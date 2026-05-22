import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_stack_utils import *
from .model_utils import *
from scipy.optimize import linear_sum_assignment
from utils import loss_utils
import pc_util
import itertools


class EdgeAttentionNet(nn.Module):
    def __init__(self, model_cfg, input_channel):
        super().__init__()
        self.model_cfg = model_cfg
        self.freeze = False

        self.att_layer = PairedPointAttention(input_channel)
        num_feature = self.att_layer.num_output_feature
        self.shared_fc = LinearBN(num_feature, num_feature)
        self.drop = nn.Dropout(0.5)
        self.cls_fc = nn.Linear(num_feature, 1)

        if self.training:
            self.train_dict = {}
            pos_weight = torch.tensor([20.0])
            # 1. 使用标准 Loss，并添加 pos_weight 解决正负样本不均
            # 20.0 表示给正样本（存在的边）更高的权重
            pos_weight = torch.tensor([20.0]) 
            self.cls_loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            self.loss_weight = self.model_cfg.LossWeight

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch_dict):
        if self.training:
            self.train_dict = {}
        batch_idx = batch_dict['keypoint'][:, 0]
        point_fea = batch_dict['keypoint_features']
        device = point_fea.device

        bin_label_list = []
        pair_idx_list, pair_idx_list1, pair_idx_list2 = [], [], []
        
        matches = batch_dict.get('matches', None)
        edge_label = batch_dict.get('edges', [])
        
        idx = 0
        for i in range(batch_dict['batch_size']):
            mask = batch_idx == i
            num_pts = mask.sum().item()
            
            if num_pts <= 1:
                idx += num_pts
                continue
                
            if self.training:
                if i >= len(edge_label):
                    idx += num_pts
                    continue
                    
                curr_match = matches[mask]
                valid_match_mask = (curr_match != -1)
                
                if valid_match_mask.sum() < 2:
                    idx += num_pts
                    continue
                
                # 1. 生成所有预测组合
                pair_idx = list(itertools.combinations(range(num_pts), 2))
                pair_idx = torch.tensor(pair_idx, device=device)
                
                # 2. 准备 GT 集合
                curr_gt = edge_label[i]
                gt_edge_set = set()
                
                if curr_gt is not None:
                    # 将数据转为 numpy 以便处理
                    if torch.is_tensor(curr_gt):
                        curr_gt_np = curr_gt.cpu().numpy()
                    else:
                        curr_gt_np = np.array(curr_gt)

                    # --- 核心修复：强制检查维度 ---
                    if curr_gt_np.ndim == 1 and curr_gt_np.size == 2:
                        # 如果只有一条边被挤压成了 [v1, v2]，转为 [[v1, v2]]
                        curr_gt_np = curr_gt_np[np.newaxis, :]
                    
                    if curr_gt_np.ndim == 2 and curr_gt_np.shape[1] == 2:
                        # 确保每个 e 都是可迭代的数组 [v1, v2]
                        curr_gt_np = curr_gt_np[np.sum(curr_gt_np, axis=1) >= 0]
                        gt_edge_set = set([tuple(sorted((int(e[0]), int(e[1])))) for e in curr_gt_np])

                # 3. 生成标签
                match_np = curr_match.cpu().numpy()
                match_edges = [tuple(sorted((int(match_np[p[0]]), int(match_np[p[1]])))) for p in pair_idx.tolist()]
                
                label = torch.tensor([e in gt_edge_set for e in match_edges], device=device, dtype=torch.float)
                
                # DEBUG 打印
                bin_label_list.append(label)
                pair_idx_list.append(pair_idx)
                pair_idx_list1.append(pair_idx[:, 0] + idx)
                pair_idx_list2.append(pair_idx[:, 1] + idx)
                
            else:
                # 推理模式
                pair_idx = list(itertools.combinations(range(num_pts), 2))
                pair_idx = torch.tensor(pair_idx, device=device)
                pair_idx_list.append(pair_idx)
                pair_idx_list1.append(pair_idx[:, 0] + idx)
                pair_idx_list2.append(pair_idx[:, 1] + idx)

            idx += num_pts

        # --- 后续预测逻辑保持不变 ---
        if len(pair_idx_list1) > 0:
            p1, p2 = torch.cat(pair_idx_list1).long(), torch.cat(pair_idx_list2).long()
            edge_fea = self.att_layer(point_fea[p1], point_fea[p2])
            edge_pred = self.cls_fc(self.drop(self.shared_fc(edge_fea)))
            
            batch_dict['pair_points'] = torch.cat(pair_idx_list, 0)
            batch_dict['edge_score'] = torch.sigmoid(edge_pred).view(-1)
            
            if self.training and len(bin_label_list) > 0:
                self.train_dict['label'] = torch.cat(bin_label_list)
                self.train_dict['edge_pred'] = edge_pred.squeeze(-1)
        else:
            batch_dict['pair_points'] = point_fea.new_zeros((0, 2), dtype=torch.long)
            batch_dict['edge_score'] = point_fea.new_zeros((0,))
        
        return batch_dict

    def loss(self, loss_dict, disp_dict, batch_dict=None):
        # 1. 获取模型预测和真值标签
        # 确保 forward 中 self.train_dict['edge_pred'] 已经 squeeze 过了或者在这里处理
        if 'edge_pred' not in self.train_dict or 'label' not in self.train_dict:
            zero = next(self.parameters()).sum() * 0.0
            loss_dict.update({
                'edge_cls_loss': 0.0,
                'edge_loss': 0.0
            })
            disp_dict.update({'edge_acc': 0.0})
            return zero, loss_dict, disp_dict
        pred_logits = self.train_dict['edge_pred'].view(-1) 
        label_cls = self.train_dict['label'].view(-1)
        
        # 2. 计算分类 Loss
        cls_loss = self.cls_loss_func(pred_logits, label_cls)
        
        # 如果你有自定义的 get_cls_loss 也可以用，但建议保持一致：
        # cls_loss = self.get_cls_loss(pred_logits, label_cls, self.loss_weight.get('cls_weight', 1.0))
        
        total_loss = cls_loss # 如果后续有其他 loss (如 offset loss) 再相加
        
        # 3. 更新 loss 统计字典
        loss_dict.update({
            'edge_cls_loss': cls_loss.item(),
            'edge_loss': total_loss.item()
        })

        # 4. 计算准确率 (Accuracy/Recall)
        with torch.no_grad():
            # 将 logits 转为 0/1 预测值
            pred_probs = torch.sigmoid(pred_logits)
            pred_labels = (pred_probs >= 0.5).float()
            
            # 统计正样本数量 (label 为 1 的边)
            num_positives = torch.sum(label_cls == 1).item()
            
            if num_positives > 0:
                # 这里计算的是 Positive Recall (召回率)：
                # 即在所有真实的边中，有多少被模型正确预测出来了
                correct_positives = torch.sum((pred_labels == 1) & (label_cls == 1)).item()
                acc = correct_positives / num_positives
            else:
                # 训练初期如果没有匹配到边，acc 设为 0
                acc = 0.0

        # 5. 更新显示字典
        disp_dict.update({'edge_acc': acc})
        
        return total_loss, loss_dict, disp_dict

    def get_cls_loss(self, pred, label, weight):
        positives = label > 0
        negatives = label == 0
        cls_weights = (negatives * 1.0 + positives * 1.0).float()
        pos_normalizer = positives.sum().float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_loss_src = self.cls_loss_func(pred.squeeze(-1), label, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum()

        cls_loss = cls_loss * weight
        return cls_loss


class PairedPointAttention(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.edge_att1 = nn.Sequential(
            nn.Linear(input_channel, input_channel),
            nn.BatchNorm1d(input_channel),
            nn.ReLU(),
            nn.Linear(input_channel, input_channel),
            nn.Sigmoid(),
        )
        self.edge_att2 = nn.Sequential(
            nn.Linear(input_channel, input_channel),
            nn.BatchNorm1d(input_channel),
            nn.ReLU(),
            nn.Linear(input_channel, input_channel),
            nn.Sigmoid(),
        )
        self.fea_fusion_layer = nn.MaxPool1d(2)

        self.num_output_feature = input_channel

    def forward(self, point_fea1, point_fea2):
        fusion_fea = point_fea1 + point_fea2
        att1 = self._forward_attention_mlp(self.edge_att1, fusion_fea)
        att2 = self._forward_attention_mlp(self.edge_att2, fusion_fea)
        att_fea1 = point_fea1 * att1
        att_fea2 = point_fea2 * att2
        fea = torch.cat([att_fea1.unsqueeze(1), att_fea2.unsqueeze(1)], 1)
        fea = self.fea_fusion_layer(fea.permute(0, 2, 1)).squeeze(-1)
        return fea

    @staticmethod
    def _forward_attention_mlp(layers, x):
        for layer in layers:
            if isinstance(layer, nn.BatchNorm1d) and layer.training and x.dim() == 2 and x.shape[0] == 1:
                x = F.batch_norm(
                    x,
                    layer.running_mean,
                    layer.running_var,
                    layer.weight,
                    layer.bias,
                    training=False,
                    momentum=layer.momentum,
                    eps=layer.eps,
                )
            else:
                x = layer(x)
        return x






