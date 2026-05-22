import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .model_utils import *
from utils import loss_utils


class PointNet2(nn.Module):
    def __init__(self, model_cfg, in_channel=3):
        super().__init__()
        self.model_cfg = model_cfg
        self.sa1 = PointNetSAModule(256, 0.1, 16, in_channel, [32, 32, 64])
        self.sa2 = PointNetSAModule(128, 0.2, 16, 64, [64, 64, 128])
        self.sa3 = PointNetSAModule(64, 0.4, 16, 128, [128, 128, 256])
        self.sa4 = PointNetSAModule(16, 0.8, 16, 256, [256, 256, 512])
        self.fp4 = PointNetFPModule(768, [256, 256])
        self.fp3 = PointNetFPModule(384, [256, 256])
        self.fp2 = PointNetFPModule(320, [256, 128])
        self.fp1 = PointNetFPModule(128, [128, 128, 128])
        self.shared_fc = Conv1dBN(128, 128)
        self.drop = nn.Dropout(0.5)
        self.offset_fc = nn.Conv1d(128, 3, 1)
        self.cls_fc = nn.Conv1d(128, 1, 1)
        self.init_weights()
        self.num_output_feature = 128
        if self.training:
            self.train_dict = {}
            self.add_module(
                'cls_loss_func',
                loss_utils.SigmoidBCELoss()
            )
            self.add_module(
                'reg_loss_func',
                loss_utils.WeightedSmoothL1Loss()
            )
            self.loss_weight = self.model_cfg.LossWeight

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
        points = batch_dict['points']
        vectors = batch_dict['vectors']

        B = int(batch_dict['batch_size'])
        if points.dim() == 3:
            point_input = points.contiguous()
            B, N, _ = point_input.shape
        elif points.dim() == 2:
            N = points.shape[0] // B
            point_input = points.view(B, N, -1).contiguous()
        else:
            raise ValueError(f"Unsupported points shape: {points.shape}")

        xyz_batch = point_input[..., :3].contiguous()
        batch_dict['points'] = xyz_batch.reshape(B * N, 3)

        if self.training:
            # 传入重塑后的 xyz_batch
            offset, cls = self.assign_targets(xyz_batch, vectors, self.model_cfg.PosRadius)
            self.train_dict.update({
                'offset_label': offset,
                'cls_label': cls
            })

        
        l0_xyz = xyz_batch
        l0_fea = point_input.permute(0, 2, 1).contiguous()

        l1_xyz, l1_fea = self.sa1(l0_xyz, l0_fea)
        l2_xyz, l2_fea = self.sa2(l1_xyz, l1_fea)
        l3_xyz, l3_fea = self.sa3(l2_xyz, l2_fea)
        l4_xyz, l4_fea = self.sa4(l3_xyz, l3_fea)

        l3_fea = self.fp4(l3_xyz, l4_xyz, l3_fea, l4_fea)
        l2_fea = self.fp3(l2_xyz, l3_xyz, l2_fea, l3_fea)
        l1_fea = self.fp2(l1_xyz, l2_xyz, l1_fea, l2_fea)
        l0_fea = self.fp1(l0_xyz, l1_xyz, None, l1_fea)

        x = self.drop(self.shared_fc(l0_fea))
        pred_offset = self.offset_fc(x).permute(0, 2, 1)
        pred_cls = self.cls_fc(x).permute(0, 2, 1)
        if self.training:
            self.train_dict.update({
                'cls_pred': pred_cls,
                'offset_pred': pred_offset
            })
        batch_dict['point_features'] = l0_fea.permute(0, 2, 1)
        batch_dict['point_pred_score'] = torch.sigmoid(pred_cls).squeeze(-1)
        batch_dict['point_pred_offset'] = pred_offset * self.model_cfg.PosRadius
        return batch_dict

    def loss(self, loss_dict, disp_dict,batch_dict=None):
        pred_cls, pred_offset = self.train_dict['cls_pred'], self.train_dict['offset_pred']
        label_cls, label_offset = self.train_dict['cls_label'], self.train_dict['offset_label']
        cls_loss = self.get_cls_loss(pred_cls, label_cls, self.loss_weight['cls_weight'])
        reg_loss = self.get_reg_loss(pred_offset, label_offset, label_cls, self.loss_weight['reg_weight'])
        loss = cls_loss + reg_loss
        loss_dict.update({
            'pts_cls_loss': cls_loss.item(),
            'pts_offset_loss': reg_loss.item(),
            'pts_loss': loss.item()
        })

        pred_cls = pred_cls.squeeze(-1)
        label_cls = label_cls.squeeze(-1)
        pred_logit = torch.sigmoid(pred_cls)
        pred = torch.where(pred_logit >= 0.5, pred_logit.new_ones(pred_logit.shape), pred_logit.new_zeros(pred_logit.shape))
        pos_count = torch.sum(label_cls == 1).item()
        acc = torch.sum((pred == label_cls) & (label_cls == 1)).item() / max(pos_count, 1)
        #acc = torch.sum(pred == label_cls).item() / len(label_cls.view(-1))
        disp_dict.update({'pts_acc': acc})
        return loss, loss_dict, disp_dict

    def get_cls_loss(self, pred, label, weight):
        batch_size = int(pred.shape[0])
        positives = label > 0
        negatives = label == 0
        cls_weights = (negatives * 1.0 + positives * 1.0).float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_loss_src = self.cls_loss_func(pred.squeeze(-1), label, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * weight
        return cls_loss

    def get_reg_loss(self, pred, label, cls_label, weight):
        batch_size = int(pred.shape[0])
        positives = cls_label > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        reg_loss_src = self.reg_loss_func(pred, label, weights=reg_weights)  # [N, M]
        reg_loss = reg_loss_src.sum() / batch_size
        reg_loss = reg_loss * weight
        return reg_loss


    def assign_targets(self, points, gvs, radius):
        """
        Args:
            points: (B, N, 3) 还原后的点云
            gvs: (B, M, 3) 地面真值关键点
        """
        B, N, _ = points.shape
        device = points.device

        # 1. 计算距离矩阵 (B, N, M)
        # 计算每个点到该 Batch 内所有 GT 关键点的距离
        dists = torch.cdist(points, gvs) 

        # 2. 获取每个点最近的 GT 关键点索引
        # min_dist: (B, N), min_idx: (B, N)
        min_dist, min_idx = torch.min(dists, dim=2)

        # 3. 构造用于 gather 的索引 (B, N, 3)
        # 必须扩展到 3 维才能从 (B, M, 3) 的 gvs 中提取坐标
        expanded_idx = min_idx.unsqueeze(-1).expand(B, N, 3).long()

        # 4. 提取对应的 GT 关键点坐标
        # 现在 gvs (B, M, 3) 和 expanded_idx (B, N, 3) 维度一致，报错消失
        target_points = torch.gather(gvs, 1, expanded_idx) 

        # 5. 计算偏移位移
        dis = (target_points - points) / radius

        # 6. 掩码处理
        # 只有在 radius 范围内的点才被视为“正样本”
        mask = (min_dist > radius).unsqueeze(-1)
        dis.masked_fill_(mask, 0) # 非关键点周围的点，偏移设为 0

        # 7. 分类标签 (B, N)
        label = (min_dist <= radius).float()

        return dis, label


class PointNetSAModuleMSG(nn.Module):
    def __init__(self, npoint, radii, nsamples, in_channel, mlps, use_xyz=True):
        """
        PointNet Set Abstraction Module
        :param npoint: int
        :param radii: list of float, radius in ball_query
        :param nsamples: list of int, number of samples in ball_query
        :param in_channel: int
        :param mlps: list of list of int
        :param use_xyz: bool
        """
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        mlps = [[in_channel] + mlp for mlp in mlps]
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        for i in range(len(radii)):
            r = radii[i]
            nsample = nsamples[i]
            mlp = mlps[i]
            if use_xyz:
                mlp[0] += 3
            self.groupers.append(QueryAndGroup(r, nsample, use_xyz) if npoint is not None else GroupAll(use_xyz))
            self.mlps.append(Conv2ds(mlp))

    def forward(self, xyz, features, new_xyz=None):
        new_features_list = []
        xyz = xyz.contiguous()
        xyz_flipped = xyz.permute(0, 2, 1)
        if new_xyz is None:
            new_xyz = gather_operation(xyz_flipped, furthest_point_sample(
                xyz, self.npoint)).permute(0, 2, 1) if self.npoint is not None else None

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)  

            new_features = self.mlps[i](new_features)  
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(-1)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointNetSAModule(PointNetSAModuleMSG):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, use_xyz=True):
        super().__init__(npoint, [radius], [nsample], in_channel, [mlp], use_xyz)


class PointNetFPModule(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp = Conv2ds([in_channel] + mlp)

    def forward(self, pts1, pts2, fea1, fea2):
        """
        :param pts1: (B, n, 3) 
        :param pts2: (B, m, 3)  n > m
        :param fea1: (B, C1, n)
        :param fea2: (B, C2, m)
        :return:
            new_features: (B, mlp[-1], n)
        """
        if pts2 is not None:
            dist, idx = three_nn(pts1, pts2)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = three_interpolate(fea2, idx, weight)
        else:
            interpolated_feats = fea2.expand(*fea2.size()[0:2], pts1.size(1))

        if fea1 is not None:
            new_features = torch.cat([interpolated_feats, fea1], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)



class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int, wd: float = 1.0, wf: float = 0.0) -> torch.Tensor:
        """纯 PyTorch 实现最远点采样 (FPS)"""
        B, N, _ = xyz.size()
        device = xyz.device
        output = torch.zeros(B, npoint, dtype=torch.int32, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        batch_indices = torch.arange(B, dtype=torch.long, device=device)
        for i in range(npoint):
            output[:, i] = farthest.int()
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(ctx, grad_out): return ()

furthest_point_sample = FurthestPointSampling.apply

class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """(B, C, N), (B, npoint) -> (B, C, npoint)"""
        B, C, N = features.size()
        idx = idx.long()
        output = features.gather(2, idx.unsqueeze(1).expand(-1, C, -1))
        return output

    @staticmethod
    def backward(ctx, grad_out):
        # 简化版 backward，满足大部分非训练采样需求
        return None, None

gather_operation = GatherOperation.apply

class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B, N, 3), (B, M, 3) -> (B, N, 3), (B, N, 3)"""
        dist = torch.cdist(unknown, known)
        dist, idx = dist.sort(dim=-1)
        return torch.sqrt(dist[:, :, :3]), idx[:, :, :3].int()

    @staticmethod
    def backward(ctx, a=None, b=None): return ()

three_nn = ThreeNN.apply

class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        B, C, M = features.shape
        N = idx.shape[1]
        idx = idx.long()
        expanded_features = features.gather(2, idx.view(B, 1, N * 3).expand(-1, C, -1))
        expanded_features = expanded_features.view(B, C, N, 3)
        output = torch.sum(expanded_features * weight.unsqueeze(1), dim=-1)
        return output

    @staticmethod
    def backward(ctx, grad_out): return None, None, None

three_interpolate = ThreeInterpolate.apply

class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        B, C, N = features.shape
        npoint, nsample = idx.shape[1], idx.shape[2]
        idx = idx.long().view(B, npoint * nsample)
        output = features.gather(2, idx.unsqueeze(1).expand(-1, C, -1))
        return output.view(B, C, npoint, nsample)

    @staticmethod
    def backward(ctx, grad_out): return None, None

grouping_operation = GroupingOperation.apply

class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        B, N, _ = xyz.shape
        npoint = new_xyz.shape[1]
        device = xyz.device
        dist = torch.cdist(new_xyz, xyz)
        idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, npoint, 1)
        idx[dist > radius] = N
        idx = idx.sort(dim=-1)[0][:, :, :nsample]
        fill_idx = idx[:, :, 0:1].repeat(1, 1, nsample)
        mask = (idx == N)
        idx[mask] = fill_idx[mask]
        ctx.mark_non_differentiable(idx)
        return idx.int()

    @staticmethod
    def backward(ctx, grad_out): return ()

ball_query = BallQuery.apply

class BallCenterQuery(Function):
    @staticmethod
    def forward(ctx, radius: float, point: torch.Tensor, key_point: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(point, key_point)
        min_dist, idx = torch.min(dist, dim=-1)
        idx[min_dist > radius] = -1
        ctx.mark_non_differentiable(idx)
        return idx.int()

    @staticmethod
    def backward(ctx, grad_out): return ()

ball_center_query = BallCenterQuery.apply


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        # _, idx = pointnet_util.knn_query(self.nsample, xyz, new_xyz)
        xyz_trans = xyz.permute(0, 2, 1)
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.permute(0, 2, 1).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class GroupAll(nn.Module):
    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        grouped_xyz = xyz.permute(0, 2, 1).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
