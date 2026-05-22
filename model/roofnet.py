from .pointnet2 import PointNet2
from .cluster_refine import ClusterRefineNet
from .edge_pred_net import EdgeAttentionNet
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN



class RoofNet(nn.Module):
    def __init__(self, model_cfg, input_channel=3):
        super().__init__()
        self.use_edge = bool(model_cfg.get('use_edge', True))
        self.model_cfg = model_cfg
        
        # 配置处理
        self.use_building3d_format = model_cfg.get('use_building3d_format', False)
        self.use_color = model_cfg.get('use_color', False)
        self.use_intensity = model_cfg.get('use_intensity', False)
        
        # 确定输入通道数
        if self.use_color and not self.use_intensity:
            input_channel = 6  # xyz + rgb
        elif not self.use_color and self.use_intensity:
            input_channel = 4  # xyz + intensity
        elif self.use_color and self.use_intensity:
            input_channel = 7  # xyz + rgb + intensity
        else:
            input_channel = 3  # 仅xyz
        self.input_channel = input_channel
        
        # 初始化子网络
        self.keypoint_det_net = PointNet2(
            model_cfg.PointNet2, 
            in_channel=input_channel
        )
        
        self.cluster_refine_net = ClusterRefineNet(
            model_cfg.ClusterRefineNet,
            input_channel=self.keypoint_det_net.num_output_feature
        )
        self.edge_att_net = EdgeAttentionNet(
            model_cfg.EdgeAttentionNet,
            input_channel=self.cluster_refine_net.num_output_feature
        )

    def forward(self, batch_dict):
        # 数据格式适配
        batch_dict = self._adapt_batch_dict(batch_dict)
        
        # 特征提取
        batch_dict = self.keypoint_det_net(batch_dict)
        
        # 可选：边处理
        if self.use_edge:
            batch_dict = self.cluster_refine_net(batch_dict)
            batch_dict = self.edge_att_net(batch_dict)
        
        # 训练/推理
        if self.training:
            return self._compute_loss(batch_dict)
        else:
            return batch_dict
    
    def _adapt_batch_dict(self, batch_dict):
        """适配不同数据格式"""
        adapted = batch_dict.copy()
        
        # 点云数据
        if 'point_clouds' in adapted:
            # 提取正确的特征维度
            points = adapted['point_clouds']
            if points.shape[-1] > 3:
                if self.use_color and not self.use_intensity:
                    # 使用xyz+颜色（前6维）
                    adapted['points'] = points[..., :6]
                elif not self.use_color and self.use_intensity:
                    # 使用xyz+强度
                    if points.shape[-1] >= 4:
                        adapted['points'] = torch.cat([
                            points[..., :3], 
                            points[..., 7:8]  # 强度在第7维
                        ], dim=-1)
                elif self.use_color and self.use_intensity:
                    # 使用所有特征
                    adapted['points'] = points
                else:
                    # 仅使用xyz
                    adapted['points'] = points[..., :3]
            else:
                adapted['points'] = points
        
        # 标签数据
        if 'wf_vertices' in adapted:
            adapted['vectors'] = adapted['wf_vertices']
        if 'wf_edges' in adapted:
            adapted['edges'] = adapted['wf_edges']
        
        return adapted
    
    def _compute_loss(self, batch_dict):
        """计算总损失"""
        loss = 0
        loss_dict = {}
        disp_dict = {}
        
        # 关键点检测损失
        kp_loss, loss_dict, disp_dict = self.keypoint_det_net.loss(
            loss_dict, disp_dict, batch_dict
        )
        loss += kp_loss
        
        # 可选：边相关损失
        if self.use_edge:
            cr_loss, loss_dict, disp_dict = self.cluster_refine_net.loss(
                loss_dict, disp_dict, batch_dict
            )
            loss += cr_loss
            
            ea_loss, loss_dict, disp_dict = self.edge_att_net.loss(
                loss_dict, disp_dict, batch_dict
            )
            loss += ea_loss
        
        return loss, loss_dict, disp_dict
    
    def enable_edge_processing(self, enable=True):
        """启用/禁用边处理"""
        self.use_edge = enable
        if enable and not hasattr(self, 'cluster_refine_net'):
            # 动态创建边处理网络
            self.cluster_refine_net = ClusterRefineNet(
                self.model_cfg.ClusterRefineNet,
                input_channel=self.keypoint_det_net.num_output_feature
            )
            self.edge_att_net = EdgeAttentionNet(
                self.model_cfg.EdgeAttentionNet,
                input_channel=self.cluster_refine_net.num_output_feature
            )
