import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import DBSCAN
from torch.autograd import Function

# --- 移除 import pc_util ---

class Conv2ds(nn.Sequential):
    def __init__(self, cns):
        super().__init__()
        for i in range(len(cns) - 1):
            in_cn, out_cn = cns[i], cns[i + 1]
            self.add_module('conv%d' % (i + 1), Conv2dBN(in_cn, out_cn))

class Conv2dBN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.bn = nn.BatchNorm2d(out_channel)
        self.conv = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x), inplace=True))

class Conv1ds(nn.Sequential):
    def __init__(self, cns):
        super().__init__()
        for i in range(len(cns) - 1):
            in_cn, out_cn = cns[i], cns[i + 1]
            self.add_module('conv%d' % (i + 1), Conv1dBN(in_cn, out_cn))

class Conv1dBN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.bn = nn.BatchNorm1d(out_channel)
        self.conv = nn.Conv1d(in_channel, out_channel, 1)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x), inplace=True))

class Linears(nn.Sequential):
    def __init__(self, cns):
        super().__init__()
        for i in range(len(cns) - 1):
            in_cn, out_cn = cns[i], cns[i + 1]
            self.add_module('linear%d' % (i + 1), LinearBN(in_cn, out_cn))

class LinearBN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.bn = nn.BatchNorm1d(out_channel)
        self.linear = nn.Linear(in_channel, out_channel)

    def forward(self, x):
        x = F.relu(self.linear(x), inplace=True)
        if self.bn.training and x.dim() == 2 and x.shape[0] == 1:
            return F.batch_norm(
                x,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                self.bn.bias,
                training=False,
                momentum=self.bn.momentum,
                eps=self.bn.eps,
            )
        return self.bn(x)

def _remap_state_keys_for_current_model(checkpoint_state, model_state):
    remapped = {}
    for key, value in checkpoint_state.items():
        target_key = key
        if target_key not in model_state:
            candidate = key.replace('.shared_fc.conv.', '.shared_fc.linear.')
            if candidate in model_state and model_state[candidate].shape == value.shape:
                target_key = candidate
        remapped[target_key] = value
    return remapped


def load_params_with_optimizer(net, filename, to_cpu=False, optimizer=None, logger=None):
    if not os.path.isfile(filename):
        raise FileNotFoundError
    if logger is not None:
        logger.info('==> Loading parameters from checkpoint: %s', filename)
    map_location = torch.device('cpu') if to_cpu else None
    checkpoint = torch.load(filename, map_location=map_location)
    epoch = checkpoint.get('epoch', -1)
    it = checkpoint.get('it', 0.0)
    model_state = _remap_state_keys_for_current_model(checkpoint['model_state'], net.state_dict())
    missing_keys, unexpected_keys = net.load_state_dict(model_state, strict=False)
    missing_keys = [k for k in missing_keys if not k.endswith('cls_loss_func.pos_weight')]
    if logger is not None and (missing_keys or unexpected_keys):
        logger.warning('Checkpoint loaded with missing keys: %s', missing_keys[:20])
        logger.warning('Checkpoint loaded with unexpected keys: %s', unexpected_keys[:20])
    if optimizer is not None:
        optim_state = checkpoint.get('optimizer_state', None)
        if optim_state is not None:
            if logger is not None:
                logger.info('==> Loading optimizer parameters from checkpoint')
            optimizer.load_state_dict(optim_state)
    if logger is not None:
        logger.info('==> Done')
    return it, epoch

# --- 纯 PyTorch/Sklearn 替代方案 ---

class DBSCANCluster(Function):
    @staticmethod
    def forward(ctx, eps: float, min_pts: int, point: torch.Tensor) -> torch.Tensor:
        """使用 Scikit-Learn 替代 C++ dbscan_wrapper"""
        B, N, _ = point.size()
        device = point.device
        # 推荐使用 torch.full 替代旧版 torch.cuda.IntTensor
        idx = torch.full((B, N), -1, dtype=torch.int32, device=device)
        
        point_np = point.detach().cpu().numpy()
        for b in range(B):
            db = DBSCAN(eps=eps, min_samples=min_pts).fit(point_np[b])
            idx[b] = torch.from_numpy(db.labels_).to(device).int()
            
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, grad_out): return (None, None, None)

dbscan_cluster = DBSCANCluster.apply

class GetClusterPts(Function):
    @staticmethod
    def forward(ctx, point: torch.Tensor, cluster_idx: torch.Tensor) -> torch.Tensor:
        """使用 PyTorch 原生算子替代 C++ cluster_pts_wrapper"""
        B, N = cluster_idx.size()
        # 计算最大簇类数
        M = int(cluster_idx.max() + 1)
        if M <= 0: # 处理全为噪声的情况
            return torch.zeros((B, 1, 3), device=point.device) - 10.0, torch.zeros((B, 1), dtype=torch.int32, device=point.device)

        device = point.device
        key_pts = torch.zeros((B, M, 3), device=device)
        num_cluster = torch.zeros((B, M), dtype=torch.int32, device=device)

        for b in range(B):
            for m in range(M):
                mask = (cluster_idx[b] == m)
                count = mask.sum()
                if count > 0:
                    num_cluster[b, m] = count.int()
                    key_pts[b, m] = point[b][mask].mean(dim=0)
                else:
                    key_pts[b, m] = -10.0 # 模拟原代码 key_pts[key_pts * 1e4 == 0] = -1e1

        ctx.mark_non_differentiable(key_pts)
        ctx.mark_non_differentiable(num_cluster)
        return key_pts, num_cluster

    @staticmethod
    def backward(ctx, grad_out): return (None, None)

get_cluster_pts = GetClusterPts.apply
