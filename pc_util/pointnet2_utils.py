import torch
import torch.nn as nn
from torch.autograd import Function

# --------------------------------------------------
# 1. 替代 ball_query (球查询)
# --------------------------------------------------
def ball_query(radius, nsample, xyz, new_xyz):
    """
    用纯 PyTorch 实现替代 CUDA 版 ball_query
    xyz: (B, N, 3) 原始点云
    new_xyz: (B, S, 3) 中心点
    """
    device = xyz.device
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    
    # 计算距离矩阵 (B, S, N)
    dist = torch.cdist(new_xyz, xyz) 
    
    # 找到距离在 radius 以内的索引
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    group_idx[dist > radius] = N  # 超过半径的标记为 N (待会会被截断或填补)
    
    # 排序取前 nsample 个
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    
    # 填充：如果某个球内点数不够，用第一个点的索引填充
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    
    return group_idx.int()

# --------------------------------------------------
# 2. 替代 gather_operation (特征收集)
# --------------------------------------------------
def gather_operation(features, idx):
    """
    features: (B, C, N)
    idx: (B, npoint)
    """
    B, C, N = features.shape
    idx = idx.long()
    # 转换维度进行收集
    res = features.gather(2, idx.unsqueeze(1).expand(-1, C, -1))
    return res

# --------------------------------------------------
# 3. 替代 furthest_point_sample (FPS 采样)
# --------------------------------------------------
def furthest_point_sample(xyz, npoint):
    """
    xyz: (B, N, 3)
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids.int()

# --------------------------------------------------
# 4. 其他模型可能调用的类 (如有)
# --------------------------------------------------
class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        return furthest_point_sample(xyz, npoint)

    @staticmethod
    def backward(ctx, grad_out):
        return None, None

def grouping_operation(features, idx):
    """
    features: (B, C, N)
    idx: (B, S, nsample)
    """
    B, C, N = features.shape
    S, nsample = idx.shape[1], idx.shape[2]
    idx = idx.long().view(B, S * nsample)
    res = gather_operation(features, idx)
    return res.view(B, C, S, nsample)