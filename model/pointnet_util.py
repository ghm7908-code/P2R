import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Tuple

# --- 模拟原始接口，不再导入 pc_util ---

def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """纯 PyTorch 实现 FPS 采样"""
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

def gather_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """纯 PyTorch 实现特征收集 (B, C, N) -> (B, C, npoint)"""
    B, C, N = features.shape
    idx = idx.long()
    return features.gather(2, idx.unsqueeze(1).expand(-1, C, -1))

def grouping_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """纯 PyTorch 实现分组 (B, C, N) -> (B, C, npoint, nsample)"""
    B, C, N = features.shape
    npoint, nsample = idx.shape[1], idx.shape[2]
    idx = idx.long().view(B, npoint * nsample)
    res = gather_operation(features, idx)
    return res.view(B, C, npoint, nsample)

def ball_query(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """纯 PyTorch 实现球查询"""
    device = xyz.device
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    dist = torch.cdist(new_xyz, xyz) # (B, S, N)
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    group_idx[dist > radius] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx.int()

def three_nn(unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """实现 3NN 查找"""
    dist = torch.cdist(unknown, known)
    dist, idx = dist.sort(dim=-1)
    return torch.sqrt(dist[:, :, :3]), idx[:, :, :3].int()

def three_interpolate(features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """实现特征插值"""
    B, C, M = features.shape
    N = idx.shape[1]
    idx = idx.long()
    expanded_features = features.gather(2, idx.view(B, 1, N * 3).expand(-1, C, -1))
    expanded_features = expanded_features.view(B, C, N, 3)
    return torch.sum(expanded_features * weight.unsqueeze(1), dim=-1)

def ball_center_query(radius: float, point: torch.Tensor, key_point: torch.Tensor) -> torch.Tensor:
    """实现 Center Query"""
    dist = torch.cdist(point, key_point) # (B, N, npoint)
    min_dist, min_idx = torch.min(dist, dim=-1)
    min_idx[min_dist > radius] = -1
    return min_idx.int()

def knn_query(nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """实现 KNN Query"""
    dist = torch.cdist(new_xyz, xyz)
    dist, idx = dist.sort(dim=-1)
    return torch.sqrt(dist[:, :, :nsample]), idx[:, :, :nsample].int()