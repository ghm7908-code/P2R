import torch
import torch.nn as nn
from torch.autograd import Function, Variable
import pc_util


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor,
                new_xyz: torch.Tensor, new_xyz_batch_cnt: torch.Tensor):
        B = xyz_batch_cnt.shape[0]
        M = new_xyz.shape[0]
        device = xyz.device
        idx = torch.zeros((M, nsample), dtype=torch.int32, device=device)
        
        # 逐 Batch 处理以确保点不会跨越不同的样本
        cur_xyz_idx = 0
        cur_new_xyz_idx = 0
        for b in range(B):
            n, m = xyz_batch_cnt[b].item(), new_xyz_batch_cnt[b].item()
            # 提取当前 Batch 的点
            batch_xyz = xyz[cur_xyz_idx : cur_xyz_idx + n]
            batch_new_xyz = new_xyz[cur_new_xyz_idx : cur_new_xyz_idx + m]
            
            # 计算距离 (m, n)
            dist = torch.cdist(batch_new_xyz, batch_xyz)
            # 寻找半径内的点并填充
            for i in range(m):
                in_ball = torch.where(dist[i] < radius)[0]
                if len(in_ball) > 0:
                    # 这里的索引需要加上起始偏移量
                    sel_idx = in_ball[torch.randperm(len(in_ball))[:nsample]] if len(in_ball) > nsample else in_ball
                    idx[cur_new_xyz_idx + i, :len(sel_idx)] = (sel_idx + cur_xyz_idx).int()
                    # 用第一个找到的点填充剩余位置
                    if len(sel_idx) < nsample:
                        idx[cur_new_xyz_idx + i, len(sel_idx):] = (sel_idx[0] + cur_xyz_idx).int()
                else:
                    idx[cur_new_xyz_idx + i] = cur_xyz_idx # 默认指向该 batch 第一个点
            
            cur_xyz_idx += n
            cur_new_xyz_idx += m

        empty_ball_mask = (idx[:, 0] == -1) # 简单模拟原逻辑
        return idx, empty_ball_mask

    @staticmethod
    def backward(ctx, a=None): return (None,) * 6


ball_query = BallQuery.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, features_batch_cnt: torch.Tensor,
                idx: torch.Tensor, idx_batch_cnt: torch.Tensor):
        """
        纯 PyTorch 实现，替代 pc_util.group_points_wrapper_stack
        Args:
            features: (N1 + N2 ..., C) 
            idx: (M1 + M2 ..., nsample) 索引
        Returns:
            output: (M1 + M2 ..., C, nsample)
        """
        # 1. 参数校验与预处理
        M, nsample = idx.size()
        N, C = features.size()
        B = idx_batch_cnt.shape[0]
        
        # 2. 核心逻辑：利用 PyTorch 的高级索引直接提取特征
        # idx 的形状是 (M, nsample)，存储的是 0 到 N-1 之间的全局索引
        # features[idx.long()] 的形状会变成 (M, nsample, C)
        idx_long = idx.long()
        output = features[idx_long] # (M, nsample, C)

        # 3. 调整维度顺序以符合原输出格式 (M, C, nsample)
        output = output.permute(0, 2, 1).contiguous()

        # 保存用于 backward 的变量
        ctx.for_backwards = (B, N, idx_long, features_batch_cnt, idx_batch_cnt)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            grad_out: (M, C, nsample)
        Returns:
            grad_features: (N, C)
        """
        B, N, idx_long, features_batch_cnt, idx_batch_cnt = ctx.for_backwards
        M, C, nsample = grad_out.size()
        device = grad_out.device

        # 创建梯度占位符
        grad_features = torch.zeros((N, C), device=device)

        # 将 grad_out 转换回 (M, nsample, C)
        grad_out_reshaped = grad_out.permute(0, 2, 1).contiguous()

        # 核心逻辑：使用 index_add_ 将梯度累加回对应的原始点索引位置
        # 我们需要将 idx 展平处理
        flat_idx = idx_long.view(-1) # (M * nsample)
        flat_grad = grad_out_reshaped.view(-1, C) # (M * nsample, C)
        
        grad_features.index_add_(0, flat_idx, flat_grad)

        return grad_features, None, None, None

grouping_operation = GroupingOperation.apply


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        Args:
            radius: float, radius of ball
            nsample: int, maximum number of features to gather in the ball
            use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor,
                new_xyz: torch.Tensor, new_xyz_batch_cnt: torch.Tensor,
                features: torch.Tensor = None):
        """
        Args:
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            features: (N1 + N2 ..., C) tensor of features to group

        Returns:
            new_features: (M1 + M2, C, nsample) tensor
        """
        assert xyz.shape[0] == xyz_batch_cnt.sum(), 'xyz: %s, xyz_batch_cnt: %s' % (str(xyz.shape), str(new_xyz_batch_cnt))
        assert new_xyz.shape[0] == new_xyz_batch_cnt.sum(), \
            'new_xyz: %s, new_xyz_batch_cnt: %s' % (str(new_xyz.shape), str(new_xyz_batch_cnt))

        # idx: (M1 + M2 ..., nsample), empty_ball_mask: (M1 + M2 ...)
        idx, empty_ball_mask = ball_query(self.radius, self.nsample, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt)
        grouped_xyz = grouping_operation(xyz, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, 3, nsample)
        grouped_xyz -= new_xyz.unsqueeze(-1)

        grouped_xyz[empty_ball_mask] = 0

        if features is not None:
            grouped_features = grouping_operation(features, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, C, nsample)
            grouped_features[empty_ball_mask] = 0
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (M1 + M2 ..., C + 3, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features, idx


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int):
        """
        Args:
            ctx:
            xyz: (B, N, 3) where N > npoint
            npoint: int, number of features in the sampled set

        Returns:
            output: (B, npoint) tensor containing the set
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        pc_util.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown: torch.Tensor, unknown_batch_cnt: torch.Tensor, 
                known: torch.Tensor, known_batch_cnt: torch.Tensor):
        B = unknown_batch_cnt.shape[0]
        M = unknown.shape[0]
        dist2 = torch.zeros((M, 3), device=unknown.device)
        idx = torch.zeros((M, 3), dtype=torch.int32, device=unknown.device)

        last_u, last_k = 0, 0
        for b in range(B):
            u_cnt, k_cnt = unknown_batch_cnt[b].item(), known_batch_cnt[b].item()
            u_xyz = unknown[last_u:last_u+u_cnt]
            k_xyz = known[last_k:last_k+k_cnt]
            
            # 计算距离并取前 3 个最近点
            dists = torch.cdist(u_xyz, k_xyz)
            d, i = dists.topk(3, largest=False, dim=-1)
            
            dist2[last_u:last_u+u_cnt] = d ** 2
            idx[last_u:last_u+u_cnt] = (i + last_k).int()
            
            last_u += u_cnt
            last_k += k_cnt
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor):
        # features: (M, C), idx: (N, 3), weight: (N, 3)
        ctx.save_for_backward(idx, weight)
        ctx.M = features.shape[0]
        
        # 利用高级索引进行加权求和
        idx_long = idx.long()
        # features[idx_long] 维度是 (N, 3, C)
        # weight 维度是 (N, 3)，需扩展为 (N, 3, 1)
        output = (features[idx_long] * weight.unsqueeze(-1)).sum(dim=1)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, weight = ctx.saved_tensors
        M = ctx.M
        grad_features = torch.zeros((M, grad_out.shape[1]), device=grad_out.device)
        
        # 梯度反向累加
        for i in range(3):
            grad_features.index_add_(0, idx[:, i].long(), grad_out * weight[:, i].unsqueeze(-1))
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


if __name__ == '__main__':
    pass
