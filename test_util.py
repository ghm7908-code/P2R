import glob
import tqdm
import os
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import itertools
from model.pointnet_util import *
from model.model_utils import *

# 可视化相关导入
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

def writePoints(points, clsRoad):
    with open(clsRoad, 'w+') as file1:
        for i in range(len(points)):
            point = points[i]
            file1.write(str(point[0]))
            file1.write(' ')
            file1.write(str(point[1]))
            file1.write(' ')
            file1.write(str(point[2]))
            file1.write('\n')

def writeEdges(edges, clsRoad):
    with open(clsRoad, 'w+') as file1:
        for i in range(len(edges)):
            edge = edges[i]
            file1.write(str(edge[0] + 1))
            file1.write(' ')
            file1.write(str(edge[1] + 1))
            file1.write(' ')
            file1.write('\n')

def assign_targets(points, gvs, radius):
    idx = ball_center_query(radius, points, gvs).type(torch.int64)
    batch_size = gvs.size()[0]
    idx_add = torch.arange(batch_size).to(idx.device).unsqueeze(-1).repeat(1, idx.shape[-1]) * gvs.shape[1]
    gvs = gvs.view(-1, 3)
    idx_add += idx
    target_points = gvs[idx_add.view(-1)].view(batch_size, -1, 3)
    dis = target_points - points
    dis[idx < 0] = 0
    dis /= radius
    label = torch.where(idx >= 0, torch.ones(idx.shape).to(idx.device), torch.zeros(idx.shape).to(idx.device))
    return dis, label

def _safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def test_model(model, data_loader, logger, edge_thresh=0.5, point_match_thresh=0.1):
    if len(data_loader.dataset) == 0:
        raise RuntimeError("The test split is empty. Check --data_path and --split.")

    model.use_edge = True
    statistics = {
        'tp_pts': 0,
        'num_label_pts': 0,
        'num_pred_pts': 0,
        'pts_bias': np.zeros(3, np.float64),
        'tp_edges': 0,
        'num_label_edges': 0,
        'num_pred_edges': 0,
    }

    dataloader_iter = iter(data_loader)
    with tqdm.trange(0, len(data_loader), desc='test', dynamic_ncols=True) as tbar:
        for _ in tbar:
            batch = next(dataloader_iter)
            load_data_to_gpu(batch)
            with torch.no_grad():
                batch = model(batch)
                load_data_to_cpu(batch)
            eval_process(batch, statistics, edge_thresh=edge_thresh, point_match_thresh=point_match_thresh)

    bias = statistics['pts_bias'] / max(statistics['tp_pts'], 1)
    metrics = {
        'pts_recall': _safe_div(statistics['tp_pts'], statistics['num_label_pts']),
        'pts_precision': _safe_div(statistics['tp_pts'], statistics['num_pred_pts']),
        'pts_bias_x': bias[0],
        'pts_bias_y': bias[1],
        'pts_bias_z': bias[2],
        'edge_recall': _safe_div(statistics['tp_edges'], statistics['num_label_edges']),
        'edge_precision': _safe_div(statistics['tp_edges'], statistics['num_pred_edges']),
    }
    logger.info('pts_recall: %f', metrics['pts_recall'])
    logger.info('pts_precision: %f', metrics['pts_precision'])
    logger.info('pts_bias: %f, %f, %f', bias[0], bias[1], bias[2])
    logger.info('edge_recall: %f', metrics['edge_recall'])
    logger.info('edge_precision: %f', metrics['edge_precision'])
    return metrics


def eval_process(batch, statistics, edge_thresh=0.5, point_match_thresh=0.1):
    batch_size = batch['batch_size']
    keypoints = batch.get('keypoint', np.zeros((0, 4), dtype=np.float32))
    refined_pts = batch.get('refined_keypoint', np.zeros((0, 3), dtype=np.float32))
    label_pts = batch['vectors']
    edge_scores = batch.get('edge_score', np.zeros((0,), dtype=np.float32))
    pair_points = batch.get('pair_points', np.zeros((0, 2), dtype=np.int64))
    label_edges = batch['edges']
    mm_pts = batch['minMaxPt']

    edge_offset = 0
    for i in range(batch_size):
        mm_pt = mm_pts[i]
        min_pt = mm_pt[0]
        max_pt = mm_pt[1]
        delta_pt = max_pt - min_pt

        p_pts = refined_pts[keypoints[:, 0] == i] if len(keypoints) else np.zeros((0, 3), dtype=np.float32)
        l_pts = label_pts[i]
        l_pts = l_pts[np.sum(l_pts, -1, keepdims=False) > -2e1]

        p_to_l = {}
        if len(p_pts) > 0 and len(l_pts) > 0:
            vec_a = np.sum(p_pts ** 2, -1)
            vec_b = np.sum(l_pts ** 2, -1)
            dist_matrix = vec_a.reshape(-1, 1) + vec_b.reshape(1, -1) - 2 * np.matmul(p_pts, np.transpose(l_pts))
            dist_matrix = np.sqrt(np.maximum(dist_matrix, 0.0) + 1e-6)
            p_ind, l_ind = linear_sum_assignment(dist_matrix)
            mask = dist_matrix[p_ind, l_ind] < point_match_thresh
            tp_ind, tl_ind = p_ind[mask], l_ind[mask]
            dis = np.abs(((p_pts[tp_ind] * delta_pt) + min_pt) - ((l_pts[tl_ind] * delta_pt) + min_pt))
            p_to_l = {int(p): int(l) for p, l in zip(tp_ind, tl_ind)}
            statistics['tp_pts'] += len(tp_ind)
            statistics['pts_bias'] += np.sum(dis, 0)

        statistics['num_label_pts'] += len(l_pts)
        statistics['num_pred_pts'] += len(p_pts)

        gt_edges = label_edges[i]
        gt_edges = gt_edges[np.sum(gt_edges, -1, keepdims=False) >= 0]
        gt_edge_set = {tuple(sorted((int(e[0]), int(e[1])))) for e in gt_edges}

        num_pairs = len(p_pts) * (len(p_pts) - 1) // 2
        sample_pairs = pair_points[edge_offset: edge_offset + num_pairs]
        sample_scores = edge_scores[edge_offset: edge_offset + num_pairs]
        edge_offset += num_pairs

        pred_mask = sample_scores > edge_thresh
        pred_pairs = sample_pairs[pred_mask] if len(sample_pairs) else np.zeros((0, 2), dtype=np.int64)
        tp_edges = 0
        for a, b in pred_pairs:
            a, b = int(a), int(b)
            if a in p_to_l and b in p_to_l:
                if tuple(sorted((p_to_l[a], p_to_l[b]))) in gt_edge_set:
                    tp_edges += 1

        statistics['tp_edges'] += tp_edges
        statistics['num_label_edges'] += len(gt_edge_set)
        statistics['num_pred_edges'] += len(pred_pairs)


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        # 1. 处理 Tensor (最常见情况)
        if isinstance(val, torch.Tensor):
            batch_dict[key] = val.cuda().contiguous()
        
        # 2. 处理 Numpy (防止某些字段没转成 Tensor)
        elif isinstance(val, np.ndarray):
            # 注意：某些字段如索引需要是 long 型，坐标需要是 float 型
            if np.issubdtype(val.dtype, np.integer):
                batch_dict[key] = torch.from_numpy(val).long().cuda()
            else:
                batch_dict[key] = torch.from_numpy(val).float().cuda()
        
        # 3. 处理 List (Point2Roof 的 edges 经常以列表形式存储)
        elif isinstance(val, list):
            batch_dict[key] = [v.cuda() if isinstance(v, torch.Tensor) else v for v in val]
            
    return batch_dict

def load_data_to_cpu(batch_dict):
    for key, val in batch_dict.items():
        # 1. 处理 Tensor
        if isinstance(val, torch.Tensor):
            batch_dict[key] = val.detach().cpu().numpy()
        
        # 2. 处理 List (如果有的话)
        elif isinstance(val, list):
            batch_dict[key] = [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in val]
            
    return batch_dict


# ============================================================================
# 可视化功能函数
# ============================================================================

def save_wireframe_obj(vertices, edges, obj_path):
    """
    将顶点和边保存为 OBJ 格式的线框模型
    
    Args:
        vertices: Nx3 顶点坐标 (numpy array 或 torch tensor)
        edges: Mx2 边索引 (numpy array 或 torch tensor)
        obj_path: 输出 OBJ 文件路径
    """
    # 转换为 numpy
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
    if isinstance(edges, torch.Tensor):
        edges = edges.cpu().numpy()
    
    with open(obj_path, 'w') as f:
        # 写入顶点
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # 写入边 (OBJ 中边使用 'l' 前缀，索引从1开始)
        for e in edges:
            f.write(f"l {e[0]+1} {e[1]+1}\n")
    
    print(f"已保存线框模型: {obj_path}")


def save_roof_ply(vertices, edges, point_cloud, ply_path, gt_color=[0, 255, 0], pred_color=[255, 0, 0]):
    """
    将 Roof 模型保存为 PLY 格式，包含顶点颜色
    
    Args:
        vertices: Nx3 顶点坐标
        edges: Mx2 边索引
        point_cloud: 点云数据 (可选，用于可视化)
        ply_path: 输出 PLY 文件路径
        gt_color: 真值顶点颜色 [R, G, B]
        pred_color: 预测顶点颜色 [R, G, B]
    """
    import trimesh
    
    # 转换为 numpy
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
    if isinstance(edges, torch.Tensor):
        edges = edges.cpu().numpy()
    
    # 创建线框网格
    mesh = trimesh.Trimesh(vertices=vertices, edges=edges)
    
    # 添加颜色
    colors = np.zeros((len(vertices), 4), dtype=np.uint8)
    colors[:, :3] = pred_color
    colors[:, 3] = 255  # Alpha
    mesh.visual.vertex_colors = colors
    
    # 保存
    mesh.export(ply_path)
    print(f"已保存 PLY 模型: {ply_path}")


def visualize_3d_comparison(point_cloud, gt_vertices, gt_edges, pred_vertices, pred_edges, 
                           output_path, view_angle=(30, 45)):
    """
    生成真值与预测的 3D 对比可视化图
    
    Args:
        point_cloud: 点云数据 (Nx3)
        gt_vertices: 真值顶点 (Mx3)
        gt_edges: 真值边 (Kx2)
        pred_vertices: 预测顶点 (Px3)
        pred_edges: 预测边 (Qx2)
        output_path: 输出图片路径
        view_angle: 视角 (elev, azim)
    """
    fig = plt.figure(figsize=(16, 6))
    
    # 子图1: 点云 + 真值 Roof
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Ground Truth', fontsize=14)
    
    # 绘制点云 (灰色，透明度较低)
    if point_cloud is not None:
        ax1.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
                   c='lightgray', s=1, alpha=0.3)
    
    # 绘制真值 Roof (绿色)
    for edge in gt_edges:
        pts = gt_vertices[edge]
        ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'g-', linewidth=2)
    ax1.scatter(gt_vertices[:, 0], gt_vertices[:, 1], gt_vertices[:, 2], 
               c='green', s=30, marker='o')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # 子图2: 点云 + 预测 Roof
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Prediction', fontsize=14)
    
    # 绘制点云 (灰色，透明度较低)
    if point_cloud is not None:
        ax2.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
                   c='lightgray', s=1, alpha=0.3)
    
    # 绘制预测 Roof (红色)
    for edge in pred_edges:
        pts = pred_vertices[edge]
        ax2.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'r-', linewidth=2)
    ax2.scatter(pred_vertices[:, 0], pred_vertices[:, 1], pred_vertices[:, 2], 
               c='red', s=30, marker='o')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=view_angle[0], azim=view_angle[1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存对比图: {output_path}")


def visualize_multi_view(point_cloud, gt_vertices, gt_edges, pred_vertices, pred_edges, 
                        output_path):
    """
    生成多视角对比可视化图 (4个视角)
    
    Args:
        point_cloud: 点云数据 (Nx3)
        gt_vertices: 真值顶点 (Mx3)
        gt_edges: 真值边 (Kx2)
        pred_vertices: 预测顶点 (Px3)
        pred_edges: 预测边 (Qx2)
        output_path: 输出图片路径
    """
    fig = plt.figure(figsize=(16, 12))
    
    views = [(30, 0), (30, 90), (30, 180), (30, 270)]
    
    for idx, (elev, azim) in enumerate(views):
        # 真值
        ax = fig.add_subplot(4, 2, idx * 2 + 1, projection='3d')
        ax.set_title(f'Ground Truth - View {idx+1}', fontsize=10)
        
        if point_cloud is not None:
            ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
                      c='lightgray', s=0.5, alpha=0.2)
        
        for edge in gt_edges:
            pts = gt_vertices[edge]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'g-', linewidth=1.5)
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        
        # 预测
        ax = fig.add_subplot(4, 2, idx * 2 + 2, projection='3d')
        ax.set_title(f'Prediction - View {idx+1}', fontsize=10)
        
        if point_cloud is not None:
            ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
                      c='lightgray', s=0.5, alpha=0.2)
        
        for edge in pred_edges:
            pts = pred_vertices[edge]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'r-', linewidth=1.5)
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存多视角对比图: {output_path}")


def save_point_cloud_ply(point_cloud, ply_path, color=None):
    """
    保存点云为 PLY 格式
    
    Args:
        point_cloud: 点云数据 (Nx3)
        ply_path: 输出路径
        color: 可选颜色 [R, G, B] 或每个点的颜色 (Nx3)
    """
    import trimesh
    
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.cpu().numpy()
    
    # 创建点云
    pc = trimesh.PointCloud(vertices=point_cloud)
    
    # 添加颜色
    if color is not None:
        if isinstance(color, list) or (isinstance(color, np.ndarray) and color.ndim == 1):
            colors = np.tile(color, (len(point_cloud), 1))
        else:
            colors = np.array(color)
        pc.colors = np.hstack([colors, 255 * np.ones((len(colors), 1), dtype=np.uint8)])
    
    pc.export(ply_path)
    print(f"已保存点云: {ply_path}")


def process_and_visualize_sample(batch, sample_id, output_dir):
    """
    处理单个样本并生成所有可视化结果
    
    Args:
        batch: 模型输出 batch
        sample_id: 样本ID
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取数据
    pts_pred = batch['refined_keypoint']  # 预测的关键点
    pts_label = batch['vectors'][0]  # 真值关键点
    
    # 匹配点用于获取边
    edge_pred = batch['edge_score']
    
    # 保存 OBJ
    gt_obj_path = os.path.join(output_dir, f"{sample_id}_gt.obj")
    pred_obj_path = os.path.join(output_dir, f"{sample_id}_pred.obj")
    
    if isinstance(pts_label, torch.Tensor):
        pts_label = pts_label.cpu().numpy()
    
    # 保存真值 (边需要从原始数据获取)
    # 这里假设 batch 中包含边信息
    if 'edges' in batch:
        gt_edges = batch['edges'][0]
        if isinstance(gt_edges, torch.Tensor):
            gt_edges = gt_edges.cpu().numpy()
        save_wireframe_obj(pts_label, gt_edges, gt_obj_path)
    
    # 保存预测
    save_wireframe_obj(pts_label, np.zeros((0, 2), dtype=np.int32), pred_obj_path)
    
    # 生成对比图
    if 'point_clouds' in batch:
        point_cloud = batch['point_clouds']
        if isinstance(point_cloud, torch.Tensor):
            point_cloud = point_cloud.cpu().numpy()
        
        comparison_path = os.path.join(output_dir, f"{sample_id}_comparison.png")
        visualize_3d_comparison(
            point_cloud, 
            pts_label, 
            gt_edges if 'gt_edges' in dir() else np.zeros((0, 2), dtype=np.int32),
            pts_pred,
            np.zeros((0, 2), dtype=np.int32),
            comparison_path
        )
    
    print(f"样本 {sample_id} 可视化完成: {output_dir}")
