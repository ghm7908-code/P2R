import glob
import tqdm
import os
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
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


# ============================================================================
# APCalculator: precision / recall / F1 / ACO / WED (from evaluate_wireframe.py)
# ============================================================================

def parse_obj_index_static(token, num_vertices):
    raw = int(token.split("/")[0])
    if raw > 0:
        return raw - 1
    return num_vertices + raw


def hausdorff_distance_line(p_line, t_line, sample_points=20):
    """Compute Hausdorff distance matrix between two sets of line segments."""
    n_pred, n_gt = p_line.shape[0], t_line.shape[0]
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float64)

    all_lines = np.concatenate((p_line, t_line), axis=0)
    weights = np.linspace(0.0, 1.0, sample_points, dtype=np.float64).reshape(1, sample_points, 1)
    all_points = all_lines[:, 0, :][:, np.newaxis, :] + weights * (
        all_lines[:, 1, :][:, np.newaxis, :] - all_lines[:, 0, :][:, np.newaxis, :]
    )

    distance_matrix = cdist(
        all_points[:n_pred, :, :].reshape(-1, 3),
        all_points[n_pred:n_pred + n_gt, :, :].reshape(-1, 3),
        "euclidean",
    )
    distance_matrix = distance_matrix.reshape(n_pred, sample_points, n_gt, sample_points)
    distance_matrix = np.transpose(distance_matrix, axes=(0, 2, 1, 3))
    h_pt_value = distance_matrix.min(-1).max(-1, keepdims=True)
    h_tp_value = distance_matrix.min(-2).max(-1, keepdims=True)
    hausdorff_matrix = np.concatenate((h_pt_value, h_tp_value), axis=-1)
    hausdorff_matrix = hausdorff_matrix.max(-1)
    return hausdorff_matrix


def graph_edit_distance(pd_vertices, pd_edges, gt_vertices, gt_edges, wed_v):
    wed_e = 0.0
    if len(pd_vertices) > 0:
        distances = cdist(pd_vertices, gt_vertices)
        wed_v += float(np.min(distances, axis=1).sum())
        min_indices = np.argmin(distances, axis=1)
        pd_vertices = pd_vertices.copy()
        for i, index in enumerate(min_indices):
            pd_vertices[i] = gt_vertices[index]
        unique_pd_vertices = np.unique(pd_vertices, axis=0)
        renew_pd_edges = pd_edges.copy()
        for i, point in enumerate(unique_pd_vertices):
            v_indices = np.where((pd_vertices == point).all(axis=1))[0]
            for v_index in v_indices:
                renew_pd_edges[pd_edges == v_index] = i
        renew_pd_edges = np.unique(renew_pd_edges, axis=0)

        gt_edges_copy = gt_edges.copy()
        for edge in renew_pd_edges:
            e1_index = np.where((gt_vertices == unique_pd_vertices[edge[0]]).all(axis=1))[0]
            e2_index = np.where((gt_vertices == unique_pd_vertices[edge[1]]).all(axis=1))[0]
            if len(e1_index) == 0 or len(e2_index) == 0:
                wed_e += np.linalg.norm(unique_pd_vertices[edge[0]] - unique_pd_vertices[edge[1]])
                continue
            matched_edge = np.array(sorted([e1_index[0], e2_index[0]]))
            exists = np.where((gt_edges == matched_edge).all(axis=1))[0]
            if len(exists):
                mask = np.any(gt_edges_copy != matched_edge, axis=1)
                gt_edges_copy = gt_edges_copy[mask]
            else:
                wed_e += np.linalg.norm(unique_pd_vertices[edge[0]] - unique_pd_vertices[edge[1]])
    else:
        gt_edges_copy = gt_edges.copy()
        wed_v = 0.0

    for edge in gt_edges_copy:
        wed_e += np.linalg.norm(gt_vertices[edge[0]] - gt_vertices[edge[1]])

    sum_distance = 0.0
    for edge in gt_edges:
        sum_distance += np.linalg.norm(gt_vertices[edge[0]] - gt_vertices[edge[1]])

    if sum_distance <= 1e-12:
        return 0.0
    return float((wed_e + wed_v) / sum_distance)


def computer_edges_wed(edges, vertices):
    index = []
    for edge in edges:
        indices = []
        for point in edge:
            matching_indices = np.where((vertices == point).all(axis=1))[0]
            indices.append(matching_indices[0] if len(matching_indices) > 0 else -1)
        index.append(indices)
    if len(index) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.sort(np.asarray(index, dtype=np.int32), axis=-1)


def remove_corners(corner_a, corner_b):
    if len(corner_a) == 0:
        return corner_a.reshape(0, 3)
    if len(corner_b) == 0:
        return corner_a.copy()
    corner_a_view = corner_a.view([("", corner_a.dtype)] * corner_a.shape[1])
    corner_b_view = corner_b.view([("", corner_b.dtype)] * corner_b.shape[1])
    corner = np.setdiff1d(corner_a_view, corner_b_view).view(corner_a.dtype).reshape(-1, corner_a.shape[1])
    return corner


class APCalculator:
    """Precision/recall/F1 calculator with Hungarian matching and Hausdorff distance."""

    def __init__(self, distance_thresh=0.1, confidence_thresh=0.7):
        self.distance_thresh = distance_thresh
        self.confidence_thresh = confidence_thresh
        self.sample_count = 0
        self.reset()

    def compute_metrics(self, batch):
        batch_size = len(batch["predicted_vertices"])
        self.sample_count += batch_size

        batch_predicted_corners = batch["predicted_vertices"]
        batch_predicted_edges = batch["predicted_edges"]
        batch_pred_edges_vertices = batch["pred_edges_vertices"]
        batch_label_corners = batch["wf_vertices"]
        batch_label_edges = batch["wf_edges"]
        batch_label_edges_vertices = batch["wf_edges_vertices"]

        for b in range(batch_size):
            predicted_corners = np.asarray(batch_predicted_corners[b], dtype=np.float64).reshape(-1, 3)
            predicted_edges = np.asarray(batch_predicted_edges[b], dtype=np.int32).reshape(-1, 2)
            pred_edges_vertices = np.asarray(batch_pred_edges_vertices[b], dtype=np.float64).reshape(-1, 2, 3)
            label_corners = np.asarray(batch_label_corners[b], dtype=np.float64).reshape(-1, 3)
            label_edges = np.asarray(batch_label_edges[b], dtype=np.int32).reshape(-1, 2)
            label_edges_vertices = np.asarray(batch_label_edges_vertices[b], dtype=np.float64).reshape(-1, 2, 3)

            tp_edges = 0
            tp_fp_edges = len(predicted_edges)
            tp_fn_edges = len(label_edges)
            distances = 0.0

            pr_corners = np.zeros((0, 2, 3), dtype=np.float64)
            gt_corners = np.zeros((0, 2, 3), dtype=np.float64)
            matched_pred_edge_indices = np.zeros((0,), dtype=np.int32)
            matched_gt_edge_indices = np.zeros((0,), dtype=np.int32)

            if len(predicted_edges) != 0 and len(label_edges_vertices) != 0:
                edge_distance = hausdorff_distance_line(pred_edges_vertices, label_edges_vertices)
                predict_indices, label_indices = linear_sum_assignment(edge_distance)
                edge_mask = edge_distance[predict_indices, label_indices] <= self.distance_thresh
                matched_pred_edge_indices = predict_indices[edge_mask]
                matched_gt_edge_indices = label_indices[edge_mask]
                pr_corners = pred_edges_vertices[matched_pred_edge_indices]
                gt_corners = label_edges_vertices[matched_gt_edge_indices]
                tp_edges = int(edge_mask.sum())

                un_match_pr_corners = remove_corners(
                    predicted_corners, np.unique(pr_corners.reshape(-1, 3), axis=0) if len(pr_corners) else np.zeros((0, 3))
                )
                un_match_gt_corners = remove_corners(
                    label_corners, np.unique(gt_corners.reshape(-1, 3), axis=0) if len(gt_corners) else np.zeros((0, 3))
                )

                additional_corner_matches = 0
                if len(un_match_pr_corners) > 0 and len(un_match_gt_corners) > 0:
                    distance_matrix = cdist(un_match_pr_corners, un_match_gt_corners)
                    un_match_predict_indices, un_match_label_indices = linear_sum_assignment(distance_matrix)
                    un_match_mask = (
                        distance_matrix[un_match_predict_indices, un_match_label_indices] <= self.distance_thresh
                    )
                    distances += float(
                        distance_matrix[
                            un_match_predict_indices[un_match_mask],
                            un_match_label_indices[un_match_mask],
                        ].sum()
                    )
                    additional_corner_matches = int(un_match_mask.sum())

                matched_pred_vertices = (
                    np.unique(pr_corners.reshape(-1, 3), axis=0) if len(pr_corners) else np.zeros((0, 3), dtype=np.float64)
                )
                matched_gt_vertices = (
                    np.unique(gt_corners.reshape(-1, 3), axis=0) if len(gt_corners) else np.zeros((0, 3), dtype=np.float64)
                )

                tp_corners = len(matched_pred_vertices) + additional_corner_matches
                tp_fp_corners = len(predicted_corners)
                tp_fn_corners = len(label_corners)

                if len(matched_pred_vertices) > 0 and len(matched_gt_vertices) > 0:
                    distance_matrix = cdist(matched_pred_vertices, matched_gt_vertices)
                    distances += float(np.min(distance_matrix, axis=1).sum())

                if len(matched_pred_edge_indices) > 0:
                    predicted_corners_for_wed = np.unique(label_edges_vertices.reshape(-1, 3), axis=0)
                    submission_edges = computer_edges_wed(label_edges_vertices, predicted_corners_for_wed)
                    wed = graph_edit_distance(
                        predicted_corners_for_wed,
                        submission_edges.copy(),
                        label_corners.copy(),
                        label_edges.copy(),
                        distances,
                    )
                else:
                    wed = graph_edit_distance(
                        np.zeros((0, 3), dtype=np.float64),
                        np.zeros((0, 2), dtype=np.int32),
                        label_corners.copy(),
                        label_edges.copy(),
                        distances,
                    )

            else:
                if len(predicted_corners) > 0 and len(label_corners) > 0:
                    distance_matrix = cdist(predicted_corners, label_corners)
                    predict_indices, label_indices = linear_sum_assignment(distance_matrix)
                    mask = distance_matrix[predict_indices, label_indices] <= self.distance_thresh
                    distances = float(distance_matrix[predict_indices[mask], label_indices[mask]].sum())
                    tp_corners = int(mask.sum())
                else:
                    distances = 0.0
                    tp_corners = 0

                tp_fp_corners = len(predicted_corners)
                tp_fn_corners = len(label_corners)
                tp_edges = 0
                tp_fp_edges = 0
                tp_fn_edges = len(label_edges)
                wed = 1.0

            self.ap_dict["tp_corners"] += tp_corners
            self.ap_dict["tp_fp_corners"] += tp_fp_corners
            self.ap_dict["tp_fn_corners"] += tp_fn_corners
            self.ap_dict["distance"] += distances
            self.ap_dict["wed"] += wed
            self.ap_dict["tp_edges"] += tp_edges
            self.ap_dict["tp_fp_edges"] += tp_fp_edges
            self.ap_dict["tp_fn_edges"] += tp_fn_edges

    def output_accuracy(self):
        self.ap_dict["average_corner_offset"] = _safe_div(self.ap_dict["distance"], self.ap_dict["tp_corners"])
        self.ap_dict["average_wed"] = _safe_div(self.ap_dict["wed"], self.sample_count)
        self.ap_dict["corners_precision"] = _safe_div(self.ap_dict["tp_corners"], self.ap_dict["tp_fp_corners"])
        self.ap_dict["corners_recall"] = _safe_div(self.ap_dict["tp_corners"], self.ap_dict["tp_fn_corners"])

        cp = self.ap_dict["corners_precision"]
        cr = self.ap_dict["corners_recall"]
        self.ap_dict["corners_f1"] = _safe_div(2 * cp * cr, cp + cr)

        self.ap_dict["edges_precision"] = _safe_div(self.ap_dict["tp_edges"], self.ap_dict["tp_fp_edges"])
        self.ap_dict["edges_recall"] = _safe_div(self.ap_dict["tp_edges"], self.ap_dict["tp_fn_edges"])

        ep = self.ap_dict["edges_precision"]
        er = self.ap_dict["edges_recall"]
        self.ap_dict["edges_f1"] = _safe_div(2 * ep * er, ep + er)

        return {
            "ACO": self.ap_dict["average_corner_offset"],
            "WED": self.ap_dict["average_wed"],
            "CP": self.ap_dict["corners_precision"],
            "CR": self.ap_dict["corners_recall"],
            "CF1": self.ap_dict["corners_f1"],
            "EP": self.ap_dict["edges_precision"],
            "ER": self.ap_dict["edges_recall"],
            "EF1": self.ap_dict["edges_f1"],
            "support_samples": self.sample_count,
            "tp_corners": self.ap_dict["tp_corners"],
            "tp_fp_corners": self.ap_dict["tp_fp_corners"],
            "tp_fn_corners": self.ap_dict["tp_fn_corners"],
            "tp_edges": self.ap_dict["tp_edges"],
            "tp_fp_edges": self.ap_dict["tp_fp_edges"],
            "tp_fn_edges": self.ap_dict["tp_fn_edges"],
        }

    def reset(self):
        self.ap_dict = {
            "tp_corners": 0,
            "tp_fp_corners": 0,
            "tp_fn_corners": 0,
            "distance": 0.0,
            "tp_edges": 0,
            "wed": 0.0,
            "tp_fp_edges": 0,
            "tp_fn_edges": 0,
            "average_corner_offset": 0.0,
            "corners_precision": 0.0,
            "corners_recall": 0.0,
            "corners_f1": 0.0,
            "edges_precision": 0.0,
            "edges_recall": 0.0,
            "edges_f1": 0.0,
        }
        self.sample_count = 0


# ============================================================================
# 主要测试函数
# ============================================================================

def test_model(model, data_loader, logger, edge_thresh=0.5, point_match_thresh=0.1,
               ap_distance_thresh=0.1):
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

    # Initialize APCalculator for comprehensive metrics
    ap_calculator = APCalculator(distance_thresh=ap_distance_thresh)

    dataloader_iter = iter(data_loader)
    with tqdm.trange(0, len(data_loader), desc='test', dynamic_ncols=True) as tbar:
        for _ in tbar:
            batch = next(dataloader_iter)
            load_data_to_gpu(batch)
            with torch.no_grad():
                batch = model(batch)
                load_data_to_cpu(batch)
            eval_process(batch, statistics, edge_thresh=edge_thresh, point_match_thresh=point_match_thresh)
            eval_process_ap(batch, ap_calculator, edge_thresh=edge_thresh)

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

    # Merge APCalculator metrics
    ap_metrics = ap_calculator.output_accuracy()
    metrics.update(ap_metrics)

    logger.info('========== Simple Matching Metrics ==========')
    logger.info('pts_recall: %f', metrics['pts_recall'])
    logger.info('pts_precision: %f', metrics['pts_precision'])
    logger.info('pts_bias: %f, %f, %f', bias[0], bias[1], bias[2])
    logger.info('edge_recall: %f', metrics['edge_recall'])
    logger.info('edge_precision: %f', metrics['edge_precision'])
    logger.info('========== APCalculator Metrics (Hungarian + Hausdorff) ==========')
    logger.info('Corner Precision (CP): %f', metrics['CP'])
    logger.info('Corner Recall (CR): %f', metrics['CR'])
    logger.info('Corner F1 (CF1): %f', metrics['CF1'])
    logger.info('Edge Precision (EP): %f', metrics['EP'])
    logger.info('Edge Recall (ER): %f', metrics['ER'])
    logger.info('Edge F1 (EF1): %f', metrics['EF1'])
    logger.info('Average Corner Offset (ACO): %f', metrics['ACO'])
    logger.info('Wireframe Edit Distance (WED): %f', metrics['WED'])
    logger.info('Support samples: %d', metrics['support_samples'])
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


def eval_process_ap(batch, ap_calculator, edge_thresh=0.5):
    """
    Build per-sample predictions in the APCalculator batch format and
    accumulate comprehensive metrics (CP/CR/CF1, EP/ER/EF1, ACO, WED).
    """
    batch_size = batch['batch_size']
    keypoints = batch.get('keypoint', np.zeros((0, 4), dtype=np.float32))
    refined_pts = batch.get('refined_keypoint', np.zeros((0, 3), dtype=np.float32))
    label_pts = batch['vectors']
    edge_scores = batch.get('edge_score', np.zeros((0,), dtype=np.float32))
    pair_points = batch.get('pair_points', np.zeros((0, 2), dtype=np.int64))
    label_edges = batch['edges']

    predicted_vertices_list = []
    predicted_edges_list = []
    pred_edges_vertices_list = []
    wf_vertices_list = []
    wf_edges_list = []
    wf_edges_vertices_list = []

    edge_offset = 0
    for i in range(batch_size):
        # --- GT data ---
        l_pts = label_pts[i]
        l_pts = l_pts[np.sum(l_pts, -1, keepdims=False) > -2e1]
        l_edges = label_edges[i]
        l_edges = l_edges[np.sum(l_edges, -1, keepdims=False) >= 0]

        if len(l_pts) > 0 and len(l_edges) > 0:
            gt_edge_verts = np.stack([l_pts[l_edges[:, 0]], l_pts[l_edges[:, 1]]], axis=1)
        elif len(l_pts) > 0:
            gt_edge_verts = np.zeros((0, 2, 3), dtype=np.float64)
        else:
            gt_edge_verts = np.zeros((0, 2, 3), dtype=np.float64)

        wf_vertices_list.append(l_pts)
        wf_edges_list.append(l_edges)
        wf_edges_vertices_list.append(gt_edge_verts)

        # --- Prediction data ---
        p_pts = refined_pts[keypoints[:, 0] == i] if len(keypoints) else np.zeros((0, 3), dtype=np.float32)
        num_pairs = len(p_pts) * (len(p_pts) - 1) // 2
        sample_pairs = pair_points[edge_offset: edge_offset + num_pairs]
        sample_scores = edge_scores[edge_offset: edge_offset + num_pairs]
        edge_offset += num_pairs

        pred_mask = sample_scores > edge_thresh
        pred_edges = sample_pairs[pred_mask] if len(sample_pairs) else np.zeros((0, 2), dtype=np.int64)

        if len(p_pts) > 0 and len(pred_edges) > 0:
            pred_edge_verts = np.stack([p_pts[pred_edges[:, 0]], p_pts[pred_edges[:, 1]]], axis=1)
        elif len(p_pts) > 0:
            pred_edge_verts = np.zeros((0, 2, 3), dtype=np.float64)
        else:
            pred_edge_verts = np.zeros((0, 2, 3), dtype=np.float64)

        predicted_vertices_list.append(p_pts)
        predicted_edges_list.append(pred_edges)
        pred_edges_vertices_list.append(pred_edge_verts)

    ap_batch = {
        "predicted_vertices": predicted_vertices_list,
        "predicted_edges": predicted_edges_list,
        "pred_edges_vertices": pred_edges_vertices_list,
        "wf_vertices": wf_vertices_list,
        "wf_edges": wf_edges_list,
        "wf_edges_vertices": wf_edges_vertices_list,
    }
    ap_calculator.compute_metrics(ap_batch)


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


def visualize_predictions(model, data_loader, output_dir, logger,
                          vis_sample_id=None, vis_all=False, edge_thresh=0.5):
    """
    Visualize model predictions for specified samples or all samples.

    Args:
        model: RoofNet model in eval mode
        data_loader: DataLoader (batch_size=1 recommended)
        output_dir: Directory to save visualizations
        logger: Logger instance
        vis_sample_id: Specific sample ID to visualize (e.g. 'sample_001')
        vis_all: If True, visualize all samples
        edge_thresh: Edge score threshold
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    model.use_edge = True

    dataloader_iter = iter(data_loader)
    vis_count = 0

    with torch.no_grad():
        for batch_idx in range(len(data_loader)):
            batch = next(dataloader_iter)
            load_data_to_gpu(batch)
            batch = model(batch)
            load_data_to_cpu(batch)

            sample_ids = batch.get('frame_id', [str(batch_idx)])
            for i in range(batch['batch_size']):
                sid = sample_ids[i] if i < len(sample_ids) else str(batch_idx)

                # Filter by vis_sample_id if specified
                if vis_sample_id is not None and not vis_all:
                    if sid != vis_sample_id:
                        continue

                # Extract data for this sample
                keypoints = batch.get('keypoint', np.zeros((0, 4), dtype=np.float32))
                refined_pts = batch.get('refined_keypoint', np.zeros((0, 3), dtype=np.float32))
                edge_scores = batch.get('edge_score', np.zeros((0,), dtype=np.float32))
                pair_points = batch.get('pair_points', np.zeros((0, 2), dtype=np.int64))
                label_pts = batch['vectors'][i]
                label_pts = label_pts[np.sum(label_pts, -1, keepdims=False) > -2e1]
                label_edges = batch['edges'][i]
                label_edges = label_edges[np.sum(label_edges, -1, keepdims=False) >= 0]
                point_cloud = batch.get('points')

                # Predicted vertices for this sample
                p_pts = refined_pts[keypoints[:, 0] == i] if len(keypoints) else np.zeros((0, 3), dtype=np.float32)

                # Predicted edges for this sample
                num_pairs = len(p_pts) * (len(p_pts) - 1) // 2
                # Find the correct edge offset for this sample
                edge_offset = 0
                for prev_i in range(i):
                    prev_pts = refined_pts[keypoints[:, 0] == prev_i] if len(keypoints) else np.zeros((0, 3))
                    edge_offset += len(prev_pts) * (len(prev_pts) - 1) // 2

                sample_pairs = pair_points[edge_offset: edge_offset + num_pairs]
                sample_scores = edge_scores[edge_offset: edge_offset + num_pairs]
                pred_mask = sample_scores > edge_thresh
                pred_edges = sample_pairs[pred_mask] if len(sample_pairs) else np.zeros((0, 2), dtype=np.int64)

                # Save OBJ files
                gt_obj_path = output_dir / f"{sid}_gt.obj"
                pred_obj_path = output_dir / f"{sid}_pred.obj"
                save_wireframe_obj(label_pts, label_edges, str(gt_obj_path))
                save_wireframe_obj(p_pts, pred_edges, str(pred_obj_path))
                logger.info("Exported %s and %s", gt_obj_path, pred_obj_path)

                # Generate comparison image
                if point_cloud is not None:
                    if isinstance(point_cloud, torch.Tensor):
                        pc = point_cloud.cpu().numpy()
                    elif isinstance(point_cloud, np.ndarray):
                        if point_cloud.ndim == 3:
                            pc = point_cloud[i]
                        else:
                            pc = point_cloud
                    else:
                        pc = point_cloud

                    comparison_path = output_dir / f"{sid}_comparison.png"
                    visualize_3d_comparison(
                        pc, label_pts, label_edges, p_pts, pred_edges,
                        str(comparison_path)
                    )

                vis_count += 1
                if vis_sample_id is not None and not vis_all:
                    logger.info("Visualization complete for sample: %s", sid)
                    return

    logger.info("Visualization complete: %d samples saved to %s", vis_count, output_dir)
