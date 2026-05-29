import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate wireframe OBJ predictions with the same matching logic as ap_calculator.py."
    )
    parser.add_argument("--pred_dir", required=True, help="Directory containing predicted wireframe .obj files")
    parser.add_argument("--gt_dir", required=True, help="Directory containing ground-truth wireframe .obj files")
    parser.add_argument("--pred_ext", default=".obj", help="Prediction file extension")
    parser.add_argument("--gt_ext", default=".obj", help="Ground-truth file extension")
    parser.add_argument("--names_file", default="", help="Optional text file of sample ids to evaluate")
    parser.add_argument("--distance_thresh", default=0.1, type=float,
                        help="Distance threshold used by the original APCalculator matching logic")
    parser.add_argument("--confidence_thresh", default=0.7, type=float,
                        help="Compatibility argument from APCalculator; currently not used by the original logic")
    parser.add_argument("--output_json", default="", help="Optional JSON output path")
    parser.add_argument("--output_csv", default="", help="Optional CSV output path")
    return parser.parse_args()


def parse_obj_index(token, num_vertices):
    raw = int(token.split("/")[0])
    if raw > 0:
        return raw - 1
    return num_vertices + raw


def load_wireframe_obj(obj_path):
    vertices = []
    edges = []

    with open(obj_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            tag = parts[0]

            if tag == "v":
                vertices.append([float(v) for v in parts[1:4]])
                continue

            if tag == "l":
                ids = [parse_obj_index(token, len(vertices)) for token in parts[1:]]
                for start, end in zip(ids[:-1], ids[1:]):
                    if start != end:
                        edges.append(sorted((start, end)))

    vertices = np.asarray(vertices, dtype=np.float64)
    if len(edges) == 0:
        edges = np.zeros((0, 2), dtype=np.int32)
    else:
        edges = np.unique(np.asarray(edges, dtype=np.int32), axis=0)

    if len(edges) == 0 or len(vertices) == 0:
        edge_vertices = np.zeros((0, 2, 3), dtype=np.float64)
    else:
        edge_vertices = vertices[edges]

    return {
        "vertices": vertices.reshape(-1, 3),
        "edges": edges.reshape(-1, 2),
        "edge_vertices": edge_vertices.reshape(-1, 2, 3),
    }


def resolve_pairs(pred_dir, gt_dir, pred_ext, gt_ext, names_file):
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    pred_map = {path.stem: path for path in pred_dir.glob(f"*{pred_ext}")}
    gt_map = {path.stem: path for path in gt_dir.glob(f"*{gt_ext}")}

    if names_file:
        with open(names_file, "r", encoding="utf-8") as handle:
            selected = {line.strip() for line in handle if line.strip()}
        pred_map = {name: path for name, path in pred_map.items() if name in selected}
        gt_map = {name: path for name, path in gt_map.items() if name in selected}

    common_names = sorted(pred_map.keys() & gt_map.keys())
    missing_pred = sorted(gt_map.keys() - pred_map.keys())
    missing_gt = sorted(pred_map.keys() - gt_map.keys())

    return [(name, pred_map[name], gt_map[name]) for name in common_names], missing_pred, missing_gt


def hausdorff_distance_line(p_line, t_line, sample_points=20):
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


def computer_edges(edges, vertices):
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


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


class APCalculator:
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
                    adjusted_pred_edges_vertices = pred_edges_vertices.copy()
                    adjusted_pred_edges_vertices[matched_pred_edge_indices] = label_edges_vertices[matched_gt_edge_indices]

                    # Keep the original WED construction logic from ap_calculator.py.
                    predicted_corners_for_wed = np.unique(label_edges_vertices.reshape(-1, 3), axis=0)
                    submission_edges = computer_edges(label_edges_vertices, predicted_corners_for_wed)
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
        self.ap_dict["average_corner_offset"] = safe_divide(self.ap_dict["distance"], self.ap_dict["tp_corners"])
        self.ap_dict["average_wed"] = safe_divide(self.ap_dict["wed"], self.sample_count)
        self.ap_dict["corners_precision"] = safe_divide(self.ap_dict["tp_corners"], self.ap_dict["tp_fp_corners"])
        self.ap_dict["corners_recall"] = safe_divide(self.ap_dict["tp_corners"], self.ap_dict["tp_fn_corners"])

        cp = self.ap_dict["corners_precision"]
        cr = self.ap_dict["corners_recall"]
        self.ap_dict["corners_f1"] = safe_divide(2 * cp * cr, cp + cr)

        self.ap_dict["edges_precision"] = safe_divide(self.ap_dict["tp_edges"], self.ap_dict["tp_fp_edges"])
        self.ap_dict["edges_recall"] = safe_divide(self.ap_dict["tp_edges"], self.ap_dict["tp_fn_edges"])

        ep = self.ap_dict["edges_precision"]
        er = self.ap_dict["edges_recall"]
        self.ap_dict["edges_f1"] = safe_divide(2 * ep * er, ep + er)

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


def build_batch(pred_obj, gt_obj):
    pred = load_wireframe_obj(pred_obj)
    gt = load_wireframe_obj(gt_obj)
    return {
        "predicted_vertices": [pred["vertices"]],
        "predicted_edges": [pred["edges"]],
        "pred_edges_vertices": [pred["edge_vertices"]],
        "wf_vertices": [gt["vertices"]],
        "wf_edges": [gt["edges"]],
        "wf_edges_vertices": [gt["edge_vertices"]],
    }


def write_csv(output_csv, results):
    fieldnames = [
        "name",
        "ACO",
        "WED",
        "CP",
        "CR",
        "CF1",
        "EP",
        "ER",
        "EF1",
        "support_samples",
        "tp_corners",
        "tp_fp_corners",
        "tp_fn_corners",
        "tp_edges",
        "tp_fp_edges",
        "tp_fn_edges",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    args = parse_args()

    pairs, missing_pred, missing_gt = resolve_pairs(
        args.pred_dir, args.gt_dir, args.pred_ext, args.gt_ext, args.names_file
    )
    if not pairs:
        raise RuntimeError("No matched prediction/ground-truth OBJ pairs were found.")

    overall_calculator = APCalculator(
        distance_thresh=args.distance_thresh,
        confidence_thresh=args.confidence_thresh,
    )

    per_file_results = []
    for index, (name, pred_path, gt_path) in enumerate(pairs, start=1):
        batch = build_batch(pred_path, gt_path)
        overall_calculator.compute_metrics(batch)

        single_calculator = APCalculator(
            distance_thresh=args.distance_thresh,
            confidence_thresh=args.confidence_thresh,
        )
        single_calculator.compute_metrics(batch)
        result = {"name": name}
        result.update(single_calculator.output_accuracy())
        per_file_results.append(result)

        if index % 50 == 0 or index == len(pairs):
            print(f"Processed {index}/{len(pairs)} pairs")

    summary = overall_calculator.output_accuracy()

    print("\nSummary")
    print(f"pairs: {len(pairs)}")
    print(f"ACO: {summary['ACO']}")
    print(f"CP: {summary['CP']}")
    print(f"CR: {summary['CR']}")
    print(f"CF1: {summary['CF1']}")
    print(f"EP: {summary['EP']}")
    print(f"ER: {summary['ER']}")
    print(f"EF1: {summary['EF1']}")
    print(f"WED: {summary['WED']}")

    if missing_pred:
        print(f"Warning: {len(missing_pred)} GT files have no matching prediction.")
    if missing_gt:
        print(f"Warning: {len(missing_gt)} prediction files have no matching GT.")

    if args.output_csv:
        write_csv(args.output_csv, per_file_results)
        print(f"Wrote CSV to {args.output_csv}")

    if args.output_json:
        payload = {
            "summary": summary,
            "missing_predictions": missing_pred,
            "missing_ground_truth": missing_gt,
            "per_file": per_file_results,
        }
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote JSON to {args.output_json}")


if __name__ == "__main__":
    main()
