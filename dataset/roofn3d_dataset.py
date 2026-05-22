import os
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset


def _cfg_get(cfg, key, default=None):
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def read_pts(pts_file):
    pts = np.loadtxt(pts_file, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] < 3:
        raise ValueError(f"Point file must contain at least xyz columns: {pts_file}")
    return pts[:, :3]


def _parse_obj_index(token):
    token = token.split("/")[0]
    if not token:
        raise ValueError("empty OBJ index")
    return int(token) - 1


def load_obj(obj_file):
    vertices, edges = [], set()
    with open(obj_file, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            tag = vals[0].lower()
            if tag == "v" and len(vals) >= 4:
                vertices.append(vals[1:4])
            elif tag in {"f", "l"} and len(vals) >= 3:
                ids = [_parse_obj_index(v) for v in vals[1:]]
                if len(ids) == 2:
                    edges.add(tuple(sorted(ids)))
                else:
                    for a, b in zip(ids, ids[1:] + ids[:1]):
                        edges.add(tuple(sorted((a, b))))

    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.size == 0:
        raise ValueError(f"No vertices found in OBJ file: {obj_file}")

    if edges:
        edges = np.asarray(sorted(edges), dtype=np.int64)
    else:
        edges = np.zeros((0, 2), dtype=np.int64)
    return vertices, edges


def _choose_obj_for_xyz(xyz_path):
    parent = xyz_path.parent
    split_dir = parent.parent if parent.name.lower() in {"xyz", "points", "pointcloud", "point_clouds"} else parent
    preferred = [
        parent / "polygon.obj",
        parent / "framework.obj",
        parent / "wireframe.obj",
        parent / f"{xyz_path.stem}.obj",
        split_dir / "gt" / f"{xyz_path.stem}.obj",
        split_dir / "gts" / f"{xyz_path.stem}.obj",
        split_dir / "gt_obj" / f"{xyz_path.stem}.obj",
        split_dir / "gt_objs" / f"{xyz_path.stem}.obj",
        split_dir / "gt_framework" / f"{xyz_path.stem}.obj",
        split_dir / "gt_frameworks" / f"{xyz_path.stem}.obj",
    ]
    for obj_path in preferred:
        if obj_path.exists():
            return obj_path

    obj_files = sorted(parent.glob("*.obj"))
    if len(obj_files) == 1:
        return obj_files[0]
    return None


def _sample_id_from_xyz(xyz_path):
    if xyz_path.stem.lower() in {"points", "point", "cloud"}:
        return xyz_path.parent.name
    return xyz_path.stem


def _sample_from_directory(dir_path, sample_id=None):
    xyz_path = dir_path / "points.xyz"
    if not xyz_path.exists():
        xyz_files = sorted(dir_path.glob("*.xyz"))
        xyz_path = xyz_files[0] if xyz_files else xyz_path
    obj_path = _choose_obj_for_xyz(xyz_path)
    return _sample_from_paths(xyz_path, obj_path, sample_id or dir_path.name)


def _resolve_list_path(item, list_dir):
    path = Path(item)
    if path.is_absolute():
        return path
    return list_dir / path


def _sample_from_paths(xyz_path, obj_path=None, sample_id=None):
    xyz_path = Path(xyz_path)
    obj_path = Path(obj_path) if obj_path is not None else _choose_obj_for_xyz(xyz_path)
    if xyz_path.exists() and obj_path is not None and obj_path.exists():
        return {
            "sample_id": sample_id or _sample_id_from_xyz(xyz_path),
            "xyz_path": str(xyz_path),
            "obj_path": str(obj_path),
        }
    return None


def _infer_split_from_list_name(list_name):
    lower_name = list_name.lower()
    if "valid" in lower_name or "val" in lower_name:
        return "val"
    if "test" in lower_name:
        return "test"
    if "train" in lower_name:
        return "train"
    return None


def _candidate_roots_for_list(list_dir):
    roots = [list_dir]
    lower_name = list_dir.name.lower()
    if lower_name.startswith("bwformer") or lower_name.startswith("processed"):
        roots.append(list_dir.parent)
    return roots


def _obj_candidates_for_stem(split_dir, stem, xyz_path):
    parent = xyz_path.parent
    candidates = [
        parent / f"{stem}.obj",
        parent / "polygon.obj",
        parent / "framework.obj",
        parent / "wireframe.obj",
        split_dir / f"{stem}.obj",
    ]
    obj_subdirs = [
        "obj",
        "objs",
        "object",
        "objects",
        "gt",
        "gts",
        "gt_obj",
        "gt_objs",
        "gt_framework",
        "gt_frameworks",
        "framework",
        "frameworks",
        "wireframe",
        "wireframes",
        "polygon",
        "polygons",
        "annot",
        "annots",
        "label",
        "labels",
    ]
    for subdir in obj_subdirs:
        candidates.append(split_dir / subdir / f"{stem}.obj")
    return candidates


def _sample_from_stem(split_dir, stem):
    xyz_candidates = [
        split_dir / f"{stem}.xyz",
        split_dir / stem / "points.xyz",
        split_dir / stem / f"{stem}.xyz",
    ]
    xyz_subdirs = [
        "xyz",
        "points",
        "point",
        "pointcloud",
        "point_cloud",
        "pointclouds",
        "point_clouds",
        "pc",
        "pcs",
        "pts",
    ]
    for subdir in xyz_subdirs:
        xyz_candidates.append(split_dir / subdir / f"{stem}.xyz")

    for xyz_path in xyz_candidates:
        if not xyz_path.exists():
            continue
        sample = _sample_from_paths(xyz_path, sample_id=stem)
        if sample is not None:
            return sample
        for obj_path in _obj_candidates_for_stem(split_dir, stem, xyz_path):
            sample = _sample_from_paths(xyz_path, obj_path, stem)
            if sample is not None:
                return sample
    return None


def _sample_from_raw_identifier(line, list_dir, list_name):
    split = _infer_split_from_list_name(list_name)
    split_names = [split] if split is not None else ["train", "val", "test"]
    item = Path(line)
    item_without_suffix = item.with_suffix("") if item.suffix in {".npy", ".pkl"} else item
    candidate_items = [item]
    if item_without_suffix != item:
        candidate_items.append(item_without_suffix)

    for root in _candidate_roots_for_list(list_dir):
        for split_name in split_names:
            split_dirs = [root / split_name]
            if root == list_dir:
                split_dirs.append(root)
            for split_dir in split_dirs:
                if not split_dir.exists():
                    continue
                for candidate_item in candidate_items:
                    candidate = split_dir / candidate_item
                    if candidate.is_dir():
                        sample = _sample_from_directory(candidate, candidate.name)
                        if sample is not None:
                            return sample

                    if candidate.suffix == ".xyz":
                        sample = _sample_from_paths(candidate, sample_id=candidate.stem)
                        if sample is not None:
                            return sample
                    else:
                        sample = _sample_from_paths(candidate.with_suffix(".xyz"), sample_id=candidate.name)
                        if sample is not None:
                            return sample
                        sample = _sample_from_directory(candidate, candidate.name)
                        if sample is not None:
                            return sample

                    stem = candidate_item.stem if candidate_item.suffix else candidate_item.name
                    sample = _sample_from_stem(split_dir, stem)
                    if sample is not None:
                        return sample
    return None


def _sample_from_list_line(line, list_dir, list_name):
    if line.startswith("{"):
        item = json.loads(line)
        xyz_path = _resolve_list_path(item["xyz_path"], list_dir)
        obj_path = _resolve_list_path(item["obj_path"], list_dir) if item.get("obj_path") else None
        return _sample_from_paths(xyz_path, obj_path, item.get("sample_id"))

    delimiter = "\t" if "\t" in line else "," if "," in line else None
    if delimiter is not None:
        parts = [part.strip() for part in line.split(delimiter) if part.strip()]
        if len(parts) >= 2:
            return _sample_from_paths(_resolve_list_path(parts[0], list_dir), _resolve_list_path(parts[1], list_dir))

    item_path = _resolve_list_path(line, list_dir)
    if item_path.is_dir():
        return _sample_from_directory(item_path, item_path.name)

    sample = _sample_from_paths(item_path)
    if sample is not None:
        return sample
    return _sample_from_raw_identifier(line, list_dir, list_name)


def discover_samples(data_path):
    path = Path(data_path)
    if path.is_file():
        samples = []
        missing = []
        list_dir = path.parent
        with open(path, "r") as f:
            for line in f:
                item = line.strip()
                if not item or item.startswith("#"):
                    continue
                sample = _sample_from_list_line(item, list_dir, path.name)
                if sample is not None:
                    samples.append(sample)
                else:
                    missing.append(item)
        if missing:
            preview = ", ".join(missing[:5])
            warnings.warn(
                f"Skipped {len(missing)} unresolved entries from {path}. "
                f"First unresolved entries: {preview}",
                RuntimeWarning,
            )
        return samples

    if not path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

    samples = []
    seen = set()
    for xyz_path in sorted(path.rglob("*.xyz")):
        obj_path = _choose_obj_for_xyz(xyz_path)
        if obj_path is None or not obj_path.exists():
            continue
        key = (str(xyz_path.resolve()), str(obj_path.resolve()))
        if key in seen:
            continue
        seen.add(key)
        samples.append(
            {
                "sample_id": _sample_id_from_xyz(xyz_path),
                "xyz_path": str(xyz_path),
                "obj_path": str(obj_path),
            }
        )
    return samples


class RoofN3dDataset(Dataset):
    def __init__(self, data_path, transform, data_cfg, logger=None):
        self.samples = discover_samples(data_path)
        self.npoint = int(_cfg_get(data_cfg, "NPOINT", _cfg_get(data_cfg, "num_points", 4096)))
        self.normalize_with_full_cloud = bool(_cfg_get(data_cfg, "normalize_with_full_cloud", True))
        self.transform = transform

        if logger is not None:
            logger.info("Dataset path: %s", data_path)
            logger.info("Total samples: %d", len(self.samples))
            if len(self.samples) == 0:
                logger.warning("No .xyz/.obj sample pairs were found under %s", data_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]
        all_points = read_pts(sample["xyz_path"])
        vectors, edges = load_obj(sample["obj_path"])

        if self.normalize_with_full_cloud:
            min_pt, max_pt = self._square_bounds(all_points)

        points = self.transform(all_points.copy())
        points = self._sample_points(points)

        if not self.normalize_with_full_cloud:
            min_pt, max_pt = self._square_bounds(points)

        denom = np.maximum(max_pt - min_pt, 1e-6)
        points = (points - min_pt) / denom
        vectors = (vectors - min_pt) / denom

        min_max = np.stack([min_pt, max_pt], axis=0).astype(np.float32)
        return {
            "points": points.astype(np.float32),
            "vectors": vectors.astype(np.float32),
            "edges": edges.astype(np.int64),
            "frame_id": sample["sample_id"],
            "sample_id": sample["sample_id"],
            "xyz_path": sample["xyz_path"],
            "obj_path": sample["obj_path"],
            "minMaxPt": min_max,
        }

    def _sample_points(self, points):
        if len(points) == 0:
            raise ValueError("Empty point cloud")
        if len(points) >= self.npoint:
            idx = np.random.choice(len(points), self.npoint, replace=False)
        else:
            extra = np.random.choice(len(points), self.npoint - len(points), replace=True)
            idx = np.concatenate([np.arange(len(points)), extra])
        np.random.shuffle(idx)
        return points[idx]

    @staticmethod
    def _square_bounds(points):
        min_pt = np.min(points, axis=0).astype(np.float64)
        max_pt = np.max(points, axis=0).astype(np.float64)
        min_xyz = np.min(min_pt)
        max_xyz = np.max(max_pt)
        return np.full(3, min_xyz, dtype=np.float64), np.full(3, max_xyz, dtype=np.float64)

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)

        batch_size = len(batch_list)
        ret = {}
        for key, val in data_dict.items():
            if key == "points":
                ret[key] = np.stack(val, axis=0).astype(np.float32)
            elif key == "vectors":
                max_vec = max(len(x) for x in val)
                batch_vecs = np.ones((batch_size, max_vec, 3), dtype=np.float32) * -1e1
                for k in range(batch_size):
                    batch_vecs[k, : len(val[k]), :] = val[k]
                ret[key] = batch_vecs
            elif key == "edges":
                max_edges = max((len(x) for x in val), default=0)
                batch_edges = np.ones((batch_size, max_edges, 2), dtype=np.int64) * -10
                for k in range(batch_size):
                    if len(val[k]) > 0:
                        batch_edges[k, : len(val[k]), :] = val[k]
                ret[key] = batch_edges
            elif key == "minMaxPt":
                ret[key] = np.stack(val, axis=0).astype(np.float32)
            elif key in {"frame_id", "sample_id", "xyz_path", "obj_path"}:
                ret[key] = val
            else:
                ret[key] = np.stack(val, axis=0)

        ret["batch_size"] = batch_size
        return ret


def writePoints(points, clsRoad):
    out_dir = os.path.dirname(clsRoad)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(clsRoad, "w") as file1:
        for point in points:
            file1.write(f"{point[0]} {point[1]} {point[2]}\n")
