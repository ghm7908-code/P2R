from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from dataset.roofn3d_dataset import RoofN3dDataset

__all__ = {
    "RoofN3dDataset": RoofN3dDataset,
}


def _cfg_get(cfg, key, default=None):
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


class GaussianTransform:
    def __init__(self, sigma=(0.005, 0.015), clip=0.05, p=0.8):
        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, points):
        if np.random.rand(1) >= self.p:
            return points
        last_sigma = np.random.rand(1) * (self.sigma[1] - self.sigma[0]) + self.sigma[0]
        row, col = points.shape
        jittered = np.clip(last_sigma * np.random.randn(row, col), -self.clip, self.clip)
        return points + jittered


def resolve_split_path(root_or_split_path, data_cfg, training=True, split=None):
    path = Path(root_or_split_path)
    if path.is_file():
        return path

    if split is None:
        split = _cfg_get(data_cfg, "train_split", "train") if training else _cfg_get(data_cfg, "val_split", "val")

    if path.name in {"train", "val", "test"}:
        return path

    if path.is_dir():
        if training:
            target_count = _cfg_get(data_cfg, "subset_count", 4096)
            list_candidates = [
                path / f"train_list_subset_{target_count}.txt",
                path / "train_list_subset_4096.txt",
                path / "train_list_subset_2048.txt",
                path / "train_list.txt",
                path / "train.txt",
            ]
        elif split == "test":
            list_candidates = [
                path / "test_list.txt",
                path / "test.txt",
            ]
        else:
            list_candidates = [
                path / "valid_list.txt",
                path / "val_list.txt",
                path / "val.txt",
                path / "test.txt",
            ]
        for list_path in list_candidates:
            if list_path.exists():
                return list_path

    split_path = path / split
    if split_path.exists():
        return split_path

    legacy_list = path / ("train.txt" if training else "test.txt")
    if legacy_list.exists():
        return legacy_list

    return split_path


def build_dataloader(path, batch_size, data_cfg, workers=16, logger=None, training=True, split=None):
    data_path = resolve_split_path(path, data_cfg, training=training, split=split)
    if logger is not None:
        logger.info("Resolved dataset split path: %s", data_path)

    if training:
        transform = GaussianTransform(sigma=(0.005, 0.010), clip=0.05, p=0.8)
    else:
        transform = GaussianTransform(sigma=(0.005, 0.010), clip=0.05, p=0.0)

    dataset = RoofN3dDataset(data_path, transform, data_cfg, logger)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=workers,
        collate_fn=dataset.collate_batch,
        shuffle=training,
    )
    return dataloader
