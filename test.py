import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch

from dataset.data_utils import build_dataloader
from model import model_utils
from model.roofnet import RoofNet
from test_util import load_data_to_cpu, load_data_to_gpu, save_wireframe_obj, test_model, visualize_predictions
from utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description="Point2Roof testing script")
    parser.add_argument("--cfg_file", type=str, default="./model_cfg.yaml")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--test_tag", type=str, default="default")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--edge_thresh", type=float, default=0.5)
    parser.add_argument("--point_thresh", type=float, default=0.1)
    parser.add_argument("--ap_distance_thresh", type=float, default=0.1,
                        help="Distance threshold for APCalculator Hungarian matching")
    parser.add_argument("--save_obj", action="store_true")
    parser.add_argument("--allow_random", action="store_true")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization mode")
    parser.add_argument("--vis_sample_id", type=str, default=None, help="Sample ID to visualize")
    parser.add_argument("--vis_all", action="store_true", help="Visualize all samples")
    args = parser.parse_args()

    cfg = common_utils.cfg_from_yaml_file(args.cfg_file)
    if args.data_path:
        cfg.DATA.root_dir = args.data_path
    cfg.DATA.NPOINT = cfg.DATA.get("NPOINT", cfg.DATA.get("num_points", 4096))
    cfg.DATA.use_color = cfg.DATA.get("use_color", False)
    cfg.DATA.use_intensity = cfg.DATA.get("use_intensity", False)
    cfg.MODEL.use_edge = cfg.MODEL.get("use_edge", True)
    return args, cfg


def get_input_channels(cfg):
    channels = 3
    if cfg.DATA.get("use_color", False):
        channels += 3
    if cfg.DATA.get("use_intensity", False):
        channels += 1
    return channels


def find_checkpoint(args, output_dir, root_dir):
    if args.ckpt:
        return Path(args.ckpt)

    candidates = []
    for ckpt_dir in [output_dir / "ckpt", output_dir, root_dir]:
        candidates.extend(Path(p) for p in glob.glob(str(ckpt_dir / "*checkpoint_epoch_*.pth")))
    root_default = root_dir / "checkpoint_epoch_90.pth"
    if root_default.exists():
        candidates.append(root_default)

    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def export_predictions(model, data_loader, output_dir, edge_thresh, logger):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.use_edge = True

    with torch.no_grad():
        for batch in data_loader:
            load_data_to_gpu(batch)
            pred = model(batch)
            load_data_to_cpu(pred)
            load_data_to_cpu(batch)

            keypoints = pred.get("keypoint", np.zeros((0, 4), dtype=np.float32))
            refined = pred.get("refined_keypoint", np.zeros((0, 3), dtype=np.float32))
            pair_points = pred.get("pair_points", np.zeros((0, 2), dtype=np.int64))
            edge_scores = pred.get("edge_score", np.zeros((0,), dtype=np.float32))

            edge_offset = 0
            for i, sample_id in enumerate(batch["frame_id"]):
                mask = keypoints[:, 0] == i if len(keypoints) else np.zeros((0,), dtype=bool)
                pts = refined[mask]
                min_pt, max_pt = batch["minMaxPt"][i]
                pts_orig = pts * (max_pt - min_pt) + min_pt

                num_pairs = len(pts) * (len(pts) - 1) // 2
                sample_pairs = pair_points[edge_offset: edge_offset + num_pairs]
                sample_scores = edge_scores[edge_offset: edge_offset + num_pairs]
                edge_offset += num_pairs

                pred_edges = sample_pairs[sample_scores > edge_thresh] if len(sample_pairs) else np.zeros((0, 2), dtype=np.int64)
                obj_path = output_dir / f"{sample_id}_pred.obj"
                save_wireframe_obj(pts_orig, pred_edges, str(obj_path))
                logger.info("Exported %s: points=%d edges=%d", obj_path, len(pts_orig), len(pred_edges))


def main():
    args, cfg = parse_config()
    if args.gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    output_dir = cfg.ROOT_DIR / "output" / args.test_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(output_dir / "test_log.txt")
    logger.info("********************** Start testing **********************")
    logger.info("data_path: %s", cfg.DATA.root_dir)
    logger.info("split: %s", args.split)

    test_loader = build_dataloader(
        cfg.DATA.root_dir,
        args.batch_size,
        cfg.DATA,
        workers=cfg.DATA.get("num_workers", 16),
        logger=logger,
        training=False,
        split=args.split,
    )

    input_channels = get_input_channels(cfg)
    model = RoofNet(cfg.MODEL, input_channel=input_channels).cuda()
    model.eval()

    ckpt_path = find_checkpoint(args, output_dir, cfg.ROOT_DIR)
    if ckpt_path is None:
        if not args.allow_random:
            raise FileNotFoundError("No checkpoint found. Pass --ckpt or use --allow_random for a smoke test.")
        logger.warning("No checkpoint found; evaluating random initialized weights.")
    else:
        model_utils.load_params_with_optimizer(model, str(ckpt_path), optimizer=None, logger=logger)
        logger.info("Loaded checkpoint: %s", ckpt_path)

    metrics = test_model(
        model,
        test_loader,
        logger,
        edge_thresh=args.edge_thresh,
        point_match_thresh=args.point_thresh,
        ap_distance_thresh=args.ap_distance_thresh,
    )

    if args.save_obj:
        vis_loader = build_dataloader(
            cfg.DATA.root_dir,
            1,
            cfg.DATA,
            workers=cfg.DATA.get("num_workers", 16),
            logger=logger,
            training=False,
            split=args.split,
        )
        export_predictions(model, vis_loader, output_dir / "test_results", args.edge_thresh, logger)

    if args.visualize:
        logger.info("Visualization mode enabled")
        vis_loader = build_dataloader(
            cfg.DATA.root_dir,
            1,
            cfg.DATA,
            workers=cfg.DATA.get("num_workers", 16),
            logger=logger,
            training=False,
            split=args.split,
        )
        vis_output_dir = output_dir / "visualizations"
        vis_output_dir.mkdir(parents=True, exist_ok=True)
        visualize_predictions(
            model, vis_loader, vis_output_dir, logger,
            vis_sample_id=args.vis_sample_id,
            vis_all=args.vis_all,
            edge_thresh=args.edge_thresh,
        )

    logger.info("metrics: %s", metrics)


if __name__ == "__main__":
    main()
