#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评估 PoseNet 3DoF checkpoint 与 odometry baselines。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuralbev_lo.bev.rasterizer import bev_config_from_mapping  # noqa: E402
from neuralbev_lo.data.pair_dataset import (  # noqa: E402
    KittiAdjacentPairDataset,
    pose_label_stats_from_config,
    resolve_sequences_for_split,
)
from neuralbev_lo.eval.odometry_metrics import (  # noqa: E402
    build_baseline_predictions,
    evaluate_baselines,
    integrate_relative_poses,
    save_metrics_csv,
    save_metrics_json,
)
from neuralbev_lo.models.losses import denormalize_pose_batch  # noqa: E402
from neuralbev_lo.models.posenet import build_posenet_from_config, stack_bev_pair  # noqa: E402
from neuralbev_lo.training.checkpoint import load_checkpoint  # noqa: E402
from neuralbev_lo.training.train_posenet import SyntheticPairDataset, build_dataloader  # noqa: E402
from neuralbev_lo.utils.config import load_yaml_config  # noqa: E402
from neuralbev_lo.utils.logging import log_info, log_warn  # noqa: E402
from neuralbev_lo.viz.render_trajectory import save_trajectory_overlay  # noqa: E402

DEFAULT_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "configs" / "train" / "posenet_3dof.yaml"
DEFAULT_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs" / "metrics"


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Evaluate PoseNet 3DoF odometry baselines.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def _resolve_data_root(config: dict[str, Any], override: Path | None) -> Path:
    """从命令行、环境变量或 dataset config 解析 KITTI 根目录。"""

    if override is not None:
        return override
    dataset_config_path = PROJECT_ROOT / str(config.get("data", {}).get("dataset_config"))
    dataset_config = load_yaml_config(dataset_config_path)
    dataset_section = dataset_config.get("dataset", {})
    if not isinstance(dataset_section, dict):
        raise ValueError("dataset config must contain a dataset mapping")
    env_name = str(dataset_section.get("data_root_env", "KITTI_ROOT"))
    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        return Path(env_value)
    return PROJECT_ROOT / str(dataset_section.get("default_data_root", "data/kitti_odometry"))


def _select_sequence(config: dict[str, Any], split: str, override: str | None) -> str:
    """选择一个连续 sequence 用于短轨迹评估。"""

    if override is not None and override.strip():
        clean = override.strip()
        return clean.zfill(2) if clean.isdigit() else clean
    return resolve_sequences_for_split(config, split)[0]


def _synthetic_eval_data(max_pairs: int) -> tuple[SyntheticPairDataset, np.ndarray, np.ndarray, np.ndarray, str]:
    """构造合成 eval dataset 与物理 GT 标签。"""

    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    dataset = SyntheticPairDataset(num_samples=max_pairs, channels=4, height=16, width=16)
    gt_relative = dataset.targets.astype(np.float64)
    pose_mean = np.zeros(3, dtype=np.float32)
    pose_std = np.ones(3, dtype=np.float32)
    return dataset, gt_relative, pose_mean, pose_std, "synthetic"


def _kitti_eval_data(
    config: dict[str, Any],
    *,
    data_root: Path,
    sequence: str,
    max_pairs: int,
) -> tuple[KittiAdjacentPairDataset, np.ndarray, np.ndarray, np.ndarray, str]:
    """构造 KITTI eval dataset 与物理 GT 标签。"""

    bev_config = bev_config_from_mapping(config.get("bev", {}))
    pose_stats = pose_label_stats_from_config(config)
    dataset = KittiAdjacentPairDataset(
        data_root,
        [sequence],
        bev_config,
        max_pairs_per_sequence=max_pairs,
        pose_stats=pose_stats,
        normalize_targets=True,
    )
    gt_relative = np.asarray([sample.target_pose_3dof for sample in dataset.samples], dtype=np.float64)
    return dataset, gt_relative, pose_stats.mean, pose_stats.std, sequence


@torch.no_grad()
def _predict_learned_relative(
    config: dict[str, Any],
    checkpoint_path: Path,
    dataset,
    *,
    pose_mean: np.ndarray,
    pose_std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """加载 checkpoint 并预测反归一化后的 learned relative poses。"""

    model = build_posenet_from_config(config).to(device)
    metadata = load_checkpoint(checkpoint_path, model=model, map_location=device)
    checkpoint_config = metadata.get("config", {})
    if isinstance(checkpoint_config, dict) and checkpoint_config.get("pose"):
        config_pose_norm = config.get("pose", {}).get("pose_norm", {})
        checkpoint_pose_norm = checkpoint_config.get("pose", {}).get("pose_norm", {})
        if config_pose_norm and checkpoint_pose_norm and config_pose_norm != checkpoint_pose_norm:
            raise ValueError("runtime config pose_norm does not match checkpoint pose_norm")
    model.eval()
    dataloader = build_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions: list[np.ndarray] = []
    mean_tensor = torch.as_tensor(pose_mean, dtype=torch.float32, device=device)
    std_tensor = torch.as_tensor(pose_std, dtype=torch.float32, device=device)
    for batch in dataloader:
        bev_prev = batch[0].to(device=device, dtype=torch.float32)
        bev_curr = batch[1].to(device=device, dtype=torch.float32)
        prediction_norm = model(stack_bev_pair(bev_prev, bev_curr))
        prediction = denormalize_pose_batch(prediction_norm, mean_tensor, std_tensor)
        predictions.append(prediction.detach().cpu().numpy().astype(np.float64))
    if not predictions:
        raise ValueError("learned prediction received no batches")
    return np.concatenate(predictions, axis=0)


def main() -> int:
    """执行 baseline evaluation 并保存 metrics 与轨迹图。"""

    args = _parse_args()
    try:
        config = load_yaml_config(args.config)
        device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
        if args.synthetic:
            dataset, gt_relative, pose_mean, pose_std, sequence = _synthetic_eval_data(args.max_pairs)
        else:
            data_root = _resolve_data_root(config, args.data_root)
            sequence = _select_sequence(config, args.split, args.sequence)
            dataset, gt_relative, pose_mean, pose_std, sequence = _kitti_eval_data(
                config,
                data_root=data_root,
                sequence=sequence,
                max_pairs=args.max_pairs,
            )

        learned_relative = None
        if args.checkpoint is not None:
            learned_relative = _predict_learned_relative(
                config,
                args.checkpoint,
                dataset,
                pose_mean=pose_mean,
                pose_std=pose_std,
                device=device,
                batch_size=args.batch_size,
            )
        baselines = build_baseline_predictions(gt_relative, learned_relative=learned_relative)
        rows = evaluate_baselines(gt_relative, baselines)

        prefix = f"{sequence}_eval"
        metrics_json = args.output_dir / f"{prefix}_metrics.json"
        metrics_csv = args.output_dir / f"{prefix}_metrics.csv"
        trajectory_path = args.output_dir / f"{prefix}_trajectory.png"
        metadata = {
            "config": str(args.config),
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "sequence": sequence,
            "synthetic": bool(args.synthetic),
            "max_pairs": int(args.max_pairs),
            "device": str(device),
        }
        save_metrics_json(rows, metrics_json, metadata=metadata)
        save_metrics_csv(rows, metrics_csv)
        trajectories = {
            "gt": integrate_relative_poses(gt_relative),
            "zero_motion": integrate_relative_poses(baselines["zero_motion"]),
            "constant_velocity": integrate_relative_poses(baselines["constant_velocity"]),
        }
        if learned_relative is not None:
            trajectories["learned"] = integrate_relative_poses(learned_relative)
        save_trajectory_overlay(trajectories, trajectory_path, title=f"PoseNet eval {sequence}")
    except Exception as exc:  # noqa: BLE001 - CLI 需要上下文后返回非零。
        log_warn("PoseNet evaluation failed", config=args.config, checkpoint=args.checkpoint, error=exc)
        return 1

    best_row = min(rows, key=lambda row: float(row["ate_rmse_m"]))
    log_info(
        "PoseNet evaluation completed",
        sequence=sequence,
        baselines=",".join(str(row["baseline"]) for row in rows),
        best_baseline=best_row["baseline"],
        metrics_json=metrics_json,
        metrics_csv=metrics_csv,
        trajectory=trajectory_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
