#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练 Week 6 轻量级 PoseNet 3DoF baseline。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Final

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
from neuralbev_lo.models.posenet import build_posenet_from_config  # noqa: E402
from neuralbev_lo.training.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from neuralbev_lo.training.train_posenet import (  # noqa: E402
    SyntheticPairDataset,
    build_dataloader,
    train_one_epoch,
    validate,
)
from neuralbev_lo.utils.config import load_yaml_config  # noqa: E402
from neuralbev_lo.utils.logging import log_info, log_warn  # noqa: E402
from neuralbev_lo.utils.seed import set_global_seed  # noqa: E402

DEFAULT_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "configs" / "train" / "posenet_3dof.yaml"
DEFAULT_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs" / "checkpoints"


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Train the Week 6 PoseNet 3DoF baseline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--max-val-pairs", type=int, default=None)
    parser.add_argument("--overfit-batches", "--overfit_batches", dest="overfit_batches", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic pairs instead of KITTI.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA AMP.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
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


def _loss_weights_from_config(config: dict[str, Any]) -> tuple[float, float, float]:
    """读取 loss 权重配置。"""

    training_section = config.get("training", {})
    if not isinstance(training_section, dict):
        raise ValueError("config.training must be a mapping")
    raw_weights = training_section.get("loss_weights", [1.0, 1.0, 1.0])
    if not isinstance(raw_weights, (list, tuple)) or len(raw_weights) != 3:
        raise ValueError("training.loss_weights must contain 3 values")
    return float(raw_weights[0]), float(raw_weights[1]), float(raw_weights[2])


def _build_datasets(args: argparse.Namespace, config: dict[str, Any]):
    """根据参数构造 train/val dataset 和 pose 统计量。"""

    if args.synthetic:
        channels = len(config.get("bev", {}).get("channels", [0, 1, 2, 3]))
        num_samples = max(8, int(args.max_train_pairs or 16))
        train_dataset = SyntheticPairDataset(num_samples=num_samples, channels=channels, height=32, width=32)
        val_dataset = SyntheticPairDataset(num_samples=8, channels=channels, height=32, width=32, seed=20260609)
        pose_mean = torch.zeros(3, dtype=torch.float32)
        pose_std = torch.ones(3, dtype=torch.float32)
        return train_dataset, val_dataset, pose_mean, pose_std, "synthetic"

    data_root = _resolve_data_root(config, args.data_root)
    bev_config = bev_config_from_mapping(config.get("bev", {}))
    pose_stats = pose_label_stats_from_config(config)
    train_sequences = resolve_sequences_for_split(config, "train")
    val_sequences = resolve_sequences_for_split(config, "val")
    train_dataset = KittiAdjacentPairDataset(
        data_root,
        train_sequences,
        bev_config,
        max_pairs_per_sequence=args.max_train_pairs,
        pose_stats=pose_stats,
        normalize_targets=True,
    )
    val_dataset = KittiAdjacentPairDataset(
        data_root,
        val_sequences,
        bev_config,
        max_pairs_per_sequence=args.max_val_pairs,
        pose_stats=pose_stats,
        normalize_targets=True,
    )
    return train_dataset, val_dataset, pose_stats.mean, pose_stats.std, str(data_root)


def main() -> int:
    """执行 PoseNet 训练或 tiny overfit。"""

    args = _parse_args()
    try:
        config = load_yaml_config(args.config)
        seed = int(config.get("experiment", {}).get("seed", 20260608))
        set_global_seed(seed)
        torch.manual_seed(seed)
        device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
        training_section = config.get("training", {})
        if not isinstance(training_section, dict):
            raise ValueError("config.training must be a mapping")
        epochs = int(args.epochs or training_section.get("epochs", 1))
        batch_size = int(args.batch_size or training_section.get("batch_size", 4))
        if args.overfit_batches is not None and args.overfit_batches <= 0:
            raise ValueError("overfit-batches must be positive when provided")
        if epochs <= 0:
            raise ValueError("epochs must be positive")

        train_dataset, val_dataset, pose_mean, pose_std, data_source = _build_datasets(args, config)
        train_loader = build_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = build_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        model = build_posenet_from_config(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_section.get("learning_rate", 1.0e-3)))
        start_epoch = 0
        if args.resume is not None:
            metadata = load_checkpoint(args.resume, model=model, optimizer=optimizer, map_location=device)
            start_epoch = int(metadata["epoch"]) + 1

        loss_weights = _loss_weights_from_config(config)
        use_amp = bool(training_section.get("amp", True)) and not args.no_amp
        best_val_loss = float("inf")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        log_info(
            "PoseNet training started",
            config=args.config,
            data_source=data_source,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            train_samples=len(train_dataset),
            val_samples=len(val_dataset),
            overfit_batches=args.overfit_batches,
            amp=use_amp and device.type == "cuda",
            seed=seed,
        )

        for epoch in range(start_epoch, start_epoch + epochs):
            train_result = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device=device,
                loss_weights=loss_weights,
                amp=use_amp,
                max_batches=args.overfit_batches,
            )
            val_result = validate(
                model,
                val_loader if args.overfit_batches is None else train_loader,
                device=device,
                pose_mean=pose_mean,
                pose_std=pose_std,
                loss_weights=loss_weights,
                max_batches=args.overfit_batches,
            )
            metrics = {
                "train_loss": train_result.loss,
                "val_loss": val_result.loss,
                "translation_error_m": float(val_result.translation_error_m or 0.0),
                "yaw_error_rad": float(val_result.yaw_error_rad or 0.0),
            }
            latest_path = args.output_dir / "posenet_3dof_latest.pt"
            save_checkpoint(
                latest_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                config=config,
            )
            if val_result.loss < best_val_loss:
                best_val_loss = val_result.loss
                save_checkpoint(
                    args.output_dir / "posenet_3dof_best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=metrics,
                    config=config,
                )
            log_info(
                "PoseNet epoch completed",
                epoch=epoch,
                train_loss=f"{train_result.loss:.6f}",
                val_loss=f"{val_result.loss:.6f}",
                translation_error_m=f"{float(val_result.translation_error_m or 0.0):.6f}",
                yaw_error_rad=f"{float(val_result.yaw_error_rad or 0.0):.6f}",
                checkpoint=latest_path,
            )
    except Exception as exc:  # noqa: BLE001 - CLI 需要完整上下文后返回非零。
        log_warn("PoseNet training failed", config=args.config, error=exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
