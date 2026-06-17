#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行 NeuralBEV-LO 合成 data-to-model smoke。

该 smoke 不依赖 KITTI 数据或 CUDA：它构造一个确定性合成 BEV pair batch，执行
PoseNet 前向、weighted SmoothL1 loss、checkpoint save/load round trip，并打印
结构化日志。后续真实 KITTI smoke 可以复用同一模型与 loss 接口。
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Final

import torch

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuralbev_lo.models.losses import weighted_smooth_l1_loss  # noqa: E402
from neuralbev_lo.models.posenet import PoseNet3DoF, stack_bev_pair  # noqa: E402
from neuralbev_lo.training.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from neuralbev_lo.training.train_posenet import SyntheticPairDataset, build_dataloader  # noqa: E402
from neuralbev_lo.utils.config import load_yaml_config  # noqa: E402
from neuralbev_lo.utils.logging import log_info, log_warn  # noqa: E402
from neuralbev_lo.utils.seed import set_global_seed  # noqa: E402

DEFAULT_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "configs" / "eval" / "kitti_eval.yaml"


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Run NeuralBEV-LO synthetic data-to-model smoke.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--synthetic", action="store_true", help="Run without KITTI data.")
    parser.add_argument("--stage", choices=("model",), default="model")
    return parser.parse_args()


def main() -> int:
    """执行合成 data-to-model smoke。"""

    args = _parse_args()
    config = load_yaml_config(args.config)
    seed = int(config.get("experiment", {}).get("seed", 20260608))
    set_global_seed(seed)
    torch.manual_seed(seed)

    if not args.synthetic:
        log_warn("Only synthetic smoke is implemented; proceeding in synthetic mode")

    dataset = SyntheticPairDataset(num_samples=4, channels=4, height=16, width=16, seed=seed)
    dataloader = build_dataloader(dataset, batch_size=2, shuffle=False, num_workers=0)
    bev_prev, bev_curr, target, _metadata = next(iter(dataloader))
    model = PoseNet3DoF(in_channels=8, hidden_channels=8)
    prediction = model(stack_bev_pair(bev_prev, bev_curr))
    loss = weighted_smooth_l1_loss(prediction, target)
    if not torch.isfinite(loss):
        log_warn("Synthetic smoke loss is not finite", loss=float(loss.detach().cpu().item()))
        return 1

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "synthetic_smoke.pt"
        expected = prediction.detach()
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=None,
            epoch=0,
            metrics={"loss": float(loss.detach().cpu().item())},
            config={"smoke": "synthetic"},
        )
        reloaded = PoseNet3DoF(in_channels=8, hidden_channels=8)
        load_checkpoint(checkpoint_path, model=reloaded)
        actual = reloaded(stack_bev_pair(bev_prev, bev_curr)).detach()
        torch.testing.assert_close(actual, expected)

    log_info(
        "Synthetic data-to-model smoke completed",
        config=args.config,
        stage=args.stage,
        prediction_shape=tuple(prediction.shape),
        loss=f"{float(loss.detach().cpu().item()):.6f}",
        seed=seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
