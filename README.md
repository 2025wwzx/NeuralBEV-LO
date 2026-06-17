# NeuralBEV-LO

Learning-based LiDAR Odometry for Streaming 4D BEV Mapping.

NeuralBEV-LO is an open-source research prototype, not a production autonomous-driving stack. The first target is a small, reproducible KITTI Odometry pipeline that converts sequential LiDAR frames into BEV tensors, builds a ground-truth-pose BEV memory baseline, trains a lightweight relative-pose network, and compares learned-pose BEV memory against simple baselines.

## Repository

- Remote: `git@github.com:2025wwzx/NeuralBEV-LO.git`
- Workspace: `E:\AAAworkspace\study\NeuralBEV-LO`
- Primary dataset target: KITTI Odometry
- First milestone: M0 reproducible skeleton

## Current Scope

The project has reached the M1 data-to-BEV baseline: KITTI Odometry loading,
single-frame BEV rendering, and GT-pose BEV memory demos are runnable. The
current Week 5 focus is the adjacent-frame pair dataset and train-only 3DoF
pose-label normalization pipeline for the first PoseNet training loop.

## Environment

The current local diagnostic snapshot is:

```text
Python: 3.12.10
GPU: NVIDIA GeForce RTX 5080
Driver CUDA reported by nvidia-smi: 13.2
PyTorch observed locally: 2.12.0+cu132
torch.cuda.is_available(): True
```

The project metadata allows Python `>=3.10,<3.13`. CUDA and PyTorch installation commands should be documented only after `scripts/check_env.py` confirms the working local combination.

Week 1 fallback ladder:

1. Use the native Windows Python + CUDA environment when `torch.cuda.is_available()` is true.
2. Use WSL2 only if Windows CUDA package compatibility blocks PyTorch or OpenCV.
3. Keep CPU-only synthetic smoke tests runnable even when KITTI data or CUDA is unavailable.

## Quick Smoke Commands

```powershell
python scripts/check_env.py
python -m pytest
python -c "import neuralbev_lo"
python scripts/run_pipeline_smoke.py --synthetic
```

## KITTI M1 Demo Commands

After placing KITTI Odometry under `data/kitti_odometry/`, either as a merged
`sequences/` + `poses/` root or as the official split `data_odometry_*` folders:

```powershell
python scripts/preview_sequence.py --config configs/dataset/kitti.yaml --sequence 00 --data-root data/kitti_odometry
python scripts/render_bev_frame.py --config configs/train/posenet_3dof.yaml --sequence 00 --frame-index 0 --data-root data/kitti_odometry
python scripts/build_bev_memory_demo.py --config configs/eval/kitti_eval.yaml --sequence 00 --frames 100 --data-root data/kitti_odometry
```

The memory demo writes a current / naive / GT-pose BEV comparison image under
`outputs/figures/`.

## KITTI Week 5 Label Preview

The default pose target is adjacent-frame `dx, dy, yaw` in the previous
LiDAR/BEV frame. Pose normalization statistics are computed from train
sequences `00`-`06` only and recorded in `configs/train/posenet_3dof.yaml`.

```powershell
python scripts/preview_pose_labels.py --config configs/train/posenet_3dof.yaml --sequence 00 --data-root data/kitti_odometry --max-pairs 20
python scripts/preview_pose_labels.py --config configs/train/posenet_3dof.yaml --data-root data/kitti_odometry --write-stats outputs/metrics/pose_label_stats_train.json
```

The pair dataset API returns `(bev_prev, bev_curr, target_pose_3dof, metadata)`
and is covered by `tests/test_pair_dataset.py`.

## Data

Large datasets are not committed. See `data/README.md` for the expected KITTI layout and environment-variable options.

## Documentation

- `NeuralBEV-LO-execution-plan.md` - full 12-week execution plan.
- `docs/data_contract.md` - dataset and artifact rules.
- `docs/coordinate_system.md` - KITTI pose, LiDAR pose, BEV grid, and warp conventions.
- `docs/architecture.md` - evolving architecture notes.
- `docs/experiment_log.md` - experiment and failure-case log.
- `docs/resume_notes.md` - artifact-backed resume wording.
