# NeuralBEV-LO

Learning-based LiDAR Odometry for Streaming 4D BEV Mapping.

NeuralBEV-LO is an open-source research prototype, not a production autonomous-driving stack. The first target is a small, reproducible KITTI Odometry pipeline that converts sequential LiDAR frames into BEV tensors, builds a ground-truth-pose BEV memory baseline, trains a lightweight relative-pose network, and compares learned-pose BEV memory against simple baselines.

## Repository

- Remote: `git@github.com:2025wwzx/NeuralBEV-LO.git`
- Workspace: `E:\AAAworkspace\study\NeuralBEV-LO`
- Primary dataset target: KITTI Odometry
- First milestone: M0 reproducible skeleton

## Current Scope

The project has reached the Week 11/M3 research-demo stage: KITTI Odometry
loading, BEV rasterization, GT-pose memory, lightweight PoseNet relative-pose
inference, learned-pose memory comparison, odometry/BEV metrics, failure-case
logging, and a short demo video are runnable. The learned checkpoint is still a
smoke-scale prototype and is documented as weak; GT-pose memory remains the
upper-bound reference.

## Architecture

The v0.1 pipeline stays deliberately small:

```text
KITTI Velodyne + poses
  -> validated point clouds and LiDAR poses
  -> BEV tensors
  -> adjacent 3DoF labels
  -> PoseNet3DoF relative-pose prediction
  -> temporal BEV memory update
  -> odometry, BEV consistency metrics, figures, and M3 demo video
```

See `docs/architecture.md` for the architecture and coordinate-system diagrams.

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

## Week 6 PoseNet Smoke Commands

The first learned-odometry baseline is a lightweight CNN PoseNet. It stacks two
BEV frames by channel and predicts normalized `dx, dy, yaw`.

```powershell
python scripts/run_pipeline_smoke.py --synthetic
python scripts/train_posenet_3dof.py --config configs/train/posenet_3dof.yaml --synthetic --cpu --epochs 1 --batch-size 4 --overfit-batches 4 --output-dir outputs/checkpoints/week6_synthetic_smoke
python scripts/train_posenet_3dof.py --config configs/train/posenet_3dof.yaml --data-root data/kitti_odometry --cpu --epochs 1 --batch-size 2 --max-train-pairs 1 --max-val-pairs 1 --overfit-batches 1 --output-dir outputs/checkpoints/week6_kitti_tiny_smoke
```

Checkpoints are generated under `outputs/checkpoints/` and are not committed.

## Week 7 Odometry Eval Commands

Week 7 evaluates relative-pose baselines and integrates short trajectories. The
current smoke-scale learned checkpoint is expected to be weak; failures are
logged instead of hidden.

```powershell
python scripts/eval_posenet.py --config configs/train/posenet_3dof.yaml --synthetic --max-pairs 6 --output-dir outputs/metrics/week7_synthetic_eval
python scripts/eval_posenet.py --config configs/train/posenet_3dof.yaml --checkpoint outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt --data-root data/kitti_odometry --sequence 07 --max-pairs 5 --cpu --output-dir outputs/metrics/week7_kitti_tiny_eval
```

The eval command writes metrics JSON/CSV and a trajectory overlay PNG under the
chosen output directory.

## Week 8 Learned-Pose BEV Memory Commands

Week 8 connects PoseNet predictions to temporal BEV memory. It writes a
current / naive / GT-pose / learned-pose comparison image, a trajectory overlay,
and BEV memory metrics.

```powershell
python scripts/build_bev_memory_demo.py --config configs/eval/kitti_eval.yaml --train-config configs/train/posenet_3dof.yaml --pose-source learned --checkpoint outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt --sequence 07 --frames 5 --data-root data/kitti_odometry --cpu --output-dir outputs/figures/week8_kitti_tiny_memory --metrics-dir outputs/metrics/week8_kitti_tiny_memory
```

The current smoke checkpoint is weak; the expected Week 8 result is a runnable
comparison plus a documented failure case, not a strong learned map yet.

## Week 9 BEV Consistency Metric Commands

Week 9 adds per-frame BEV consistency metrics to the memory demo. The command
now writes final memory metrics JSON/CSV, per-frame consistency JSON/CSV, and an
alignment curve PNG. Use `--resolution-m` to run a small resolution ablation
without editing YAML.

```powershell
python scripts/build_bev_memory_demo.py --config configs/eval/kitti_eval.yaml --train-config configs/train/posenet_3dof.yaml --pose-source learned --checkpoint outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt --sequence 07 --frames 5 --resolution-m 0.5 --data-root data/kitti_odometry --cpu --output-dir outputs/figures/week9_kitti_tiny_consistency_05 --metrics-dir outputs/metrics/week9_kitti_tiny_consistency_05
python scripts/build_bev_memory_demo.py --config configs/eval/kitti_eval.yaml --train-config configs/train/posenet_3dof.yaml --pose-source learned --checkpoint outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt --sequence 07 --frames 5 --resolution-m 0.25 --data-root data/kitti_odometry --cpu --output-dir outputs/figures/week9_kitti_tiny_consistency_025 --metrics-dir outputs/metrics/week9_kitti_tiny_consistency_025
```

Metric definitions are stored in the consistency JSON metadata. The current
defaults use density only: `alignment_iou` is thresholded current BEV vs memory
BEV mean IoU with `occupancy_threshold=0.1`, and `flicker_score` is mean
pixel-wise standard deviation over the configured memory window. These are
short-sequence engineering indicators; they do not prove global map correctness,
height consistency, or learned odometry quality.

## Week 10 Robustness and Runtime Commands

Week 10 adds lightweight point filtering variants and optional runtime stage
logging to the memory demo. Safe eval configs document conservative defaults for
CPU smoke runs and RTX 5080 runs.

```powershell
python scripts/build_bev_memory_demo.py --config configs/eval/kitti_cpu_smoke_safe.yaml --train-config configs/train/posenet_3dof.yaml --pose-source learned --checkpoint outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt --sequence 07 --frames 5 --profile-runtime --data-root data/kitti_odometry --cpu --output-dir outputs/figures/week10_cpu_safe --metrics-dir outputs/metrics/week10_cpu_safe
python scripts/build_bev_memory_demo.py --config configs/eval/kitti_rtx5080_safe.yaml --train-config configs/train/posenet_3dof.yaml --pose-source learned --checkpoint outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt --sequence 07 --frames 5 --profile-runtime --filter-z-range -1.5 3.0 --filter-distance-range 0.0 50.0 --data-root data/kitti_odometry --output-dir outputs/figures/week10_filtered_smoke --metrics-dir outputs/metrics/week10_filtered_smoke
```

`--filter-z-range` clips LiDAR points by height before BEV rasterization, and
`--filter-distance-range` clips by horizontal radial distance. `--profile-runtime`
prints stage logs for data loading, rasterization, inference, BEV warp, and
rendering. These logs are for bottleneck diagnosis only; they are not benchmark
numbers unless the hardware, data range, and thread settings are fixed.

## Week 11 M3 Demo and Report

The M3 demo uses the Week 10 CPU-safe learned-memory path, then combines the
memory comparison image and trajectory overlay into an MP4. The memory image
contains current BEV, naive memory, GT-pose memory, and learned-pose memory; the
trajectory image shows GT, naive, and learned short trajectories.

```powershell
python scripts/build_bev_memory_demo.py --config configs/eval/kitti_cpu_smoke_safe.yaml --train-config configs/train/posenet_3dof.yaml --pose-source learned --checkpoint outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt --sequence 07 --frames 5 --data-root data/kitti_odometry --cpu --output-dir outputs/figures/week11_m3_demo --metrics-dir outputs/metrics/week11_m3_demo
python scripts/build_m3_demo_video.py --memory-image outputs/figures/week11_m3_demo/07_000000_000004_memory.png --trajectory-image outputs/figures/week11_m3_demo/07_000000_000004_trajectory.png --output outputs/figures/week11_m3_demo/neuralbev_lo_m3_demo.mp4 --seconds 6 --fps 6 --title "NeuralBEV-LO M3 KITTI seq07 smoke demo"
```

Generated M3 artifacts:

- `outputs/figures/week11_m3_demo/neuralbev_lo_m3_demo.mp4`
- `outputs/figures/week11_m3_demo/07_000000_000004_memory.png`
- `outputs/figures/week11_m3_demo/07_000000_000004_trajectory.png`
- `outputs/figures/week11_m3_demo/07_000000_000004_alignment_curve.png`
- `outputs/metrics/week11_m3_demo/07_000000_000004_memory_metrics.json`
- `docs/m3_report.md`

On the current smoke run, learned-pose memory remains worse than naive memory
(`occupancy_iou` 0.542051 vs 0.689678), while GT-pose memory is the reference
upper bound (`occupancy_iou` 1.0). This is a reproducibility and integration
demo, not a claim of production odometry accuracy.

## Week 12 Reproducibility Freeze

Week 12 freezes the v0.1 research-prototype command path. The release readiness
checker verifies required docs/scripts, runtime artifact ignore rules, and the
pending release-tag state.

```powershell
python scripts/check_env.py
python scripts/run_pipeline_smoke.py --synthetic
python -m pytest
python scripts/train_posenet_3dof.py --config configs/train/posenet_3dof.yaml --synthetic --cpu --epochs 1 --batch-size 4 --overfit-batches 4 --output-dir outputs/checkpoints/week12_synthetic_freeze
python scripts/eval_posenet.py --config configs/train/posenet_3dof.yaml --synthetic --max-pairs 6 --output-dir outputs/metrics/week12_synthetic_eval
python scripts/build_bev_memory_demo.py --config configs/eval/kitti_cpu_smoke_safe.yaml --train-config configs/train/posenet_3dof.yaml --pose-source learned --checkpoint outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt --sequence 07 --frames 5 --data-root data/kitti_odometry --cpu --output-dir outputs/figures/week12_m3_freeze --metrics-dir outputs/metrics/week12_m3_freeze
python scripts/build_m3_demo_video.py --memory-image outputs/figures/week12_m3_freeze/07_000000_000004_memory.png --trajectory-image outputs/figures/week12_m3_freeze/07_000000_000004_trajectory.png --output outputs/figures/week12_m3_freeze/neuralbev_lo_v0_1_demo.mp4 --seconds 6 --fps 6 --title "NeuralBEV-LO v0.1 research prototype"
python scripts/check_release_readiness.py --allow-pending-tag
```

See `docs/release_checklist.md` before tagging. The tag
`v0.1-research-prototype` must only be created after explicit user approval.

## Data

Large datasets are not committed. See `data/README.md` for the expected KITTI layout and environment-variable options.

## Limitations

- "4D BEV" in this repository means a 2D BEV memory evolving over time, not a
  dense `(x, y, z, t)` reconstruction.
- The current PoseNet checkpoint is smoke-scale and is not trained enough to
  beat simple baselines on KITTI tiny validation.
- BEV consistency metrics use density-channel occupancy by default; they do not
  prove global map correctness or height consistency.
- `data/`, `outputs/`, and checkpoints are runtime artifacts and must not be
  committed.

## Documentation

- `NeuralBEV-LO-execution-plan.md` - full 12-week execution plan.
- `docs/data_contract.md` - dataset and artifact rules.
- `docs/coordinate_system.md` - KITTI pose, LiDAR pose, BEV grid, and warp conventions.
- `docs/architecture.md` - evolving architecture notes.
- `docs/m3_report.md` - Week 11 M3 demo and technical report summary.
- `docs/release_checklist.md` - Week 12 reproducibility and tag checklist.
- `docs/experiment_log.md` - experiment and failure-case log.
- `docs/resume_notes.md` - artifact-backed resume wording.
