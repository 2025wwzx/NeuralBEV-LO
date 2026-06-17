# NeuralBEV-LO v0.1 Research Prototype Release Notes

Tag target: `v0.1-research-prototype`

This release freezes NeuralBEV-LO as a resume/GitHub-ready research prototype.
It demonstrates a reproducible KITTI Odometry pipeline for streaming BEV memory,
not production-grade odometry.

## Highlights

- Loads KITTI Odometry Velodyne, calibration, and pose files from the official
  split archive layout or a merged `sequences/` + `poses/` layout.
- Rasterizes sequential LiDAR frames into BEV tensors and builds naive,
  GT-pose, and learned-pose temporal BEV memories.
- Runs a lightweight PoseNet3DoF checkpoint for adjacent relative-pose
  inference on a short KITTI sequence.
- Writes final memory metrics, per-frame BEV consistency metrics, comparison
  figures, a trajectory overlay, a demo MP4, and a concise final report.
- Includes artifact-aware release readiness checks so the local release demo
  outputs are verified before tagging.

## Reproduce The Release Demo

```powershell
python scripts/run_v0_1_release_demo.py
python scripts/check_release_readiness.py --allow-pending-tag --require-artifacts
python -m pytest
```

Expected local release-demo outputs:

- `outputs/figures/v0_1_release_demo/07_000000_000004_memory.png`
- `outputs/figures/v0_1_release_demo/07_000000_000004_trajectory.png`
- `outputs/figures/v0_1_release_demo/07_000000_000004_alignment_curve.png`
- `outputs/figures/v0_1_release_demo/neuralbev_lo_v0_1_release_demo.mp4`
- `outputs/metrics/v0_1_release_demo/07_000000_000004_memory_metrics.json`
- `outputs/metrics/v0_1_release_demo/07_000000_000004_consistency_metrics.json`
- `outputs/reports/v0_1_release_demo.txt`

## Metrics Snapshot

The current v0.1 smoke run uses KITTI sequence `07`, frames `0` through `4`,
CPU inference, and `configs/eval/kitti_cpu_smoke_safe.yaml`.

```text
final occupancy_iou:
  naive: 0.689678
  gt_pose: 1.000000
  learned_pose: 0.542051

mean per-frame alignment / flicker:
  naive: alignment 0.744705, flicker 0.001996
  gt_pose: alignment 0.791612, flicker 0.008939
  learned_pose: alignment 0.612804, flicker 0.026458
```

Interpretation: GT-pose memory is the upper-bound reference. The learned-pose
memory is intentionally reported as a smoke-scale failure case because it
underperforms naive memory on this short run.

## Limitations

- "4D BEV" means a 2D BEV memory evolving over time, not a dense `(x, y, z, t)`
  reconstruction.
- The PoseNet checkpoint is a smoke-scale integration checkpoint, not a strong
  odometry model.
- The BEV consistency metrics are short-sequence engineering indicators and do
  not prove global map correctness.
- KITTI data, generated outputs, checkpoints, videos, and metrics are runtime
  artifacts and are intentionally not committed.

## Tagging

Do not create or push `v0.1-research-prototype` until the user explicitly
approves it in the current conversation.
