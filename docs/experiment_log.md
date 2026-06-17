# Experiment Log

Use this file to record reproducible runs, failures, artifacts, and suspected causes.

## Template

```text
date:
run_id:
config:
command:
device:
result:
artifacts:
notes:
```

Week 1 has no training experiment yet. Environment diagnostics come from `scripts/check_env.py`.

## 2026-06-17 M1 GT-Pose BEV Memory

```text
date: 2026-06-17
run_id: m1_gt_pose_bev_memory_kitti00
config: configs/eval/kitti_eval.yaml
dataset: KITTI Odometry sequence 00, split layout under data/kitti_odometry
commands:
  python scripts/preview_sequence.py --config configs/dataset/kitti.yaml --sequence 00 --data-root data/kitti_odometry
  python scripts/prepare_kitti.py --config configs/dataset/kitti.yaml --sequence 00 --data-root data/kitti_odometry
  python scripts/render_bev_frame.py --config configs/train/posenet_3dof.yaml --sequence 00 --frame-index 0 --data-root data/kitti_odometry
  python scripts/build_bev_memory_demo.py --config configs/eval/kitti_eval.yaml --sequence 00 --frames 10 --data-root data/kitti_odometry
  python scripts/build_bev_memory_demo.py --config configs/eval/kitti_eval.yaml --sequence 00 --frames 50 --data-root data/kitti_odometry
  python scripts/build_bev_memory_demo.py --config configs/eval/kitti_eval.yaml --sequence 00 --frames 100 --data-root data/kitti_odometry
result:
  sequence 00 validated with 4541 frames, 4541 timestamps, and 4541 poses.
  Single-frame BEV and GT-pose memory comparison figures were generated.
artifacts:
  outputs/manifests/kitti_sequence_00_manifest.json
  outputs/figures/kitti_00_000000_bev.png
  outputs/figures/kitti_00_000000_000009_gt_memory.png
  outputs/figures/kitti_00_000000_000049_gt_memory.png
  outputs/figures/kitti_00_000000_000099_gt_memory.png
notes:
  This is the Week 4/M1 upper-bound baseline. It uses GT adjacent poses and does not train PoseNet yet.
```

## 2026-06-17 Week 5 Pose Label Preview

```text
date: 2026-06-17
run_id: week5_pose_label_preview_kitti_train
config: configs/train/posenet_3dof.yaml
dataset: KITTI Odometry train split sequences 00-06, split layout under data/kitti_odometry
commands:
  python scripts/preview_pose_labels.py --config configs/train/posenet_3dof.yaml --sequence 00 --data-root data/kitti_odometry --max-pairs 20
  python scripts/preview_pose_labels.py --config configs/train/posenet_3dof.yaml --data-root data/kitti_odometry --write-stats outputs/metrics/pose_label_stats_train.json --json
result:
  train split pose-normalization count is 15230 adjacent pairs from sequences 00-06 only.
  train mean dx,dy,yaw is [1.026173, -0.000203, -0.000044].
  train std dx,dy,yaw is [0.469382, 0.020209, 0.017428].
  sequence 00 first 20-pair preview mean is [0.864453, 0.030725, 0.001977].
artifacts:
  outputs/metrics/pose_label_stats_train.json
notes:
  These stats are written into configs/train/posenet_3dof.yaml for Week 6 PoseNet target normalization.
  Validation and test sequences are not used for pose-normalization statistics.
```

## 2026-06-17 Week 6 PoseNet Smoke

```text
date: 2026-06-17
run_id: week6_posenet_smoke
config: configs/train/posenet_3dof.yaml
dataset:
  synthetic BEV pairs for CPU data-to-model smoke and tiny overfit
  KITTI Odometry tiny subset with max 1 adjacent pair per train/val sequence
commands:
  python -m pytest tests/test_posenet_training.py
  python -m pytest
  python scripts/run_pipeline_smoke.py --synthetic
  python scripts/train_posenet_3dof.py --config configs/train/posenet_3dof.yaml --synthetic --cpu --epochs 1 --batch-size 4 --overfit-batches 4 --output-dir outputs/checkpoints/week6_synthetic_smoke
  python scripts/train_posenet_3dof.py --config configs/train/posenet_3dof.yaml --data-root data/kitti_odometry --cpu --epochs 1 --batch-size 2 --max-train-pairs 1 --max-val-pairs 1 --overfit-batches 1 --output-dir outputs/checkpoints/week6_kitti_tiny_smoke
result:
  Week 6 tests passed: 6 passed in tests/test_posenet_training.py.
  Full test suite passed: 47 passed.
  Synthetic data-to-model smoke completed with finite loss 0.136254 and prediction shape (2, 3).
  Synthetic CPU overfit command saved posenet_3dof_latest.pt with train_loss 0.075349 and val_loss 0.107954.
  KITTI tiny CPU command used 7 train samples and 2 val samples, saving posenet_3dof_latest.pt with train_loss 0.129203.
artifacts:
  outputs/checkpoints/week6_synthetic_smoke/posenet_3dof_latest.pt
  outputs/checkpoints/week6_synthetic_smoke/posenet_3dof_best.pt
  outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt
  outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_best.pt
notes:
  This is a smoke-scale Week 6 baseline, not a full KITTI training run.
  The CLI supports both --overfit-batches and --overfit_batches spellings.
```

## 2026-06-17 Week 7 Odometry Eval Smoke

```text
date: 2026-06-17
run_id: week7_odometry_eval_smoke
config: configs/train/posenet_3dof.yaml
dataset:
  synthetic eval with 6 adjacent pairs
  KITTI Odometry sequence 07 tiny eval with 5 adjacent pairs
checkpoint:
  outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt
commands:
  python -m pytest tests/test_odometry_eval.py
  python -m pytest
  python scripts/eval_posenet.py --config configs/train/posenet_3dof.yaml --synthetic --max-pairs 6 --output-dir outputs/metrics/week7_synthetic_eval
  python scripts/eval_posenet.py --config configs/train/posenet_3dof.yaml --checkpoint outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt --data-root data/kitti_odometry --sequence 07 --max-pairs 5 --cpu --output-dir outputs/metrics/week7_kitti_tiny_eval
result:
  Week 7 tests passed: 5 passed in tests/test_odometry_eval.py.
  Full test suite passed: 52 passed.
  Synthetic eval saved metrics and trajectory; gt_label_echo ATE is 0.0 and zero_motion ATE is 1.764426.
  KITTI tiny eval saved zero_motion, constant_velocity, gt_label_echo, and learned baselines.
  KITTI tiny learned ATE is 2.794398, worse than zero_motion ATE 0.295942.
artifacts:
  outputs/metrics/week7_synthetic_eval/synthetic_eval_metrics.json
  outputs/metrics/week7_synthetic_eval/synthetic_eval_metrics.csv
  outputs/metrics/week7_synthetic_eval/synthetic_eval_trajectory.png
  outputs/metrics/week7_kitti_tiny_eval/07_eval_metrics.json
  outputs/metrics/week7_kitti_tiny_eval/07_eval_metrics.csv
  outputs/metrics/week7_kitti_tiny_eval/07_eval_trajectory.png
notes:
  The learned checkpoint is smoke-scale and not expected to beat baselines yet.
  This records the Week 7 accepted failure case: learned pose does not beat zero-motion on KITTI tiny eval.
  Full KITTI training or a stronger overfit checkpoint is needed before claiming learned odometry quality.
```

## 2026-06-17 Week 8 Learned-Pose BEV Memory

```text
date: 2026-06-17
run_id: week8_learned_pose_bev_memory_kitti_tiny
config: configs/eval/kitti_eval.yaml
train_config: configs/train/posenet_3dof.yaml
dataset: KITTI Odometry sequence 07, first 5 frames
checkpoint: outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt
commands:
  python -m pytest tests/test_learned_bev_memory.py
  python -m pytest
  python scripts/build_bev_memory_demo.py --config configs/eval/kitti_eval.yaml --train-config configs/train/posenet_3dof.yaml --pose-source learned --checkpoint outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt --sequence 07 --frames 5 --data-root data/kitti_odometry --cpu --output-dir outputs/figures/week8_kitti_tiny_memory --metrics-dir outputs/metrics/week8_kitti_tiny_memory
result:
  Week 8 tests passed: 4 passed in tests/test_learned_bev_memory.py.
  Full test suite passed: 56 passed.
  Four-panel memory image, trajectory overlay, and metrics JSON were generated.
  GT-pose remains the upper bound with mean_abs_error 0.0 and occupancy_iou 1.0.
  naive memory: mean_abs_error 0.016233, occupancy_iou 0.689449.
  learned-pose memory: mean_abs_error 0.026720, occupancy_iou 0.541895.
artifacts:
  outputs/figures/week8_kitti_tiny_memory/07_000000_000004_memory.png
  outputs/figures/week8_kitti_tiny_memory/07_000000_000004_trajectory.png
  outputs/metrics/week8_kitti_tiny_memory/07_000000_000004_memory_metrics.json
notes:
  This is the Week 8 accepted failure case: learned-pose memory is worse than naive memory on the tiny KITTI smoke run.
  Suspected cause: the checkpoint is smoke-scale and was not trained long enough to produce useful odometry.
  Full training or a stronger overfit checkpoint is needed before claiming learned-pose BEV quality.
```
