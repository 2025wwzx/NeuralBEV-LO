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

## 2026-06-17 Week 9 BEV Consistency Metrics and Resolution Ablation

```text
date: 2026-06-17
run_id: week9_bev_consistency_kitti_tiny
config: configs/eval/kitti_eval.yaml
train_config: configs/train/posenet_3dof.yaml
dataset: KITTI Odometry sequence 07, first 5 frames
checkpoint: outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt
metrics_definition:
  alignment_iou: mean IoU between thresholded current BEV and thresholded memory BEV.
  flicker_score: mean pixel-wise standard deviation over the memory history window.
  occupancy_threshold: 0.1
  selected_channels: density
  flicker_window: 5
commands:
  python -m pytest tests/test_bev_consistency_metrics.py
  python -m pytest tests/test_learned_bev_memory.py tests/test_bev_consistency_metrics.py
  python scripts/build_bev_memory_demo.py --config configs/eval/kitti_eval.yaml --train-config configs/train/posenet_3dof.yaml --pose-source learned --checkpoint outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt --sequence 07 --frames 5 --resolution-m 0.5 --data-root data/kitti_odometry --cpu --output-dir outputs/figures/week9_kitti_tiny_consistency_05 --metrics-dir outputs/metrics/week9_kitti_tiny_consistency_05
  python scripts/build_bev_memory_demo.py --config configs/eval/kitti_eval.yaml --train-config configs/train/posenet_3dof.yaml --pose-source learned --checkpoint outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt --sequence 07 --frames 5 --resolution-m 0.25 --data-root data/kitti_odometry --cpu --output-dir outputs/figures/week9_kitti_tiny_consistency_025 --metrics-dir outputs/metrics/week9_kitti_tiny_consistency_025
result:
  Week 9 tests passed: 4 passed in tests/test_bev_consistency_metrics.py.
  Week 8 + Week 9 local tests passed: 8 passed.
  0.5m final memory occupancy_iou: naive 0.689449, gt_pose 1.0, learned_pose 0.541895.
  0.5m mean per-frame consistency: naive alignment 0.743782 / flicker 0.002002; gt_pose alignment 0.791565 / flicker 0.008974; learned_pose alignment 0.612429 / flicker 0.026496.
  0.25m final memory occupancy_iou: naive 0.541576, gt_pose 1.0, learned_pose 0.470163.
  0.25m mean per-frame consistency: naive alignment 0.651811 / flicker 0.001256; gt_pose alignment 0.732911 / flicker 0.006948; learned_pose alignment 0.507377 / flicker 0.016087.
artifacts:
  outputs/figures/week9_kitti_tiny_consistency_05/07_000000_000004_memory.png
  outputs/figures/week9_kitti_tiny_consistency_05/07_000000_000004_trajectory.png
  outputs/figures/week9_kitti_tiny_consistency_05/07_000000_000004_alignment_curve.png
  outputs/metrics/week9_kitti_tiny_consistency_05/07_000000_000004_memory_metrics.json
  outputs/metrics/week9_kitti_tiny_consistency_05/07_000000_000004_memory_metrics.csv
  outputs/metrics/week9_kitti_tiny_consistency_05/07_000000_000004_consistency_metrics.json
  outputs/metrics/week9_kitti_tiny_consistency_05/07_000000_000004_consistency_metrics.csv
  outputs/figures/week9_kitti_tiny_consistency_025/07_000000_000004_memory.png
  outputs/figures/week9_kitti_tiny_consistency_025/07_000000_000004_trajectory.png
  outputs/figures/week9_kitti_tiny_consistency_025/07_000000_000004_alignment_curve.png
  outputs/metrics/week9_kitti_tiny_consistency_025/07_000000_000004_memory_metrics.json
  outputs/metrics/week9_kitti_tiny_consistency_025/07_000000_000004_memory_metrics.csv
  outputs/metrics/week9_kitti_tiny_consistency_025/07_000000_000004_consistency_metrics.json
  outputs/metrics/week9_kitti_tiny_consistency_025/07_000000_000004_consistency_metrics.csv
notes:
  These metrics are short-sequence engineering indicators, not proof of global map correctness.
  alignment_iou currently uses density only; the same threshold is not valid for height channels without redefining the metric.
  flicker_score is affected by memory decay alpha 0.9 and flicker_window 5, so lower flicker can also mean heavier smoothing.
  The learned checkpoint remains smoke-scale and is still worse than naive memory in both tested resolutions.
  Resolution ablation changes rasterizer resolution only; the PoseNet checkpoint is not retrained per resolution.
```

## 2026-06-17 Week 10 Robustness, Runtime, and Failure Analysis

```text
date: 2026-06-17
run_id: week10_robustness_runtime_failure_analysis
config:
  configs/eval/kitti_eval.yaml
  configs/eval/kitti_cpu_smoke_safe.yaml
  configs/eval/kitti_rtx5080_safe.yaml
dataset: KITTI Odometry sequence 07
checkpoint: outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt
commands:
  python scripts/check_env.py
  python -m pytest tests/test_smoke.py
  python scripts/render_bev_frame.py --config configs/train/posenet_3dof.yaml --sequence 07 --frame-index 0 --data-root data/kitti_odometry --output-dir outputs/figures/week10_m1_verify
  python scripts/build_bev_memory_demo.py --config configs/eval/kitti_eval.yaml --train-config configs/train/posenet_3dof.yaml --pose-source gt --sequence 07 --frames 5 --data-root data/kitti_odometry --cpu --output-dir outputs/figures/week10_m1_gt_memory_verify --metrics-dir outputs/metrics/week10_m1_gt_memory_verify
  python scripts/eval_posenet.py --config configs/train/posenet_3dof.yaml --checkpoint outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt --data-root data/kitti_odometry --sequence 07 --max-pairs 5 --cpu --output-dir outputs/metrics/week10_m2_verify
  python -m pytest tests/test_week10_robustness.py
  python -m pytest tests/test_learned_bev_memory.py tests/test_bev_consistency_metrics.py tests/test_week10_robustness.py
  python scripts/build_bev_memory_demo.py --config configs/eval/kitti_eval.yaml --train-config configs/train/posenet_3dof.yaml --pose-source learned --checkpoint outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt --sequence 07 --frames 5 --start-frame 0 --resolution-m 0.5 --profile-runtime --data-root data/kitti_odometry --cpu --output-dir outputs/figures/week10_failure_case_a_start0_05 --metrics-dir outputs/metrics/week10_failure_case_a_start0_05
  python scripts/build_bev_memory_demo.py --config configs/eval/kitti_eval.yaml --train-config configs/train/posenet_3dof.yaml --pose-source learned --checkpoint outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt --sequence 07 --frames 5 --start-frame 0 --resolution-m 0.25 --profile-runtime --filter-z-range -1.5 3.0 --filter-distance-range 0.0 50.0 --data-root data/kitti_odometry --cpu --output-dir outputs/figures/week10_failure_case_b_filtered_025 --metrics-dir outputs/metrics/week10_failure_case_b_filtered_025
  python scripts/build_bev_memory_demo.py --config configs/eval/kitti_eval.yaml --train-config configs/train/posenet_3dof.yaml --pose-source learned --checkpoint outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt --sequence 07 --frames 5 --start-frame 20 --resolution-m 0.5 --profile-runtime --filter-z-range -3.0 2.0 --filter-distance-range 0.0 70.0 --data-root data/kitti_odometry --cpu --output-dir outputs/figures/week10_failure_case_c_start20_filtered --metrics-dir outputs/metrics/week10_failure_case_c_start20_filtered
  python scripts/build_bev_memory_demo.py --config configs/eval/kitti_cpu_smoke_safe.yaml --train-config configs/train/posenet_3dof.yaml --pose-source learned --checkpoint outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt --sequence 07 --frames 5 --data-root data/kitti_odometry --output-dir outputs/figures/week10_cpu_safe --metrics-dir outputs/metrics/week10_cpu_safe
  python scripts/build_bev_memory_demo.py --config configs/eval/kitti_rtx5080_safe.yaml --train-config configs/train/posenet_3dof.yaml --pose-source learned --checkpoint outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt --sequence 07 --frames 5 --profile-runtime --filter-z-range -1.5 3.0 --filter-distance-range 0.0 50.0 --data-root data/kitti_odometry --output-dir outputs/figures/week10_filtered_smoke --metrics-dir outputs/metrics/week10_filtered_smoke
result:
  M0 environment check completed and tests/test_smoke.py passed: 3 passed.
  M1 single-frame BEV render and GT-pose memory demo completed.
  M2 PoseNet eval path completed with zero_motion, constant_velocity, gt_label_echo, and learned baselines.
  Week 10 tests passed: 3 passed in tests/test_week10_robustness.py.
  Week 8-10 local tests passed: 11 passed.
  Runtime logs now report data_loading, rasterization, inference, warp, and rendering stages.
  Point filters support height clipping and horizontal radial distance clipping before BEV rasterization.
  Safe config smoke runs completed: CPU safe reported device=cpu; RTX 5080 safe reported device=cuda on this machine.
safe_configs:
  CPU smoke: configs/eval/kitti_cpu_smoke_safe.yaml uses device=cpu, amp=false, batch_size=1, max_frames=5, resolution_m=0.5.
  RTX 5080 conservative: configs/eval/kitti_rtx5080_safe.yaml uses device=cuda, amp=true, batch_size=2, max_frames=20, resolution_m=0.25.
failure_cases:
  - id: week10_case_a_start0_05
    artifact:
      outputs/figures/week10_failure_case_a_start0_05/07_000000_000004_memory.png
      outputs/figures/week10_failure_case_a_start0_05/07_000000_000004_trajectory.png
      outputs/figures/week10_failure_case_a_start0_05/07_000000_000004_alignment_curve.png
      outputs/metrics/week10_failure_case_a_start0_05/07_000000_000004_memory_metrics.json
    observation: learned_pose final occupancy_iou 0.541895 is worse than naive 0.689449; learned mean alignment 0.612429 is worse than naive 0.743782 and flicker is higher at 0.026496.
    suspected_cause: the checkpoint is smoke-scale and predicts adjacent motion poorly, so learned warps smear BEV memory instead of stabilizing it.
    mitigation_idea: train on full train split, add stronger validation, and reject learned updates when confidence or consistency falls below a threshold.
  - id: week10_case_b_filtered_025
    artifact:
      outputs/figures/week10_failure_case_b_filtered_025/07_000000_000004_memory.png
      outputs/figures/week10_failure_case_b_filtered_025/07_000000_000004_trajectory.png
      outputs/figures/week10_failure_case_b_filtered_025/07_000000_000004_alignment_curve.png
      outputs/metrics/week10_failure_case_b_filtered_025/07_000000_000004_memory_metrics.json
    observation: with 0.25m BEV plus z/range filtering, learned_pose occupancy_iou drops to 0.182357 while naive remains 0.405640; mean learned alignment is only 0.334335.
    suspected_cause: finer resolution makes small pose errors more visible, and filtering removes many far/height points so the weak learned pose has less redundant evidence.
    mitigation_idea: retrain or finetune at the target resolution and tune filters on validation data instead of applying them after a smoke-scale checkpoint.
  - id: week10_case_c_start20_filtered
    artifact:
      outputs/figures/week10_failure_case_c_start20_filtered/07_000020_000024_memory.png
      outputs/figures/week10_failure_case_c_start20_filtered/07_000020_000024_trajectory.png
      outputs/figures/week10_failure_case_c_start20_filtered/07_000020_000024_alignment_curve.png
      outputs/metrics/week10_failure_case_c_start20_filtered/07_000020_000024_memory_metrics.json
    observation: learned_pose and naive final occupancy_iou are nearly tied, 0.524814 vs 0.525206, but learned flicker is much higher at 0.027997 vs naive 0.003799.
    suspected_cause: learned updates do not reliably improve short-window alignment and introduce temporal instability even when final IoU looks similar.
    mitigation_idea: use temporal consistency gating, compare against constant-velocity pose priors, and record per-frame uncertainty before accepting learned memory updates.
notes:
  The learned model is still a smoke-level prototype, not a production or SOTA odometry model.
  Runtime logs are local diagnostic numbers; do not compare them across machines without fixed hardware and thread settings.
  data/ and outputs/ remain git-ignored runtime directories and must not be committed.
```

## 2026-06-17 Week 11 M3 Demo, Documentation, and Technical Report

```text
date: 2026-06-17
run_id: week11_m3_demo
config: configs/eval/kitti_cpu_smoke_safe.yaml
train_config: configs/train/posenet_3dof.yaml
dataset: KITTI Odometry sequence 07, first 5 frames
checkpoint: outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt
commands:
  python -m pytest tests/test_m3_demo_video.py
  python scripts/build_bev_memory_demo.py --config configs/eval/kitti_cpu_smoke_safe.yaml --train-config configs/train/posenet_3dof.yaml --pose-source learned --checkpoint outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt --sequence 07 --frames 5 --data-root data/kitti_odometry --cpu --output-dir outputs/figures/week11_m3_demo --metrics-dir outputs/metrics/week11_m3_demo
  python scripts/build_m3_demo_video.py --memory-image outputs/figures/week11_m3_demo/07_000000_000004_memory.png --trajectory-image outputs/figures/week11_m3_demo/07_000000_000004_trajectory.png --output outputs/figures/week11_m3_demo/neuralbev_lo_m3_demo.mp4 --seconds 6 --fps 6 --title "NeuralBEV-LO M3 KITTI seq07 smoke demo"
result:
  M3 video tests passed: 2 passed in tests/test_m3_demo_video.py.
  M3 demo video generated with 36 frames at 6 FPS.
  The video combines the memory panel and trajectory overlay.
  learned_pose final occupancy_iou 0.542051 remains worse than naive 0.689678; gt_pose remains 1.0.
  learned_pose mean alignment 0.612804 and flicker 0.026458 are worse than naive alignment 0.744705 and flicker 0.001996.
artifacts:
  outputs/figures/week11_m3_demo/neuralbev_lo_m3_demo.mp4
  outputs/figures/week11_m3_demo/07_000000_000004_memory.png
  outputs/figures/week11_m3_demo/07_000000_000004_trajectory.png
  outputs/figures/week11_m3_demo/07_000000_000004_alignment_curve.png
  outputs/metrics/week11_m3_demo/07_000000_000004_memory_metrics.json
  outputs/metrics/week11_m3_demo/07_000000_000004_consistency_metrics.json
  docs/architecture.md
  docs/resume_notes.md
  docs/m3_report.md
notes:
  This is an M3 integration demo and technical-report checkpoint, not a claim that learned odometry is strong.
  README, architecture notes, resume notes, and m3_report.md now tie claims to saved commands and artifact paths.
  "4D BEV" means a 2D BEV memory evolving over time, not a dense x/y/z/t reconstruction.
```

## 2026-06-17 Week 12 Cleanup, Reproducibility Freeze, and v0.1 Packaging

```text
date: 2026-06-17
run_id: week12_v0_1_reproducibility_freeze
commands:
  python scripts/run_v0_1_release_demo.py
  python scripts/check_env.py
  python scripts/run_pipeline_smoke.py --synthetic
  python -m pytest
  python scripts/train_posenet_3dof.py --config configs/train/posenet_3dof.yaml --synthetic --cpu --epochs 1 --batch-size 4 --overfit-batches 4 --output-dir outputs/checkpoints/week12_synthetic_freeze
  python scripts/eval_posenet.py --config configs/train/posenet_3dof.yaml --synthetic --max-pairs 6 --output-dir outputs/metrics/week12_synthetic_eval
  python scripts/build_bev_memory_demo.py --config configs/eval/kitti_cpu_smoke_safe.yaml --train-config configs/train/posenet_3dof.yaml --pose-source learned --checkpoint outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt --sequence 07 --frames 5 --data-root data/kitti_odometry --cpu --output-dir outputs/figures/week12_m3_freeze --metrics-dir outputs/metrics/week12_m3_freeze
  python scripts/build_m3_demo_video.py --memory-image outputs/figures/week12_m3_freeze/07_000000_000004_memory.png --trajectory-image outputs/figures/week12_m3_freeze/07_000000_000004_trajectory.png --output outputs/figures/week12_m3_freeze/neuralbev_lo_v0_1_demo.mp4 --seconds 6 --fps 6 --title "NeuralBEV-LO v0.1 research prototype"
  python scripts/write_v0_1_report.py
  python scripts/check_release_readiness.py --allow-pending-tag --require-artifacts
  python scripts/check_release_readiness.py --allow-pending-tag
result:
  One-command v0.1 release demo wrapper now runs BEV memory, video, and final report generation.
  Environment check completed on Python 3.12.10 with RTX 5080 CUDA-enabled PyTorch.
  Synthetic data-to-model smoke completed with loss 0.136254.
  Full test suite passed: 67 passed.
  Synthetic training freeze saved outputs/checkpoints/week12_synthetic_freeze/posenet_3dof_latest.pt with train_loss 0.075349 and val_loss 0.107954.
  Synthetic eval wrote zero_motion, constant_velocity, and gt_label_echo baselines; gt_label_echo ATE is 0.0.
  KITTI M3 freeze wrote memory metrics, consistency metrics, figures, and v0.1 demo video.
  v0.1 demo video exists with 36 frames at 6 FPS.
  v0.1 final report writes config, device, checkpoint, metrics, limitations, and artifact paths.
  Artifact-aware release readiness verifies the one-command release demo outputs before tagging.
  Release readiness passed all static checks except release_tag, which is intentionally pending until user approval.
metrics:
  final occupancy_iou: naive 0.689678, gt_pose 1.0, learned_pose 0.542051.
  mean per-frame consistency: naive alignment 0.744705 / flicker 0.001996; gt_pose alignment 0.791612 / flicker 0.008939; learned_pose alignment 0.612804 / flicker 0.026458.
artifacts:
  outputs/figures/v0_1_release_demo/neuralbev_lo_v0_1_release_demo.mp4
  outputs/reports/v0_1_release_demo.txt
  outputs/checkpoints/week12_synthetic_freeze/posenet_3dof_latest.pt
  outputs/metrics/week12_synthetic_eval/synthetic_eval_metrics.json
  outputs/metrics/week12_synthetic_eval/synthetic_eval_metrics.csv
  outputs/metrics/week12_synthetic_eval/synthetic_eval_trajectory.png
  outputs/figures/week12_m3_freeze/07_000000_000004_memory.png
  outputs/figures/week12_m3_freeze/07_000000_000004_trajectory.png
  outputs/figures/week12_m3_freeze/07_000000_000004_alignment_curve.png
  outputs/figures/week12_m3_freeze/neuralbev_lo_v0_1_demo.mp4
  outputs/reports/week12_v0_1_final_report.txt
  outputs/metrics/week12_m3_freeze/07_000000_000004_memory_metrics.json
  outputs/metrics/week12_m3_freeze/07_000000_000004_consistency_metrics.json
  docs/release_checklist.md
notes:
  No release tag was created in this run. Create and push v0.1-research-prototype only after explicit user approval.
  data/, outputs/, work/, checkpoints, and raw arrays remain ignored runtime artifacts and must not be committed.
```
