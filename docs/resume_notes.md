# Resume Notes

## Wording Guardrail

In this project, "4D BEV" means a 2D spatial BEV memory evolving over time. It
does not claim a full dense `(x, y, z, t)` reconstruction.

Avoid saying the learned odometry is strong or production-ready. The current
checkpoint is smoke-scale; the honest claim is that the end-to-end pipeline,
baselines, metrics, failure analysis, and demo artifacts are reproducible.

## Project Title

```text
NeuralBEV-LO: Learning-based LiDAR Odometry for Streaming 4D BEV Mapping
```

## Artifact-Backed Bullets

```text
- Built an end-to-end PyTorch/KITTI Odometry prototype that converts sequential Velodyne frames into BEV tensors, predicts adjacent 3DoF motion, and updates a temporal BEV memory; M3 demo artifacts are generated under outputs/figures/week11_m3_demo/.
- Implemented KITTI split-layout loading, LiDAR pose conversion, BEV rasterization, differentiable BEV memory warping, odometry metrics, BEV consistency metrics, and runtime/failure-case logging.
- Compared learned-pose BEV memory against naive and GT-pose memory baselines; documented smoke-scale failure cases where learned memory underperforms naive memory instead of hiding weak results.
- Produced a reproducible M3 demo video combining current/naive/GT/learned BEV memory with trajectory overlay, plus architecture notes and an experiment log tied to saved commands and artifact paths.
```

## Evidence To Cite

- Demo video: `outputs/figures/week11_m3_demo/neuralbev_lo_m3_demo.mp4`
- M3 memory panel: `outputs/figures/week11_m3_demo/07_000000_000004_memory.png`
- M3 trajectory overlay: `outputs/figures/week11_m3_demo/07_000000_000004_trajectory.png`
- M3 metrics: `outputs/metrics/week11_m3_demo/07_000000_000004_memory_metrics.json`
- Failure cases: `docs/experiment_log.md`
- Architecture summary: `docs/architecture.md`
- M3 report: `docs/m3_report.md`
