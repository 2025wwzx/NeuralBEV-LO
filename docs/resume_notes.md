# Resume Notes

Do not use this project on a resume until the artifacts in `NeuralBEV-LO-execution-plan.md` are complete.

## Wording Guardrail

In this project, "4D BEV" means a 2D spatial BEV memory evolving over time. It does not claim a full dense `(x, y, z, t)` reconstruction.

## Draft Title

```text
NeuralBEV-LO: Learning-based LiDAR Odometry for Streaming 4D BEV Mapping
```

## Draft Bullets

These bullets are placeholders until backed by artifacts:

```text
- Built a PyTorch research prototype that converts sequential LiDAR frames into a temporal BEV memory using learned relative pose estimation and differentiable BEV warping.
- Implemented KITTI LiDAR loading, BEV rasterization, relative pose regression, temporal memory update, odometry metrics, and BEV consistency visualization.
- Evaluated learned odometry against ground-truth-pose accumulation and simple motion baselines using trajectory metrics and BEV memory quality indicators.
```
