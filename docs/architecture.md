# Architecture Notes

The first architecture milestone is intentionally small:

```text
KITTI files -> data validation -> BEV tensors -> smoke pipeline
```

Later milestones add:

1. GT-pose BEV memory baseline.
2. Adjacent-frame 3DoF labels.
3. Lightweight PoseNet.
4. Learned-pose BEV memory.
5. Odometry and BEV consistency metrics.

Heavy sparse convolution, Transformers, ROS, loop closure, and custom CUDA kernels are out of scope for v0.1.
