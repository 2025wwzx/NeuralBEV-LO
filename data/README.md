# Data Directory

Large datasets and generated caches stay out of git.

Expected KITTI Odometry layout:

```text
data/kitti_odometry/
  sequences/
    00/
      calib.txt
      times.txt
      velodyne/
        000000.bin
        000001.bin
        ...
  poses/
    00.txt
    01.txt
    ...
```

You may also set `KITTI_ROOT` to point at another KITTI Odometry root. Runtime code should resolve the dataset root from `configs/dataset/kitti.yaml` or `KITTI_ROOT`, never from a hard-coded absolute path.

Do not commit:

- KITTI `.bin` frames.
- Pose downloads copied from the dataset.
- Cached BEV arrays.
- Checkpoints, videos, metrics, or logs.

Use `outputs/` for generated artifacts.
