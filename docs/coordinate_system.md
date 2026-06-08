# Coordinate System Contract

## KITTI Pose Source

KITTI Odometry `poses/<seq>.txt` is treated as camera-frame world pose, commonly `T_world_cam0`.

The internal project pose convention is LiDAR-frame world pose:

```text
T_world_lidar
```

Before computing labels, code must convert camera-frame pose to LiDAR-frame pose using `Tr_velo_to_cam` from `calib.txt`.

## Relative Pose Label

Adjacent relative pose is:

```text
T_prev_curr = inv(T_world_prev) @ T_world_curr
```

The first learning target is 3DoF:

```text
dx, dy, yaw
```

`dx`, `dy`, and `yaw` are expressed in the previous LiDAR/BEV frame unless a config explicitly says otherwise.

## BEV Grid

Default BEV convention:

- x points forward.
- y points left.
- x range starts at `0m`.
- y range is symmetric around the ego vehicle.
- Pixel row/column mapping must be tested with synthetic fixtures before KITTI demos.

## Warp Direction

BEV memory warp uses the inverse transform required by `torch.nn.functional.grid_sample`.

Synthetic tests must cover:

- identity warp
- translation
- yaw
- inverse transform
- boundary padding

For example, a 10m forward translation at 0.25m resolution corresponds to a 40-cell shift within a one-cell tolerance after interpolation.
