# Data Contract

## Dataset Target

Phase 1 starts with KITTI Odometry. SemanticKITTI, nuScenes, OpenOccupancy, and semantic labels are later-stage extensions.

## Required KITTI Files

For each sequence used by a command, the project expects:

- `sequences/<seq>/velodyne/*.bin`
- `sequences/<seq>/calib.txt`
- `sequences/<seq>/times.txt`
- `poses/<seq>.txt`

The loader also supports the official split download layout without moving files:

- `data_odometry_velodyne/dataset/sequences/<seq>/velodyne/*.bin`
- `data_odometry_calib/dataset/sequences/<seq>/calib.txt`
- `data_odometry_calib/dataset/sequences/<seq>/times.txt`
- `data_odometry_poses/dataset/poses/<seq>.txt`

The implementation must validate that timestamps, poses, and Velodyne frames have consistent counts before processing.

## Point Cloud Shape

Velodyne `.bin` files are expected to load as `float32[N, 4]` with columns:

```text
x, y, z, intensity
```

Malformed, empty, or non-finite point clouds should produce structured warnings or validation errors rather than silent corruption.

## Artifact Rules

Generated artifacts must live under `outputs/` using a run id:

```text
{experiment_name}_{YYYYMMDD_HHMMSS}
```

Commands that write artifacts must print the resolved `run_id` and output directory.

## Split Rules

Default splits:

```text
smoke: sequence 00, first 20-50 adjacent pairs
train: sequences 00-06
val: sequences 07-08
test/demo: sequences 09-10
```

Pose-normalization statistics must be computed from the train split only. Validation and test labels must not influence normalization.
