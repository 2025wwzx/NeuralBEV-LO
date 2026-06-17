# Architecture Notes

NeuralBEV-LO is a small research prototype for LiDAR odometry and temporal BEV
memory. The v0.1 scope is intentionally narrow: KITTI Odometry input,
density/height/intensity BEV tensors, a lightweight 3DoF PoseNet, and metrics
that compare learned-pose memory against simple baselines.

## Pipeline

```mermaid
flowchart LR
    A["KITTI Odometry\nVelodyne, calib, poses"] --> B["Data validation\npaths, poses, point clouds"]
    B --> C["Point filtering\nz and radial range"]
    C --> D["BEV rasterizer\nC x H x W tensor"]
    D --> E["Adjacent pair dataset\nprev/current BEV"]
    E --> F["PoseNet3DoF\npredict dx, dy, yaw"]
    B --> G["GT relative pose\nupper-bound label"]
    F --> H["Learned-pose memory update"]
    G --> I["GT-pose memory update"]
    D --> J["Naive memory baseline"]
    H --> K["Figures and metrics"]
    I --> K
    J --> K
    K --> L["M3 demo video\nmemory panel + trajectory overlay"]
```

## Coordinate System

```mermaid
flowchart TB
    A["KITTI poses.txt\nT_world_cam0"] --> B["Calibration\nTr_velo_to_cam"]
    B --> C["Internal pose\nT_world_lidar"]
    C --> D["Relative label\nT_prev_curr = inv(T_prev) @ T_curr"]
    D --> E["3DoF target\n dx, dy, yaw in previous LiDAR frame"]
    E --> F["BEV grid\nx forward -> row\ny left -> column"]
    F --> G["BEV warp\ninverse sampling for grid_sample"]
```

The BEV grid uses x as forward distance and y as left distance. Tensor shape is
`[C, H, W]`, where H indexes x bins and W indexes y bins. The ego vehicle is not
assumed to be at the image center because the default x range starts at `0m`.

## Implemented Components

- `neuralbev_lo/data/kitti_dataset.py` loads KITTI Odometry in either merged or
  official split layout.
- `neuralbev_lo/data/pose_utils.py` converts KITTI camera poses into internal
  LiDAR poses and builds adjacent relative labels.
- `neuralbev_lo/bev/rasterizer.py` converts `float32[N,4]` point clouds into
  BEV tensors with density, height, and intensity channels.
- `neuralbev_lo/bev/memory.py` updates naive, GT-pose, and learned-pose BEV
  memories.
- `neuralbev_lo/models/posenet.py` defines the lightweight 3DoF PoseNet.
- `neuralbev_lo/eval/odometry_metrics.py` and `neuralbev_lo/eval/bev_metrics.py`
  save odometry and BEV consistency metrics.
- `scripts/build_bev_memory_demo.py` is the main end-to-end demo command.
- `scripts/build_m3_demo_video.py` packages the memory panel and trajectory
  overlay into the Week 11 demo MP4.

## Out Of Scope

The current v0.1 scope does not include sparse convolution backbones, loop
closure, map optimization, ROS integration, semantic labels, or custom CUDA
kernels. Learned-pose results are smoke-scale and documented as weaker than the
GT-pose upper bound and often weaker than naive memory.
