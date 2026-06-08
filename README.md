# NeuralBEV-LO

Learning-based LiDAR Odometry for Streaming 4D BEV Mapping.

NeuralBEV-LO is an open-source research prototype, not a production autonomous-driving stack. The first target is a small, reproducible KITTI Odometry pipeline that converts sequential LiDAR frames into BEV tensors, builds a ground-truth-pose BEV memory baseline, trains a lightweight relative-pose network, and compares learned-pose BEV memory against simple baselines.

## Repository

- Remote: `git@github.com:2025wwzx/NeuralBEV-LO.git`
- Workspace: `E:\AAAworkspace\study\NeuralBEV-LO`
- Primary dataset target: KITTI Odometry
- First milestone: M0 reproducible skeleton

## Current Scope

The Week 1 deliverable establishes the project skeleton, configuration layout, data contract, coordinate-system contract, structured logging helper, environment diagnostics, and smoke tests. Modeling code, KITTI parsing, BEV rasterization, and training are intentionally deferred to later phases in `NeuralBEV-LO-execution-plan.md`.

## Environment

The current local diagnostic snapshot is:

```text
Python: 3.12.10
GPU: NVIDIA GeForce RTX 5080
Driver CUDA reported by nvidia-smi: 13.2
PyTorch observed locally: 2.12.0+cu132
torch.cuda.is_available(): True
```

The project metadata allows Python `>=3.10,<3.13`. CUDA and PyTorch installation commands should be documented only after `scripts/check_env.py` confirms the working local combination.

Week 1 fallback ladder:

1. Use the native Windows Python + CUDA environment when `torch.cuda.is_available()` is true.
2. Use WSL2 only if Windows CUDA package compatibility blocks PyTorch or OpenCV.
3. Keep CPU-only synthetic smoke tests runnable even when KITTI data or CUDA is unavailable.

## Quick Smoke Commands

```powershell
python scripts/check_env.py
python -m pytest
python -c "import neuralbev_lo"
python scripts/run_pipeline_smoke.py --synthetic
```

## Data

Large datasets are not committed. See `data/README.md` for the expected KITTI layout and environment-variable options.

## Documentation

- `NeuralBEV-LO-execution-plan.md` - full 12-week execution plan.
- `docs/data_contract.md` - dataset and artifact rules.
- `docs/coordinate_system.md` - KITTI pose, LiDAR pose, BEV grid, and warp conventions.
- `docs/architecture.md` - evolving architecture notes.
- `docs/experiment_log.md` - experiment and failure-case log.
- `docs/resume_notes.md` - artifact-backed resume wording.
