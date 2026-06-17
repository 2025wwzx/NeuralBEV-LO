# v0.1 Release Checklist

This checklist freezes NeuralBEV-LO as a presentable research prototype. It does
not claim production odometry accuracy.

## Required Before Tagging

- [x] Full test suite passes with `python -m pytest`.
- [x] Environment smoke passes with `python scripts/check_env.py`.
- [x] Synthetic data-to-model smoke passes with `python scripts/run_pipeline_smoke.py --synthetic`.
- [x] KITTI M1 single-frame BEV render path is documented.
- [x] KITTI GT-pose memory path is documented.
- [x] KITTI learned-pose M3 demo path is documented.
- [x] One-command v0.1 release demo wrapper is documented.
- [x] M3 demo video command is documented.
- [x] v0.1 final report command is documented.
- [x] v0.1 release manifest command is documented.
- [x] Failure cases and limitations are documented in `docs/experiment_log.md`.
- [x] Architecture and coordinate-system docs are present.
- [x] Resume notes are artifact-backed and include the 4D BEV wording guardrail.
- [x] v0.1 GitHub release-note draft exists in `docs/release_notes_v0_1.md`.
- [x] `data/`, `outputs/`, `work/`, checkpoints, and raw arrays are ignored by git.
- [x] `scripts/check_release_readiness.py --allow-pending-tag` passes.
- [x] `scripts/check_release_readiness.py --allow-pending-tag --require-artifacts` passes after running the one-command release demo.
- [x] `scripts/check_release_readiness.py --allow-pending-tag --require-artifacts --require-clean-git` passes after final commit and push.
- [ ] User explicitly approves creating tag `v0.1-research-prototype`.
- [ ] After approval, create and push tag `v0.1-research-prototype`.

## Reproducibility Freeze Command Sequence

One-command release demo:

```powershell
python scripts/run_v0_1_release_demo.py
```

Expanded command sequence:

```powershell
python scripts/check_env.py
python scripts/run_pipeline_smoke.py --synthetic
python -m pytest
python scripts/train_posenet_3dof.py --config configs/train/posenet_3dof.yaml --synthetic --cpu --epochs 1 --batch-size 4 --overfit-batches 4 --output-dir outputs/checkpoints/week12_synthetic_freeze
python scripts/eval_posenet.py --config configs/train/posenet_3dof.yaml --synthetic --max-pairs 6 --output-dir outputs/metrics/week12_synthetic_eval
python scripts/build_bev_memory_demo.py --config configs/eval/kitti_cpu_smoke_safe.yaml --train-config configs/train/posenet_3dof.yaml --pose-source learned --checkpoint outputs/checkpoints/week6_kitti_tiny_smoke/posenet_3dof_latest.pt --sequence 07 --frames 5 --data-root data/kitti_odometry --cpu --output-dir outputs/figures/week12_m3_freeze --metrics-dir outputs/metrics/week12_m3_freeze
python scripts/build_m3_demo_video.py --memory-image outputs/figures/week12_m3_freeze/07_000000_000004_memory.png --trajectory-image outputs/figures/week12_m3_freeze/07_000000_000004_trajectory.png --output outputs/figures/week12_m3_freeze/neuralbev_lo_v0_1_demo.mp4 --seconds 6 --fps 6 --title "NeuralBEV-LO v0.1 research prototype"
python scripts/write_v0_1_report.py
python scripts/write_release_manifest.py
python scripts/check_release_readiness.py --allow-pending-tag --require-artifacts
python scripts/check_release_readiness.py --allow-pending-tag --require-artifacts --require-clean-git
python scripts/check_release_readiness.py --allow-pending-tag
```

## Tagging

Tagging is intentionally manual. After reviewing the generated artifacts and this
checklist, and the release-note draft in `docs/release_notes_v0_1.md`, the user
can approve:

```powershell
git tag v0.1-research-prototype
git push origin v0.1-research-prototype
```

Do not create or push the tag without explicit user approval in the current
conversation.
