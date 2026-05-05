# Slice 009 Summary

## Scope

Live 518 Reconstruction Verification.

## Result

- Verified the HY-World targeted reconstruction/export tests.
- Verified the local live asset state under `weights/hy-world-2`.
- Wrote `outputs/hyworld2/live-setup-check/trace.json` with a clean `asset-validation` blocker.
- Fixed the unrelated TRELLIS SLat full-suite collection blocker by restoring exact chunked self-attention exports and guard behavior.

## Evidence

- `uv run pytest -q tests/test_hyworld2_tools.py tests/test_hyworld2_inference.py tests/test_hyworld2_export.py` -> 27 passed.
- `uv run mlx-spatial-hyworld2 validate weights/hy-world-2` -> `ready=False` with missing `HY-WorldMirror-2.0/model.safetensors` and `HY-WorldMirror-2.0/config.yaml or HY-WorldMirror-2.0/config.json`.
- `uv run mlx-spatial-hyworld2 reconstruct weights/hy-world-2 inputs/trellis2/demo-rgb-background.png --output outputs/hyworld2/live-setup-check --heads depth,normal,points --memory-profile safe --trace-output outputs/hyworld2/live-setup-check/trace.json` -> exit 2, clean `asset-validation` blocker.
- `uv run pytest -q tests/test_hyworld2_tools.py tests/test_hyworld2_preprocess.py tests/test_hyworld2_assets.py tests/test_hyworld2_inference.py tests/test_hyworld2_worldmirror.py tests/test_hyworld2_heads.py tests/test_hyworld2_export.py tests/test_model_assets.py` -> 78 passed.
- `uv run pytest -q tests/test_trellis2_slat.py` -> 22 passed.
- `uv run pytest -q` -> 319 passed, 5 skipped.
- `git diff --check` -> passed.

## Stop Reason

All approved slices are complete. Live reconstruction output is blocked only by missing local HY-WorldMirror weights/config.
