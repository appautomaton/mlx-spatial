# ATTEMPT: TRELLIS.2 End-to-End Inference

## Command

```bash
uv run python - <<'PY'
from mlx_spatial import Trellis2InferencePipeline, validate_trellis2_assets
root = 'weights/trellis2'
image = '.agent/work/trellis-e2e-inference-attempt/sample-input.txt'
validation = validate_trellis2_assets(root)
print('ready=', validation.ready)
print('missing=', list(validation.missing))
report = Trellis2InferencePipeline(root).attempt(image, load_probes=True)
print('completed=', report.completed)
print('completed_stages=', list(report.completed_stages))
if report.blocker:
    print('blocker_stage=', report.blocker.stage)
    print('blocker_operation=', report.blocker.operation)
    print('blocker_reference=', report.blocker.reference)
    print('blocker_reason=', report.blocker.reason)
    print('blocker_next_slice=', report.blocker.next_slice)
PY
```

## Real Asset Readiness

- `ready=True`
- `missing=[]`
- Root: `weights/trellis2`

## Sample Input

- Path: `.agent/work/trellis-e2e-inference-attempt/sample-input.txt`
- Purpose: local file-existence input for the first attempt boundary.
- Note: this slice validates pipeline wiring and blocker behavior; it does not implement image decoding or preprocessing yet.

## Outcome

- `completed=False`
- Completed stages:
  - `input-image`
  - `asset-config-validation`
  - `checkpoint-probe-readiness`

## Blocker Ledger

- Stage: `image-preprocessing-background`
- Operation: `MLX/Python image preprocessing and background removal boundary`
- Reference: `vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:127-162`
- Reason: `stage is traced but not implemented in MLX`
- Next slice: `implement image-preprocessing-background for TRELLIS.2 inference`

## Interpretation

The first end-to-end attempt validates local real weights and loads configured probes, then stops before image preprocessing/background handling. This is the expected first structured blocker from `FLOW.md`.
