# Verification: TRELLIS.2 Forward Trace Conditioning

**Date:** 2026-05-02
**Verifier:** Codex `auto-verify`

## Commands Run

```bash
uv run pytest
uv run python -c "from mlx_spatial import Trellis2InferencePipeline; r=Trellis2InferencePipeline('weights/trellis2').attempt_forward_trace('inputs/trellis2/demo-alpha.webp'); print('completed=', r.completed_stages); print('outputs=', r.outputs); print('blocker_stage=', r.blocker.stage if r.blocker else None); print('operation=', r.blocker.operation if r.blocker else None); print('reason=', r.blocker.reason if r.blocker else None)"
uv run python -c "<fake TRELLIS.2 fixture with simulated conditioning>"
uv run python -c "from mlx_spatial import discover_trellis2_conditioning_config, default_dinov3_root; d=discover_trellis2_conditioning_config('weights/trellis2'); c=d.config; print(...)"
rg "import (torch|torchvision|transformers|huggingface_hub|onnx)|from (torch|torchvision|transformers|huggingface_hub|onnx)" src pyproject.toml
git status --short --ignored
```

## Criteria

### Criterion 1: Forward-trace entry point runs alpha preprocessing and invokes conditioning

- **Result:** PASS
- **Evidence:** Live command reported `completed= ('input-image', 'asset-config-validation', 'checkpoint-probe-readiness', 'image-preprocessing-background')`, then `blocker_stage= image-conditioning` and `operation= local DINOv3 asset validation`.
- **Gap:** none

### Criterion 2: Fake fixtures can advance past image-conditioning and record stage output metadata

- **Result:** PASS
- **Evidence:** Fake-fixture command with simulated conditioning reported `completed= ('input-image', 'asset-config-validation', 'checkpoint-probe-readiness', 'image-preprocessing-background', 'image-conditioning')` and `outputs= [('image-conditioning', (1, 1024, 1024), 'float32')]`.
- **Gap:** none

### Criterion 3: Fake fixtures return precise blockers for missing config, keys, or unsupported operations

- **Result:** PASS
- **Evidence:** `uv run pytest` reported `tests/test_trellis2_forward.py ...........`; these tests cover malformed pipeline config, missing DINOv3 assets, present DINOv3 assets with missing MLX module construction, conditioning width mismatch, and sparse-flow model construction blockers.
- **Gap:** none

### Criterion 4: Real alpha attempt no longer returns the old generic image-conditioning blocker

- **Result:** PASS
- **Evidence:** Live command reported `operation= local DINOv3 asset validation`, not `MLX image feature extraction / conditioning`; reason named missing `weights/dinov3-vitl16-pretrain-lvd1689m/config.json` and `model.safetensors`.
- **Gap:** none

### Criterion 5: Real attempt evidence names input, root, completed stages, outputs, and blocker

- **Result:** PASS
- **Evidence:** `ATTEMPT.md` names `inputs/trellis2/demo-alpha.webp`, `weights/trellis2`, `outputs/trellis2/forward-trace/demo-alpha-forward-trace.json`, completed stages, empty outputs, and the `local DINOv3 asset validation` blocker.
- **Gap:** none

### Criterion 6: Default tests pass without real weights, network, HF credentials, PyTorch, TorchVision, Transformers, ONNX Runtime, or vendor imports

- **Result:** PASS
- **Evidence:** `uv run pytest` reported `94 passed, 5 skipped`; the forbidden import scan over `src pyproject.toml` returned no matches for PyTorch, TorchVision, Transformers, Hugging Face Hub, or ONNX imports.
- **Gap:** none

### Criterion 7: Runtime dependency metadata still excludes forbidden runtime dependencies

- **Result:** PASS
- **Evidence:** The forbidden dependency/import scan returned no matches in runtime surfaces; `huggingface-hub` remains dev tooling from the prior asset-download workflow, not a runtime import.
- **Gap:** none

### Criterion 8: Real weights, large outputs, and generated attempt artifacts are not tracked

- **Result:** PASS
- **Evidence:** `git status --short --ignored` reported `!! inputs/`, `!! outputs/`, and `!! weights/`.
- **Gap:** none

### Criterion 9: README documents forward-trace boundary and next known blocker

- **Result:** PASS
- **Evidence:** README documents `attempt_forward_trace(image_path)`, DINOv3 config values, the missing local DINOv3 asset paths, and the fake-fixture sparse boundary `sparse-structure-sampling` / `MLX sparse structure flow model construction`.
- **Gap:** none

## Content Checks

- **Result:** PASS
- **Audience:** README addresses developers running local TRELLIS.2 attempts by naming exact APIs, paths, and blocker fields.
- **Thesis:** The docs state that forward trace now enters image-conditioning, resolves DINOv3 config, and stops on local DINOv3 assets unless fake conditioning metadata is injected for sparse-boundary testing.
- **Source policy:** Claims are backed by local config discovery, command output, and repository paths; no external citations or quotations were introduced.
- **Anti-goals:** The docs do not claim full TRELLIS.2 inference, do not add PyTorch/Transformers runtime guidance, and do not suggest silent downloads.
- **Anti-slop scan:** No promotional or inflated completion language found.

## Summary

- **Overall:** PASS
- **Passed:** 9 of 9 criteria
- **Remaining gaps:** none for this change
- **Known next blocker:** local DINOv3 assets for `facebook/dinov3-vitl16-pretrain-lvd1689m` under `weights/dinov3-vitl16-pretrain-lvd1689m`.
- **Recommended next skill:** `auto-frame` for a DINOv3 local asset and MLX module-construction slice.
