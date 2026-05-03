# Verification: TRELLIS.2 Image Preprocessing Background

**Date:** 2026-05-02
**Verifier:** Codex `auto-verify`

## Commands Run

```bash
uv run pytest
uv run python -c "from mlx_spatial.trellis2_inference import Trellis2InferencePipeline; p=Trellis2InferencePipeline('weights/trellis2', rmbg_root='weights/rmbg2'); a=p.attempt('inputs/trellis2/demo-alpha.webp'); r=p.attempt('inputs/trellis2/demo-rgb-background.png'); print('alpha_completed=', a.completed_stages); print('alpha_blocker=', a.blocker.stage, a.blocker.operation); print('rgb_completed=', r.completed_stages); print('rgb_blocker=', r.blocker.stage, r.blocker.operation); print('rgb_reason=', r.blocker.reason)"
uv run mlx-spatial-trellis2 rmbg-validate --root weights/rmbg2
uv run python -c "from mlx_spatial import preprocess_trellis2_image; r=preprocess_trellis2_image('inputs/trellis2/demo-alpha.webp'); print('ready=', r.ready); print('input_mode=', r.image.input_mode if r.image else None); print('output_mode=', r.image.output_mode if r.image else None); print('output_size=', r.image.output_size if r.image else None); print('had_alpha=', r.image.had_input_alpha if r.image else None)"
uv run python -c "from mlx_spatial import inspect_rmbg2_key_inventory, load_rmbg2_tensors; inv=inspect_rmbg2_key_inventory('weights/rmbg2'); tensors=load_rmbg2_tensors('weights/rmbg2', names=[inv.sample_keys[0]]); print('tensor_count=', inv.tensor_count); print('prefixes=', inv.top_level_prefixes); print('loaded=', len(tensors), tensors[0].name, tensors[0].shape, tensors[0].dtype)"
uv run python -c "from mlx_spatial import rmbg2_download_command; print(rmbg2_download_command('weights/rmbg2'))"
git status --short --ignored
rg "import (torch|torchvision|transformers|huggingface_hub|onnx)|from (torch|torchvision|transformers|huggingface_hub|onnx)|vendors" src pyproject.toml
```

## Criteria

### Criterion 1: Public preprocessing API exists and is used by attempt mode

- **Result:** PASS
- **Evidence:** `uv run pytest` reported `tests/test_trellis2_inference.py ...........`, `tests/test_trellis2_preprocess.py .........`, and `83 passed, 5 skipped`; live attempt output included `alpha_completed= ('input-image', 'asset-config-validation', 'checkpoint-probe-readiness', 'image-preprocessing-background')`.
- **Gap:** none

### Criterion 2: RGBA fixtures verify decode, resize, alpha crop, composite, output mode and shape

- **Result:** PASS
- **Evidence:** `uv run pytest` reported `tests/test_trellis2_preprocess.py .........`; live preprocessing output reported `ready= True`, `input_mode= RGBA`, `output_mode= RGB`, `output_size= (704, 704)`, and `had_alpha= True`.
- **Gap:** none

### Criterion 3: RGB fixtures verify RMBG path selection and deterministic blocker behavior when assets are absent

- **Result:** PASS
- **Evidence:** `uv run pytest` reported `tests/test_trellis2_preprocess.py .........` and `tests/test_trellis2_inference.py ...........`; these include RGB missing-RMBG and configured-RMBG blocker cases.
- **Gap:** none

### Criterion 4: RMBG asset validation is deterministic and offline

- **Result:** PASS
- **Evidence:** `uv run mlx-spatial-trellis2 rmbg-validate --root weights/rmbg2` reported `ready=True`, `present=4`, `missing=0`.
- **Gap:** none

### Criterion 5: Manual RMBG download/help surface names repo, root, and gated/non-commercial boundary

- **Result:** PASS
- **Evidence:** `uv run python -c "from mlx_spatial import rmbg2_download_command; ..."` reported `('uv', 'run', 'hf', 'download', 'briaai/RMBG-2.0', '--local-dir', 'weights/rmbg2')`; README states `briaai/RMBG-2.0` is gated, non-commercial, and not downloaded during import, tests, validation, or inference attempts.
- **Gap:** none

### Criterion 6: Local RMBG safetensors can be inspected/loaded as MLX arrays

- **Result:** PASS
- **Evidence:** `uv run python -c "from mlx_spatial import inspect_rmbg2_key_inventory, load_rmbg2_tensors; ..."` reported `tensor_count= 754`, `prefixes= ('bb', 'decoder', 'squeeze_module')`, and `loaded= 1 bb.layers.0.blocks.0.attn.proj.bias (192,) float32`.
- **Gap:** none

### Criterion 7: If BiRefNet runnable parity is reached, RGB preprocessing completes

- **Result:** PASS
- **Evidence:** The runnable-parity condition was not reached; the alternate incomplete-port criterion below was exercised. Live RGB attempt reported a precise blocker rather than claiming completion.
- **Gap:** none

### Criterion 8: If BiRefNet is incomplete, the blocker is precise and not generic

- **Result:** PASS
- **Evidence:** Live RGB attempt reported `rgb_blocker= image-preprocessing-background MLX BiRefNet deformable convolution` and `rgb_reason= RMBG-2.0 BiRefNet imports torchvision.ops.deform_conv2d, but mlx.nn has no DeformConv2d implementation`.
- **Gap:** none

### Criterion 9: RGBA-alpha real attempt advances to image-conditioning

- **Result:** PASS
- **Evidence:** Live alpha attempt reported `alpha_completed= ('input-image', 'asset-config-validation', 'checkpoint-probe-readiness', 'image-preprocessing-background')` and `alpha_blocker= image-conditioning MLX image feature extraction / conditioning`.
- **Gap:** none

### Criterion 10: Default test suite passes without forbidden runtime requirements

- **Result:** PASS
- **Evidence:** `uv run pytest` reported `83 passed, 5 skipped`; `pyproject.toml` runtime dependencies include `mlx`, `numpy`, `pillow>=12.2.0`, and `safetensors`. `huggingface-hub>=0.36` is in the dev group. The forbidden-import scan found no runtime imports of PyTorch, TorchVision, Transformers, Hugging Face Hub, or ONNX in `src`.
- **Gap:** none

### Criterion 11: Real weights, large images, and generated outputs are not tracked

- **Result:** PASS
- **Evidence:** `git status --short --ignored` showed `!! inputs/`, `!! outputs/`, and `!! weights/`.
- **Gap:** none

### Criterion 12: README or CLI help documents preprocessing behavior and remaining boundary

- **Result:** PASS
- **Evidence:** README documents `weights/rmbg2/`, `rmbg-validate`, `rmbg-download-command`, gated/non-commercial status, alpha preprocessing behavior, RGB/RMBG behavior, the `MLX BiRefNet deformable convolution` blocker, and the next `image-conditioning` boundary.
- **Gap:** none

## Content Checks

- **Result:** PASS
- **Audience:** README addresses developers running local TRELLIS.2/RMBG attempts with exact commands and ignored path conventions.
- **Thesis:** The docs state that alpha preprocessing now works, RGB requires local RMBG, and the current RGB blocker is deformable convolution in BiRefNet.
- **Source policy:** No external citations or long quotations were introduced; local facts are backed by command output and file paths.
- **Anti-goals:** The docs do not claim full TRELLIS.2 image-to-3D inference, do not imply silent downloads, and do not add PyTorch/ONNX fallback guidance.
- **Anti-slop scan:** No promotional or inflated completion language found.

## Summary

- **Overall:** PASS
- **Passed:** 12 of 12 criteria
- **Remaining gaps:** none for this change
- **Known next blocker:** `image-conditioning` for alpha inputs; `MLX BiRefNet deformable convolution` for RGB/RMBG background removal.
- **Recommended next skill:** `auto-frame` for the next change, likely `trellis-image-conditioning`.
