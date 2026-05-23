# Slice 3 — Image Conditioner Adapter (LITO-B)

Parallel-safe with Slices 1, 2, 4. Consumes only source-contract fixtures from `tests/fixtures/lito/cond_*.safetensors` (generated in Slice 0B).

## Inputs

- `INTAKE.md`, `SPEC.md`, `PLAN.md`, this slice file
- `spec/gap-matrix.md` § LITO-B — final tolerance (closed in Slice 0)
- `slices/00-vendor-assets-routing.md` § Upstream Coverage Audit — records that LiTo uses PyTorch `SpatialDinov2`
- `tests/fixtures/lito/manifest.json` — source contract/function metadata for conditioner fixtures
- `tests/fixtures/lito/cond_input_*.safetensors`, `cond_output_*.safetensors`

## Upstream Files to Read First

(filled in by Slice 0; placeholders below)

1. `vendors/ml-lito/src/lito/trainers/lito_dit_trainer.py::get_image_conditioning` and `::load_img_encoder`
2. `vendors/ml-lito/src/lito/models/dino.py::SpatialDinov2`
3. The point where the encoder output is consumed by the DiT (to confirm the expected feature shape and dtype)

## Reference Patterns in mlx-spatial

- `src/mlx_spatial/trellis2_dinov3.py` and `trellis2_dinov3_forward.py` — if LiTo uses DINOv3, reuse this directly
- `src/mlx_spatial/hyworld2_vit.py` — alternative ViT-based encoder reference
- `src/mlx_spatial/sam3d_condition.py` — closest parallel to a "thin adapter" pattern
- `src/mlx_spatial/trellis2_rmbg.py` and `trellis2_rmbg_forward.py` — image preprocessing reference (background removal, normalization)

## Implementation Outline

Two cases, decided by Slice 0:

### Case A: Upstream uses an encoder mlx-spatial already has (e.g., DINOv3)

Write `src/mlx_spatial/lito_condition.py` as a **thin adapter** that:

1. Imports the existing encoder (e.g., `from mlx_spatial.trellis2_dinov3 import …`)
2. Wraps it with any LiTo-specific pre/post processing (image normalization stats, feature reshape, projection layer if upstream adds one on top of the encoder backbone)
3. Loads the LiTo-specific weights for projections, NOT the encoder backbone (which is shared and loaded separately)

### Case B: Upstream uses a novel encoder

Write `src/mlx_spatial/lito_condition.py` as a **full port**:

1. Read the upstream encoder class
2. Replicate in MLX float16 with the existing pattern shape from `hyworld2_vit.py`
3. Load weights via `lito_assets`

In both cases, write `tests/test_lito_condition.py`:

- `test_cond_matches_source_contract_input_0`, `_1`, `_2`
- `test_cond_output_shape` — matches DiT's expected `cond` input shape
- `test_cond_uses_float16`

## Parity Probe Specification

```python
import mlx.core as mx
import safetensors

fix_in = safetensors.load_file("tests/fixtures/lito/cond_input_0.safetensors")
fix_out = safetensors.load_file("tests/fixtures/lito/cond_output_0.safetensors")["cond_tokens"]

cond = LitoCondition.load(weights_root="weights/lito")
ours = cond(fix_in["straight_rgb"], fix_in["alpha"])

assert ours.shape == fix_out.shape
assert ours.dtype == mx.float16
mx.allclose(ours, fix_out, atol=<from gap-matrix>, rtol=<from gap-matrix>)
```

## Verification

```bash
uv run pytest tests/test_lito_condition.py -v
uv run pytest tests/test_lito_condition.py -m torch_parity  # optional CPU/MPS Torch parity only; no CUDA-only deps
```

## Slice-Specific Risks

- **Encoder identification surprise:** Slice 0 must name the exact encoder. If the audit is wrong (e.g., upstream uses a modified DINOv3 with extra projections), the adapter case may silently produce wrong features. The parity probe catches this.
- **Image preprocessing drift:** normalization stats, resize interpolation, and color-space conversions are easy to get wrong. Read the demo's preprocessing path end-to-end before writing the adapter; mirror the same `mx.array` dtype and channel order.

## Done When

All verification commands pass AND parity holds for all ≥ 3 fixtures AND the adapter clearly indicates (in module docstring) which case (A reuse vs. B port) was taken and why.
