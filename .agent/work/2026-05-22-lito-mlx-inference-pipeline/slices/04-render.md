# Slice 4 — LF-Conditioned 3DGS Render (LITO-E)

Parallel-safe with Slices 1, 2, 3. Consumes only source-contract fixtures from `tests/fixtures/lito/render_*.safetensors` / `render_*.png` (generated in Slice 0B).

**Default mode: adapter-only.** Wraps `src/mlx_spatial/gs_rasterize.py` without modifying it. If Slice 0's Risk F audit reported `adapter infeasible`, this slice does NOT start until the Risk F decision checkpoint resolves (see PLAN.md).

## Inputs

- `INTAKE.md`, `SPEC.md`, `PLAN.md`, this slice file
- `spec/gap-matrix.md` § LITO-E — final image-similarity threshold (closed in Slice 0)
- `slices/00-vendor-assets-routing.md` § Risk F Audit — confirms adapter feasibility OR records which sub-option was approved
- `tests/fixtures/lito/manifest.json` — source contract/function metadata for render fixtures
- `tests/fixtures/lito/render_input_*.safetensors`, `render_output_*.png` (or `.safetensors`, whichever Slice 0 chose)

## Upstream Files to Read First

(filled in by Slice 0; placeholders below)

1. `vendors/ml-lito/libraries/plibs/src/plibs/gs_utils.py::render_3dgs_gsplat` and `vendors/ml-lito/src/lito/trainers/lito_trainer.py::render_gaussians` — vendor gsplat/PyTorch render source reference
2. The Gaussian-parameters tensor schema expected by upstream (xyz, scale, rotation, opacity, color/SH coefficients, **plus LF conditioning tensor**)
3. The camera convention used (intrinsics format, extrinsics convention, projection)

## Reference Patterns in mlx-spatial

- `src/mlx_spatial/gs_rasterize.py` — the existing `GaussianSplatRenderer` (do not modify; wrap)
- `src/mlx_spatial/metal/` — the Metal compute kernel (do not modify in adapter-only mode)
- `src/mlx_spatial/hyworld2_sh.py` — spherical harmonics evaluation (degrees 0–4 land in Phase 2)
- `src/mlx_spatial/hyworld2_camera.py` — camera conversions
- `src/mlx_spatial/hyworld2_export.py` — example of using `rasterize_gaussians` and emitting rendered output

## Implementation Outline

### Adapter-only (default)

Write `src/mlx_spatial/lito_render.py`:

1. `class LitoRenderer:` constructor takes weights root and optional LF-conditioning parameters
2. `def __call__(self, gaussians: dict, camera: dict, lf_condition: mx.array | None = None) -> mx.array:`
   - Apply LF conditioning to Gaussian parameters BEFORE handing off to `gs_rasterize.GaussianSplatRenderer`. The LF layer is the only new GS-side surface.
   - Call `gs_rasterize.GaussianSplatRenderer(...)` with the (possibly LF-modulated) Gaussian params
   - Return the rasterized image
3. Document in the module docstring: "Adapter-only — does not modify `gs_rasterize.py`. LF conditioning is applied as a pre-rasterization layer."

### If Risk F option B was chosen (shared-base refactor)

Extract a shared base inside `src/mlx_spatial/gs_rasterize.py` (e.g., `_GaussianSplatRendererBase`) that both HY-World 2.0 and LiTo depend on. Move HY-World-specific assumptions into a `HyWorld2GaussianSplatRenderer` subclass. Write `LitoGaussianSplatRenderer` subclass. The refactor must preserve all HY-World 2.0 behavior (verified by green regression sweep).

In either mode, write `tests/test_lito_render.py`:

- `test_render_matches_source_contract_input_0` — load Gaussian params + camera fixture, render, compare to local contract render via image-similarity threshold
- `test_render_uses_float16_inputs`
- `test_lito_render_does_not_modify_gs_rasterize` (adapter-only mode) — a meta-test that asserts no behavioral change to `gs_rasterize.py` exports

## Performance Instrumentation

Render is the final stage before export and a likely time hotspot. Instrument wall time and peak active memory for the render call. Mirror the DiT slice's instrumentation pattern (eval barriers, `mx.metal.get_peak_memory`). Record into the test fixture so Slice 5's aggregate `LitoGenerationResult.metrics["render"]` can compare to the per-slice baseline.

```python
import time, mlx.core as mx

mx.metal.reset_peak_memory()
t0 = time.perf_counter()
img = renderer(gaussians=vendor_in, camera={"intrinsics": K, "extrinsics": E}, lf_condition=lf)
mx.eval(img)
wall = time.perf_counter() - t0
peak_gb = mx.metal.get_peak_memory() / (1024 ** 3)
```

Render generally fits comfortably under the 90 GB soft threshold; if it does not, the rasterizer is allocating more memory than the existing HY-World render does — investigate before weakening the threshold.

## Parity Probe Specification

```python
import mlx.core as mx
import safetensors
from PIL import Image
import numpy as np

fix_in = safetensors.load_file("tests/fixtures/lito/render_input_0.safetensors")
fix_out = np.array(Image.open("tests/fixtures/lito/render_output_0.png"))  # or load from .safetensors

renderer = LitoRenderer.load(weights_root="weights/lito")
ours = renderer(
    gaussians=fix_in,  # xyz, scale, rotation, opacity, SH, lf_condition
    camera={"intrinsics": fix_in["K"], "extrinsics": fix_in["E"]},
    lf_condition=fix_in["lf"],
)
ours_np = (np.array(ours) * 255).clip(0, 255).astype("uint8")

psnr = 10 * np.log10(255**2 / np.mean((ours_np.astype(float) - fix_out.astype(float)) ** 2))
assert psnr >= <from gap-matrix>  # e.g., >= 30 dB
```

## Verification

```bash
uv run pytest tests/test_lito_render.py -v
uv run pytest tests/test_hyworld2_*.py  # HY-World regression sweep — mandatory regardless of Risk F mode
uv run pytest tests/test_gs_rasterize*.py  # if any
```

## Slice-Specific Risks

- **LF-conditioning math:** "Light field" conditioning could mean several things — per-Gaussian view-dependent color modulation, per-Gaussian attenuation, or a separately rendered LF that composites with the GS output. Read upstream carefully; do not assume.
- **Camera convention mismatch:** `gs_rasterize.py` may assume OpenCV (right-down-forward) while LiTo's upstream may use OpenGL (right-up-back) or another. The adapter must convert; the conversion is a known correctness pitfall.
- **Image-similarity threshold calibration:** PSNR is a coarse metric. If vendor's rasterizer uses different tile sizes or floating-point accumulation order, pixel-level diffs are expected. Threshold is per-Slice-0 measurement; do not weaken it without recording why.
- **HY-World regression (option B only):** if the shared-base refactor changes any HY-World code path, even non-behaviorally, the HY-World regression sweep is the gate. Run it both before and after the refactor.
- **CUDA-only upstream render:** vendor gsplat may rely on CUDA-specific code. Treat it as static source reference and port the tensor schema/math into the existing MLX/Metal rasterizer path.

## Done When

All verification commands pass AND image-similarity threshold from `spec/gap-matrix.md` holds AND HY-World regression sweep is green AND `src/mlx_spatial/gs_rasterize.py` is unchanged (adapter-only mode) or its public surface preserves all HY-World call sites (option B mode).
