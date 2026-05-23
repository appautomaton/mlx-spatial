# Slice 0B — Source Contracts + MLX-Compatible Fixtures + Tolerances (LITO-A)

This slice is the remaining serial gate before P1. It converts Slice 0A's upstream audit into source-contract metadata and deterministic local fixtures that downstream subagents can consume without a live vendor runtime.

## Inputs

- `SPEC.md`
- `PLAN.md`
- `spec/gap-matrix.md`
- `slices/00-vendor-assets-routing.md`
- `vendors/ml-lito/` (gitignored shallow clone; source reference only)
- `weights/lito-raw/` and `weights/lito-mlx/` (gitignored local weights)

## No-CUDA Contract

- CUDA is not allowed locally, in runtime dependencies, in dev dependencies, or in tests.
- Torch with CPU/MPS is allowed only for optional `torch_parity` probes when the relevant path runs without CUDA-only packages.
- Upstream CUDA/PyTorch/gsplat code is static source reference for the MLX inference implementation. It must not be used as an execution gate.
- Do not install or require xformers, flash-attention, CUDA-backed gsplat, or any CUDA wheel.
- Do not import vendor runtime modules from `scripts/lito/write_contract_fixtures.py` or `scripts/lito/validate_fixtures.py`.

## Source Contract

Record the source entry point for each module in `tests/fixtures/lito/manifest.json`:

| Fixture group | Upstream source reference | Local fixture role |
|---|---|---|
| `tokenizer_input_*.safetensors`, `tokenizer_output_*.safetensors` | `lito.trainers.lito_trainer.LightTokenizationTrainer.get_latents(...)` and `lito.models.spoint_encoder.SPointEncoder` | Fixed point-cloud/ray/rgb inputs plus 8192 x 32 latent-shape/range contract outputs |
| `cond_input_*.safetensors`, `cond_output_*.safetensors` | `LiToDiTTrainer.get_image_conditioning(straight_rgb, alpha)` and `lito.models.dino.SpatialDinov2` | Fixed RGBA inputs plus image-token feature shape/order contract outputs |
| `dit_input_*.safetensors`, `dit_step_*_*.safetensors` | upstream MLX DiT and `lito.odelibs.ode_solvers.odeint(...)` trajectory semantics | Fixed latent/condition inputs plus deterministic microtrajectory checkpoints |
| `render_input_*.safetensors`, `render_output_*.png` | `LightTokenizationTrainer.render_gaussians(...)` and `plibs.gs_utils.render_3dgs_gsplat(...)` | Fixed Gaussian/camera inputs plus local image/mask contract output |

The manifest must not label these as vendor numerical outputs. Use `backend: "source_contract_local"` unless a future non-CUDA oracle is actually executed.

## Fixture Strategy

Keep fixtures small but boundary-complete:

- tokenizer: 3 fixed point-cloud inputs and 3 latent contract outputs
- image conditioner: 3 fixed RGBA inputs and 3 feature contract outputs
- DiT: 1 fixed input x 1 fixed seed; write input, step 0, mid step, final step
- render: 1 fixed render input and 1 output image

Use deterministic NumPy/PIL/safetensors generation only. The fixture generator may encode simple local mathematical transforms so tests can catch axis, dtype, shape, seed, and normalization mistakes, but it must not pretend to be a vendor-output parity capture.

## Manifest

Write `tests/fixtures/lito/manifest.json` with one entry per fixture group:

```json
{
  "tokenizer": {
    "backend": "source_contract_local",
    "upstream_entry": "lito.trainers.lito_trainer.LightTokenizationTrainer.get_latents",
    "fixture_role": "shape_dtype_range_contract",
    "files": ["tokenizer_input_0.safetensors", "tokenizer_output_0.safetensors"],
    "dtype": "float32",
    "shape": {"latent_tokens": [1, 8192, 32]},
    "license": "synthetic local fixture; no Apple sample redistribution"
  }
}
```

The exact schema may add fields, but it must preserve backend, upstream entry, fixture role, files, dtype, shape, seed where applicable, and license/distribution status.

## Validation Helpers

`scripts/lito/write_contract_fixtures.py` generates fixtures and manifest.

`scripts/lito/validate_fixtures.py` checks:

- manifest exists and is valid JSON
- every manifest-listed file exists
- safetensors files can be opened and have at least one tensor
- render PNG files can be opened by PIL
- required groups are present: `tokenizer`, `condition`, `dit`, `render`

Both helpers must avoid vendor imports, MLX imports, and CUDA dependencies.

## Acceptance Criteria

- `slices/00b-reference-fixtures.md` records this no-CUDA contract and the upstream source entry points.
- `scripts/lito/write_contract_fixtures.py tests/fixtures/lito --overwrite` creates `tests/fixtures/lito/manifest.json` plus all required fixture files.
- `tests/fixtures/lito/manifest.json` records backend/source/function metadata for all required groups and marks local fixtures as source-contract fixtures.
- `scripts/lito/validate_fixtures.py tests/fixtures/lito` passes in the project environment.
- `spec/gap-matrix.md` records concrete source-contract tolerances and names optional non-CUDA parity probes separately.
- No scratch files leak into the repo working tree.

## Verification

```bash
uv run python scripts/lito/write_contract_fixtures.py tests/fixtures/lito --overwrite
test -f tests/fixtures/lito/manifest.json
find tests/fixtures/lito -maxdepth 1 -type f | grep -E "(tokenizer|dit|cond|render)_"
uv run python scripts/lito/validate_fixtures.py tests/fixtures/lito
grep -E "source-contract|MLX-compatible|no CUDA" .agent/work/2026-05-22-lito-mlx-inference-pipeline/spec/gap-matrix.md
find . -maxdepth 3 \( -name 'lito-fixtures-*' -o -name 'lito-fixture-env-*' \) -print -quit | grep . && echo "FAIL: scratch leak detected" || echo "OK: no scratch leaks"
```

## Historical Probe Evidence

Attempted 2026-05-23 in isolated vendor environment:

- Created `/tmp/lito-fixture-env` with CPython 3.11.15.
- Installed/imported enough of the upstream stack to prove the dependency shape: torch/torchvision/torchaudio, lightning, numpy/scipy, OpenCV, scikit-image, requests, timm, lpips, transformers, open3d, mlx, PyTorch3D, spz, gsplat, and TRELLIS support deps.
- Initialized missing gitignored upstream submodules:
  - `vendors/ml-lito/third_party/TRELLIS`
  - `vendors/ml-lito/third_party/TRELLIS/trellis/representations/mesh/flexicubes`
- `lito.trainers.lito_trainer`, `lito.trainers.lito_dit_trainer`, `plibs.gs_utils`, and `gsplat` import in that isolated environment.
- `torch.backends.mps.is_available()` returned `False`, `torch.cuda.is_available()` returned `False`, and importing MLX emitted no usable Metal device in this headless sandbox.
- Loading `weights/lito-raw/lito_new.ckpt` with unused voxel/mesh decoder construction disabled reaches the tokenizer model on CPU.
- Running `LightTokenizationTrainer.get_latents(...)` then falls into upstream localized attention. The no-CUDA path selects an xformers-backed implementation; installing xformers on macOS arm64 failed with `clang++: error: unsupported option '-fopenmp'`.

Conclusion: this evidence is useful for static source analysis and for knowing where CUDA-only assumptions live. It is not a reason to require CUDA. The active Slice 0B path is local source-contract fixture generation.
