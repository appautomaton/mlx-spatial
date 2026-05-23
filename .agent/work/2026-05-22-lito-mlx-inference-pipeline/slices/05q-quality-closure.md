# Slice 5Q Quality Closure Detail

## Trigger

User inspection rejected the current checkpoint-backed LiTo PLY quality: the file is schema-valid, but surfaces are broken. AC-07 stays open until real-object outputs are visually credible. `inputs/lito/smoke.png` is a color blob and cannot be used as qualitative evidence.

## Hard Constraints

- No CUDA execution and no CUDA dependencies.
- No runtime imports from `torch`, `vendors/ml-lito`, `xformers`, `flash_attn`, CUDA-backed `gsplat`, or vendor LiTo modules.
- Upstream PyTorch/CUDA files are static reference only.
- Optional Torch probes, if used at all, must be CPU/MPS-only, external to runtime dependencies, and non-blocking if unavailable.
- Work only on inference. Do not add training, new weight formats, or broad refactors.

## Quality Oracle

Command checks must prove:

- PLY is checkpoint-backed, not source-contract smoke.
- Gaussian fields are finite.
- Bbox, opacity, scaling, quaternion norms, and SH/color fields are not obviously degenerate.
- Regeneration is reproducible for fixed `--seed`.

Human inspection must prove:

- The generated object has coherent surfaces for at least `inputs/trellis2/teacup.png`.
- Broken, exploded, hollow, or blob-like surfaces are not accepted as AC-07 completion.

## Parallel Audit Group

Run these in parallel after Slice 5Q-0. They are read-only except for their own orchestration notes.

### Audit A: Conditioning

Inspect upstream and local preprocessing/image-conditioning contracts:

- Upstream: LiTo demo/preprocess path, `dino.py`, `lito_dit_trainer.get_image_conditioning`, patch encoder/RGBA branch.
- Local: `src/mlx_spatial/lito_condition.py`, `src/mlx_spatial/lito_real_backend.py`, `src/mlx_spatial/lito_inference.py`.
- Output: `orchestration/quality-audit-conditioning.md`.

Required note sections: `Confirmed Matches`, `Confirmed Mismatches`, `Unknowns`, `Fix Target`.

### Audit B: Sampler

Inspect upstream and local latent/sampler contracts:

- Upstream: `inference_sample_latent_mlx`, `forward_with_cfg`, DiT/rectified-flow sampler code, timestep schedule, guidance scale, latent initialization.
- Local: `src/mlx_spatial/lito_dit.py`, `src/mlx_spatial/lito_real_backend.py`.
- Output: `orchestration/quality-audit-sampler.md`.

Required note sections: `Confirmed Matches`, `Confirmed Mismatches`, `Unknowns`, `Fix Target`.

### Audit C: Decode

Inspect upstream and local voxel/init-coordinate/Gaussian decode contracts:

- Upstream: `lito_trainer.inference_init_coords_for_decoder`, `inference_estimate_gaussians_mlx`, `GaussianDecoderXv`, localized-voxel attention, `decode_gs`, export/render field ordering.
- Local: `src/mlx_spatial/lito_real_backend.py`, `src/mlx_spatial/lito_render.py`, any PLY writer/export helpers.
- Output: `orchestration/quality-audit-decode.md`.

Required note sections: `Confirmed Matches`, `Confirmed Mismatches`, `Unknowns`, `Fix Target`.

## Implementation Routing

- The coordinator chooses one Slice 5Q-2 fix target after reading all audit notes and the baseline inspector stats.
- Implement one mismatch class per pass. Do not patch conditioning, sampler, and decode in one change unless a single test proves they are one coupled bug.
- Do not run two implementers concurrently against `src/mlx_spatial/lito_real_backend.py`; that file is a shared write target.
- After each implementation pass, run the quality inspector and record whether the failure moved, narrowed, or resolved.
- If two targeted fix passes do not improve the quality stats or visual output, return to `auto-plan` with the audit notes and baseline/fix JSON as evidence.

## Final Gate

Slice 5Q-3 ends at `human-verify` because visual surface credibility is not fully command-verifiable. After human acceptance, Slice 5Q-4 updates artifacts and routes to `auto-verify`.
