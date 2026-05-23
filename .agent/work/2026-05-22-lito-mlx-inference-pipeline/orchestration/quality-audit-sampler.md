# quality-audit-sampler

## Status

DONE_WITH_CONCERNS

## Confirmed Matches

- Upstream `vendors/ml-lito/src/lito/trainers/lito_dit_trainer.py:760-929` and local `src/mlx_spatial/lito_real_backend.py:693-783` both sample `(B,8192,32)` normalized latent noise, integrate to `1.0`, then unnormalize with latent mean/std.
- Upstream recommended public settings are `heun`, `20` steps, `cfg_scale=3.0`, `mlx_compute_dtype="float16"` from `vendors/ml-lito/demos/lito/fastapi_lito_demo.py:326-345` and `:555-672`; local pipeline passes `heun`, caller `num_steps`, and caller `cfg_scale` through `src/mlx_spatial/lito_inference.py:240-249`.
- Timestep schedule matches for active methods: upstream `mx.linspace(self.t_eps, 1.0, ode_num_steps)` at `lito_dit_trainer.py:880-906`; local `mx.linspace(float(t_eps), 1.0, int(num_steps))` at `lito_real_backend.py:760`.
- Euler/Heun update equations match upstream MLX ODE solver `vendors/ml-lito/src/lito/mlx/odelibs/ode_solvers.py:49-136` and local loop `lito_real_backend.py:762-778`.
- CFG batching/math matches active dropout-enabled config: upstream `forward_with_cfg` duplicates first half and returns `uncond + scale * (cond - uncond)` at `vendors/ml-lito/src/lito/mlx/models/dit.py:369-400`; local duplicates latent/cond and applies same formula at `lito_real_backend.py:728-758`.
- Timestep embedding, condition embedding, position tokens, block execution, SwiGLU, and final AdaLN line up structurally: upstream `vendors/ml-lito/src/lito/mlx/models/dit.py:330-365`; local `lito_real_backend.py:672-690` and `:1651-1762`.
- Local real-weight inventory confirms expected DiT shape: tests assert `pos_mtx=(8192,1152)`, condition projection `(1152,2048)`, and timestep projection `(1152,64)` at `tests/test_lito_real_backend.py:866-885`.
- Baseline quality issue is real but not sampler-stat obvious: `orchestration/quality-baseline.md` records valid checkpoint-backed PLY schema/stats with user visual broken surfaces.

## Confirmed Mismatches

- Local backend sampler defaults `cfg_scale=1.0` in `DirectMlxLitoBackend.sample_dit_latents` / `sample_lito_dit_latents` (`src/mlx_spatial/lito_real_backend.py:188`, `:700`), while upstream `inference_sample_latent_mlx` defaults to `3.0` (`lito_dit_trainer.py:760-768`). The pipeline default still passes `3.0`, so this is not the observed baseline path unless another caller bypassed `LitoInferencePipeline`.
- Upstream supports `heun_<alpha>` nonuniform schedules (`lito_dit_trainer.py:885-899`); local supports only `"euler"` and `"heun"` (`lito_real_backend.py:762-780`). This is not active for the baseline command.
- Upstream MLX sampler has a CFG validity fallback for no-dropout checkpoints (`lito_dit_trainer.py:814-827`); local always enables CFG when `cfg_scale > 1.0` (`lito_real_backend.py:728-758`). Released config uses `cond_drop_prob: 0.1`, so this is a portability concern, not a current smoking gun.
- Local `mlx_compute_dtype` is configured but not honored inside the direct backend sampler; weights and math are loaded/run as float32. Upstream demo uses `float16` for MLX sampling. This is numerical/perf drift, unlikely by itself to create broken surfaces.

## Unknowns

- No golden upstream/local numeric trace was found for a full real DiT velocity step or full final sampled latent.
- No saved baseline latent statistics or intermediate velocity norms are referenced in `quality-baseline.md`, so sampler plausibility cannot be separated from decoder/init-coordinate drift by artifact inspection alone.
- Whether broken surfaces begin before or after `decode_sampled_latents_to_gaussians` remains unproven without recording sampled latent, sparse-structure/init coords, and final Gaussian decoder inputs.

## Fix Target

No high-confidence mismatch found in the active sampler. Highest-value sampler follow-up is instrumentation or golden comparison for one upstream MLX `run_lito_dit_velocity` step and final sampled latent before changing sampler logic.

## Verification

- Read-only inspections only; no CUDA, no dependency install, no upstream runtime execution, no source edits.
- Inspected `vendors/ml-lito`, `src/mlx_spatial/lito_dit.py`, `src/mlx_spatial/lito_real_backend.py`, `src/mlx_spatial/lito_inference.py`, `tests/test_lito_dit.py`, `tests/test_lito_real_backend.py`, and `orchestration/quality-baseline.md`.

## Concerns

- Sampler is structurally aligned but not proven numerically parity-correct.
- Current tests are too weak to catch full 28-block real-checkpoint sampler drift.
- Visual failure likely needs a decoder/init-coordinate fix first if one-step sampler parity later passes.
