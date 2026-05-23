# Slice 2 — Flow-Matching DiT (LITO-D)

Parallel-safe with Slices 1, 3, 4. Consumes only source-contract fixtures from `tests/fixtures/lito/dit_*.safetensors` (generated in Slice 0B).

## Inputs

- `INTAKE.md`, `SPEC.md`, `PLAN.md`, this slice file
- `spec/gap-matrix.md` § LITO-D — final per-step drift budget (closed in Slice 0)
- `slices/00-vendor-assets-routing.md` § Upstream Coverage Audit — records DiT MLX coverage and the PyTorch conditioning dependency
- `tests/fixtures/lito/manifest.json` — source contract/function metadata for DiT fixtures
- `tests/fixtures/lito/dit_input_*.safetensors`, `dit_step_0_*.safetensors`, `dit_step_mid_*.safetensors`, `dit_step_final_*.safetensors`

## Upstream Files to Read First

(filled in by Slice 0; placeholders below)

1. `vendors/ml-lito/src/lito/mlx/models/dit.py` — DiT class definition
2. `vendors/ml-lito/src/lito/odelibs/ode_solvers.py` — rectified flow integrator / sampler source contract
3. `vendors/ml-lito/src/lito/trainers/lito_dit_trainer.py` — inference sampling wrapper and hyperparameter plumbing
4. The image-conditioning entry point (this slice does NOT implement LITO-B — it consumes a vendor-reference conditioned feature as fixture input)

## Reference Patterns in mlx-spatial

- `src/mlx_spatial/hyworld2_transformer.py` — block patterns, attention with QK-norm, RoPE
- `src/mlx_spatial/trellis2_forward.py` and `src/mlx_spatial/trellis2_inference.py` — flow patterns (rectified flow Euler integrator)
- `src/mlx_spatial/hyworld2_inference.py` — `HYWORLD2_MEMORY_PROFILES` shape — clone it as `LITO_MEMORY_PROFILES` in this slice

## Implementation Outline

1. Read upstream DiT class. Map each block to `hyworld2_transformer.py` patterns where shapes match; replicate verbatim where novel.
2. Write `src/mlx_spatial/lito_dit.py`:
   - `LitoDiTConfig` dataclass
   - `class LitoDiT(mx.nn.Module):`
   - `def __call__(self, latent: mx.array, cond: mx.array, t: mx.array) -> mx.array:` — one denoising step
   - `def sample(self, cond: mx.array, num_steps: int, seed: int) -> mx.array:` — full rectified-flow integration; reuse `trellis2_forward.py` flow integrator if shapes align
3. Define `LITO_MEMORY_PROFILES` (e.g., `("default", "low")`) and select layer-streaming or chunked-attention behavior accordingly. Mirror `HYWORLD2_MEMORY_PROFILES` shape.
4. Default dtype `float16`; `float32` annotated inline only where stability demands (likely: RoPE phase accumulation, softmax denominator).
5. Write `tests/test_lito_dit.py`:
  - `test_dit_step_0_matches_source_contract` — single forward step at `t=0`, fixed seed -> compare to `dit_step_0_*.safetensors`
  - `test_dit_step_mid_matches_source_contract` — at mid-step
  - `test_dit_step_final_matches_source_contract` — at final step
  - `test_dit_full_trajectory_matches_source_contract` — full `sample(...)` matches `dit_step_final_*.safetensors`
   - `test_dit_memory_profile_default_fits_m4max` — sanity bound on resident memory (use `mlx_memory.py` utilities if helpful)

## Memory Instrumentation (mandatory)

DiT is the largest single consumer in the pipeline. This slice MUST measure and bound peak active memory per profile.

`LITO_MEMORY_PROFILES = ("safe", "balanced", "large")` with default `"balanced"`. Each profile's `sample(...)` runs at upstream-recommended `num_steps` (from Slice 0's `## Recommended Settings`) and stays under **90 GB** active memory on the 128 GB dev system.

**Slice ownership note (resolves cross-slice hazard):** Slice 2 uses plain `assert` statements for its memory tests. It does **not** import `LitoMemoryLimitExceeded` — that exception class is owned by Slice 5 (`lito_inference.py`), where the orchestration-layer monitor lives. Slice 2 runs in parallel with Slices 1, 3, 4 and must not depend on Slice 5 artifacts.

Use MLX-native APIs with eval boundaries:

```python
import mlx.core as mx

mx.metal.reset_peak_memory()
out = dit.sample(cond, num_steps=LITO_RECOMMENDED_NUM_STEPS, seed=42)
mx.eval(out)                      # force computation
peak_bytes = mx.metal.get_peak_memory()
peak_gb = peak_bytes / (1024 ** 3)
assert peak_gb < 90.0, f"balanced profile peaked at {peak_gb:.1f} GB; soft threshold is 90 GB"
```

Add `test_dit_memory_balanced_stays_under_90gb`, `test_dit_memory_safe_stays_well_under_threshold`, and `test_dit_memory_large_stays_under_90gb` to `tests/test_lito_dit.py`. The `safe` profile should fit comfortably; `large` may approach 90 GB but never cross it in this slice's tests. (The 100 GB hard-ceiling raise behavior is Slice 5's responsibility, tested in `tests/test_lito_memory_limits.py` where `LitoMemoryLimitExceeded` is defined and importable.)

If `balanced` exceeds 90 GB at upstream-recommended `num_steps`, **the slice does not weaken the threshold.** Options in priority order:
1. Tile / stream layers (mirror `HYWORLD2_MEMORY_PROFILES` streaming patterns)
2. Move large intermediate buffers to `float16` where they are not stability-critical (and annotate the few `float32` exceptions inline)
3. Reduce attention chunk size in the default profile
4. If none of the above suffice, surface as a slice-specific risk — but do not silently raise the threshold; that is an SPEC change, not a slice decision

## Parity Probe Specification

For each fixture set (one fixed input x one fixed seed, >= 1 set):

```python
import mlx.core as mx
import safetensors

fix_in = safetensors.load_file("tests/fixtures/lito/dit_input_0.safetensors")
step_0 = safetensors.load_file("tests/fixtures/lito/dit_step_0_0.safetensors")["latent"]
step_final = safetensors.load_file("tests/fixtures/lito/dit_step_final_0.safetensors")["latent"]

dit = LitoDiT.load(weights_root="weights/lito")
ours_step_0 = dit(fix_in["latent"], fix_in["cond"], mx.array(0.0))
ours_final = dit.sample(fix_in["cond"], num_steps=fix_in["num_steps"], seed=int(fix_in["seed"]))

assert ours_step_0.dtype == mx.float16
mx.allclose(ours_step_0, step_0, atol=<from gap-matrix>, rtol=<from gap-matrix>)
mx.allclose(ours_final, step_final, atol=<from gap-matrix-final>, rtol=<from gap-matrix-final>)
```

Drift budget per step is recorded in `spec/gap-matrix.md` after Slice 0 measures vendor numerics. Expect monotonic growth with step index.

## Verification

```bash
uv run pytest tests/test_lito_dit.py -v
uv run pytest tests/test_lito_dit.py -m torch_parity  # optional CPU/MPS Torch parity only; no CUDA-only deps
```

## Slice-Specific Risks

- **Rectified flow integrator subtleties:** Euler steps with rectified-flow targets are simple in principle but the schedule (timestep spacing, conditioning injection mechanism) can differ between implementations. Match upstream exactly; do not "optimize" the schedule.
- **Memory pressure:** DiT inference at full image-conditioned latent grid on M4 Max may not fit at float16 without streaming. If `test_dit_memory_profile_default_fits_m4max` fails, add a `low` profile that tiles attention or streams layers; do not silently drop precision.
- **Numerical drift across many steps:** if a 50-step sample diverges from vendor by step 30, the issue is usually an accumulated `float16` underflow at softmax — try `float32` softmax just at the denominator and annotate.

## Done When

All verification commands pass AND per-step drift is within budget AND `LITO_MEMORY_PROFILES` default fits M4 Max unified memory.
