# Slice 1 — Tokenizer (LITO-C)

Parallel-safe with Slices 2, 3, 4. Consumes only source-contract fixtures from `tests/fixtures/lito/tokenizer_*.safetensors` (generated in Slice 0B).

## Inputs

- `INTAKE.md`, `SPEC.md`, `PLAN.md`, this slice file
- `spec/gap-matrix.md` § LITO-C — final per-tensor tolerances (closed in Slice 0)
- `slices/00-vendor-assets-routing.md` § Upstream Coverage Audit — records that tokenizer source reference is PyTorch because upstream has no full MLX tokenizer encoder
- `tests/fixtures/lito/manifest.json` — source contract/function metadata for tokenizer fixtures
- `tests/fixtures/lito/tokenizer_input_*.safetensors`, `tokenizer_output_*.safetensors` — ≥ 3 fixed inputs

## Upstream Files to Read First

(filled in by Slice 0; placeholders below)

1. `vendors/ml-lito/src/lito/trainers/lito_trainer.py::LightTokenizationTrainer.get_latents` — module-boundary source contract
2. `vendors/ml-lito/src/lito/models/spoint_encoder.py` — tokenizer encoder architecture and localized attention source reference
3. Any utility modules the tokenizer imports (positional encoding, attention variants, MLPs)

## Reference Patterns in mlx-spatial

- `src/mlx_spatial/hyworld2_layers.py` — generic MLX attention, MLP, norm patterns
- `src/mlx_spatial/hyworld2_transformer.py` — block patterns
- `src/mlx_spatial/sam3d_slat.py` and `src/mlx_spatial/trellis2_slat.py` — neighbors in shape (latent producers, not tokenizers, but similar dtype discipline)

The tokenizer architecture is **novel** — distinct from O-Voxel and ShapeVAE. Do not pattern-match it to either; read upstream first.

## Implementation Outline

1. Read upstream tokenizer class and hyperparameters
2. Write `src/mlx_spatial/lito_tokenizer.py` with:
   - `LitoTokenizerConfig` dataclass (matching upstream hyperparameters)
   - `class LitoTokenizer(mx.nn.Module):` with `__init__(self, config: LitoTokenizerConfig)` and `def __call__(self, point_cloud: mx.array) -> mx.array:` returning the 8192 × 32 latent
   - Weights loaded via `lito_assets` from `weights/lito[-mlx]/`
3. Default tensor dtype: MLX `float16`. Justify any `float32` use inline (one short comment per occurrence — e.g., `# float32 for softmax stability`).
4. Write `tests/test_lito_tokenizer.py`:
   - `test_tokenizer_matches_source_contract_input_0`, `_input_1`, `_input_2` — load fixture pair, run our tokenizer, compare via `mlx_spatial.lito_parity.compare_tensors` (build this helper if it doesn't exist yet; mirror `hyworld2_parity.compare_hyworld2_parity_tensors`)
   - `test_tokenizer_output_shape` — `(8192, 32)`
   - `test_tokenizer_uses_float16` — `result.dtype is mx.float16`

## Parity Probe Specification

For each fixture pair `(tokenizer_input_<n>.safetensors, tokenizer_output_<n>.safetensors)`, compare against the source-contract output recorded in the fixture manifest. Do not treat these files as captured vendor numerical outputs unless the manifest backend says a non-CUDA oracle was actually executed.

```python
import mlx.core as mx
import safetensors

fix_in = safetensors.load_file("tests/fixtures/lito/tokenizer_input_0.safetensors")
fix_out = safetensors.load_file("tests/fixtures/lito/tokenizer_output_0.safetensors")["latent_tokens"]

tokenizer = LitoTokenizer.load(weights_root="weights/lito")  # or weights/lito-mlx
ours = tokenizer(fix_in["xyz_w"], fix_in["rgb"], fix_in["ray_origin_direction_w"])

assert ours.shape == fix_out.shape
assert ours.dtype == mx.float16
mx.allclose(ours, fix_out, atol=<from gap-matrix>, rtol=<from gap-matrix>)
```

## Verification

```bash
uv run pytest tests/test_lito_tokenizer.py -v
uv run pytest tests/test_lito_tokenizer.py -m torch_parity  # optional CPU/MPS Torch parity only; no CUDA-only deps
```

## Slice-Specific Risks

- **Novel architecture surprises:** the 8192 × 32 latent shape suggests learned point-cloud tokenization. The exact tokenization scheme (cross-attention from learned queries? FPS + clustering? Voxel quantization?) is not derivable from the latent shape alone — read upstream carefully.
- **Hyperparameter drift:** if upstream `LitoTokenizerConfig` has dozens of fields, replicate verbatim; do not "tidy up" defaults that look unused, since untested params may turn on a code path the vendor takes.
- **CUDA-only upstream ops:** localized attention may be xformers/flash-attention-backed upstream. Port those operations to MLX-compatible primitives; do not require xformers, flash-attention, or CUDA.

## Done When

All verification commands pass AND parity tolerance from `spec/gap-matrix.md` holds for all ≥ 3 fixtures AND no `float32` line in `lito_tokenizer.py` is missing its justification comment.

## Hand-Off to Next Stage

Slice 5 (integration) replaces fixture-loading with the actual upstream module output. This slice writes the module; Slice 5 wires it up.
