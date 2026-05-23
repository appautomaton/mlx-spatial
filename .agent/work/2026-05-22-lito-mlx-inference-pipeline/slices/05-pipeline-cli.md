# Slice 5 — Pipeline + CLI + End-to-End Smoke (LITO-F integration)

Depends on Slices 1, 2, 3, 4. Parallel-safe with Slice 6.

This slice wires the completed MLX source-contract modules into a runnable local pipeline: image -> conditioner -> tokenizer -> DiT -> render -> export. It does not import vendor runtime code and it does not require CUDA. The output is a source-contract 3DGS smoke artifact, not a claim of full vendor numerical parity. After the 2026-05-23 guard correction, the default `generate` path is reserved for checkpoint-backed LiTo output and must fail closed until that real backend lands; source-contract smoke generation requires `--source-contract-smoke`.

## Inputs

- `INTAKE.md`, `SPEC.md`, `PLAN.md`, this slice file
- All modules from Slices 1–4 (`lito_tokenizer.py`, `lito_dit.py`, `lito_condition.py`, `lito_render.py`)
- `lito_assets.py` from Slice 0
- A small synthetic or user-provided sample input under `inputs/lito/` or a test-created temporary image. Do not commit Apple generated samples because Slice 0 recorded CC BY-NC-ND constraints.

## Reference Patterns in mlx-spatial

- `src/mlx_spatial/hyworld2.py` — CLI shape with `validate`, `inspect`, `download-command`, `reconstruct` subcommands
- `src/mlx_spatial/hyworld2_inference.py` — orchestration with `HYWORLD2_MEMORY_PROFILES`, staged execution, dataclass output
- `src/mlx_spatial/sam3d.py` and `src/mlx_spatial/trellis2.py` — alternative CLI shapes for cross-reference

## Implementation Outline

### `src/mlx_spatial/lito_inference.py`

Module-level constants (from Slice 0's `## Recommended Settings`):

```python
# LITO_RECOMMENDED_* — upstream-verified defaults. Each constant names its source.
LITO_RECOMMENDED_NUM_STEPS: int = ...           # source: vendors/ml-lito/<file>:<lines> or <URL>
LITO_RECOMMENDED_SEED_POLICY: str = ...         # source: ...
LITO_RECOMMENDED_RESOLUTION: tuple[int, int] = ...  # source: ...
LITO_RECOMMENDED_CFG_SCALE: float = ...         # source: ...  (or rectified-flow equivalent)
# ... any other knob Slice 0 captured
```

Memory monitor:

```python
class LitoMemoryLimitExceeded(RuntimeError):
    """Raised when peak active memory crosses the 100 GB hard ceiling."""

SOFT_THRESHOLD_GB = 90.0
HARD_CEILING_GB = 100.0

def _check_memory(stage: str) -> float:
    peak_gb = mx.metal.get_peak_memory() / (1024 ** 3)
    if peak_gb >= HARD_CEILING_GB:
        raise LitoMemoryLimitExceeded(f"stage {stage}: peak active memory {peak_gb:.1f} GB exceeded {HARD_CEILING_GB} GB ceiling")
    if peak_gb >= SOFT_THRESHOLD_GB:
        logger.warning(f"stage {stage}: peak active memory {peak_gb:.1f} GB crossed {SOFT_THRESHOLD_GB} GB soft threshold")
    return peak_gb
```

Result dataclass:

```python
@dataclass(frozen=True)
class LitoGenerationResult:
    gaussians: mx.array | dict[str, mx.array]
    rendered_image: mx.array | None
    metadata: dict[str, Any]
    metrics: dict[str, dict[str, float]]  # stage → {wall_time_s, peak_active_memory_gb, peak_cache_memory_gb}
```

`class LitoInferencePipeline:` orchestrates the full pipeline:

1. `__init__(self, weights_root: str | Path, memory_profile: str = "balanced")` — `memory_profile` choices: `("safe", "balanced", "large")` matching SAM 3D convention
2. `def generate(self, image_path: str | Path, num_steps: int = LITO_RECOMMENDED_NUM_STEPS, seed: int | None = None, cfg_scale: float = LITO_RECOMMENDED_CFG_SCALE, ...) -> LitoGenerationResult:`
   For each stage (`preprocess`, `condition`, `tokenize`, `dit`, `decode`, `render`, `export`):
   - `mx.metal.reset_peak_memory()`
   - `t0 = time.perf_counter()`
   - run the stage
   - `mx.eval(stage_output)` (force computation before measurement)
   - record `wall_time_s = time.perf_counter() - t0` and `peak_active_memory_gb = _check_memory(stage)` and `peak_cache_memory_gb = mx.metal.get_cache_memory() / (1024 ** 3)` (if available in MLX version)
   - accumulate into `metrics[stage]`
   - if `_check_memory` raised, the exception propagates and the pipeline aborts cleanly

3. `LITO_MEMORY_PROFILES: dict[str, ...]` — parallel to `HYWORLD2_MEMORY_PROFILES`. Choices: `("safe", "balanced", "large")` matching SAM 3D's `scripts/sam3d/reconstruct.py` convention. Default `"balanced"`. Each profile tested against the 90 GB soft threshold in Slice 2's DiT test.

### `src/mlx_spatial/lito.py`

CLI entry point — mirror `hyworld2.py`:

```python
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apple LiTo MLX inference")
    parser.add_argument("--root", default=LITO_DEFAULT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate, inspect, download-command — mirror hyworld2.py

    generate = subparsers.add_parser("generate")
    generate.add_argument("image_path")
    generate.add_argument("--output", required=True)
    generate.add_argument("--format", choices=["ply", "splat", "safetensors"], default="ply")
    generate.add_argument("--memory-profile", choices=("safe", "balanced", "large"), default="balanced")
    generate.add_argument("--num-steps", type=int, default=LITO_RECOMMENDED_NUM_STEPS)
    generate.add_argument("--cfg-scale", type=float, default=LITO_RECOMMENDED_CFG_SCALE)
    generate.add_argument("--seed", type=int, default=None)
    generate.add_argument("--print-metrics", action="store_true", help="print per-stage wall time and peak memory")
    # ...
```

After the pipeline runs, if `--print-metrics` is set, log the `LitoGenerationResult.metrics` dict in a readable table (one row per stage).

### `scripts/lito/generate.py` (mandatory deliverable)

Mirror `scripts/sam3d/reconstruct.py`. The script:

1. Sets `sys.path` to include the in-repo `src/` so it works without `uv pip install -e .`
2. Defines argparse with the upstream-recommended defaults (same as the CLI's `generate` subcommand)
3. Delegates to `mlx_spatial.lito.main` with normalized args

```python
#!/usr/bin/env python3
"""Run Apple LiTo image-to-3DGS generation with mlx-spatial recommended defaults.

Example:
    python scripts/lito/generate.py inputs/lito/sample.png \\
      --output outputs/lito/sample.ply

Defaults are taken from mlx_spatial.lito_inference.LITO_RECOMMENDED_*, which
in turn trace to upstream apple/ml-lito (see slices/00-vendor-assets-routing.md
§ Recommended Settings).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))


def main(argv: list[str] | None = None) -> int:
    from mlx_spatial.lito_inference import (
        LITO_RECOMMENDED_NUM_STEPS,
        LITO_RECOMMENDED_CFG_SCALE,
    )

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="input image")
    parser.add_argument("--root", default="weights/lito", help="weights root (lito or lito-mlx)")
    parser.add_argument("--output", type=Path, required=True, help="output 3DGS file")
    parser.add_argument("--format", choices=("ply", "splat", "safetensors"), default="ply")
    parser.add_argument("--memory-profile", choices=("safe", "balanced", "large"), default="balanced")
    parser.add_argument("--num-steps", type=int, default=LITO_RECOMMENDED_NUM_STEPS)
    parser.add_argument("--cfg-scale", type=float, default=LITO_RECOMMENDED_CFG_SCALE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-metrics", action="store_true")
    args = parser.parse_args(argv)

    from mlx_spatial.lito import main as lito_main
    return lito_main([
        "generate",
        str(args.image),
        "--root", str(args.root),
        "--output", str(args.output),
        "--format", args.format,
        "--memory-profile", args.memory_profile,
        "--num-steps", str(args.num_steps),
        "--cfg-scale", str(args.cfg_scale),
        "--seed", str(args.seed),
        *(["--print-metrics"] if args.print_metrics else []),
    ])


if __name__ == "__main__":
    sys.exit(main())
```

Add an entry to `scripts/README.md` (one line under a `## lito/` section, mirroring the existing per-pipeline sections).

Output writers per format:

- `.ply` — gsplat-standard. If runtime code imports `plyfile`, add `plyfile` to runtime `dependencies`; otherwise hand-write the PLY header (mirror `hyworld2_export.py`) and keep `plyfile` dev-only for validation.
- `.splat` — PlayCanvas/web format. Simple binary; may alias to the PLY smoke writer with a clear metadata note if no upstream tool needs it in this slice.
- `.safetensors` — param dump; trivial via `safetensors`

### `pyproject.toml` updates

In `[project.scripts]`, append exactly one line:

```
mlx-spatial-lito = "mlx_spatial.lito:main"
```

In `[dependency-groups.dev]`, add `torch` only if optional CPU/MPS `torch_parity` tests need it. Do not add CUDA extras, xformers, flash-attention, or CUDA-backed gsplat. Add `plyfile` for PLY validation in tests. Add `plyfile` to runtime `dependencies` only if the runtime writer imports it; if the writer hand-writes PLY, do not add a runtime dependency. `uv lock` updates `uv.lock` naturally.

After both edits, run `uv build` to confirm a clean wheel and sdist still produce; this is the AC-08 safety check that replaced the strict release-boundary anti-goal.

### Sample Input Acquisition

Slice 0's license decision drives this:

- Slice 0 recorded that modified Apple generated samples should not be redistributed. Tests should create synthetic images under `/tmp` or use an untracked local `inputs/lito/` image if present. Do not add committed Apple sample images.

### Tests

`tests/test_lito_inference.py`:
- `test_full_pipeline_runs_on_sample_input` — end-to-end source-contract pipeline without crashing
- `test_lito_inference_imports_without_vendors` — temporarily moves `vendors/ml-lito` aside and re-imports
- `test_metrics_dict_has_all_stages` — asserts `metrics` contains keys `preprocess`, `condition`, `tokenize`, `dit`, `decode`, `render`, `export`; each entry has `wall_time_s` and `peak_active_memory_gb` keys
- `test_recommended_constants_have_upstream_source_comments` — meta-test that greps `lito_inference.py` for `LITO_RECOMMENDED_*` and asserts each has a comment naming the upstream source

`tests/test_lito_memory_limits.py` (new, mandatory):
- `test_soft_threshold_warning_at_90gb` — uses a synthetic large input or mocked `mx.metal.get_peak_memory` to assert the warning path fires
- `test_hard_ceiling_raises_at_100gb` — uses a synthetic input or mocked `mx.metal.get_peak_memory` to assert `LitoMemoryLimitExceeded` raises and the pipeline aborts cleanly

`tests/test_lito_cli.py`:
- `test_cli_validate_returns_zero_on_valid_weights`
- `test_cli_generate_fails_closed_without_smoke_flag` — default `generate` with missing weights exits non-zero and writes no output.
- `test_cli_generate_rejects_placeholder_weights_without_smoke_flag` — default `generate` with placeholder safetensors exits non-zero and writes no output.
- `test_cli_generate_produces_ply` — runs `mlx-spatial-lito generate /tmp/<sample>.png --output /tmp/test.ply --format ply --source-contract-smoke` and asserts the file has a valid deterministic PLY header/body. **Write to `/tmp/`, not the repo working tree.**
- `test_cli_generate_format_safetensors`
- (optional) `test_cli_generate_format_splat`
- `test_cli_generate_print_metrics_logs_all_stages`
- `test_scripts_lito_generate_wrapper_works` — invokes `python scripts/lito/generate.py inputs/lito/<sample>.png --output /tmp/test_wrapper.ply` and asserts exit 0 + valid PLY

## Verification

```bash
# Module-level
uv run pytest tests/test_lito_inference.py tests/test_lito_cli.py tests/test_lito_memory_limits.py -v

# CLI exposed?
uv run mlx-spatial-lito --help

# End-to-end smoke via the library entry point
uv run mlx-spatial-lito validate weights/lito 2>/dev/null || uv run mlx-spatial-lito validate weights/lito-mlx
uv run mlx-spatial-lito generate inputs/lito/<sample>.png --output outputs/lito/<sample>.ply --format ply --print-metrics --source-contract-smoke
python -c "from pathlib import Path; data=Path('outputs/lito/<sample>.ply').read_text().splitlines(); print(data[0], data[2])"

# End-to-end smoke via the standalone wrapper script
uv run python scripts/lito/generate.py inputs/lito/<sample>.png --output /tmp/lito-smoke-wrapper.ply --source-contract-smoke

# Vendors-absent import
mv vendors/ml-lito /tmp/_lito_stash && uv run python -c "from mlx_spatial.lito import LitoInferencePipeline; print('OK')" && mv /tmp/_lito_stash vendors/ml-lito

# Wheel-build safety (AC-08 replacement for the old release-boundary check)
uv build

# Full regression sweep — no existing pipeline regresses
uv run pytest

# AC-15 hygiene check
git status --porcelain | grep -vE "^.. (src/mlx_spatial/lito|tests/test_lito|scripts/lito|scripts/README|docs/lito|pyproject\\.toml|uv\\.lock|inputs/lito)" && echo "FAIL: unexpected files" || echo "OK"
```

## Slice-Specific Risks

- **Latent → Gaussian decode boundary:** the bridge between DiT output (latent) and renderer input (explicit Gaussian params with xyz, scale, rotation, opacity, color/SH, LF) is the most likely place for an integration bug. Slice 0 must identify where this decode lives in upstream and record it. If it lives in the DiT module, Slice 2 owns it; if in the render module, Slice 4 owns it; if standalone, this slice owns it.
- **End-to-end smoke strength:** the corrected no-CUDA plan uses source-contract modules and synthetic/local images for smoke. This proves wiring, CLI, metrics, export, and no-CUDA behavior, not full vendor numerical parity.
- **CLI subcommand mismatch with prior pipelines:** if `hyworld2.py` uses `reconstruct_root` as a positional, the LiTo CLI should not silently diverge. Mirror exact argparse conventions where the operation is analogous.
- **`uv build` regression:** after editing `pyproject.toml`, run `uv build` to confirm the wheel and sdist still produce cleanly (AC-08). If `uv build` fails, the dev-dep addition has reached into the wheel surface — investigate whether `torch` or `plyfile` accidentally got listed under runtime `dependencies` instead of `[dependency-groups.dev]`.

## Done When

All verification commands pass AND `mlx-spatial-lito generate` produces a valid `.ply` from a sample input AND `from mlx_spatial.lito import LitoInferencePipeline` works with `vendors/` absent AND no existing pipeline test regresses.
