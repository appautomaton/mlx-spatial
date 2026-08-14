# Development

## Setup

```bash
uv sync
export MLX_SPATIAL_TEST_SCRATCH="$(mktemp -d /tmp/mlx-spatial-test.XXXXXX)"
mkdir -p "$MLX_SPATIAL_TEST_SCRATCH"/{inputs,outputs,artifacts,logs,cache}
export PYTHONPYCACHEPREFIX="$MLX_SPATIAL_TEST_SCRATCH/cache/pycache"
uv run pytest -q --basetemp "$MLX_SPATIAL_TEST_SCRATCH/artifacts/pytest"
```

The unified `mlx-spatial` package requires Python 3.13 on an Apple Silicon Mac.
Its integrated SpatialKit extension is built as part of the same distribution.

## Targeted Checks

SAM3D-focused checks:

```bash
uv run pytest tests/test_sam3d*.py -q \
  --basetemp "$MLX_SPATIAL_TEST_SCRATCH/artifacts/pytest-sam3d"
uv run python scripts/sam3d/reconstruct.py --help
uv run python scripts/sam3d/inspect_trace.py --help
```

Package checks:

```bash
release_tmp="$(mktemp -d /tmp/mlx-spatial-release.XXXXXX)"
uv build --out-dir "$release_tmp/dist"
uv run python scripts/packaging/check_release_artifacts.py \
  "$release_tmp"/dist/mlx_spatial-*.tar.gz \
  "$release_tmp"/dist/mlx_spatial-*.whl
uv run python scripts/packaging/check_release_artifacts.py --git-hygiene
```

CLI smoke checks:

```bash
uv run mlx-spatial-sam3d --help
uv run mlx-spatial-trellis2 --help
uv run mlx-spatial-hyworld2 --help
uv run mlx-spatial-lito --help
uv run mlx-spatial-mapanything --help
uv run mlx-spatial-pixal3d --help
```

## Local Assets

Keep large and gated assets out of git:

```text
weights/
inputs/
outputs/
vendors/
```

Tests should pass without downloading gated weights unless they are explicitly marked as optional parity or local-inference checks. Runtime commands that need weights should fail with structured blockers instead of fabricating outputs.

## Temporary Test And Audit Data

Create one task-specific directory for any check that writes files. If setup
already created `MLX_SPATIAL_TEST_SCRATCH`, reuse it:

```bash
export MLX_SPATIAL_TEST_SCRATCH="$(mktemp -d /tmp/mlx-spatial-test.XXXXXX)"
mkdir -p "$MLX_SPATIAL_TEST_SCRATCH"/{inputs,outputs,artifacts,logs,cache}
export PYTHONPYCACHEPREFIX="$MLX_SPATIAL_TEST_SCRATCH/cache/pycache"
uv run pytest --basetemp "$MLX_SPATIAL_TEST_SCRATCH/artifacts/pytest"
```

Keep generated fixtures, parity bundles, browser dependencies, logs, caches,
and model outputs below that root. `outputs/` in the repository is reserved for
user-requested inference results, not test or audit scratch data. Preserve the
temporary root only when its artifacts are needed for diagnosis; otherwise
remove it after recording the relevant result.

## Representative Pipeline Fixtures

Routine integration coverage uses generated miniature checkpoints, composed
pipeline fixtures, and reviewed real-weight-derived boundary patches. Every
fixture states its provenance and limits. None requires the local production
weight bundles.

- TRELLIS.2 generates and quantizes a miniature selective INT8 bundle, then
  runs through the real SpatialKit GLB path.
- Pixal3D combines a synthetic full-orchestration fixture with a compact
  real-weight-derived decoder patch.
- LiTo runs its complete source-contract generation path.
- SAM3D preserves CLI reconstruction, mesh extraction, and GLB writing around
  deterministic conditioning and flow boundaries.
- HY-World 2 runs fixture reconstruction and writes its staged artifacts below
  pytest temporary storage.
- MapAnything generates a miniature checkpoint and runs asset inspection,
  safetensors loading, encoder, multi-view information sharing, prediction
  heads, geometry postprocess, and NPZ writing.

### TRELLIS.2 Synthetic Miniature

The TRELLIS.2 golden test does not read `weights/` or download model assets. It
generates a miniature source checkpoint, applies the production selective INT8
quantizer, and runs image conditioning, sparse sampling, shape and texture SLat
sampling, both decoders, artifact serialization, and the real SpatialKit export
path with miniature export settings.

```bash
uv run pytest tests/test_trellis2_golden_fixture.py -q
```

Reviewed tensor and GLB expectations live in
`tests/data/trellis2_miniature_golden.json`. Do not regenerate that manifest
automatically during tests. Rebaseline it only after reviewing an intentional
inference-contract change.

### Pixal3D Real-Weight-Derived Decoder Patch

`tests/data/pixal3d_derived_golden/` contains a 16-cubed spatial patch captured
from an official-sample run of `TencentARC/Pixal3D` revision
`0b31f9160aa400719af409098bff7936a932f726`. The source used unquantized BF16
flow and FP16 decoder checkpoints. The committed patch is under 64 KiB and
replays decoded O-Voxel validation, mesh extraction, remeshing, MLX QEM, UV,
texture baking, and GLB writing without the 22 GB source bundle.

```bash
uv run pytest tests/test_pixal3d_derived_golden.py -q
```

The default test verifies provenance, checksums, and decoded contracts. The
bounded Metal-backed replay normally completes in under one second. It does not
replace the synthetic full-pipeline Pixal3D test: the derived fixture begins at
the decoder-output boundary.

Rebaseline only from a reviewed real inference result:

```bash
uv run python scripts/pixal3d/write_derived_golden_fixture.py \
  outputs/pixal3d/real-smoke-moge-balanced-decoders1100k \
  outputs/pixal3d/real-smoke-moge-balanced-decoders1100k/trace.json \
  tests/data/pixal3d_derived_golden \
  --source-revision 0b31f9160aa400719af409098bff7936a932f726
```

### MapAnything Generated Miniature Scene

The MapAnything fixture generates its checkpoint at runtime from deterministic
tiny tensors. The committed manifest records fixture provenance, covered and
excluded scope, stage order, output schemas, and tolerant numerical summaries.
It is a pipeline regression fixture, not official-weight parity.

```bash
uv run pytest tests/test_mapanything_scene_pipeline.py \
  -m 'integration and not real_assets' -q
```

Reviewed expectations live in
`tests/data/mapanything_miniature_scene_golden.json`. Rebaseline them only after
reviewing an intentional production pipeline change.

## Editing Constraints

- Prefer existing module boundaries over new abstractions.
- Keep model-specific behavior inside the relevant `sam3d_*`, `trellis2_*`, `hyworld2_*`, `lito_*`, `mapanything_*`, or `pixal3d_*` modules.
- Keep shared primitives model-neutral.
- Do not add generated outputs, converted weights, vendor checkouts, or agent state to package artifacts.
- Use structured parsers for model metadata and safetensors; avoid ad hoc parsing when a local helper exists.
- Keep scripts self-documented with argparse and stable defaults.

## Documentation Style

Stable docs should be short enough to scan and precise enough for agents to execute:

- Lead with the supported path before maintainer or parity paths.
- Name exact asset roots, input roots, output roots, and CLIs.
- Prefer current commands over historical transcripts.
- Put dated verification evidence in release notes, PRs, or `.agent/work/` artifacts, not stable docs.
- Do not describe internal slice decisions unless the decision still affects the public runtime contract.

## Reference Parity

Reference parity work belongs in targeted tests or dev-only scripts. It should record:

- source checkpoint path
- converted checkpoint path
- tensor count
- missing or extra tensors
- shape mismatches
- max absolute difference

Store heavy audit outputs under `MLX_SPATIAL_TEST_SCRATCH`. Commit only small,
portable fixtures that are intentionally part of the test or documentation
contract; committed metadata must not contain developer-machine absolute paths.

## Worktree Hygiene

This repo often has local generated files and experimental model outputs. Before release review:

```bash
git status --short
uv run python scripts/packaging/check_release_artifacts.py --git-hygiene
```

Separate release-readiness changes from unrelated pipeline implementation changes. Do not revert unrelated dirty files unless that is the explicit task.
