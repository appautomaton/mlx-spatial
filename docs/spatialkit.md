# SpatialKit

Native C++, Metal, and Python export primitives integrated into the
`mlx-spatial` distribution as `mlx_spatial.spatialkit`. The API boundary accepts
ordinary Python buffers, NumPy arrays, and decoded files; callers do not install
or publish a second package.

## Install From This Checkout

```bash
uv sync
```

The root build uses scikit-build-core and nanobind to place `_native` and the
Metal kernel resource below `mlx_spatial/spatialkit`. It requires Python 3.13,
Apple Silicon, and macOS 14 or newer. A separate Metal command-line compiler
component is not required because the runtime can compile the packaged kernel
source through the Metal API.

For SpatialKit development:

```bash
export MLX_SPATIAL_TEST_SCRATCH="$(mktemp -d /tmp/mlx-spatial.XXXXXX)"
mkdir -p "$MLX_SPATIAL_TEST_SCRATCH"/{inputs,outputs,artifacts,logs,cache}
export PYTHONPYCACHEPREFIX="$MLX_SPATIAL_TEST_SCRATCH/cache/pycache"
uv run pytest tests/spatialkit \
  --basetemp "$MLX_SPATIAL_TEST_SCRATCH/artifacts/pytest"
```

## Decoded Pixal3D Export

Create one task-specific scratch root before an audit or heavy export, or reuse
the root created during an earlier conversion session:

```bash
export MLX_SPATIAL_TEST_SCRATCH="$(mktemp -d /tmp/mlx-spatial.XXXXXX)"
mkdir -p "$MLX_SPATIAL_TEST_SCRATCH"/{inputs,outputs,artifacts,logs,cache}
```

Then call the decoded-NPZ entry point:

```python
import os
from pathlib import Path

from mlx_spatial.spatialkit import export_pixal3d_glb

scratch = Path(os.environ["MLX_SPATIAL_TEST_SCRATCH"])
result = export_pixal3d_glb(
    "inputs/mlx-spatialkit/pixal3d-1024-cascade-decoded-pbr",
    scratch / "outputs" / "pixal3d-preview",
)
print(result.glb.path)
print(result.diagnostics_path)
```

`inputs/mlx-spatialkit/` is a legacy fixture-directory name retained so large
local decoded artifacts and existing oracle metadata do not need to move. It
does not name an installable package; runtime imports use
`mlx_spatial.spatialkit` from the single `mlx-spatial` distribution.

The decoded directory must contain:

```text
shape_decoder_fields.npz
texture_decoder_pbr.npz
```

The exporter writes `model.glb`, `diagnostics.json`, and optional visual parity
sidecars below the requested output directory.

## Quality Paths

The default is a fast preview contract:

- simplifier: `spatial-cluster`
- UV backend: `face-atlas`
- texture postprocess: `legacy-dilation`
- quality preset: `preview`

`quality_preset="reference-target"` raises the face target and enables strict
diagnostics, but it does not automatically select every available reference
component. In particular, it still selects the topology-aware preview
simplifier unless QEM/remesh are explicitly requested.

The native implementation currently includes:

- FlexiDualGrid mesh extraction and mesh diagnostics
- a CuMesh-compatible narrow-band UDF double-cover behavior control; it is not
  a faithful single-surface remesh
- native QEM edge-collapse simplification
- experimental single-layer native QEM and MLX/Metal batched-QEM pipelines,
  with deterministic multicore native topology rebuilds
- paired-triangle face atlas and native chart UV candidates
- real pinned-xatlas `xatlas-global` and CuMesh-style `xatlas-clustered` paths
- the shared `xatlas-parallel-spatial` experiment available to Trellis2,
  SAM3D, and direct SpatialKit exports; its diagnostics report artificial
  spatial-partition cut edges
- a separate measured `xatlas-equivalent-native` implementation using native
  chart growth, LSCM, and shelf packing
- Metal UV rasterization and PBR texture baking
- trilinear source projection with bounded fallback
- native Telea-style postprocessing
- normals, PBR materials, uint16 primitive chunking, and GLB compatibility
  checks

The current single-surface geometry/UV experiment is:

```python
result = export_pixal3d_glb(
    decoded_dir,
    output_dir,
    quality_preset="reference-target",
    remesh=False,
    simplify_backend="single-layer-mlx-qem",
    uv_backend="xatlas-clustered",
    texture_postprocess="telea",
)
```

`single-layer-mlx-qem` preserves the original FlexiDualGrid surface, performs a
topology-aware coarse pass, repairs disconnected incident face fans, and then
uses conflict-free MLX/Metal QEM batches. It rejects `remesh=True`. The older
plain `qem` option still requires narrow-band remesh plus nonmanifold repair,
but that UDF path is retained as a behavior control because it creates a double
cover. Neither experimental QEM path is selected by default.

Real xatlas output and `xatlas-equivalent-native` have different correctness
contracts. The native-equivalent path promises zero flipped/overlapping UV
faces. Real xatlas may mirror complete charts and may record a small overlap
count with reference `padding=0`; its gate therefore checks the pinned xatlas
version, chart/utilization ratios, overlap ratio, affected surface area, and UV
surface coverage. Diagnostics record all measurements; backend names alone do
not clear the gates.

## Apple Silicon Execution

Use MLX where the operation is regular and numerically dense. The experimental
MLX QEM path keeps vertex quadrics, edge costs, local-minimum propagation, and
conflict selection lazy on an explicit Apple GPU stream, synchronizing once per
topology-changing round. Edge deduplication, variable-length adjacency, CSR
rebuilds, and collapse application remain on CPU because they are irregular;
their native implementations use deterministic multicore work rather than a
single Python/NumPy loop.

Performance evidence is valid only when diagnostics confirm the execution
device and the watchdog records CPU/GPU utilization, peak RSS, and zero swap
growth. Any swap growth invalidates a run.

## Current Measured Evidence

Cached decoded-NPZ runs now separate topology, geometric distance, rendered
appearance, and runtime evidence:

- The main 212k-face r512 asset and the thin violin-bow 48k-face r1024 asset
  both finish with zero boundary and nonmanifold edges and pass the shared-
  camera browser checks.
- On the violin-bow body, source-to-candidate p95 distance falls from 4.31 to
  2.04 to 1.02 voxels as remesh resolution increases from 256 to 512 to 1024.
  Large full-mesh maxima come from five minor source components, not holes in
  the watertight main component.
- Four-way `xatlas-parallel-spatial` is valid but not a speed winner. It takes
  2.79 seconds versus 1.20 seconds for `xatlas-clustered` on the 48k-face
  violin asset, and 11.52 seconds versus 1.74 seconds on the 212k-face main
  asset. It also introduces 941 and 2,999 spatial partition cuts,
  respectively. It remains an explicit compatibility/experiment path.
- The measured runs peak below 4.4 GiB RSS and record zero swap growth. MLX
  QEM and Metal texture stages use the Apple GPU; irregular topology and
  xatlas stages remain CPU work by design.

These results do not establish a production default. In particular, the UDF
path is still an offset double cover, the evidence is below the upstream
1M-face/4096-texture contract, and the fine-structure suite needs another
independent asset.

## Readiness Contract

Keep these concepts separate:

- `artifact_ready`: a parseable GLB was produced.
- `production_quality_ready`: scalar quality thresholds passed.
- `production_equivalence_ready`: reference stages, upstream settings,
  measured UV parity, rendered comparison, and deferred boundaries all passed.

The strict production-equivalence gate is intentionally conservative. Before
changing any Pixal3D or TRELLIS.2 pipeline default, the current work is:

- extend the cached decoded-NPZ evidence to a second fine-structure fixture
- establish full reference-scale `target_faces=1000000` and
  `texture_size=4096` evidence within time and memory budgets
- add an object-level preservation gate for fine structures such as the
  violin-bow fixture
- decide whether the measured single-layer MLX QEM candidate is strong enough
  to replace the current default exporter

Do not describe QEM, narrow-band UDF remesh, or reference UV as missing; those
implementations exist. Do not describe the UDF double cover as a solved
single-surface remesh, or the experimental MLX QEM path as production-ready.

## Tests And Artifacts

Fast tests use pytest's temporary directories. Run them with an explicit task
root when you need artifact isolation:

```bash
uv run pytest tests/spatialkit \
  --basetemp "$MLX_SPATIAL_TEST_SCRATCH/artifacts/pytest"
```

Heavy tests require real decoded fixtures and, for Metal paths, compatible
Apple hardware:

```bash
uv run pytest tests/spatialkit -m heavy \
  --basetemp "$MLX_SPATIAL_TEST_SCRATCH/artifacts/pytest-heavy"
```

Oracle environments, caches, generated GLBs, screenshots, JSON/HTML reports,
and logs must remain below `MLX_SPATIAL_TEST_SCRATCH`. Committed anchor metadata
may contain stable repository-relative fixture identifiers and numeric hashes,
but never developer-machine absolute paths.

## Browser Visual Proof

Browser rendering is dev-only and not a package dependency:

```bash
npm install \
  --prefix "$MLX_SPATIAL_TEST_SCRATCH/cache/render-deps" \
  playwright@1.60.0 three@0.181.2
NODE_PATH="$MLX_SPATIAL_TEST_SCRATCH/cache/render-deps/node_modules" \
  node scripts/spatialkit/render_glb_visual_parity.cjs \
  --candidate "$MLX_SPATIAL_TEST_SCRATCH/outputs/candidate/model.glb" \
  --reference "$MLX_SPATIAL_TEST_SCRATCH/inputs/reference/model.glb" \
  --output-dir "$MLX_SPATIAL_TEST_SCRATCH/artifacts/browser-render" \
  --visual-report "$MLX_SPATIAL_TEST_SCRATCH/artifacts/visual-parity.json"
```

The report is deterministic inspection evidence. It is not by itself proof of
xatlas equivalence, CUDA/cuMesh equivalence, or perceptual identity.
