# mlx-spatialkit

Native C++ and Metal spatial export primitives for `mlx-spatial`.

This package is intentionally independent from MLX. It accepts ordinary Python
buffers, NumPy arrays, and files at the binding boundary, while native code owns
the expensive mesh, texture, and export stages.

Pixal3D decoded export entry point:

```python
from mlx_spatialkit import export_pixal3d_glb

result = export_pixal3d_glb(
    "inputs/mlx-spatialkit/pixal3d-1024-cascade-decoded-pbr",
    "/tmp/mlx-spatialkit-pixal3d",
)
print(result.glb.path)
print(result.diagnostics_path)
```

To run the reference-target quality gate, use:

```python
result = export_pixal3d_glb(
    "inputs/mlx-spatialkit/pixal3d-1024-cascade-decoded-pbr",
    "/tmp/mlx-spatialkit-pixal3d-reference",
    quality_preset="reference-target",
)
print(result.diagnostics["quality"]["production_thresholds"])
```

To exercise the native chart UV candidate on decoded Pixal3D tensors, opt in
explicitly:

```python
result = export_pixal3d_glb(
    "inputs/mlx-spatialkit/pixal3d-1024-cascade-decoded-pbr",
    "/tmp/mlx-spatialkit-pixal3d-chart",
    uv_backend="native-chart",
    chart_angle_degrees=45.0,
)
print(result.diagnostics["quality"]["native_chart_uv_candidate"])
```

When `tile_padding` is not supplied, Pixal3D exports resolve padding by UV
backend: `face-atlas` keeps `0.08`, while `native-chart` uses `0.02`.
Diagnostics record both `settings.tile_padding` and
`settings.tile_padding_source`; explicit caller padding is preserved.

The real fixture test is opt-in with `pytest -m heavy` and writes generated
GLB/diagnostic artifacts under `/tmp`.

## Quality Tier

`mlx-spatialkit` separates preview exports from the reference-target quality
gate. The native texture path uses Metal exact sparse-voxel sampling, bounded
sparse-neighbor fallback, and native UV-surface dilation. Diagnostics record raw
exact coverage, fallback-filled texels, final visible base-color coverage,
runtime, and RSS samples.

Default preview simplification still uses `spatial-cluster` with
`quality_tier=geometry_aware_preview`. `quality_preset="reference-target"`
selects the native `topology-aware` simplifier, which uses representative source
vertices inside the topology-guarded clustering flow and reports
`quality_tier=production` only when its self-checks pass.

The UV stage uses a native paired-triangle face atlas: two unrelated triangle
faces share one atlas tile in complementary halves. This is still not xatlas
charting, but it reduces atlas waste and lets the Metal texture bake report
`atlas_faces_per_tile=2`.

For arbitrary non-atlas UV meshes, the Metal texture bake path now builds a
bounded UV-space face-bin index before dispatch. Those bakes report
`backend=metal-uv-binned-nearest` plus bin grid, face-reference, max-candidate,
and guard diagnostics, so chart UVs do not fall back to an
O(texture_pixels * faces) scan. `make_native_chart_uvs` is available as an
opt-in native chart candidate: it groups edge-connected smooth faces by a
normal-angle threshold, splits oversized charts into deterministic spatial
chunks, duplicates vertices at chart boundaries, packs charts through
deterministic local-frame/PCA projection plus aspect-aware shelf packing, and
feeds this binned Metal path. Projection evaluates a deterministic 19-candidate
PCA-centered rotation search at 5-degree steps and reports the candidate count
and step size in UV stats. Existing Pixal3D exports still use the paired-triangle
face atlas by default, and this chart candidate is not xatlas chart parity.
`export_pixal3d_glb(...,
uv_backend="native-chart")` wires the same candidate into the real decoded
Pixal3D export path and records chart count, duplicate ratio, UV-bin
diagnostics, large-chart split counts, chart rect fill, shelf packing
efficiency, and `xatlas_chart_parity=false`.

Chart exports report separate artifact and quality readiness under
`quality.native_chart_uv_candidate`. A chart export can be artifact-ready when
it writes a valid GLB and bakes through `metal-uv-binned-nearest`, while still
being `quality_blocked` if global texture coverage or UV-surface occupancy is
too low. In that case `result.quality_warnings` includes
`native_chart_uv_candidate_quality_blocked`, and the failed checks identify the
specific coverage/utilization blocker.

The current large-chart splitter, bounded rotation search, shelf packer, and
tighter native-chart padding improve the real fixture chart candidate versus the
older fixed-axis/equal-grid chart path. It can still report `quality_blocked`
until global texture coverage clears the readiness floor.

For high-resolution exports, the Metal texture path resolves nearest-voxel
fallback and native dilation budgets from the atlas tile size. Atlas textures
scale `fallback_radius` within `12..24` and `dilation_max_passes` within
`8..64`, with a bounded 4096 floor for dense atlases; non-atlas UVs keep the
lower defaults. The 4096 reference-target gate now passes production coverage
on the real Pixal3D decoded fixture while recording the resolved budgets,
actual dilation passes, and stage RSS peaks.

`quality_preset="reference-target"` resolves the face target from the checked-in
Pixal3D reference trace when available and records threshold checks for topology,
face-count ratio, raw/final texture coverage, and backend tier. The current
reference-target heavy fixture passes face-count, topology, and final global
coverage thresholds, with final visible coverage around `0.602` versus the older
one-triangle atlas baseline of about `0.269`; it now also passes the backend-tier
gate with `production_quality_ready=true`. The 4096 heavy fixture separately
passes production texture coverage. Explicit upstream-style
`target_faces=1000000`, `texture_size=4096` export also has a separate
`quality.upstream_export_settings` gate; when that gate passes, the 1M-face
setting deferral is removed. This is still not a claim of upstream xatlas
charting or CUDA/cuMesh remesh parity.

Native GLB writing now emits `NORMAL` attributes and splits large meshes into
chunk-local primitives with `UNSIGNED_SHORT` indices. Diagnostics record
`quality.glb_viewer_compatibility` to check parseability, material/texture
presence, normals on every primitive, uint16-only indices, local index bounds,
and chunking for large meshes. This targets stricter viewers such as macOS
Preview/Quick Look; it does not change decoded model outputs or imply xatlas
chart parity.

When the checked-in reference GLB is available, reference-target export also
writes a `visual_parity/` sidecar next to `model.glb`: `visual_parity.json`,
`index.html`, and extracted candidate/reference base-color PNGs. The report
compares GLB mesh counts, texture dimensions, and embedded base-color coverage
against the reference GLB. It is deterministic inspection evidence, not a
browser-rendered screenshot or xatlas chart-equivalence proof. When comparing a
4096 candidate against the checked-in 1024 reference GLB, texture-resolution
mismatch is expected and should stay visible in the report. Default deferred
visual parity boundaries now stay limited to xatlas chart parity and 1M-face
export-setting parity; the 1M boundary is removed for explicit 1M/4096 exports
only after upstream-setting readiness passes.

Pixal3D export diagnostics also include a `memory` summary with observed process
RSS peaks per stage, backed by `ps` RSS samples and `resource.getrusage`
high-water RSS. These numbers explain host-process memory behavior during
stages such as `texture_bake` and `write_glb`; they are not full system memory
pressure, Activity Monitor app-memory equivalence, or Metal allocator telemetry.

For browser-rendered visual proof, keep the browser stack outside the package
runtime and install it under `/tmp`:

```bash
npm install --prefix /tmp/mlx-spatialkit-render-deps playwright@1.60.0 three@0.181.2
NODE_PATH=/tmp/mlx-spatialkit-render-deps/node_modules \
  node ../../scripts/spatialkit/render_glb_visual_parity.cjs \
  --candidate /tmp/export/model.glb \
  --reference ../../inputs/mlx-spatialkit/pixal3d-1024-cascade-glb-reference/model.glb \
  --output-dir /tmp/export/visual_parity/browser_render \
  --visual-report /tmp/export/visual_parity/visual_parity.json
```

The script writes a browser screenshot, HTML report, and JSON checks under the
given `/tmp` output directory. It proves that Chrome/Three.js can render both
GLBs nonblank across fixed views; it is not exact perceptual scoring.
