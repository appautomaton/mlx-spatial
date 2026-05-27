# Pixal3D

Pixal3D is TencentARC's projection-conditioned image-to-3D pipeline. In
`mlx-spatial`, Pixal3D is currently an implementation track. The runtime can
use converted local NAF safetensors to build high-resolution projected DINOv3
features without Torch, and can write textured GLB after decoded shape and
texture tensors are available.

Implemented now:

- local asset validation and checkpoint inspection for `TencentARC/Pixal3D`
- MoGe-derived auto-camera through the existing converted MLX MoGe
  pointmap/intrinsics runtime
- Pixal3D manual-FOV camera math matching the upstream inference script as an
  explicit override
- sparse-stage DINOv3 hidden-state extraction through the shared MLX DINOv3
  helper
- projection conditioning from DINOv3 hidden states: global tokens plus
  view-aligned sparse-structure grid features
- Pixal3D `image_attn_mode="proj"` block math in the shared sparse-structure
  and SLat flow boundaries
- sparse-structure FlowEuler probing through the shared MLX sparse flow helper
- sparse decoder coordinate extraction when a compatible sparse decoder
  checkpoint/config is available
- coordinate-indexed 512 shape SLat probing with either explicit NAF-upsampled
  features or MLX NAF-projected features from local converted weights
- shared shape decoder LR-to-HR coordinate upsample, Pixal3D HR token
  quantization, and HR coordinate artifact writing
- coordinate-indexed 1024 shape SLat and texture SLat probing through the same
  explicit-override or MLX NAF-projection path
- shared FlexiDualGrid shape decoder execution after HR shape SLat, writing
  decoded 7-channel shape fields
- shared guided texture decoder execution after texture SLat, writing decoded
  6-channel PBR voxel attributes
- shared FlexiDualGrid mesh extraction, Mac-native texture baking, and
  Pixal3D-labeled textured GLB writing after decoded tensors are available
- cascade stage planning for `1024_cascade` and `1536_cascade`
- trace output, `sparse_projection.npz`, `sparse_structure.npz`, and
  shape/texture/decode intermediate artifacts as each MLX boundary completes

Still blocked:

- missing converted NAF weights block NAF-projected stages until
  `weights/naf/naf_release.safetensors` is created locally
- missing converted MoGe weights block auto-camera until
  `weights/sam-3d-objects-mlx/moge/model.safetensors` is present; pass
  `--manual-fov` to use the deterministic override

## Assets

Download the upstream Pixal3D weights manually:

```bash
uv run mlx-spatial-pixal3d download-command weights/pixal3d
```

That prints:

```bash
uv run hf download TencentARC/Pixal3D --local-dir weights/pixal3d
```

Then validate:

```bash
uv run mlx-spatial-pixal3d validate weights/pixal3d
uv run mlx-spatial-pixal3d inspect weights/pixal3d --limit 5
```

The upstream Hugging Face metadata currently identifies the Pixal3D model repo
as MIT-licensed. Respect any Hugging Face access gates and upstream model-card
terms when downloading or redistributing outputs.

Pixal3D image conditioning also needs local DINOv3 ViT-L/16 assets:

```bash
uv run mlx-spatial-trellis2 dinov3-download-command weights/dinov3-vitl16-pretrain-lvd1689m
uv run hf download facebook/dinov3-vitl16-pretrain-lvd1689m \
  config.json model.safetensors \
  --local-dir weights/dinov3-vitl16-pretrain-lvd1689m
```

Pixal3D's upstream projection stages use Valeo NAF. Convert the NAF release
checkpoint locally for the Torch-free runtime:

```bash
uv run --group torch-ref python scripts/pixal3d/convert_naf.py \
  --output weights/naf/naf_release.safetensors
```

Pixal3D auto-camera reuses the package's existing converted MLX MoGe root. The
default is `weights/sam-3d-objects-mlx/moge`, normally supplied by the public
`appautomaton/sam-3d-objects-mlx` bundle:

```bash
uv run hf download appautomaton/sam-3d-objects-mlx \
  --local-dir weights/sam-3d-objects-mlx
```

This is a MoGe-derived MLX auto-camera path, not an exact claim of upstream
`Ruicheng/moge-2-vitl` parity.

## Recommended Run

Use the vendored sample image from the shallow upstream checkout:

```bash
python scripts/pixal3d/generate.py vendors/Pixal3D/assets/images/0_img.png \
  --root weights/pixal3d \
  --dino-root weights/dinov3-vitl16-pretrain-lvd1689m \
  --moge-root weights/sam-3d-objects-mlx/moge \
  --naf-root weights/naf \
  --output-dir outputs/pixal3d/sample \
  --pipeline-type 1024_cascade
```

When Pixal3D, MoGe, DINOv3, and NAF assets are present, the current expected
output is:

```text
outputs/pixal3d/sample/
  trace.json
  sparse_projection.npz
  sparse_structure.npz          # written after sparse decoder coordinates are available
  shape_slat_lr.npz             # written after LR NAF projection succeeds
  shape_slat_hr_coordinates.npz # written after compatible shape decoder upsample
  shape_slat_hr.npz             # written after HR NAF projection succeeds
  texture_slat.npz              # written after texture NAF projection succeeds
  shape_decoder_fields.npz      # written after shared FlexiDualGrid shape decode
  texture_decoder_pbr.npz       # written after guided texture PBR voxel decode
  model.glb                     # written after mesh extraction and texture baking
```

If converted MoGe weights are missing, omitted `--manual-fov` returns a
structured `camera-setup` blocker with the MoGe root and memory profile. If the
DINOv3 assets are missing, the CLI returns an `image-conditioning` blocker with
the exact root and download command. If converted NAF weights are missing,
NAF-projected stages return a structured `naf-assets` blocker with the expected
safetensors path and conversion command. When MoGe, DINOv3, NAF, sparse-flow,
and sparse-decoder assets are present, the runtime can write
`sparse_structure.npz`, build coordinate-sampled NAF projections, probe 512
shape SLat, write `shape_slat_lr.npz`, upsample guarded HR coordinates, write
`shape_slat_hr_coordinates.npz`, probe 1024 shape and texture SLat stages as
downstream assets permit, run the shared shape and texture decoders, then write
a Pixal3D-labeled textured GLB.

## Settings

- pipeline type: use `1024_cascade` on Apple Silicon by default
- high-memory mode: `1536_cascade`
- seed: `42`
- max tokens: `49152`, matching the upstream cascade guard
- shape upsample token limit: `1000000`; this is the MLX compute guard before
  HR coordinate quantization/reduction
- shape decoder token limit: `1100000`
- texture decoder token limit: `1100000`
- texture size: `1024`
- GLB face target: `50000`
- xatlas face guard: `auto`
- texture bake backend: `kdtree`
- GLB export backend: `internal`; optional `spatialkit` uses `mlx-spatialkit`
  when importable and records a diagnostics JSON sidecar
- MoGe root: `weights/sam-3d-objects-mlx/moge`
- MoGe memory profile: `balanced`; alternatives are `safe` and `large`
- manual FOV override: radians, for example `--manual-fov 0.2`
- DINOv3 root: `weights/dinov3-vitl16-pretrain-lvd1689m`
- NAF root: `weights/naf`
- NAF coordinate chunk size: `8192`
- sample image: `vendors/Pixal3D/assets/images/0_img.png`

The cascade planner starts at 1024 or 1536 output resolution, quantizes HR
coordinates onto `resolution / 16`, and steps the HR resolution down by 128
until the sparse token count is below `max_num_tokens` or the 1024 floor is
reached.

## Runtime Boundary

Runtime modules are Torch-free:

- `pixal3d_assets.py`: asset manifest, validation, config parsing, and probes
- `pixal3d_camera.py`: MoGe intrinsics camera conversion, manual-FOV camera
  override, and cascade planning
- `pixal3d_projection.py`: projection grid, FOV projection, feature sampling,
  coordinate-indexed feature selection, and explicit NAF map override support
- `naf.py`: Torch-free converted NAF safetensors loading, image encoder, RoPE,
  and coordinate-sampled neighborhood attention
- `pixal3d_export.py`: intermediate projection, sparse-coordinate, HR
  coordinate, shape SLat, texture SLat, shape decoder, and texture decoder NPZ
  artifact writing plus Pixal3D-labeled GLB writing
- `pixal3d_inference.py`: staged orchestration, MLX MoGe auto-camera handoff,
  trace metadata, export settings, and blockers
- `trellis2_dinov3.py`, `trellis2_dinov3_forward.py`: shared MLX DINOv3
  hidden-state extraction
- `trellis2_sparse_structure.py`: shared sparse FlowEuler probing, sparse
  decoder boundary checks, and config-gated Pixal3D projection attention
- `trellis2_decode.py`: shared shape decoder coordinate upsample plus
  full shape/texture decoder execution
- `trellis2_export.py`: shared mesh postprocess, texture baking, and GLB payload
  helpers
- `trellis2_slat.py`: shared SLat flow boundary with config-gated Pixal3D
  projection attention

Dev-only PyTorch reference capture and NAF checkpoint conversion are setup
workflows. Runtime imports remain Torch-free.

## Native Spatialkit Export

The default GLB path stays internal and requires no `mlx-spatialkit` install.
For decoded NPZ to GLB work, opt into the native companion backend:

```bash
python scripts/pixal3d/generate.py vendors/Pixal3D/assets/images/0_img.png \
  --root weights/pixal3d \
  --dino-root weights/dinov3-vitl16-pretrain-lvd1689m \
  --moge-root weights/sam-3d-objects-mlx/moge \
  --naf-root weights/naf \
  --output-dir /tmp/mlx-spatialkit-pixal3d \
  --pipeline-type 1024_cascade \
  --glb-export-backend spatialkit
```

`spatialkit` consumes the same `shape_decoder_fields.npz` and
`texture_decoder_pbr.npz` artifacts and writes `model.glb` plus
`diagnostics.json` next to the output unless `--glb-diagnostics-path` is set.
If `mlx_spatialkit` is not importable, the pipeline falls back to the internal
writer and records the fallback reason in trace metadata. Real fixture tests for
this path are marked `heavy` and write generated artifacts under `/tmp`.

Spatialkit diagnostics separate artifact readiness from production-quality
readiness. The current native texture path uses Metal exact sparse-voxel
sampling plus bounded fallback/fill and records raw exact coverage, final visible
base-color coverage, fallback-filled texels, timings, and RSS samples. Preview
mesh simplification is reported as `spatial-cluster` with
`quality_tier=geometry_aware_preview`. Reference-target exports request the
native `topology-aware` simplifier, which keeps the topology guard and chooses
representative source vertices for clustered output vertices. When the checked-in
Pixal3D reference trace is available, diagnostics include `reference_comparison`
face-count and coverage ratios.

For 4096 reference-target texture export, spatialkit scales the Metal nearest
fallback radius and native dilation pass budget from the atlas tile size instead
of using a fixed low budget. Diagnostics report the resolved
`fallback_radius`, `dilation_max_passes`, and actual `dilation_pass_count`. The
real decoded Pixal3D fixture now passes the production final-coverage threshold
at `texture_size=4096` while keeping generated artifacts under `/tmp`.
Dense 4096 atlases use an additional bounded floor so explicit upstream-style
`target_faces=1000000`, `texture_size=4096` export can fill enough visible
texture coverage without changing decoded model outputs.

The native UV path now uses paired-triangle face-atlas packing. Each atlas tile
can hold two unrelated triangle faces in complementary halves, and diagnostics
report `packing=paired-triangles`, `faces_per_tile=2`, and texture-bake
`atlas_faces_per_tile=2`. This improves global texture coverage without adding
xatlas as a package dependency, but it is not equivalent to xatlas chart
generation or a production remesh backend.

When a caller supplies arbitrary non-atlas UVs, the Metal bake no longer scans
every face for every texel. It builds a bounded UV-space face-bin index and
reports `backend=metal-uv-binned-nearest` with bin grid, face-reference,
max-candidate, and guard diagnostics. Remaining UV-surface holes are filled by
a bounded native nearest-visible surface pass, with raw exact coverage,
fallback-filled texels, surface-filled texels, and final visible coverage
reported separately. `mlx-spatialkit` also exposes `make_native_chart_uvs` as an
opt-in native chart candidate. It groups
edge-connected smooth faces by a configurable normal-angle threshold, reuses
vertices within a chart, splits oversized charts into deterministic spatial
chunks, applies bounded low-fill chart splitting when a chart under-fills its
projected rectangle, duplicates only at chart boundaries, and bakes through the
binned Metal path. Chart projection uses a deterministic local-frame/PCA
rotation before deterministic aspect-aware shelf packing. Projection evaluates
19 PCA-centered rotation candidates at 5-degree steps and reports the candidate
count, step, oversized and low-fill split counts, chart rect fill, packing
efficiency, and packed-bounds diagnostics. Existing Pixal3D exports still use
the paired-triangle face-atlas fast path; this candidate does not replace xatlas
charting or prove CUDA/cuMesh remesh parity.

For direct decoded-NPZ conversion, `mlx_spatialkit.export_pixal3d_glb` accepts
`uv_backend="native-chart"` and `chart_angle_degrees=45.0` as an opt-in chart
candidate export. That path records `native_chart_uv_candidate` diagnostics,
bakes through `metal-uv-binned-nearest`, and keeps `xatlas_chart_parity=false`.
The default remains `uv_backend="face-atlas"` for stability. If `tile_padding`
is not supplied, face-atlas exports keep `0.08`; native-chart exports resolve
to sub-texel `0.001` padding and record `settings.tile_padding_source` so the
contract is visible.
When the checked-in reference trace has xatlas unwrap metrics, spatialkit also
records `quality.xatlas_chart_parity`: reference xatlas chart count/utilization,
native chart count/UV-surface occupancy, ratio fields, deficit fields, checks,
and `parity_ready=false`. The `xatlas_utilization_equivalence` check uses a
`0.95` utilization-ratio target and currently fails for the native-chart path,
which keeps the remaining boundary measurable without adding xatlas as a
`mlx-spatialkit` runtime dependency or claiming chart equivalence.
The native low-fill splitter now uses a higher bounded fill target, one extra
split depth, and 4-face/2-face-child minimums for small low-fill charts. This
improves the reference-target fixture's chart fill and xatlas-utilization ratio
while keeping xatlas parity explicitly false.
Eligible low-fill charts also evaluate both local centroid split axes and five
fixed split positions, then accept the best improving split.
`low_fill_split_partition_candidate_count` records the bounded partition search.
`native_chart_uv_candidate` separates `artifact_ready` from `quality_ready`:
the current chart path writes a valid GLB and clears the scalar native-chart
coverage checks after UV-surface fill, while preserving `xatlas_chart_parity=false`
and raw/exact/final coverage diagnostics.
When paired with `quality_preset="reference-target"`, the opt-in native-chart
path also passes production threshold checks and deterministic GLB/PNG visual
comparison on the real fixture. That closes the reference-target native-chart
readiness gate, but it is still scalar readiness. The stricter
`quality.production_equivalence.ready` and
`result.production_equivalence_ready` fields remain false while xatlas chart
parity, 1M/4096 upstream-setting parity, or deterministic visual-comparison
boundaries remain open.

The export diagnostics now keep geometry-hole evidence separate from UV chart
coverage. Native `mesh_metrics` reports boundary vertices, closed boundary
loops, open boundary chains, small boundary loops, and max boundary component
size. It also reports open-boundary edge totals, small-open-component totals,
simple open-chain count, branched open-component count, endpoint count, and
branch-vertex count. `export_pixal3d_glb` records those metrics for both the
source and final export meshes, so visible holes can be evaluated as topology
evidence before we change simplification, repair, or UV charting. The current
reference-target fixture's open components are branched rather than simple
endpoint-to-endpoint chains, so open-boundary repair remains a separate design
step.
For the topology-aware production simplifier, `mlx-spatialkit` also performs a
bounded small-loop fill after simplification. The repair uses projected
ear-clipping for closed boundary loops up to 8 edges by default, respects the
remaining target-face budget, and rejects patches that would introduce
degenerate, duplicate, or nonmanifold faces. If the first ear-clipping diagonal
is topology-blocked, the native path can try bounded alternate ear-clipping
triangulations for the same loop under the same guards. If projected
ear-clipping fails, a conservative centroid-fan fallback may fill loops up to the
same 8-edge policy cap. The repair also makes one bounded second pass and may
fill small simple cycles, with a more conservative 6-edge policy cap, discovered
inside branched open-boundary components. Diagnostics report policy caps,
effective caps, method counts, alternative-triangulation counts, branch-cycle
counts, rejection reason counts, budget, and faces added. This addresses small
geometry holes separately from xatlas chart parity, endpoint-chain repair, or
open-boundary remeshing.
The public export parameter `small_boundary_loop_fill_max_edges` defaults to
`8` for this measured policy, clamps the effective fallback and branched-cycle
caps, and `0` disables the fill when comparing geometry repair against the
unpatched simplifier output.
For texture seam robustness, the native bake fills a bounded no-face gutter
after UV-surface fill. The gutter copies RGB and metallic/roughness values into
padding texels that can be touched by linear texture filtering, but keeps alpha
and coverage status unchanged so UV-surface and visible-coverage metrics do not
inflate. Diagnostics report the gutter pass count and filled texel count, and
visual comparison reports raw RGB footprint separately from visible RGB coverage.
When the same opt-in native-chart backend is run with explicit upstream-style
`target_faces=1000000`, `texture_size=4096`, the real fixture passes both
`quality.upstream_export_settings` and `quality.native_chart_uv_candidate`
readiness. The 1024 reference GLB visual comparison is still expected to report
face-count and texture-resolution mismatches; that is an honest reference-target
comparison boundary, not a native-chart export failure.

For decoded NPZ validation, `mlx_spatialkit.export_pixal3d_glb` also accepts
`quality_preset="reference-target"`. That preset resolves the face target from
the checked-in reference trace when available and records
`production_thresholds` for reference availability, preset, backend tier,
topology exportability, face-count ratio, final coverage ratio, and raw coverage
reporting. Current reference-target diagnostics are expected to stay
`artifact_ready=true`; on the current heavy fixture they also reach
`production_quality_ready=true` because backend tier, face-count, topology,
final coverage, raw coverage reporting, preset, and reference checks all pass.
`production_quality_ready` is the scalar reference-target threshold result; use
`production_equivalence_ready` to decide whether the export can be treated as
full Pixal3D production-equivalent. Current diagnostics keep that stricter flag
false because xatlas parity is still deferred.
Both the default face-atlas path and the opt-in native-chart path have
reference-target real-fixture gates.
Explicit upstream-style `target_faces=1000000`, `texture_size=4096` export has
a separate `quality.upstream_export_settings` section that checks target faces,
texture size, backend tier, target reach, face retention, artifact readiness,
and final coverage. This closes the 1M/4096 setting-readiness boundary when the
check passes. The native-chart backend has a matching 1M/4096 real-fixture gate,
but it is not full upstream xatlas charting, xatlas chart equivalence, or
CUDA/cuMesh remesh parity.

Native spatialkit GLB output also records
`quality.glb_viewer_compatibility`. The writer emits `NORMAL` attributes and
splits large meshes into chunk-local primitives using `UNSIGNED_SHORT` indices
instead of one large uint32-indexed primitive. The compatibility gate checks
parseability, material/texture presence, normals, uint16 index accessors, local
index bounds, and large-mesh chunking. This is intended to reduce strict-viewer
failure modes in macOS Preview/Quick Look and similar tools; it does not replace
xatlas charting or CUDA/cuMesh remesh parity.

When the reference GLB is available, `mlx_spatialkit.export_pixal3d_glb` writes
a `visual_parity/` directory next to the generated GLB. It contains
`visual_parity.json`, an `index.html` texture preview, and extracted
candidate/reference base-color PNGs. The diagnostics JSON includes the compact
visual-comparison summary and paths. This report compares GLB structure, face
counts, texture dimensions, and embedded texture coverage; it is not xatlas
chart parity. The checked-in reference GLB is 1024, so a 4096 candidate should
honestly report texture-resolution mismatch even when its production coverage
gate passes. Default deferred visual parity boundaries now stay limited to
xatlas chart parity and 1M-face export-setting parity.
The 1M setting boundary is removed for explicit 1M/4096 exports only after
`quality.upstream_export_settings.all_passed=true`; xatlas chart parity remains
deferred. For explicit 1M/4096 native-chart exports, the deferred list should
shrink to `["not_xatlas_chart_parity"]` even though the compact visual summary
is not all-passed against the 1024 reference GLB.

The same diagnostics JSON includes a `memory` summary for spatialkit exports.
It records aggregate process RSS samples, observed per-stage RSS peaks, and
`resource.getrusage` high-water RSS. Use this to compare stages such as
`extract_mesh`, `texture_bake`, and `write_glb` within one run. It does not
measure full system memory pressure, exact Activity Monitor app-memory values,
MLX allocator state, or Metal heap residency.
The Metal texture bake releases the Python GIL only while the command buffer is
committed and waited on, so Python monitor threads can continue sampling during
the GPU wait. Python buffer loading and nanobind result construction remain
GIL-held.

Browser-rendered visual proof is available as dev tooling, not a package
runtime dependency. Install pinned Playwright/Three dependencies under `/tmp`
and run `scripts/spatialkit/render_glb_visual_parity.cjs` against the generated
`model.glb` and the checked-in reference GLB. The script writes
`browser_render_report.json`, `comparison.png`, and `index.html` under
`visual_parity/browser_render/` and augments `visual_parity.json` when requested.
This still does not claim xatlas chart parity, CUDA/cuMesh remesh parity, or
exact perceptual equivalence.
