# SAM3D MLX Mesh Decoder Algorithm Parity Plan

## Goal

Port the SAM3D Objects mesh decoder path closely enough that the existing MLX `reconstruct` command can produce a real Blender-readable `mesh.glb` from generated SLat coordinates/features without PyTorch, CUDA, or fake geometry.

## Architecture Approach

Use the current SAM3D MLX pipeline through `gaussians.ply` unchanged, then replace the `mesh-decoder` blocker with a strict mesh path:

- refactor the existing SLat decoder helper so mesh can run the shared transformer torso without applying the Gaussian `out_layer`;
- implement mesh-only `SparseSubdivideBlock3d` using the existing sparse tensor, sparse conv, sparse linear, group norm, and SiLU patterns;
- interpret `slat_decoder_mesh` output in the official SparseFeatures2Mesh layout: `sdf`, `deform`, `weights`, optional `color`;
- assemble guarded dense vertex/cube fields and run a NumPy/MLX FlexiCubes-style inference extractor;
- write a basic GLB with existing export utilities and trace all mesh counts, resolutions, memory estimates, and blockers.

No shipped runtime import may use PyTorch, CUDA, `spconv`, `gsplat`, `nvdiffrast`, `kaolin`, or vendored runtime modules.

## Ordered Task Sequence

### Slice 1: Mesh Decoder Contract And Torso Split

**Objective:** Add a mesh decoder config/tensor contract and split the shared SLat decoder transformer from the final output projection.
**Execution:** direct
**Depends on:** none
**Touches:** `src/mlx_spatial/sam3d_decoder.py`, `src/mlx_spatial/sam3d_contract.py`, `tests/test_sam3d_decoder.py`, `tests/test_sam3d_contract.py`
**Context budget:** ~6% of context window
**Produces:** Mesh decoder config parsing, tensor loading checks, and a reusable decoder torso function that preserves the existing Gaussian path.
**Acceptance criteria:**
- Mesh config reads active `SLatMeshDecoderTdfyWrapper` fields including resolution, channels, blocks, heads, window size, and representation config.
- Tensor loading requires `input_layer.`, `blocks.`, `upsample.`, and `out_layer.` prefixes for `slat_decoder_mesh`.
- Existing Gaussian decoder tests still pass without behavior change.
- A fixture can run the decoder torso and apply either Gaussian or mesh output projection separately.
**Verification:** `uv run pytest -q tests/test_sam3d_decoder.py tests/test_sam3d_contract.py tests/test_sam3d_gaussian.py`
**Auto-continue:** yes

### Slice 2: SparseSubdivideBlock3d In MLX

**Objective:** Port official mesh upsample blocks: sparse group norm, SiLU, sparse subdivide, 3x3 sparse conv stack, zeroed second conv, and optional skip projection.
**Execution:** subagent recommended
**Depends on:** Slice 1
**Touches:** `src/mlx_spatial/sam3d_mesh.py` or `src/mlx_spatial/sam3d_decoder.py`, `src/mlx_spatial/sam3d_slat.py`, `tests/test_sam3d_mesh.py`, `tests/test_sam3d_decoder.py`
**Context budget:** ~10% of context window
**Produces:** `run_sam3d_mesh_decoder_features` or equivalent that returns final sparse mesh feature rows and subdivision metadata before surface extraction.
**Acceptance criteria:**
- Sparse subdivide expands every cube coord to its eight children and duplicates feature rows in official corner order.
- `SparseSubdivideBlock3d` fixture matches expected shape and skip behavior when channels change.
- Group norm handles channel counts below 32 and uses official epsilon/affine behavior.
- Mesh decoder feature trace records token counts after each subdivision.
**Verification:** `uv run pytest -q tests/test_sam3d_mesh.py tests/test_sam3d_decoder.py`
**Auto-continue:** yes

### Slice 3: SparseFeatures2Mesh Field Assembly

**Objective:** Convert sparse mesh feature rows into official-layout dense SDF, deform, weight, and optional color fields with guards.
**Execution:** subagent recommended
**Depends on:** Slice 2
**Touches:** `src/mlx_spatial/sam3d_mesh.py`, `tests/test_sam3d_mesh.py`, `tests/test_sam3d_export.py`
**Context budget:** ~9% of context window
**Produces:** SparseFeatures2Mesh-style layout parser, sparse cube-to-vertex aggregation, dense field assembly, deformed grid vertex computation, and memory estimates.
**Acceptance criteria:**
- Feature channel layout matches official sizes: `sdf=8`, `deform=24`, `weights=21`, optional `color=48`.
- `sparse_cube2verts` deterministic fixture averages shared cube-corner attributes correctly.
- Dense SDF initializes missing vertices outside with `1` and applies `sdf_bias = -1 / extraction_resolution`.
- Dense weights initialize missing cubes to zero and deformed vertices follow upstream `grid/res - 0.5 + tanh(deform)` math.
- Guard overrun returns a structured blocker object and does not write an artifact.
**Verification:** `uv run pytest -q tests/test_sam3d_mesh.py tests/test_sam3d_export.py`
**Auto-continue:** yes

### Slice 4: FlexiCubes Surface Core

**Objective:** Port the inference-only FlexiCubes surface-core math needed before triangulation: surface cubes, surface edges, case ids, weight activations, and dual vertex candidates.
**Execution:** subagent recommended
**Depends on:** Slice 3
**Touches:** `src/mlx_spatial/sam3d_mesh.py`, optional `src/mlx_spatial/sam3d_flexicubes_tables.py`, `tests/test_sam3d_mesh.py`
**Context budget:** ~12% of context window
**Produces:** NumPy/MLX FlexiCubes core primitives with official tables converted to repo-local constants.
**Acceptance criteria:**
- Surface cube detection matches sign-change semantics on tiny dense SDF fixtures.
- Surface edge identification deduplicates shared grid edges and marks cube edge maps.
- Weight activations match official inference rules: `beta=tanh()*0.99+1`, `alpha=tanh()*0.99+1`, `gamma=sigmoid()*0.99+0.005`.
- Case id computation uses the official check table and handles non-ambiguous fixtures exactly.
- A tiny fixture produces non-empty dual vertex candidates without triangulation.
**Verification:** `uv run pytest -q tests/test_sam3d_mesh.py`
**Auto-continue:** yes

### Slice 5: FlexiCubes Triangulation And Mesh Result

**Objective:** Complete inference triangulation so dense fields produce vertices, faces, optional vertex colors, and mesh stats.
**Execution:** subagent recommended
**Depends on:** Slice 4
**Touches:** `src/mlx_spatial/sam3d_mesh.py`, optional `src/mlx_spatial/sam3d_flexicubes_tables.py`, `src/mlx_spatial/sam3d_export.py`, `tests/test_sam3d_mesh.py`, `tests/test_sam3d_export.py`
**Context budget:** ~13% of context window
**Produces:** `Sam3dMeshExtractResult` with non-empty vertices/faces for deterministic fields and optional color arrays suitable for the basic GLB writer.
**Acceptance criteria:**
- Tiny deterministic SDF field extracts a non-empty triangle mesh.
- Faces reference valid vertices and vertices are finite float32.
- Optional colors are sigmoid-activated and aligned with mesh vertices.
- Empty/no-surface fields return a structured `mesh-decoder` blocker rather than an empty GLB.
- The basic GLB writer accepts extracted mesh output without special casing.
**Verification:** `uv run pytest -q tests/test_sam3d_mesh.py tests/test_sam3d_export.py`
**Auto-continue:** yes

### Slice 6: Reconstruct Integration And Trace

**Objective:** Replace the `mesh-decoder` blocker in `reconstruct` with the real mesh decoder, extraction, GLB export, and trace metadata.
**Execution:** direct
**Depends on:** Slice 5
**Touches:** `src/mlx_spatial/sam3d_inference.py`, `src/mlx_spatial/sam3d.py`, `tests/test_sam3d_tools.py`, `tests/test_sam3d_decoder.py`, `tests/test_sam3d_export.py`
**Context budget:** ~8% of context window
**Produces:** CLI path that writes both official-field `gaussians.ply` and basic `mesh.glb` when `--glb-output` is provided.
**Acceptance criteria:**
- Gaussian-only command still exits `0` and does not require mesh tensors.
- GLB command loads `slat_decoder_mesh`, runs mesh decode/extract/export, and appends `mesh-decoder` and `glb-export` to completed stages.
- Trace records input SLat shape, subdivision token counts, extraction resolution, memory estimates, vertex count, face count, GLB bytes, and blocker details when blocked.
- Tests prove requested GLB never falls back to point, cube-preview, or Gaussian-derived geometry.
**Verification:** `uv run pytest -q tests/test_sam3d_tools.py tests/test_sam3d_decoder.py tests/test_sam3d_export.py tests/test_sam3d_gaussian.py`
**Auto-continue:** no

### Slice 7: Live Acceptance And Regression

**Objective:** Run the official sample through `gaussians.ply + mesh.glb`, verify Blender import when available, and run regressions.
**Execution:** direct
**Depends on:** Slice 6
**Touches:** `outputs/sam3d/human-object/`, `.agent/.automaton/state/current.json`, `.agent/steering/STATUS.md`
**Context budget:** ~6% of context window
**Produces:** Fresh live artifacts and verification evidence for the active change.
**Acceptance criteria:**
- Live command writes non-empty `outputs/sam3d/human-object/gaussians.ply`.
- Live command writes non-empty `outputs/sam3d/human-object/mesh.glb`.
- Trace reaches `glb-export` with no blocker and nonzero mesh vertex/face counts.
- Blender headless import passes if `blender` is installed; if not installed, GLB header/container tests still pass and the missing Blender binary is recorded.
- Focused SAM3D tests, full pytest, and diff check pass.
**Verification:** `uv run mlx-spatial-sam3d reconstruct weights/sam-3d-objects-mlx vendors/sam-3d-objects/notebook/images/human_object/image.png --mask vendors/sam-3d-objects/notebook/images/human_object/0.png --moge-root weights/moge-vitl-mlx --output outputs/sam3d/human-object/gaussians.ply --glb-output outputs/sam3d/human-object/mesh.glb --seed 42 --memory-profile large --trace-output outputs/sam3d/human-object/trace.json && uv run pytest -q tests/test_sam3d_assets.py tests/test_sam3d_condition.py tests/test_sam3d_decoder.py tests/test_sam3d_export.py tests/test_sam3d_gaussian.py tests/test_sam3d_mesh.py tests/test_sam3d_tools.py && uv run pytest -q && git diff --check`
**Auto-continue:** no

## Execution Routing And Topology

- Serial dependency chain: Slice 1 -> Slice 2 -> Slice 3 -> Slice 4 -> Slice 5 -> Slice 6 -> Slice 7.
- Subagent recommended slices: 2, 3, 4, and 5 because they touch sparse math, cross module boundaries, and need focused implementation/review loops.
- Direct slices: 1, 6, and 7 because they are contract/refactor, orchestration, or verification work.
- Parallel-safe groups: none. The slices share mesh decoder files and should not be edited concurrently.
- Auto-continue chain: Slices 1 through 5 may continue after their verification passes. Slice 6 is a checkpoint before live generation because integration failures may require plan adjustment. Slice 7 is final verification.
- Review topology for subagent slices: implementer -> spec reviewer -> quality reviewer, using the per-slice acceptance criteria and commands above.

## Verification Commands

- Slice 1: `uv run pytest -q tests/test_sam3d_decoder.py tests/test_sam3d_contract.py tests/test_sam3d_gaussian.py`
- Slice 2: `uv run pytest -q tests/test_sam3d_mesh.py tests/test_sam3d_decoder.py`
- Slice 3: `uv run pytest -q tests/test_sam3d_mesh.py tests/test_sam3d_export.py`
- Slice 4: `uv run pytest -q tests/test_sam3d_mesh.py`
- Slice 5: `uv run pytest -q tests/test_sam3d_mesh.py tests/test_sam3d_export.py`
- Slice 6: `uv run pytest -q tests/test_sam3d_tools.py tests/test_sam3d_decoder.py tests/test_sam3d_export.py tests/test_sam3d_gaussian.py`
- Slice 7:
  ```bash
  uv run mlx-spatial-sam3d reconstruct weights/sam-3d-objects-mlx vendors/sam-3d-objects/notebook/images/human_object/image.png --mask vendors/sam-3d-objects/notebook/images/human_object/0.png --moge-root weights/moge-vitl-mlx --output outputs/sam3d/human-object/gaussians.ply --glb-output outputs/sam3d/human-object/mesh.glb --seed 42 --memory-profile large --trace-output outputs/sam3d/human-object/trace.json
  uv run pytest -q tests/test_sam3d_assets.py tests/test_sam3d_condition.py tests/test_sam3d_decoder.py tests/test_sam3d_export.py tests/test_sam3d_gaussian.py tests/test_sam3d_mesh.py tests/test_sam3d_tools.py
  uv run pytest -q
  git diff --check
  ```

## Context Budget For This Change

Estimated total implementation context is ~64% if done in one continuous session. The safer route is slice-by-slice execution with concise orchestration artifacts:

- Slice 1: ~6%
- Slice 2: ~10%
- Slice 3: ~9%
- Slice 4: ~12%
- Slice 5: ~13%
- Slice 6: ~8%
- Slice 7: ~6%

If FlexiCubes triangulation grows beyond Slice 5's budget, split it into table/case-id extraction and triangulation/export without changing the acceptance target.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan follows the current SAM3D dataflow cleanly from SLat through mesh decoder features, SparseFeatures2Mesh, FlexiCubes-style extraction, and basic GLB export without broadening the CLI or adding forbidden runtime dependencies.
- Concern: The FlexiCubes triangulation port and dense extraction guards remain high-risk because they combine large-memory arrays, official table semantics, and nontrivial surface topology in the same execution path.
- Action: Start `auto-execute` at Slice 1 and keep Slices 4 and 5 as hard checkpoints if FlexiCubes surface extraction needs to be split further.
- Verified: Checked active state, STATUS, PLAN, DESIGN, slice routing, acceptance criteria, verification commands, strict no-fallback constraints, and memory-guard coverage.
