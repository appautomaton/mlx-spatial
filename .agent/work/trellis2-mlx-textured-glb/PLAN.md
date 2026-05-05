# PLAN: TRELLIS.2 MLX Textured GLB

## Goal

Extend the current MLX TRELLIS.2 shape pipeline so an RGB or RGBA image can produce a real textured `.glb` asset through MLX inference and Mac-native export surfaces, with structured blockers instead of fake outputs when an exact stage is not yet implemented.

## Architecture Approach

Build the textured path on top of the working `generate-shape` stages. Reuse the existing image preprocessing, DINOv3 conditioning, sparse structure, shape SLat, shape decoder, and FlexiDualGrid mesh extraction as the geometry base. Add a separate `generate-textured` command that runs texture SLat, texture decoder, mesh/voxel texture baking, and GLB export. The command must either write a Blender-readable textured GLB or return a structured blocker with the deepest completed stage.

Runtime model compute stays in MLX. NumPy/Pillow may handle sparse/index/image assembly. GLB/UV/export can use a small internal writer or a lightweight Mac-native dependency only after an explicit dependency decision.

## Ordered Task Sequence

### Slice 1: Texture Pipeline Contract Discovery

**Objective:** Map the upstream TRELLIS.2 texture route, checkpoints, decoder inputs, MeshWithVoxel coupling, and GLB/export expectations into repo-local contracts.
**Execution:** direct
**Depends on:** none
**Touches:** `.agent/work/trellis2-mlx-textured-glb/`, `README.md` if docs need a contract note, texture-related tests as needed
**Context budget:** ~8%
**Produces:** A concise contract note in this work folder plus tests for discovered texture route metadata.
**Acceptance criteria:**
- Identifies texture SLat model keys and texture decoder checkpoint/config paths for `512`, `1024`, `1024_cascade`, and `1536_cascade`.
- Identifies which shape outputs guide texture decoding and baking.
- Identifies the smallest viable GLB/UV/export strategy for the repo.
- Adds or updates tests that fail if expected texture route metadata is absent.
**Verification:** `uv run pytest -q tests/test_trellis2_slat.py tests/test_trellis2_decode.py tests/test_trellis2_tools.py`
**Auto-continue:** no

### Slice 2: `generate-textured` Command And Structured Blockers

**Objective:** Add the user-facing command, output path policy, trace/result types, and early blockers before expensive texture compute.
**Execution:** direct
**Depends on:** Slice 1
**Touches:** `src/mlx_spatial/trellis2.py`, `src/mlx_spatial/trellis2_inference.py`, `src/mlx_spatial/trellis2_export.py`, CLI/inference tests
**Context budget:** ~10%
**Produces:** `generate-textured` accepts `.glb` under `outputs/`, rejects invalid output paths, and stops at the first unimplemented texture stage with trace metadata.
**Acceptance criteria:**
- `.glb` is required for `generate-textured`; `.obj` remains the `generate-shape` output.
- Paths outside `outputs/` are rejected.
- Missing texture assets/configs produce precise blockers before model execution.
- Existing `generate-shape` behavior is unchanged.
**Verification:** `uv run pytest -q tests/test_trellis2_tools.py tests/test_trellis2_inference.py tests/test_trellis2_export.py`
**Auto-continue:** yes

### Slice 3: Exact Texture SLat Execution

**Objective:** Run the MLX texture SLat branch from real sparse/shape state using upstream route semantics and resource guards.
**Execution:** subagent recommended
**Depends on:** Slice 2
**Touches:** `src/mlx_spatial/trellis2_slat.py`, `src/mlx_spatial/trellis2_inference.py`, SLat tests
**Context budget:** ~14%
**Produces:** Texture SLat coordinates/features and route metadata, or a structured blocker for unsupported route/token guards.
**Acceptance criteria:**
- Texture SLat route selection matches upstream pipeline type semantics.
- Texture SLat consumes real shape SLat/shape coordinates from the current run.
- Exact mode does not use synthetic texture inputs or silent approximate fallbacks.
- Trace reports texture token counts, conditioning resolution, and model route.
**Verification:** `uv run pytest -q tests/test_trellis2_slat.py tests/test_trellis2_inference.py`
**Auto-continue:** no

### Slice 4: Texture Decoder To Concrete Texture Representation

**Objective:** Decode texture SLat through the MLX texture decoder into a concrete texture-side representation suitable for surface baking.
**Execution:** subagent recommended
**Depends on:** Slice 3
**Touches:** `src/mlx_spatial/trellis2_decode.py`, `src/mlx_spatial/trellis2_inference.py`, decoder tests
**Context budget:** ~15%
**Produces:** Texture decoder output with shape/guide metadata, or a structured blocker naming the exact missing decoder operation.
**Acceptance criteria:**
- Loads and validates texture decoder checkpoint/config keys.
- Runs `from_latent` and decoder levels needed for texture output under configured guards.
- Uses shape decoder subdivision/guide metadata where upstream requires it.
- Trace reports texture decoder output shapes and completed decoder levels.
**Verification:** `uv run pytest -q tests/test_trellis2_decode.py tests/test_trellis2_inference.py`
**Auto-continue:** no

### Slice 5: Mesh/Voxel Coupling And Baking Fixtures

**Objective:** Implement the Mac-native mesh/voxel texture mapping surface with deterministic fake texture fixtures before live GLB export.
**Execution:** subagent recommended
**Depends on:** Slice 4
**Touches:** new or existing mesh/texture/export modules, `tests/test_trellis2_export.py`, new fixture tests
**Context budget:** ~14%
**Produces:** Deterministic mesh plus texture-field fixture can bake an image/atlas or equivalent surface color payload without model weights.
**Acceptance criteria:**
- Fixture mesh receives non-empty UVs or an equivalent GLB-compatible texture mapping.
- Baking uses texture decoder-style fields, not constant colors.
- Texture image/payload is non-empty and deterministic.
- No PyTorch/CUDA runtime dependency is introduced.
**Verification:** `uv run pytest -q tests/test_trellis2_export.py`
**Auto-continue:** no

### Slice 6: GLB Writer And Blender Fixture Verification

**Objective:** Write a valid textured GLB from deterministic fixture geometry and texture data, then verify it in Blender.
**Execution:** subagent recommended
**Depends on:** Slice 5
**Touches:** GLB/export module, export tests, optional dependency metadata if approved
**Context budget:** ~12%
**Produces:** A fixture-generated `.glb` under a temp/output path with mesh and material/texture data.
**Acceptance criteria:**
- GLB contains mesh geometry, indices/accessors/buffers, material, and image/texture data.
- Blender headless import reports at least one mesh and at least one material or image texture.
- Export path policy remains limited to `outputs/` for user commands.
**Verification:** `uv run pytest -q tests/test_trellis2_export.py && blender --background --python-expr \"import bpy; p='outputs/trellis2/fixture-textured.glb'; bpy.ops.import_scene.gltf(filepath=p); print('GLB_OK', len([o for o in bpy.context.scene.objects if o.type=='MESH']), len(bpy.data.materials), len(bpy.data.images))\"`
**Auto-continue:** no

### Slice 7: End-To-End Textured Generation Integration

**Objective:** Connect image preprocessing, shape generation, texture SLat, texture decoder, baking, and GLB export in `generate-textured`.
**Execution:** subagent recommended
**Depends on:** Slice 6
**Touches:** `src/mlx_spatial/trellis2_inference.py`, `src/mlx_spatial/trellis2.py`, texture/export modules, integration tests
**Context budget:** ~15%
**Produces:** `generate-textured` can write a real textured GLB from fixtures and can run live until success or a precise blocker.
**Acceptance criteria:**
- Fake TRELLIS fixture runs full command path and writes a non-empty textured GLB.
- Live command uses real image preprocessing and real shape path before texture stages.
- Blockers include deepest completed stage and texture/decoder/export metadata.
- Existing shape OBJ command still passes.
**Verification:** `uv run pytest -q tests/test_trellis2_inference.py tests/test_trellis2_export.py && uv run pytest -q`
**Auto-continue:** no

### Slice 8: Live 512 Textured GLB Verification

**Objective:** Run the real local 512 pipeline and verify the generated GLB in Blender.
**Execution:** direct
**Depends on:** Slice 7
**Touches:** `outputs/trellis2/`, verification notes
**Context budget:** ~8%
**Produces:** `outputs/trellis2/image-textured.glb` or a final structured blocker report naming the next implementation gap.
**Acceptance criteria:**
- Live command either writes a non-empty textured GLB or returns a precise blocker accepted as the next implementation slice.
- On success, Blender imports the GLB and reports mesh plus material/texture data.
- On success, current shape OBJ live command still works.
- Verification evidence is recorded in this work folder.
**Verification:** `uv run mlx-spatial-trellis2 generate-textured weights/trellis2 inputs/trellis2/image.png --output outputs/trellis2/image-textured.glb --pipeline-type 512 --seed 42 --rmbg-root weights/rmbg2 && blender --background --python-expr \"import bpy; p='outputs/trellis2/image-textured.glb'; bpy.ops.import_scene.gltf(filepath=p); print('GLB_OK', len([o for o in bpy.context.scene.objects if o.type=='MESH']), len(bpy.data.materials), len(bpy.data.images))\"`
**Auto-continue:** no

## Execution Routing And Topology

- Slices 1 and 2 are direct and may be done in one serial pass after verification.
- Slices 3 through 7 are subagent-recommended because they cross model, decoder, export, and command boundaries.
- Slice 8 is direct verification.
- Auto-continue chain: Slice 2 may continue into Slice 3 only if all blockers are command-surface blockers and no dependency decision remains.
- Checkpoints: Slices 1, 3, 4, 5, 6, 7, and 8 are checkpoints.
- Parallel-safe groups: none by default. Texture SLat and GLB writer work may become parallel-safe only after Slice 1 fixes disjoint write sets and dependency choices.

## Verification Commands

- Slice 1: `uv run pytest -q tests/test_trellis2_slat.py tests/test_trellis2_decode.py tests/test_trellis2_tools.py`
- Slice 2: `uv run pytest -q tests/test_trellis2_tools.py tests/test_trellis2_inference.py tests/test_trellis2_export.py`
- Slice 3: `uv run pytest -q tests/test_trellis2_slat.py tests/test_trellis2_inference.py`
- Slice 4: `uv run pytest -q tests/test_trellis2_decode.py tests/test_trellis2_inference.py`
- Slice 5: `uv run pytest -q tests/test_trellis2_export.py`
- Slice 6: `uv run pytest -q tests/test_trellis2_export.py` plus Blender fixture import.
- Slice 7: `uv run pytest -q tests/test_trellis2_inference.py tests/test_trellis2_export.py && uv run pytest -q`
- Slice 8: live `generate-textured` command plus Blender GLB import.

## Context Budget For This Change

Estimated total implementation context: ~96% spread across checkpointed sessions, with no single slice intended to exceed ~15%. Execution should preserve this plan and stop at structured blockers rather than broadening scope inside a slice.

## Recommended Review

Run `auto-eng-review` before execution. This plan intentionally selects the high-risk end-to-end path, touches learned model execution and export semantics, and may introduce a GLB/UV dependency decision.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan keeps the textured GLB path grounded in the working shape pipeline and divides model inference, decoder output, baking, GLB writing, and live verification into checkpointed slices.
- Concern: The end-to-end target still carries material risk around texture decoder guide semantics, MeshWithVoxel equivalence, UV or atlas strategy, and GLB dependency choice.
- Action: Start with Slice 1 and make the GLB/UV dependency decision explicit before implementing Slice 2 or any export writer.
- Verified: PLAN.md and DESIGN.md were checked for slice order, data flow, runtime boundaries, blocker handling, verification commands, and dependency risk.
