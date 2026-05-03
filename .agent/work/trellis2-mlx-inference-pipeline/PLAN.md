# PLAN: TRELLIS.2 MLX Inference Pipeline

## Goal

Implement a blocker-driven, MLX-first TRELLIS.2 image-to-3D inference pipeline from real image input through DINOv3 conditioning, sparse structure sampling/decoding, SLat shape and texture sampling/decoding, and final mesh/export boundary, producing a real artifact when feasible or the first exact blocker at each remaining stage.

## Architecture Approach

Keep `Trellis2InferencePipeline.attempt_forward_trace(...)` as the single runtime spine. Add stage-specific MLX helpers in small modules, wire one stage at a time, and require each real stage to either run from real upstream tensors or return a precise `Trellis2ForwardBlocker`. Default tests use fake configs/checkpoints; explicit local real-weight probes are allowed only as verification commands outside default fixture paths.

## Ordered Task Sequence

### Slice 1: Full Pipeline Contract Discovery

**Objective:** Parse the complete TRELLIS.2 pipeline contract from local config and vendored source into typed stage metadata.
**Execution:** direct
**Depends on:** none
**Touches:** `src/mlx_spatial/trellis2_forward.py`, `tests/test_trellis2_forward.py`, `README.md`
**Context budget:** ~8% of context window
**Produces:** Typed config discovery for sparse flow, sparse decoder, shape SLat, shape decoder, texture SLat, texture decoder, sampler configs, normalization values, and default pipeline type.
**Acceptance criteria:**
- Fake `pipeline.json` tests validate all required model keys and sampler fields.
- Missing or malformed fields produce stage-specific blockers.
- Existing DINOv3 conditioning discovery behavior is preserved.
**Verification:** `uv run pytest tests/test_trellis2_forward.py tests/test_trellis2_inference.py`
**Auto-continue:** yes

### Slice 2: Sparse FlowEuler And Sparse Flow Skeleton

**Objective:** Implement the sparse structure sampler contract, FlowEuler guidance interval schedule, noise shape validation, and first MLX sparse flow module skeleton.
**Execution:** subagent recommended
**Depends on:** Slice 1
**Touches:** `src/mlx_spatial/trellis2_sparse_structure.py`, `src/mlx_spatial/trellis2_forward.py`, `tests/test_trellis2_sparse_structure.py`, `tests/test_trellis2_forward.py`
**Context budget:** ~14% of context window
**Produces:** Fake-fixture sparse sampling path that validates `(B, 8, 16, 16, 16)` noise, conditioning width, timestep schedule, CFG interval behavior, and checkpoint key groups.
**Acceptance criteria:**
- The previous generic `MLX sparse structure flow model construction` blocker is replaced by either sampled sparse metadata in fake mode or a more precise first real sparse-flow blocker.
- Tests cover timestep rescaling, guidance interval boundaries, missing checkpoint groups, shape mismatch, and dtype reporting.
**Verification:** `uv run pytest tests/test_trellis2_sparse_structure.py tests/test_trellis2_forward.py`
**Auto-continue:** no

### Slice 3: Real Sparse Structure Flow Forward Attempt

**Objective:** Load the selected sparse flow checkpoint tensors and execute as much of `SparseStructureFlowModel` as MLX supports before stopping at the first exact model/op/memory blocker.
**Execution:** subagent recommended
**Depends on:** Slice 2
**Touches:** `src/mlx_spatial/trellis2_sparse_structure.py`, `src/mlx_spatial/trellis2_forward.py`, `tests/test_trellis2_sparse_structure.py`, `README.md`
**Context budget:** ~15% of context window
**Produces:** Real-weight sparse flow attempt integrated after DINOv3 conditioning.
**Acceptance criteria:**
- The real alpha forward trace completes DINOv3 and reaches a blocker more specific than sparse flow construction, or produces real sparse flow sample metadata.
- Blocker names the exact unsupported transformer component, checkpoint key, MLX op, dtype, shape, or memory boundary.
- Default tests still avoid loading the full checkpoint.
**Verification:** `uv run pytest tests/test_trellis2_sparse_structure.py tests/test_trellis2_forward.py && uv run python -m mlx_spatial.trellis2 attempt-forward-trace weights/trellis2 inputs/trellis2/demo-alpha.webp --dino-root weights/dinov3-vitl16-pretrain-lvd1689m`
**Auto-continue:** no

### Slice 4: Sparse Structure Decoder Boundary

**Objective:** Implement or precisely block sparse structure decoder execution from real sampled sparse latents into sparse coordinates.
**Execution:** subagent recommended
**Depends on:** Slice 3
**Touches:** `src/mlx_spatial/trellis2_sparse_structure.py`, `src/mlx_spatial/trellis2_forward.py`, `tests/test_trellis2_sparse_structure.py`, `tests/test_trellis2_forward.py`
**Context budget:** ~12% of context window
**Produces:** Decoder key-map/config parser, fake decoder fixture, and real decoder attempt/blocker.
**Acceptance criteria:**
- Real path only appends sparse coordinate output after decoder execution succeeds.
- Decoder blockers distinguish checkpoint, sparse op, thresholding, coordinate extraction, and memory issues.
- Fake tests validate coordinate shape and ordering contracts.
**Verification:** `uv run pytest tests/test_trellis2_sparse_structure.py tests/test_trellis2_forward.py`
**Auto-continue:** no

### Slice 5: Shape SLat Sampling Boundary

**Objective:** Implement or precisely block shape SLat flow sampling for the configured `512`, `1024`, or `1024_cascade` path.
**Execution:** subagent recommended
**Depends on:** Slice 4
**Touches:** `src/mlx_spatial/trellis2_slat.py`, `src/mlx_spatial/trellis2_forward.py`, `src/mlx_spatial/trellis2_inference.py`, `tests/test_trellis2_slat.py`, `tests/test_trellis2_forward.py`
**Context budget:** ~15% of context window
**Produces:** Shape SLat config/key-map/sampler contract, fake sparse-coordinate conditioned sampling fixture, and real attempt/blocker.
**Acceptance criteria:**
- Shape SLat stage consumes real sparse coordinates or stops before it.
- Cascade routing is explicit and tested.
- Blockers name exact shape-flow checkpoint groups, sparse-coordinate conditioning, sampler behavior, op gap, dtype, shape, or memory issue.
**Verification:** `uv run pytest tests/test_trellis2_slat.py tests/test_trellis2_forward.py tests/test_trellis2_inference.py`
**Auto-continue:** no

### Slice 6: Texture SLat Sampling Boundary

**Objective:** Implement or precisely block texture SLat sampling conditioned on real image features and real `shape_slat`, before any decoder stage.
**Execution:** subagent recommended
**Depends on:** Slice 5
**Touches:** `src/mlx_spatial/trellis2_slat.py`, `src/mlx_spatial/trellis2_forward.py`, `tests/test_trellis2_slat.py`, `tests/test_trellis2_forward.py`
**Context budget:** ~12% of context window
**Produces:** Texture SLat config/key-map/sampler contract, fake `shape_slat` conditioned sampling fixture, and real texture SLat attempt/blocker.
**Acceptance criteria:**
- Texture SLat stage consumes real `shape_slat` or stops before it.
- Blockers identify exact texture-flow checkpoint groups, shape-SLat conditioning, sampler behavior, op gap, dtype, shape, or memory issue.
**Verification:** `uv run pytest tests/test_trellis2_slat.py tests/test_trellis2_forward.py`
**Auto-continue:** no

### Slice 7: Decode Latents Shape And Texture Boundary

**Objective:** Implement or precisely block TRELLIS.2 `decode_latent(...)` behavior that decodes shape and texture from real `shape_slat`, real `tex_slat`, and resolution.
**Execution:** subagent recommended
**Depends on:** Slice 6
**Touches:** `src/mlx_spatial/trellis2_decode.py`, `src/mlx_spatial/trellis2_forward.py`, `tests/test_trellis2_decode.py`, `tests/test_trellis2_forward.py`
**Context budget:** ~12% of context window
**Produces:** Shape decoder and texture decoder parsers/key maps, fake decoded-latent fixture, and real decoder attempt/blocker.
**Acceptance criteria:**
- Real path appends decoded shape/texture output only after decoder execution succeeds.
- Blockers separate shape decoder construction, texture decoder construction, sparse tensor layout, iso-surface/mesh extraction, texture voxel guidance, UV/material handling, dtype, shape, and memory issues.
**Verification:** `uv run pytest tests/test_trellis2_decode.py tests/test_trellis2_forward.py`
**Auto-continue:** no

### Slice 8: Export Boundary, Docs, And Final Verification

**Objective:** Implement final artifact metadata/export handling and close the spec with current real boundary evidence.
**Execution:** direct
**Depends on:** Slice 7
**Touches:** `src/mlx_spatial/trellis2_export.py`, `src/mlx_spatial/trellis2_forward.py`, `src/mlx_spatial/trellis2_inference.py`, `tests/test_trellis2_export.py`, `README.md`, `.agent/work/trellis2-mlx-inference-pipeline/VERIFY.md`
**Context budget:** ~10% of context window
**Produces:** Export result metadata, output path policy, final docs, and verification artifact.
**Acceptance criteria:**
- If a real mesh/texture artifact exists, it is written only under ignored `outputs/` and reported with structured metadata.
- If export is blocked, the blocker names the exact mesh/export dependency or format issue.
- README and `VERIFY.md` report the deepest real completed stage and current blocker.
**Verification:** `uv run pytest && uv run python -m mlx_spatial.trellis2 attempt-forward-trace weights/trellis2 inputs/trellis2/demo-alpha.webp --dino-root weights/dinov3-vitl16-pretrain-lvd1689m`
**Auto-continue:** no

## Execution Routing And Topology

- Topology: serial checkpoint chain. Each slice depends on the previous slice because downstream stages require real upstream outputs or exact blockers.
- Auto-continue chain: Slice 1 may continue into Slice 2 after verification. All later slices are checkpoints because real-weight blockers may change the next implementation boundary.
- Parallel-safe groups: none. The runtime spine and dataclass contracts are shared.
- Recommended route: run `auto-eng-review` before `auto-execute` because this plan crosses shared trace contracts, new modules, real checkpoint loading, and multiple decoder/export boundaries.

## Verification Commands

- Slice 1: `uv run pytest tests/test_trellis2_forward.py tests/test_trellis2_inference.py`
- Slice 2: `uv run pytest tests/test_trellis2_sparse_structure.py tests/test_trellis2_forward.py`
- Slice 3: `uv run pytest tests/test_trellis2_sparse_structure.py tests/test_trellis2_forward.py && uv run python -m mlx_spatial.trellis2 attempt-forward-trace weights/trellis2 inputs/trellis2/demo-alpha.webp --dino-root weights/dinov3-vitl16-pretrain-lvd1689m`
- Slice 4: `uv run pytest tests/test_trellis2_sparse_structure.py tests/test_trellis2_forward.py`
- Slice 5: `uv run pytest tests/test_trellis2_slat.py tests/test_trellis2_forward.py tests/test_trellis2_inference.py`
- Slice 6: `uv run pytest tests/test_trellis2_slat.py tests/test_trellis2_forward.py`
- Slice 7: `uv run pytest tests/test_trellis2_decode.py tests/test_trellis2_forward.py`
- Slice 8: `uv run pytest && uv run python -m mlx_spatial.trellis2 attempt-forward-trace weights/trellis2 inputs/trellis2/demo-alpha.webp --dino-root weights/dinov3-vitl16-pretrain-lvd1689m`

## Context Budget For This Change

Estimated total: ~98% of a full context window if executed as one continuous chain. Execution should checkpoint after each slice, and any slice that ends with a real blocker should pause for review before continuing deeper.

## Review: Engineering

- Verdict: needs_correction
- Strength: The plan preserves the existing blocker-driven `attempt_forward_trace(...)` spine and decomposes the broad TRELLIS.2 pipeline into serial, testable checkpoints.
- Concern: Slice 3 and Slice 8 use the non-existent CLI subcommand `attempt-forward` instead of the implemented `attempt-forward-trace`, so their verification commands would fail before testing the intended runtime boundary.
- Action: Run `auto-plan` to correct the real-weight verification commands and recheck the stage order before execution.
- Verified: state, status, SPEC, DESIGN, PLAN, CLI subcommand inventory, and inference stage order checked.

## Plan Correction

- Corrected Slice 3 and Slice 8 verification commands to use `attempt-forward-trace weights/trellis2 inputs/trellis2/demo-alpha.webp --dino-root weights/dinov3-vitl16-pretrain-lvd1689m`, matching the implemented CLI positional arguments.

## Review: Engineering

- Verdict: needs_correction
- Strength: The corrected plan now uses the implemented `attempt-forward-trace` CLI shape and keeps verification grounded in existing test and trace surfaces.
- Concern: Slice 6 and Slice 7 invert the TRELLIS.2 data flow because vendored `run(...)` samples texture SLat from `shape_slat` before `decode_latent(...)` decodes shape and texture together, while the plan and design make texture SLat depend on a decoded shape boundary.
- Action: Run `auto-plan` to reorder the SLat and decoder slices so texture SLat sampling follows shape SLat sampling and both decoders follow texture SLat.
- Verified: state, status, SPEC, DESIGN, corrected PLAN, CLI help, vendored TRELLIS.2 run order, and current inference stage order checked.

## Plan Correction

- Reordered Slice 6 and Slice 7 to match vendored TRELLIS.2 `run(...)`: texture SLat sampling now follows shape SLat sampling, and combined shape/texture decode now follows texture SLat.
- Updated the DESIGN data contract so texture SLat depends on image features and `shape_slat`, not a decoded shape or mesh boundary.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The corrected plan now preserves the existing blocker-driven runtime spine, uses valid verification commands, and matches TRELLIS.2 latent ordering through texture SLat and combined decode.
- Concern: Real sparse flow, SLat, decoder, and export implementation may still stop at MLX op, memory, dtype, or checkpoint-layout blockers, but the plan explicitly treats those as first-class outcomes.
- Action: Run `auto-execute` starting with Slice 1.
- Verified: state, status, SPEC, DESIGN, corrected PLAN, CLI help, vendored TRELLIS.2 run order, and verification command coverage checked.

## Execution Evidence

- Slice 1 completed by direct route: `discover_trellis2_conditioning_config(...)` now parses the full TRELLIS.2 model, sampler, normalization, and default pipeline contract while preserving existing DINOv3 conditioning discovery.
- Slice 1 verification passed: `uv run pytest tests/test_trellis2_forward.py tests/test_trellis2_inference.py` reported `28 passed`.
- Slice 2 completed by direct route: added sparse-structure FlowEuler schedule, noise-shape validation, fake sampling metadata, and a sharper real sparse boundary blocker.
- Slice 2 verification passed: `uv run pytest tests/test_trellis2_sparse_structure.py tests/test_trellis2_forward.py` reported `22 passed`.
- Full-suite check passed after Slice 2: `uv run pytest` reported `126 passed, 5 skipped`.
- After Slice 2, the real alpha trace blocked at `sparse-structure-sampling` / `MLX sparse structure transformer stack forward`.
- Execution window stopped after Slice 2 because Slice 2 has `Auto-continue: no`.
- Slice 3 completed by direct route: added a real sparse-flow forward probe that loads selected `input_layer` tensors, validates block-0 checkpoint keys, executes the MLX sparse input projection, and stops at the first unported `ModulatedTransformerCrossBlock`.
- Slice 3 targeted verification passed: `uv run pytest tests/test_trellis2_sparse_structure.py tests/test_trellis2_forward.py` reported `25 passed`.
- Full-suite check passed after Slice 3: `uv run pytest` reported `129 passed, 5 skipped`.
- Slice 3 real alpha trace passed: `uv run python -m mlx_spatial.trellis2 attempt-forward-trace weights/trellis2 inputs/trellis2/demo-alpha.webp --dino-root weights/dinov3-vitl16-pretrain-lvd1689m` completed through `image-conditioning` and now blocks at `sparse-structure-sampling` / `MLX sparse structure ModulatedTransformerCrossBlock forward`.
- Slice 3 live blocker reason confirms the real sparse input projection executed with output shape `(1, 4096, 1536)` before stopping at sparse 3D RoPE/shared-modulation block execution.
- Execution window stopped after Slice 3 because Slice 3 has `Auto-continue: no`.
- Slice 4 completed by direct route: added sparse structure decoder config parsing, decoder checkpoint key/shape probing, sparse decoder coordinate extraction, and a forward-level decoder boundary dispatcher that does not append coordinates unless decoder execution succeeds.
- Slice 4 targeted verification passed: `uv run pytest tests/test_trellis2_sparse_structure.py tests/test_trellis2_forward.py` reported `34 passed`.
- Full-suite check passed after Slice 4: `uv run pytest` reported `138 passed, 5 skipped`.
- Main alpha trace remains correctly gated at `sparse-structure-sampling` / `MLX sparse structure ModulatedTransformerCrossBlock forward`.
- Standalone decoder boundary reports `sparse-structure-decoding` / `sparse structure decoder config/checkpoint probe` because `weights/trellis2/microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.json` is not present locally.
- Execution window stopped after Slice 4 because Slice 4 has `Auto-continue: no`.
- Slice 5 completed by direct route: added SLat flow config parsing, shape-SLat pipeline route selection for `512`, `1024`, `1024_cascade`, and `1536_cascade`, real checkpoint key probing, sparse-coordinate validation, and first MLX sparse-feature input projection.
- Slice 5 targeted verification passed: `uv run pytest tests/test_trellis2_slat.py tests/test_trellis2_forward.py tests/test_trellis2_inference.py` reported `43 passed`.
- Full-suite check passed after Slice 5: `uv run pytest` reported `150 passed, 5 skipped`.
- Main alpha trace remains correctly gated at `sparse-structure-sampling` / `MLX sparse structure ModulatedTransformerCrossBlock forward`.
- Standalone shape-SLat boundary without coordinates reports `shape-slat-sampling` / `shape SLat upstream sparse coordinate availability` for the configured `1024_cascade` route.
- Standalone shape-SLat boundary with fake sparse coordinates loads real local `slat_flow_img2shape_dit_1_3B_512_bf16` tensors, executes input projection to `(2, 1536)`, and blocks at `MLX shape SLat ModulatedSparseTransformerCrossBlock forward`.
- Execution window stopped after Slice 5 because Slice 5 has `Auto-continue: no`.
- Slice 6 completed by direct route: added texture-SLat route selection, upstream `shape_slat` availability blocking, shape-SLat feature validation, concat noise+shape feature projection, and real texture-flow checkpoint probing.
- Slice 6 targeted verification passed: `uv run pytest tests/test_trellis2_slat.py tests/test_trellis2_forward.py` reported `41 passed`.
- Full-suite check passed after Slice 6: `uv run pytest` reported `159 passed, 5 skipped`.
- Main alpha trace remains correctly gated at `sparse-structure-sampling` / `MLX sparse structure ModulatedTransformerCrossBlock forward`.
- Standalone texture-SLat boundary without `shape_slat` reports `texture-slat-sampling` / `texture SLat upstream shape_slat availability` for the configured `1024_cascade` route.
- Standalone texture-SLat boundary with fake `shape_slat` loads real local `slat_flow_imgshape2tex_dit_1_3B_1024_bf16` tensors, concatenates `(2, 32)` noise and `(2, 32)` shape features to `(2, 64)`, executes input projection to `(2, 1536)`, and blocks at `MLX texture SLat ModulatedSparseTransformerCrossBlock forward`.
- Execution window stopped after Slice 6 because Slice 6 has `Auto-continue: no`.
- Slice 7 completed by direct route: added shape/texture latent decoder config parsing, decoder checkpoint key/shape probing, shape/texture SLat layout validation, from-latent projections, and combined `decode_latent(...)` blocker routing.
- Slice 7 targeted verification passed: `uv run pytest tests/test_trellis2_decode.py tests/test_trellis2_forward.py` reported `33 passed`.
- Full-suite check passed after Slice 7: `uv run pytest` reported `168 passed, 5 skipped`.
- Main alpha trace remains correctly gated at `sparse-structure-sampling` / `MLX sparse structure ModulatedTransformerCrossBlock forward`.
- Standalone decode boundary without latents reports upstream `shape_slat` or `texture_slat` availability blockers.
- Standalone decode boundary with fake `shape_slat` and `tex_slat` loads real local shape and texture decoder tensors, executes both `from_latent` projections to `(2, 1024)`, and blocks at `MLX shape latent decoder SparseConvNeXt/FlexiDualGrid forward`.
- Execution window stopped after Slice 7 because Slice 7 has `Auto-continue: no`.
- Slice 8 completed by direct route: added TRELLIS.2 export path validation, ignored-outputs artifact writing metadata, and export readiness/blocker assessment against forward traces.
- Slice 8 targeted verification passed: `uv run pytest tests/test_trellis2_export.py` reported `6 passed`.
- Full-suite check passed after Slice 8: `uv run pytest` reported `174 passed, 5 skipped`.
- Slice 8 real alpha trace passed: `uv run python -m mlx_spatial.trellis2 attempt-forward-trace weights/trellis2 inputs/trellis2/demo-alpha.webp --dino-root weights/dinov3-vitl16-pretrain-lvd1689m` completed `input-image`, `asset-config-validation`, `checkpoint-probe-readiness`, `image-preprocessing-background`, and `image-conditioning`, then blocked at `sparse-structure-sampling` / `MLX sparse structure ModulatedTransformerCrossBlock forward`.
- Export boundary now validates `.glb` and `.obj` output paths under ignored `outputs/` and reports `mesh-export` / `upstream inference completion before export` while the real trace is blocked upstream.
- Execution window stopped after Slice 8 because all planned slices are complete.
