# PLAN: TRELLIS.2 Forward Trace Conditioning

## Goal

Run the verified alpha-input TRELLIS.2 attempt forward in MLX using local `weights/trellis2`, replacing the current `image-conditioning` blanket blocker with either real conditioning output and the next downstream attempt or the first exact unsupported operation, module, config, or checkpoint-key blocker.

## Architecture Approach

- Add a focused forward-trace layer beside the existing readiness/attempt code instead of changing `dry_run(...)` semantics.
- Keep `attempt(...)` stable and add an explicit forward-trace entry point, likely `attempt_forward_trace(...)`, for aggressive compute probing.
- Represent forward-trace outputs separately from blockers so a stage can record tensor metadata before the next blocker.
- Split image-conditioning into config discovery, MLX image tensor preparation, DINOv3 asset/port assessment, and downstream sparse-structure boundary dispatch.
- Treat missing external DINOv3 assets as a concrete `image-conditioning` blocker, because `weights/trellis2/pipeline.json` references `facebook/dinov3-vitl16-pretrain-lvd1689m` outside the TRELLIS.2 checkpoint bundle.
- Use fake fixtures for default tests and real `weights/trellis2` only for ignored local evidence.

## Ordered Task Sequence

### Slice 1: Forward-Trace Contracts And Config Discovery

**Objective:** Add forward-trace result/output dataclasses and parse the TRELLIS.2 conditioning/downstream config needed for the trace.
**Execution:** direct
**Depends on:** none
**Touches:** `src/mlx_spatial/trellis2_forward.py`, `src/mlx_spatial/__init__.py`, `tests/test_trellis2_forward.py`
**Context budget:** ~8% of context window
**Produces:** public forward-trace dataclasses plus fake-fixture config discovery for `image_cond_model`, expected conditioning width, and first downstream sparse-flow config.
**Acceptance criteria:**
- Fake `pipeline.json` and checkpoint config fixtures expose `DinoV3FeatureExtractor`, `facebook/dinov3-vitl16-pretrain-lvd1689m`, default resolution choice, and `cond_channels`.
- Missing or malformed config returns a structured blocker rather than a JSON/key exception.
- Public exports are available from `mlx_spatial`.
**Verification:** `uv run pytest tests/test_trellis2_forward.py`
**Auto-continue:** yes

### Slice 2: Pipeline Forward-Trace Entry Point

**Objective:** Wire a new forward-trace attempt entry point after alpha preprocessing without changing existing attempt behavior.
**Execution:** direct
**Depends on:** Slice 1
**Touches:** `src/mlx_spatial/trellis2_inference.py`, `src/mlx_spatial/trellis2_forward.py`, `tests/test_trellis2_inference.py`, `tests/test_trellis2_forward.py`
**Context budget:** ~8% of context window
**Produces:** `Trellis2InferencePipeline` forward-trace method that completes preprocessing and enters `image-conditioning`.
**Acceptance criteria:**
- Existing `attempt(...)` tests still pass with the current `image-conditioning` blocker behavior.
- The forward-trace method records completed stages through `image-preprocessing-background`.
- Fake fixture blockers from the forward-trace layer propagate with the existing blocker shape.
- Invalid/missing image behavior remains unchanged.
**Verification:** `uv run pytest tests/test_trellis2_inference.py tests/test_trellis2_forward.py`
**Auto-continue:** yes

### Slice 3: MLX Image Tensor Preparation

**Objective:** Convert the preprocessed PIL image into the normalized MLX image tensor expected by the DINOv3 conditioning path.
**Execution:** direct
**Depends on:** Slice 2
**Touches:** `src/mlx_spatial/trellis2_forward.py`, `tests/test_trellis2_forward.py`
**Context budget:** ~7% of context window
**Produces:** deterministic image resize, channel layout, dtype, and normalization helper for conditioning input.
**Acceptance criteria:**
- Tiny generated image fixtures produce MLX arrays with deterministic shape, dtype, and normalized values.
- The helper uses the DINO normalization constants from the reference path.
- No TorchVision transforms or vendor imports are used.
**Verification:** `uv run pytest tests/test_trellis2_forward.py`
**Auto-continue:** yes

### Slice 4: DINOv3 MLX Port Assessment

**Objective:** Attempt the MLX-first DINOv3 image-conditioning path far enough to either produce conditioning tensor metadata or return the first exact DINOv3 asset/op/key blocker.
**Execution:** subagent recommended
**Depends on:** Slice 3
**Touches:** `src/mlx_spatial/trellis2_forward.py`, optional focused modules under `src/mlx_spatial/`, `tests/test_trellis2_forward.py`, `.agent/work/trellis-forward-trace-conditioning/CONDITIONING.md`
**Context budget:** ~15% of context window
**Produces:** executable conditioning assessment and a local report naming the DINOv3 model, asset state, attempted MLX boundary, and blocker or output metadata.
**Acceptance criteria:**
- The implementation checks for local DINOv3 assets/config without network downloads.
- If local DINOv3 assets are absent, the blocker names `facebook/dinov3-vitl16-pretrain-lvd1689m` and the explicit local asset requirement.
- If local DINOv3 assets are present, the implementation attempts explicit MLX module/key/op assessment and names the first unsupported boundary.
- Fake-fixture tests can simulate successful conditioning metadata and precise blockers.
**Verification:** `uv run pytest tests/test_trellis2_forward.py tests/test_trellis2_inference.py`
**Auto-continue:** no

### Slice 5: Sparse-Structure Boundary Dispatch

**Objective:** Dispatch from conditioning output metadata into the first downstream sparse-structure boundary and return a precise next blocker.
**Execution:** direct
**Depends on:** Slice 4
**Touches:** `src/mlx_spatial/trellis2_forward.py`, `tests/test_trellis2_forward.py`
**Context budget:** ~10% of context window
**Produces:** sparse-structure boundary probe that validates conditioning width against `cond_channels` and inspects first sparse-flow checkpoint requirements.
**Acceptance criteria:**
- Fake successful conditioning output advances completed stages past `image-conditioning`.
- Conditioning width mismatch returns a `sparse-structure-sampling` blocker with expected and actual widths.
- Compatible fake conditioning output reaches a sparse-flow module/sampler blocker rather than stopping at generic image conditioning.
- No full sampler or decoder execution is claimed.
**Verification:** `uv run pytest tests/test_trellis2_forward.py tests/test_trellis2_inference.py`
**Auto-continue:** yes

### Slice 6: Real Local Forward Trace Evidence

**Objective:** Run the forward trace against `inputs/trellis2/demo-alpha.webp` and local `weights/trellis2`, recording how far it gets.
**Execution:** direct
**Depends on:** Slice 5
**Touches:** ignored output paths, `.agent/work/trellis-forward-trace-conditioning/ATTEMPT.md`
**Context budget:** ~6% of context window
**Produces:** local attempt evidence with completed stages, tensor metadata if any, and final blocker.
**Acceptance criteria:**
- Evidence names input path, TRELLIS.2 root, conditioning config, completed stages, outputs, and blocker.
- The real local alpha attempt no longer reports the old generic `image-conditioning` blocker.
- Any generated JSON or images are under ignored `outputs/`.
**Verification:** `uv run python -c "from mlx_spatial import Trellis2InferencePipeline; r=Trellis2InferencePipeline('weights/trellis2').attempt_forward_trace('inputs/trellis2/demo-alpha.webp'); print(r.completed_stages); print(r.blocker.stage if r.blocker else None); print(r.blocker.operation if r.blocker else None)"`
**Auto-continue:** yes

### Slice 7: Documentation And Full Verification

**Objective:** Document the new forward-trace boundary and prove the default suite still passes.
**Execution:** direct
**Depends on:** Slice 6
**Touches:** `README.md`, `.agent/work/trellis-forward-trace-conditioning/PLAN.md`
**Context budget:** ~5% of context window
**Produces:** README notes and final plan evidence.
**Acceptance criteria:**
- README or TRELLIS docs describe the forward-trace method, DINOv3 conditioning dependency, and next known blocker.
- PLAN.md records slice-by-slice execution evidence.
- Full default tests pass without real weights, network, Hugging Face credentials, PyTorch, TorchVision, Transformers, ONNX Runtime, or vendor imports.
- Git status confirms real weights and generated outputs are ignored or absent.
**Verification:** `uv run pytest`
**Auto-continue:** no

## Execution Routing And Topology

- Slices 1-3 are direct and serial; they establish contracts, routing, and deterministic tensor preparation.
- Slice 4 is a checkpoint and uses `subagent recommended` because DINOv3 availability/key/op assessment is the highest-risk part and may touch multiple files.
- Slice 5 resumes direct execution after Slice 4 returns either conditioning metadata or a precise DINOv3 blocker simulation path for tests.
- Slices 6-7 are direct evidence/documentation closeout.
- Auto-continue chain: Slices 1 -> 2 -> 3 may continue after tests pass; stop after Slice 4 for blocker review; Slices 5 -> 6 -> 7 may continue only if Slice 5 has passing tests.
- Parallel-safe groups: none. The forward-trace report shape and pipeline integration are shared surfaces.
- Recommended pre-execution review: `auto-eng-review`, because the plan introduces a new public attempt method, a forward-trace result contract, and a risky DINOv3 MLX assessment.

## Verification Commands

- Slice 1: `uv run pytest tests/test_trellis2_forward.py`
- Slice 2: `uv run pytest tests/test_trellis2_inference.py tests/test_trellis2_forward.py`
- Slice 3: `uv run pytest tests/test_trellis2_forward.py`
- Slice 4: `uv run pytest tests/test_trellis2_forward.py tests/test_trellis2_inference.py`
- Slice 5: `uv run pytest tests/test_trellis2_forward.py tests/test_trellis2_inference.py`
- Slice 6: `uv run python -c "from mlx_spatial import Trellis2InferencePipeline; r=Trellis2InferencePipeline('weights/trellis2').attempt_forward_trace('inputs/trellis2/demo-alpha.webp'); print(r.completed_stages); print(r.blocker.stage if r.blocker else None); print(r.blocker.operation if r.blocker else None)"`
- Slice 7: `uv run pytest`

## Context Budget For This Change

- Estimated total: ~59% of context window across planning and execution.
- Largest slice: Slice 4 at ~15% because DINOv3 model availability and MLX op/key assessment may expose the first hard blocker.
- Checkpoints: after Slice 4, before real local evidence, and before final verification.
- Expected closeout: the alpha forward trace should replace the generic `image-conditioning` blocker with either conditioning output metadata plus a downstream blocker, or a precise DINOv3 asset/op/key blocker.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan separates forward-trace contracts, pipeline routing, image tensor preparation, DINOv3 assessment, sparse-boundary dispatch, and real evidence into independently verifiable slices.
- Concern: Slice 4 may stop at a missing external DINOv3 asset or unsupported MLX DINOv3 module before real conditioning tensors are produced, so execution must preserve the precise-blocker contract instead of broadening into downloads or PyTorch parity.
- Action: Proceed with `auto-execute`, but stop after Slice 4 if the DINOv3 asset/op/key blocker changes the planned downstream dispatch route.
- Verified: SPEC.md, DESIGN.md, PLAN.md, state pointers, slice dependencies, verification commands, DINOv3 config dependency, fake-fixture strategy, default-test constraints, and blocker semantics checked.

## Execution Evidence

- Slice 1: PASS. Added `trellis2_forward.py` dataclasses and config discovery for `image_cond_model`, DINOv3 model name, conditioning resolution, expected feature width, and sparse-flow config; `uv run pytest tests/test_trellis2_forward.py` reported `8 passed`.
- Slice 2: PASS. Added `Trellis2InferencePipeline.attempt_forward_trace(...)` without changing existing `attempt(...)` behavior; `uv run pytest tests/test_trellis2_forward.py tests/test_trellis2_inference.py` reported `19 passed`.
- Slice 3: PASS. Added deterministic DINOv3 image tensor preparation using PIL resize, BCHW layout, float32 dtype, and DINO normalization constants; `uv run pytest tests/test_trellis2_forward.py` reported `8 passed`.
- Slice 4: PASS with checkpoint blocker. Added local DINOv3 asset/port assessment, fake successful conditioning-output metadata, and present-asset module-construction blocker coverage; live `inputs/trellis2/demo-alpha.webp` forward trace now blocks at `image-conditioning` / `local DINOv3 asset validation` for missing `weights/dinov3-vitl16-pretrain-lvd1689m/config.json` and `model.safetensors`. `uv run pytest tests/test_trellis2_forward.py tests/test_trellis2_inference.py` reported `19 passed`; full `uv run pytest` reported `91 passed, 5 skipped`. Recorded details in `CONDITIONING.md`.
- Slice 5: PASS. Added sparse-structure boundary dispatch that validates conditioning feature width against `cond_channels`, inspects required sparse-flow checkpoint keys, and returns `sparse-structure-sampling` / `MLX sparse structure flow model construction` instead of a generic image-conditioning blocker for compatible fake conditioning metadata. `uv run pytest tests/test_trellis2_forward.py tests/test_trellis2_inference.py` reported `22 passed`.
- Slice 6: PASS. Recorded real local forward-trace evidence in `ATTEMPT.md` and ignored JSON under `outputs/trellis2/forward-trace/demo-alpha-forward-trace.json`; the real alpha attempt completes through `image-preprocessing-background` and blocks at `image-conditioning` / `local DINOv3 asset validation`. The planned command reported `('input-image', 'asset-config-validation', 'checkpoint-probe-readiness', 'image-preprocessing-background')`, `image-conditioning`, and `local DINOv3 asset validation`.
- Slice 7: PASS. Updated README with `attempt_forward_trace(...)`, DINOv3 conditioning config, local DINOv3 asset blocker, and fake-fixture sparse-boundary behavior. Full `uv run pytest` reported `94 passed, 5 skipped`; `git status --short --ignored` confirmed `inputs/`, `outputs/`, and `weights/` are ignored.
