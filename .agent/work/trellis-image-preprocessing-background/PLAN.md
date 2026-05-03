# PLAN: TRELLIS.2 Image Preprocessing Background

## Goal

Implement the TRELLIS.2 `image-preprocessing-background` stage in `mlx_spatial` so real image inputs are decoded, normalized, alpha-cropped/composited, and, when local gated RMBG weights are present, processed through an MLX-native BiRefNet background-removal path before the pipeline advances to `image-conditioning` or a precise lower-level blocker.

## Architecture Approach

- Keep the feature inside the existing `mlx-spatial` package and CLI surfaces.
- Add a small image preprocessing module for deterministic PIL-based decode, alpha detection, max-side resize, alpha bbox crop, and RGB-over-alpha composite.
- Update `Trellis2InferencePipeline.attempt(...)` to run real preprocessing after asset/probe readiness and before the existing stage blocker flow.
- Add local RMBG asset metadata and tooling beside the existing TRELLIS.2 asset tooling, with explicit gated/non-commercial wording and no implicit downloads.
- Add MLX safetensors inspection/loading for RMBG first, then attempt the BiRefNet architecture/key mapping in a checkpointed slice.
- Treat an incomplete BiRefNet port as acceptable only when it returns a precise blocker naming the first unsupported module, operation, or weight key.
- Keep default tests independent from real weights, network, PyTorch, TorchVision, Transformers, Hugging Face Hub, ONNX Runtime, and vendor imports.

## Ordered Task Sequence

### Slice 1: Deterministic Image Preprocessor

**Objective:** Implement the local image decode and alpha-based TRELLIS preprocessing path.
**Execution:** direct
**Depends on:** none
**Touches:** `pyproject.toml`, `uv.lock`, `src/mlx_spatial/trellis2_preprocess.py`, `src/mlx_spatial/__init__.py`, `tests/test_trellis2_preprocess.py`
**Context budget:** ~10% of context window
**Produces:** public preprocessing API and tests for RGBA alpha images.
**Acceptance criteria:**
- Image paths are decoded as actual images.
- Useful RGBA alpha is detected without RMBG.
- Max-side resize clamps to 1024 while preserving aspect ratio.
- Alpha foreground bbox crop and RGB-over-alpha composite match the reference semantics.
- Empty/no foreground alpha returns a structured stage blocker rather than raising an incidental array error.
- Public exports are available from `mlx_spatial`.
**Verification:** `uv run pytest tests/test_trellis2_preprocess.py`
**Auto-continue:** yes

### Slice 2: Pipeline Integration For Alpha Inputs

**Objective:** Wire deterministic preprocessing into TRELLIS.2 attempt mode so alpha images advance past `image-preprocessing-background`.
**Execution:** direct
**Depends on:** Slice 1
**Touches:** `src/mlx_spatial/trellis2_inference.py`, `tests/test_trellis2_inference.py`
**Context budget:** ~8% of context window
**Produces:** attempt-mode integration and updated blocker behavior.
**Acceptance criteria:**
- `attempt(...)` rejects invalid/non-image files with a structured `input-image` or preprocessing blocker.
- With fake TRELLIS.2 assets and a tiny RGBA-alpha image, completed stages include `image-preprocessing-background`.
- With a successful alpha preprocessing path, the first blocker moves to `image-conditioning`.
- Existing readiness/dry-run behavior remains deterministic.
**Verification:** `uv run pytest tests/test_trellis2_preprocess.py tests/test_trellis2_inference.py`
**Auto-continue:** yes

### Slice 3: RMBG Asset Tooling

**Objective:** Add explicit local RMBG asset validation and manual download/help surfaces without network side effects.
**Execution:** direct
**Depends on:** Slice 2
**Touches:** `src/mlx_spatial/model_assets.py`, `src/mlx_spatial/trellis2.py`, `src/mlx_spatial/__init__.py`, `tests/test_trellis2_tools.py`, `README.md`
**Context budget:** ~8% of context window
**Produces:** RMBG asset descriptor, validator, CLI/help command, public exports, and docs.
**Acceptance criteria:**
- `weights/rmbg2` or the chosen local root is documented as the RMBG root.
- Validator reports present/missing RMBG files deterministically.
- Manual download/help output names `briaai/RMBG-2.0`, the expected local root, and gated/non-commercial status.
- Runtime dependencies still exclude Hugging Face Hub, PyTorch, TorchVision, and Transformers.
**Verification:** `uv run pytest tests/test_trellis2_tools.py tests/test_model_assets.py`
**Auto-continue:** yes

### Slice 4: RMBG Safetensors Loading And Key Inventory

**Objective:** Add MLX-based RMBG safetensors inspection/loading helpers and checkpoint-key diagnostics.
**Execution:** direct
**Depends on:** Slice 3
**Touches:** `src/mlx_spatial/trellis2_rmbg.py`, `src/mlx_spatial/__init__.py`, `tests/test_trellis2_rmbg.py`
**Context budget:** ~8% of context window
**Produces:** fake-fixture-backed RMBG tensor inspection/loading helpers and key inventory reporting.
**Acceptance criteria:**
- Fake RMBG safetensors can be inspected and loaded as MLX arrays.
- Missing or unsupported files fail deterministically.
- Helpers expose enough checkpoint-key information to drive BiRefNet key mapping.
- No real RMBG weights are required in default tests.
**Verification:** `uv run pytest tests/test_trellis2_rmbg.py tests/test_trellis2_tools.py`
**Auto-continue:** no

### Slice 5: MLX BiRefNet Port Attempt

**Objective:** Attempt the MLX-native BiRefNet architecture/key-mapping path for local RMBG weights.
**Execution:** subagent recommended
**Depends on:** Slice 4
**Touches:** `src/mlx_spatial/trellis2_rmbg.py`, optional focused modules under `src/mlx_spatial/`, `tests/test_trellis2_rmbg.py`, `.agent/work/trellis-image-preprocessing-background/RMBG_PORT.md`
**Context budget:** ~15% of context window
**Produces:** runnable MLX BiRefNet forward path or a precise module/op/key blocker documented in `RMBG_PORT.md`.
**Acceptance criteria:**
- The implementation attempts explicit MLX module construction and safetensors key mapping.
- If compatible local real RMBG assets are present, the path either produces an alpha matte or records the first concrete blocker.
- The blocker names the exact module, operation, weight key, or architecture mismatch rather than a generic "not implemented".
- Default tests use fake module/key fixtures and do not require real RMBG weights.
**Verification:** `uv run pytest tests/test_trellis2_rmbg.py tests/test_trellis2_preprocess.py tests/test_trellis2_inference.py`
**Auto-continue:** no

### Slice 6: RGB Preprocessing Integration

**Objective:** Route RGB or fully opaque images through the RMBG path and preserve precise blocker semantics.
**Execution:** direct
**Depends on:** Slice 5
**Touches:** `src/mlx_spatial/trellis2_preprocess.py`, `src/mlx_spatial/trellis2_inference.py`, `tests/test_trellis2_preprocess.py`, `tests/test_trellis2_inference.py`
**Context budget:** ~8% of context window
**Produces:** RGB attempt behavior using MLX RMBG when available or returning an RMBG-specific blocker.
**Acceptance criteria:**
- RGB and fully opaque RGBA inputs require RMBG rather than fake alpha masks.
- Missing RMBG assets return a structured blocker at `image-preprocessing-background`.
- Incomplete RMBG port blockers propagate through `attempt(...)` with the existing blocker shape.
- Successful RMBG output follows the same crop/composite path as supplied-alpha images.
**Verification:** `uv run pytest tests/test_trellis2_preprocess.py tests/test_trellis2_inference.py tests/test_trellis2_rmbg.py`
**Auto-continue:** yes

### Slice 7: Real Local Attempt Evidence

**Objective:** Run real local attempts against `weights/trellis2` and local sample images to prove the new boundary.
**Execution:** direct
**Depends on:** Slice 6
**Touches:** ignored output paths, `.agent/work/trellis-image-preprocessing-background/ATTEMPT.md`
**Context budget:** ~6% of context window
**Produces:** `ATTEMPT.md` with commands, sample image details, completed stages, and blocker/output evidence.
**Acceptance criteria:**
- RGBA-alpha local attempt completes `image-preprocessing-background` and blocks at `image-conditioning`.
- RGB local attempt either completes RMBG preprocessing or records a precise RMBG asset/port blocker.
- Evidence states whether local `weights/rmbg2` assets were present.
- No real weights or generated large outputs are tracked.
**Verification:** `uv run python -c "from mlx_spatial.trellis2_inference import Trellis2InferencePipeline; print(Trellis2InferencePipeline('weights/trellis2').dry_run(load_probes=False).blocker.stage)"`
**Auto-continue:** yes

### Slice 8: Documentation And Full Verification

**Objective:** Document the completed preprocessing boundary and prove the default suite still passes.
**Execution:** direct
**Depends on:** Slice 7
**Touches:** `README.md`, `.agent/work/trellis-image-preprocessing-background/PLAN.md`
**Context budget:** ~5% of context window
**Produces:** README notes and final plan evidence.
**Acceptance criteria:**
- README or CLI docs describe alpha preprocessing, RGB/RMBG behavior, local RMBG asset requirement, license/gated boundary, and next `image-conditioning` blocker.
- PLAN.md execution evidence is updated slice by slice.
- Full test suite passes without real weights, network, Hugging Face credentials, PyTorch, TorchVision, Transformers, ONNX Runtime, or vendor imports.
- Git status confirms real weights and generated outputs are ignored or absent.
**Verification:** `uv run pytest`
**Auto-continue:** no

## Execution Routing And Topology

- Slices 1-4 are direct and serial; they may auto-continue through local deterministic tests.
- Slice 4 is a checkpoint before architecture porting because real RMBG assets may be absent and key inventory may change the exact porting route.
- Slice 5 is subagent recommended because it may touch multiple files and requires focused architecture/key-mapping work.
- Slice 6 resumes direct integration once Slice 5 returns either a runnable alpha-matte path or a precise blocker.
- Slices 7-8 are direct verification/documentation closeout.
- Parallel-safe groups: none. The stage contract, blocker shape, and preprocessing/RMBG integration share write surfaces.
- Recommended pre-execution review: `auto-eng-review`, because the plan introduces a new dependency boundary and an MLX architecture port attempt.

## Verification Commands

- Slice 1: `uv run pytest tests/test_trellis2_preprocess.py`
- Slice 2: `uv run pytest tests/test_trellis2_preprocess.py tests/test_trellis2_inference.py`
- Slice 3: `uv run pytest tests/test_trellis2_tools.py tests/test_model_assets.py`
- Slice 4: `uv run pytest tests/test_trellis2_rmbg.py tests/test_trellis2_tools.py`
- Slice 5: `uv run pytest tests/test_trellis2_rmbg.py tests/test_trellis2_preprocess.py tests/test_trellis2_inference.py`
- Slice 6: `uv run pytest tests/test_trellis2_preprocess.py tests/test_trellis2_inference.py tests/test_trellis2_rmbg.py`
- Slice 7: `uv run python -c "from mlx_spatial.trellis2_inference import Trellis2InferencePipeline; print(Trellis2InferencePipeline('weights/trellis2').dry_run(load_probes=False).blocker.stage)"`
- Slice 8: `uv run pytest`

## Context Budget For This Change

- Estimated total: ~68% of context window across planning and execution.
- Largest slice: Slice 5 at ~15% because BiRefNet architecture/key mapping may expose unknown MLX gaps.
- Checkpoints: after Slice 4 and Slice 5.
- Expected closeout: deterministic alpha preprocessing should advance to `image-conditioning`; RGB preprocessing should either run MLX RMBG or report a concrete RMBG blocker.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan separates deterministic preprocessing, pipeline integration, RMBG asset handling, safetensors inventory, and BiRefNet porting into verifiable slices with clear checkpoint boundaries.
- Concern: Slice 5 may expose unsupported MLX architecture operations or incompatible RMBG key mappings, and Slice 1 may require a deliberate Pillow dependency update.
- Action: Proceed with `auto-execute`, but stop after Slice 4 if RMBG asset/key inventory changes the BiRefNet port route and record the exact blocker before implementing Slice 5.
- Verified: PLAN.md, DESIGN.md, SPEC.md, state pointers, slice dependencies, verification commands, blocker semantics, dependency boundaries, and default-test constraints checked.

## Execution Evidence

- Slice 1: PASS. Added Pillow as a deliberate runtime dependency, implemented `trellis2_preprocess.py`, exported the preprocessing API, and verified RGBA decode, max-side resize, alpha crop/composite, RGB blocker, empty-alpha blocker, and invalid-image blocker with `uv run pytest tests/test_trellis2_preprocess.py` reporting `6 passed`.
- Slice 2: PASS. Wired preprocessing into `Trellis2InferencePipeline.attempt(...)`; alpha inputs now complete `image-preprocessing-background` and block next at `image-conditioning`, while invalid images and RGB inputs return structured preprocessing blockers; `uv run pytest tests/test_trellis2_preprocess.py tests/test_trellis2_inference.py` reported `16 passed`.
- Slice 3: PASS. Added `RMBG2_ASSETS`, RMBG validation, manual `rmbg-download-command`, public exports, CLI tests, and README notes for gated/non-commercial local assets; `uv run pytest tests/test_trellis2_tools.py tests/test_model_assets.py` reported `17 passed`.
- Slice 4: PASS. Added `trellis2_rmbg.py` with fake-fixture-backed RMBG safetensors inspection, selected tensor loading, and key inventory helpers; `uv run pytest tests/test_trellis2_rmbg.py tests/test_trellis2_tools.py` reported `16 passed`.
- Execution-window verification: PASS. `uv run pytest tests/test_trellis2_preprocess.py tests/test_trellis2_inference.py tests/test_trellis2_tools.py tests/test_model_assets.py tests/test_trellis2_rmbg.py` reported `38 passed`; full `uv run pytest` reported `77 passed, 5 skipped`.
- Checkpoint after Slice 4: local `weights/rmbg2` assets are absent; `uv run mlx-spatial-trellis2 rmbg-validate --root weights/rmbg2` reported missing `model.safetensors`, `config.json`, `BiRefNet_config.py`, and `birefnet.py`.
- Slice 5: BLOCKED. Downloaded and validated local `weights/rmbg2` assets, added executable `assess_rmbg2_mlx_port(...)`, and verified the first concrete blocker: RMBG-2.0 uses `torchvision.ops.deform_conv2d` in the `ASPPDeformable` decoder path, while this MLX runtime has no `mlx.nn.DeformConv2d`; `uv run pytest tests/test_trellis2_rmbg.py tests/test_trellis2_preprocess.py tests/test_trellis2_inference.py` reported `23 passed`. Recorded the blocker in `RMBG_PORT.md`.
- Slice 6: PASS. Routed RGB and fully opaque inputs through configured RMBG assessment, preserved generic missing-RMBG behavior when no root is configured, propagated the BiRefNet deformable-convolution blocker through `attempt(...)`, and kept alpha inputs on the deterministic crop/composite path. `uv run pytest tests/test_trellis2_preprocess.py tests/test_trellis2_inference.py tests/test_trellis2_rmbg.py` reported `27 passed`.
- Slice 7: PASS. Added local attempt evidence in `ATTEMPT.md`; `inputs/trellis2/demo-alpha.webp` completed `image-preprocessing-background` and blocked next at `image-conditioning`, while `inputs/trellis2/demo-rgb-background.png` with local `weights/rmbg2` blocked at `MLX BiRefNet deformable convolution`. Ignored snapshots were written under `outputs/trellis2/`.
- Slice 8: PASS. Updated README with alpha preprocessing, RGB/RMBG behavior, local RMBG asset requirements, gated/non-commercial boundary, and current `image-conditioning`/RMBG blockers. `uv run python -c "from mlx_spatial.trellis2_inference import Trellis2InferencePipeline; print(Trellis2InferencePipeline('weights/trellis2').dry_run(load_probes=False).blocker.stage)"` reported `image-preprocessing-background`; full `uv run pytest` reported `83 passed, 5 skipped`.
