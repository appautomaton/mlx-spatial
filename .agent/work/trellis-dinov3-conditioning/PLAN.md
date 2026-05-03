# PLAN: TRELLIS.2 DINOv3 Conditioning

## Goal

Make the TRELLIS.2 forward trace resolve local `facebook/dinov3-vitl16-pretrain-lvd1689m` assets and attempt MLX `DinoV3FeatureExtractor` construction/forward, producing real conditioning tensor metadata or the first exact DINOv3 config, key, shape, module, or operation blocker.

## Architecture Approach

Keep the existing forward-trace/result contract. Add an offline DINOv3 asset and inspection layer, wire it into `assess_dinov3_conditioning(...)`, and use fake safetensor fixtures for default tests. Real Hugging Face asset acquisition stays explicit and checkpointed. Runtime code must remain free of PyTorch, TorchVision, Transformers, Hugging Face Hub, ONNX Runtime, and vendor imports.

## Ordered Task Sequence

### Slice 1: DINOv3 Asset Surface

**Objective:** Add deterministic local asset metadata, validation helpers, exports, and CLI/help text for `facebook/dinov3-vitl16-pretrain-lvd1689m`.
**Execution:** direct
**Depends on:** none
**Touches:** `src/mlx_spatial/model_assets.py`, `src/mlx_spatial/trellis2.py`, `src/mlx_spatial/__init__.py`, `tests/test_model_assets.py`, `tests/test_trellis2_tools.py`
**Context budget:** ~8% of context window
**Produces:** Public DINOv3 asset validation and manual download/help surface.
**Acceptance criteria:**
- DINOv3 manifest validates `config.json` and `model.safetensors` under `weights/dinov3-vitl16-pretrain-lvd1689m`.
- Help text names `facebook/dinov3-vitl16-pretrain-lvd1689m`, the local root, expected files, and possible authentication or terms acceptance.
- Validation performs no network access and no model loading.
**Verification:** `uv run pytest tests/test_model_assets.py tests/test_trellis2_tools.py`
**Auto-continue:** yes

### Slice 2: DINOv3 Config and Checkpoint Inventory

**Objective:** Inspect local DINOv3 config/checkpoint structure with offline MLX-compatible readers and tiny fake safetensor fixtures.
**Execution:** subagent recommended
**Depends on:** Slice 1
**Touches:** `src/mlx_spatial/trellis2_dinov3.py`, `src/mlx_spatial/trellis2_forward.py`, `src/mlx_spatial/__init__.py`, `tests/test_trellis2_dinov3.py`
**Context budget:** ~12% of context window
**Produces:** Deterministic config/key/shape inventory plus precise blockers for malformed or incompatible fake assets.
**Acceptance criteria:**
- Parser reports hidden size, layers, heads, patch size, MLP/intermediate size, norm behavior, image size, and expected output width when present.
- Safetensor inventory reports selected required keys and shapes without importing Transformers.
- Missing or incompatible fixture data returns a precise `image-conditioning` blocker.
**Verification:** `uv run pytest tests/test_trellis2_dinov3.py tests/test_trellis2_forward.py`
**Auto-continue:** yes

### Slice 3: Forward Trace Integration

**Objective:** Replace generic DINO asset blocking in `assess_dinov3_conditioning(...)` with the new validation and inventory layer.
**Execution:** direct
**Depends on:** Slice 2
**Touches:** `src/mlx_spatial/trellis2_forward.py`, `src/mlx_spatial/trellis2_inference.py`, `tests/test_trellis2_forward.py`, `tests/test_trellis2_inference.py`
**Context budget:** ~8% of context window
**Produces:** Forward trace that distinguishes missing assets, bad config/checkpoint assets, and DINOv3 port blockers.
**Acceptance criteria:**
- Missing local DINOv3 assets still block at `image-conditioning` / `local DINOv3 asset validation`.
- Present incompatible fake assets block with the first exact config, key, shape, module, or op issue.
- Fake-compatible conditioning output records an `image-conditioning` output and reaches the existing `sparse-structure-sampling` boundary.
**Verification:** `uv run pytest tests/test_trellis2_forward.py tests/test_trellis2_inference.py`
**Auto-continue:** yes

### Slice 4: Real Asset Acquisition Checkpoint

**Objective:** Determine whether real DINOv3 assets are present locally and record the explicit command needed if they are absent.
**Execution:** direct
**Depends on:** Slice 3
**Touches:** `.agent/work/trellis-dinov3-conditioning/ATTEMPT.md`, ignored `weights/dinov3-vitl16-pretrain-lvd1689m/` only if the user explicitly allows download
**Context budget:** ~5% of context window
**Produces:** Real local asset status and, if needed, a manual download command.
**Acceptance criteria:**
- Status names whether `config.json` and `model.safetensors` are present.
- No download happens silently.
- If assets are missing, the plan pauses with a command the user can approve/run explicitly.
**Verification:** `uv run mlx-spatial-trellis2 dinov3-validate weights/dinov3-vitl16-pretrain-lvd1689m`
**Auto-continue:** no

### Slice 5: MLX DINOv3 Construction and Forward Probe

**Objective:** Attempt the smallest honest MLX DINOv3 image-conditioning implementation or return the first exact real port blocker.
**Execution:** subagent recommended
**Depends on:** Slice 4
**Touches:** `src/mlx_spatial/trellis2_dinov3.py`, `src/mlx_spatial/trellis2_forward.py`, `tests/test_trellis2_dinov3.py`, `tests/test_trellis2_forward.py`
**Context budget:** ~15% of context window
**Produces:** MLX DINOv3 module/forward probe that emits conditioning metadata or a precise unsupported-field/key/shape/module/op blocker.
**Acceptance criteria:**
- Fake-compatible checkpoint path can run through module construction/forward and produce conditioning metadata with last dimension 1024.
- Real assets, when present, either produce real conditioning metadata or block on the first exact unmapped DINOv3 requirement.
- Runtime code does not import forbidden dependencies or vendor modules.
**Verification:** `uv run pytest tests/test_trellis2_dinov3.py tests/test_trellis2_forward.py`
**Auto-continue:** no

### Slice 6: Real Forward Attempt Evidence

**Objective:** Run the live alpha forward trace with the current local asset state and capture the truthful boundary.
**Execution:** direct
**Depends on:** Slice 5
**Touches:** `.agent/work/trellis-dinov3-conditioning/ATTEMPT.md`, ignored `outputs/`
**Context budget:** ~5% of context window
**Produces:** Fresh attempt evidence for `inputs/trellis2/demo-alpha.webp`.
**Acceptance criteria:**
- Attempt records completed stages, outputs, blocker stage, operation, reason, and next slice.
- If conditioning metadata is produced, the trace reaches the existing sparse-structure boundary and does not continue into full sampling.
- If DINOv3 is blocked, the blocker is more precise than generic asset presence when assets are present.
**Verification:** `uv run mlx-spatial-trellis2 attempt-forward-trace weights/trellis2 inputs/trellis2/demo-alpha.webp --output outputs/trellis2/demo-alpha-forward-trace.json`
**Auto-continue:** yes

### Slice 7: Docs and Full Verification

**Objective:** Update user-facing docs and verify the complete change.
**Execution:** direct
**Depends on:** Slice 6
**Touches:** `README.md`, `.agent/work/trellis-dinov3-conditioning/VERIFY.md`
**Context budget:** ~6% of context window
**Produces:** Documentation and verification evidence for the completed slice.
**Acceptance criteria:**
- Docs describe DINOv3 local asset requirements, validation/download help, and current forward-trace boundary.
- Full test suite passes without real weights or network access.
- Git inventory confirms weights and generated outputs are untracked/ignored.
**Verification:** `uv run pytest`
**Auto-continue:** no

## Execution Routing and Topology

Slices 1 to 3 can auto-continue after their verification passes. Slice 4 is a checkpoint because real asset acquisition may require authentication, terms acceptance, network access, and explicit user approval. Slice 5 is another checkpoint because the DINOv3 MLX port may reveal a new architecture or op blocker. Slices 6 to 7 can continue after Slice 5 produces either real conditioning metadata or an exact DINOv3 blocker.

Parallel-safe groups: none. The write sets overlap around `trellis2_forward.py` and the tests, and the truth of later slices depends on the previous blockers.

Recommended pre-execution gate: `auto-eng-review`, because the plan touches shared public APIs, checkpoint parsing, and a non-trivial MLX model-port boundary.

## Verification Commands

- Slice 1: `uv run pytest tests/test_model_assets.py tests/test_trellis2_tools.py`
- Slice 2: `uv run pytest tests/test_trellis2_dinov3.py tests/test_trellis2_forward.py`
- Slice 3: `uv run pytest tests/test_trellis2_forward.py tests/test_trellis2_inference.py`
- Slice 4: `uv run mlx-spatial-trellis2 dinov3-validate weights/dinov3-vitl16-pretrain-lvd1689m`
- Slice 5: `uv run pytest tests/test_trellis2_dinov3.py tests/test_trellis2_forward.py`
- Slice 6: `uv run mlx-spatial-trellis2 attempt-forward-trace weights/trellis2 inputs/trellis2/demo-alpha.webp --output outputs/trellis2/demo-alpha-forward-trace.json`
- Slice 7: `uv run pytest`

## Context Budget For This Change

Estimated total: ~59% of context window if executed in one continuous session. The real asset acquisition checkpoint and the DINOv3 port checkpoint are intentional breakpoints if the local model files or MLX op coverage change the implementation path.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan has a clean staged boundary from offline DINOv3 asset validation through fake-fixture conditioning and real local attempt evidence while preserving the existing forward-trace contract.
- Concern: Slice 5 remains the highest-risk point because fake-compatible MLX construction can prove plumbing but real DINOv3 assets may still expose an unsupported config field, checkpoint layout, tensor shape, module, or MLX operation.
- Action: Proceed with `auto-execute` through Slices 1-3, stop at the Slice 4 real asset checkpoint, and only continue into Slice 5 after the local DINOv3 asset status is recorded.
- Verified: PLAN.md, DESIGN.md, STATUS.md, canonical state pointers, slice dependencies, verification commands, forbidden runtime dependency constraints, asset checkpoint boundary, and blocker behavior were checked.

## Execution Evidence

- Slice 1: PASS. Added `DINOv3_VITL16_ASSETS`, public validation/export helpers, `dinov3-validate`, and explicit `dinov3-download-command`; `uv run pytest tests/test_model_assets.py tests/test_trellis2_tools.py` reported `19 passed`.
- Slice 2: PASS. Added `trellis2_dinov3.py` with offline DINOv3 config parsing, safetensors key/shape inventory, fake-compatible conditioning metadata, and precise blockers for missing fields and bad patch shapes; `uv run pytest tests/test_trellis2_dinov3.py tests/test_trellis2_forward.py` reported `21 passed`.
- Slice 3: PASS. Wired the DINOv3 inspection layer into `assess_dinov3_conditioning(...)` and `attempt_forward_trace(...)`; missing assets still block at `image-conditioning`, incompatible fake assets name the config/key/shape issue, and fake-compatible assets reach `sparse-structure-sampling`; `uv run pytest tests/test_trellis2_forward.py tests/test_trellis2_inference.py` reported `24 passed`.
- Slice 4: PASS. Downloaded the explicit DINOv3 assets into ignored `weights/dinov3-vitl16-pretrain-lvd1689m`; `uv run mlx-spatial-trellis2 dinov3-validate weights/dinov3-vitl16-pretrain-lvd1689m` reported `ready=True`, `present=2`, and `missing=0`.
- Slice 5: PASS with checkpoint blocker. Real local DINOv3 inventory is readable: ViT-L/16 config, hidden size 1024, 24 layers, 415 tensors, and patch embedding shape `(1024, 3, 16, 16)`. The first real MLX port blocker is `image-conditioning` / `MLX DINOv3 transformer block construction`, because embeddings/checkpoint inventory are readable but the MLX transformer layers with RoPE position embeddings are not implemented. `uv run pytest tests/test_trellis2_dinov3.py tests/test_trellis2_forward.py` reported `21 passed`.
- Slice 6: PASS. Added the `attempt-forward-trace` CLI entrypoint required by the planned verification command and refreshed the real alpha trace with local DINOv3 assets present. The trace completes through `image-preprocessing-background` and blocks at `image-conditioning` / `MLX DINOv3 transformer block construction`; the ignored JSON evidence is `outputs/trellis2/demo-alpha-forward-trace.json`. `uv run pytest tests/test_trellis2_tools.py tests/test_trellis2_forward.py tests/test_trellis2_inference.py` reported `36 passed`.
- Slice 7: PASS. Updated README with DINOv3 validation/download-help commands, local asset layout, real checkpoint inventory, and current forward-trace boundary.
- Execution-window verification: PASS. Full `uv run pytest` from the previous window reported `106 passed, 5 skipped`; targeted Slice 5 verification after real asset download reported `21 passed`; final full verification is recorded in `VERIFY.md`.
