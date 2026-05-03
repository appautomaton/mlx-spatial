# PLAN: TRELLIS.2 MLX DINOv3 Forward

## Goal

Implement the TRELLIS.2 DINOv3 image-conditioning forward path in MLX far enough to produce real conditioning tensor metadata from local `facebook/dinov3-vitl16-pretrain-lvd1689m` assets, or return the first exact MLX DINOv3 embedding, RoPE, attention, MLP, normalization, checkpoint-key, shape, or memory blocker.

## Architecture Approach

Keep `trellis2_dinov3.py` as the asset/config/blocker surface and add an MLX-only forward implementation layer for model math. Build the path in probes: selected tensor loading, patch embedding/token assembly, RoPE, attention, MLP/norm/layer-scale, repeated layers, then forward-trace integration. Use tiny fake fixtures for default tests and local real assets only for explicit attempt evidence.

## Ordered Task Sequence

### Slice 1: Forward Key Map And Tensor Loader

**Objective:** Define the DINOv3 forward checkpoint key map and load selected tensors with exact missing-key and shape blockers.
**Execution:** direct
**Depends on:** none
**Touches:** `src/mlx_spatial/trellis2_dinov3.py`, `src/mlx_spatial/trellis2_dinov3_forward.py`, `tests/test_trellis2_dinov3_forward.py`
**Context budget:** ~8% of context window
**Produces:** MLX-only key map and selected tensor loading for embeddings, one layer, final norm, and fake fixtures.
**Acceptance criteria:**
- Fake fixtures validate required embedding, attention, MLP, norm, and layer-scale keys.
- Missing fake keys return precise blockers naming the missing key.
- Real checkpoint key-map probe reports the current visible keys without loading unrelated tensors.
**Verification:** `uv run pytest tests/test_trellis2_dinov3_forward.py tests/test_trellis2_dinov3.py`
**Auto-continue:** yes

### Slice 2: Patch Embedding And Token Assembly

**Objective:** Implement patch embedding from BCHW image tensors and assemble cls/register/patch tokens.
**Execution:** direct
**Depends on:** Slice 1
**Touches:** `src/mlx_spatial/trellis2_dinov3_forward.py`, `tests/test_trellis2_dinov3_forward.py`
**Context budget:** ~8% of context window
**Produces:** Tested fake patch-token output shaped `(B, N, D)`.
**Acceptance criteria:**
- BCHW fake image input produces expected token count for configurable runtime image sizes.
- Patch embedding accepts checkpoint layout `(D, 3, patch, patch)` and returns hidden width `D`.
- cls and register tokens are inserted or intentionally skipped with documented behavior matching the checkpoint/config path.
**Verification:** `uv run pytest tests/test_trellis2_dinov3_forward.py`
**Auto-continue:** yes

### Slice 3: RoPE Probe

**Objective:** Implement or precisely block DINOv3 RoPE/position embedding behavior for token attention.
**Execution:** subagent recommended
**Depends on:** Slice 2
**Touches:** `src/mlx_spatial/trellis2_dinov3_forward.py`, `tests/test_trellis2_dinov3_forward.py`, `.agent/work/trellis-dinov3-mlx-forward/ATTEMPT.md`
**Context budget:** ~12% of context window
**Produces:** RoPE implementation for fake fixtures or a blocker naming the first unsupported RoPE dimension/layout requirement.
**Acceptance criteria:**
- Fake RoPE route is deterministic and shape-checked.
- Real config attempt reaches RoPE and either constructs required position data or returns a specific RoPE blocker.
- Blocker is more precise than `MLX DINOv3 transformer block construction`.
**Verification:** `uv run pytest tests/test_trellis2_dinov3_forward.py`
**Auto-continue:** no

### Slice 4: Single Transformer Block

**Objective:** Implement one MLX DINOv3 transformer block over fake tensors: norm1, q/k/v attention, output projection, layer scale, norm2, MLP, residuals.
**Execution:** subagent recommended
**Depends on:** Slice 3
**Touches:** `src/mlx_spatial/trellis2_dinov3_forward.py`, `tests/test_trellis2_dinov3_forward.py`
**Context budget:** ~14% of context window
**Produces:** Executable single-block fake forward path and exact blockers for missing/shape-mismatched block tensors.
**Acceptance criteria:**
- Fake block forward returns `(B, N, D)` and preserves dtype/layout expectations.
- Attention projections and MLP use checkpoint-compatible key names and shapes.
- Missing q/k/v/o, MLP, norm, or layer-scale keys are reported exactly.
**Verification:** `uv run pytest tests/test_trellis2_dinov3_forward.py tests/test_trellis2_dinov3.py`
**Auto-continue:** yes

### Slice 5: Full Fake Forward And Integration

**Objective:** Run a complete fake DINOv3 conditioning forward through `assess_dinov3_mlx_conditioning(...)` and `attempt_forward_trace(...)`.
**Execution:** subagent recommended
**Depends on:** Slice 4
**Touches:** `src/mlx_spatial/trellis2_dinov3.py`, `src/mlx_spatial/trellis2_forward.py`, `tests/test_trellis2_dinov3.py`, `tests/test_trellis2_forward.py`, `tests/test_trellis2_dinov3_forward.py`
**Context budget:** ~12% of context window
**Produces:** Fake executable DINOv3 path that records conditioning output metadata and reaches the existing sparse boundary.
**Acceptance criteria:**
- Fake executable fixture produces conditioning metadata with last dimension 1024.
- `attempt_forward_trace(...)` records `image-conditioning` output for fake executable assets.
- Existing fake-conditioner and missing-asset tests remain valid.
**Verification:** `uv run pytest tests/test_trellis2_dinov3.py tests/test_trellis2_forward.py tests/test_trellis2_dinov3_forward.py`
**Auto-continue:** yes

### Slice 6: Real Forward Probe

**Objective:** Attempt the real local ViT-L/16 DINOv3 forward at TRELLIS.2 conditioning resolution or return the first exact real blocker.
**Execution:** subagent recommended
**Depends on:** Slice 5
**Touches:** `src/mlx_spatial/trellis2_dinov3.py`, `src/mlx_spatial/trellis2_dinov3_forward.py`, `.agent/work/trellis-dinov3-mlx-forward/ATTEMPT.md`, ignored `outputs/`
**Context budget:** ~15% of context window
**Produces:** Real alpha attempt evidence with either conditioning output metadata or a precise blocker.
**Acceptance criteria:**
- Real attempt does not fake output.
- Real blocker, if any, names the exact embedding/RoPE/attention/MLP/norm/key/shape/dtype/layout/memory boundary.
- If real conditioning output is produced, it dispatches to the existing sparse-structure boundary and does not enter full sampling.
**Verification:** `uv run mlx-spatial-trellis2 attempt-forward-trace weights/trellis2 inputs/trellis2/demo-alpha.webp --output outputs/trellis2/dinov3-mlx-forward-alpha.json`
**Auto-continue:** no

### Slice 7: Docs And Full Verification

**Objective:** Document the current MLX DINOv3 forward boundary and verify the complete change.
**Execution:** direct
**Depends on:** Slice 6
**Touches:** `README.md`, `.agent/work/trellis-dinov3-mlx-forward/VERIFY.md`
**Context budget:** ~6% of context window
**Produces:** User-facing docs and final verification evidence.
**Acceptance criteria:**
- README describes the new MLX DINOv3 forward boundary and current blocker or output metadata.
- Full test suite passes.
- Runtime dependencies still exclude forbidden packages.
- Real weights and generated outputs remain ignored.
**Verification:** `uv run pytest`
**Auto-continue:** no

## Execution Routing and Topology

Slices 1 and 2 can auto-continue because they are bounded fake-fixture implementation work. Slice 3 is a checkpoint because RoPE is the first likely architecture mismatch. Slices 4 and 5 can continue after RoPE has a concrete route or blocker because they stay fake-fixture based. Slice 6 is a checkpoint because it runs the real local ViT-L/16 path and may expose memory or unsupported-op blockers. Slice 7 closes documentation and full verification after the real boundary is known.

Parallel-safe groups: none. The write set centers on `trellis2_dinov3.py`, the new forward module, and shared forward tests, so serial execution is safer.

Recommended pre-execution gate: `auto-eng-review`, because this plan implements non-trivial model math, checkpoint key mapping, and memory-sensitive real inference probing.

## Verification Commands

- Slice 1: `uv run pytest tests/test_trellis2_dinov3_forward.py tests/test_trellis2_dinov3.py`
- Slice 2: `uv run pytest tests/test_trellis2_dinov3_forward.py`
- Slice 3: `uv run pytest tests/test_trellis2_dinov3_forward.py`
- Slice 4: `uv run pytest tests/test_trellis2_dinov3_forward.py tests/test_trellis2_dinov3.py`
- Slice 5: `uv run pytest tests/test_trellis2_dinov3.py tests/test_trellis2_forward.py tests/test_trellis2_dinov3_forward.py`
- Slice 6: `uv run mlx-spatial-trellis2 attempt-forward-trace weights/trellis2 inputs/trellis2/demo-alpha.webp --output outputs/trellis2/dinov3-mlx-forward-alpha.json`
- Slice 7: `uv run pytest`

## Context Budget For This Change

Estimated total: ~75% of context window if attempted in one continuous session. The RoPE checkpoint and real ViT-L/16 probe checkpoint are intentional breakpoints. If Slice 6 produces a memory or unsupported-op blocker, the change can still close as successful if the blocker is exact and more specific than the previous transformer-construction placeholder.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan decomposes DINOv3 forward execution into key-map, embedding, RoPE, block, fake integration, and real-probe slices with explicit verification and checkpoint boundaries.
- Concern: RoPE semantics and full 512-resolution ViT-L/16 memory behavior remain high-risk, so Slice 3 and Slice 6 must stop on exact blockers instead of expanding into unplanned parity or optimization work.
- Action: Proceed with `auto-execute` through Slices 1-2, stop at the Slice 3 RoPE checkpoint, and continue only after the RoPE route or blocker is recorded.
- Verified: PLAN.md, DESIGN.md, STATUS.md, canonical state pointers, slice dependencies, verification commands, checkpoint boundaries, fake-fixture strategy, real-weight boundary, and forbidden runtime dependency constraints were checked.

## Execution Evidence

### Slices 1-3: Forward Key Map, Patch Tokens, RoPE Probe

- Route used: direct.
- Implemented `src/mlx_spatial/trellis2_dinov3_forward.py` for DINOv3 forward key mapping, selected tensor loading, patch embedding/token assembly, and RoPE geometry validation.
- Updated `src/mlx_spatial/trellis2_dinov3.py` to parse DINOv3 forward config fields and dispatch real conditioning attempts into the forward probe.
- Added `tests/test_trellis2_dinov3_forward.py` and updated DINOv3 fixtures to use the real Hugging Face checkpoint key layout.
- Targeted verification passed: `uv run pytest tests/test_trellis2_dinov3_forward.py tests/test_trellis2_dinov3.py tests/test_trellis2_forward.py` reported `28 passed`.
- Full verification passed after the execution artifact updates: `uv run pytest` reported `113 passed, 5 skipped`.
- Real local alpha trace passed through key map, patch embedding, token assembly, and RoPE geometry; it now stops at `image-conditioning` / `MLX DINOv3 attention block forward`.
- Attempt artifact: `.agent/work/trellis-dinov3-mlx-forward/ATTEMPT.md`.
- Generated ignored trace: `outputs/trellis2/dinov3-mlx-forward-alpha.json`.

### Slice 4: Single Transformer Block

- Route used: direct.
- Implemented one MLX DINOv3 transformer block in `src/mlx_spatial/trellis2_dinov3_forward.py`: LayerNorm, q/k/v projections, DINOv3 2D RoPE on patch-token q/k tensors, scaled dot-product attention, output projection, layer-scale residual, MLP, and second layer-scale residual.
- Added fake-fixture coverage for block output shape/dtype and exact missing block-tensor blockers in `tests/test_trellis2_dinov3_forward.py`.
- Targeted verification passed: `uv run pytest tests/test_trellis2_dinov3_forward.py tests/test_trellis2_dinov3.py` reported `17 passed`.
- Full verification passed: `uv run pytest` reported `115 passed, 5 skipped`.
- Real local alpha trace evaluated the first MLX DINOv3 transformer block and moved the blocker to `image-conditioning` / `MLX DINOv3 full layer stack forward`.
- Generated ignored trace: `outputs/trellis2/dinov3-mlx-forward-alpha.json`.

### Slices 5-6: Full Forward Integration And Real Forward Probe

- Route used: direct.
- Implemented complete MLX DINOv3 layer-stack execution in `src/mlx_spatial/trellis2_dinov3_forward.py`: all configured transformer layers plus final layer normalization.
- Updated `src/mlx_spatial/trellis2_forward.py` so fake executable traces can set `image_cond_model.args.image_size` while the default TRELLIS.2 conditioning resolution remains 512.
- Added fake executable forward trace coverage in `tests/test_trellis2_forward.py` and full-layer/final-norm coverage in `tests/test_trellis2_dinov3_forward.py`.
- Targeted verification passed: `uv run pytest tests/test_trellis2_dinov3_forward.py tests/test_trellis2_dinov3.py tests/test_trellis2_forward.py` reported `33 passed`.
- Full verification passed: `uv run pytest` reported `118 passed, 5 skipped`.
- Real local alpha trace produced DINOv3 conditioning output `cond` with shape `(1, 1029, 1024)` and dtype `float32` after 24 transformer layers.
- The current real blocker moved to `sparse-structure-sampling` / `MLX sparse structure flow model construction`.
- Generated ignored trace: `outputs/trellis2/dinov3-mlx-forward-alpha.json`.

### Slice 7: Docs And Full Verification

- Route used: direct.
- Updated `README.md` to describe the current MLX DINOv3 forward boundary: real local DINOv3 conditioning now completes and the active blocker is sparse-structure sampling.
- Added final verification artifact `.agent/work/trellis-dinov3-mlx-forward/VERIFY.md`.
- Full verification passed: `uv run pytest` reported `118 passed, 5 skipped`.
- Runtime dependency check confirmed `pyproject.toml` still does not add Torch, Transformers, TensorFlow, or JAX as package runtime dependencies.
