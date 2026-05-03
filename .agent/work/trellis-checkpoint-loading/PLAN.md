# PLAN: TRELLIS.2 Checkpoint Inspection and MLX Loading

## Goal

Enable local inspection of real TRELLIS.2 checkpoint assets and loading of selected checkpoint tensors into MLX arrays, while keeping default tests independent of network access, real weights, Hugging Face credentials, PyTorch, Transformers, and vendor imports.

## Architecture Approach

- Add a small model-neutral checkpoint module for safetensors inspection and selected-tensor loading.
- Represent tensor metadata with a simple public data structure containing name, shape, dtype, and source.
- Keep file-format support safetensors-only for this slice; unsupported suffixes fail explicitly.
- Use fake/minimal safetensors fixtures in tests so verification remains local and deterministic.
- Document manual real-weight placement and optional inspection/loading workflows without invoking downloads from code or tests.

## Ordered Task Sequence

### Slice 1: Safetensors Fixture and Dependency Boundary

**Objective:** Establish safetensors as the only supported checkpoint format for this slice and prove tiny fixture creation works in tests.
**Execution:** direct
**Depends on:** none
**Touches:** `pyproject.toml`, `tests/`
**Context budget:** ~5% of context window
**Produces:** minimal safetensors test fixture pattern and any required lightweight dependency declaration.
**Acceptance criteria:**
- Default dependencies do not include PyTorch, Transformers, or Hugging Face Hub.
- Tests can create a tiny safetensors checkpoint fixture without network access or real weights.
- Unsupported dependency assumptions are visible in test setup or package config.
**Verification:** `uv run pytest tests/test_checkpoint.py`
**Auto-continue:** yes

**Execution correction:** `safetensors.mlx` imports NumPy internally, so Slice 1 adds `numpy` as the minimal companion dependency for safetensors checkpoint creation/loading while still excluding PyTorch, Transformers, and Hugging Face Hub.

### Slice 2: Checkpoint Inspection API

**Objective:** Implement deterministic tensor metadata inspection for local safetensors checkpoints.
**Execution:** direct
**Depends on:** Slice 1
**Touches:** `src/mlx_spatial/checkpoint.py`, `src/mlx_spatial/__init__.py`, `tests/test_checkpoint.py`
**Context budget:** ~8% of context window
**Produces:** public inspection helper and metadata tests.
**Acceptance criteria:**
- Inspection returns tensor name, shape, dtype, and source path for fixture tensors.
- Metadata order is deterministic.
- Exact-name and prefix filters work for inspection.
- Missing paths, unsupported suffixes, invalid filters, and no-match filters raise clear errors.
**Verification:** `uv run pytest tests/test_checkpoint.py`
**Auto-continue:** yes

### Slice 3: Selected Tensor MLX Loading API

**Objective:** Load selected safetensors checkpoint tensors into MLX arrays without loading unrelated tensors by API contract.
**Execution:** direct
**Depends on:** Slice 2
**Touches:** `src/mlx_spatial/checkpoint.py`, `src/mlx_spatial/__init__.py`, `tests/test_checkpoint.py`
**Context budget:** ~8% of context window
**Produces:** public selected-tensor loader and MLX numeric tests.
**Acceptance criteria:**
- Loader returns a dictionary keyed by tensor name with `mlx.core.array` values.
- Loaded tensor shapes and numeric values match fixture values.
- Exact-name and prefix filters work for loading.
- Loading without a selection, missing requested tensors, invalid filters, missing paths, and unsupported suffixes raise clear errors.
**Verification:** `uv run pytest tests/test_checkpoint.py`
**Auto-continue:** yes

### Slice 4: TRELLIS.2 Asset and Documentation Workflow

**Objective:** Connect checkpoint inspection/loading to the TRELLIS.2 local asset workflow without requiring real weights by default.
**Execution:** direct
**Depends on:** Slice 3
**Touches:** `src/mlx_spatial/model_assets.py`, `README.md`, `tests/test_model_assets.py`, `tests/test_checkpoint.py`
**Context budget:** ~8% of context window
**Produces:** README workflow and any refined manifest entries needed for checkpoint files.
**Acceptance criteria:**
- README documents local placement under `weights/trellis2/`.
- README documents manual download pattern, inspection API or command snippet, and MLX loading snippet.
- README states unsupported boundaries: no full inference, no block parity, no decoder, no mesh/GLB export, no automatic downloads.
- Asset validation remains deterministic and does not require real files in default tests.
**Verification:** `uv run pytest tests/test_model_assets.py tests/test_checkpoint.py`
**Auto-continue:** yes

### Slice 5: Full Verification

**Objective:** Prove the whole repository still satisfies default dependency and test constraints after checkpoint loading support.
**Execution:** direct
**Depends on:** Slice 4
**Touches:** test suite, documentation evidence
**Context budget:** ~4% of context window
**Produces:** final verification evidence for `auto-verify`.
**Acceptance criteria:**
- Full test suite passes.
- No real checkpoint artifacts are present in tracked source paths.
- Manual real-weight verification steps remain documentation-only.
**Verification:** `uv run pytest`
**Auto-continue:** no

## Execution Routing and Topology

- Slice 1: direct, auto-continue to Slice 2 after verification passes.
- Slice 2: direct, auto-continue to Slice 3 after verification passes.
- Slice 3: direct, auto-continue to Slice 4 after verification passes.
- Slice 4: direct, auto-continue to Slice 5 after verification passes.
- Slice 5: direct, checkpoint boundary before `auto-verify`.
- Parallel-safe groups: none. The slices share API and fixture contracts and should run serially.
- Subagents: not required. Consider `auto-eng-review` before execution because this adds a new public API and dependency boundary.

## Verification Commands

- Slice 1: `uv run pytest tests/test_checkpoint.py`
- Slice 2: `uv run pytest tests/test_checkpoint.py`
- Slice 3: `uv run pytest tests/test_checkpoint.py`
- Slice 4: `uv run pytest tests/test_model_assets.py tests/test_checkpoint.py`
- Slice 5: `uv run pytest`

## Context Budget For This Change

- Estimated total: ~33% of context window across planning and execution.
- Largest slice: Slice 2 or Slice 3 at ~8% each.
- Expected execution can remain direct because each slice is bounded to one subsystem plus tests/docs.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan is cleanly sliced around one data path from local safetensors metadata to selected MLX arrays, with deterministic fake-fixture tests guarding the dependency and no-network constraints.
- Concern: The exact safetensors API behavior and dependency placement are not yet proven in this repository, so Slice 1 must validate fixture writing and parsing before public API work continues.
- Action: Execute Slice 1 first and stop if safetensors cannot create and inspect tiny local fixtures without PyTorch, Transformers, Hugging Face Hub, network access, or real weights.
- Verified: PLAN.md and DESIGN.md reviewed for architecture fit, data flow, edge cases, test strategy, rollback safety, and dependency risk.

## Execution Evidence

- Slice 1: PASS. Added `safetensors` and the required `numpy` companion dependency for safetensors' MLX path; `uv run pytest tests/test_checkpoint.py` passed with `8 passed`.
- Slice 2: PASS. Added deterministic checkpoint metadata inspection with exact-name and prefix filters; `uv run pytest tests/test_checkpoint.py` passed with `8 passed`.
- Slice 3: PASS. Added selected tensor loading into MLX arrays; `uv run pytest tests/test_checkpoint.py` passed with `8 passed`.
- Slice 4: PASS. Added TRELLIS.2 checkpoint README workflow while preserving fake-file asset validation; `uv run pytest tests/test_model_assets.py tests/test_checkpoint.py` passed with `13 passed`.
- Slice 5: PASS. Full suite `uv run pytest` passed with `44 passed, 5 skipped`.
