# PLAN: TRELLIS.2 Real-Weight Tooling

## Goal

Add reusable TRELLIS.2 real-weight tooling that validates local assets, inspects configured safetensors checkpoints, and load-probes selected tensor groups into MLX arrays without requiring real weights, network access, Hugging Face tooling, PyTorch, Transformers, or vendor imports in default tests.

## Architecture Approach

- Build a small TRELLIS.2 orchestration module over the existing asset and checkpoint helpers.
- Represent probe groups as static data with checkpoint-relative paths, exact names or prefixes, and reference notes.
- Use `vendors/trellis-mac` as the primary reference during execution to choose conservative groups and the needed HF asset targets, while keeping implementation free of vendor imports.
- Add tests with temporary fake TRELLIS.2 roots and tiny safetensors files to prove validation, inspection, probe matching, and load probing.
- Add `huggingface_hub` as a dev-only dependency for the `huggingface-cli` operator workflow, not as a base runtime dependency.
- Add a thin CLI entrypoint only if it remains a direct wrapper over the public API and can be tested without real weights.

## Ordered Task Sequence

### Slice 1: Reference-Aware Probe Group Design

**Objective:** Define conservative TRELLIS.2 probe groups and local tooling contracts without importing vendor code.
**Execution:** direct
**Depends on:** none
**Touches:** `vendors/trellis-mac` read-only, `vendors/TRELLIS.2` read-only, `.agent/work/trellis-real-weight-tools/DESIGN.md`
**Context budget:** ~8% of context window
**Produces:** finalized probe group contract notes in `DESIGN.md` or plan execution evidence.
**Acceptance criteria:**
- Probe groups identify checkpoint-relative paths and exact names or prefixes.
- Each probe group has a reference note naming whether it is based on `trellis-mac`, original TRELLIS.2 awareness, or conservative placeholder behavior.
- No implementation imports vendor modules or reads vendor paths at runtime.
**Verification:** direct review of `DESIGN.md` plus later tests in Slice 2 and Slice 3
**Auto-continue:** yes

### Slice 2: Dev-Only HF CLI Boundary

**Objective:** Make `huggingface-cli` available as dev tooling without adding Hugging Face packages to base runtime dependencies.
**Execution:** direct
**Depends on:** Slice 1
**Touches:** `pyproject.toml`, `uv.lock`, `tests/test_trellis2_tools.py`
**Context budget:** ~5% of context window
**Produces:** dev dependency declaration and tests that assert Hugging Face tooling is dev-only.
**Acceptance criteria:**
- `huggingface_hub` is present only in the dev dependency group.
- Base runtime dependencies still exclude Hugging Face Hub, PyTorch, and Transformers.
- Tests assert the dependency boundary.
**Verification:** `uv run pytest tests/test_trellis2_tools.py`
**Auto-continue:** yes

**Execution correction:** `huggingface-hub==1.13.0` does not expose a `cli` extra in this environment; the dev dependency is declared as `huggingface-hub>=0.36`, which provides the `huggingface-cli` command.

**Execution correction:** Real TRELLIS.2 flow checkpoints contain BF16 tensors; `safetensors.mlx` metadata inspection works but selected BF16 materialization raises a `TypeError`, so `load_checkpoint_tensors` falls back to `mlx.core.load` for BF16 files and then returns only the requested names.

**Execution correction:** `huggingface-cli` is deprecated and refused to run in this environment; the supported dev command is `uv run hf download ...`, and the reusable command helper returns that form.

### Slice 3: TRELLIS.2 Tooling API

**Objective:** Implement public TRELLIS.2 helpers for asset validation, configured checkpoint inspection, and named probe selection.
**Execution:** direct
**Depends on:** Slice 2
**Touches:** `src/mlx_spatial/trellis2.py`, `src/mlx_spatial/__init__.py`, `tests/test_trellis2_tools.py`
**Context budget:** ~10% of context window
**Produces:** tested public API for fake-root validation and deterministic checkpoint inspection.
**Acceptance criteria:**
- API validates a fake TRELLIS.2 asset root using `TRELLIS2_ASSETS` semantics and reports deterministic readiness.
- API inspects fake safetensors checkpoints by configured checkpoint path and by named probe group.
- API surfaces missing roots, missing checkpoint files, unsupported formats, empty selections, and no-match probes with clear errors.
- Public exports are covered by tests.
**Verification:** `uv run pytest tests/test_trellis2_tools.py`
**Auto-continue:** yes

### Slice 4: Load-Probe API and Optional CLI

**Objective:** Add reusable load-probe behavior for named TRELLIS.2 probe groups and optionally expose it through thin CLI commands.
**Execution:** direct
**Depends on:** Slice 3
**Touches:** `src/mlx_spatial/trellis2.py`, `pyproject.toml`, `tests/test_trellis2_tools.py`
**Context budget:** ~10% of context window
**Produces:** load-probe API and, if small enough, CLI entrypoint(s) for local operator use.
**Acceptance criteria:**
- Load-probe returns MLX arrays or deterministic MLX-derived summaries for selected fake tensors.
- Output ordering is deterministic by group, checkpoint, and tensor name.
- Empty probe selections and no-match groups raise clear errors.
- If CLI is added, it runs against fake fixtures in tests and does not require real weights or downloads.
**Verification:** `uv run pytest tests/test_trellis2_tools.py`
**Auto-continue:** yes

### Slice 5: Operator Documentation

**Objective:** Document the manual real-weight workflow for validation, inspection, and load probing without committing real outputs.
**Execution:** direct
**Depends on:** Slice 4
**Touches:** `README.md`, `tests/test_trellis2_tools.py`
**Context budget:** ~6% of context window
**Produces:** README instructions for HF CLI setup/download expectation, local root convention, validation, inspection, load-probe commands, and unsupported boundaries.
**Acceptance criteria:**
- README states HF CLI is installed separately and is not a runtime dependency.
- README documents `weights/trellis2/` as the local root and keeps automatic downloads out of code and tests.
- README includes validation, inspection, and load-probe examples using the implemented API or CLI.
- README states no full inference, model construction, block execution, decoder, mesh/GLB, `.pt`/`.pth`, or real-weight report is included.
**Verification:** `uv run pytest tests/test_trellis2_tools.py`
**Auto-continue:** yes

### Slice 6: Real-Weight Download Attempt

**Objective:** Attempt to download the needed TRELLIS.2 model assets with dev-provided `huggingface-cli` into ignored `weights/trellis2/` and validate the resulting local layout when environment conditions permit.
**Execution:** direct
**Depends on:** Slice 5
**Touches:** ignored `weights/trellis2/`, execution evidence only
**Context budget:** ~6% of context window
**Produces:** download/validation evidence or a clear blocker report if repo ID, network, credentials, or disk space prevent the attempt.
**Acceptance criteria:**
- Download command uses `uv run huggingface-cli` or equivalent dev-environment invocation.
- Download target is `weights/trellis2/` and no downloaded artifacts are tracked.
- If the download succeeds, `validate_model_assets("weights/trellis2")` reports deterministic readiness details.
- If the download cannot run, the blocker is explicit and default tests remain unaffected.
**Verification:** `uv run python -c "from mlx_spatial import validate_model_assets; r = validate_model_assets('weights/trellis2'); print('ready=', r.ready); print('present=', list(r.present)); print('missing=', list(r.missing))"`
**Auto-continue:** yes

### Slice 7: Full Verification

**Objective:** Prove the repository still satisfies default local-test and dependency boundaries after real-weight tooling is added.
**Execution:** direct
**Depends on:** Slice 6
**Touches:** test suite, dependency config, artifact evidence
**Context budget:** ~4% of context window
**Produces:** final execution evidence for `auto-verify`.
**Acceptance criteria:**
- Full test suite passes.
- No real weights or generated real-weight outputs are committed.
- Base dependencies still exclude Hugging Face Hub, PyTorch, Transformers, and vendor imports; Hugging Face Hub is dev-only.
**Verification:** `uv run pytest`
**Auto-continue:** no

## Execution Routing and Topology

- Slice 1: direct, read-only reference discovery, auto-continue to Slice 2 after the probe group contract is recorded.
- Slice 2: direct, auto-continue to Slice 3 after dependency-boundary tests pass.
- Slice 3: direct, auto-continue to Slice 4 after `uv run pytest tests/test_trellis2_tools.py` passes.
- Slice 4: direct, auto-continue to Slice 5 after `uv run pytest tests/test_trellis2_tools.py` passes.
- Slice 5: direct, auto-continue to Slice 6 after targeted tests pass.
- Slice 6: direct, best-effort external download checkpoint; continue to full verification whether it succeeds or reports an environmental blocker.
- Slice 7: direct, checkpoint boundary before `auto-verify`.
- Parallel-safe groups: none. Probe group contracts, API, CLI, and docs share one contract and should remain serial.
- Subagents: not required. Use `auto-eng-review` before execution because this adds public tooling and optional command entrypoints.

## Verification Commands

- Slice 1: direct review of `DESIGN.md` and absence of runtime vendor imports; verified concretely by later tests.
- Slice 2: `uv run pytest tests/test_trellis2_tools.py`
- Slice 3: `uv run pytest tests/test_trellis2_tools.py`
- Slice 4: `uv run pytest tests/test_trellis2_tools.py`
- Slice 5: `uv run pytest tests/test_trellis2_tools.py`
- Slice 6: `uv run python -c "from mlx_spatial import validate_model_assets; r = validate_model_assets('weights/trellis2'); print('ready=', r.ready); print('present=', list(r.present)); print('missing=', list(r.missing))"`
- Slice 7: `uv run pytest`

## Context Budget For This Change

- Estimated total: ~44% of context window across planning and execution.
- Largest slices: Slice 2 and Slice 3 at ~10% each.
- Reference discovery is intentionally bounded to selecting probe group names and checkpoint-relative paths, not porting vendor implementation.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan keeps the real-weight path isolated behind dev-only Hugging Face tooling while preserving fake-fixture tests and runtime dependency boundaries.
- Concern: Slice 6 depends on external conditions and exact TRELLIS.2 asset identity, so it may produce a blocker report instead of downloaded weights even when the code slices are correct.
- Action: Execute Slices 1-5 first, then attempt Slice 6 only with `uv run huggingface-cli` into ignored `weights/trellis2/` and record any repo, network, credential, or disk-space blocker explicitly.
- Verified: PLAN.md and DESIGN.md reviewed for architecture fit, data flow, edge cases, test strategy, rollback safety, dependency risk, and external download isolation.

## Execution Evidence

- Slice 1: PASS. Probe groups were defined for sparse structure flow, shape SLat flow, texture SLat flow, shape decoder, and texture decoder; reference basis was recorded in `DESIGN.md` from `vendors/trellis-mac/generate.py` and original TRELLIS.2 checkpoint loading code.
- Slice 2: PASS. Added `huggingface-hub>=0.36` as a dev-only dependency; `uv lock` resolved it; `uv run pytest tests/test_trellis2_tools.py` passed with `10 passed`.
- Slice 3: PASS. Added `mlx_spatial.trellis2` validation and inspection APIs with public exports; `uv run pytest tests/test_trellis2_tools.py` passed with `10 passed`.
- Slice 4: PASS. Added load-probe API and `mlx-spatial-trellis2` CLI; `uv run pytest tests/test_checkpoint.py tests/test_trellis2_tools.py` passed with `18 passed`.
- Slice 5: PASS. README documents dev HF CLI, local root, validation, inspection, load-probe commands, and unsupported boundaries; targeted tests passed with `10 passed`.
- Slice 6: PASS. `uv run huggingface-cli download ...` was blocked by CLI deprecation, then `uv run hf download microsoft/TRELLIS.2-4B --local-dir weights/trellis2` completed with local path `weights/trellis2`; validation reported `ready=True`, all 12 manifest paths present, and no missing paths.
- Slice 6 real load probes: PASS. `shape-decoder`, `texture-decoder`, `sparse-structure-flow`, `shape-slat-flow`, and `texture-slat-flow` probes loaded selected real tensors into MLX arrays; BF16 flow checkpoints required the recorded `mx.load` fallback.
- Slice 7: PASS. Full suite `uv run pytest` passed with `54 passed, 5 skipped`; `git status --short --ignored` shows `weights/` as ignored.
