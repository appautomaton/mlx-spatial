# PLAN: TRELLIS.2 End-to-End Inference Attempt

## Goal

Create the first inference-only TRELLIS.2 image-to-3D attempt in `mlx_spatial` by tracing the working `vendors/trellis-mac` path, wiring the equivalent MLX pipeline stages around local real weights, and producing either a runnable partial result or a precise blocker ledger for missing MLX compute stages.

## Architecture Approach

- Start with read-only discovery of the `trellis-mac` and original TRELLIS.2 inference flow.
- Encode the discovered flow as a small inference-only pipeline contract with explicit stage names.
- Reuse existing TRELLIS.2 asset validation, checkpoint inspection, and real-weight probe loading.
- Implement readiness/dry-run first so the pipeline can prove local configuration without model compute.
- Implement an execution attempt that accepts an image path and stops at the first unimplemented MLX compute stage with a structured blocker.
- Treat a precise blocker ledger as a valid output for this first end-to-end attempt.

## Ordered Task Sequence

### Slice 1: Reference Flow Trace

**Objective:** Capture the real `trellis-mac` image-to-3D inference stage order and MLX replacement points.
**Execution:** direct
**Depends on:** none
**Touches:** `vendors/trellis-mac` read-only, `vendors/TRELLIS.2` read-only, `.agent/work/trellis-e2e-inference-attempt/FLOW.md`, `.agent/work/trellis-e2e-inference-attempt/DESIGN.md`
**Context budget:** ~10% of context window
**Produces:** `FLOW.md` with stage order, reference files/lines, model/config loading points, sampler/decoder/export path, and likely MLX blockers.
**Acceptance criteria:**
- Flow artifact names each image-to-3D stage in order.
- Flow artifact cites `trellis-mac` runnable reference and original TRELLIS.2 architecture files.
- Flow artifact identifies initial MLX replacement points and likely blockers.
- Runtime implementation remains absent or vendor-free in this slice.
**Verification:** direct review of `.agent/work/trellis-e2e-inference-attempt/FLOW.md`
**Auto-continue:** yes

### Slice 2: Pipeline Contract and Blocker Types

**Objective:** Implement a public inference-only pipeline contract with deterministic stage and blocker structures.
**Execution:** direct
**Depends on:** Slice 1
**Touches:** `src/mlx_spatial/trellis2_inference.py`, `src/mlx_spatial/__init__.py`, `tests/test_trellis2_inference.py`
**Context budget:** ~10% of context window
**Produces:** public pipeline API, stage model, blocker model, and unit tests.
**Acceptance criteria:**
- Pipeline exposes deterministic stage order matching `FLOW.md`.
- Blocker structure includes stage, operation, reference, reason, and next-slice recommendation.
- No runtime imports from `vendors/`.
- Public exports are covered by tests.
**Verification:** `uv run pytest tests/test_trellis2_inference.py`
**Auto-continue:** yes

### Slice 3: Readiness and Dry-Run Mode

**Objective:** Add readiness/dry-run behavior that validates assets, configs, weight probes, and stage wiring without model compute.
**Execution:** direct
**Depends on:** Slice 2
**Touches:** `src/mlx_spatial/trellis2_inference.py`, `tests/test_trellis2_inference.py`
**Context budget:** ~10% of context window
**Produces:** dry-run report for fake fixtures and local real-weight readiness path.
**Acceptance criteria:**
- Dry-run validates fake asset roots in default tests.
- Dry-run reports configured stages and whether each is ready, blocked, or unimplemented.
- Dry-run can call existing TRELLIS.2 probe tooling without requiring real weights in default tests.
- Errors for missing assets/configs are deterministic and tested.
**Verification:** `uv run pytest tests/test_trellis2_inference.py tests/test_trellis2_tools.py`
**Auto-continue:** yes

### Slice 4: Execution Attempt Mode

**Objective:** Add an image-path execution attempt that advances through implemented stages and stops at a precise missing-compute blocker.
**Execution:** direct
**Depends on:** Slice 3
**Touches:** `src/mlx_spatial/trellis2_inference.py`, `tests/test_trellis2_inference.py`
**Context budget:** ~10% of context window
**Produces:** attempt mode and fake-fixture tests for image input validation and blocker behavior.
**Acceptance criteria:**
- Attempt accepts an image path and validates file existence without needing real image model compute in default tests.
- Attempt runs readiness checks first.
- Attempt stops at the first unimplemented compute stage with a structured blocker rather than fake output.
- Tests cover missing image, valid fake image, and blocker ledger shape.
**Verification:** `uv run pytest tests/test_trellis2_inference.py`
**Auto-continue:** yes

### Slice 5: Real Local Attempt

**Objective:** Run the pipeline attempt against local `weights/trellis2/` and a local sample image or record a precise missing-input blocker.
**Execution:** direct
**Depends on:** Slice 4
**Touches:** ignored output paths only, `.agent/work/trellis-e2e-inference-attempt/ATTEMPT.md`
**Context budget:** ~8% of context window
**Produces:** `ATTEMPT.md` recording real asset readiness, sample image source, attempt command, output paths or blocker ledger.
**Acceptance criteria:**
- Real attempt validates `weights/trellis2/` if available.
- Real attempt uses a local sample image if available or generated as a tiny local fixture.
- Outcome is recorded as completed output paths or a precise blocker ledger.
- No generated large outputs are tracked.
**Verification:** `uv run python -c "from mlx_spatial import validate_trellis2_assets; r = validate_trellis2_assets('weights/trellis2'); print('ready=', r.ready); print('missing=', list(r.missing))"`
**Auto-continue:** yes

### Slice 6: Documentation and Full Verification

**Objective:** Document the inference attempt boundary and prove default tests still pass.
**Execution:** direct
**Depends on:** Slice 5
**Touches:** `README.md`, test suite, `.agent/work/trellis-e2e-inference-attempt/PLAN.md`
**Context budget:** ~6% of context window
**Produces:** README notes and final execution evidence.
**Acceptance criteria:**
- README documents readiness/dry-run, attempt mode, and blocker semantics.
- README states inference-only and no-training boundaries.
- Full test suite passes without real weights, network, Hugging Face credentials, PyTorch, Transformers, or vendor imports.
- Git status shows real weights and generated outputs are ignored or absent.
**Verification:** `uv run pytest`
**Auto-continue:** no

## Execution Routing and Topology

- Slice 1: direct, read-only discovery, auto-continue to Slice 2 after `FLOW.md` exists.
- Slice 2: direct, auto-continue to Slice 3 after targeted tests pass.
- Slice 3: direct, auto-continue to Slice 4 after targeted tests pass.
- Slice 4: direct, auto-continue to Slice 5 after targeted tests pass.
- Slice 5: direct, real local attempt checkpoint; continue to Slice 6 after `ATTEMPT.md` records output or blocker.
- Slice 6: direct, checkpoint boundary before `auto-verify`.
- Parallel-safe groups: none. Discovery, API contract, dry-run, and attempt mode share one stage/blocker contract.
- Subagents: not required, but `auto-eng-review` is recommended before execution because the scope intentionally accepts blockers and touches a new public inference surface.

## Verification Commands

- Slice 1: direct review of `.agent/work/trellis-e2e-inference-attempt/FLOW.md`
- Slice 2: `uv run pytest tests/test_trellis2_inference.py`
- Slice 3: `uv run pytest tests/test_trellis2_inference.py tests/test_trellis2_tools.py`
- Slice 4: `uv run pytest tests/test_trellis2_inference.py`
- Slice 5: `uv run python -c "from mlx_spatial import validate_trellis2_assets; r = validate_trellis2_assets('weights/trellis2'); print('ready=', r.ready); print('missing=', list(r.missing))"`
- Slice 6: `uv run pytest`

## Context Budget For This Change

- Estimated total: ~54% of context window across planning and execution.
- Largest slices: Slice 1 through Slice 4 at ~10% each.
- Expected outcome is a bounded pipeline attempt plus blocker evidence, not guaranteed full GLB generation.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan separates reference tracing, pipeline contract, dry-run readiness, attempt mode, and real execution evidence so the new inference surface can advance without pretending full TRELLIS.2 compute exists.
- Concern: Slice 5 may stop at the first missing MLX compute stage, so execution must record a precise blocker rather than expanding scope into a large unplanned model port.
- Action: Execute Slice 1 first and keep `FLOW.md` as the source of truth for stage order, then stop any later attempt at the first structured blocker that names the missing operation and next slice.
- Verified: PLAN.md and DESIGN.md reviewed for architecture fit, data flow, edge cases, test strategy, rollback safety, dependency risk, and blocker semantics.

## Execution Evidence

- Slice 1: PASS. Wrote `FLOW.md` with stage order, references to `vendors/trellis-mac` and `vendors/TRELLIS.2`, MLX replacement points, and expected first blockers.
- Slice 2: PASS. Added `Trellis2InferencePipeline`, stage constants, report dataclasses, and blocker dataclass; `uv run pytest tests/test_trellis2_inference.py` passed with `8 passed`.
- Slice 3: PASS. Added dry-run readiness checks for fake assets and optional probe loading; `uv run pytest tests/test_trellis2_inference.py tests/test_trellis2_tools.py` passed with `18 passed`.
- Slice 4: PASS. Added image-path attempt mode that stops at a structured blocker; `uv run pytest tests/test_trellis2_inference.py` passed with `8 passed`.
- Slice 5: PASS. Real local attempt against `weights/trellis2/` reported `ready=True`, completed stages `input-image`, `asset-config-validation`, and `checkpoint-probe-readiness`, then stopped at blocker stage `image-preprocessing-background` with reference `vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:127-162`.
- Slice 6: PASS. README documents readiness/dry-run, attempt mode, blocker semantics, inference-only scope, and no-training boundary; `uv run pytest` passed with `62 passed, 5 skipped`; `git status --short --ignored` shows `weights/` as ignored.
