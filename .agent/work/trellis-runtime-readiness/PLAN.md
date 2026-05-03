# PLAN: TRELLIS.2 MLX Runtime Readiness

## Goal

Build a TRELLIS.2 MLX runtime readiness layer that can execute a minimal weighted sparse convolution reference primitive, validate local model asset readiness, and document the bring-up path without downloading or loading real model weights by default.

## Architecture Approach

Extend the current sparse primitive layer in place for weighted sparse convolution, using the existing `(target_index, source_index, kernel_index)` map row contract and the verified gather/scatter semantics.

Add a small model-asset readiness module with a TRELLIS.2 manifest/config and deterministic validation against local files. Keep it dependency-free: no Hugging Face imports, no PyTorch imports, no network calls, no checkpoint loading.

Use README as the user-facing runtime readiness guide. The docs must preserve the spec thesis: define executable MLX sparse compute and asset validation boundaries before downloading checkpoints.

## Content Constraints

- **Artifact target:** `README.md`
- **Audience:** Engineers using this repo to build or port spatial model inference components; they know Python and MLX but need explicit sparse compute and local asset handling contracts.
- **Thesis:** The repo becomes ready for TRELLIS.2 weights by defining executable MLX sparse compute and asset validation boundaries first, not by downloading checkpoints before code can consume them.
- **Voice:** Technical reference voice. Short direct sentences. State constraints and commands plainly.
- **Content anti-goals:** no hype, no claims of full TRELLIS.2 inference, no vague future-work ending, no hidden dependency claims.
- **Channel:** README reference documentation.
- **Source policy:** existing repository behavior, this spec/design, and local vendor names only. Do not add external factual claims about remote model files unless marked as a placeholder pattern.
- **Factual risk:** medium. Technical claims must match implemented APIs and tests.
- **Format:** Markdown reference sections with commands and explicit unsupported boundaries.

## Ordered Task Sequence

### Slice 1: Weighted Sparse Convolution Reference

**Objective:** Add a public MLX weighted sparse convolution helper with exact small-array tests.

**Execution:** direct

**Depends on:** none

**Touches:** `src/mlx_spatial/sparse_conv.py`, `src/mlx_spatial/__init__.py`, `tests/test_sparse_weighted_conv.py`

**Context budget:** ~13% of context window

**Produces:** Public weighted sparse convolution helper and MLX-only tests.

**Acceptance criteria:**

- Helper is public and documented with map, feature, weight, and output shape contracts.
- Helper accepts `source_features` shaped `(source_count, in_channels)`, `map_rows` shaped `(m, 3)`, `kernel_weights` shaped `(kernel_count, in_channels, out_channels)`, and `target_count`.
- Helper computes `source_features[source_index] @ kernel_weights[kernel_index]` per map row and sums into `target_index`.
- Duplicate target indices accumulate deterministically.
- Empty maps return zero output shaped `(target_count, out_channels)`.
- Invalid map shape, non-integer map rows, invalid feature/weight rank, channel mismatch, out-of-bounds source/target/kernel indices, and invalid target count raise `ValueError`.
- Existing sparse map and gather/scatter tests continue to pass.

**Verification:** `uv run pytest tests/test_sparse_weighted_conv.py tests/test_sparse_feature.py tests/test_sparse_conv.py`

**Auto-continue:** yes

### Slice 2: Optional Weighted Parity Scaffold

**Objective:** Add an optional local PyTorch parity scaffold for weighted sparse convolution, skipped by default.

**Execution:** direct

**Depends on:** Slice 1

**Touches:** `tests/test_sparse_weighted_conv_parity.py`, `pyproject.toml` only if an existing marker is insufficient

**Context budget:** ~5% of context window

**Produces:** Gated PyTorch parity test for the weighted reference primitive.

**Acceptance criteria:**

- Test is marked `torch_parity` and skips unless `MLX_SPATIAL_RUN_TORCH_PARITY=1`.
- Test imports PyTorch only inside the gated loader.
- Test compares MLX output against an equivalent small PyTorch loop.
- PyTorch remains absent from base dependencies.

**Verification:** `uv run pytest tests/test_sparse_weighted_conv_parity.py tests/test_sparse_weighted_conv.py`

**Auto-continue:** yes

### Slice 3: TRELLIS.2 Asset Readiness

**Objective:** Add dependency-free TRELLIS.2 asset manifest/config and local validation tests using tiny fake files.

**Execution:** subagent recommended

**Depends on:** none

**Touches:** `src/mlx_spatial/model_assets.py`, `src/mlx_spatial/__init__.py`, `tests/test_model_assets.py`, `.gitignore`

**Context budget:** ~14% of context window

**Produces:** TRELLIS.2 asset manifest/config, validation helper, tests, and local weight ignore protection.

**Acceptance criteria:**

- Manifest/config names TRELLIS.2 and contains expected relative asset paths without storing weights.
- Validation helper accepts a local root path and reports deterministic present/missing entries without downloading or importing optional tooling.
- Missing-file and present-file behavior is covered with temporary fake files.
- Local `weights/` artifacts are ignored or the docs choose an out-of-repo cache convention; if using `weights/`, root `.gitignore` protects it.
- Default tests do not require network access, Hugging Face credentials, real model weights, vendors, or local absolute paths.

**Verification:** `uv run pytest tests/test_model_assets.py`

**Auto-continue:** yes

### Slice 4: Runtime Readiness Documentation

**Objective:** Update README to document weighted sparse compute, asset readiness, Hugging Face CLI pattern, validation workflow, and unsupported boundaries.

**Execution:** direct

**Depends on:** Slices 1 and 3

**Touches:** `README.md`

**Context budget:** ~8% of context window

**Produces:** README runtime readiness guide.

**Acceptance criteria:**

- README documents weighted sparse convolution shape contracts and deterministic accumulation semantics.
- README documents local asset convention and validation workflow.
- README includes a Hugging Face CLI download command pattern without requiring the CLI or network in default tests.
- README clearly states unsupported boundaries: no full TRELLIS.2 inference, no checkpoint loading, no decoder, no mesh/GLB export.
- README names concrete next slices: checkpoint inspection/loading, TRELLIS sparse block parity, decoder/mesh path.
- Content follows the specified audience, thesis, voice, source policy, and anti-goals.

**Verification:** `uv run pytest`

**Auto-continue:** no

## Execution Routing And Topology

- Slice 1 route: direct. It extends one existing sparse primitive subsystem and has targeted tests.
- Slice 2 route: direct. It adds optional parity scaffolding and reuses the existing parity marker.
- Slice 3 route: subagent recommended. It crosses runtime asset config, public exports, tests, and repository ignore policy.
- Slice 4 route: direct. It is a docs-only final integration slice, but content constraints must be checked during verification.
- Auto-continue chain: Slice 1 may continue into Slice 2 after targeted tests pass. Slice 2 may continue into Slice 3 after parity scaffold is confirmed skipped by default. Slice 3 may continue into Slice 4 after asset tests pass.
- Checkpoints: stop after Slice 4 for full verification.
- Parallel-safe groups: Slice 1 and Slice 3 are conceptually independent, but plan execution should remain serial because both may update `src/mlx_spatial/__init__.py`.
- Subagents: optional only for Slice 3. No subagent is required.

## Verification Commands

- Slice 1: `uv run pytest tests/test_sparse_weighted_conv.py tests/test_sparse_feature.py tests/test_sparse_conv.py`
- Slice 2: `uv run pytest tests/test_sparse_weighted_conv_parity.py tests/test_sparse_weighted_conv.py`
- Slice 3: `uv run pytest tests/test_model_assets.py`
- Slice 4: `uv run pytest`
- Optional local parity after default verification: `MLX_SPATIAL_RUN_TORCH_PARITY=1 uv run pytest -m torch_parity`

## Execution Evidence

- Slice 1: PASS. `uv run pytest tests/test_sparse_weighted_conv.py tests/test_sparse_feature.py tests/test_sparse_conv.py` completed with `17 passed` after correcting one test expectation arithmetic error.
- Slice 2: PASS. `uv run pytest tests/test_sparse_weighted_conv_parity.py tests/test_sparse_weighted_conv.py` completed with `5 passed, 1 skipped`.
- Slice 3: PASS. Subagent implementer, spec reviewer, and quality reviewer all returned approved statuses; coordinator reran `uv run pytest tests/test_model_assets.py` with `5 passed`.
- Slice 4: PASS. `uv run pytest` completed with `36 passed, 5 skipped`.
- Optional PyTorch parity remains gated by `MLX_SPATIAL_RUN_TORCH_PARITY=1` and skipped by default.

## Context Budget For This Change

Estimated total: ~40% of context window.

This is intentionally larger than prior primitive slices but remains one capability: runtime readiness for TRELLIS.2 sparse compute and local asset validation.

## Recommended Next Skill

`auto-verify`

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan has clear slice boundaries, explicit verification commands, and a simple data flow from sparse map rows through weighted MLX accumulation to local asset validation.
- Concern: The TRELLIS.2 asset manifest is necessarily provisional and the weighted sparse convolution layout is a reference contract rather than verified model-layer parity.
- Action: Proceed with `auto-execute`, keeping Slice 1 tests tied to the documented reference layout and Slice 3 documentation explicit that exact checkpoint filenames may be refined later.
- Verified: PLAN.md, DESIGN.md, execution topology, weighted sparse data flow, asset validation boundary, dependency boundary, edge-case coverage, and test strategy were checked.
