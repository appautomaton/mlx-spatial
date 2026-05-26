# Plan: Pixal3D MLX Support

## Goal

Execute the Pixal3D support contract in [SPEC.md](SPEC.md): local Pixal3D assets through an MLX image-to-3D pipeline, with a package CLI, documented settings, runtime dependency hygiene, and AppleGPU memory guards.

## Architecture Approach

Use [DESIGN.md](DESIGN.md). Treat Pixal3D as its own model family, but reuse existing TRELLIS.2 MLX primitives wherever the math and checkpoint structure match. The main new work is Pixal3D projection conditioning, projection attention integration, camera setup, and export/runtime orchestration.

Runtime and dev boundaries:

- Runtime: `src/mlx_spatial/pixal3d*.py`, MLX/NumPy/Pillow, local safetensors/JSON helpers, existing Mac-native export helpers.
- Dev reference: `tools/`, `torch-ref` group, `vendors/Pixal3D`, explicit `PIXAL3D_TORCH_REF=1`.

## Execution Routing and Topology

Default: direct, serial, continue after each verified slice.

Overrides:

- Slice 3: subagent recommended because projection conditioning touches DINOv3, camera geometry, grid sampling, and NAF behavior.
- Slice 4: subagent recommended because projection attention changes the shared sparse/dense flow path.
- Slice 6: subagent recommended because export may cross mesh, PBR texture, and GLB boundaries.

Parallel-safe groups: none. The slices depend on prior source/asset and model-boundary work.

Checkpoints: none. Entry into execution itself is the Automaton checkpoint.

## Requirement Traceability

| SPEC ID | Satisfied by |
|---|---|
| PIXAL3D-01 | Slices 2, 7 |
| PIXAL3D-02 | Slices 1, 2 |
| PIXAL3D-03 | Slices 1, 7 |
| PIXAL3D-04 | Slice 3 |
| PIXAL3D-05 | Slices 4, 5 |
| PIXAL3D-06 | Slice 5 |
| PIXAL3D-07 | Slice 6 |
| PIXAL3D-08 | Slice 6 |
| PIXAL3D-09 | Slice 7 |
| PIXAL3D-10 | Slice 7 |

## Ordered Slice Sequence

### Slice 1: Source, Asset, and Reference Boundary

**Objective:** Add Pixal3D asset manifests and guarded PyTorch reference tooling so later MLX work has real source/weight evidence.

**Acceptance criteria:**
- `weights/pixal3d` validation checks `pipeline.json` plus required Pixal3D JSON/safetensors checkpoint paths.
- CLI helper can print the HF download command for `TencentARC/Pixal3D`.
- Checkpoint inspection groups cover sparse structure flow/decoder, shape 512/1024 flows, texture 1024 flow, shape decoder, and texture decoder.
- Dev reference capture refuses to run unless `PIXAL3D_TORCH_REF=1` is set and never affects runtime imports.

**Touches:** `src/mlx_spatial/model_assets.py`, `src/mlx_spatial/pixal3d_assets.py`, `src/mlx_spatial/pixal3d.py`, `tools/pixal3d_dump_torch_reference.py`, `tests/test_pixal3d_assets.py`, `tests/test_pixal3d_parity.py`

**Verification:** `uv run pytest tests/test_pixal3d_assets.py tests/test_pixal3d_parity.py -q`

**Status:** complete
**Evidence:** added `PIXAL3D_ASSETS`, `src/mlx_spatial/pixal3d_assets.py`, `src/mlx_spatial/pixal3d_parity.py`, and `tools/pixal3d_dump_torch_reference.py`; targeted Pixal3D Slice 1-2 test run passed with `20 passed`, including asset manifest/config/probe coverage and the `PIXAL3D_TORCH_REF` reference guard.
**Risks / next:** heavy Torch tensor capture is metadata-only for this slice; Slice 3 owns projection parity tensors once the MLX projection boundary exists.

### Slice 2: Public Runtime Skeleton and CLI

**Objective:** Add a Pixal3D runtime API/CLI/script skeleton that validates inputs and reports structured blockers before model execution is filled in.

**Acceptance criteria:**
- `pyproject.toml` exposes `mlx-spatial-pixal3d`.
- `uv run mlx-spatial-pixal3d --help`, `download-command`, `validate`, and `inspect` work without Torch.
- A recommended script under `scripts/pixal3d/` prints clear defaults and routes to the package CLI.
- Runtime result/blocker dataclasses define the sample input, camera params, completed stages, output artifact path, and exact blocker stage.

**Depends on:** Slice 1

**Touches:** `pyproject.toml`, `src/mlx_spatial/pixal3d.py`, `src/mlx_spatial/pixal3d_inference.py`, `scripts/pixal3d/generate.py`, `tests/test_pixal3d_cli.py`, `tests/test_pixal3d_inference.py`

**Verification:** `uv run pytest tests/test_pixal3d_cli.py tests/test_pixal3d_inference.py -q && uv run mlx-spatial-pixal3d --help`

**Status:** complete
**Evidence:** added `mlx-spatial-pixal3d`, package CLI subcommands, a recommended `scripts/pixal3d/generate.py` wrapper, and a runtime skeleton that validates inputs/assets/config and returns structured blockers at camera or projection-conditioning boundaries; `uv run mlx-spatial-pixal3d --help`, `uv run python scripts/pixal3d/generate.py --help`, `git diff --check`, and the targeted Pixal3D tests passed.
**Risks / next:** runtime intentionally stops before model execution; Slice 3 begins projection conditioning without changing shared TRELLIS.2 flow math yet.

### Slice 3: Projection Conditioning

**Objective:** Implement MLX Pixal3D image conditioning: DINOv3 global tokens plus camera-aware projected 3D-grid features for sparse, shape, and texture stages.

**Acceptance criteria:**
- MLX projection grid matches upstream coordinate convention, front-view transform, FOV projection, feature sampling, and grid resolutions.
- Manual FOV path can build projection conditioning without MoGe.
- DINOv3 global and projected feature shapes match Pixal3D stage expectations.
- NAF-dependent 2048-channel projection behavior is either implemented or represented by a concrete blocker with lower-risk partial stage coverage where possible.
- Tests cover projection math and shape contracts without requiring real multi-GB Pixal3D weights.

**Execution:** subagent recommended

**Depends on:** Slices 1, 2

**Touches:** `src/mlx_spatial/pixal3d_projection.py`, `src/mlx_spatial/trellis2_dinov3*.py` if extension is needed, `tests/test_pixal3d_projection.py`

**Verification:** `uv run pytest tests/test_pixal3d_projection.py -q`

**Status:** complete
**Evidence:** added `src/mlx_spatial/pixal3d_projection.py`, public package exports, and `tests/test_pixal3d_projection.py`; the projection tests cover the rotated grid convention, front-view distance transform, center projection, BHWC/BCHW bilinear sampling, sparse-stage global/projected shapes, and the NAF-dependent 2048-channel blocker/override path. `uv run pytest tests/test_pixal3d_projection.py -q` passed with `10 passed`; the combined Pixal3D target set passed with `30 passed`, the Pixal3D runtime forbidden-import scan was clean, and `git diff --check` passed.
**Risks / next:** NAF itself remains explicitly blocked until an MLX equivalent is implemented; Slice 4 owns projection-attention integration and shared TRELLIS.2 regression coverage.

### Slice 4: Projection Attention and Flow Boundaries

**Objective:** Extend MLX flow execution so Pixal3D `image_attn_mode="proj"` checkpoints can run sparse-structure and SLat blocks.

**Acceptance criteria:**
- Dense and sparse cross-attention support the Pixal3D `ProjectAttention` behavior: global cross-attention plus per-token `proj_linear(proj_context)`.
- Pixal3D flow configs parse `image_attn_mode` and `proj_in_channels` without breaking existing TRELLIS.2 configs.
- Sparse structure flow/decoder and SLat flow probes can run through at least block-0 with Pixal3D checkpoint shapes.
- Existing TRELLIS.2 tests continue to pass.

**Execution:** subagent recommended

**Depends on:** Slice 3

**Touches:** `src/mlx_spatial/trellis2_sparse_structure.py`, `src/mlx_spatial/trellis2_slat.py`, optional `src/mlx_spatial/pixal3d_flow.py`, `tests/test_pixal3d_flow.py`, selected TRELLIS.2 regression tests

**Verification:** `uv run pytest tests/test_pixal3d_flow.py tests/test_trellis2_forward.py tests/test_trellis2_slat.py -q`

**Status:** complete
**Evidence:** extended sparse-structure and SLat flow configs with `image_attn_mode` and `proj_in_channels`, added config-gated Pixal3D `proj` attention that computes upstream-style `global_cross_attention + proj_linear(projected_context)`, and kept TRELLIS.2 `cross` mode on the existing path. Added `tests/test_pixal3d_flow.py` for Pixal3D config parsing, dense/sparse projection-attention math, and wrapper-key checkpoint probes. Verification passed: `uv run pytest tests/test_pixal3d_flow.py tests/test_trellis2_forward.py tests/test_trellis2_slat.py -q` -> `55 passed, 2 deselected`; `uv run pytest tests/test_pixal3d_*.py -q` -> `36 passed`; the targeted forbidden-import scan and `git diff --check` passed.
**Risks / next:** Slice 5 must align projected grid features to real sparse coordinates during orchestration; Slice 4 only proves the model block boundary accepts already-aligned projection context.

### Slice 5: Pixal3D Pipeline Orchestration and Camera

**Objective:** Wire the Pixal3D cascade path through preprocessing, camera params, sparse structure, shape 512-to-HR cascade, texture 1024, and decoders with memory-aware stage transitions.

**Acceptance criteria:**
- Pipeline supports `1024_cascade` and `1536_cascade`, with `1024_cascade` as the recommended Apple Silicon default.
- Manual FOV path works end to end through the implemented MLX model boundaries.
- Auto-camera path reuses existing MLX MoGe support or returns a structured MoGe asset/implementation blocker.
- `max_num_tokens` and HR resolution reduction follow upstream Pixal3D behavior.
- Trace metadata records stages, selected resolution, token counts, timing, and MLX memory counters when available.

**Depends on:** Slice 4

**Touches:** `src/mlx_spatial/pixal3d_camera.py`, `src/mlx_spatial/pixal3d_inference.py`, `tests/test_pixal3d_pipeline.py`

**Verification:** `uv run pytest tests/test_pixal3d_pipeline.py -q`

**Status:** complete
**Evidence:** added `src/mlx_spatial/pixal3d_camera.py` for upstream-compatible manual-FOV distance, cascade stage planning, and the HR token-limit reduction loop; updated `src/mlx_spatial/pixal3d_inference.py` to record camera params, sampler config, stage plan, timings, and MLX memory snapshots. Auto-camera now returns a structured MoGe blocker, while manual FOV can either block honestly at Pixal3D DINOv3 hidden-state extraction or, when `projection_hidden_states` are supplied, run through sparse-stage projection conditioning and stop at the sparse-structure flow boundary. Verification passed: `uv run pytest tests/test_pixal3d_pipeline.py -q` -> `3 passed`; `uv run pytest tests/test_pixal3d_*.py -q` -> `44 passed`; Slice 4 shared-flow regression rerun passed with `55 passed, 2 deselected`; `git diff --check` passed.
**Risks / next:** runtime still does not extract real Pixal3D DINOv3 hidden states from an image or execute full sparse/decoder/texture checkpoints; Slice 6 owns output artifact behavior from the implemented boundary.

### Slice 6: Output Artifact and Sample Generation

**Objective:** Produce a user-visible Pixal3D output from a sample image, with GLB when supported or a precise export blocker after completed MLX predictions.

**Acceptance criteria:**
- A sample input is available under `inputs/pixal3d/` or the script documents use of `vendors/Pixal3D/assets/images/0_img.png` without packaging it.
- Recommended script runs the Pixal3D pipeline with explicit output under `outputs/pixal3d/<run>/`.
- Successful path writes textured GLB when Mac-native export is compatible.
- If textured GLB is blocked, output includes completed MLX intermediate predictions and a blocker naming the missing export operation, dependency, and next implementation target.
- Optional real-weight smoke command uses `/tmp` or `outputs/` only.

**Execution:** subagent recommended

**Depends on:** Slice 5

**Touches:** `src/mlx_spatial/pixal3d_export.py`, `src/mlx_spatial/pixal3d_inference.py`, `scripts/pixal3d/generate.py`, `tests/test_pixal3d_export.py`

**Verification:** `uv run pytest tests/test_pixal3d_export.py -q && uv run python scripts/pixal3d/generate.py vendors/Pixal3D/assets/images/0_img.png --root weights/pixal3d --output-dir /tmp/pixal3d-smoke --pipeline-type 1024_cascade --manual-fov 0.2`

**Status:** complete
**Evidence:** added `src/mlx_spatial/pixal3d_export.py` and `tests/test_pixal3d_export.py`; a completed sparse projection boundary now writes `sparse_projection.npz` with global/projected features plus metadata, and the trace records artifact paths. Updated `scripts/pixal3d/generate.py` help plus `scripts/README.md` to use the vendored sample `vendors/Pixal3D/assets/images/0_img.png` without packaging it. Verification passed: `uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py tests/test_pixal3d_inference.py tests/test_pixal3d_cli.py -q` -> `15 passed`; `uv run pytest tests/test_pixal3d_*.py -q` -> `46 passed`; `uv run python scripts/pixal3d/generate.py --help` passed. `weights/pixal3d` is absent in this checkout, so the exact real-weight smoke could not run; a fake-root smoke with the vendored sample wrote `/tmp/pixal3d-smoke/trace.json` and returned the expected structured `image-conditioning` blocker.
**Risks / next:** textured GLB output remains blocked until Pixal3D DINOv3 hidden-state extraction, checkpoint execution, decoder handoff, and texture export are wired.

### Slice 7: Docs, Regression, and Package Hygiene

**Objective:** Prove Pixal3D support is documented, dependency-clean, and non-regressive across the package.

**Acceptance criteria:**
- README, docs index, architecture docs, release docs, and scripts README mention Pixal3D consistently.
- `docs/pixal3d.md` explains assets, license/access notes, recommended settings, memory profile, and output/export boundary.
- Runtime import scan shows no forbidden Torch/CUDA/vendor imports in `src/mlx_spatial`.
- Existing full test suite passes.
- Lock/build/artifact checks pass and exclude local assets/vendors/generated outputs.

**Depends on:** Slices 1-6

**Touches:** `README.md`, `docs/README.md`, `docs/architecture.md`, `docs/development.md`, `docs/release.md`, `docs/pixal3d.md`, `scripts/README.md`, tests as needed

**Verification:** `uv run pytest -q && uv run python <AST forbidden import scan> && uv lock --check && rm -rf dist && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build && python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-*.tar.gz dist/mlx_spatial-*-py3-none-any.whl && python scripts/packaging/check_release_artifacts.py --git-hygiene`

**Status:** complete
**Evidence:** added `docs/pixal3d.md` and updated README, docs index, architecture, development, release, and scripts docs for the Pixal3D CLI, asset root, recommended `1024_cascade` manual-FOV command, vendored sample image, current blockers, and intermediate NPZ artifact. Verification passed: `uv run pytest tests/test_pixal3d_*.py -q` -> `46 passed`; `uv run mlx-spatial-pixal3d --help` and `uv run python scripts/pixal3d/generate.py --help` passed; full `uv run pytest -q` -> `853 passed, 10 skipped, 27 deselected, 2 warnings`; AST runtime import scan found no forbidden `torch`, `torchvision`, `cv2`, `nvdiffrast`, `cumesh`, `natten`, or `flash_attn` imports; `uv lock --check`, `git diff --check`, build, artifact checker, and git hygiene passed. After live release checks showed `v0.0.2` was already tagged, released, and available on PyPI, the release metadata was retargeted to `0.0.3`; the fresh build produced `dist/mlx_spatial-0.0.3.tar.gz` and `dist/mlx_spatial-0.0.3-py3-none-any.whl`, with `docs/pixal3d.md`, `scripts/pixal3d/generate.py`, `pixal3d_camera.py`, and `pixal3d_export.py` present in the expected artifacts.
**Risks / next:** the original regex-style import scan is overly broad for this repo because existing non-Pixal3D modules mention upstream package names in docstrings/status text; AST import scanning is the actionable runtime dependency check.

## Aggregate Verification Commands

| Gate | Command |
|---|---|
| Pixal3D targeted tests | `uv run pytest tests/test_pixal3d_*.py -q` |
| Runtime forbidden import scan | `! rg -n "import (torch|torchvision|cv2)|from (torch|torchvision|cv2)|vendors/Pixal3D|o_voxel|nvdiffrast|cumesh|natten|flash_attn" src/mlx_spatial` |
| Full regression | `uv run pytest -q` |
| Package hygiene | `uv lock --check && rm -rf dist && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build && python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-*.tar.gz dist/mlx_spatial-*-py3-none-any.whl && python scripts/packaging/check_release_artifacts.py --git-hygiene` |

## Review Recommendation

Run `auto-eng-review` before `auto-execute`. This plan modifies shared TRELLIS.2 flow primitives and introduces large-model Pixal3D runtime paths, so a short engineering review is useful before code changes.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan preserves the runtime/dev split, names concrete asset and model boundaries, and gives each risky Pixal3D stage a targeted verification command.
- Concern: The riskiest path is Slice 3 through Slice 5 because NAF-derived 2048-channel projection features and Pixal3D projection attention may force changes in shared TRELLIS.2 flow code.
- Action: Execute Slice 1 and Slice 2 first, then require projection and TRELLIS.2 regression tests before any shared flow changes are treated as complete.
- Verified: canonical SPEC, DESIGN, PLAN, Pixal3D HF config paths, local vendor state, and existing TRELLIS.2 flow/projection touchpoints were checked.
