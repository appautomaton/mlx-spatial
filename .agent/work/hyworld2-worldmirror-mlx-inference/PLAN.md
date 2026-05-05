# HY-World-2.0 WorldMirror MLX Inference Plan

## Goal

Build a staged MLX inference infrastructure for HY-World-2.0 WorldMirror 2.0 that can reconstruct scene/world geometry from image or video-frame inputs, starting with depth, normals, camera metadata, and point-cloud PLY outputs, while preserving official pipeline semantics and strict memory guards.

## Architecture Approach

Use the official HY-World-2.0 WorldMirror pipeline as the behavioral reference, but expose it through the repo's existing blocker-driven MLX style. The smallest correct design is a separate `hyworld2` command surface with modular asset validation, preprocessing, checkpoint inspection, MLX model execution, selectable heads, export helpers, and trace metadata.

The implementation should not try to port Voyager, training, FSDP, NCCL, CUDA `gsplat`, or real-time Gaussian rendering. Gaussian attributes can be represented as a staged head once depth/normal/points are real; rendering remains a later milestone.

## Ordered Task Sequence

### Slice 1: HY-World Asset And CLI Contract

**Objective:** Add a `mlx-spatial-hyworld2` command surface with deterministic asset validation and setup/help commands.
**Execution:** direct
**Depends on:** none
**Touches:** `pyproject.toml`, `src/mlx_spatial/model_assets.py` or `src/mlx_spatial/hyworld2_assets.py`, `src/mlx_spatial/hyworld2.py`, `tests/test_hyworld2_tools.py`
**Context budget:** ~8% of context window
**Produces:** `validate`, `inspect`, and `download-command` scaffolding for `weights/hy-world-2/HY-WorldMirror-2.0`.
**Acceptance criteria:**
- `validate` reports valid assets or names exact missing `model.safetensors` and config files.
- Config may be `config.yaml` or `config.json`; validation reports which one resolved.
- Runtime dependencies still exclude PyTorch, CUDA, `gsplat`, and Hugging Face runtime clients.
**Verification:** `uv run pytest -q tests/test_hyworld2_tools.py tests/test_model_assets.py`
**Auto-continue:** yes

### Slice 2: Reconstruction Trace, Blockers, And Path Policy

**Objective:** Add the `reconstruct` command skeleton with structured trace/blocker behavior and output-path validation.
**Execution:** direct
**Depends on:** Slice 1
**Touches:** `src/mlx_spatial/hyworld2.py`, `src/mlx_spatial/hyworld2_inference.py`, `tests/test_hyworld2_inference.py`
**Context budget:** ~8%
**Produces:** A no-fake reconstruction pipeline that can stop cleanly at missing input, missing assets, or unimplemented model stages.
**Acceptance criteria:**
- Output paths outside `outputs/` are rejected.
- The command records `completed_stages`, requested heads, memory profile, and blocker details.
- Missing weights/configs block at `asset-validation`; missing inputs block at `input-discovery`.
**Verification:** `uv run pytest -q tests/test_hyworld2_tools.py tests/test_hyworld2_inference.py`
**Auto-continue:** yes

### Slice 3: Official Input Preprocessing Contract

**Objective:** Port the official input discovery and image preprocessing contract for small image sets.
**Execution:** direct
**Depends on:** Slice 2
**Touches:** `src/mlx_spatial/hyworld2_preprocess.py`, `src/mlx_spatial/hyworld2_inference.py`, `tests/test_hyworld2_preprocess.py`, `tests/test_hyworld2_inference.py`
**Context budget:** ~10%
**Produces:** Image-directory/single-image preprocessing into MLX tensors shaped `[1, S, 3, H, W]` with traceable frame selection.
**Acceptance criteria:**
- Supports deterministic image ordering and a bounded frame count.
- Balanced profile resolves to a 518 target-size path for the first live milestone.
- Trace includes original size, processed size, frame count, patch grid, and estimated token count.
**Verification:** `uv run pytest -q tests/test_hyworld2_preprocess.py tests/test_hyworld2_inference.py`
**Auto-continue:** yes

### Slice 4: Checkpoint And Config Routing

**Objective:** Parse WorldMirror config/checkpoint metadata and map safetensors keys into official component groups without full model execution.
**Execution:** subagent recommended
**Depends on:** Slice 1
**Touches:** `src/mlx_spatial/hyworld2_assets.py`, `src/mlx_spatial/hyworld2_inference.py`, `tests/test_hyworld2_assets.py`, `tests/test_hyworld2_inference.py`
**Context budget:** ~10%
**Produces:** Config loader, safetensors key grouping, resolved model config, and staged readiness metadata.
**Acceptance criteria:**
- Fake safetensors/config fixtures can validate and inspect component key groups.
- Missing key groups produce precise blockers.
- Config defaults reflect official WorldMirror model fields needed for `VisualGeometryTransformer`, `CameraHead`, and DPT heads.
**Verification:** `uv run pytest -q tests/test_hyworld2_assets.py tests/test_hyworld2_inference.py`
**Auto-continue:** no

### Slice 5: MLX VisualGeometryTransformer Core

**Objective:** Implement the MLX backbone contract needed to produce intermediate token lists and `patch_start_idx`.
**Execution:** subagent recommended
**Depends on:** Slice 3, Slice 4
**Touches:** `src/mlx_spatial/hyworld2_worldmirror.py`, optional `src/mlx_spatial/hyworld2_layers.py`, `tests/test_hyworld2_worldmirror.py`
**Context budget:** ~15%
**Produces:** Patch/token assembly, special tokens, RoPE position handling, frame/global attention blocks, intermediate token capture, and memory guards.
**Acceptance criteria:**
- Small deterministic fixtures produce expected token shapes for model sizes used in tests.
- Query-chunked full attention matches dense attention on small fixtures.
- Activation/token guards block before unsafe global attention allocations.
- No approximate/windowed attention path is used in exact reconstruction.
**Verification:** `uv run pytest -q tests/test_hyworld2_worldmirror.py tests/test_hyworld2_inference.py`
**Auto-continue:** no

### Slice 6: MLX Camera And Dense Heads

**Objective:** Port the staged heads needed for camera metadata, depth, normals, and points.
**Execution:** subagent recommended
**Depends on:** Slice 5
**Touches:** `src/mlx_spatial/hyworld2_heads.py`, `src/mlx_spatial/hyworld2_worldmirror.py`, `tests/test_hyworld2_heads.py`
**Context budget:** ~15%
**Produces:** MLX `CameraHead`, DPT-style feature projection/fusion, official activations, and frame-chunked head execution.
**Acceptance criteria:**
- Depth, normal, point, and camera fixtures return official-shaped outputs.
- Head activation functions match deterministic references: `exp`, `expp1`, `inv_log`, `norm`, and `linear`.
- Frame-chunked head execution matches unchunked execution on small fixtures.
**Verification:** `uv run pytest -q tests/test_hyworld2_heads.py tests/test_hyworld2_worldmirror.py`
**Auto-continue:** no

### Slice 7: Staged Reconstruction Orchestration And Exports

**Objective:** Wire preprocessing, checkpoint routing, MLX model stages, selected heads, and concrete file exports into `reconstruct`.
**Execution:** subagent recommended
**Depends on:** Slice 6
**Touches:** `src/mlx_spatial/hyworld2_inference.py`, `src/mlx_spatial/hyworld2_export.py`, `src/mlx_spatial/hyworld2.py`, `tests/test_hyworld2_export.py`, `tests/test_hyworld2_inference.py`
**Context budget:** ~12%
**Produces:** Depth/normal image or array outputs, camera metadata, `points.ply`, and trace JSON for fixture-backed reconstruction.
**Acceptance criteria:**
- Fixture reconstruction writes non-empty staged outputs under `outputs/hyworld2/`.
- `--heads` enables only requested heads and disables/export-blocks the rest explicitly.
- Point-cloud PLY has vertices and deterministic color/coordinate formatting.
**Verification:** `uv run pytest -q tests/test_hyworld2_export.py tests/test_hyworld2_inference.py tests/test_hyworld2_tools.py`
**Auto-continue:** yes

### Slice 8: Gaussian Attribute Stage Contract

**Objective:** Add Gaussian-head staging without CUDA rendering or fake preview output.
**Execution:** direct
**Depends on:** Slice 7
**Touches:** `src/mlx_spatial/hyworld2_heads.py`, `src/mlx_spatial/hyworld2_inference.py`, `src/mlx_spatial/hyworld2_export.py`, `tests/test_hyworld2_inference.py`
**Context budget:** ~8%
**Produces:** A `gs` head route that either exports exact available Gaussian attributes or returns a structured renderer/export blocker.
**Acceptance criteria:**
- Requesting `--heads gs` never imports CUDA `gsplat`.
- If GS model pieces are unavailable, blocker names `gaussian-head` or `gaussian-export` exactly.
- Trace distinguishes Gaussian attributes from Gaussian rendering.
**Verification:** `uv run pytest -q tests/test_hyworld2_inference.py tests/test_hyworld2_heads.py`
**Auto-continue:** yes

### Slice 9: Live 518 Reconstruction Verification

**Objective:** Run the real command against local HY-World weights if present, otherwise prove the setup blocker is precise and actionable.
**Execution:** direct
**Depends on:** Slice 8
**Touches:** `outputs/hyworld2/`, verification notes only
**Context budget:** ~6%
**Produces:** Either a non-empty live `points.ply` plus trace, or a verified missing-weights blocker.
**Acceptance criteria:**
- With weights present, the live command writes non-empty point-cloud PLY and trace metadata.
- Without weights, `validate` and `reconstruct` name exact missing files and exit cleanly.
- Existing TRELLIS.2 tests still pass.
**Verification:** `uv run pytest -q tests/test_hyworld2_tools.py tests/test_hyworld2_inference.py tests/test_hyworld2_export.py && uv run pytest -q`
**Auto-continue:** no

## Execution Routing And Topology

- Slices 1-3 are a serial auto-continue chain because they build the CLI, blocker, and preprocessing surface.
- Slice 4 is a checkpoint because real checkpoint/config structure may force naming or routing corrections.
- Slices 5-7 are the core model and export implementation and should use subagents when execution mode allows it, but their write sets should remain ordered to avoid model-contract conflicts.
- Slice 8 can auto-continue after Slice 7 because it is a staged contract on top of the same prediction/export surface.
- Slice 9 is a live verification checkpoint and should not auto-continue.
- Parallel-safe groups: none for code-writing slices until Slice 5 is complete. During execution, read-only official-source audits may run in parallel with direct implementation, but code writes should follow the ordered slices above.

## Verification Commands

- Slice 1: `uv run pytest -q tests/test_hyworld2_tools.py tests/test_model_assets.py`
- Slice 2: `uv run pytest -q tests/test_hyworld2_tools.py tests/test_hyworld2_inference.py`
- Slice 3: `uv run pytest -q tests/test_hyworld2_preprocess.py tests/test_hyworld2_inference.py`
- Slice 4: `uv run pytest -q tests/test_hyworld2_assets.py tests/test_hyworld2_inference.py`
- Slice 5: `uv run pytest -q tests/test_hyworld2_worldmirror.py tests/test_hyworld2_inference.py`
- Slice 6: `uv run pytest -q tests/test_hyworld2_heads.py tests/test_hyworld2_worldmirror.py`
- Slice 7: `uv run pytest -q tests/test_hyworld2_export.py tests/test_hyworld2_inference.py tests/test_hyworld2_tools.py`
- Slice 8: `uv run pytest -q tests/test_hyworld2_inference.py tests/test_hyworld2_heads.py`
- Slice 9: `uv run pytest -q tests/test_hyworld2_tools.py tests/test_hyworld2_inference.py tests/test_hyworld2_export.py && uv run pytest -q`
- Final hygiene: `git diff --check`

## Context Budget For This Change

Estimated total implementation context is ~92% across multiple execution sessions. No single slice should exceed ~15%. The likely high-risk slices are Slice 5 and Slice 6 because they port the backbone attention and DPT-style heads; both should stop at structured blockers rather than stretching into live execution when parity or memory behavior is uncertain.

## Recommended Next Skill

`auto-execute`

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan has a clean staged architecture, concrete blockers, and testable slices that avoid PyTorch/CUDA runtime dependencies while preserving the official WorldMirror parity target.
- Concern: Slices 5 and 6 carry significant rework risk because the VisualGeometryTransformer and DPT heads combine large attention, RoPE, chunking, and checkpoint-key mapping that may expose parity gaps not visible in fixture-only tests.
- Action: Start `auto-execute` with Slice 1 and keep Slice 4 as a hard checkpoint before any full backbone/head port begins.
- Verified: PLAN.md, DESIGN.md, STATUS.md, active state pointers, official WorldMirror pipeline references, slice dependencies, verification commands, memory-guard requirements, and dependency boundaries were checked.

## Execution Evidence

- Slice 1 complete: added `mlx-spatial-hyworld2` CLI, WorldMirror asset validation, checkpoint inspection scaffolding, download-command, and tool tests.
- Slice 2 complete: added structured reconstruction trace/blockers, output path guard, head normalization, memory-profile routing, and no-fake reconstruction skeleton.
- Slice 3 complete: added official-style small image preprocessing into MLX `[1, S, 3, H, W]` tensors with sorted image discovery, 14-multiple target sizing, RGBA-over-white handling, and trace metadata.
- Slice 4 complete: added WorldMirror config parsing/defaults, checkpoint component grouping, requested-head-aware missing-group blockers, corrupt-safetensors blockers, and checkpoint metadata in reconstruction traces.
- Subagent orchestration: official-source explorer completed; repo-pattern explorer completed; Slice 4 implementer completed; Slice 4 spec review approved; Slice 4 quality review requested two fixes; fix implementer completed; quality re-review approved.
- Verification: `uv run pytest -q tests/test_hyworld2_tools.py tests/test_hyworld2_preprocess.py tests/test_hyworld2_assets.py tests/test_hyworld2_inference.py tests/test_model_assets.py` -> 39 passed.
- Live setup check: `uv run mlx-spatial-hyworld2 validate weights/hy-world-2` -> precise missing `HY-WorldMirror-2.0/model.safetensors` and config blocker.
- Full regression: `uv run pytest -q` -> 280 passed, 5 skipped.
- Hygiene: `git diff --check` on HY-World changed files -> passed.
- Stop reason: Slice 4 is the approved hard checkpoint before the Slice 5 VisualGeometryTransformer port.
- Slice 5 complete: added MLX VisualGeometryTransformer token assembly, official-style first/later-frame special token slots, optional condition slots, frame/global exact attention blocks, q/k RoPE application, intermediate patch-token capture, exact query-chunked attention, token/attention guards, and deterministic fixture tensor allocation guards.
- Subagent orchestration: Slice 5 implementer completed; official-source explorer completed; spec review requested frame/global/intermediate corrections then approved; quality review requested RoPE/condition/qkv fixes, then requested fixture-allocation memory guard, and approved after fix.
- Verification: `uv run pytest -q tests/test_hyworld2_worldmirror.py tests/test_hyworld2_inference.py` -> 25 passed.
- Bundle verification: `uv run pytest -q tests/test_hyworld2_tools.py tests/test_hyworld2_preprocess.py tests/test_hyworld2_assets.py tests/test_hyworld2_inference.py tests/test_hyworld2_worldmirror.py tests/test_model_assets.py` -> 54 passed.
- Full regression: `uv run pytest -q` -> 295 passed, 5 skipped.
- Hygiene: `git diff --check -- src/mlx_spatial/hyworld2_worldmirror.py tests/test_hyworld2_worldmirror.py` -> passed.
- Stop reason: Slice 5 has `Auto-continue: no`; next execution window starts with Slice 6 Camera and Dense Heads.
- Slice 6 complete: added fixture-backed MLX camera and DPT dense heads, official-shaped camera/depth/normal/points outputs, activation helpers for `exp`, `expp1`, `inv_log`, `norm`, and `linear`, full-intermediate camera token routing, patch-only DPT feature routing, frame chunking, per-chunk `mx.eval` memory boundaries, and structured blockers.
- Subagent orchestration: official-source explorer completed; repo-pattern explorer completed; Slice 6 implementer completed; coordinator requested full-intermediate camera parity and cleanup fixes; spec review approved; quality review requested chunk-eval and zero-frame blockers; quality re-review approved.
- Verification: `uv run pytest -q tests/test_hyworld2_heads.py tests/test_hyworld2_worldmirror.py` -> 30 passed.
- Hygiene: `git diff --check -- src/mlx_spatial/hyworld2_heads.py src/mlx_spatial/hyworld2_worldmirror.py tests/test_hyworld2_heads.py tests/test_hyworld2_worldmirror.py` -> passed.
- Stop reason: Slice 6 has `Auto-continue: no`; user requested continued ownership, so the next execution window starts with Slice 7 Staged Reconstruction Orchestration And Exports.
- Slice 7 complete: added explicit fixture reconstruction orchestration, deterministic depth/normal/camera/PLY/trace exports, selected-head execution and disabled-head metadata, Gaussian head blocker, output cleanup for reused fixture directories, and CLI `--fixture-tensors`.
- Subagent orchestration: Slice 7 implementer completed; spec review approved; quality review requested stale-output cleanup; quality re-review approved. Full-suite review also observed unrelated TRELLIS SLat dirty-worktree import drift outside Slice 7.
- Verification: `uv run pytest -q tests/test_hyworld2_export.py tests/test_hyworld2_inference.py tests/test_hyworld2_tools.py` -> 27 passed.
- Hygiene: `git diff --check -- src/mlx_spatial/hyworld2_inference.py src/mlx_spatial/hyworld2_export.py src/mlx_spatial/hyworld2.py tests/test_hyworld2_export.py tests/test_hyworld2_inference.py` -> passed.
- Continue reason: Slice 7 has `Auto-continue: yes`; next execution window starts with Slice 8 Gaussian Attribute Stage Contract.
- Slice 8 complete: added MLX Gaussian attribute staging through the dense-head path, deterministic `gaussian/attributes.npz` and metadata exports, trace metadata separating attributes from rendering, and a `gaussian-export` blocker instead of any CUDA `gsplat` renderer import or fake preview.
- Verification: `uv run pytest -q tests/test_hyworld2_inference.py tests/test_hyworld2_heads.py` -> 29 passed.
- Hygiene: `git diff --check -- src/mlx_spatial/hyworld2_heads.py src/mlx_spatial/hyworld2_inference.py src/mlx_spatial/hyworld2_export.py tests/test_hyworld2_inference.py tests/test_hyworld2_heads.py` -> passed.
- Continue reason: Slice 8 has `Auto-continue: yes`; next execution window starts with Slice 9 Live 518 Reconstruction Verification.
- Slice 9 complete: verified the live setup path against `weights/hy-world-2`, wrote a trace for the missing-weights reconstruction check, and fixed the unrelated TRELLIS SLat collection blocker by restoring exact chunked self-attention exports and guard behavior.
- Live setup check: `uv run mlx-spatial-hyworld2 validate weights/hy-world-2` -> `ready=False` with exact missing `HY-WorldMirror-2.0/model.safetensors` and `HY-WorldMirror-2.0/config.yaml or HY-WorldMirror-2.0/config.json`.
- Live reconstruct check: `uv run mlx-spatial-hyworld2 reconstruct weights/hy-world-2 inputs/trellis2/demo-rgb-background.png --output outputs/hyworld2/live-setup-check --heads depth,normal,points --memory-profile safe --trace-output outputs/hyworld2/live-setup-check/trace.json` -> clean `asset-validation` blocker with missing asset metadata.
- HY-World bundle verification: `uv run pytest -q tests/test_hyworld2_tools.py tests/test_hyworld2_preprocess.py tests/test_hyworld2_assets.py tests/test_hyworld2_inference.py tests/test_hyworld2_worldmirror.py tests/test_hyworld2_heads.py tests/test_hyworld2_export.py tests/test_model_assets.py` -> 78 passed.
- TRELLIS regression fix verification: `uv run pytest -q tests/test_trellis2_slat.py` -> 22 passed.
- Full regression: `uv run pytest -q` -> 319 passed, 5 skipped.
- Final hygiene: `git diff --check` -> passed.
- Stop reason: all approved slices are complete; live reconstruction awaits local HY-WorldMirror weights/config.
