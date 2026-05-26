# Plan: MapAnything MLX Parity Backbone

## Goal

Execute the first parity-first MapAnything MLX implementation slice from [SPEC.md](SPEC.md): assets/config, preprocessing parity, reference capture, first encoder-stage parity, and Desk smoke.

## Architecture Approach

Use a staged MLX port, not a broad model clone. The first executable model boundary is the image encoder prefix: DINOv2-style patch embedding plus block 0, backed by explicit checkpoint keys such as `encoder.model.patch_embed.*` and `encoder.model.blocks.0.*`. This boundary is small enough to parity-test and still sits on the path to full image-only MapAnything inference.

Runtime package code must not import `vendors/map-anything`, Torch, TorchVision, UniCeption, or OpenCV. Dev-only reference capture lives under `tools/` and is guarded by `MAPANYTHING_TORCH_REF=1`.

## Execution Routing and Topology

Default: direct, serial, continuation after verification.

Overrides:
- Slice 4: subagent recommended, because it ports model math and maps checkpoint tensors.

Parallel-safe groups: none. Each slice builds on artifacts and contracts from the previous slice.

Checkpoints: none.

## Requirement Traceability

| SPEC AC | Satisfied by |
|---|---|
| MA-01 | Slices 1, 4, 5 |
| MA-02 | Slices 1, 5 |
| MA-03 | Slice 1 |
| MA-04 | Slices 1, 4 |
| MA-05 | Slice 2 |
| MA-06 | Slices 3, 4 |
| MA-07 | Slice 5 |
| MA-08 | Slice 3 |
| MA-09 | Slice 5 |

## Ordered Slice Sequence

### Slice 1: MapAnything Assets and Config Boundary

**Objective:** Add local MapAnything asset/config inspection helpers that validate `weights/map-anything` and expose the config facts needed by later slices.

**Acceptance criteria:**
- `src/mlx_spatial/mapanything_assets.py` validates `config.json` and `model.safetensors` under a configurable root.
- The helper parses the local config into a small MLX-facing config object with encoder size, patch size, data norm type, info-sharing depth/dim/head count, and prediction output type.
- Checkpoint inspection reports deterministic groups for `encoder`, `info_sharing`, geometric encoders, dense head, pose head, and scale head.
- Tests prove missing files/config fields fail clearly and that the current local checkpoint group layout is recognized when present.

**Touches:** `src/mlx_spatial/mapanything_assets.py`, `src/mlx_spatial/__init__.py`, `tests/test_mapanything_assets.py`

**Produces:** Asset validation and checkpoint group API for later slices.

**Verification:** `uv run pytest tests/test_mapanything_assets.py tests/test_checkpoint.py -q`

**Status:** complete
**Evidence:** added `src/mlx_spatial/mapanything_assets.py`, public exports in `src/mlx_spatial/__init__.py`, and `tests/test_mapanything_assets.py`; adjusted stale heavy-framework dependency assertion in `tests/test_checkpoint.py` to allow existing `huggingface-hub`; `uv run pytest tests/test_mapanything_assets.py tests/test_checkpoint.py -q` passed with 18 tests.
**Risks / next:** none for Slice 1.

### Slice 2: Image Preprocessing Parity

**Objective:** Implement MLX/Pillow/NumPy MapAnything image loading that matches the vendored image-only `load_images(...)` contract without runtime Torch/TorchVision/UniCeption.

**Acceptance criteria:**
- `src/mlx_spatial/mapanything_preprocess.py` discovers supported images deterministically and implements the fixed-resolution mapping for `resolution_set=518`.
- Desk images preprocess to the expected 4:3 bucket `(H, W) = (392, 518)` with patch size 14.
- DINOv2 normalization constants are local and tested for dtype/range/shape behavior.
- Unit tests cover synthetic aspect ratios, unsupported inputs, RGB conversion, stride handling, and Desk metadata.
- Optional Torch reference comparison checks the MLX output against vendored `load_images(...)` for the Desk pair when `MAPANYTHING_TORCH_REF=1`.

**Touches:** `src/mlx_spatial/mapanything_preprocess.py`, `tests/test_mapanything_preprocess.py`, optional `tests/test_mapanything_preprocess_parity.py`

**Produces:** Runtime-safe preprocessing path and optional vendor parity test.

**Verification:** `uv run pytest tests/test_mapanything_preprocess.py -q && MAPANYTHING_TORCH_REF=1 uv run --group torch-ref pytest tests/test_mapanything_preprocess_parity.py -q`

**Status:** complete
**Evidence:** added `src/mlx_spatial/mapanything_preprocess.py`, public preprocessing exports, `tests/test_mapanything_preprocess.py`, and opt-in `tests/test_mapanything_preprocess_parity.py`; `uv run pytest tests/test_mapanything_preprocess.py -q` passed with 9 tests; `MAPANYTHING_TORCH_REF=1 uv run --group torch-ref pytest tests/test_mapanything_preprocess_parity.py -q` passed with 1 parity test against vendored `load_images`.
**Risks / next:** none for Slice 2.

### Slice 3: Dev-Only Torch Reference Capture

**Objective:** Add guarded parity-bundle utilities and a dev-only reference script for capturing vendored MapAnything tensors.

**Acceptance criteria:**
- `src/mlx_spatial/mapanything_parity.py` can write/load/compare `.npz` tensor bundles with metadata that states `runtime_depends_on_torch=False`.
- `tools/mapanything_dump_torch_reference.py` refuses to run unless `MAPANYTHING_TORCH_REF=1` is set.
- The reference script uses `torch-ref`, `vendors/map-anything`, `weights/map-anything`, and `inputs/map-anything/desk` to capture preprocessing output and the selected encoder-prefix tensors into `/tmp` or an explicit output path.
- Tests cover bundle round-trips, missing/value/shape mismatch reporting, and the env guard without requiring Torch.

**Touches:** `src/mlx_spatial/mapanything_parity.py`, `tools/mapanything_dump_torch_reference.py`, `tests/test_mapanything_parity.py`

**Produces:** The parity harness required for model-stage numeric checks.

**Verification:** `uv run pytest tests/test_mapanything_parity.py -q && python tools/mapanything_dump_torch_reference.py weights/map-anything inputs/map-anything/desk --output /tmp/mapanything-ref-guard.npz 2>/tmp/mapanything-ref-guard.err; test $? -eq 2; rg MAPANYTHING_TORCH_REF /tmp/mapanything-ref-guard.err`

**Status:** complete
**Evidence:** added `src/mlx_spatial/mapanything_parity.py`, public parity exports, `tools/mapanything_dump_torch_reference.py`, and `tests/test_mapanything_parity.py`; `uv run pytest tests/test_mapanything_parity.py -q` passed with 7 tests; plain-python guarded script invocation returned 2 before heavy imports, mentioned `MAPANYTHING_TORCH_REF`, and did not write the requested bundle.
**Risks / next:** Slice 4 must use a real env-enabled reference bundle generated by this script and verify the metadata has `torch_hub_disabled=true`.

### Slice 4: MLX Encoder Prefix Parity

**Objective:** Port the MapAnything image encoder prefix to MLX and verify it against captured Torch reference tensors.

**Acceptance criteria:**
- `src/mlx_spatial/mapanything_model.py` defines the first MLX model-stage boundary: patch embedding plus encoder block 0 for the local DINOv2 giant configuration.
- Weight loading maps `encoder.model.patch_embed.*` and `encoder.model.blocks.0.*` explicitly from `model.safetensors`.
- Tests verify tensor shapes, required key presence, dtype choices, and numerical parity against a deterministic captured reference bundle within documented tolerances.
- The implementation reuses existing MLX layer idioms where practical and avoids importing vendor or Torch code.

**Execution:** subagent recommended

**Depends on:** Slices 1, 2, 3

**Touches:** `src/mlx_spatial/mapanything_model.py`, `tests/test_mapanything_model.py`, optional small fixture under `tests/fixtures/mapanything/`

**Produces:** First real MLX model-stage implementation on the MapAnything path.

**Verification:** `uv run pytest tests/test_mapanything_model.py -q`

**Status:** complete
**Evidence:** added `src/mlx_spatial/mapanything_model.py`, public encoder-prefix exports, `tests/test_mapanything_model.py`, and `tests/fixtures/mapanything/encoder_prefix_tiny_reference.npz`; `uv run pytest tests/test_mapanything_model.py -q` passed with 7 tests; `uv run pytest tests/test_mapanything_*.py -q` passed with 33 tests and 1 expected optional skip. Generated `/tmp/mapanything-desk-prefix-reference.npz` with `MAPANYTHING_TORCH_REF=1 uv run --group torch-ref python tools/mapanything_dump_torch_reference.py weights/map-anything inputs/map-anything/desk --prefix-only --device mps --output /tmp/mapanything-desk-prefix-reference.npz`; metadata had `torch_hub_disabled=true`, `torch_hub_pretrained=false`, `prefix_only=true`, and `reference_model=vendored_dinov2_vitg14`. Real-weight Desk parity passed for `encoder.patch_embed`, `encoder.tokens`, and `encoder.block0` at `atol=1.2e-2`, `rtol=1e-3`; patch embedding max abs error was 0.0, tokens max abs error was 4.77e-7, and block 0 max/mean abs errors were 0.01055 / 7.86e-5.
**Risks / next:** Slice 5 should keep the tolerance metadata visible in trace output because block 0 parity is limited by attention backend numeric drift rather than preprocessing or positional interpolation.

### Slice 5: Desk Smoke and Package Boundary

**Objective:** Wire the asset, preprocessing, parity metadata, and encoder-prefix pieces into a first Desk smoke path without claiming full reconstruction.

**Acceptance criteria:**
- `src/mlx_spatial/mapanything_inference.py` exposes a small `MapAnythingPrefixPipeline` or equivalent that validates assets, preprocesses input images, runs the implemented MLX encoder prefix, and returns trace metadata/tensor summaries.
- A test runs the Desk pair through the smoke path using local assets when present and skips cleanly when weights are absent.
- Package/runtime imports work with `vendors/` absent from `PYTHONPATH`.
- No runtime dependency additions introduce Torch/TorchVision/UniCeption/OpenCV.
- Artifact hygiene checks confirm `vendors/`, `weights/`, and `inputs/` are excluded from wheel/sdist output.

**Depends on:** Slices 1, 2, 4

**Touches:** `src/mlx_spatial/mapanything_inference.py`, `src/mlx_spatial/__init__.py`, `tests/test_mapanything_inference.py`, optional `scripts/mapanything/smoke.py`, `scripts/README.md` if a script is added

**Produces:** First executable MapAnything MLX smoke surface for the official Desk images.

**Verification:** `uv run pytest tests/test_mapanything_assets.py tests/test_mapanything_preprocess.py tests/test_mapanything_model.py tests/test_mapanything_inference.py -q && uv lock --check && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build && python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-0.0.2.tar.gz dist/mlx_spatial-0.0.2-py3-none-any.whl`

**Status:** complete
**Evidence:** added `src/mlx_spatial/mapanything_inference.py`, public prefix-pipeline exports, and `tests/test_mapanything_inference.py`; the smoke pipeline validates encoder assets, preprocesses images, loads explicit encoder-prefix weights, runs patch embedding plus block 0, and returns tensor summaries/trace metadata without claiming full reconstruction. `uv run pytest tests/test_mapanything_*.py -q` passed with 39 tests and 1 expected optional skip. The exact Slice 5 verification command passed: 32 targeted tests, `uv lock --check`, `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build`, and `python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-0.0.2.tar.gz dist/mlx_spatial-0.0.2-py3-none-any.whl`; artifact hygiene reported `checked 2 artifact(s)`.
**Risks / next:** none for this slice; the implemented surface remains an encoder-prefix smoke path, not full MapAnything reconstruction.

## Aggregate Verification Commands

| Stage | Command |
|---|---|
| Default regression | `uv run pytest tests/test_mapanything_*.py -q` |
| Opt-in Torch reference | `MAPANYTHING_TORCH_REF=1 uv run --group torch-ref pytest tests/test_mapanything_preprocess_parity.py -q` |
| Package hygiene | `uv lock --check && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build && python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-0.0.2.tar.gz dist/mlx_spatial-0.0.2-py3-none-any.whl` |

## Review Recommendation

Run `auto-eng-review` before execution. The plan is bounded, but Slice 4 touches model math and checkpoint mapping, where a short engineering review is useful before code changes.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan follows the repo's existing MLX-first, asset-inspection, and parity-bundle patterns while choosing a small model boundary that can be verified before deeper MapAnything porting.
- Concern: Slice 3 and Slice 4 are sensitive to reference-capture drift because vendored MapAnything defaults can use Torch Hub unless the local-weight path explicitly forces `model.encoder.uses_torch_hub=false`.
- Action: Proceed to `auto-execute` starting at Slice 1, and when implementing Slice 3 require the guarded reference script to initialize MapAnything from local `weights/map-anything` config and checkpoint with Torch Hub disabled before any Slice 4 parity fixture is accepted.
- Verified: Read the canonical SPEC and PLAN, current Automaton state, vendored `load_images` and local initialization paths, checkpoint key groups, and existing checkpoint/assets/parity helper patterns.
