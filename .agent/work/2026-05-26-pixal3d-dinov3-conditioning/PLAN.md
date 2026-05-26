# Plan: Pixal3D DINOv3 Conditioning

## Goal

Execute [SPEC.md](SPEC.md): make Pixal3D runtime build sparse-stage projection conditioning from an input image through the existing MLX DINOv3 helper.

## Architecture Approach

Use the shared TRELLIS.2 DINOv3 functions already exported by `mlx_spatial`: `prepare_dinov3_image_tensor`, `assess_dinov3_mlx_conditioning`, and the existing DINOv3 asset convention. Pixal3D owns only orchestration, CLI plumbing, blocker metadata, and projection artifact handoff.

## Execution Routing and Topology

Default route: direct, serial.

Parallel-safe groups: none.

Checkpoint after: none. The user has approved continuing the active goal; this plan stays inside the known missing runtime boundary.

## Requirement Traceability

| SPEC ID | Satisfied by |
|---|---|
| PXDINO-01 | Slice 1 |
| PXDINO-02 | Slice 1 |
| PXDINO-03 | Slice 1 |
| PXDINO-04 | Slice 2 |
| PXDINO-05 | Slice 2 |

## Ordered Slice Sequence

### Slice 1: Runtime DINOv3 Wiring

**Objective:** Route Pixal3D manual-FOV inference from input image to MLX DINOv3 hidden states and sparse projection conditioning.

**Acceptance criteria:**
- `Pixal3DInferencePipeline.generate` accepts `dino_root`.
- Package CLI and script expose `--dino-root`.
- Missing DINOv3 assets return an actionable `image-conditioning` blocker.
- Fake DINOv3 assets let a valid image reach `sparse_projection.npz` without `projection_hidden_states`.

**Touches:** `src/mlx_spatial/pixal3d_inference.py`, `src/mlx_spatial/pixal3d.py`, `scripts/pixal3d/generate.py`, `tests/test_pixal3d_inference.py`, `tests/test_pixal3d_pipeline.py`

**Verification:** `uv run pytest tests/test_pixal3d_inference.py tests/test_pixal3d_pipeline.py tests/test_trellis2_dinov3.py tests/test_trellis2_dinov3_forward.py -q`

**Status:** complete
**Evidence:** added `dino_root` plumbing to `Pixal3DInferencePipeline.generate`, `mlx-spatial-pixal3d generate`, and `scripts/pixal3d/generate.py`; runtime now validates local DINOv3 assets, prepares a 512px DINOv3 image tensor, calls the shared MLX DINOv3 conditioning helper, and feeds returned hidden states into sparse projection conditioning. Added fake-DINOv3 tests for both missing-asset blocker metadata and image-to-`sparse_projection.npz` execution, and corrected shared fake DINOv3 conditioning to include the CLS token in synthetic hidden states. Verification passed: `uv run pytest tests/test_trellis2_dinov3.py tests/test_trellis2_forward.py tests/test_pixal3d_pipeline.py -q` -> `40 passed, 2 deselected`; `uv run pytest tests/test_pixal3d_*.py -q` -> `48 passed`.
**Risks / next:** only the sparse-stage DINOv3 path is wired; Pixal3D sparse flow, decoder handoff, high-resolution NAF features, and GLB export remain later blockers.

### Slice 2: Docs and Release Hygiene

**Objective:** Make the new DINOv3 runtime boundary clear in docs and prove package hygiene remains intact.

**Acceptance criteria:**
- `docs/pixal3d.md` and Pixal3D script help describe `--dino-root` and current remaining blocker.
- Pixal3D targeted tests pass.
- AST forbidden import scan remains clean.
- Lock/build/artifact hygiene remains valid for the current package version.

**Depends on:** Slice 1

**Touches:** `docs/pixal3d.md`, `.agent/work/2026-05-26-pixal3d-dinov3-conditioning/PLAN.md`

**Verification:** `uv run pytest tests/test_pixal3d_*.py -q && uv run mlx-spatial-pixal3d generate --help && uv run python scripts/pixal3d/generate.py --help && uv run python - <<'PY'\nimport ast\nfrom pathlib import Path\nforbidden = {'torch','torchvision','cv2','nvdiffrast','cumesh','natten','flash_attn'}\nviolations = []\nfor path in Path('src/mlx_spatial').rglob('*.py'):\n    tree = ast.parse(path.read_text(), filename=str(path))\n    for node in ast.walk(tree):\n        if isinstance(node, ast.Import):\n            for alias in node.names:\n                if alias.name.split('.')[0] in forbidden:\n                    violations.append(f'{path}:{node.lineno}: import {alias.name}')\n        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split('.')[0] in forbidden:\n            violations.append(f'{path}:{node.lineno}: from {node.module} import ...')\nif violations:\n    print('\\n'.join(violations))\n    raise SystemExit(1)\nprint('AST forbidden import scan passed')\nPY\nuv lock --check && git diff --check && rm -rf dist && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build && python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-*.tar.gz dist/mlx_spatial-*-py3-none-any.whl`

**Status:** complete
**Evidence:** updated `docs/pixal3d.md`, `README.md`, `docs/architecture.md`, and `scripts/README.md` to document `--dino-root`, DINOv3 asset setup, the new sparse projection artifact path, and the remaining sparse-flow/decoder blockers. Verification passed: `uv run pytest tests/test_pixal3d_*.py -q` -> `48 passed`; `uv run mlx-spatial-pixal3d generate --help` and `uv run python scripts/pixal3d/generate.py --help` show `--dino-root`; AST forbidden import scan passed; `uv lock --check` and `git diff --check` passed; `rm -rf dist && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build && python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-*.tar.gz dist/mlx_spatial-*-py3-none-any.whl && python scripts/packaging/check_release_artifacts.py --git-hygiene` built and checked `dist/mlx_spatial-0.0.3.tar.gz` plus `dist/mlx_spatial-0.0.3-py3-none-any.whl`; full `uv run pytest -q` -> `855 passed, 10 skipped, 27 deselected, 2 warnings`.
**Risks / next:** `weights/pixal3d` is not present in this checkout, so real Pixal3D checkpoint execution was not smoke-tested; the next implementation cycle should target sparse flow/checkpoint execution and sparse decoder handoff.
