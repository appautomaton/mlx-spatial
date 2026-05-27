# mlx-spatialkit Native GLB Core Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-native-glb-core/SPEC.md`: build a native `mlx-spatialkit` package that converts Pixal3D decoded NPZ artifacts into a textured GLB for `mlx-spatial`.

## Architecture Approach

Use the companion design in `.agent/work/2026-05-27-mlx-spatialkit-native-glb-core/DESIGN.md`. The execution rule is strict: C++ / Objective-C++ / Metal 4 owns the mesh, texture-bake, and GLB hot paths; Python owns thin validation, I/O orchestration, and tests.

## Execution Routing And Topology

- Default execution: direct.
- Subagent routes: recommended for read-only risk audits, implementation review, and disjoint worker slices after ownership is assigned.
- Parallel-safe groups: none; slices touch the same native package and integration boundary.
- Checkpoints: none inside the approved plan. Execution should continue slice-by-slice after each verification passes.
- Review recommendation: run `auto-eng-review` before `auto-execute` because this plan introduces native C++/Metal code, package build plumbing, and memory/thread-safety constraints.
- Commit rhythm: commit after verified slices or small verified slice groups. Before each commit, inspect `git status --short`, review the staged diff, exclude `/tmp` outputs and ignored heavy artifacts, and do not push or tag unless the user explicitly asks.

## Agentic Collaboration Model

- Use parallel agents for read-only audits when they can answer independent questions, such as CMake/package risk, Metal runtime risk, GLB/export correctness, or mesh algorithm parity.
- Use worker agents only after file ownership is explicit and disjoint; workers must not edit the same native files concurrently.
- Keep dependent implementation slices serial at the integration boundary: build skeleton -> contracts -> mesh extraction -> mesh processing -> GLB/UV -> Metal bake -> real fixture -> `mlx-spatial` integration.
- Main agent owns integration, final diff review, verification, and commits.
- Subagents may run verification in parallel with non-overlapping implementation, but final pass/fail claims come from main-agent command output.

## Slice Order At A Glance

| Order | Slice | Purpose | Depends on |
|---|---|---|---|
| 1 | Native package and build skeleton | Prove the nested package, native extension, CMake, nanobind, and Metal probe can build. | none |
| 2 | Python API and native input contracts | Establish the thin Python surface and native Pixal3D NPZ validation. | 1 |
| 3 | Native FlexiDualGrid mesh extraction | Move shape fields -> triangle mesh into C++ with parity tests. | 2 |
| 4 | Native mesh metrics, cleanup, simplification | Make extracted meshes diagnosable and export-ready enough for first GLB success. | 3 |
| 5 | Native UV interface and GLB writer | Produce UV-ready geometry and write valid GLB containers. | 4 |
| 6 | Metal 4 texture bake | Move texture/PBR raster and sampling hot loops into Metal. | 5 |
| 7 | Real Pixal3D fixture export | Prove decoded Pixal3D fixture -> GLB under `/tmp` with diagnostics. | 6 |
| 8 | mlx-spatial integration and docs | Connect the native backend to `mlx-spatial` and document usage/cleanliness rules. | 7 |

## Ordered Slice Sequence

### Slice 1: Native Package And Build Skeleton

**Objective:** Create the nested `packages/mlx-spatialkit` package with `scikit-build-core`, `nanobind`, CMake, minimal native extension import, and Metal toolchain probe.

**Acceptance criteria:**
- `packages/mlx-spatialkit/pyproject.toml`, `CMakeLists.txt`, `src/mlx_spatialkit`, `cpp`, `metal`, and `tests` exist.
- Package-local build dependencies are limited to native build tooling such as `scikit-build-core` and `nanobind`; runtime dependencies do not include MLX, Torch, or `mlx-spatial`.
- Wheel and sdist builds succeed on macOS arm64 when `xcrun metal -v` works.
- The build compiles a tiny Objective-C++ bridge and `.metal` kernel so Metal support is proven at compile time.
- Build failure is explicit when the Metal compiler is missing.

**Verification:** `xcrun metal -v && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear && python - <<'PY'\nimport pathlib, tarfile, zipfile\nblocked = {'inputs', 'outputs', 'vendors', 'weights'}\nfor path in pathlib.Path('/tmp/mlx-spatialkit-dist').glob('*'):\n    if path.suffix == '.whl':\n        names = zipfile.ZipFile(path).namelist()\n    elif ''.join(path.suffixes[-2:]) == '.tar.gz':\n        with tarfile.open(path) as archive:\n            names = archive.getnames()\n    else:\n        continue\n    bad = [n for n in names if blocked.intersection(pathlib.PurePosixPath(n).parts)]\n    assert not bad, (path, bad[:5])\nprint('artifact clean')\nPY`

**Touches:** `packages/mlx-spatialkit/**`

**Status:** complete
**Evidence:** Added `packages/mlx-spatialkit` with `scikit-build-core`/`nanobind` build files, a minimal `_native` module, Objective-C++ Metal bridge, and `.metal` probe kernel; `xcrun metal -v && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear && ...` passed and printed `artifact clean`; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit pytest tests -q` passed with `1 passed`.
**Risks / next:** Slice 2 must replace the placeholder Python API surface with native Pixal3D input-contract validators.

### Slice 2: Python API And Native Input Contracts

**Objective:** Add thin Python APIs and native validators for Pixal3D decoded shape/texture contracts.

**Acceptance criteria:**
- Public APIs validate `Nx4 int32`, `Nx7 float32`, and `Nx6 float32` contracts with clear errors.
- Validation crosses the nanobind boundary; Python does not reimplement native validation logic beyond path/NPZ loading.
- Tests cover valid arrays, wrong dtype, wrong rank, wrong channel count, empty inputs, and unsupported batch cases.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit pytest tests/test_contracts.py -q`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit`, `packages/mlx-spatialkit/cpp`, `packages/mlx-spatialkit/tests`

### Slice 3: Native FlexiDualGrid Mesh Extraction

**Objective:** Port FlexiDualGrid field-to-mesh extraction from the current Python `ovoxel.py` behavior into C++.

**Acceptance criteria:**
- C++ extraction produces deterministic vertices/faces for small synthetic fixtures.
- Synthetic parity tests compare native output to `src/mlx_spatial/ovoxel.py:flexi_dual_grid_fields_to_mesh`.
- The implementation handles no-quad cases and validation failures without crashes or partial outputs.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit pytest tests/test_flexi_dual_grid.py -q`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/cpp/flexi_dual_grid.cpp`, `packages/mlx-spatialkit/src/mlx_spatialkit/mesh.py`, `packages/mlx-spatialkit/tests`

### Slice 4: Native Mesh Metrics, Cleanup, And Simplification Interface

**Objective:** Add native mesh diagnostics plus enough cleanup/simplification behavior to prepare extracted Pixal3D meshes for UV/export.

**Acceptance criteria:**
- Metrics include vertex/face counts, degenerate faces, duplicate faces, boundary edges, nonmanifold edges, components, and export-blocking reasons.
- Cleanup removes degenerates, duplicates, unreferenced vertices, and small components in native code.
- Simplification is behind a native-owned interface and can reduce fixture meshes enough for downstream UV/export.
- Tests prove diagnostics are actionable and no Python per-face loops own the cleanup path.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit pytest tests/test_mesh_processing.py -q`

**Depends on:** Slice 3

**Touches:** `packages/mlx-spatialkit/cpp/mesh_metrics.cpp`, `packages/mlx-spatialkit/cpp/mesh_cleanup.cpp`, `packages/mlx-spatialkit/cpp/simplify.cpp`, `packages/mlx-spatialkit/tests`

### Slice 5: Native UV Interface And GLB Writer

**Objective:** Add the UV-ready geometry interface and native GLB writer needed for a textured GLB.

**Acceptance criteria:**
- UV interface is native-owned and replaceable; C++ `xatlas` wrapping is allowed if isolated.
- GLB writer emits valid GLB header/chunks, mesh buffers, texture image payloads, materials, and metadata needed for Pixal3D exports.
- Tests validate GLB structure and basic scene metadata without relying on the real fixture.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit pytest tests/test_glb_writer.py -q`

**Depends on:** Slice 4

**Touches:** `packages/mlx-spatialkit/cpp/glb_writer.cpp`, `packages/mlx-spatialkit/cpp/*uv*`, `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests`

### Slice 6: Metal 4 Texture Bake

**Objective:** Implement the Metal-backed texture/PBR bake path for UV raster/sampling work.

**Acceptance criteria:**
- Objective-C++ bridge compiles and loads a Metal 4 pipeline.
- Texture bake returns base-color/PBR texture buffers and coverage diagnostics.
- Tests cover a tiny synthetic mesh/voxel case with deterministic output.
- Failure paths report missing Metal/toolchain/device errors clearly.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit pytest tests/test_texture_bake.py -q`

**Depends on:** Slice 5

**Touches:** `packages/mlx-spatialkit/metal`, `packages/mlx-spatialkit/cpp/bindings.cpp`, `packages/mlx-spatialkit/src/mlx_spatialkit/texture.py`, `packages/mlx-spatialkit/tests`

### Slice 7: Real Pixal3D Fixture Export

**Objective:** Wire the native pipeline end-to-end for the ignored real Pixal3D decoded fixture and write the generated GLB plus diagnostics to `/tmp`.

**Acceptance criteria:**
- `export_pixal3d_glb(...)` consumes `inputs/mlx-spatialkit/pixal3d-1024-cascade-decoded-pbr`.
- Generated `model.glb` and diagnostics JSON are written under `/tmp`.
- Diagnostics include stage timings, memory samples where practical, mesh counts, cleanup counts, UV stats, texture coverage, and output byte size.
- The GLB is valid and previewable enough for the existing reference viewer/manual inspection path.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy`

**Depends on:** Slice 6

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests`, `/tmp/mlx-spatialkit-*`

### Slice 8: mlx-spatial Integration And Docs

**Objective:** Add the `mlx-spatial` integration boundary and documentation so Pixal3D export can use `mlx_spatialkit` coherently when available.

**Acceptance criteria:**
- `mlx-spatial` integration is explicit and does not make `mlx-spatialkit` a required runtime dependency unless deliberately chosen in implementation.
- Existing Pixal3D export tests are updated or extended for the native-backend selection behavior.
- Docs explain decoded NPZ -> GLB usage, heavy fixture expectations, `/tmp` output policy, and fallback behavior.
- Package artifact checks confirm ignored local assets are not bundled.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear && python - <<'PY'\nimport pathlib, tarfile, zipfile\nblocked = {'inputs', 'outputs', 'vendors', 'weights'}\nfor path in pathlib.Path('/tmp/mlx-spatialkit-dist').glob('*'):\n    if path.suffix == '.whl':\n        names = zipfile.ZipFile(path).namelist()\n    elif ''.join(path.suffixes[-2:]) == '.tar.gz':\n        with tarfile.open(path) as archive:\n            names = archive.getnames()\n    else:\n        continue\n    bad = [n for n in names if blocked.intersection(pathlib.PurePosixPath(n).parts)]\n    assert not bad, (path, bad[:5])\nprint('artifact clean')\nPY`

**Depends on:** Slice 7

**Touches:** `src/mlx_spatial`, `docs`, `scripts`, `tests`, `packages/mlx-spatialkit`

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| SK-01 | Slice 1 |
| SK-02 | Slice 1 |
| SK-03 | Slices 1, 8 |
| SK-04 | Slice 2 |
| SK-05 | Slice 3 |
| SK-06 | Slice 4 |
| SK-07 | Slice 4 |
| SK-08 | Slice 6 |
| SK-09 | Slice 5 |
| SK-10 | Slice 7 |
| SK-11 | Slice 8 |
| SK-12 | Slices 7, 8 |

## Aggregate Verification Commands

| Stage | Command |
|---|---|
| Build skeleton | `xcrun metal -v && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear` |
| Package tests | `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit pytest tests -q` |
| Root integration tests | `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` |
| Heavy real fixture | `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy` |

## Execution Notes

- Use `/tmp/mlx-spatialkit-*` for generated GLBs, diagnostics, build dist outputs, and heavy scratch data.
- Read the existing Python reference before replacing behavior in each slice.
- Keep commits and diffs narrow: package scaffold first, native contracts second, integration last.
- Preserve a clean commit rhythm during implementation: verified, logically scoped commits; no generated heavy artifacts; no push/tag without explicit user direction.
- Do not mark the objective complete until the real Pixal3D decoded fixture exports a valid GLB through `mlx_spatialkit` and the integration/documentation checks pass.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan has a coherent native-first slice order, explicit C++/Metal ownership of hot paths, real Pixal3D fixture verification, artifact cleanliness checks, and a practical collaboration model.
- Concern: The highest-risk areas remain native packaging, Metal compile/runtime behavior, CPU UV dependency isolation, mesh simplification quality, and memory pressure on the real fixture.
- Action: Start `auto-execute` with Slice 1 and do not advance past each slice until its verification command passes and the diff is reviewed for generated artifacts.
- Verified: Automaton state, SPEC.md, DESIGN.md, PLAN.md, current Pixal3D/TRELLIS export hot-path references, Metal compiler availability, package command shape, and parallel packaging audit result.
