# mlx-spatialkit Export Quality Hardening Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-export-quality-hardening/SPEC.md`: make the real Pixal3D turtle export visually coherent and make spatialkit quality limitations explicit instead of hidden behind valid GLB tests.

## Architecture Approach

Use `.agent/work/2026-05-27-mlx-spatialkit-export-quality-hardening/DESIGN.md`. The execution rule is strict: Python may orchestrate and inspect small metadata, but native C++/Objective-C++/Metal must own texture fill, coverage counting, mesh processing, and GLB payload generation.

## Execution Routing And Topology

- Default execution: direct, serial, continuation after each slice verification.
- Subagent routes: recommended for read-only audits and final quality review; optional for implementation only with disjoint file ownership.
- Parallel-safe groups: none for implementation; texture, export diagnostics, tests, and docs share contracts.
- Checkpoints: none inside the approved plan. Execution should continue slice-by-slice after verification passes.
- Review recommendation: run `auto-eng-review` before `auto-execute` because this change touches native Metal texture paths and quality gates.
- Commit rhythm: commit after verified slices or small verified slice groups; do not push, tag, or release.

## Ordered Slice Sequence

### Slice 1: Texture Coverage Diagnostics

**Objective:** Make native texture bake diagnostics distinguish raw exact hits, surface texels, fallback/fill status, and final visible coverage.

**Acceptance criteria:**
- Texture stats expose `uv_surface_texel_count`, `exact_sampled_texel_count`, `fallback_filled_texel_count`, `missing_texel_count`, `out_of_grid_texel_count`, `visible_base_color_texel_count`, `raw_coverage_ratio`, and `final_visible_coverage_ratio`.
- Existing `sampled_texel_count` either remains as a compatibility alias or is replaced with clearly documented migration in tests.
- Synthetic tests include exact hits, missing sparse voxels, and no-face texels.
- Python wrappers only forward native stats; they do not compute per-texel coverage.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_texture_bake.py -q`

**Touches:** `packages/mlx-spatialkit/metal/texture_bake.mm`, `packages/mlx-spatialkit/metal/kernels/texture_bake.metal`, `packages/mlx-spatialkit/src/mlx_spatialkit/texture.py`, `packages/mlx-spatialkit/tests/test_texture_bake.py`

### Slice 2: Native Texture Fill/Fallback

**Objective:** Fill missing UV-rasterized surface texels through a native fallback path so the base-color texture is visually coherent instead of sparse dots.

**Acceptance criteria:**
- Missing surface texels receive coherent base color through native C++/Objective-C++/Metal logic.
- Diagnostics report raw exact-hit coverage separately from final visible coverage.
- Synthetic tests prove fallback fill changes missing surface texels from zero RGBA to visible color and increments fallback counters.
- Allocation guards remain enforced and error paths stay deterministic.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_texture_bake.py tests/test_glb_writer.py -q`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/metal`, `packages/mlx-spatialkit/cpp`, `packages/mlx-spatialkit/src/mlx_spatialkit/texture.py`, `packages/mlx-spatialkit/tests`

### Slice 3: GLB Embedded Texture Quality Gate

**Objective:** Add tests/helpers that inspect embedded GLB baseColor PNG payloads so sparse textures cannot pass as successful exports.

**Acceptance criteria:**
- Test helper extracts the baseColor image from a GLB JSON/BIN payload without relying on external viewers.
- Fast GLB tests assert nonzero RGB/alpha coverage on a synthetic GLB.
- Heavy real-fixture test asserts final visible base-color coverage and embedded PNG coverage exceed thresholds that fail the current `~1.15%` output.
- Generated extracted PNGs, if any, are written only under `/tmp`.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py tests/test_real_pixal3d_export.py -q -m "not heavy"`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/tests/test_glb_writer.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, optional test helper under `packages/mlx-spatialkit/tests/`

### Slice 4: Real Turtle Export Quality Verification

**Objective:** Run the real decoded Pixal3D turtle fixture through spatialkit and prove the output is visually coherent by diagnostics and embedded texture inspection.

**Acceptance criteria:**
- Heavy test writes `model.glb` and `diagnostics.json` under `/tmp`.
- Diagnostics show high final visible coverage relative to UV surface texels and record runtime/RSS samples.
- Embedded baseColor PNG inspection agrees with diagnostics within a small tolerance.
- The current sparse-dot output would fail this test.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy`

**Depends on:** Slice 3

**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, `/tmp/mlx-spatialkit-*`

### Slice 5: Mesh Simplifier Quality Tier And Export Readiness Semantics

**Objective:** Stop presenting the current face-stride simplifier as quality remeshing and make export readiness semantics honest.

**Acceptance criteria:**
- Simplifier diagnostics include `backend`, `algorithm`, and `quality_tier`.
- Current face-stride behavior is labeled `face-stride-preview` / `preview` unless replaced by a bounded quality-aware native simplifier.
- Export diagnostics distinguish artifact readiness from production-quality parity.
- Tests fail if the preview simplifier is labeled production or quality-aware.
- Export does not ignore non-empty export-blocking mesh reasons when setting readiness metadata.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py tests/test_real_pixal3d_export.py -q -m "not heavy"`

**Depends on:** Slice 4

**Touches:** `packages/mlx-spatialkit/cpp/simplify.cpp`, `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests/test_mesh_processing.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

### Slice 6: Docs And Clean Verification

**Objective:** Align package/root docs and final verification with the actual quality tier, diagnostics, `/tmp` policy, and no-release posture.

**Acceptance criteria:**
- Docs explain spatialkit texture coverage diagnostics, preview simplifier tier, real-fixture quality gate, fallback behavior, and `/tmp` heavy output policy.
- Root Pixal3D docs remain coherent with optional `--glb-export-backend spatialkit`.
- Package and root artifact checks confirm `inputs/`, `outputs/`, `vendors/`, `weights/`, and generated `/tmp` artifacts are not bundled.
- Worktree contains only intentional source/test/doc/Automaton artifact changes.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear && python - <<'PY'\nimport pathlib, tarfile, zipfile\nblocked = {'inputs', 'outputs', 'vendors', 'weights'}\nfor path in pathlib.Path('/tmp/mlx-spatialkit-dist').glob('*'):\n    if path.suffix == '.whl':\n        names = zipfile.ZipFile(path).namelist()\n    elif ''.join(path.suffixes[-2:]) == '.tar.gz':\n        with tarfile.open(path) as archive:\n            names = archive.getnames()\n    else:\n        continue\n    bad = [n for n in names if blocked.intersection(pathlib.PurePosixPath(n).parts)]\n    assert not bad, (path, bad[:5])\nprint('spatialkit artifact clean')\nPY`

**Depends on:** Slice 5

**Touches:** `packages/mlx-spatialkit/README.md`, `scripts/README.md`, `docs`, package tests, build artifacts under `/tmp`

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| SKQ-01 | Slice 1 |
| SKQ-02 | Slices 2, 4 |
| SKQ-03 | Slices 1, 2 |
| SKQ-04 | Slices 3, 4 |
| SKQ-05 | Slice 5 |
| SKQ-06 | Slice 4 |
| SKQ-07 | Slice 6 |
| SKQ-08 | Slice 6 |

## Aggregate Verification Commands

| Scope | Command |
|---|---|
| Fast package quality gate | `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` |
| Heavy real fixture quality gate | `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy` |
| Package artifact cleanliness | `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear` plus package-content check |
| Root smoke integration | `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` |

## Execution Notes

- Use `/tmp/mlx-spatialkit-*` for generated GLBs, extracted textures, diagnostics, and build outputs.
- Do not push, tag, publish, or change release metadata in this change.
- Before marking complete, open or inspect the generated real turtle GLB and report the exact `model.glb` and `diagnostics.json` paths.
- Commit after verified slice groups; keep Automaton artifacts, source changes, tests, and docs coherent in the commit history.
