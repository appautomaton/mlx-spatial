# mlx-spatial MapAnything Release Readiness Plan

## Goal

Execute the release-readiness contract in [SPEC.md](SPEC.md): prove and patch
the PyPI package surface for MapAnything MLX inference before the next tag.

## Execution Routing and Topology

Default route: direct, serial.

Parallel-safe groups: none.

Checkpoints:

- Stop before publish/tag actions. Publishing and tag pushes are outside this
  plan.
- Stop before broad commit splitting if unrelated dirty state needs human
  grouping decisions.

Recommended review: `auto-eng-review` is useful before executing release
patches because this plan touches packaging, workflow, public docs, and memory
evidence. If execution proceeds directly, every slice must leave concrete
verification evidence in this PLAN.

## Requirement Traceability

| SPEC ID | Satisfied by |
| --- | --- |
| REL-MA-01 | Slice 1 |
| REL-MA-02 | Slice 2 |
| REL-MA-03 | Slice 2 |
| REL-MA-04 | Slice 3 |
| REL-MA-05 | Slice 3 |
| REL-MA-06 | Slice 4 |
| REL-MA-07 | Slice 4 |
| REL-MA-08 | Slice 5 |
| REL-MA-09 | Slice 5 |
| REL-MA-10 | Slice 6 |

## Ordered Slice Sequence

### Slice 1: Automaton State Alignment

**Objective:** Move durable Automaton state from the verified scene-generation
change to this release-readiness change.

**Acceptance criteria:**

- `current.json` names `2026-05-26-mlx-spatial-mapanything-release-readiness`.
- Canonical SPEC and PLAN pointers resolve.
- `get-context.mjs` reports this change without diagnostics.

**Verification:** `node .agent/.automaton/scripts/get-context.mjs`

**Touches:** `.agent/work/2026-05-26-mlx-spatial-mapanything-release-readiness/`, `.agent/.automaton/state/current.json`

**Status:** complete
**Evidence:** wrote `INTAKE.md`, `SPEC.md`, and this `PLAN.md`; ran `node .agent/.automaton/scripts/sync-status.mjs --active-change 2026-05-26-mlx-spatial-mapanything-release-readiness --canonical-spec .agent/work/2026-05-26-mlx-spatial-mapanything-release-readiness/SPEC.md --canonical-plan .agent/work/2026-05-26-mlx-spatial-mapanything-release-readiness/PLAN.md --stage plan`, then entered execute stage through `sync-status.mjs --stage execute`. `node .agent/.automaton/scripts/get-context.mjs` reports this active change, `stage: execute`, canonical SPEC, canonical PLAN, and no diagnostics.
**Risks / next:** none.

### Slice 2: Package Metadata and Workflow Version Audit

**Objective:** Ensure the package CLI surface and release workflow match the
current intended version and MapAnything entry point.

**Acceptance criteria:**

- `pyproject.toml` includes `mlx-spatial-mapanything = "mlx_spatial.mapanything:main"`.
- `.github/workflows/workflow.yaml` does not hard-code stale `0.0.1` artifact
  names when `pyproject.toml` is `0.0.2`.
- `uv run mlx-spatial-mapanything --help` succeeds.

**Verification:** `uv run mlx-spatial-mapanything --help && rg -n "0\\.0\\.1|mlx-spatial-mapanything|mlx_spatial.mapanything" pyproject.toml .github/workflows/workflow.yaml`

**Touches:** `pyproject.toml`, `.github/workflows/workflow.yaml`

**Status:** complete
**Evidence:** changed `.github/workflows/workflow.yaml` artifact checking from stale `dist/mlx_spatial-0.0.1.*` filenames to version-agnostic `dist/mlx_spatial-*.tar.gz` and `dist/mlx_spatial-*-py3-none-any.whl`. `uv run mlx-spatial-mapanything --help` succeeded. A Python assertion confirmed the workflow no longer contains `dist/mlx_spatial-0.0.1` and does contain both artifact patterns. `rg -n "mlx-spatial-mapanything|mlx_spatial.mapanything|dist/mlx_spatial-" pyproject.toml .github/workflows/workflow.yaml` shows the MapAnything entry point and the workflow artifact patterns.
**Risks / next:** none.

### Slice 3: Runtime Boundary and Docs/Script Coherence

**Objective:** Prove the runtime dependency boundary and public docs/scripts are
coherent for MapAnything.

**Acceptance criteria:**

- Base dependencies exclude Torch, TorchVision, UniCeption, OpenCV, and CUDA-only packages.
- Runtime package code does not import those stacks or `vendors/map-anything`.
- MapAnything docs and script help agree on weights root, `fixed_mapping`,
  stride `1`, checkpoint-derived patch size, `.npz` output, and no mesh/3DGS
  export claim.

**Verification:** `uv run pytest tests/test_mapanything_inference.py tests/test_checkpoint.py -q && python scripts/mapanything/generate_scene.py --help && rg -n "fixed_mapping|patch_size=checkpoint_config|Gaussian Splat|mesh|torch|vendor" README.md docs/mapanything.md scripts/README.md scripts/mapanything/generate_scene.py src/mlx_spatial pyproject.toml`

**Touches:** `README.md`, `docs/mapanything.md`, `scripts/README.md`, `scripts/mapanything/generate_scene.py`, runtime dependency tests if needed

**Status:** complete
**Evidence:** `uv run pytest tests/test_mapanything_inference.py tests/test_checkpoint.py -q` passed with 19 tests. `python scripts/mapanything/generate_scene.py --help` shows `fixed_mapping`, stride `1`, config-derived patch size, `weights/map-anything`, `.npz` output, and no mesh/3DGS export claim. Runtime import scan `! rg -n "import (torch|torchvision|cv2)|from (torch|torchvision|uniception|cv2)|vendors/map-anything" src/mlx_spatial` passed. Docs/script scan confirmed MapAnything wording across `README.md`, `docs/mapanything.md`, `scripts/README.md`, and `scripts/mapanything/generate_scene.py`.
**Risks / next:** post-verify audit found two release-surface gaps and both were corrected. `docs/release.md` post-publish checks listed only `mlx-spatial-sam3d --help`; it now includes `mlx-spatial-mapanything --help`. Full pytest also found `huggingface-hub` in base runtime dependencies; `pyproject.toml` now keeps `huggingface-hub>=0.36` in the `dev` dependency group only. Targeted dependency tests passed with 4 tests, and `uv run hf --help` confirms the repo dev CLI still exists.

### Slice 4: Real Desk Smoke and Memory Evidence

**Objective:** Run the real MapAnything Desk scene path and capture memory/timing
evidence for AppleGPU release readiness.

**Acceptance criteria:**

- Local Desk run writes `scene.npz` and `trace.json`.
- Output schema matches the Torch-reference semantic layout plus MLX `extrinsics`.
- Peak process memory or MLX peak memory is recorded in this PLAN.
- The run uses the recommended script path, not a hidden one-off command.

**Verification:** `/usr/bin/time -l uv run python scripts/mapanything/generate_scene.py inputs/map-anything/desk --output-dir /tmp/mapanything-release-smoke`

**Touches:** `PLAN.md` evidence only unless the run exposes a blocker.

**Status:** complete
**Evidence:** `/usr/bin/time -l uv run python scripts/mapanything/generate_scene.py inputs/map-anything/desk --output-dir /tmp/mapanything-release-smoke` succeeded with `profile=production-like`, `resize_mode=fixed_mapping`, `stride=1`, `patch_size=checkpoint_config`, and `runtime_depends_on_torch=false`. It wrote `/tmp/mapanything-release-smoke/scene.npz` and `trace.json`; trace reports 2 frames, target size `(518, 392)`, and 11 completed stages through `scene-postprocess`. Schema check confirmed `images (2,392,518,3)`, `depth/confidence/masks (2,392,518)`, `intrinsics (2,3,3)`, `camera_poses/extrinsics (2,4,4)`, and `world_points (2,392,518,3)`, all `float32`. `/usr/bin/time -l` recorded `1.50 real` and `5,867,470,848` bytes maximum resident set size, approximately 5.47 GiB process RSS, with no page faults or swaps.
**Risks / next:** RSS is process-level memory, not pure MLX heap; it is adequate release evidence for the Desk smoke, but future stage-level memory tracking should use MLX counters if regressions appear.

### Slice 5: Tests, Build, and Artifact Contents

**Objective:** Run release-grade checks and prove MapAnything files are included
while blocked local paths are excluded.

**Acceptance criteria:**

- Targeted MapAnything tests pass.
- `uv lock --check` passes.
- `rm -rf dist && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` passes.
- Artifact checker passes on the generated wheel and sdist.
- Artifact contents include MapAnything runtime modules, `docs/mapanything.md`,
  and `scripts/mapanything/generate_scene.py`.

**Verification:** `uv run pytest tests/test_mapanything_*.py -q && uv lock --check && rm -rf dist && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build && python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-*.tar.gz dist/mlx_spatial-*-py3-none-any.whl`

**Touches:** packaging metadata or artifact checker if gaps are found

**Status:** complete
**Evidence:** `uv run pytest tests/test_mapanything_*.py -q` passed with 69 tests and 5 opt-in parity tests skipped. `uv lock --check` passed. Initial artifact check against `dist/mlx_spatial-*` failed because stale local `dist/mlx_spatial-0.1.0.*` artifacts were still present and contained blocked `.agent/.codex/.claude` paths. Corrected the release docs/workflow/plan to clean `dist/` before build, then ran `rm -f dist/mlx_spatial-*.tar.gz dist/mlx_spatial-*-py3-none-any.whl && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build`; it built only `dist/mlx_spatial-0.0.2.tar.gz` and `dist/mlx_spatial-0.0.2-py3-none-any.whl`. `python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-*.tar.gz dist/mlx_spatial-*-py3-none-any.whl` passed with 2 artifacts. A contents check confirmed the sdist includes `docs/mapanything.md`, `scripts/mapanything/generate_scene.py`, MapAnything source modules, and MapAnything tests; the wheel includes MapAnything runtime modules and neither artifact includes `weights/`, `inputs/`, `outputs/`, `vendors/`, `.agent/`, or `.codex/`.
Full release audit later found `huggingface-hub` in base dependencies; moved it to the `dev` dependency group, ran `uv lock`, and re-ran the release gates. Fresh verification passed: `uv run pytest -q` -> 807 passed, 10 skipped, 27 deselected, 2 warnings; `uv lock --check` passed; `rm -rf dist && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` rebuilt `dist/mlx_spatial-0.0.2.tar.gz` and `dist/mlx_spatial-0.0.2-py3-none-any.whl`; artifact checker and git hygiene passed. Clean wheel smoke passed in `/tmp/mlx-spatial-wheel-smoke`: installed only base runtime dependencies plus `mlx-spatial==0.0.2`, imported `mlx_spatial`, and ran installed `mlx-spatial-mapanything --help`.
**Risks / next:** none for current artifacts; local release preflight must keep the new `rm -rf dist` cleanup step to avoid stale generated artifacts.

### Slice 6: Git Boundary and Release Notes for Human Review

**Objective:** Make the release-relevant dirty state and remaining non-release
dirty groups explicit without reverting unrelated work.

**Acceptance criteria:**

- `git status --short` is reviewed.
- Release-relevant files are listed separately from unrelated Automaton/scaffold
  dirt.
- No publish/tag action is performed.
- If any release blocker remains, it is recorded as a `VERIFY-GAP` note here.

**Verification:** `git status --short && python scripts/packaging/check_release_artifacts.py --git-hygiene`

**Touches:** `PLAN.md` evidence; package checker only if the hygiene gate is wrong.

**Status:** complete
**Evidence:** `git status --short` reviewed. Release-relevant current groups are `.github/workflows/workflow.yaml`, `README.md`, `docs/{README,architecture,development,release,mapanything}.md`, `scripts/README.md`, `scripts/mapanything/`, `pyproject.toml`, MapAnything runtime modules under `src/mlx_spatial/mapanything*.py`, MapAnything tests/fixtures, and `tools/mapanything_dump_torch_*.py`. Existing mixed dirty groups remain outside this release slice and were not reverted: `.codex/skills/*`, `.agent/.automaton/*`, older script-polish changes, LiTo docs/assets changes, and other prior generated Automaton work folders. `python scripts/packaging/check_release_artifacts.py --git-hygiene` passed. No publish, tag, push, or commit action was performed.
**Risks / next:** commit grouping remains a human review decision because the worktree is mixed; release artifact hygiene is clean.

## Aggregate Verification Commands

```bash
node .agent/.automaton/scripts/get-context.mjs
uv run mlx-spatial-mapanything --help
uv run pytest tests/test_mapanything_*.py -q
uv lock --check
rm -rf dist
UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build
python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-*.tar.gz dist/mlx_spatial-*-py3-none-any.whl
python scripts/packaging/check_release_artifacts.py --git-hygiene
```
