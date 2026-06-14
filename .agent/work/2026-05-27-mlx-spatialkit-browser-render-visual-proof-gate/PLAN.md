# mlx-spatialkit Browser Render Visual Proof Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-browser-render-visual-proof-gate/SPEC.md`: add a dev-only browser-rendered visual proof gate for Pixal3D candidate/reference GLBs.

## Architecture Approach

Use `.agent/work/2026-05-27-mlx-spatialkit-browser-render-visual-proof-gate/DESIGN.md`. Keep browser tooling as opt-in dev tooling invoked from scripts with dependencies installed under `/tmp`; package runtime stays dependency-light.

## Execution Routing And Topology

- Default execution: direct, serial, continue after verification.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after the full verify gate; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Browser Render Script

**Objective:** Add a dev-only Playwright/Three browser render script for candidate/reference GLB comparison.

**Acceptance criteria:**
- Script validates required args and prints dependency/setup guidance.
- Script serves local GLBs and Three.js from caller-provided Node modules.
- Script writes browser render JSON, screenshot PNG, and HTML report under the requested output directory.
- Script can augment `visual_parity.json` when checks pass.

**Verification:** `node --check scripts/spatialkit/render_glb_visual_parity.cjs && node scripts/spatialkit/render_glb_visual_parity.cjs --help`

**Touches:** `scripts/spatialkit/render_glb_visual_parity.cjs`

**Status:** complete
**Evidence:** Added `scripts/spatialkit/render_glb_visual_parity.cjs` with CLI validation, setup guidance, local HTTP serving for GLBs/Three.js, Chrome/Playwright rendering, screenshot/report output, and optional `visual_parity.json` augmentation. `node --check scripts/spatialkit/render_glb_visual_parity.cjs && node scripts/spatialkit/render_glb_visual_parity.cjs --help` passed.
**Risks / next:** Slice 2 must prove the browser renderer works on generated GLB fixtures with `/tmp` Node deps.

### Slice 2: Synthetic Browser Smoke Proof

**Objective:** Prove the script works on deterministic small GLBs without real Pixal3D artifacts.

**Acceptance criteria:**
- Test fixture GLBs are generated under `/tmp`.
- Pinned Playwright/Three dependencies are installed under `/tmp`.
- Browser render report checks pass and artifacts exist.
- No generated artifacts enter the repo.

**Verification:** `rm -rf /tmp/mlx-spatialkit-render-smoke /tmp/mlx-spatialkit-render-deps && mkdir -p /tmp/mlx-spatialkit-render-smoke /tmp/mlx-spatialkit-render-deps && npm install --prefix /tmp/mlx-spatialkit-render-deps playwright@1.60.0 three@0.181.2 && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit python - <<'PY'
from pathlib import Path
import numpy as np
from mlx_spatialkit import make_face_atlas_uvs, textured_glb_payload
root = Path("/tmp/mlx-spatialkit-render-smoke")
for name, color in (("candidate", [255, 0, 0, 255]), ("reference", [0, 160, 255, 255])):
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    vertices = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0]], dtype=np.float32)
    faces = np.array([[0,1,2], [1,3,2]], dtype=np.int64)
    mesh = make_face_atlas_uvs(vertices, faces, tile_padding=0.0)
    base = np.zeros((16, 16, 4), dtype=np.uint8)
    base[:, :] = np.array(color, dtype=np.uint8)
    mr = np.zeros((16, 16, 3), dtype=np.uint8)
    (out / "model.glb").write_bytes(textured_glb_payload(mesh, base_color_rgba=base, metallic_roughness=mr))
PY
NODE_PATH=/tmp/mlx-spatialkit-render-deps/node_modules node scripts/spatialkit/render_glb_visual_parity.cjs --candidate /tmp/mlx-spatialkit-render-smoke/candidate/model.glb --reference /tmp/mlx-spatialkit-render-smoke/reference/model.glb --output-dir /tmp/mlx-spatialkit-render-smoke/browser_render && node -e 'const fs=require("fs"); const r=JSON.parse(fs.readFileSync("/tmp/mlx-spatialkit-render-smoke/browser_render/browser_render_report.json","utf8")); if (!r.summary.all_passed) process.exit(1); for (const p of Object.values(r.artifacts)) if (!fs.existsSync(p)) process.exit(1);'`

**Depends on:** Slice 1

**Touches:** no repo source beyond Slice 1

**Status:** complete
**Evidence:** Fixed Three.js module-root resolution and ran the synthetic smoke command with pinned Playwright/Three under `/tmp`. The script rendered generated fixture GLBs into `/tmp/mlx-spatialkit-render-smoke/browser_render`, produced `browser_render_report.json`, `comparison.png`, and `index.html`, and the JSON reported `all_passed=true` with visible-pixel ratios `[1, 1, 1]`.
**Risks / next:** Slice 3 must prove the same browser render path on the real Pixal3D reference-target GLB output.

### Slice 3: Real Pixal3D Render Proof

**Objective:** Prove the real spatialkit reference-target output and checked-in Pixal3D reference GLB render in browser and augment `visual_parity.json`.

**Acceptance criteria:**
- Heavy reference-target export writes candidate GLB and `visual_parity.json` under `/tmp`.
- Browser render script writes `browser_render/` artifacts under the export sidecar.
- Browser render report checks pass for candidate and reference.
- Augmented `visual_parity.json` records browser render artifacts and no longer lists `not_browser_rendered_visual_proof`.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy && latest=$(ls -dt /tmp/mlx-spatialkit-reference-target-export-* | head -n 1) && NODE_PATH=/tmp/mlx-spatialkit-render-deps/node_modules node scripts/spatialkit/render_glb_visual_parity.cjs --candidate "$latest/model.glb" --reference inputs/mlx-spatialkit/pixal3d-1024-cascade-glb-reference/model.glb --output-dir "$latest/visual_parity/browser_render" --visual-report "$latest/visual_parity/visual_parity.json" && node -e 'const fs=require("fs"); const latest=fs.readdirSync("/tmp").filter(n=>n.startsWith("mlx-spatialkit-reference-target-export-")).map(n=>"/tmp/"+n).sort((a,b)=>fs.statSync(b).mtimeMs-fs.statSync(a).mtimeMs)[0]; const r=JSON.parse(fs.readFileSync(latest+"/visual_parity/browser_render/browser_render_report.json","utf8")); const v=JSON.parse(fs.readFileSync(latest+"/visual_parity/visual_parity.json","utf8")); if (!r.summary.all_passed || !v.summary.browser_rendered_visual_proof) process.exit(1); if (v.deferred_parity_boundaries.includes("not_browser_rendered_visual_proof")) process.exit(1);'`

**Depends on:** Slice 2

**Touches:** generated `/tmp` artifacts only

**Status:** complete
**Evidence:** Ran the heavy reference-target fixture and browser render script against the generated candidate GLB and checked-in reference GLB. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy` passed with `2 passed, 3 deselected`. Browser render artifacts were written under `/tmp/mlx-spatialkit-reference-target-export-44123/visual_parity/browser_render`; `browser_render_report.json` reported `all_passed=true` and visible-pixel ratios `[1.0131792054368192, 1.0137264870360956, 1.0112140690653086]`. The augmented `visual_parity.json` removed `not_browser_rendered_visual_proof` while preserving `not_xatlas_chart_parity`, `not_4096_texture_parity`, and `not_1m_face_export_setting_parity`.
**Risks / next:** Slice 4 must document the dev-only browser workflow and verify repo/package hygiene.

### Slice 4: Docs And Full Verification

**Objective:** Document the browser render proof workflow and verify repo/package hygiene.

**Acceptance criteria:**
- Docs explain the `/tmp` Node dependency setup, output artifacts, and proof boundary.
- Full package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 3

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** Updated `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, and `scripts/README.md` with the `/tmp` Playwright/Three setup, browser render artifacts, and proof boundaries. Full verification passed: `git diff --check`; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` passed with `48 passed, 2 deselected`; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` passed with `35 passed`; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear` built wheel and sdist under `/tmp`; artifact inspection found no generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, or pytest cache entries.
**Risks / next:** Ready for independent Automaton verify. Exact perceptual scoring, xatlas parity, 4096 texture, and 1M-face parity remain deferred.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| BRG-01 | Slice 1 |
| BRG-02 | Slice 2 |
| BRG-03 | Slice 3 |
| BRG-04 | Slice 4 |
| BRG-05 | Slice 4 |

## Execution Notes

- Keep Playwright/Three out of `packages/mlx-spatialkit/pyproject.toml`.
- Keep generated screenshots and Node dependencies under `/tmp`.
- Do not claim xatlas, 4096 texture, 1M-face setting, or exact perceptual parity.
