# mlx-spatialkit Browser Render Visual Proof Gate Design

## Current Gap

The current `visual_parity/` sidecar contains deterministic GLB and texture metrics:

```text
visual_parity/
  visual_parity.json
  index.html
  candidate_base_color.png
  reference_base_color.png
```

It intentionally does not prove that a browser can render the generated GLB.

## Tool Shape

Add a dev-only Node script:

```text
scripts/spatialkit/render_glb_visual_parity.cjs
  --candidate /tmp/.../model.glb
  --reference inputs/.../model.glb
  --output-dir /tmp/.../visual_parity/browser_render
  --visual-report /tmp/.../visual_parity/visual_parity.json
```

The script uses caller-provided Node dependencies, normally installed under
`/tmp/mlx-spatialkit-render-deps`:

```bash
npm install --prefix /tmp/mlx-spatialkit-render-deps playwright@1.60.0 three@0.181.2
NODE_PATH=/tmp/mlx-spatialkit-render-deps/node_modules node scripts/spatialkit/render_glb_visual_parity.cjs ...
```

This keeps Playwright and Three.js out of package runtime dependencies.

## Browser Flow

```text
local HTTP server
  -> serves candidate/reference GLBs
  -> serves Three.js files from /tmp node_modules
  -> browser page loads GLTFLoader
  -> renders candidate/reference across fixed view directions
  -> captures screenshot PNG
  -> returns visible-pixel metrics
```

The render page uses transparent WebGL canvases and counts nontransparent
pixels per canvas. This is a sanity gate for renderability and gross visual
coverage, not a perceptual similarity metric.

## Artifacts

```text
browser_render/
  browser_render_report.json
  comparison.png
  index.html
```

If `--visual-report` points to an existing `visual_parity.json`, the script
adds a `browser_render` section, adds artifact paths, and removes
`not_browser_rendered_visual_proof` only when all browser render checks pass.

## Boundary

This closes the "browser rendered proof exists" gap. It does not close:

- xatlas chart parity
- 4096 texture parity
- 1M-face export-setting parity
- exact perceptual or screenshot-difference scoring
