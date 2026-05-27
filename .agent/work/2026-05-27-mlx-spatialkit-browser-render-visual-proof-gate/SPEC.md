# mlx-spatialkit Browser Render Visual Proof Gate Spec

## Bounded Goal

Add a dev-only browser-rendered visual proof gate for Pixal3D reference-target GLBs so `mlx-spatialkit` can produce screenshot evidence for candidate/reference comparability in addition to deterministic GLB/texture metrics.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant C++/Metal hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

The current visual parity report proves GLB structure, texture dimensions, face ratios, and embedded texture coverage. It still explicitly defers browser-rendered visual proof. This change closes that proof gap without adding browser or Three.js dependencies to the package runtime.

## Work Scale And Shape

- Scale: capability hardening
- Shape: visual parity proof and tooling

## Selected Lenses

- **product:** A developer should be able to open a rendered screenshot and see candidate/reference GLBs side by side.
- **engineering:** Keep browser tooling outside package runtime dependencies and use `/tmp` for generated render artifacts.
- **runtime:** Render proof should operate on existing GLB files; it must not rerun inference or model decoding.

## Target User Or Stakeholder

Apple Silicon `mlx-spatial` developers validating Pixal3D decoded NPZ artifacts into GLB output through the `mlx-spatialkit` companion backend.

## Current Evidence

- `compare_textured_glbs(...)` writes `visual_parity.json`, an `index.html` texture preview, and extracted base-color PNGs.
- `visual_parity.json` still records `not_browser_rendered_visual_proof` in `deferred_parity_boundaries`.
- No repo-local browser renderer or Three.js viewer exists.
- This machine can launch installed Google Chrome headlessly with Playwright when `playwright` and `three` are installed under `/tmp`.

## Required Outcome

1. A repo script renders candidate/reference GLBs in a real browser using Three.js and installed Chrome.
2. The script writes artifacts under a caller-provided output directory, normally under `/tmp`: screenshot PNG, browser render JSON, and lightweight HTML report.
3. The script records machine-readable checks: both GLBs loaded, all configured views rendered, visible pixel counts are nonzero, and the candidate/reference visible-pixel ratio is within a broad sanity range.
4. When pointed at an existing `visual_parity.json`, the script augments that report with browser-render evidence and removes `not_browser_rendered_visual_proof` only when the browser render checks pass.
5. Heavy verification proves the real spatialkit reference-target GLB and the checked-in Pixal3D reference GLB render successfully.
6. Docs explain the dev-only dependency setup and what the screenshot proof does and does not prove.

## Browser Render Targets

| ID | Target | Required evidence |
|---|---|---|
| BRG-01 | Browser renderer script exists | `node --check` passes and script help documents required args/deps. |
| BRG-02 | Synthetic smoke render | Script renders small fixture GLBs under `/tmp` using Chrome/Three/Playwright. |
| BRG-03 | Real fixture render | Heavy reference-target export plus script produces screenshot and JSON artifacts under `/tmp`. |
| BRG-04 | Visual report augmentation | Existing `visual_parity.json` gains browser-render artifacts and no longer lists `not_browser_rendered_visual_proof` when checks pass. |
| BRG-05 | Runtime dependency boundary | Package metadata remains free of Playwright/Three runtime dependencies. |

## Constraints

- Keep generated render artifacts and Node dependencies under `/tmp`.
- Do not add Playwright, Three.js, Pillow, or browser tooling to `mlx-spatialkit` runtime dependencies.
- Do not rerun model inference or decoded NPZ generation.
- Do not change geometry, texture, or production quality thresholds.
- Do not claim xatlas chart parity, 4096-texture parity, or 1M-face setting parity.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **Browser environment risk:** Headless WebGL can differ by browser and GPU backend. Mitigation: use broad nonblank/ratio checks and preserve screenshots for review.
- **False confidence risk:** Screenshots are visual evidence, not exact perceptual equivalence. Mitigation: keep deterministic GLB/texture metrics and deferred xatlas/4096/1M boundaries.
- **Dependency drift risk:** Dev-only npm packages can change. Mitigation: verification installs pinned Playwright and Three versions under `/tmp`.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| BRG-01 | Script is syntactically valid and self-documents setup. | `node --check scripts/spatialkit/render_glb_visual_parity.cjs` and `node ... --help` pass. |
| BRG-02 | Browser render smoke proof works on synthetic GLBs. | Generate two fixture GLBs under `/tmp`, run the script with `/tmp` Node deps, and assert JSON/check/artifact outputs. |
| BRG-03 | Real Pixal3D candidate/reference render proof works. | Run heavy reference-target export, run browser render script against candidate/reference GLBs, and assert browser render report plus augmented `visual_parity.json`. |
| BRG-04 | Docs state dev-only dependency and proof boundary. | README/Pixal3D/scripts docs mention browser render proof and limitations. |
| BRG-05 | Repo/package hygiene holds. | Package tests, root Pixal3D tests, build artifact inspection, and `git status --short` stay clean. |

## Scope Coverage Decisions

- **Included:** dev-only browser render script, synthetic smoke proof, real-fixture browser render proof, `visual_parity.json` augmentation, docs, package/root/build verification.
- **Deferred:** exact screenshot perceptual scoring, xatlas-equivalent unwrap, 4096 texture baking, 1M-face upstream export setting parity, Metal heap telemetry, memory optimization.
- **Anti-goals:** browser tooling as package runtime dependency, generated screenshots in repo, claims of exact visual/perceptual equivalence.

## Blocking Questions Or Assumptions

No blocking questions remain.

Assumption: a browser-rendered screenshot sidecar is the right next proof layer because existing deterministic metrics already pass but explicitly defer browser-rendered evidence.
