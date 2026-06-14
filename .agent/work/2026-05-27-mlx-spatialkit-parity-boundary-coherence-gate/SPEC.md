# mlx-spatialkit Parity Boundary Coherence Gate Spec

## Bounded Goal

Make Pixal3D visual-comparison diagnostics report only parity boundaries that are still true after the 4096 texture and browser-render proof gates.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant C++/Metal hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

## Work Scale And Shape

- Scale: bug-sized contract hardening
- Shape: diagnostics parity and coverage

## Selected Lenses

- **engineering:** Keep deferred-boundary labels generated from real current capabilities instead of stale historical blockers.
- **product:** A developer reading `diagnostics.json` should see the remaining production gaps accurately.

## Current Evidence

`packages/mlx-spatialkit/src/mlx_spatialkit/glb_compare.py` still lists these visual-comparison deferrals unconditionally:

- `not_xatlas_chart_parity`
- `not_4096_texture_parity`
- `not_1m_face_export_setting_parity`
- `not_browser_rendered_visual_proof`

The latest verified Automaton phases closed the 4096 coverage gate and browser-render proof gate, so the remaining unconditional deferrals are now stale for those two labels. The true remaining boundaries are xatlas chart parity and 1M-face export-setting parity.

## Required Outcome

1. Visual-comparison diagnostics no longer report `not_4096_texture_parity` or `not_browser_rendered_visual_proof` as default deferred boundaries.
2. Visual-comparison diagnostics continue to report `not_xatlas_chart_parity` and `not_1m_face_export_setting_parity`.
3. Tests assert both absence of closed deferrals and presence of true remaining deferrals.
4. Docs describe the remaining boundaries consistently with the diagnostics.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| PBC-01 | Default visual comparison emits no stale 4096/browser deferrals. | Unit or fixture test asserts the labels are absent. |
| PBC-02 | Default visual comparison still preserves true remaining boundaries. | Unit or fixture test asserts xatlas and 1M labels are present. |
| PBC-03 | Real reference-target heavy diagnostics align with the contract. | Existing heavy export test asserts the updated boundary set. |
| PBC-04 | Docs stay coherent with diagnostics. | README/Pixal3D/scripts docs no longer imply 4096/browser proof are still deferred. |
| PBC-05 | Repo hygiene holds. | Focused package tests, root Pixal3D tests, `git diff --check`, and package build/artifact inspection pass. |

## Scope Coverage Decisions

- **Included:** visual-comparison boundary label cleanup, tests, docs, package/root/build verification, Automaton artifacts.
- **Deferred:** implementing xatlas chart parity, implementing 1M-face export-setting parity, changing reference GLB assets, changing 4096 browser-render screenshot evidence.
- **Anti-goals:** hiding true remaining production gaps, weakening visual-comparison checks, adding dependencies, writing heavy generated artifacts outside `/tmp`, tagging/pushing/releasing.

## Constraints

- Keep generated and heavy verification artifacts under `/tmp`.
- Do not change model inference or decoded NPZ artifacts.
- Do not relax quality thresholds.
- Do not claim xatlas chart parity or 1M-face setting parity.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **False-completion risk:** Removing stale labels could look like full production parity. Mitigation: keep xatlas and 1M labels explicit and asserted.
- **Drift risk:** Docs and tests may disagree with generated diagnostics. Mitigation: assert the boundary set in both focused and heavy tests.

## Blocking Questions Or Assumptions

Assumption: Phase 8 and Phase 7 verification evidence is sufficient to remove the stale `not_4096_texture_parity` and `not_browser_rendered_visual_proof` default labels; the remaining unclosed boundaries are xatlas chart parity and 1M-face export-setting parity.
