# Pixal3D GLB Quality Rebaseline Spec

## Bounded Goal

Rebaseline and consolidate Pixal3D GLB quality work so `mlx-spatial` only treats artifacts as ready when input generation, native export, visual diagnostics, and remaining production-quality parity gaps are separated and verified against real fixtures.

## Broader Intent

The next implementation cycle must move from screenshot-driven patching to source-grounded quality work. Prior work made real progress, but it also created a false completion signal: the rendered-visual plan recorded completed evidence while the artifact still reported `rendered_visual_ready=false`. This spec turns that mismatch into the first problem to fix, consolidates the 2026-05-27 `mlx-spatialkit` work log as historical evidence, and frames the remaining Pixal3D and `mlx-spatialkit` gaps as one coherent quality objective.

## Work Scale And Shape

- **Scale:** capability-sized rebaseline.
- **Shape:** parity and diagnostics work across Pixal3D input conditioning, decoded fixture provenance, native GLB export, and readiness gates.

## Selected Lenses

- **product:** Users need to know whether a generated GLB is actually usable, not merely written.
- **engineering:** Implementation must follow the Pixal3D/o-voxel/CuMesh reference contracts where relevant and keep unresolved native gaps explicit.
- **runtime:** Heavy GLBs, screenshots, browser renders, and scratch comparisons stay under `/tmp`; repo artifacts stay compact and reloadable.

## Required Outcome

1. Prior rendered-visual work is reclassified honestly as a partial checkpoint, not a completed quality objective.
2. Diagnostics and tests distinguish structural artifact readiness, rendered visual readiness, browser-render proof, production quality, and reference parity.
3. Artifact comparisons use a stable A/B/C taxonomy backed by a manifest: decoded model output, native `mlx-spatialkit` GLB, and reference/control GLB.
4. Pixal3D input handling matches upstream preprocessing and stage-conditioning contracts enough to avoid raw-background leakage and to isolate generation defects from export defects.
5. Native export gaps are mapped by layer before implementation: topology/remesh/QEM, open-boundary repair, UV unwrap, texture sampling, texture postprocess/render padding, material packing, and normals/viewer compatibility.
6. The next heavy fixture checks cover the actual local fixture set under `inputs/mlx-spatialkit`, including the base Pixal3D 1024 cascade fixture and at least one independent violin/bow lineage.
7. The spec preserves unresolved parity gaps as blockers rather than hiding them behind passing scalar counters.
8. Existing working contracts stay valid: Pixal3D texture coordinate order, PBR channel packing, flexible dual-grid extraction, viewer compatibility checks, and current non-heavy test fixtures are regression-protected before new quality work is considered progress.
9. The 2026-05-27 `mlx-spatialkit` work dirs are consolidated into a compact rollup that preserves evidence without turning every historical micro-spec into active backlog.
10. The next implementation focus is explicit: production-quality geometry/export parity, led by open-boundary topology, true QEM edge-collapse, narrow-band remesh, and unwrap parity, not another readiness or screenshot-tuning loop.

## Source Evidence

Detailed harvested signals are normative in [spec/gap-matrix.md](spec/gap-matrix.md). The 2026-05-27 historical work consolidation is normative in [spec/legacy-2026-05-27-rollup.md](spec/legacy-2026-05-27-rollup.md).

Summary evidence:

- The prior `2026-05-27-mlx-spatialkit-rendered-visual-correctness` plan records completed evidence while also recording `rendered_visual_ready=false` and residual artifacts.
- Raw violin input generated a white sheet; Pixal3D-style alpha preprocessing removed the sheet, proving that input preprocessing and export quality are separate problems.
- Upstream Pixal3D preprocesses the raw image, then feeds the same preprocessed RGB image to MoGe and generation; our MLX path currently opens the raw path for MoGe, DINO, and NAF.
- Upstream Pixal3D uses stage-specific image conditioning; our MLX path currently risks reusing 512-resolution conditioning for 1024-resolution stages.
- The native exporter writes usable GLBs and has fixed the Pixal3D texture coordinate order, but still reports open-chain blockers and non-reference topology behavior.
- Subagent audits agree that extraction is comparatively close, while remesh/QEM/open-chain repair, UV unwrap parity, texture postprocess, and diagnostic naming remain real gaps.
- The 55 `2026-05-27-mlx-spatialkit-*` work dirs represent one iterative quality-debugging chain; the active spec supersedes them as the working contract while keeping them available as forensic evidence.

## Constraints

- Do not create a new roadmap backlog; the roadmap should stay empty unless a separate roadmap-scale decision is made.
- Do not claim quality success from `visual_comparison.summary.all_passed`, `artifact_ready`, structural GLB parseability, or browser visible-pixel checks alone.
- Do not add xatlas, Torch, CUDA, or MLX as required dependencies of `mlx-spatialkit`.
- Do not line-port CUDA or vendor code; use vendor code as behavior reference.
- Do not start release, tag, push, or publish work.
- Do not leave generated heavy artifacts in the repo.
- Do not edit historical `.agent/work/*` specs or plans to rewrite history; supersede stale prior artifacts from the active change only.
- Do not create another broad chain of micro-specs for the same GLB-quality problem; consolidate evidence here and split only independently verifiable production-quality implementation work.
- Do not remove or silently rename existing public knobs, diagnostics, or tests unless the active plan names the migration and verifies compatibility.

## Acceptance Criteria

| ID | Requirement | Check |
|---|---|---|
| PQR-01 | Prior false-completion evidence is corrected. | The previous rendered-visual change is recorded as a partial checkpoint, and no current artifact or state implies the quality goal is complete while `rendered_visual_ready=false`. |
| PQR-02 | Readiness terms are unambiguous. | Diagnostics/tests separate `artifact_ready`, `rendered_visual_ready`, `browser_rendered_visual_proof`, `production_quality_ready`, and `production_equivalence_ready`; scalar `summary.all_passed` cannot be used as visual success. |
| PQR-03 | Artifact provenance is explicit. | Every heavy comparison writes a manifest that records A/B/C roles, `lineage_id`, decoded input, native GLB, reference/control GLB, commands/settings, diagnostics paths, and browser-proof artifacts when present. |
| PQR-04 | Pixal3D input parity is measurable. | Tests or focused fixtures prove upstream-style preprocessing happens before MoGe/DINO/NAF conditioning and catches raw-background leakage before full generation. |
| PQR-05 | Stage-conditioning risk is bounded. | `ss` and `shape_512` may use 512-resolution conditioning; `shape_1024` and `tex_1024` use 1024-resolution conditioning or fail closed on mismatched patch grids, with stage metadata in traces. |
| PQR-06 | Export topology gaps are treated as blockers. | Diagnostics classify clean loops, simple open chains, branched open chains, non-manifold edges, heuristic QEM, and missing narrow-band remesh without claiming production parity until those blockers are resolved. |
| PQR-07 | UV/bake/material gaps are isolated by tests. | Separate tests distinguish unwrap, `coverage_status` sampling classes, postprocess, render padding, material packing, normals, and viewer/render effects. |
| PQR-08 | Two-fixture quality evidence exists. | The base Pixal3D 1024 cascade fixture and at least one independent violin/bow lineage both produce native/reference artifacts with honest readiness outcomes. |
| PQR-09 | Repo hygiene holds. | `git diff --check` passes, focused tests pass, heavy outputs remain outside tracked files, and no release/tag/push work is introduced. |
| PQR-10 | Comparison manifests are authoritative. | Comparisons fail closed if A and C do not share lineage or if the manifest is missing, ambiguous, or mismatched. |
| PQR-11 | Existing working behavior remains valid. | Focused regression tests preserve the known-working texture coordinate order, PBR packing, mesh extraction/metrics, GLB writer/viewer compatibility, and Pixal3D fixture boundaries. |
| PQR-12 | Legacy work is consolidated. | The 2026-05-27 `mlx-spatialkit` work dirs are summarized by bucket in an active-change rollup and are treated as historical evidence, not active backlog. |
| PQR-13 | Next quality work is focused. | The spec names production-quality geometry/export parity as the next implementation direction and prevents readiness/browser/scalar gates from standing in for actual GLB quality. |

## Scope Coverage Decisions

- **Included:** Automaton state correction through active artifacts, readiness naming, artifact manifest taxonomy, 2026-05-27 work-log consolidation, Pixal3D input preprocessing contract, stage-conditioning risk, native export gap attribution, UV/bake/material isolation, two-fixture verification, and regression protection for existing working contracts.
- **Deferred:** full 1M/4096 export parity, exact xatlas replacement, and full production-grade QEM/DC implementation unless accepted as a separate focused production-geometry implementation change.
- **Anti-goals:** declaring the previous checkpoint complete, optimizing for one screenshot, creating another 05-27-style micro-spec chain, using a single turtle fixture as proof of generality, or treating visually coherent silhouette as production quality.

## Risks

- **Scope collapse:** Splitting input and export too early could hide their interaction. Mitigation: keep one rebaseline spec with separate acceptance checks.
- **Overfitting:** Fixes tuned to turtle/city can regress violin/bow. Mitigation: require two-fixture evidence.
- **False green diagnostics:** Existing counters can pass while visual readiness fails. Mitigation: readiness names and blockers must be stricter than artifact write success.
- **Native scope creep:** Remesh/QEM/xatlas parity can grow large. Mitigation: this spec may plan blockers and measurement gates before implementing the full production path.

## Assumptions

- The user wants this framed as the next coherent spec, not as a roadmap phase.
- `mlx-spatialkit` remains dependency-light and native-performance oriented.
- `/tmp/CuMesh`, `vendors/Pixal3D`, and `vendors/TRELLIS.2` are behavior references available locally for planning and verification.
