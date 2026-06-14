# mlx-spatialkit Reference Visual Parity Gate Spec

## Bounded Goal

Add a deterministic reference visual-comparison gate for Pixal3D reference-target exports so `mlx-spatialkit` can prove and explain visual comparability against the checked-in Pixal3D GLB, not only scalar production-threshold readiness.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant C++/Metal hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

The current reference-target gate now passes `production_quality_ready=true`, but docs still correctly state that this is not full upstream xatlas, 4096-texture, or 1M-face parity. This change adds the next proof layer: machine-readable GLB comparison and reviewer-friendly preview artifacts under `/tmp`.

## Work Scale And Shape

- Scale: capability hardening
- Shape: GLB inspection + reference comparison diagnostics + real-fixture proof artifacts

## Selected Lenses

- **engineering:** Keep export hot paths native; use Python only for post-export GLB inspection/reporting.
- **runtime:** Comparison must not load model weights or run full inference; it should parse existing GLB payloads and embedded PNGs with bounded memory.
- **product:** A user should be able to inspect why a spatialkit export is visually comparable or where it still differs from the intended Pixal3D GLB.

## Target User Or Stakeholder

Apple Silicon `mlx-spatial` developers validating Pixal3D decoded NPZ artifacts into GLB output through the `mlx-spatialkit` companion backend.

## Current Evidence

- Latest reference-target spatialkit GLB under `/tmp` reports `production_quality_ready=true`.
- Spatialkit GLB structure:
  - one mesh, one material, two embedded PNG images
  - final faces: `198618`
  - base-color texture: `1024x1024`
  - final visible base-color coverage: about `0.602`
- Checked-in reference GLB at `inputs/mlx-spatialkit/pixal3d-1024-cascade-glb-reference/model.glb` has:
  - one mesh, one material, two embedded PNG images
  - final faces: `212542`
  - reference trace reports `unwrap_backend=xatlas-parallel-spatial`, `bake_backend=xatlas-kdtree`, raw coverage about `0.413`, final coverage `1.0`
- Existing tests have a minimal GLB/PNG parser, but it is test-only and does not support all PNG filters used by the reference GLB.

## Required Outcome

1. `mlx-spatialkit` exposes package-level GLB inspection/comparison utilities for GLB 2.0 payload structure, mesh counts, material textures, embedded PNG dimensions, and texture coverage.
2. Reference-target `export_pixal3d_glb` automatically writes a visual-comparison report when the checked-in reference GLB is available.
3. Visual-comparison artifacts are written next to the generated GLB under the chosen output directory, normally `/tmp` for heavy tests.
4. Diagnostics include the visual-comparison JSON path and summary metrics: face ratio, vertex ratio, texture resolution match, base-color coverage ratio, RGB coverage ratio, image names, and pass/fail checks.
5. The heavy real fixture test asserts the visual-comparison report exists and proves the current export is comparable by the defined thresholds.
6. The report must not redefine upstream parity: xatlas, 4096 texture, and 1M-face export settings remain deferred unless separately implemented and verified.

## Visual Parity Targets

| ID | Target | Required evidence |
|---|---|---|
| VPG-01 | GLB parseability | Candidate and reference GLBs parse as GLB 2.0 with one textured mesh and embedded PNG textures. |
| VPG-02 | Face-count closeness | Candidate face count remains within the existing `80-125%` reference range. |
| VPG-03 | Texture compatibility | Candidate and reference base-color textures have matching resolution and format. |
| VPG-04 | Coverage comparability | Candidate base-color alpha/RGB coverage is at least `0.50` of reference base-color coverage. |
| VPG-05 | Reviewer artifact | A JSON report and lightweight HTML/texture-preview artifact are written under the export output directory. |
| VPG-06 | Honesty boundary | Report states remaining non-goals: not xatlas chart parity, not 4096 texture parity, not 1M-face setting parity. |

## Constraints

- Keep heavy/generated artifacts under `/tmp` in tests and development runs.
- Do not add heavyweight image or browser dependencies for the package-level comparison.
- Do not change model inference or decoded NPZ generation.
- Do not relax existing production thresholds.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **False confidence risk:** Texture coverage is not full 3D rendering. Mitigation: name the report as a visual-comparison aid and keep deferred parity boundaries explicit.
- **Parser risk:** PNG filters differ across generated/reference images. Mitigation: implement the standard PNG row filters needed for coverage analysis.
- **Artifact risk:** Visual reports could pollute the repo. Mitigation: heavy tests write under `/tmp`; package artifacts are inspected for generated files.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| VPG-01 | Package-level GLB/PNG inspection exists. | Unit tests parse synthetic and real-style GLB payloads, including filtered PNG rows. |
| VPG-02 | Reference-target export emits visual-comparison diagnostics when the reference GLB is available. | Heavy test asserts visual-comparison JSON/HTML/texture artifacts exist under `/tmp`. |
| VPG-03 | Visual-comparison checks are thresholded and honest. | Heavy diagnostics assert face ratio, texture resolution match, and coverage ratio pass while deferred xatlas/4096/1M parity is listed. |
| VPG-04 | Existing production gate remains intact. | Existing reference-target production readiness assertions still pass without threshold relaxation. |
| VPG-05 | Docs explain how to read the report. | README/Pixal3D docs mention visual-comparison artifacts and remaining boundaries. |
| VPG-06 | Repo/package hygiene holds. | Package tests, heavy fixture test, root Pixal3D tests, build artifact inspection, and `git status --short` stay clean. |

## Scope Coverage Decisions

- **Included:** package-level GLB parser/comparator, PNG coverage with filters, reference-target diagnostics integration, JSON/HTML preview output, real-fixture heavy gate, docs, package/root/build verification.
- **Deferred:** browser-rendered GLB screenshots, xatlas-equivalent unwrap, 4096 texture baking, 1M-face upstream export setting parity, full inference reruns.
- **Anti-goals:** visual claims without machine-readable metrics, repo-checked generated artifacts, heavyweight image/browser dependencies, threshold relaxation.

## Blocking Questions Or Assumptions

No blocking questions remain.

Assumption: a deterministic GLB/texture comparison gate is the next useful step because the current scalar production gate passes, but the broader goal still needs visual comparability evidence against the checked-in reference GLB.
