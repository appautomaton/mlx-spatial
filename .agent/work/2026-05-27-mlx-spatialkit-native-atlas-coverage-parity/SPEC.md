# mlx-spatialkit Native Atlas Coverage Parity Spec

## Bounded Goal

Improve the native Pixal3D reference-target export path by replacing one-triangle-per-tile UV packing with a denser native atlas strategy that raises real-fixture global texture coverage while keeping production readiness threshold-gated.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: native C++/Metal hot paths, quality-first geometry and texture output, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no hidden preview behavior behind passing tests.

This change is a Phase 3 step. It targets the measured texture coverage blocker from the verified reference-target gate; it does not claim full production parity unless every production threshold passes.

## Work Scale And Shape

- Scale: capability hardening
- Shape: native algorithm + parity gap closure + real-fixture verification

## Selected Lenses

- **engineering:** Native C++ owns atlas generation; Python only selects the path and records diagnostics.
- **runtime:** The atlas must handle about 200k reference-target faces without Python hot loops or repo-polluting artifacts.
- **product:** Reference-target exports should get visibly closer to the intended Pixal3D output and explain remaining blockers precisely.

## Target User Or Stakeholder

Apple Silicon `mlx-spatial` developers validating Pixal3D decoded NPZ artifacts into GLB output through the `mlx-spatialkit` companion backend.

## Current Evidence

- Reference-target export resolves `target_faces=212542` and produces about `198618` simplified faces.
- Face-count ratio and topology thresholds pass.
- `production_quality_ready=false` because:
  - simplifier tier is still `geometry_aware_preview`
  - global final texture coverage is about `0.269`, below the `0.50` threshold
- Current face-atlas assigns one triangle per rectangular tile, leaving roughly half of every tile unused before padding. That directly depresses global texture coverage even when UV-surface coverage is reasonable.

## Required Outcome

1. Native atlas generation can pack two triangle faces into one tile when safe, using complementary half-tile UVs.
2. Diagnostics report atlas packing mode, faces per tile, tile count, utilization estimate, and output mesh sizes.
3. The reference-target export path uses the denser native atlas by default without changing preview artifact readiness semantics.
4. Real Pixal3D reference-target heavy diagnostics show higher final global texture coverage than the prior `0.269` baseline and keep production readiness false unless all thresholds pass.
5. Tests and package artifacts stay clean, with heavy generated outputs under `/tmp`.

## Constraints

- Keep `mlx-spatialkit` independent from MLX, Torch, xatlas, and root `mlx-spatial` runtime imports.
- Do not introduce Python per-face/per-vertex atlas hot loops.
- Keep GLB geometry valid: finite vertices/UVs, faces in range, no texture-coordinate violations.
- Do not relax production thresholds to make the new atlas pass.
- Do not tag, release, publish, or push.

## Risks

- **Coverage risk:** Higher UV utilization may still not hit the production threshold because sparse voxel sampling remains the bottleneck. Mitigation: record before/after coverage and keep threshold failures explicit.
- **Visual risk:** Pairing unrelated triangles in one tile improves utilization but is not equivalent to xatlas charting. Mitigation: label backend honestly as native atlas, not xatlas parity.
- **Runtime risk:** Larger dense atlas stats must not allocate per-pixel or per-face Python structures. Mitigation: keep atlas work in C++ and verify memory samples on heavy export.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| ACP-01 | Native atlas supports paired triangle packing. | C++/Python tests assert two triangle faces share one atlas tile with complementary UV halves and backend diagnostics report paired mode. |
| ACP-02 | Export path uses dense atlas for reference-target preset. | Real fixture diagnostics show the UV backend/mode and higher UV/global coverage than the prior reference-target baseline. |
| ACP-03 | Production readiness remains threshold-gated. | Tests confirm failed backend/coverage thresholds keep `production_quality_ready=false`; no threshold is relaxed. |
| ACP-04 | Package/runtime hygiene holds. | Package tests, heavy fixture test, root Pixal3D tests, build artifact inspection, and `git status --short` stay clean. |

## Scope Coverage Decisions

- **Included:** native paired atlas packing, diagnostics, reference-target integration, coverage tests, docs if user-facing behavior changes.
- **Deferred:** full xatlas-equivalent unwrap, production remesh backend tier, replacing sparse-nearest Metal sampling with a full projection bake.
- **Anti-goals:** external xatlas dependency inside `mlx-spatialkit`, Python atlas hot loops, claiming production readiness without threshold evidence.

## Blocking Questions Or Assumptions

No blocking questions remain.

Assumption: a paired triangle atlas is the smallest native change likely to improve the measured global coverage blocker while preserving package independence and memory safety.
