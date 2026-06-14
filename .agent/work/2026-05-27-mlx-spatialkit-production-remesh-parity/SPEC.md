# mlx-spatialkit Production Remesh Parity Spec

## Bounded Goal

Move `mlx-spatialkit` from preview spatial clustering toward production Pixal3D export parity by adding a reference-target export preset, a stronger native remesh/simplification path, and hard readiness thresholds proven on the real Pixal3D fixture.

## Broader Intent

The broader thread goal remains: `mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`, with quality-first geometry, performant C++/Metal hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no hidden half-stubbed behavior.

This change is the next Phase 2 cycle. It does not claim the whole backend is production-ready unless the evidence meets the acceptance thresholds below; if it exposes remaining unwrap or bake gaps, those gaps must remain explicit.

## Work Scale And Shape

- Scale: capability hardening
- Shape: parity + native algorithm + real-fixture verification

## Selected Lenses

- **engineering:** Native C++ owns geometry hot paths; Python only selects presets, orchestrates, and records diagnostics.
- **runtime:** Real fixtures are multi-million-face meshes on Apple Silicon, so algorithms must be bounded, memory-aware, and tested under `/tmp` output paths.
- **product:** A user should be able to choose a Pixal3D reference-target export path and see honest production-readiness status instead of preview artifacts labeled as ready.

## Target User Or Stakeholder

Apple Silicon `mlx-spatial` developers validating Pixal3D decoded model outputs into GLB artifacts that are visually and structurally comparable to the intended Pixal3D export path.

## Current Evidence

- Current verified spatialkit heavy output is artifact-ready but not production-ready:
  - backend: `spatial-cluster`
  - quality tier: `geometry_aware_preview`
  - simplified faces: about `43,632` for target `50,000`
  - final coverage ratio vs reference: about `0.216`
  - production readiness: `false`
- Checked-in Pixal3D reference trace at `inputs/mlx-spatialkit/pixal3d-1024-cascade-glb-reference/trace.json` reports:
  - source faces: `8,304,022`
  - final faces: `212,542`
  - final vertices: `95,281`
  - unwrap backend: `xatlas-parallel-spatial`
  - bake backend: `xatlas-kdtree`
  - raw coverage: about `0.413`
  - final coverage: `1.0`
  - xatlas face guard: `300,000`

## Required Outcome

1. A named Pixal3D production/reference-target export preset exists and resolves settings from the checked-in reference trace when available.
2. A native geometry path stronger than `spatial-cluster` is implemented or selected for the production preset, with diagnostics that explain algorithm, target, final mesh counts, topology cleanup, and why it is or is not production-tier.
3. Export diagnostics include production-readiness threshold checks, not only backend labels.
4. The real Pixal3D fixture can be exported under `/tmp` with the production/reference-target preset and compared against reference face count and coverage metrics.
5. The code keeps `production_quality_ready=false` unless every production threshold passes.

## Parity Targets

| ID | Target | Production threshold for this cycle |
|---|---|---|
| RMP-01 | Face-count parity | Final faces are within 80-125% of reference final faces, or diagnostics state the exact blocker. |
| RMP-02 | Topology exportability | Export metrics have no blocking reasons. |
| RMP-03 | Texture coverage | Final visible coverage is at least 50% of reference final coverage and raw coverage is reported against reference. |
| RMP-04 | Backend honesty | Production readiness requires a non-preview simplifier tier and passing thresholds; preview paths remain artifact-only. |
| RMP-05 | Runtime hygiene | Heavy generated GLB/diagnostics stay under `/tmp`; package/root tests and artifact checks stay clean. |

## Constraints

- Keep `mlx-spatialkit` independent from MLX, Torch, and root `mlx-spatial` runtime imports.
- Do not use Python per-face/per-vertex hot loops for production geometry.
- Prefer native C++ for remesh/simplification and existing Metal for texture bake.
- Do not tag, release, publish, push, or change release metadata.
- Keep heavy generated artifacts under `/tmp`.
- Do not hide a failed threshold behind `ready=true`; `artifact_ready` and `production_quality_ready` must remain separate.

## Risks

- **Algorithm risk:** A first native remesh upgrade may improve face-count parity but still trail upstream quality. Mitigation: threshold diagnostics decide production readiness, not labels.
- **Runtime risk:** A full QEM edge-collapse over 8M faces may be expensive. Mitigation: implement a bounded native path and verify timing/RSS on the real fixture.
- **Texture risk:** Face-atlas UVs may remain the dominant coverage blocker even with better remesh. Mitigation: keep unwrap/bake gaps explicit in diagnostics and tests.
- **Scope risk:** Upstream-style xatlas and final texture parity may require another cycle. Mitigation: this SPEC demands measurable improvement and honest blockers, not false production claims.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| RMP-01 | A production/reference-target preset resolves target faces and face guard from the reference trace when available. | Unit tests cover preset resolution with and without `trace.json`. |
| RMP-02 | Native geometry path is stronger than `spatial-cluster` for the production preset. | C++/binding tests assert the production preset reports a non-preview backend or an explicit blocker; no Python geometry hot loop is introduced. |
| RMP-03 | Production readiness is threshold-gated. | Tests prove `production_quality_ready` only becomes true when face-count, topology, coverage, and backend-tier thresholds pass. |
| RMP-04 | Real fixture exports under `/tmp` with reference-target settings. | Heavy test writes GLB/diagnostics under `/tmp` and asserts reference comparison fields and threshold details. |
| RMP-05 | Diagnostics are coherent. | Diagnostics separate `artifact_ready`, `production_quality_ready`, threshold pass/fail details, backend tier, timing, and RSS samples. |
| RMP-06 | Repo/package cleanliness holds. | `git diff --check`, package tests, root Pixal3D tests, and package build artifact inspection pass. |

## Scope Coverage Decisions

- **Included:** production/reference-target preset, native remesh/simplification improvement or explicit blocker, production threshold diagnostics, real Pixal3D heavy gate, package/root verification, docs if user-facing settings change.
- **Deferred:** claiming full production backend status if thresholds fail, replacing all root `mlx-spatial` internal export paths, release/tag/publish work, external dependency adoption that breaks package independence.
- **Anti-goals:** Python hot-loop remeshing, hiding preview behavior behind passing tests, checking generated heavy outputs into the repo, changing model inference logic.

## Blocking Questions Or Assumptions

No blocking questions remain.

Assumption: the next useful production-parity step is a reference-target geometry/export preset with hard threshold diagnostics; if the native remesh path still cannot pass thresholds, the verified outcome is an explicit, test-backed blocker for the next unwrap/remesh cycle rather than a production-ready claim.
