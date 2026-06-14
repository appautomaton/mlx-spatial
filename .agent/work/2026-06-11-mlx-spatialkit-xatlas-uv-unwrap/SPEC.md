# mlx-spatialkit Reference-Parity UV Unwrap Spec

## Bounded Goal

Add a native, dependency-light **reference-parity UV unwrap** backend to `mlx-spatialkit` — cone-angle chart clustering → xatlas-equivalent chart segmentation → low-distortion parameterization → padded atlas packing — so the export stage contract's `xatlas_unwrap` check passes **honestly** (`reference_matched`, overlap-free, distortion-bounded UVs) on both fixtures, replacing reliance on the heuristic-quarantined `face-atlas` / `native-chart` backends.

## Broader Intent

Production-quality native geometry/export parity with the Pixal3D/TRELLIS reference. This is **stage 3 of 4**: narrow-band remesh (verified `2026-05-28`) → QEM edge-collapse (verified `2026-05-29`) → **UV unwrap (this change)** → texture postprocess/inpaint. Texture bake quality is gated on UVs: the current backends are projection/tile heuristics with no distortion or overlap guarantees, and the stage contract quarantines them (`export.py:1315-1329`, `unwrap_reference = uv_backend.startswith("xatlas")` at `export.py:1249`).

## Work Scale And Shape

- **Scale:** capability-sized — a two-stage unwrap pipeline (cluster → chart/parameterize/pack) plus the UV-quality metrics needed to prove it.
- **Shape:** reference parity + structural addition (a new UV backend alongside the existing two), verified by UV-quality metrics against a dev-time oracle on two fixtures.

## Selected Lenses

- **engineering:** the core is chart segmentation + parameterization correctness (zero overlap, bounded distortion, watertight-input seams handled honestly).
- **runtime:** charting + parameterization over ~50k-face fixtures CPU-first with a measured budget; reference xatlas is CPU C++, so CPU-native parity is realistic.

## Required Outcome (parity target)

Reference (behavior-only, no CUDA/vendor line-port): `CuMesh.uv_unwrap` (`/tmp/CuMesh/cumesh/cumesh.py:408`), as invoked by `o_voxel.postprocess.to_glb` (`vendors/TRELLIS.2/o-voxel/o_voxel/postprocess.py:201-210`) right after remesh+simplify. It is a **two-stage** pipeline:

1. **Stage A — fast cone clustering** (`compute_charts`, `cumesh.py:361`; kernels `/tmp/CuMesh/src/atlas.cu:1071`): greedy cost-based chart agglomeration bounded by a normal-cone half-angle, with chart-area and perimeter/area-ratio penalties, then refinement + boundary smoothing. Production knobs (pinned by `to_glb` defaults, `postprocess.py:28-31`): `threshold_cone_half_angle_rad = radians(90)`, `refine_iterations = 0`, `global_iterations = 1`, `smooth_strength = 1`, plus CuMesh defaults `area_penalty_weight = 0.1`, `perimeter_area_ratio_weight = 0.0001`.
2. **Stage B — xatlas per cluster** (`/tmp/CuMesh/cumesh/xatlas.py:55,104`): each cluster fed to real xatlas — chart growth under the default cost weights (`normal_deviation 2.0, roundness 0.01, straightness 6.0, normal_seam 4.0, texture_seam 0.5, max_cost 2.0, max_iterations 1`), LSCM-style parameterization, then packing with default `PackOptions` (`padding 0, bilinear True, rotate_charts True, rotate_charts_to_axis True, brute_force False`, resolution estimated).

Native parity requires:

1. A native **two-stage unwrap backend** (cluster → chart → parameterize → pack) in `packages/mlx-spatialkit/cpp/` exposed alongside (not replacing) `face-atlas` / `native-chart`, selected via the existing `uv_backend` param (`export.py:226`, validator `export.py:1075-1079` widened).
2. **UV validity guarantees** the heuristics lack: zero overlapping triangles in UV space; all UVs finite and packed in [0,1]; per-chart geometric distortion (stretch) measured and bounded.
3. **Behavior parity, oracle-anchored:** the xatlas half of the reference is plain CPU C++, so the pip `xatlas` wheel (same library CuMesh vendors) is a runnable **dev-time-only oracle** on Apple Silicon. Parity = our metrics (chart count, atlas utilization, stretch distribution, seam/duplicated-vertex ratio) within stated tolerances of the oracle's numbers on the same fixture meshes; stage-A semantics ported behaviorally from the readable `atlas.cu` source.
4. **Honest gate:** `xatlas_unwrap` stage reports `reference_matched` (`export.py:1315-1329`) only via the new backend with its proof passing; `face-atlas` / `native-chart` remain `heuristic_quarantined`; the existing `_xatlas_chart_parity_summary` (`export.py:1682-1840`) is wired to real numbers, not vacuous defaults.
5. **End-to-end fixture proof:** QEM-decimated watertight meshes (both fixtures, 50k target, the verified stage-2 output) → native unwrap → texture bake → GLB, with a bounded bake round-trip error and inspectable artifacts under `/tmp`.

## Constraints

- Dependency-light native C++/Metal: no required MLX, Torch, CUDA, or **xatlas** deps (carry-forward). The pip-xatlas oracle lives in a `/tmp` venv and is used only to generate recorded anchor numbers; tests pin those numbers as constants and must pass without xatlas installed.
- Behavior-reference only; do not line-port `atlas.cu` or vendor/copy xatlas sources.
- Do not remove or rename `face-atlas` / `native-chart`, their knobs, stats keys, or tests; the new backend is additive. Default `uv_backend` stays `"face-atlas"` until proven (opt-in first, mirroring the QEM rollout).
- **Input condition (recorded, not fixed here):** unwrap input is the stage-2 QEM output — fully manifold with 5–7 recorded residual boundary loops per fixture. The unwrap must handle these open boundaries correctly (they become chart boundaries); closing them is the deferred non-manifold-tolerant-QEM follow-on and must NOT be pulled into this change.
- Reuse existing native primitives where they fit (`triangle_bvh.hpp` for distortion sampling; the `native-chart` adjacency/packing scaffolding where it genuinely matches the reference semantics).
- Heavy artifacts stay under `/tmp`; no release/tag/push.

## Acceptance Criteria

| ID | Requirement | Check |
|---|---|---|
| UVU-01 | Native reference-parity unwrap backend exists and is callable. | New `uv_backend` value accepted by the validator; on a synthetic closed mesh it returns finite UVs in [0,1] with a coherent vmap/face remap (same return shape contract as existing backends). |
| UVU-02 | Stage-A clustering parity. | Cone half-angle invariant machine-checked (no face normal beyond threshold from its chart's cone axis); area + perimeter/area penalty and smoothing semantics implemented; cluster count recorded on both fixtures at the pinned production knobs. |
| UVU-03 | Chart segmentation parity (stage B). | Chart count on fixture meshes within a stated factor of the pip-xatlas oracle run on the same input at reference options (anchor numbers recorded in-test); cost-weight semantics (normal deviation / roundness / straightness / seams, `max_cost`) drive growth. |
| UVU-04 | Overlap-free, bounded-distortion parameterization. | Zero UV-triangle overlaps (machine-checked); per-chart stretch metrics (e.g. L2/Linf texel stretch vs 3D) recorded and within stated tolerance of oracle distribution; degenerate/flipped UV triangles = 0. |
| UVU-05 | Packing parity. | Padding semantics honored (no inter-chart bleed at the proof texture size); atlas utilization within stated tolerance of oracle; chart rotation-to-hull-axis or equivalent implemented; resolution/texels-per-unit semantics supported. |
| UVU-06 | Honest gate flip. | `xatlas_unwrap` stage `reference_matched` with the new backend; `heuristic_quarantined` preserved for old backends; `_xatlas_chart_parity_summary` reports real measured values and the **hardcoded `parity_ready: False`** (`export.py:1730/1823`, consumed at `:2335`) becomes a computed verdict that flips only on the measured proof; no production-blocker regression (`export.py:1461-1465` untouched semantics). |
| UVU-07 | Two-fixture end-to-end proof. | Both fixtures: QEM watertight 50k mesh → native unwrap → bake → GLB under `/tmp`; bake round-trip error bounded (recorded metric); open-boundary charts handled (chart count at boundaries recorded; no overlap/distortion blowup attributable to the residual loops). |
| UVU-08 | Determinism. | Same input + options → byte-identical UVs/vertices/faces, in-process and cross-process (PYTHONHASHSEED-varied), mirroring the QEM determinism contract. |
| UVU-09 | Runtime/memory recorded and bounded. | Unwrap at fixture scale completes within a recorded numeric budget (time + delta peak RSS, suite-order-independent per the stage-2 lesson); no OOM. |
| UVU-10 | Dependency-light preserved. | No new required deps; native build + import succeed; full suite passes in an environment without pip-xatlas. |
| UVU-11 | Regression contracts preserved. | `face-atlas` / `native-chart` backends, stats keysets, `test_glb_writer` / `test_mesh_processing` / export / remesh / simplify suites, and all public knobs stay green and intact. |

## Scope Coverage Decisions

- **Included:** the two-stage native unwrap backend; UV validity metrics (overlap, stretch, flipped tris) as new native diagnostics; oracle-anchored parity numbers; honest gate flip + parity-summary wiring; opt-in export param routing; two-fixture end-to-end proof with bake round-trip; determinism; perf budget; regression.
- **Deferred (own follow-on changes):** non-manifold-tolerant QEM (0-loop input parity — explicitly out, decided pre-framing); texture postprocess / Telea-equivalent inpaint (stage 4); reference-scale 1M-vertex / 4096-texture parity (carry-forward); `brute_force` packing and `block_align` (reference defaults are off); Metal acceleration unless CPU proves insufficient.
- **Anti-goals:** removing or regressing the existing UV backends; line-porting CUDA or vendoring xatlas; flipping the default `uv_backend` before the proof; claiming production-ready GLB from unwrap alone (stage 4 pending); declaring parity from chart counts alone without the overlap/distortion checks; gaming the `startswith("xatlas")` gate with a renamed heuristic.

## Assumptions

- Pip `xatlas` (0.0.9-line, per `vendors/sam-3d-objects/requirements.txt:86`) is behaviorally the same library CuMesh vendors (`_cumesh_xatlas`); version-pinned oracle numbers are recorded with the version string. If oracle install fails on this host, parity tolerances fall back to source-derived bounds with the gap recorded — flagged to the user before the proof slice, not silently.
- The pinned production knobs are `to_glb`'s defaults (cone 90°, refine 0, global 1, smooth 1); Pixal3D's texturing-pipeline no-kwargs call (`trellis2_texturing.py:306`) is a fallback path, not the parity target.

## Risks

- **Parameterization is the hard core:** LSCM-equivalent flattening with guaranteed zero overlap is algorithmically heavier than anything in stages 1–2; a planar-projection shortcut would pass small synthetics and fail distortion parity on curved charts. Mitigate: distortion + overlap metrics land before the backend (oracle of record), as `nonmanifold_vertices` did for QEM.
- **90° cone half-angle means clustering is permissive** — most segmentation pressure lands on stage B's cost-driven growth; under-implementing stage B and leaning on stage A would show up as chart-count/distortion divergence from the oracle. The parity tolerances must be tight enough to catch this.
- **Oracle drift:** pip xatlas vs CuMesh's vendored binding may differ in version/options mapping; record exact versions and a small option-mapping table with the anchors.
- **Perf:** xatlas-equivalent growth is greedy with per-face cost re-evaluation; naive native implementation may be O(F²)-ish at fixture scale (stage-2 precedent: two such bugs found late). Budget assertions land with the proof slice, not after.
- **Boundary seams:** the 5–7 residual input loops produce real chart boundaries; bake bleed at those seams may be visually confusable with stage-4 (inpaint) gaps — diagnostics must attribute them.
