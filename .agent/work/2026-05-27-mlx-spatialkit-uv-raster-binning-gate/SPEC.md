# mlx-spatialkit UV Raster Binning Gate Spec

## Bounded Goal

Make the native Metal texture bake path production-viable for arbitrary chart UVs by replacing the non-atlas all-faces-per-pixel scan with a bounded UV-space face-bin acceleration path and explicit diagnostics.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant Metal/C++ hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

## Work Scale And Shape

- Scale: capability hardening
- Shape: native Metal texture-bake performance foundation for xatlas/chart parity

## Selected Lenses

- **engineering:** Move the texture bake path closer to vendored Pixal3D's UV-rasterization model instead of depending on face-atlas-specific lookup.
- **runtime:** Keep arbitrary UV rasterization bounded in CPU memory and GPU work before attempting native chart generation.
- **product:** A future xatlas-like chart UV mesh should be bakeable by spatialkit without falling into an unbounded O(texture_pixels * faces) scan.

## Current Evidence

- Vendored Pixal3D uses `cumesh.uv_unwrap`, `nvdiffrast` UV rasterization, barycentric interpolation of mesh positions, and PBR voxel sampling.
- Current spatialkit real export uses native paired-triangle face atlas lookup, which is fast because face lookup is derived from atlas tile coordinates.
- The generic non-atlas Metal path exists, but it scans every face for every texel. That is acceptable for tiny tests and not production-scale for charted 200k-face or 1M-face meshes.
- The remaining explicit production deferral is `not_xatlas_chart_parity`; a scalable arbitrary-UV raster bake is a prerequisite for closing that gap honestly.

## Required Outcome

1. Non-atlas texture baking builds a bounded UV-space face-bin index on CPU and uses it in Metal so per-pixel face search scans only local candidate faces.
2. Diagnostics distinguish `metal-uv-binned-nearest` from face-atlas and brute-force UV scan, including bin grid size, face-reference counts, max candidates per bin, and guard status.
3. Focused tests prove the binned path produces the same surface/color behavior as the existing arbitrary-UV path on small fixtures and reports bounded candidate counts.
4. A stress fixture with many arbitrary-UV faces proves the binned path avoids the all-faces scan contract and stays within memory guards.
5. Docs explain that this closes the arbitrary-UV raster-bake scalability prerequisite, but does not yet implement native xatlas chart generation or CUDA/cuMesh remesh parity.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| UVBIN-01 | Non-atlas UV bake uses a binned Metal lookup path by default. | Focused test asserts `backend=metal-uv-binned-nearest` and bin diagnostics are present for provided UV meshes. |
| UVBIN-02 | Binned path preserves existing arbitrary-UV bake behavior on small fixtures. | Focused test compares key sampled texels/coverage against the existing expectations. |
| UVBIN-03 | Candidate search is bounded and diagnosable. | Stress test asserts `max_bin_candidate_faces` is far below total face count and `bin_reference_count` is guarded/reported. |
| UVBIN-04 | Face-atlas path and 1M/4096 gates remain intact. | Existing focused texture tests and heavy Pixal3D export gate pass, with xatlas deferral unchanged. |
| UVBIN-05 | Docs and hygiene hold. | Docs updated; package/root/build verification and artifact inspection pass. |

## Scope Coverage Decisions

- **Included:** CPU UV bin construction, Metal binned non-atlas raster lookup, diagnostics, focused/stress tests, real fixture regression, docs, package/root/build verification.
- **Deferred:** native xatlas chart generation, CUDA/cuMesh remesh parity, changing the real Pixal3D exporter from face atlas to chart UVs, exact perceptual scoring.
- **Anti-goals:** claiming xatlas parity, relaxing production thresholds, adding package runtime dependencies, writing heavy artifacts outside `/tmp`, tagging/pushing/releasing.

## Constraints

- Keep hot raster work in Metal; Python should not own per-pixel loops.
- Keep bin construction memory-bounded with explicit guard errors.
- Preserve the face-atlas fast path and existing real fixture behavior.
- Do not change decoded model outputs.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **Bin explosion risk:** Large UV triangles can overlap many bins. Mitigation: cap bin-reference count and report guard failures explicitly.
- **Performance proof risk:** Unit tests cannot prove full production speed. Mitigation: assert structural candidate-count reduction and preserve heavy Pixal3D gates.
- **False parity risk:** Binned arbitrary UV bake could be mistaken for xatlas chart generation. Mitigation: keep `not_xatlas_chart_parity` deferred and document this as a prerequisite only.

## Blocking Questions Or Assumptions

Assumption: arbitrary chart UV raster scalability should be implemented before native chart generation, because otherwise a future chart UV mesh would fall back to an unbounded face scan and fail the performance objective.
