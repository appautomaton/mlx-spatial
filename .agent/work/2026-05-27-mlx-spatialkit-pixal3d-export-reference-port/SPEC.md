# mlx-spatialkit Pixal3D Export Reference Port Spec

## Bounded Goal

Make `mlx-spatialkit` generate high-quality Pixal3D GLB artifacts by replacing heuristic-driven export tuning with a reference-first native C++/Metal port of the Pixal3D `o_voxel.postprocess.to_glb` export contract.

## Broader Intent

`mlx-spatialkit` should be the dependable native export backend for `mlx-spatial`: visually coherent GLB output, native hot paths, memory-aware execution on Apple Silicon, thread-safe native ownership, honest diagnostics, and no production-quality claims that are not backed by source-grounded parity evidence.

## Work Scale And Shape

- Scale: capability hardening
- Shape: reference parity + native algorithm port + visual-quality closure

## Selected Lenses

- **product:** A Pixal3D decoded scene should open in ordinary GLB viewers as the intended object with coherent geometry, colors, and material texture, not as a hole-patched or smeared approximation.
- **engineering:** The implementation should follow the upstream stage contract before adding local alternatives; C++/Metal owns mesh, UV, raster, projection, and sampling hot paths.
- **runtime:** Heavy real-fixture exports stay under `/tmp`, record wall-time and RSS, and avoid retaining large arrays after their stage is complete.

## Target User Or Stakeholder

Apple Silicon `mlx-spatial` developers converting Pixal3D decoded `shape_decoder_fields.npz` and `texture_decoder_pbr.npz` into inspectable GLB artifacts without Torch, CUDA, or hidden preview-quality shortcuts.

## Source Evidence

- Pixal3D inference exports through `o_voxel.postprocess.to_glb` with `decimation_target=1000000`, `texture_size=4096`, `remesh=True`, `remesh_band=1`, and `remesh_project=0`: `vendors/Pixal3D/inference.py:263`.
- The export reference performs CuMesh hole fill, BVH construction, optional remesh, simplify/cleanup, UV unwrap, UV-space rasterization, original-mesh BVH projection, sparse PBR voxel sampling, inpaint, material creation, and GLB coordinate conversion: `vendors/TRELLIS.2/o-voxel/o_voxel/postprocess.py:14`.
- CuMesh hole fill is perimeter-limited boundary-loop centroid-fan filling, not arbitrary projected ear clipping: `/tmp/CuMesh/src/clean_up.cu:450`.
- CuMesh simplification is QEM-style edge collapse with edge-length and skinny-triangle costs, boundary awareness, cost propagation, and thresholded collapse loops: `/tmp/CuMesh/src/simplify.cu:531`.
- CuMesh remesh is narrow-band dual contouring driven by BVH unsigned-distance queries, sparse voxel hashing, quad split selection, and optional projection back to the source mesh: `/tmp/CuMesh/cumesh/remeshing.py:24`.
- CuMesh unwrap delegates chart packing to xatlas through a wrapper surface after accelerated chart clustering; in this cycle that is behavior reference for native chart/packing metrics, not approval to add a required dependency: `/tmp/CuMesh/cumesh/cumesh.py:408`.
- CuMesh and vendored TRELLIS.2/o-voxel are MIT-compatible implementation references; `xatlas` is a reference metric source unless explicitly approved as a dependency later; `nvdiffrast` is NVIDIA Source Code License and must be treated as behavior reference only, not copied or line-ported.
- Current `mlx-spatialkit` already has an end-to-end native pipeline in `packages/mlx-spatialkit/src/mlx_spatialkit/export.py:213`, but current risk is in local mesh/UV/texture fill heuristics rather than missing orchestration.

## Required Outcome

1. A Pixal3D export reference map exists in the active plan/code comments/tests as executable stage expectations, not a loose narrative.
2. The native GLB path uses a source-grounded stage order:
   - decoded NPZ validation
   - FlexiDualGrid mesh extraction
   - CuMesh-style hole fill and cleanup semantics
   - narrow-band DC remesh or an explicitly measured equivalent
   - QEM-like simplification/repair sequence
   - xatlas-behavior-compatible native unwrap diagnostics and chart/packing behavior, without making xatlas a required dependency unless explicitly approved later
   - UV raster bake behavior equivalent to nvdiffrast rasterize/interpolate without copying restricted code
   - original high-resolution mesh BVH projection before PBR voxel sampling
   - trilinear sparse voxel sampling and seam fill/inpaint with diagnostics that separate true samples from fills
3. Existing local heuristics that produce fewer holes but smeared or low-granularity color are reverted, disabled, or demoted unless they are justified against the reference contract.
4. The real Pixal3D decoded fixture exports a visually coherent GLB under `/tmp`, with geometry identity and texture quality inspected against the existing reference GLB and diagnostics.
5. Remaining gaps, including xatlas chart equivalence, 1M-face/4096-texture parity, or full remesh parity, stay explicit in diagnostics and docs until they are actually closed.

## Constraints

- `mlx-spatialkit` remains importable without MLX, Torch, or CUDA.
- Python may orchestrate, load arrays, and write diagnostics. Python must not own per-face, per-texel, per-voxel, BVH, remesh, or simplification hot loops.
- Use native C++/Objective-C++/Metal for performance-critical work, with deterministic allocation guards and clear exceptions.
- Do not copy or structurally port `nvdiffrast` CUDA kernels; use it only as behavior reference for rasterization/interpolation semantics.
- Do not add required or vendored `xatlas` in this cycle without explicit approval; use its behavior and CuMesh/o-voxel stage contracts as references for native chart/packing diagnostics and implementation.
- Keep heavy generated GLBs, screenshots, diagnostics, and scratch exports under `/tmp`.
- Do not tag, release, publish, push, or change release metadata in this scope.
- Preserve git hygiene: read files before editing, keep changes scoped, and commit in coherent checkpoints once implementation slices are verified.
- Use read-only subagent audits for large independent source searches, then integrate results through the coordinator instead of scattering roadmap/spec fragments.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| PXR-01 | The active plan has one source-grounded stage map from Pixal3D/o-voxel/CuMesh/xatlas/nvdiffrast behavior to `mlx-spatialkit` modules. | Plan or linked slice cites concrete source paths and names the owner module for every stage. |
| PXR-02 | Hole fill/cleanup behavior matches CuMesh semantics or records a measured blocker. | Focused C++ tests cover perimeter-limited centroid-fan hole fill, duplicate/degenerate cleanup, nonmanifold handling, and boundary metrics. |
| PXR-03 | The geometry path stops relying on local preview clustering as the quality path. | Diagnostics for the Pixal3D quality preset report a reference-grounded remesh/simplify backend or an explicit blocker, not hidden `geometry_aware_preview` success. |
| PXR-04 | UV unwrap behavior is xatlas-behavior-compatible without an unapproved required xatlas dependency. | Tests and diagnostics report chart count/utilization/coverage against the reference trace; package build stays free of required xatlas wiring unless explicitly approved later. |
| PXR-05 | Texture bake projects UV samples back to the original high-resolution mesh before voxel sampling. | Native bake tests prove projected positions are used; real-fixture diagnostics report projection usage and sample/fill counts separately. |
| PXR-06 | Metal replaces CUDA raster/grid sampling hot behavior without copying restricted code. | Code inspection and tests show native Metal/C++ raster/interpolate/sample implementation; `nvdiffrast` remains behavior reference only. |
| PXR-07 | The real decoded Pixal3D fixture produces a visually coherent GLB under `/tmp`. | Heavy test writes GLB/diagnostics under `/tmp`; visual compare and manual Preview/browser inspection no longer show the known hole-patch smear regression. |
| PXR-08 | Memory and thread-safety contracts hold. | Diagnostics include stage timing/RSS; native code avoids unguarded mutable global state and releases stage-local large arrays. |
| PXR-09 | Docs match behavior. | `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, and script help describe the active quality preset, reference gaps, `/tmp` output policy, and remaining non-equivalence. |
| PXR-10 | Repo cleanliness holds. | `git diff --check`, package tests, root Pixal3D integration tests, and `git status --short` show no generated heavy artifacts or unrelated churn. |

## Scope Coverage Decisions

- **Included:** CuMesh-style hole fill/cleanup, narrow-band DC remesh or measured equivalent, QEM-like simplification/repair, xatlas-behavior-compatible native unwrap diagnostics/behavior, UV raster bake, original-mesh BVH projection, sparse PBR voxel sampling, texture seam fill diagnostics, visual comparison, docs, heavy `/tmp` verification, and any necessary revert/demotion of unsafe local heuristics.
- **Deferred:** release/tag/publish work, pushing local tags, claiming full CUDA/cuMesh parity before evidence, copying `nvdiffrast`, broad model-inference changes outside decoded NPZ to GLB export, and publishing generated heavy artifacts.
- **Anti-goals:** continuing one-problem-at-a-time heuristic patching, accepting valid GLB structure as visual success, hiding preview algorithms behind `production_quality_ready=true`, adding Python hot loops for native work, or expanding the roadmap into many micro-phases again.

## Risks

- **Reference-port size:** The real upstream path spans mesh topology, remesh, unwrap, raster, projection, and texture sampling. Mitigation: keep one spec but implement through ordered, verifiable slices.
- **License/provenance risk:** `nvdiffrast` cannot be copied safely, and xatlas is not an approved required dependency in this cycle. Mitigation: behavior tests and clean native implementation; MIT projects retain notices only if explicitly approved for copied code later.
- **Quality regression risk:** More hole filling can preserve silhouette while damaging color granularity. Mitigation: require visual compare, embedded texture inspection, and sample/fill diagnostics before accepting a geometry repair.
- **Runtime risk:** Reference-style remesh and BVH projection can be expensive on 8M-face source meshes. Mitigation: native hot paths, stage memory ownership, `/tmp` heavy runs, and RSS/timing gates.
- **Scope drift risk:** xatlas, CuMesh, remesh, and bake are each large. Mitigation: the plan must map all stages but keep acceptance tied to decoded Pixal3D GLB quality, not a general-purpose mesh library.

## Blocking Questions Or Assumptions

No blocking questions remain.

Assumptions:

- The current decoded Pixal3D 1024 cascade fixture and existing reference GLB are sufficient as the primary real-fixture gate for this cycle.
- It is acceptable to revert or demote recent native heuristics if the reference audit shows they are the source of smear, holes, or misleading readiness status.
- xatlas is not introduced as a required native dependency unless explicitly approved later; current work uses it only as behavior reference and metric vocabulary.
