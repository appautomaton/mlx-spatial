# mlx-spatialkit GLB Viewer Compatibility Gate Spec

## Bounded Goal

Make native Pixal3D GLB exports structurally friendlier to macOS Preview/Quick Look and other strict viewers by adding normals, avoiding large single-primitive `UNSIGNED_INT` index buffers, and reporting this compatibility contract in diagnostics.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant Metal/C++ hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

## Work Scale And Shape

- Scale: capability hardening
- Shape: native GLB writer and diagnostics parity

## Selected Lenses

- **engineering:** Bring the native GLB writer closer to upstream Pixal3D/trimesh export expectations by carrying normals and emitting viewer-compatible primitive chunks.
- **runtime:** Keep the writer native, bounded, and memory-conscious for 200k-face and 1M-face exports.
- **product:** A user opening `model.glb` in common viewers should not see an uncolored point-like placeholder when the artifact is meant to be a textured mesh.

## Current Evidence

- The current spatialkit GLB writer emits only `POSITION` and `TEXCOORD_0` attributes from `packages/mlx-spatialkit/cpp/glb_writer.cpp`.
- The checked Pixal3D reference/output GLB has one primitive, `312093` vertices, `212542` faces, `UNSIGNED_INT` indices (`componentType=5125`), and no `NORMAL` attribute.
- Browser visual proof renders the checked GLB, but macOS Preview/Quick Look is the user-visible complaint surface and may be stricter about normals and large `UNSIGNED_INT` primitives.
- Vendored Pixal3D carries `vertex_normals` into its `trimesh.Trimesh(... visual=TextureVisuals(...))` export path after CUDA uv unwrap and texture baking.

## Required Outcome

1. Native GLB payloads include a finite normalized `NORMAL` accessor and every mesh primitive references it.
2. Native GLB payloads split large geometry into local primitive chunks whose index accessors use `UNSIGNED_SHORT` and stay at or below `65535` local vertices.
3. Pixal3D export diagnostics include an explicit GLB viewer-compatibility section checking normals, primitive chunking, index component type, material/texture presence, and parseability.
4. Real Pixal3D fixture verification proves the reference-target export writes a textured GLB with normals, uint16-only primitive indices, browser render proof, and the existing xatlas chart parity deferral still intact.
5. Docs explain that this improves strict-viewer compatibility and GLB structure, but does not implement xatlas charting or upstream CUDA/cuMesh remesh parity.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| GVC-01 | GLB writer emits `NORMAL` attributes for all primitives. | Focused writer test parses GLB JSON and asserts each primitive has `NORMAL` and the normal accessor is `VEC3` float32. |
| GVC-02 | Large meshes avoid `UNSIGNED_INT` primitive indices. | Focused writer test with more than 65535 vertices asserts multiple primitives and every index accessor uses `5123`. |
| GVC-03 | Export diagnostics expose viewer compatibility readiness. | Focused export/diagnostic test asserts `quality.glb_viewer_compatibility.all_passed=true` with named checks. |
| GVC-04 | Real Pixal3D fixture passes the compatibility gate. | Heavy real export under `/tmp` reports normals, uint16-only chunked indices, browser proof still available, and `not_xatlas_chart_parity` remains deferred. |
| GVC-05 | Docs and hygiene hold. | Docs updated; focused package tests, root Pixal3D tests, build to `/tmp`, and artifact inspection pass. |

## Scope Coverage Decisions

- **Included:** native normal generation, primitive chunking, GLB inspection/diagnostic coverage, real fixture proof, docs, package/root/build verification.
- **Deferred:** xatlas chart parity, CUDA/cuMesh remesh parity, exact macOS Preview automation, changing decoded model outputs, release/tag/push/publish work.
- **Anti-goals:** claiming xatlas parity, adding runtime Python GLB writer fallback, adding package runtime dependencies, writing heavy artifacts outside `/tmp`, relaxing existing production thresholds.

## Constraints

- Keep the hot writer path native C++.
- Keep generated and heavy artifacts under `/tmp`.
- Do not change model inference or decoded NPZ artifacts.
- Preserve existing texture material semantics and embedded PBR PNGs.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **GLB size/runtime risk:** Chunk-local vertex buffers may duplicate shared vertices in generic meshes. Mitigation: greedy chunking with bounded local buffers and real fixture memory checks.
- **False parity risk:** Viewer-compatible GLB structure could be mistaken for xatlas/cuMesh parity. Mitigation: keep `not_xatlas_chart_parity` explicit.
- **Preview automation risk:** `qlmanage` thumbnailing may be unreliable in headless/session contexts. Mitigation: use structural GLB checks plus browser render proof, and document the boundary.

## Blocking Questions Or Assumptions

Assumption: the next valuable production step is to harden GLB structure for strict viewers before attempting xatlas chart equivalence, because the observed user-facing failure is a viewer/opening symptom rather than a missing decoded model stage.
