# mlx-spatialkit Export Quality Hardening Spec

## Bounded Goal

Make `mlx-spatialkit` reject and fix the current sparse-color Pixal3D GLB export so the real 1024 cascade turtle fixture produces a visually coherent colored GLB, not only a structurally valid GLB.

## Broader Intent

`mlx-spatialkit` is intended to become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant C++/Metal hot paths, consistent diagnostics/contracts, memory-aware execution, thread-safe native ownership, and no hidden half-stubbed behavior behind passing tests.

This change is the next cycle toward that broader goal. It does not claim full production remeshing parity; it closes the immediate visual-quality failure and makes remaining quality limitations explicit and test-visible.

## Work Scale And Shape

- Scale: capability hardening
- Shape: parity + coverage + native performance + diagnostics

## Selected Lenses

- **product:** The generated GLB should be inspectable in ordinary viewers without appearing as uncolored dots over a vague surface.
- **engineering:** Native C++/Metal owns per-face, per-texel, per-voxel, and mesh-processing hot paths; Python stays orchestration-only.
- **runtime:** The real fixture is multi-million-token and must remain memory-aware, thread-safe, and repo-clean, with heavy outputs under `/tmp`.

## Target User Or Stakeholder

Apple Silicon `mlx-spatial` developers who need Pixal3D image-to-GLB output that is visually useful, fast enough to iterate on, and honest about quality/performance limitations.

## Evidence From Current Repo

- Current `export_pixal3d_glb(...)` produces a valid GLB from `inputs/mlx-spatialkit/pixal3d-1024-cascade-decoded-pbr`.
- The current real turtle run wrote `/tmp/mlx-spatialkit-pixal3d-glb-current/model.glb` in about `4.7s`, with peak RSS about `3.25 GB`.
- Current geometry path extracted `8,304,022` source faces and simplified to `50,000` faces.
- Current GLB embeds base-color and metallic-roughness textures, but the base-color texture has only `12,023 / 1,048,576` colored texels, about `1.15%` coverage.
- Current diagnostics record `missing_texel_count=346245`, which means many UV-rasterized texels hit mesh surface but failed sparse voxel sampling.
- Current tests accept `sampled_texel_count > 0`, so they can pass while the visual output is mostly uncolored.
- Current native simplification in `packages/mlx-spatialkit/cpp/simplify.cpp` selects evenly spaced source faces and compacts vertices; that is a preview-quality placeholder unless it is replaced or clearly surfaced as such.

## Required Outcome

1. Texture bake diagnostics distinguish:
   - UV-rasterized texels
   - exact sparse-voxel hits
   - native fallback-filled texels
   - still-missing texels
   - final visible base-color texel coverage
2. The Metal/C++ texture path fills the real Pixal3D fixture's UV-covered surface with coherent base color using native hot paths, not Python per-texel loops.
3. Real-fixture tests inspect the embedded GLB base-color PNG or exported diagnostics strongly enough to fail the current sparse-dot output.
4. Mesh simplification quality is no longer hidden:
   - either replace the current face-stride simplifier with a bounded quality-aware native simplifier, or
   - label it explicitly as preview-quality in diagnostics and tests while preventing production-ready claims.
5. The package docs and `mlx-spatial` integration docs explain the quality tier, diagnostics, `/tmp` heavy-artifact policy, and remaining limitations.

## Constraints

- `mlx-spatialkit` remains independent from MLX, Torch, and root `mlx-spatial` runtime imports.
- Python may load files, call native APIs, and inspect small metadata. Python must not own per-face, per-texel, per-voxel, or per-neighbor hot loops.
- Metal code must keep explicit allocation guards and deterministic error paths.
- Native code must be safe for independent concurrent Python calls; no unguarded mutable global caches.
- Heavy generated GLBs, diagnostics, textures, and benchmark outputs must stay under `/tmp`.
- Do not tag, release, publish, or push as part of this change.

## Risks

- **Texture fallback cost:** wider sparse-voxel search can become expensive. Mitigation: bounded search radius, diagnostics for fallback work, and real-fixture runtime/RSS recording.
- **Color bleeding:** image-space dilation can cross face-atlas tiles. Mitigation: prefer model-space native fallback or tile-aware fill, and test surface coverage separately from full texture coverage.
- **Simplifier scope creep:** full production remesh is too large for this cycle. Mitigation: make the current quality tier explicit and only replace the simplifier if the bounded native path is practical.
- **Preview mismatch:** macOS Preview may display GLB materials differently from web viewers. Mitigation: inspect embedded GLB texture payloads and diagnostics, not only Preview screenshots.
- **Memory pressure:** real decoded artifacts and native buffers are large. Mitigation: keep stage-local ownership, delete released arrays, and retain RSS samples in diagnostics.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| SKQ-01 | Texture diagnostics expose surface coverage, exact hits, fallback fills, still-missing texels, and final visible coverage. | `tests/test_texture_bake.py` asserts the new stats on synthetic fixtures. |
| SKQ-02 | The native texture path no longer emits sparse-dot base color for the real turtle fixture. | Heavy test asserts final visible base-color coverage is high relative to UV-rasterized coverage and rejects the current `~1.15%` result. |
| SKQ-03 | Texture filling remains native-owned. | Code inspection and tests show filling is implemented in C++/Objective-C++/Metal, with Python only calling the native API. |
| SKQ-04 | GLB texture payload is verified, not only GLB header/chunks. | Heavy test extracts or inspects the embedded baseColor PNG and asserts non-sparse alpha/RGB coverage. |
| SKQ-05 | Mesh simplification quality is honest. | Diagnostics and tests identify the simplifier backend/quality tier and fail any production-ready claim while a preview simplifier is still used. |
| SKQ-06 | Real fixture stays operational under `/tmp`. | `pytest -m heavy` writes generated GLB/diagnostics under `/tmp` and records runtime/RSS samples. |
| SKQ-07 | Docs are consistent with actual behavior. | `packages/mlx-spatialkit/README.md` and Pixal3D docs describe spatialkit quality tier, diagnostics, fallback behavior, and heavy output policy. |
| SKQ-08 | Repo cleanliness holds. | `git status --short`, package build artifact checks, and `/tmp` output policy confirm no generated heavy artifacts are tracked or bundled. |

## Scope Coverage Decisions

- **Included:** texture-coverage diagnostics, native texture fallback/fill, GLB embedded-texture validation, real turtle fixture quality gate, simplifier quality-tier surfacing, docs, and package/root verification.
- **Deferred:** exact upstream Pixal3D remesh parity, 1M-face production export target, 4096 texture target, complete xatlas replacement, broad mesh editing toolkit, and release/tag work.
- **Anti-goals:** accepting a GLB as done because its header is valid, Python per-texel repair loops, hiding preview-quality simplification, generated artifacts in the repo, or claiming production readiness before visual parity evidence exists.

## Blocking Questions Or Assumptions

No blocking questions remain.

Assumptions:

- A bounded native fallback/fill strategy is acceptable for this cycle if it makes the output visibly coherent and records that it is not full upstream remeshing parity.
- The immediate real-fixture quality target is a usable 1024 texture / 50k-face GLB under `/tmp`; upstream-style 1M-face / 4096 texture export remains a later quality phase.
