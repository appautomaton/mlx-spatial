# Pixal3D Rendered Visual Correctness Spec

## Bounded Goal

Fix the current `mlx-spatialkit` Pixal3D native-chart GLB so the real 1024 testing asset renders with coherent color, material response, surface smoothness, and hole behavior in actual GLB viewers, not just as a structurally valid artifact with passing coverage counters.

## Broader Intent

The prior reference-port checkpoint made the export path runnable and diagnostics more honest, but the latest Preview-rendered turtle/city GLB still shows broken surface appearance: gray/black metallic noise, wrong color response, visible holes or white seams, and overly weak `visual_comparison.summary.all_passed=true` diagnostics. This cycle must make rendered visual evidence authoritative before returning to 4096/upstream coverage work.

## Selected Lenses

- **product:** The turtle body, shell, legs, head, and city skyline should be readable with plausible Pixal3D color/materials in Preview or a browser GLB renderer.
- **engineering:** Diagnose and fix the actual GLB material/texture/surface path, especially reference-like texture inpaint/fill, sparse trilinear sampling, normal continuity, small-hole handling, UV holes, and rendered visual gates.
- **runtime:** Keep heavy generated outputs under `/tmp`; keep native C++/Metal on hot paths; use reproducible render/diagnostic commands instead of manual-only judgment.

## Source Evidence

- User-provided Preview screenshot of `/tmp/mlx-spatialkit-native-chart-reference-target-export-67097/model.glb` shows gray/black shiny fragmented surface, incorrect color/material response, and holes/white seams despite the object silhouette being readable.
- The same run reports `visual_comparison.summary.all_passed=true`, `artifact_ready=true`, `production_quality_ready=false`, `final_visible_coverage_ratio=0.5204620361328125`, and `uv_surface_final_visible_coverage_ratio=0.9179851371399927`; therefore the current visual gate is insufficient.
- GLB writer material JSON and PBR packing match the Pixal3D/o-voxel reference shape: `alphaMode:"OPAQUE"`, `metallicFactor:1`, `roughnessFactor:1`, and metallic-roughness packed as `R=0, G=roughness, B=metallic`.
- The bad native artifact differs materially from the reference texture: alpha coverage is about `52%` vs reference `100%`, roughness mean is about `118/255` vs reference about `254/255`, and visible base color is much darker/grayer.
- Pixal3D/o-voxel reference postprocess inpaints base color, metallic, roughness, and alpha before GLB export. The current native gutter fill copies RGB/MR into no-face texels but preserves zero alpha, which is not compatible with an opaque rendered GLB texture.
- Current sparse trilinear sampling accumulates present sparse-grid corner weights but does not normalize by present weight; this can dim base color, alpha, and roughness when corners are missing.
- Current GLB normals are recomputed after UV chart seam duplication; the reference computes normals on the mesh and maps them through unwrap, so seam-local recomputation can create fragmented highlights.
- Texture bake diagnostics still report `surface_unfilled_texel_count=43119`, `surface_fill_cross_gap_prevented_count=1349107`, and only `sampled_texel_count=33118` for the latest 1024 native-chart artifact, so holes and fill quality remain first-class suspects.
- Geometry diagnostics still show remaining boundary/open-chain components, and reference export performs small-hole filling before export; native cleanup/simplify must not let holes pass as visual-ready.

## Required Outcome

1. The current bad Preview/browser appearance is reproducible as a failing rendered-visual gate, not dismissed by texture coverage JSON.
2. Texture bake/postprocess produces a render-ready GLB texture while preserving separate diagnostics for raw sampled coverage and true UV-surface coverage.
3. Sparse trilinear sampling does not dim attributes just because some sparse-grid corners are missing.
4. GLB normals and small-hole handling do not create avoidable fragmented highlights, holes, or white seams.
5. GLB material output follows glTF PBR expectations for Pixal3D textures: base color is visible, metallic/roughness channel use is correct, and default factors do not force a metallic black/chrome look.
6. The real 1024 native-chart Pixal3D GLB renders as a visually coherent turtle/city object in a viewer-backed artifact under `/tmp`.
7. Diagnostics distinguish structural validity, rendered visual quality, production quality, and remaining QEM/DC/xatlas parity gaps.

## Constraints

- Do not continue into 1M/4096 coverage work until the 1024 rendered visual quality gate is credible.
- Do not add xatlas, Torch, CUDA, or MLX as `mlx-spatialkit` package dependencies.
- Do not claim success from `visual_comparison.summary.all_passed` unless it includes rendered evidence or is explicitly renamed/scoped as non-rendered.
- Do not hide holes by traversing no-face UV gaps or smearing arbitrary texture values across unowned atlas space.
- Heavy GLBs, screenshots, browser renders, and scratch diagnostics stay under `/tmp`.
- No release, tag, push, or publish work.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| RVC-01 | Current bad rendering is captured by an automated or scripted gate. | Candidate/reference GLB inspection fails on alpha, dark/gray base-color, roughness, and visible coverage signals, not only parseability and face ratio. |
| RVC-02 | Texture postprocess is render-ready without hiding raw coverage. | Embedded GLB base-color alpha/RGB and MR channels are filled like a render texture, while diagnostics still expose exact sampled coverage, UV-surface coverage, no-face/gutter fill counts, and unfilled surface counts. |
| RVC-03 | Sparse trilinear sampling is normalized over present corners. | Focused native texture tests prove missing sparse corners do not dim sampled color/alpha/roughness. |
| RVC-04 | Normals and hole handling stop producing obvious avoidable artifacts. | GLB normals preserve seam-safe smoothness where appropriate, and small-hole diagnostics/fill affect visual readiness. |
| RVC-05 | Real 1024 native-chart GLB is visually coherent. | Heavy real fixture writes a GLB plus evidence under `/tmp`; diagnostics show artifact-ready and rendered-visual-ready, while production-equivalence may remain false for QEM/DC/xatlas gaps. |
| RVC-06 | Diagnostics remain honest. | `production_quality_ready`, `rendered_visual_ready`, and `production_equivalence_ready` are distinct; QEM/DC/xatlas blockers remain explicit unless actually solved. |
| RVC-07 | Repo hygiene holds. | `git diff --check`, focused package tests, heavy real fixture gate, root Pixal3D integration tests, and `git status --short` show no generated heavy artifacts. |

## Scope Coverage Decisions

- **Included:** GLB material/PBR writing, metallic-roughness texture packing, base-color/alpha visibility, reference-like render texture fill, sparse trilinear normalization, normal continuity, small-hole readiness, rendered visual gate, docs/tests for current 1024 native-chart visual correctness.
- **Deferred:** 1M/4096 coverage improvement, full QEM edge-collapse, narrow-band DC remesh, exact xatlas chart parity, and replacing model inference.
- **Anti-goals:** starting a new roadmap, accepting silhouette readability as quality, relying only on PNG coverage counters, or treating Preview/browser artifacts as optional after this bug.

## Risks

- **Renderer differences:** Preview and browser/Three.js may differ. Mitigation: record the renderer used and keep output artifacts under `/tmp`; prefer gates that catch material/coverage failures independent of camera angle.
- **False material fix:** Lowering metallic factors may improve Preview while hiding wrong PBR texture packing. Mitigation: inspect GLB JSON, texture channels, and rendered output together.
- **Hole-smearing regression:** More fill can improve apparent coverage while ruining color granularity. Mitigation: keep no-face gap blocking and surface-fill diagnostics visible.

## Assumptions

- `/tmp/mlx-spatialkit-native-chart-reference-target-export-67097/model.glb` is the current representative failing artifact.
- The user-provided Preview screenshot is sufficient evidence to invalidate the old non-rendered visual pass as a production-quality gate.
