# Pixal3D GLB Quality Rebaseline Gap Matrix

This matrix is the source-grounded signal harvest for `SPEC.md`. It separates what is working from what is not, so the next plan does not rely on false completion signals.

## Artifact Taxonomy

| Label | Owns | Example | Readiness Meaning |
|---|---|---|---|
| A: decoded model output | Pixal3D input and model generation | `shape_decoder_fields.npz`, `texture_decoder_pbr.npz` | Proves inference produced fields; does not prove GLB quality. |
| B: native GLB | `mlx-spatialkit` export | `/tmp/mlx-spatialkit-*/model.glb` | Proves native export result; must be compared to A and C. |
| C: reference/control GLB | upstream/internal/xatlas path or approved control | `pixal3d-1024-cascade-glb-reference/model.glb` | Anchors expected export behavior for the same decoded input. |
| Browser/Preview proof | viewer evidence | `/tmp/.../browser_render/comparison.png` | Shows render symptoms; not a root-cause diagnosis by itself. |

Comparison manifests are authoritative. A heavy comparison must record A/B/C roles, `lineage_id`, source image/preprocess variant, decoded paths, GLB paths, trace paths, commands/settings, diagnostics paths, and browser-proof artifacts when present. Folder names are not enough; comparisons must fail closed if A and C do not share lineage.

## Working Signals

| ID | Signal | Evidence | Keep |
|---|---|---|---|
| W-01 | Native Pixal3D texture coordinate order is largely corrected. | Native bake reports and tests use `batch-x-y-z`; coordinate-order is not the main remaining turtle/violin issue. | Keep focused sentinel tests outside only-heavy fixtures. |
| W-02 | PBR channel packing appears correct. | Native writes baseColor RGBA and metallic-roughness as `R=0, G=roughness, B=metallic`, matching reference material shape. | Do not chase channel swap unless a controlled material test proves it. |
| W-03 | Flexible dual-grid extraction is comparatively close. | Audits found native extraction aligned with o-voxel inference extraction. | Treat extraction as lower-priority unless fixture metrics disprove it. |
| W-04 | Pre-simplify small-loop filling improved some artifacts. | It fills many clean loops before simplification and improved visible turtle/city coherence. | Keep as a partial checkpoint, not production parity. |
| W-05 | Browser render proof catches blank/framing failures. | Current script renders candidate/reference views and visible-pixel ratios. | Keep as evidence, but do not treat as visual-quality gate alone. |

These working signals are regression contracts for this change. New code must preserve them with focused tests before claiming progress on new quality gates.

## Confirmed Gaps

| ID | Area | What Is Wrong | Why It Matters | Evidence Direction |
|---|---|---|---|---|
| G-01 | Automaton state | Prior rendered-visual work is marked `verified` while readiness is false. | Misleads planning and hides unfinished quality work. | `current.json` vs plan evidence. |
| G-02 | Readiness naming | `summary.all_passed`, `artifact_ready`, and `quality_ready` are easy to read as visual success. | Creates false completion illusions. | Tests allow scalar pass while `rendered_visual_ready=false`. |
| G-03 | Artifact provenance | Heavy outputs lack a consistent manifest tying decoded input, native GLB, reference GLB, settings, diagnostics, and browser proof. | Makes A/B/C comparisons easy to confuse. | Violin raw/preprocessed/native artifacts showed this risk. |
| G-04 | Pixal3D preprocessing | Our MLX path feeds raw image to MoGe/DINO/NAF instead of upstream preprocessed RGB. | Causes background leakage such as the violin white sheet. | Upstream preprocess contract vs current `pixal3d_inference.py` raw opens. |
| G-05 | Stage conditioning | Our path reuses one 512 DINO hidden-state tensor across 1024 stages. | Can weaken thin structures and high-res texture detail. | Upstream has stage-specific image conditioning. |
| G-06 | Remesh | Narrow-band dual-contour remesh is missing. | Topology remains rough/open even after hole fill. | Reference routes `remesh=True` to CuMesh narrow-band remesh. |
| G-07 | Simplification | Native “topology-aware” simplifier is clustering plus QEM scoring, not QEM edge collapse. | Creates faceting and cannot claim reference simplification parity. | Native stats label QEM as not edge-collapse. |
| G-08 | Hole repair | Current repair handles small closed loops, not simple/branched open chains or complex topology. | GLBs can be artifact-ready while visibly holed. | Open-chain counts remain visual blockers; real fixture residuals are branched. |
| G-09 | UV unwrap | Native chart is not xatlas-equivalent and has lower occupancy/more island risk. | Affects granularity, seams, and texture coverage. | Native chart parity gate already reports not equivalent. |
| G-10 | Texture sampling | Native has multiple sampling regimes and fallback behavior; root TRELLIS path still has unnormalized sparse trilinear risk. | Can dim color/roughness or create inconsistent texture response. | Audits found native normalized but root path risk remains. |
| G-11 | Texture postprocess/render padding | Native dilation/BFS/gutter/render padding fill is not reference Telea-style inpaint and mutates shared output buffers across stages. | Can improve counters while creating copied/averaged regions, seam bleed, or granular patches. | Reference inpaints base color, metallic, roughness, alpha; stage-attribution tests are missing. |
| G-12 | Normals/material/viewer | Native may recompute normals after UV duplication and smooth hard edges; viewer compatibility can pass while texture semantics are wrong. | Can create fragmented highlights, over-smoothed creases, or misleading render proof. | Audit flagged seam smoothing vs hard-edge preservation risk and viewer-proof limits. |

## Measurement Tests To Plan

| Test | Purpose | Expected Decision |
|---|---|---|
| Manifest lint | Verify A/B/C roles, lineage, settings, and diagnostics are explicit before comparing artifacts. | Fail closed on mismatched or ambiguous comparisons. |
| Preprocess parity fixture | Compare raw, upstream-style preprocessed, and current MLX image inputs without full generation. | Decide exact Pixal3D preprocessing implementation and metadata. |
| Stage DINO/NAF fixture | Compare 512-only hidden states vs stage-specific 512/1024 hidden states. | Decide whether stage-specific conditioning is mandatory for quality. |
| UV-only constant field | Bake constant white/roughness=1/metallic=0 through native chart and reference/control paths. | Separate UV occupancy from texture sampling defects. |
| Sampling-only fixed mesh | Compare nearest, normalized trilinear, fallback-disabled, and CPU/reference sampling while asserting the full `coverage_status` histogram. | Separate sampling correctness from UV and postprocess. |
| Postprocess-only buffers | Apply native fill/render padding vs reference-like inpaint to identical raw texels. | Decide whether native fill is causing granularity/seam artifacts. |
| Material/normal variants | Render identical geometry/texture with controlled roughness/metallic and normals variants. | Separate viewer/material/normal effects from texture content. |
| Two-fixture heavy gate | Run the base Pixal3D 1024 cascade fixture plus one independent violin/bow lineage. | Prevent overfitting to one asset. |

## Regression Guardrails

| Guardrail | Preserves | Verification Direction |
|---|---|---|
| Texture coordinate sentinel | Pixal3D `batch-x-y-z` texture coordinate order | Existing texture bake tests must stay green before and after bake changes. |
| PBR packing sentinel | Base color RGBA and metallic-roughness channel layout | GLB writer/material tests must cover unequal channel values. |
| Extraction/metrics sentinel | Flexible dual-grid extraction and boundary metrics | Mesh processing tests must keep clean-loop/open-chain counters stable. |
| Viewer compatibility sentinel | GLB parseability, normals/material payload shape, browser blank/framing proof | Viewer compatibility remains orthogonal to visual quality readiness. |
| Fixture boundary sentinel | Existing base and violin/bow decoded fixtures | Manifest work adds lineage, not heavy generated repo artifacts. |

## Planning Implications

- Start by correcting state and readiness vocabulary before deeper implementation.
- Then implement artifact provenance and Pixal3D input parity because those determine whether a defect belongs to generation or export.
- Only after A/B/C provenance is stable should native export work proceed into UV/bake attribution and remesh/QEM/open-chain fixes.
- If narrow-band remesh/QEM proves too large, the plan should split that as a blocked production-parity subgoal while still delivering honest readiness gates.
- Historical `.agent/work/*` artifacts should be read as evidence but not edited for this change; stale meanings are superseded from the active spec/plan.
