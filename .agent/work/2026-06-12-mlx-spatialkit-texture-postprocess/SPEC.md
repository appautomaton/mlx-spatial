# SPEC — Stage 4: Reference-Parity Texture Postprocess

Change: `2026-06-12-mlx-spatialkit-texture-postprocess`
Date framed: 2026-06-12
Lenses: engineering (primary), runtime (budgets/Metal), product (visual-quality outcome)

## Bounded goal

Close stage 4 of the native Pixal3D→GLB parity roadmap: replace the quarantined dilation
inpaint with a native Telea-equivalent inpaint, make the texture bake's voxel sampling
reference-equivalent trilinear on the GPU path, and unblock+prove `texture_size=4096`
export — flipping the texture-postprocess stage gates to `reference_matched` honestly.

## Broader intent

This is the final stage of the 4-stage parity roadmap (remesh ✅, QEM ✅, UV unwrap ✅).
After this change, `export_pixal3d_glb` matches the reference
`o_voxel.postprocess.to_glb` end-to-end on every quarantined stage. The user's visual
checkpoint on stage 3 ("significant progress, not perfect") attributed residual
softness/seams to this stage; this change is expected to deliver the visible quality step.

## Work scale and shape

Scale: large (one native algorithm port + one Metal kernel upgrade + one scale proof).
Shape: parity — reference source is vendored TRELLIS.2 o-voxel; verification is
gap-ID conformance against a version-pinned dev-time oracle plus measured invariants.

## Reference semantics (parity source)

`vendors/TRELLIS.2/o-voxel/o_voxel/postprocess.py` (`to_glb`):
- **Sampling** (~lines 258–266): texel 3D positions → BVH-correct to original mesh →
  `grid_sample_3d(..., mode='trilinear')` from the dense attribute volume.
- **Inpaint** (~lines 287–292): `mask_inv = ~rasterized_coverage`;
  `cv2.inpaint(img, mask_inv, radius, cv2.INPAINT_TELEA)` with **radius 3 for
  base_color (3ch uint8)** and **radius 1 for metallic, roughness, alpha (1ch each)**.
  Note: the reference inpaints the *entire* inverse mask (gutter + background), uint8
  domain, per-channel-group.

Our current state (anchors):
- Inpaint is dilation + BFS surface fill + gutter fill —
  `metal/texture_bake.mm:883–1185`; stage reported `heuristic_quarantined`
  (`export.py` `_stage_status` for `texture_postprocess`).
- Metal kernel is nearest-only (`metal/kernels/texture_bake.metal`,
  `find_nearest_voxel_key`); a CPU trilinear path exists
  (`trilinear_sample_attributes`, `texture_bake.mm:422–499`) active under source
  projection, reported as `sampling_mode="trilinear-with-sparse-knn-fallback"`. The
  gate requires the exact string `"trilinear"` (standing honesty rule: fix by making
  the measured thing true, never by relabeling).
- 4096 blocked by pixel guard: `max_texture_pixels` defaults to `texture_size²`
  resolution in `export.py:513` and is enforced in `texture_bake.mm:1270–1279`;
  4096² = 16.8M pixels is untested for time/memory.

## Required outcome (gap matrix)

| Gap | Current | Target |
|---|---|---|
| TPP-A inpaint algorithm | dilation/BFS heuristic, quarantined | native Telea-equivalent (FMM) inpaint, parity vs cv2 oracle |
| TPP-B inpaint application | surface-only fills, alpha/coverage-preserving gutter | reference application: inverse-coverage mask, base_color r=3, metallic/roughness/alpha r=1 |
| TPP-C voxel sampling | GPU nearest (+ CPU trilinear under source projection) | GPU trilinear in the Metal kernel, reference grid_sample semantics, honest `sampling_mode="trilinear"` |
| TPP-D 4096 | pixel guard + unproven budgets | `texture_size=4096` export succeeds on both fixtures within measured budgets |
| TPP-E gates | `trilinear_pbr_sampling` + `texture_postprocess` = `heuristic_quarantined` | both `reference_matched` via computed verdicts (no hardcoding) |

## Constraints (carried forward + new)

- **Dependency-light**: NO required OpenCV/Torch/CUDA/MLX/xatlas runtime deps. OpenCV
  is a *dev-time oracle only*, version-pinned in a /tmp venv (same pattern as pip-xatlas
  0.0.11: `/tmp/uvoracle-venv`; new e.g. `/tmp/inpaint-oracle-venv` with pinned
  `opencv-python-headless`), with checked-in anchors + regeneration tool.
- **Gate honesty (standing rule)**: never silently flip gates green. `parity_ready` /
  stage statuses computed from measured invariants. The `sampling_mode=="trilinear"`
  exact-string gate is satisfied by making the kernel actually trilinear, then
  reporting the measured mode.
- **Determinism**: ordered containers, named comparators, cross-process
  PYTHONHASHSEED invariance for the new paths.
- **Budgets**: delta-RSS (not absolute) and wall-time budgets measured then bounded;
  heavy artifacts under /tmp; no release/tag/push.
- **Back-compat**: existing dilation/surface-fill path stays available (it is the
  current default behavior other tests depend on); reference postprocess is selected
  on the reference path the same opt-in way the UV backend was.
- Fixtures: the two cached Pixal3D NPZ fixtures
  (`inputs/mlx-spatialkit/{,violin-bow/}pixal3d-1024-cascade-decoded-pbr/`).

## Risks (implementation-changing)

1. **Telea port complexity**: Telea = fast-marching-method inpaint over float
   intensities with directional weighting. Bit-exactness to OpenCV is not realistic;
   parity must be defined as bounded per-pixel error vs oracle on real fixture
   masks (tolerance set from measurement, anchored).
2. **Sparse vs dense volume**: reference trilinearly samples a *dense* attribute
   volume; ours samples *sparse* voxels (KNN fallback where a trilinear corner is
   missing). Semantics at sparse-boundary texels will deviate — must be measured,
   bounded, and recorded as a documented deviation, not hidden.
3. **Inpaint-domain mismatch**: reference inpaints uint8 full-frame including
   background; our pipeline tracks coverage/alpha labels. Reconciling (apply
   Telea over inverse coverage on the reference path, keep labels consistent for
   GLB alpha) may change downstream stats baselines.
4. **4096 scale**: 16.8M texels × FMM (heap-based, serial-ish) could be slow on CPU;
   may need tiling or a bounded-band optimization. Budgets unknown until measured.
5. **Existing test baselines**: tests assert `postprocess_mode`,
   `sampling_mode`, dilation/gutter counters — intended semantic changes must update
   tests explicitly (as in slice 7 of the unwrap change), never weaken gates.

## Acceptance criteria

- **TPP-01** Native Telea-equivalent inpaint entry point (C++/Metal, no new required
  deps) callable standalone: given image + mask + radius, returns inpainted image;
  unit-tested on synthetic masks (deterministic, mask-only writes, radius respected).
- **TPP-02** Oracle parity: version-pinned cv2 `INPAINT_TELEA` anchors
  (regenerable via a checked-in tool + /tmp oracle venv) on real fixture bake
  masks for radius 3 and radius 1; our inpaint matches within a measured, anchored
  per-pixel tolerance on uint8 output; tolerance and any deviations documented.
- **TPP-03** Reference application parity: on the reference path the postprocess
  applies Telea-equivalent inpaint over the inverse coverage mask with base_color
  r=3 and metallic/roughness/alpha r=1 (reference channel grouping), replacing the
  dilation heuristic there; `postprocess_mode` reports a truthful new mode string.
- **TPP-04** GPU trilinear: the Metal bake kernel performs trilinear voxel sampling;
  GPU output matches the existing CPU `trilinear_sample_attributes` within tolerance
  on both fixtures; sparse-corner fallback semantics measured and documented;
  `sampling_mode` honestly reports `"trilinear"` (gate string) only when the
  measured path is trilinear.
- **TPP-05** Stage gates flip honestly: `trilinear_pbr_sampling` and
  `texture_postprocess` report `reference_matched` via computed verdicts; anti-gaming
  tests prove the verdicts flip back when invariants are violated.
- **TPP-06** 4096 unblocked: `export_pixal3d_glb(texture_size=4096)` on the reference
  path succeeds on both fixtures; the upstream-parity `texture_size` check passes;
  wall-time and delta-RSS measured and bounded in committed budgets.
- **TPP-07** Determinism: new inpaint + trilinear outputs byte-identical across
  processes under PYTHONHASHSEED 0/1.
- **TPP-08** Regression: full non-heavy + heavy suites green; legacy
  dilation/surface-fill behavior unchanged for non-reference paths.
- **TPP-09** Dependency check: build-file/dependency diff shows only new source
  files; suite passes with no OpenCV importable in the project venv.

## Anti-goals

- No OpenCV (or any new third-party) runtime/required dependency.
- No bit-exact OpenCV replication target — bounded-tolerance parity only.
- No bake-architecture rewrite (UV-binned rasterization stays).
- No relabeling gates to pass string checks without the measured behavior.

## Deferred / Not in scope (carried follow-ons)

- Rasterized/skyline packing for atlas utilization density (recorded, unwrap SPEC).
- Non-manifold-tolerant QEM (0-loop parity).
- Reference-scale res1024/1M-face proof (4096 *texture* is in scope here; 1M-face
  meshes are not).
- Per-texel bake round-trip quality metric (candidate diagnostic; only adopt if
  cheap during TPP-04, otherwise defer).

## Assumptions (accepted unless user objects)

- A1: Dev-time oracle pattern (pinned `opencv-python-headless` in /tmp venv +
  checked-in anchors) is acceptable, mirroring the pip-xatlas oracle.
- A2: Telea runs CPU-side C++ first (matching reference, which is CPU cv2);
  Metal port only if 4096 budgets demand it.
- A3: Reference path remains opt-in via existing preset/backend knobs; defaults
  for non-reference paths unchanged.
