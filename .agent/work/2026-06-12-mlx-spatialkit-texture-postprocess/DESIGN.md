# DESIGN — Stage 4 Texture Postprocess

Change: `2026-06-12-mlx-spatialkit-texture-postprocess` · SPEC: `SPEC.md` (TPP-01..09)

## Architecture overview

Three independent subsystem upgrades inside the existing bake (`metal/texture_bake.mm`
+ `metal/kernels/texture_bake.metal` + `texture.py` + `export.py` gates). No bake
architecture rewrite; UV-binned rasterization, BVH source projection, and the stats
contract stay.

## D1. Telea-equivalent inpaint (native C++)

Telea (2004) = fast-marching method: march the inpaint boundary inward by distance
(min-heap on T = distance to known region), painting each unknown pixel as a
normalized weighted average of known neighbors in a radius-B window. Weights =
directional (dot of vector-to-neighbor with FMM gradient direction) × geometric
(1/d²) × level-set proximity. Implemented as a standalone C++ entry
(`telea_inpaint`) in a new `cpp/inpaint.cpp`:

- Input: HxWxC uint8 image (C∈{1,3,4}), HxW uint8 mask (nonzero = inpaint), radius.
- Internal computation in float; output uint8 with round-half-up (matches cv2).
- Deterministic heap order: tie-break (T, y, x) named comparator — cv2's heap order
  is not spec'd, so bit-exactness is a non-goal; parity is bounded per-pixel error.
- CPU is fine: reference itself runs cv2 on CPU; mask sizes are boundary-band scale.
  Metal port only if 4096 budgets fail (SPEC A2).

Application parity (reference `postprocess.py:287-292`): inverse rasterization-
coverage mask; base_color RGB radius 3; metallic/roughness/alpha radius 1 each as
single channel. Our coverage labels: mask = texels NOT in {sampled(1), fallback(4),
surface_filled(5)}… reference equivalent is `rast alpha == 0` i.e. our
`no_face(0) + missing(2) + out_of_grid(3)` *before* dilation. Reference-path
postprocess therefore becomes: rasterize + trilinear → Telea over inverse coverage
(replacing dilate/BFS/gutter on that path) → alpha forced opaque like reference
(`alpha_mode='OPAQUE'`). Legacy path keeps dilation/BFS/gutter untouched, selected
by a new `postprocess` argument (`"legacy-dilation"` default, `"telea"` reference);
`postprocess_mode` stat reports truthfully (`"native-telea-inpaint"`).

Oracle: pinned `opencv-python-headless` in `/tmp/inpaint-oracle-venv` (pip-xatlas
pattern). Generator tool dumps real fixture bake channels+mask to /tmp cache, runs
cv2 in subprocess venv, writes `tests/data/inpaint_oracle_anchors.json` (provenance:
cv2 version, fixture hashes, per-channel error summaries vs our implementation are
asserted in tests, not stored as our output).

## D2. GPU trilinear sampling

Today (reference path): kernel writes nearest + surface_positions → CPU
`apply_reference_projection_and_trilinear_sampling` BVH-projects positions onto the
source mesh and CPU-trilinear-samples sparse voxels (KNN fallback at sparse
boundaries). The reference flow equivalence is already there; the gap is the
sampling leg is CPU and the gate string is `"trilinear-with-sparse-knn-fallback"`.

Design: add a **second Metal kernel pass** `mlx_spatialkit_trilinear_sample`
(in the existing metallib) that takes projected positions (computed CPU-side by the
existing BVH — BVH stays CPU) + sorted voxel keys/attrs and performs 8-corner
trilinear with present-weight renormalization, bit-matching the CPU
`trilinear_sample_attributes` math (float path identical; binary-search per corner
mirrors `find_voxel_key`). CPU KNN fallback then runs only on texels the kernel
flags (missing all corners) — unchanged semantics, same stats counters. CPU
implementation is retained as the no-Metal-device fallback and as the equivalence
test oracle (GPU == CPU exactly, or within 1 ULP-scale tolerance measured then
pinned).

Honesty: `sampling_mode` becomes `"trilinear"` ONLY because the primary measured
path is trilinear; the fallback is reported in separate stats
(`nearest_fallback_scope`, fallback texel counts) and the export gate becomes a
computed verdict: mode == "trilinear" AND `trilinear_sampled_texel_count` ≥
(surface − fallback − invalid) AND fallback fraction bounded. Documented deviation
vs reference: (a) sparse-voxel KNN fallback (reference has a dense volume),
(b) present-weight renormalization (reference grid_sample blends toward zero at the
shell boundary — recorded-not-target; renormalization avoids edge darkening).
Backend stat string updates from `metal-uv-binned-nearest` to a truthful name; the
export.py `texture_backend` check updates in the same slice (intended semantic
change, slice-7-of-unwrap pattern).

## D3. 4096 unblock

Guard semantics stay (guard is correct); the blocker is budgets + the upstream
parity check. `export_pixal3d_glb(texture_size=4096)` already resolves
`max_texture_pixels = texture_size²` (export.py:513), so the guard passes — what is
unproven is wall-time/RSS of: 16.8M-texel kernel, BVH projection of ~16.8M
positions, trilinear pass (GPU after D2), Telea band at 4096. Measure on both
fixtures, then pin budgets in heavy tests. `kMaxTextureDimension` checked ≥ 4096.
If CPU Telea blows budget: band-limit FMM (only texels within radius of coverage
boundary need full quality; far gutter converges to constant) before any Metal port.

## Verification topology

Same harness as unwrap change: per-slice pytest sections (non-heavy synthetic +
heavy fixture parity), anchors JSON + regeneration tool, anti-gaming tests on the
two flipped gates, cross-process determinism test, final dependency-diff +
full-suite wrap. Direct execution route (user's standing preference), single
human-verify checkpoint after the 4096 e2e slice (GLB visual inspection — the
stage the user flagged visually lives here).
