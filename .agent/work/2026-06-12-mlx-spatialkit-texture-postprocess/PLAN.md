# PLAN — Stage 4 Texture Postprocess

Goal: SPEC.md bounded goal — native Telea-equivalent inpaint, GPU trilinear bake
sampling, 4096 unblocked, texture gates flipped honestly. Architecture: DESIGN.md.

> Codex read-only review folded in 2026-06-13 (P0/P1): Slice 2 now freezes the raw
> pre-sampling inverse-rasterized-coverage mask as the anchored contract; Slice 3 Telea
> mask = that raw mask (snapshot before any coverage mutation); Slice 4's CPU trilinear
> leg is the equivalence oracle/debug path (NOT a "Metal-unavailable bake" fallback) and
> gains a delta-RSS budget; Slice 5 gate verdicts gain counter-conservation + raw-mask
> identity + legacy-fill-disabled invariants; Slice 2 anchors gain richer provenance.

## Execution routing and topology

- Route: **direct** (coordinator implements; standing user preference from the
  unwrap change — no subagent dispatches for implementation or review).
- Serial order 1→7; continuation is the default after each verified slice.
- Checkpoint: **human-verify after slice 6** (4096 GLB visual inspection — this is
  the stage the user flagged visually on the unwrap checkpoint).
- Parallel-safe groups: none.
- Git rhythm: one commit per verified slice, `slice N: <objective>`, strictly
  additive on the change branch; no release/tag/push.
- Test invocation: `cd packages/mlx-spatialkit && .venv/bin/python -m pytest ...`;
  rebuild via `uv pip install -p .venv/bin/python --no-build-isolation -e .` from
  `packages/mlx-spatialkit` (clean: `rm -rf /tmp/mlx-spatialkit-build`).

## Requirement traceability

| SPEC | Slices |
|---|---|
| TPP-01 inpaint core | 1 |
| TPP-02 oracle parity | 2 (2a raw-contract freeze + 2b anchored parity) |
| TPP-03 reference application | 3 |
| TPP-04 GPU trilinear | 4 |
| TPP-05 honest gates | 5 |
| TPP-06 4096 + budgets | 6 |
| TPP-07 determinism | 1 (unit), 7 (e2e) |
| TPP-08 regression | 7 |
| TPP-09 dependency-light | 7 |

## Slices

### Slice 1: Native Telea inpaint core

**Objective:** Standalone deterministic Telea (FMM) inpaint in C++ with a nanobind
entry point and synthetic-mask unit tests.
**Acceptance criteria:**
- `telea_inpaint(image, mask, radius)` binding exists (uint8 HxW / HxWxC, C∈{1,3,4});
  raises on shape/radius misuse.
- Writes only masked pixels; unmasked bytes bit-identical to input.
- Radius respected (single isolated masked pixel): known pixels beyond radius B do not
  influence its output (windowed-weight test). NOTE (Codex P1): for multi-pixel/connected
  masks Telea propagates through newly-accepted neighbors, so this bound is asserted
  per-isolated-pixel, not across a connected mask band.
- FMM order deterministic (named (T,y,x) tie-break); same input → same bytes across
  repeated calls and across PYTHONHASHSEED 0/1 subprocesses (unit-level TPP-07).
- Qualitative correctness on synthetic gradients: inpainted band continues a linear
  gradient within a small tolerance (catches dilation-style constant fill).
**Verification:** `cd packages/mlx-spatialkit && .venv/bin/python -m pytest tests/test_texture_bake.py -k telea_inpaint -q`
**Touches:** `cpp/inpaint.cpp|hpp` (new), `cpp/bindings.cpp`, `CMakeLists.txt`,
`tests/test_texture_bake.py`.

### Slice 2: cv2 Telea oracle + raw-contract freeze + anchored parity

**Objective:** Version-pinned OpenCV Telea oracle with checked-in anchors proving bounded
per-pixel parity on real fixture bake masks (radius 3 and 1), anchored against a frozen
raw pre-postprocess contract.

**2a — raw-contract freeze (Codex P0-1):** Expose/dump the *raw* pre-postprocess bake
channels (base_color RGB, metallic, roughness, alpha) and the *raw inverse rasterized-
coverage mask* — captured BEFORE any coverage mutation (before dilation/BFS/gutter and
before sampling/fallback overwrite coverage). This raw mask = `rast alpha == 0`, i.e.
`no_face(0) + missing(2) + out_of_grid(3)` (per DESIGN D1). Both the oracle and our
`telea_inpaint` run against this exact frozen contract so parity is not anchored against a
post-mutation mask.

**Acceptance criteria:**
- Tiny synthetic cv2 oracle cases run first (small hand-built image+mask) to catch FMM
  divergence early, before fixture anchoring (Codex P1-4).
- `/tmp/inpaint-oracle-venv` bootstrap documented in tool header; generator
  `tests/tools/gen_inpaint_oracle_anchors.py` (two-phase subprocess, `--reuse-cache` like
  the UV tool) dumps the frozen raw channels + raw mask, runs cv2 in the subprocess venv.
- `tests/data/inpaint_oracle_anchors.json` committed for both fixtures with rich
  provenance (Codex P1-7): cv2 version, fixture hash, **raw-mask hash**, per-channel input
  hashes, per-channel cv2-output digests, and per-channel-group **max / mean / p95**
  error summaries over the masked band (not digests alone — catch shape-preserving fraud).
- Heavy test runs our `telea_inpaint` on the same cached raw channels/mask and asserts
  per-pixel error vs oracle within a measured-then-pinned tolerance (uint8 domain),
  separately for radius 3 (RGB) and radius 1 (single-channel); tolerance recorded in
  anchors, not invented.
- Suite passes with no cv2 importable in the project venv.
**Verification:** `cd packages/mlx-spatialkit && .venv/bin/python -m pytest tests/test_texture_bake.py -k inpaint_synthetic_oracle -q && .venv/bin/python -m pytest -m heavy tests/test_real_pixal3d_export.py -k inpaint_oracle -q`
**Touches:** `tests/tools/gen_inpaint_oracle_anchors.py` (new),
`tests/data/inpaint_oracle_anchors.json` (new), `tests/test_real_pixal3d_export.py`,
`tests/test_texture_bake.py`, and the bake raw-channel/raw-mask exposure hook
(`metal/texture_bake.mm`, `src/mlx_spatialkit/texture.py`) needed to freeze the contract.

### Slice 3: Reference-path postprocess application

**Objective:** Wire Telea into the bake as the reference-path postprocess —
inverse-coverage mask, base_color r=3, metallic/roughness/alpha r=1 — keeping the
legacy dilation path default and untouched.
**Acceptance criteria:**
- `bake_pbr_texture(..., postprocess=...)` (and native arg) selects
  `"legacy-dilation"` (default, byte-identical behavior to today) or `"telea"`
  (reference application per DESIGN D1: mask = the **frozen raw inverse rasterized-
  coverage mask from Slice 2a** (`rast alpha == 0`), captured before any coverage
  mutation; channel grouping and radii per reference; alpha handled like reference
  OPAQUE).
- Alpha parity (Codex P1-5): alpha is Telea-inpainted as a single channel at r=1 like the
  reference, THEN `alphaMode=OPAQUE`; do NOT shortcut alpha bytes to 255 — texture parity
  is claimed on the inpainted alpha channel, and the test asserts the alpha bytes are the
  inpainted values, not a constant fill.
- `postprocess_mode` stat truthfully reports `"native-telea-inpaint"` on the new
  path; dilation/gutter counters report 0/disabled there.
- Coverage labels/stats remain consistent (masked-and-painted texels accounted; new
  `telea_*` counters: band size, painted texel count per channel group).
- Non-heavy unit test on synthetic bake proves path selection + mask semantics;
  heavy test on one fixture proves the reference path produces nonzero painted
  texels and no black-seam texels adjacent to coverage (seam probe).
- Legacy-path regression: existing texture bake tests pass unmodified.
**Verification:** `cd packages/mlx-spatialkit && .venv/bin/python -m pytest tests/test_texture_bake.py -q && .venv/bin/python -m pytest -m heavy tests/test_real_pixal3d_export.py -k telea_postprocess -q`
**Touches:** `metal/texture_bake.mm`, `cpp/texture_bake.hpp`, `cpp/bindings.cpp`,
`src/mlx_spatialkit/texture.py`, tests.

### Slice 4: GPU trilinear sampling pass

**Objective:** Move the reference-path trilinear sampling leg onto the GPU as a
second Metal kernel, bit-matching the CPU implementation, with KNN fallback and
stats semantics unchanged.
**Acceptance criteria:**
- New kernel `mlx_spatialkit_trilinear_sample` in `texture_bake.metal`: 8-corner
  binary-search + present-weight renormalization matching
  `trilinear_sample_attributes`; flags all-corners-missing texels for CPU KNN
  fallback (fallback math unchanged).
- GPU↔CPU equivalence: heavy test samples both paths on both fixtures; channel
  outputs identical (or within measured ≤1-LSB uint8 tolerance, pinned in test).
- CPU `trilinear_sample_attributes` leg retained as the **equivalence oracle / debug
  path** for the GPU pass — NOT framed as a "bake runs without Metal" fallback (the bake
  raster itself requires a Metal device, so a no-Metal whole-bake path is out of scope:
  Codex P0-2). Which leg produced the sampled attributes is reported in stats
  (`sampling_device` = `gpu` | `cpu-oracle`).
- `sampling_mode` reports `"trilinear"` (measured primary path), fallback reported
  via existing `nearest_fallback_*` + counter stats; `backend` string updated to a
  truthful non-"nearest" name; export.py `texture_backend` check updated in the same
  commit (intended semantic change).
- Trilinear texel counters unchanged vs CPU baseline on fixtures (same sampled /
  missing-corner / fallback counts).
- Memory (Codex P0-3): the GPU pass must not double the full-frame position buffer —
  reuse `surface_positions` in place (or stream), proven by a measured delta-RSS
  acceptance on a representative path here, so a doubled ~16.8M-texel buffer is caught at
  Slice 4, not discovered at the 4096 e2e in Slice 6.
**Verification:** `cd packages/mlx-spatialkit && .venv/bin/python -m pytest tests/test_texture_bake.py -q && .venv/bin/python -m pytest -m heavy tests/test_real_pixal3d_export.py -k gpu_trilinear -q`
**Touches:** `metal/kernels/texture_bake.metal`, `metal/texture_bake.mm`,
`src/mlx_spatialkit/export.py` (backend check), tests.

### Slice 5: Honest gate flip

**Objective:** `trilinear_pbr_sampling` and `texture_postprocess` stage gates become
computed verdicts that report `reference_matched` from measured invariants, with
anti-gaming proof.
**Acceptance criteria:**
- `trilinear_pbr_sampling`: reference_matched iff measured `sampling_mode ==
  "trilinear"` AND **counter conservation holds** (`trilinear_sampled + missing_corner_
  fallback + invalid == surface_texel_count`, exact equation, not an inequality) AND the
  **fallback fraction has an explicit denominator** (`fallback / surface_texel_count`) ≤
  pinned bound; deviations (sparse KNN, renormalization) recorded in gate detail
  (Codex P1-6).
- `texture_postprocess`: reference_matched iff `postprocess_mode ==
  "native-telea-inpaint"` AND **raw-raster-mask identity** (the texels Telea painted ==
  the frozen raw inverse-coverage mask from Slice 2a, exact set/count match) AND
  **per-channel radius stats** present and correct (base r=3, metallic/roughness/alpha
  r=1) AND **legacy dilation/BFS/gutter fill is disabled/no-op on this path** (their
  counters report 0) AND oracle-parity anchors present for the pinned cv2 version
  (checked via stats carried through diagnostics, not hardcoded).
- Anti-gaming tests (Codex P1-6): feeding legacy-path stats, mislabeled modes,
  fallback-dominated stats, a mismatched painted-vs-raw mask, nonzero legacy-fill
  counters, or absent/wrong-version anchors flips each gate back to
  `heuristic_quarantined`.
- `heuristic_quarantined` stage count drops by exactly 2 on the reference path;
  no other stage verdict changes.
**Verification:** `cd packages/mlx-spatialkit && .venv/bin/python -m pytest tests/test_real_pixal3d_export.py -k "stage_gate or parity_gate" -q && .venv/bin/python -m pytest tests/test_glb_writer.py -q`
**Touches:** `src/mlx_spatialkit/export.py`, tests.

### Slice 6: 4096 e2e + budgets

**Objective:** `export_pixal3d_glb(texture_size=4096)` on the reference path
succeeds on both fixtures within measured, pinned budgets, producing GLBs for human
inspection. The *new* proof here is **4096 specifically with Telea + GPU trilinear on the
reference path within budgets** (Codex P2-8) — not a bare guard unblock (4096 already has
baseline heavy coverage); the budgets and the reference-path combination are the
deliverable.
**Acceptance criteria:**
- Both fixtures export at 4096 with reference path (telea + GPU trilinear + QEM +
  native unwrap); upstream-parity `texture_size` check passes; guard wiring needs no
  caller workaround.
- Wall-time and delta-RSS measured first, then pinned as budgets in the heavy test
  (delta-RSS, not absolute; generous-but-real margins like the unwrap budgets).
- Telea at 4096 within budget — apply DESIGN D3 band-limited FMM only if measurement
  demands it (record decision either way).
- GLBs written under /tmp for inspection; preview-vs-4096 visual delta noted in
  evidence for the checkpoint.
**Verification:** `cd packages/mlx-spatialkit && .venv/bin/python -m pytest -m heavy tests/test_real_pixal3d_export.py -k texture_4096 -q`
**Checkpoint after:** human-verify
**Checkpoint reason:** User inspects the 4096 GLBs — this change is expected to fix
the visual residuals they flagged at the unwrap checkpoint; confirm before wrap.
**Touches:** `tests/test_real_pixal3d_export.py`, possibly `cpp/inpaint.cpp`
(band-limit), `src/mlx_spatialkit/export.py` (only if guard wiring needs it).

### Slice 7: Determinism + dependency + regression wrap

**Objective:** Cross-process determinism for the new paths, dependency-light proof,
and full-suite regression close-out.
**Acceptance criteria:**
- Cross-process e2e determinism: sha256 of base_color + metallic_roughness textures
  (reference path, telea + GPU trilinear) identical across PYTHONHASHSEED 0/1
  subprocesses on one fixture.
- Dependency diff: build/dependency files show only new source files; no new
  required deps; suite green with no cv2/torch importable in the project venv.
- Clean rebuild (`rm -rf /tmp/mlx-spatialkit-build`) then full non-heavy suite and
  full heavy suite pass; counts and durations recorded in evidence.
- PLAN evidence blocks complete for all slices; ready for auto-verify.
**Verification:** `cd packages/mlx-spatialkit && .venv/bin/python -m pytest -q -m "not heavy" && .venv/bin/python -m pytest -q -m heavy tests/test_real_pixal3d_export.py`
**Touches:** tests, PLAN evidence only.
