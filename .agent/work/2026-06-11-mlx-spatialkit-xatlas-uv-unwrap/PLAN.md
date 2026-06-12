# Plan: Reference-Parity UV Unwrap

**Goal:** SPEC at `.agent/work/2026-06-11-mlx-spatialkit-xatlas-uv-unwrap/SPEC.md` (UVU-01..11); architecture in `DESIGN.md`.

All commands run from `packages/mlx-spatialkit/` with the project venv (`.venv/bin/pytest`); rebuild = `uv pip install --no-build-isolation -e .` (clean: `rm -rf /tmp/mlx-spatialkit-build` first).

### Slice 1: Native UV-quality metrics (oracle of record)

Required:
**Objective:** Land `cpp/uv_metrics.cpp` — UV-triangle overlap count, flipped-triangle count, per-chart Sander L2/Linf texel stretch, atlas utilization — exposed via bindings, report-only.
**Acceptance criteria:**
- Synthetic with known-overlapping UVs → `uv_overlap_count > 0`; clean grid → 0. Mirrored-triangle UVs → `uv_flipped_count > 0`.
- Analytic stretch case (uniform scale → L2 == scale; anisotropic squash → Linf reflects it, within float tolerance).
- Existing backends' stats and tests untouched; metrics callable standalone on `(vertices, faces, uvs, chart_ids)` (UVU-04 mechanism, UVU-11).
- Deterministic outputs (pure functions, ordered iteration).
**Verification:** rebuild, then `.venv/bin/pytest tests/test_glb_writer.py -k uv_metrics` (new tests) and `.venv/bin/pytest tests/test_glb_writer.py -q` green.
**Touches:** `cpp/uv_metrics.cpp|hpp`, `cpp/bindings.cpp`, `CMakeLists.txt`, `tests/test_glb_writer.py`

**Status:** complete (subagent route: implementer DONE → spec APPROVED → quality CHANGES_REQUESTED → fixes → quality APPROVED)
**Evidence:** new `cpp/uv_metrics.{cpp,hpp}` + one nanobind def (`uv_quality_metrics(vertices, faces, uvs, chart_ids=None)`) + CMake source + 8 non-heavy tests. All normative metrics verified line-by-line by spec review (Sander Ss/St/Gamma formulas exact; eps-tolerant SAT excludes edge-touching; deterministic ordered iteration). Quality round fixed 2 importants: rank-1 chart_ids heap-OOB (`strides[1]`) → local strides[0]-only reader; per-pair `std::set` dedup → O(F) last-anchor stamp array (+ kMaxCellSpanPerAxis=64 large-triangle path; dedup-equivalence test pins 15/15 pairs across 289 shared cells). `pytest tests/test_glb_writer.py -q` = **23 passed** (22 pre-existing untouched + new). Orchestration detail: `orchestration/slice-001-summary.md`.
**Risks / next:** three recorded minors (no strided-chart_ids regression test; >64-cell-span path untested by fixture; checked_pairs semantics differ for atlas-spanning triangles — flagged if a later slice pins checked_pairs on such fixtures).

### Slice 2: Stage-A cone-cluster engine

Required:
**Objective:** Implement `ConeClusterer` in `cpp/uv_unwrap.cpp` — heap-driven cost-ordered chart agglomeration with cone half-angle bound, area + perimeter/area penalties, refine/smooth semantics — deterministic, exposing per-face cluster ids.
**Acceptance criteria:**
- Cone invariant machine-checked on synthetics: no face normal deviates from its cluster cone axis beyond `threshold_cone_half_angle_rad` (UVU-02).
- Penalty semantics: raising `area_penalty_weight` strictly does-not-increase max cluster area on a fixed synthetic; perimeter/area weight penalizes strip-shaped merges (unit test with a long strip + compact blob).
- Production knobs (90°, refine 0, global 1, smooth 1, 0.1, 0.0001) run end-to-end; non-default refine/smooth exercised in unit tests only.
- Determinism: identical cluster ids on repeat call; ordered containers + named heap comparator (QEM pattern); no O(F²) (spot-check: subdiv icosphere scaling ~linear in a quick timing assert with generous bound).
**Verification:** rebuild, then `.venv/bin/pytest tests/test_mesh_processing.py -k cone_cluster`.
**Execution:** subagent recommended
**Depends on:** none
**Touches:** `cpp/uv_unwrap.cpp|hpp`, `cpp/bindings.cpp`, `CMakeLists.txt`, `tests/test_mesh_processing.py`

**Status:** complete (subagent route: implementer DONE → spec APPROVED → quality APPROVED)
**Evidence:** `ConeClusterer` in new `cpp/uv_unwrap.{cpp,hpp}` (~750 lines) + `compute_uv_charts` binding with CuMesh defaults. Spec review verified the cost composition against `atlas.cu:184-192` — the kernel uses **perimeter²/area** (docstring says perimeter/area; kernel wins, documented). Three documented deviations approved (monotone enclosing cones → cone invariant is a hard guarantee; refine cone-admission; guarded division). QEM determinism discipline verified structurally (no unordered containers in TU; `ChartEdgeCostGreater` cost↑/edge_id↑/version↓; lazy invalidation + 3× compaction). 9 new tests: cube 6-planar@10°, icosphere invariant @30°/90°, area-penalty monotonicity, strip-vs-blob perimeter penalty, production knobs e2e, refine/smooth variants, exact-equality determinism, 81920-face scale (0.186 s, budget 10 s). `pytest tests/test_mesh_processing.py -q` = **49 passed**; full suite 153 passed. Orchestration: `orchestration/slice-002-summary.md`.
**Risks / next:** **(important, fold into S4)** refine path at CuMesh-default knobs (refine 100 × global 3) is unbenchmarked at scale — per-face map lookups ×300 passes could be minutes at 1.1M faces; production knobs (refine 0, global 1) avoid it, but S4 must add fixpoint early-exit + neighbor-table precompute and extend the scale test to nonzero refine. Minors recorded: round-rebuild allocation churn at 1.1M (flat adjacency if stage B needs it); rejection stats count re-rejections (not unique pairs); no threshold upper-bound validation; zero-area faces unfiltered (upstream QEM strips them); two libm-sensitive test pins (macOS-deterministic).

### Slice 3: Oracle anchors (dev-time pip xatlas)

Required:
**Objective:** Generate and commit version-pinned parity anchors: both fixtures' QEM 50k meshes → native stage-A clusters → pip xatlas per cluster (reference composition) + whole-mesh sanity run, recording chart count, utilization, stretch distribution (via slice-1 metrics), seam ratio into `tests/data/uv_oracle_anchors.json`.
**Acceptance criteria:**
- `/tmp/uvoracle-venv` with pinned pip `xatlas`; generation script `tests/tools/gen_uv_oracle_anchors.py` is re-runnable and records `xatlas_version` + option mapping (SPEC assumption).
- Anchors JSON committed with both fixtures' numbers; values sane (chart_count > stage-A cluster count is allowed; utilization in (0,1]; stretch finite).
- All tests reading anchors pass **without** xatlas importable in the project venv (UVU-10); generation path skips cleanly when xatlas absent.
- If pip xatlas cannot install on this host: STOP and surface to user (SPEC assumption fallback is a user decision, not silent).
**Verification:** `.venv/bin/python -c "import xatlas"` FAILS in project venv while `ls tests/data/uv_oracle_anchors.json` exists and `.venv/bin/pytest tests/ -k anchors -q` green; anchors regenerated once from `/tmp/uvoracle-venv` with log retained under `/tmp`.
**Depends on:** S1, S2
**Touches:** `tests/tools/gen_uv_oracle_anchors.py`, `tests/data/uv_oracle_anchors.json`, `tests/test_glb_writer.py`

### Slice 4: Stage-B chart growth + projection baseline

Required:
**Objective:** Implement `ChartBuilder` (xatlas ChartOptions cost semantics within each cluster) plus the PCA/orthographic parameterization path with flip/stretch acceptance test, producing per-chart UVs for planar-ish charts.
**Acceptance criteria:**
- Growth cost implements normal-deviation/roundness/straightness/normal-seam/texture-seam weights with `max_cost` stop and `max_iterations=1` reseed semantics; weights are knobs defaulted to reference values (UVU-03 mechanism).
- On fixture meshes (cached QEM 50k from S3 setup), chart count within the stated tolerance factor of the per-cluster oracle anchors (tolerance recorded in test, target ≤1.5×; tighten in S8 evidence if measured tighter).
- Projection acceptance: charts accepted only when `uv_flipped_count==0` and Linf stretch ≤ recorded threshold; rejected charts marked for S5 (LSCM) — count recorded in stats.
- Deterministic chart ids/UVs across repeat calls.
**Verification:** rebuild, then `.venv/bin/pytest tests/test_mesh_processing.py -k chart_growth` and `.venv/bin/pytest -m heavy tests/test_real_pixal3d_export.py -k chart_count_parity`.
**Execution:** subagent recommended
**Depends on:** S2, S3
**Touches:** `cpp/uv_unwrap.cpp`, `tests/`

### Slice 5: LSCM parameterization + zero-overlap repair (hard core)

Required:
**Objective:** Native LSCM (sparse conformal system, two pinned boundary vertices, CG solver) for charts that fail projection, plus overlap repair by bounded chart split + re-parameterize, establishing the zero-overlap invariant.
**Acceptance criteria:**
- LSCM on analytic curved charts (hemisphere cap, cylinder segment): `uv_flipped_count==0`, L2 stretch strictly better than projection on the same chart, finite UVs (UVU-04).
- Zero-overlap invariant on both fixture meshes: `uv_overlap_count==0` post-repair, split depth bounded, repair counts in stats; failure = loud stat + test failure, never silent.
- Stretch parity: fixture stretch distribution within stated tolerance of oracle anchors (UVU-04).
- CG solver deterministic (fixed iteration order, capped iterations, no parallel reductions); no O(F²) — fixture-scale parameterization completes within a generous timing assert.
- Boundary charts (the 5–7 residual input loops) parameterize without overlap/distortion blowup; per-chart boundary attribution recorded (SPEC input condition).
**Verification:** rebuild, then `.venv/bin/pytest tests/test_mesh_processing.py -k lscm` and `.venv/bin/pytest -m heavy tests/test_real_pixal3d_export.py -k "overlap or stretch_parity"`.
**Execution:** subagent recommended
**Depends on:** S4
**Touches:** `cpp/uv_unwrap.cpp`, `tests/`

### Slice 6: Packing + backend assembly + opt-in wiring

Required:
**Objective:** Chart packing (hull-axis rotation, shelf/skyline, padding + bilinear gutters, resolution/texels-per-unit) and assembly of the full `make_reference_uvs` backend, exposed via widened `uv_backend` validator — default unchanged.
**Acceptance criteria:**
- New `uv_backend="xatlas-equivalent-native"` accepted (`export.py:1075-1079` widened; invalid strings still raise); default `"face-atlas"` byte-for-byte unchanged (UVU-01, UVU-11).
- Padding honored: no two charts within `padding` texels at proof resolution (machine-checked on the packed atlas); UVs in [0,1]; utilization within stated tolerance of oracle anchors (UVU-05).
- Full backend returns the `NativeUvMesh` contract (vertices/faces/uvs/vmap/stats); stats include chart/cluster counts, rejection/repair counters, all slice-1 metrics.
- Existing two backends' stats keysets unchanged (regression assert).
**Verification:** rebuild, then `.venv/bin/pytest tests/test_glb_writer.py -k "packing or reference_uvs"` and `.venv/bin/pytest tests/test_real_pixal3d_export.py -k "uv_backend" -q` (non-heavy validator + default-preservation tests).
**Depends on:** S5
**Touches:** `cpp/uv_unwrap.cpp`, `cpp/bindings.cpp`, `src/mlx_spatialkit/export.py`, `tests/`

### Slice 7: Honest gate flip

Required:
**Objective:** Make `xatlas_unwrap` stage and `_xatlas_chart_parity_summary` honest: `parity_ready` computed from measured values, `reference_matched` only for the new backend with proof passing, old backends still quarantined.
**Acceptance criteria:**
- `parity_ready` hardcoded `False` (`export.py:1730/1823`) replaced by computed verdict (overlap==0 AND anchor tolerances met AND backend is reference); consumed unchanged at `:2335-2348` (UVU-06).
- Stage contract: new backend → `reference_matched`; `face-atlas`/`native-chart` → `heuristic_quarantined` unchanged; `production_backend_blockers` semantics untouched (`:1461-1465`).
- Parity summary reports the real measured numbers (no vacuous defaults); contract unit tests hard-assert all three backend statuses.
- Gate cannot flip on a failing metric: targeted test feeds a deliberately-bad UV result and asserts `parity_ready` stays False (anti-gaming, SPEC anti-goal).
**Verification:** `.venv/bin/pytest tests/test_real_pixal3d_export.py -k "parity_summary or stage_contract or quality_summary" -q`.
**Depends on:** S6
**Touches:** `src/mlx_spatialkit/export.py`, `tests/test_real_pixal3d_export.py`

### Slice 8: Two-fixture end-to-end proof + numeric budgets

Required:
**Objective:** Prove on both fixtures: cached NPZ → remesh → QEM 50k → reference unwrap → bake → GLB under `/tmp`, with bake round-trip error bound, parity numbers, and runtime/delta-RSS budgets.
**Acceptance criteria:**
- Both fixtures: zero UV overlaps, stretch/utilization/chart-count within anchor tolerances, `parity_ready==True`, stage `reference_matched` (UVU-07).
- Bake round-trip error recorded and bounded (metric + bound stated in test); boundary-seam texels attributed in diagnostics (input-condition handling).
- Numeric budgets: `timings_sec` for the unwrap stage and **delta** peak RSS under anchored bounds (anchor 5× first observed, QEM precedent) (UVU-09).
- Proof scope pinned: preview/50k target recorded; reference-scale 1M/4096 explicitly deferred in evidence.
**Verification:** `.venv/bin/pytest -m heavy tests/test_real_pixal3d_export.py -k "reference_uv" -v` green on both fixtures; GLBs written under each fixture's `/tmp` output dir for inspection.
**Depends on:** S6, S7
**Checkpoint after:** human-verify
**Checkpoint reason:** UV/texture quality is the user's stated goal; metrics bound distortion but visual coherence of the unwrapped+baked GLB (seam placement, texture continuity) needs human inspection in a GLB viewer before the outcome is accepted.
**Touches:** `tests/test_real_pixal3d_export.py`, `src/mlx_spatialkit/export.py` (diagnostics only if gaps found)

### Slice 9: Regression, cross-process determinism & dependency-light wrap

Required:
**Objective:** Confirm the change is additive and clean across the full suite, deterministic cross-process, and dependency-light.
**Acceptance criteria:**
- Cross-process determinism: byte-identical `uvs/vertices/faces` across two processes with different `PYTHONHASHSEED` (UVU-08).
- No new required deps (`git diff pyproject.toml CMakeLists.txt`); build + import clean; full suite green **without** xatlas importable (UVU-10).
- Full suites green: `test_glb_writer`, `test_mesh_processing`, `test_remesh`, `test_contracts`, non-heavy + heavy `test_real_pixal3d_export`; existing backends/knobs/stats intact (UVU-11).
**Verification:** clean rebuild, then `.venv/bin/pytest tests/ -q` and `.venv/bin/pytest -m heavy tests/test_real_pixal3d_export.py -q`; `git diff --stat pyproject.toml CMakeLists.txt` empty of new required deps.
**Depends on:** S6, S7, S8

## Execution routing and topology

- **Default path:** continuation S1→S2→S3→S4→S5→S6→S7→S8; one **human-verify** checkpoint after S8; S9 resumes after.
- **Subagent routes:** S2 (algorithm-heavy), S4 and S5 (algorithm-heavy core) `subagent recommended`; rest `direct`.
- **Parallel-safe groups:** none — S1/S2 are logically independent but share write sets (`bindings.cpp`, `CMakeLists.txt`); strict serial order.
- **STOP conditions:** S3 oracle-install failure (user decision); any slice unable to hold the zero-overlap invariant without weakening it (spec change, not an engineering judgment).

## Requirement traceability

| ID | Slice(s) |
|---|---|
| UVU-01 backend callable | S6 |
| UVU-02 stage-A clustering parity | S2 |
| UVU-03 chart segmentation parity | S4 (mechanism + fixture factor) |
| UVU-04 overlap-free bounded-distortion | S1 (metrics), S5 (invariant + parity) |
| UVU-05 packing parity | S6 |
| UVU-06 honest gate flip | S7 |
| UVU-07 two-fixture e2e proof | S8 |
| UVU-08 determinism | S2/S4/S5 (in-process), S9 (cross-process) |
| UVU-09 runtime/memory bounded | S8 (budgets), S9 |
| UVU-10 dependency-light | S3 (anchors without xatlas), S9 |
| UVU-11 regression preserved | S1, S6, S9 |
| SPEC input condition (residual loops) | S5 (boundary charts), S8 (seam attribution) |
| SPEC assumption (oracle version pin) | S3 |

## Architecture approach

New pattern (two-stage unwrap + LSCM + oracle anchoring) — see `DESIGN.md`. Additive: two new cpp files, one widened validator, gate computation replacing hardcoded False; no existing backend or signature changes.
