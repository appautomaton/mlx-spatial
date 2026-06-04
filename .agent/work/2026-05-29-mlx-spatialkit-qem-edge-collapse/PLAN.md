# PLAN — mlx-spatialkit QEM Edge-Collapse Simplification (rev. post eng-review)

**Goal:** Add a native, dependency-light QEM edge-collapse simplifier to `mlx-spatialkit` that decimates a watertight narrow-band remesh to a target face count while preserving closed manifold topology — replacing clustering's re-tearing — proven watertight-decimated on two cached fixtures. Spec: `SPEC.md` (QEM-01..10). Mechanism: `DESIGN.md`. Review corrections: `reviews/engineering-findings.md` (round 1: B1–B4, R1–R8) and `reviews/engineering-findings-r2.md` (round 2: M1–M4, F1–F3).

**Branch:** `mlx-spatialkit-qem-edge-collapse`. **Reference:** CuMesh `simplify`/`simplify_step` (behavior-only; `/tmp/CuMesh` gone, vendored `o-voxel` + SPEC carry the contract).

**Build/test:** scikit-build-core + nanobind, venv at `packages/mlx-spatialkit/.venv`. Native edits require an extension rebuild before pytest. Heavy fixture tests are opt-in (`-m heavy`). All proofs run **offline from cached NPZ** — no MLX inference re-run.

**What the reviews changed vs the original plan:** Round 1 added a **vertex-manifold metric** (B1), an **explicit opt-in param not the preset** (B2), the **full guard set + ordered-container determinism + heap comparator** (B1/B3/R6/R7), the **sentinel stat contract** (B4), **numeric budgets**, machine-asserted **contrast/target_reached** (R1–R5), and **cross-process determinism** (S6). Round 2 then fixed the regressions those introduced: **M1** the `nonmanifold_vertices` metric is **report-only, NOT in the status gate** (would flip clustering fixtures); **M2** S2 **forks its own qem stat-return path** (incl. the early-return at `simplify.cpp:1714`) so it can't crash on an unpopulated `ClusterResult`; **M3** `simplify_backend="qem"` **requires remesh+repair** (or auto-enables it) so it can't silently under-decimate; **M4** S3 asserts **keyset-equality** of qem vs clustering stats, not an enumerated subset.

## Ordered slice sequence
S1 → S2 → S3 → S4 → S5 (human-verify) → S6. Continue through all approved slices; windows are context batches, not stops.

### Slice 1: Vertex-manifold metric (`nonmanifold_vertices`)
Required:
**Objective:** Add a `nonmanifold_vertices` (pinch) counter to `cpp/mesh_metrics.cpp` so the dominant edge-collapse failure mode — a non-manifold *vertex* invisible to `nonmanifold_edges` — becomes detectable (B1).
**Acceptance criteria:**
- `mesh_metrics(...)` returns `nonmanifold_vertices`: number of vertices whose incident faces form >1 edge-connected fan; 0 on a clean closed manifold.
- A synthetic **pinched** mesh (two cones sharing one apex vertex) reports `nonmanifold_vertices >= 1` while `nonmanifold_edges == 0` (proves it catches what edges miss).
- **M1 — report-only, NOT gated:** `nonmanifold_vertices` is surfaced as a metric **count/class** and auto-flows into `export_metrics` via `post_metrics=mesh_metrics(simplified)` (`export.py:486`); the status-determining gate at `export.py:1445` (which branches on `nonmanifold_edges>0`) is **left unchanged** — do NOT add `nonmanifold_vertices` to it, or existing clustering fixtures (`test_real_pixal3d_export.py:292/:539`) flip `rendered_visual_blocked`→`artifact_blocked`.
**Verification:** rebuild, then `cd packages/mlx-spatialkit && .venv/bin/pytest tests/test_mesh_processing.py -k "metric or manifold"` (pinched-mesh case + clean-mesh==0) AND `.venv/bin/pytest -m heavy tests/test_real_pixal3d_export.py -k "276 or 292 or 539" 2>/dev/null || .venv/bin/pytest tests/test_mesh_processing.py -k metric` — confirm the two heavy clustering-status fixtures still assert `rendered_visual_blocked` (gate unchanged); existing mesh_metrics tests stay green.
**Touches:** `cpp/mesh_metrics.cpp` (new counter + result key), `tests/test_mesh_processing.py`. **Do NOT touch** the `export.py:1445` status gate (M1).
**Produces:** native pinch detector used as the correctness oracle for S2/S5

**Status:** complete
**Evidence:** added `nonmanifold_vertices` (union-find fan-count via existing `mesh_common::UnionFind`, deterministic) to `cpp/mesh_metrics.cpp:156-251`; two new tests in `test_mesh_processing.py` (pinch = 2 tetrahedra sharing one apex → `nonmanifold_vertices==1` while `nonmanifold_edges==0`/`boundary_loop_count==0`; clean octahedron → 0). `pytest tests/test_mesh_processing.py -q` = 25 passed. `export.py` unchanged (M1 ✓). Rebuild: `uv pip install --no-build-isolation -e .` (editable hook has rebuild=False).
**Risks / next:** none — oracle ready for S2.

### Slice 2: Core QEM edge-collapse engine (callable + correct on adversarial synthetics)
Required:
**Objective:** Implement the QEM edge-collapse engine in `cpp/simplify.cpp` routed via `backend="qem"`, with the full DESIGN guard set, ordered-container determinism, and **its own self-contained stat-return path**, reducing faces toward a target while keeping the surface closed + manifold.
**Acceptance criteria:**
- `simplify_mesh(backend="qem", target_faces=…)` reduces face count toward target via quadric-error-ordered collapses; `resolve_backend` accepts `"qem"` and the throw message lists it (R8, QEM-01).
- **M2 — self-contained return:** the qem path **forks its own stat emission** (does NOT route through the shared block at `simplify.cpp:1820-1862`, which reads an unpopulated clustering `ClusterResult best`), and likewise returns correctly via the **backend-agnostic early-return at `:1714`** when input ≤ target — so S2's own tests pass before S3 lands. S2 emits at minimum the keys its tests read; S3 widens to full keyset-equality.
- On closed inputs **including adversarial cases** — icosphere, **thin double-sided sheet**, **high-valence hub**, **valence-3 / single-tetrahedron** — output keeps `boundary_loop_count`/`*_open_chain_count`/`nonmanifold_edges`/**`nonmanifold_vertices`** at 0 (B1/R7, QEM-02).
- Collapse order is quadric-error-driven; geometric-error summary recorded (QEM-03 mechanism). Normal-flip uses `cos(theta_min)` threshold; degeneracy is a hard reject (R6). Heap **compaction** rebuilds at >3× live-edge count (R3 mechanism).
- Determinism: adjacency/re-push containers are sorted-vector/`std::map`; heap comparator `(cost ASC, edge_id ASC, version DESC)`; same input+target → identical output on a second call (QEM-06, B3 in-process).
**Verification:** rebuild, then `.venv/bin/pytest tests/test_mesh_processing.py -k qem` — new tests assert reduced faces + all four topology counters == 0 on the four synthetics + equal arrays on a repeat call + a `backend="qem"` input-already-≤-target case returns cleanly (early-return path).
**Execution:** subagent recommended
**Depends on:** S1
**Touches:** `cpp/simplify.cpp` (new engine + forked qem return path + early-return `:1714` + `resolve_backend` `:1523/1540` + `BackendSelection` `:128`), `cpp/bindings.cpp` (accept `"qem"`, default unchanged), `tests/test_mesh_processing.py`
**Detail:** `DESIGN.md`

### Slice 3: Stat-field contract, diagnostics & blocker wiring
Required:
**Objective:** Widen the qem stat dict to the full clustering keyset (legacy fields as sentinels + new QEM fields) and clear only its own blocker — without disturbing clustering diagnostics (B4/M4, QEM-04).
**Acceptance criteria:**
- **M4 — keyset-equality (the proof, not an enumeration):** on the same input, `set(qem_stats.keys()) == set(spatial_cluster_stats.keys())` — every clustering-specific field present as a `0`/`"n/a"` sentinel (covers the ~30 `add_small_loop_fill_stats`/`add_pre_simplify_loop_fill_stats` fields the DESIGN enumeration omits); new `qem_*` fields populated; `qem_simplification_backend=="native-qem-edge-collapse"`, `qem_equivalence_status=="edge-collapse"`.
- `production_blocker_values` for `qem` pushes only `missing_narrow_band_dc_remesh`, never `missing_qem_edge_collapse_simplification`; with remesh cleared, `quality_tier=="production"` and blockers empty. The qem blocker contract also holds on the **early-return path** (`:1714`) — assert a qem input-already-≤-target case (NR7).
- `topology-aware`/`spatial-cluster` still emit `qem_simplification_backend=="not_implemented"` + their blockers (regression, QEM-09).
**Verification:** rebuild, then `.venv/bin/pytest tests/test_mesh_processing.py -k "qem or topology_aware"` (incl. the keyset-equality assertion + early-return case) AND `.venv/bin/pytest tests/test_real_pixal3d_export.py -k "quality_summary or topology_blocker_map"` (non-heavy unit tests that hard-assert blocker strings) — qem clears its blocker, clustering strings unchanged.
**Depends on:** S2
**Touches:** `cpp/simplify.cpp` (`production_blocker_values` `:1543`, `add_backend_stats` `:1571`, stat block `:1820-1862`), `src/mlx_spatialkit/export.py` (confirm reads `:1223-1226`, `:1420-1452`), `tests/`

### Slice 4: Watertight input-prep + explicit opt-in wiring (not the preset)
Required:
**Objective:** Compose `remesh(repair_nonmanifold=True)` + bounded `small_boundary_loop_fill` to feed QEM a watertight input, and expose QEM via a new validated `simplify_backend` opt-in param that **requires** the watertight input-prep — leaving `_simplifier_backend_for_quality_preset` and its tests untouched (B2/M3/F1, QEM-10).
**Acceptance criteria:**
- New `export_pixal3d_glb(simplify_backend: str | None = None)`; `"qem"` overrides `requested_simplifier_backend` (`export.py:302/463`); default `None` reproduces current preset behavior byte-for-byte.
- **F1 — validated param:** a Python-side resolver accepts only `None`|`"qem"` and raises a clear `ValueError` on anything else (mirrors `_resolve_pixal3d_uv_backend`); asserted in a non-heavy unit test.
- **M3 — watertight-input guard (zero silent failures):** `simplify_backend="qem"` **requires `remesh=True` + `remesh_repair_nonmanifold=True`** — raise `ValueError` if not set (or auto-enable input-prep; pick one and assert it). Prevents the boundary-lock stall that yields a vacuously-"watertight" under-decimated mesh.
- From cached NPZ (with remesh on), pre-QEM mesh has `nonmanifold_edges==0` and `boundary_loop_count` within the recorded bound (target 0); residual loops recorded (R4).
- `_simplifier_backend_for_quality_preset` unchanged; its tests (`test_real_pixal3d_export.py:580/615-617/988/1079/1210`) still green (QEM-09).
**Verification:** `cd packages/mlx-spatialkit && .venv/bin/pytest tests/test_real_pixal3d_export.py -k "preset or simplify_backend"` (non-heavy: F1 validator rejects bad strings, M3 rejects qem-without-remesh, preset regression green) AND `.venv/bin/pytest -m heavy tests/test_real_pixal3d_export.py -k "qem and input_prep"` (pre-QEM watertight metrics + `simplify_backend="qem"` selected).
**Execution:** subagent recommended
**Depends on:** S2, S3
**Touches:** `src/mlx_spatialkit/export.py` (new validated param + M3 guard + override at `:302/463`, pipeline wiring), `cpp/simplify.cpp` (loop-fill ordering for qem input)

### Slice 5: Two-fixture watertight-decimated proof + numeric budgets + clustering contrast
Required:
**Objective:** Prove on both cached fixtures that `remesh → input-prep → QEM` yields a reduced AND watertight mesh, machine-verified better than clustering, within recorded numeric runtime/memory budgets, with the target actually reached.
**Acceptance criteria:**
- Main (turtle/city) AND violin-bow: post-QEM `mesh_metrics` watertight within bound (all four counters incl. `nonmanifold_vertices`) at the target (QEM-05).
- **Contrast machine-asserted (R5):** same input through `backend="spatial-cluster"` yields `boundary_loop_count` strictly greater than qem's (the 0→41/52 regression made into an assertion).
- **`target_reached` asserted (R4):** distinguishes hit-target from boundary-lock stall.
- **Input size recorded (R1):** assert/record `simplify_stats['source_faces']` (QEM's real input = remesh output at res1024), distinct from the target.
- **Numeric budgets (R2/R3):** `diagnostics['timings_sec']['simplify_mesh'] < BUDGET_S` and `memory['stage_peaks']['simplify_mesh']['peak_current_rss_bytes'] < BUDGET_B` (`'simplify_mesh'` added to `required_stages`); budgets anchored from a first run; reference-scale (1M/212k/4096) recorded deferred if impractical.
- **F2 — pin the proof's scope:** record `(quality_preset, target_faces, target_faces_source)` the proof exercises; note explicitly that a **preview-target** proof is a legitimate tractable-scale proof but will **not** clear the reference-target production gate (`_pixal3d_reference_stage_contract` returns `not_requested` under preview) — so the proof's scope is unambiguous.
- Geometric-error bound vs pre-simplify surface recorded (QEM-03 fixture scale).
**Verification:** `cd packages/mlx-spatialkit && .venv/bin/pytest -m heavy tests/test_real_pixal3d_export.py -k qem -v` — both fixtures green on watertightness, contrast, target_reached, source_faces, and the numeric budget assertions.
**Depends on:** S2, S3, S4
**Checkpoint after:** human-verify
**Checkpoint reason:** User's goal is "proper GLB generation" quality — human inspects the decimated watertight GLB on both fixtures (topology + visual) before the watertight-decimated outcome is accepted as proven.
**Touches:** `tests/test_real_pixal3d_export.py` (new heavy two-fixture tests), `src/mlx_spatialkit/export.py` (diagnostics: counts + quadric-error summary, QEM-10)

### Slice 6: Regression, cross-process determinism & dependency-light wrap
Required:
**Objective:** Confirm the change is additive and clean: cross-process determinism holds, no new required deps, all existing suites and public knobs stay green.
**Acceptance criteria:**
- **Cross-process determinism (B3):** `simplify_mesh(backend="qem",…)` produces byte-identical `vertices.tobytes()+faces.tobytes()` across two **separate** processes (and/or two `PYTHONHASHSEED` values) — catches ASLR-seeded hash nondeterminism the in-process S2 check cannot (QEM-06).
- No new required MLX/Torch/CUDA/xatlas dependency; native build + `import mlx_spatialkit` succeed (QEM-08).
- Full existing suites green: `test_mesh_processing`, `test_remesh`, `test_glb_writer`, `test_contracts`, and non-heavy + heavy `test_real_pixal3d_export`; clustering backends/knobs/diagnostics intact (QEM-09).
- **F3 (optional, R3 mechanism):** if cheaply expressible, one assertion that heap compaction actually fires (a stat counter `qem_heap_compactions >= 1` on a large-enough synthetic) — otherwise R3 stands as an accepted deferred-scale risk (reference-scale 1M not proven here).
**Verification:** `cd packages/mlx-spatialkit && .venv/bin/pytest tests/` (non-heavy full suite) green; new cross-process determinism test green; `.venv/bin/pytest -m heavy tests/test_real_pixal3d_export.py -k qem` green; `git diff --stat pyproject.toml CMakeLists.txt` shows no new required deps.
**Depends on:** S2, S3, S4, S5

## Execution routing and topology
- **Default path:** continuation S1→S2→S3→S4→S5. One **human-verify** checkpoint after S5; S6 resumes after.
- **Subagent routes:** S2 (algorithm-heavy) and S4 (cross-subsystem export+cpp) `subagent recommended`. S1, S3, S5, S6 `direct`.
- **Parallel-safe groups:** none — strict dependency chain; overlapping write sets on `simplify.cpp`/`export.py` (S1 on `mesh_metrics.cpp` precedes S2's proof).

## Requirement traceability
| ID | Slice(s) |
|---|---|
| QEM-01 backend exists/callable | S2 |
| QEM-02 topology-preserving | S1 (metric), S2 (synthetics), S5 (fixtures) |
| QEM-03 quadric-error fidelity | S2 (mechanism), S5 (fixture bound) |
| QEM-04 blocker clears honestly | S3 |
| QEM-05 two-fixture watertight-decimated | S5 |
| QEM-06 determinism | S2 (in-process), S6 (cross-process) |
| QEM-07 runtime/memory bounded | S5 (numeric budgets), S6 |
| QEM-08 dependency-light preserved | S6 |
| QEM-09 regression contracts preserved | S3, S4, S6 |
| QEM-10 pipeline integration + diagnostics | S4, S5 |
| Review B1 pinch guard + metric | S1, S2 |
| Review B2 opt-in not preset | S4 |
| Review B3 ordered-container + comparator + cross-proc | S2, S6 |
| Review B4 stat-field/blocker contract | S3 |
| Review R1 input≠target (source_faces) | S5 |
| Review R2/R3 numeric budget + heap compaction | S2 (compaction), S5 (budget) |
| Review R4 target_reached stall detection | S5 |
| Review R5 clustering contrast asserted | S5 |
| Review R6/R7 normal-flip threshold + low-valence guards | S2 |
| Review R8 resolve_backend arm | S2 |
| Review M1 nonmanifold_vertices report-only (not gated) | S1 |
| Review M2 fork own qem stat-return (+ early-return :1714) | S2 |
| Review M3 qem requires remesh+repair guard | S4 |
| Review M4 stat keyset-equality | S3 |
| Review F1 simplify_backend param validator | S4 |
| Review F2 pin proof preset/target | S5 |
| Review F3 heap-compaction-fires (optional) | S6 |

## Architecture approach
New pattern (QEM edge-collapse) — see `DESIGN.md`. Reuses existing quadric infra and the sorted adjacency pattern; additive new region in `simplify.cpp`, no change to the `simplify_mesh` signature (`mesh_processing.hpp:15-22`). The only non-additive surface is one new optional `export_pixal3d_glb` param (default-None preserves behavior).

## Review: Engineering

- Verdict: approved_with_risks
- Strength: All four round-2 must-fixes (M1–M4) and three followups (F1–F3) are genuinely closed with fail-if-absent verifications grounded in source, the B1–B4/R1–R8 closures are intact in the traceability table, and a third-order regression hunt found nothing — on a purely additive, cleanly-revertible architecture (new `simplify.cpp` region + one default-`None` opt-in param, no signature change).
- Concern: The proof is scoped to a tractable preview target, so reference-scale parity (1M-vertex / 212k-face / 4096-texture) is explicitly NOT proven by this change, and the R3 heap-compaction mechanism is verified only at the symptom level (bounded peak RSS), not that compaction actually fires.
- Action: Proceed to `auto-execute` Slice 1 (vertex-manifold metric) and continue the S1→S6 chain to the S5 human-verify checkpoint; carry the deferred reference-scale caveat into `auto-verify`.
- Verified: M1–M4/F1–F3 re-checked against `export.py` (:1445/:434/:237/:1091/:1189), `simplify.cpp` (:1844-1860/:1714/:1540), `mesh_metrics.cpp`, and the `:292/:539/:852` test asserts; B/R traceability intact; 6-slice dependency order sound; one cosmetic DESIGN line-range (1820-1862 vs best.* at 1844-1860) noted, non-blocking.
