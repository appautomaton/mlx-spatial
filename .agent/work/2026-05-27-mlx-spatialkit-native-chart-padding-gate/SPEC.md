# mlx-spatialkit Native Chart Padding Gate Spec

## Bounded Goal

Make Pixal3D native-chart exports use a backend-aware tighter default tile padding and prove the real chart candidate clears the UV-surface occupancy floor without changing face-atlas defaults.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant Metal/C++ hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

## Work Scale And Shape

- Scale: native chart quality/default hardening
- Shape: export contract correction

## Selected Lenses

- **engineering:** Align the export-level native-chart default with the native chart generator instead of applying face-atlas padding to chart UVs.
- **runtime:** Preserve explicit caller overrides and keep face-atlas behavior stable.
- **product:** Move the real native-chart candidate past one readiness blocker while keeping remaining blockers truthful.

## Current Evidence

- `make_native_chart_uvs` has a native default `tile_padding=0.04`, but `export_pixal3d_glb` passes its generic default `tile_padding=0.08` to both face-atlas and native-chart exports.
- A `/tmp` padding sweep on the real fixture showed native-chart occupancy improves from `0.38801002502441406` at `0.08` to `0.5065326690673828` at `0.02`.
- At `0.02`, `uv_surface_occupancy_floor` clears and the remaining native-chart blocker is `global_coverage_floor`.

## Required Outcome

1. `export_pixal3d_glb` resolves tile padding by backend: face-atlas default remains `0.08`, native-chart default becomes `0.02`, and explicit `tile_padding` overrides remain respected.
2. Diagnostics record the resolved tile padding and whether it came from backend default or explicit input.
3. Focused tests prove backend-specific default resolution and explicit override behavior.
4. The real Pixal3D native-chart export reports `uv_surface_occupancy_ratio > 0.50` and no longer lists `uv_surface_occupancy_floor` as a quality blocker.
5. Docs describe the tighter native-chart default as a native candidate quality setting, not xatlas parity or a default backend switch.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| NCPG-01 | Backend-aware tile padding defaults are explicit. | Unit tests assert face-atlas default `0.08`, native-chart default `0.02`, and explicit override passthrough. |
| NCPG-02 | Diagnostics expose the resolved contract. | Heavy fixture diagnostics include resolved `tile_padding` and `tile_padding_source`. |
| NCPG-03 | Native chart occupancy floor clears. | Heavy chart fixture reports `uv_surface_occupancy_ratio > 0.50` and no `uv_surface_occupancy_floor` blocker. |
| NCPG-04 | Face-atlas/default export behavior remains unchanged. | Full package/root Pixal3D tests still pass. |
| NCPG-05 | Docs match the parity boundary. | Docs say tighter padding improves the native chart candidate but does not claim xatlas parity or switch defaults. |

## Scope Coverage Decisions

- **Included:** tile-padding resolver, diagnostics, focused contract tests, real fixture proof, docs, regression/build verification.
- **Deferred:** solving global coverage, xatlas chart parity, overlap removal, default UV backend switch, threshold relaxation.
- **Anti-goals:** changing decoded model outputs, changing Metal bake kernels, changing chart quality floors, writing heavy artifacts outside `/tmp`, tagging/pushing/releasing.

## Constraints

- Preserve explicit caller-provided `tile_padding`.
- Keep face-atlas default at `0.08`.
- Keep generated and heavy artifacts under `/tmp`.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **Visual risk:** Tighter chart padding can increase bleed risk. Mitigation: use only for native-chart candidate and keep diagnostics truthful; explicit override remains available.
- **Contract risk:** Changing a default must be visible. Mitigation: record `tile_padding_source` and add resolver tests.
- **Regression risk:** Default argument change can affect callers. Mitigation: preserve face-atlas default and run package/root tests.

## Blocking Questions Or Assumptions

Assumption: `0.02` is a justified native-chart default because the real fixture clears the UV occupancy floor while remaining bounded and overrideable.
