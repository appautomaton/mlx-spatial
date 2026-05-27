# mlx-spatialkit Native Geometry Backend Tier Design

## Problem

The reference-target export now passes the measured face-count, topology, final
coverage, and raw-coverage checks. It still reports:

```text
backend: spatial-cluster
quality_tier: geometry_aware_preview
production_quality_ready: false
```

The next backend cannot be a rename. It needs a distinct native algorithm,
backend routing, and tests that would fail against the current spatial-cluster
implementation.

## Backend Contract

Extend the simplifier boundary from one implicit backend to an explicit backend
selection:

```text
preview/default export
  -> spatial-cluster
  -> quality_tier = geometry_aware_preview

reference-target export
  -> topology-aware native backend
  -> quality_tier = production only when backend self-checks pass
```

The Python layer remains orchestration only. It selects the backend from the
quality preset, passes the backend name to native code, and records diagnostics.

## Native Backend Shape

The first production candidate should be a native topology-aware simplifier,
implemented in C++ behind the same `simplify_mesh` result shape:

```text
cleaned mesh
  -> native topology-aware simplifier
     -> bounded candidate generation
     -> local collapse / merge validation
     -> degenerate + duplicate rejection
     -> nonmanifold guard
     -> compact mesh
  -> mesh_metrics
  -> paired atlas + Metal bake
  -> threshold gate
```

Implementation may reuse existing mesh helpers and may use spatial partitioning
as an acceleration structure, but it must not return the current
`spatial-cluster` backend under a new name. Diagnostics must expose each stage
used by a hybrid path.

## Quality Gate

`_production_thresholds()` should remain the readiness gate. The expected pass
path is:

```text
reference_available       true
quality_preset            reference-target
backend_tier              production
topology_exportability    no blockers
face_count_ratio          0.80..1.25
final_coverage_ratio      >= 0.50
raw_coverage_ratio        reported
```

If the new backend cannot justify `quality_tier=production`, diagnostics should
say why and `production_quality_ready` must remain false.

## Diagnostics

The simplifier stats should include:

- `backend`
- `algorithm`
- `quality_tier`
- `production_ready`
- `production_blockers`
- `source_faces`, `source_vertices`
- `target_faces`
- `final_faces`, `final_vertices`
- removed degenerate / duplicate / nonmanifold face counts
- target-reached and runtime-relevant counts for candidate/collapse work

Memory samples remain in the export diagnostics around existing stage labels,
especially `after_simplify_mesh` and `after_write_glb`.

## Readiness Boundary

This change can close the current backend-tier blocker only if the real fixture
passes the existing production thresholds. If it does, the active change can be
verified while the broader thread goal remains open for visual parity, higher
resolution texture/export settings, and future unwrap/bake improvements.
