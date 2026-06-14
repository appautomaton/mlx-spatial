# mlx-spatialkit Production Remesh Parity Design

## Current Gap

The verified preview path is now exportable, but it is not production-parity:

```text
decoded Pixal3D NPZ
  -> native FlexiDualGrid extraction
  -> clean mesh
  -> spatial-cluster preview simplification
  -> face-atlas UVs
  -> Metal sparse-nearest texture bake
  -> GLB + diagnostics
```

The reference trace points to a stronger target:

```text
source faces:          8,304,022
reference final faces:   212,542
unwrap:                xatlas-parallel-spatial
bake:                  xatlas-kdtree
raw coverage:          ~0.413
final coverage:        1.0
```

A probe with the current native path at `target_faces=212542` gets close on
face count, but still fails production quality because the backend remains
preview-tier and global texture coverage is only about `0.269`.

## Direction

Use a production/reference-target preset plus explicit threshold checks before
trying to claim production readiness:

```text
quality_preset=reference-target
  -> load checked-in reference trace when available
  -> resolve target faces from reference final_faces
  -> run native export
  -> compare face-count, topology, coverage, backend tier, runtime/RSS
  -> production_quality_ready = all thresholds pass
```

Then improve the native hot path behind the same contract. This avoids moving
goalposts: every geometry or texture change must improve the same real-fixture
threshold report.

## Readiness Contract

`artifact_ready` remains a mesh/GLB exportability claim. `production_quality_ready`
is a stricter parity claim and requires:

- no export blockers
- non-preview simplifier tier
- final face count within the reference threshold
- final visible coverage threshold passes
- reference comparison fields are present

Failed thresholds are diagnostics, not hidden warnings.

## Deferred If Thresholds Fail

If the production preset remains below threshold after native geometry work, the
change should end with a precise blocker: for example UV utilization, bake
coverage, or remaining remesh backend quality. It should not relabel preview
output as production.
