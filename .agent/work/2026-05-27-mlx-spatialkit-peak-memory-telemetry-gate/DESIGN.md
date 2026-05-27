# mlx-spatialkit Peak Memory Telemetry Gate Design

## Current Gap

`export_pixal3d_glb` currently records memory snapshots like:

```text
start
after_load_npz
after_extract_mesh
after_texture_bake
after_write_glb
```

Those labels are useful but sparse. A stage can allocate and release large buffers between two labels, so diagnostics can look smaller than what a user sees in Activity Monitor.

## Monitor Shape

Add a private process memory monitor in `export.py`:

```text
_ProcessMemoryMonitor
  -> background polling thread
  -> process RSS sample provider
  -> active stage label guarded by a lock
  -> aggregate peak/current/high-water counters
  -> per-stage aggregate counters
```

The monitor keeps aggregates only:

- sample count
- observed peak `current_rss_bytes`
- observed peak `max_rss_bytes`
- peak label and stage
- per-stage start/end/peak values

It does not retain every sample.

## Stage Integration

Wrap each `_timed_stage` body:

```text
with memory_monitor.track_stage("texture_bake"):
    bake_pbr_texture(...)
```

The existing `memory_samples` dictionary remains label-based for compatibility. The new `diagnostics["memory"]` summary is the authoritative peak view.

## Thread Safety

- Use `threading.Lock` around shared counters and active-stage mutation.
- Use `threading.Event` to stop polling.
- Join the thread before writing diagnostics.
- Keep the monitor private to the export call.

## Boundary

This is host process RSS telemetry. It is not:

- full system memory pressure
- exact Activity Monitor app-memory accounting
- MLX allocator accounting
- Metal heap residency accounting

Those can be later cycles if needed.
