# mlx-spatialkit Native GLB Core Design

## Package Boundary

`mlx-spatialkit` is a nested Python distribution under `packages/mlx-spatialkit`.
Its import package is `mlx_spatialkit`. It does not depend on `mlx`, Torch, or
`mlx-spatial`; it accepts NumPy arrays, Python buffers, and file paths.

`mlx-spatial` integrates with it as an optional native export backend. The
existing Python export path is reference behavior during implementation and may
remain as fallback only if the plan slice explicitly preserves that behavior.

## Build Boundary

Use `scikit-build-core` + `nanobind` with CMake:

- C++20 for mesh contracts, metrics, extraction, cleanup, simplification, and
  GLB writing.
- Objective-C++ `.mm` bridge for Metal 4 device/pipeline setup.
- `.metal` kernels for texture raster/sampling work.
- Build-time Metal probe that fails clearly when the toolchain is unavailable.
- CMake should declare `project(... LANGUAGES CXX OBJCXX)`, locate Python with
  module development support, require `nanobind`, add the extension with
  `nanobind_add_module(...)`, and install it into `mlx_spatialkit`.
- Slice 1 must compile a tiny `.mm` bridge and `.metal` kernel so Metal support
  is proven by compilation, not only by `xcrun metal -v`.

Build outputs must stay under the package wheel or `/tmp` build directories.
Generated GLBs, diagnostics, and benchmark files go under `/tmp`.

## Public Python API

The Python layer is intentionally thin:

```python
from mlx_spatialkit.export import export_pixal3d_glb
from mlx_spatialkit.mesh import extract_flexi_dual_grid, mesh_metrics
from mlx_spatialkit.texture import bake_pbr_texture
```

The API validates paths/array metadata, calls native bindings, and returns small
result dataclasses or dictionaries. It does not loop over faces, vertices,
voxels, or texels.

## Native Data Contracts

Pixal3D decoded contracts:

- `shape_coordinates`: `Nx4 int32`, batch index in column 0.
- `shape_fields`: `Nx7 float32`, FlexiDualGrid fields.
- `texture_coordinates`: `Nx4 int32`.
- `texture_attributes`: `Nx6 float32`, base color + metallic + roughness +
  alpha layout.

Native entry points validate dtype, shape, contiguity/copy needs, non-empty
inputs, overflow-prone counts, and batch support before allocating large buffers.

## Export Pipeline

```text
NPZ loader / Python API
        |
        v
C++ FlexiDualGrid extraction
        |
        v
C++ metrics + cleanup/simplification
        |
        v
Native UV interface
        |
        v
Metal 4 PBR texture bake
        |
        v
C++ GLB writer
```

The UV interface is replaceable. First implementation may wrap C++ `xatlas` if
that gets to working GLB output faster, but `xatlas` must not leak into Python
hot loops.

## Memory And Threading Rules

- Use RAII for native buffers and Metal resources.
- Prefer job-local state; no mutable process-global mesh or texture buffers.
- Any compiled Metal pipeline/device cache must be immutable after creation or
  guarded by `std::call_once` / mutex.
- Release large intermediate buffers at stage boundaries.
- Report peak-ish process memory during heavy fixture checks when practical.
- Return deterministic errors instead of partial files when validation fails.

## Verification Strategy

Use two fixture tiers:

- Small synthetic fixtures committed under package tests for default checks.
- Real Pixal3D decoded fixture under ignored `inputs/mlx-spatialkit`, writing
  heavy outputs to `/tmp`.

Default tests prove contracts, build import, synthetic mesh parity against the
existing Python `ovoxel.py` behavior, metrics, GLB validity, and package
cleanliness. Heavy tests prove the real decoded Pixal3D fixture can export a GLB
and emit diagnostics.
