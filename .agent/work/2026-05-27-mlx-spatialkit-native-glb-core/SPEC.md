# mlx-spatialkit Native GLB Core Spec

## Bounded Goal

Build `packages/mlx-spatialkit` as a native C++ / Objective-C++ / Metal 4 package that converts Pixal3D decoded `shape_decoder_fields.npz` + `texture_decoder_pbr.npz` artifacts into a textured `model.glb` for `mlx-spatial`.

## Broader Intent

`mlx-spatial` needs a performant native export backend for Apple Silicon 3D generation outputs. The first useful target is Pixal3D export after MLX model inference has already produced decoded shape and texture/PBR voxel fields.

## Work Scale and Shape

- Scale: capability
- Shape: mixed feature + refactor + parity + coverage

## Selected Lenses

- **product**: Makes generated Pixal3D scenes usable as standard GLB artifacts without requiring users to reason about internal NPZ tensors.
- **engineering**: Creates a package boundary where native code owns the expensive mesh/export stages instead of Python loops and Python-bound helper libraries.
- **runtime**: Requires explicit memory, thread-safety, Metal toolchain, and large-fixture boundaries.

## Target User or Stakeholder

Developers using `mlx-spatial` on Apple Silicon who need local 3D generation pipelines to emit previewable textured GLB files efficiently and cleanly.

## Evidence From Current Repo

- Current root package is `mlx-spatial` under `src/mlx_spatial`; there is no existing `packages/` package yet.
- Current Pixal3D export calls `flexi_dual_grid_fields_to_mesh`, `postprocess_trellis2_mesh_for_glb`, `bake_trellis2_texture_fields_mac_native`, then `write_pixal3d_textured_glb`.
- Current export code uses Python/NumPy plus `fast_simplification`, SciPy KD-tree texture sampling, Python `xatlas`, and Python GLB assembly.
- Existing local fixtures:
  - `inputs/mlx-spatialkit/pixal3d-1024-cascade-decoded-pbr/shape_decoder_fields.npz`
  - `inputs/mlx-spatialkit/pixal3d-1024-cascade-decoded-pbr/texture_decoder_pbr.npz`
  - `inputs/mlx-spatialkit/pixal3d-1024-cascade-glb-reference/model.glb`
- Current reference GLB trace records about 8.3M source faces, about 212k exported faces, `texture_size=1024`, `xatlas_face_guard=300000`, and `unwrap_backend=xatlas-parallel-spatial`.
- `xcrun metal -v` currently works on this machine.

## Required Outcome

The repo gains a separate nested package:

```text
packages/
  mlx-spatialkit/
    pyproject.toml
    CMakeLists.txt
    src/
      mlx_spatialkit/
        __init__.py
        mesh.py
        texture.py
        export.py
    cpp/
      bindings.cpp
      mesh_contracts.cpp
      mesh_metrics.cpp
      flexi_dual_grid.cpp
      mesh_cleanup.cpp
      simplify.cpp
      glb_writer.cpp
    metal/
      texture_bake.mm
      kernels/
        texture_bake.metal
    tests/
```

The package must provide a thin Python API over native implementation for:

1. Loading/validating Pixal3D decoded NPZ artifacts.
2. Converting FlexiDualGrid shape fields to a triangle mesh.
3. Reporting mesh metrics and failure diagnostics before expensive export work.
4. Cleaning and simplifying enough of the mesh to make first GLB export reliable.
5. Producing UV-ready geometry through a native-owned interface; initial implementation may wrap C++ `xatlas` if it is isolated behind an interface.
6. Baking texture/PBR voxel attributes to GLB texture images through Metal 4-backed raster/sampling work.
7. Writing a valid textured GLB.
8. Exposing an integration point so `mlx-spatial` can call `mlx_spatialkit` when available and keep the current Python path as a fallback only if planning approves that compatibility behavior.

## Native Boundary

Hot paths must be native-owned:

```text
Pixal3D decoded NPZ
  shape: coordinates[int32 Nx4] + fields[float32 Nx7]
  texture: coordinates[int32 Nx4] + attributes[float32 Nx6]
        |
        v
Python thin API
  validate paths / arrays, pass buffers to native
        |
        v
C++ mesh core
  FlexiDualGrid extraction -> mesh metrics -> cleanup/simplify
        |
        v
Native UV interface
  replaceable wrapper, initially allowed to use C++ xatlas
        |
        v
Metal 4 texture bake
  UV raster + voxel/PBR sampling + texture image buffers
        |
        v
C++ GLB writer
  textured model.glb + export diagnostics
```

Python may own argument parsing, file path setup, small metadata dictionaries, and test orchestration. Python must not own per-face, per-vertex, per-texel, or per-voxel hot loops.

## Constraints

- Use `scikit-build-core` and `nanobind` for Python packaging/bindings unless planning finds a stronger local reason to choose otherwise.
- `mlx-spatialkit` must not depend on MLX, Torch, or `mlx-spatial`.
- Runtime input contracts are NumPy arrays, Python buffers, or file paths; native code validates shape/dtype before use.
- Metal target is Metal 4; support below Metal 4 is not required.
- Native code must use RAII, explicit ownership, bounds-checked public entry points, and deterministic error returns/exceptions at the Python boundary.
- Native work must be safe to call from multiple Python threads as independent jobs; shared mutable global state is not allowed except immutable compiled-kernel/device caches guarded appropriately.
- Heavy generated outputs, benchmark dumps, and large temporary GLBs go under `/tmp`.
- `inputs/`, `outputs/`, `vendors/`, and `weights/` remain excluded from package distributions.
- The spec does not require byte-for-byte parity with the existing GLB, but requires valid GLB output and comparable diagnostic evidence from the same decoded Pixal3D fixture.

## Risks

- **Native build risk:** CMake, Metal, and Python packaging can fail before runtime logic is testable. Mitigation: first plan slice must establish a minimal native extension build and Metal toolchain probe.
- **Mesh scale risk:** real Pixal3D decoded output has millions of voxels/faces. Mitigation: unit tests use tiny synthetic fixtures; real fixture tests are opt-in/heavy and write to `/tmp`.
- **Simplification quality risk:** first implementation may not match upstream-quality remeshing. Mitigation: acceptance requires working GLB and diagnostics, not full `cumesh` parity.
- **UV bottleneck risk:** `xatlas` is CPU-only. Mitigation: keep it behind a native replaceable interface and accelerate the texture bake first, where Metal is clearly appropriate.
- **Memory risk:** Apple unified memory can exceed comfortable limits during decoded fixture export. Mitigation: stage allocations, release buffers promptly, and track peak RSS/footprint in heavy checks.
- **Thread-safety risk:** native caches and Metal objects can introduce hidden shared state. Mitigation: define job-local state and guarded immutable caches before implementation.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| SK-01 | Package boundary exists | `packages/mlx-spatialkit` contains a buildable nested Python distribution named `mlx-spatialkit` with import package `mlx_spatialkit`. |
| SK-02 | Native extension builds | A minimal C++/nanobind extension builds through `scikit-build-core` on macOS arm64, and the build fails clearly if the Metal 4 toolchain is unavailable. |
| SK-03 | Runtime deps stay clean | `mlx-spatialkit` does not depend on MLX, Torch, or `mlx-spatial`; root `mlx-spatial` does not gain native dev-only deps as runtime requirements. |
| SK-04 | Contracts are explicit | Public APIs validate Pixal3D NPZ path/array contracts: shape coordinates `Nx4 int32`, shape fields `Nx7 float32`, texture coordinates `Nx4 int32`, texture attrs `Nx6 float32`. |
| SK-05 | Mesh extraction is native | FlexiDualGrid extraction for small synthetic fixtures is implemented in C++ and matches the existing Python `ovoxel.py` behavior on deterministic tests. |
| SK-06 | Diagnostics are actionable | Native mesh metrics report face/vertex counts, degenerate faces, duplicate faces, boundary edges, nonmanifold edges, component counts, and export-blocking reasons. |
| SK-07 | Cleanup/simplification is native-owned | Mesh cleanup/simplification runs through `mlx_spatialkit` native APIs and can reduce a fixture mesh enough for downstream UV/export without Python per-face loops. |
| SK-08 | Texture bake hot loop is Metal-backed | Texture baking uses a Metal 4 path for raster/sampling work and returns base-color/PBR texture buffers with coverage diagnostics. |
| SK-09 | GLB writer works | `mlx_spatialkit` writes a valid textured GLB from native-produced mesh/texture buffers; validation checks the file header/chunks and loads basic scene metadata. |
| SK-10 | Real Pixal3D fixture exports | An opt-in heavy test or script consumes `inputs/mlx-spatialkit/pixal3d-1024-cascade-decoded-pbr` and writes a textured GLB under `/tmp`, with diagnostics captured in JSON. |
| SK-11 | mlx-spatial integration is clear | `mlx-spatial` has a documented integration boundary for calling `mlx_spatialkit` from Pixal3D export without bundling generated fixtures or weights. |
| SK-12 | Repo cleanliness holds | Verification confirms generated heavy artifacts live in `/tmp` and package artifacts do not include `inputs/`, `outputs/`, `vendors/`, or `weights/`. |

## Scope Coverage Decisions

- **Included:** nested package structure, C++/Metal build plumbing, thin Python bindings, Pixal3D decoded-NPZ contract, native FlexiDualGrid extraction, mesh metrics/diagnostics, native cleanup/simplification interface, replaceable native UV interface, Metal 4 texture bake, GLB writer, tests, real-fixture heavy check, and `mlx-spatial` integration boundary.
- **Deferred:** full general remeshing engine, exact upstream `cumesh` parity, replacing `xatlas` completely, support below Metal 4, broad mesh editing toolkit, and non-Pixal3D model-specific export polish beyond reusable contracts.
- **Anti-goals:** Python-first implementation, diagnostic-only package, MLX dependency inside `mlx-spatialkit`, runtime Torch dependency, generated output checked into the repo, byte-for-byte GLB parity claims, and release/tag work.

## Anti-Goals

- Building a full geometry/remeshing engine in this change.
- Reproducing upstream Pixal3D/TRELLIS export internals exactly.
- Replacing every current export dependency before first GLB success.
- Supporting CUDA, Linux GPU, or Metal versions below Metal 4.
- Moving model inference into `mlx-spatialkit`.
- Publishing or releasing the package as part of this change.

## Blocking Questions or Assumptions

No blocking questions remain.

Assumptions:

- The first native version may use C++ `xatlas` behind a replaceable UV interface if that is the fastest reliable route to working GLB output.
- The real fixture GLB does not need to be byte-identical to `inputs/mlx-spatialkit/pixal3d-1024-cascade-glb-reference/model.glb`; it must be valid, previewable, and diagnostically comparable.
- Planning may decide whether `mlx-spatial` keeps the current Python export path as fallback or switches Pixal3D export directly to `mlx_spatialkit` when installed.
