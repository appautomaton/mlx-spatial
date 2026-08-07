#pragma once

#include <Python.h>

#include <cstdint>

#include <nanobind/nanobind.h>

namespace mlx_spatialkit {

namespace nb = nanobind;

// Narrow-band dual-contour remesh. Rebuilds watertight topology by isosurfacing
// a thin shell (UDF - eps) around the input surface, mirroring the behavior of
// cumesh.remeshing.remesh_narrow_band_dc (CuMesh/cumesh/remeshing.py:24)
// and src/remesh/simple_dual_contour.cu (mean-of-edge-intersections placement).
//
// vertices: (N, 3) float32; faces: (M, 3) int64; resolution: grid resolution;
// band: narrow-band width in voxels; project_back: ratio to snap vertices back
// onto the original surface (0 = keep the dual-contour shell, the Pixal3D path).
nb::dict remesh_narrow_band(
    nb::object vertices,
    nb::object faces,
    int64_t resolution,
    double band,
    double project_back,
    bool repair_nonmanifold);

// Split disconnected incident face fans onto duplicate vertices without
// deleting faces, mirroring CuMesh's non-manifold repair stage.
nb::dict repair_nonmanifold_mesh(nb::object vertices, nb::object faces);

}  // namespace mlx_spatialkit
