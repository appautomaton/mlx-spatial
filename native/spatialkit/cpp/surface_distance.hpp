#pragma once

#include <nanobind/nanobind.h>

namespace mlx_spatialkit {

// Compute exact unsigned distances from query points to a triangle mesh using
// the same native BVH primitive as remeshing and texture projection.
nanobind::dict point_to_mesh_distances(
    nanobind::object query_points,
    nanobind::object vertices,
    nanobind::object faces);

}  // namespace mlx_spatialkit
