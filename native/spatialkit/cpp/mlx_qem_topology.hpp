#pragma once

#include <cstdint>

#include <nanobind/nanobind.h>

namespace mlx_spatialkit {

nanobind::dict build_mlx_qem_topology(int64_t vertex_count, nanobind::object faces);

nanobind::dict apply_mlx_qem_collapse_batch(
    nanobind::object vertices,
    nanobind::object faces,
    nanobind::object selected_edges,
    nanobind::object boundary_vertices);

}  // namespace mlx_spatialkit
