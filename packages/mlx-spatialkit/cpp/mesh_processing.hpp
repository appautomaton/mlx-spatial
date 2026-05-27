#pragma once

#include <nanobind/nanobind.h>

namespace mlx_spatialkit {

nanobind::dict mesh_metrics(nanobind::object vertices, nanobind::object faces);

nanobind::dict clean_mesh(
    nanobind::object vertices,
    nanobind::object faces,
    int64_t min_component_faces);

nanobind::dict simplify_mesh(
    nanobind::object vertices,
    nanobind::object faces,
    int64_t target_faces,
    int64_t min_component_faces);

}  // namespace mlx_spatialkit
