#pragma once

#include <nanobind/nanobind.h>

namespace mlx_spatialkit {

nanobind::dict uv_quality_metrics(
    nanobind::object vertices,
    nanobind::object faces,
    nanobind::object uvs,
    nanobind::object chart_ids);

}  // namespace mlx_spatialkit
