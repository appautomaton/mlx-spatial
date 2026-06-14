#pragma once

#include <nanobind/nanobind.h>

namespace mlx_spatialkit {

nanobind::dict extract_flexi_dual_grid(
    nanobind::object coordinates,
    nanobind::object fields,
    int64_t grid_size);

}  // namespace mlx_spatialkit
