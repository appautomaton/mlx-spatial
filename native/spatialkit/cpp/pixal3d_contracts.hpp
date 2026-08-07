#pragma once

#include <nanobind/nanobind.h>

namespace mlx_spatialkit {

nanobind::dict validate_pixal3d_shape_fields(
    nanobind::object coordinates,
    nanobind::object fields);

nanobind::dict validate_pixal3d_texture_attributes(
    nanobind::object coordinates,
    nanobind::object attributes);

}  // namespace mlx_spatialkit
