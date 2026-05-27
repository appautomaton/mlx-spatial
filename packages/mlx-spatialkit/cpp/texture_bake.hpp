#pragma once

#include <nanobind/nanobind.h>

namespace mlx_spatialkit {

nanobind::dict bake_pbr_texture_metal(
    nanobind::object vertices,
    nanobind::object faces,
    nanobind::object uvs,
    nanobind::object texture_coordinates,
    nanobind::object texture_attributes,
    int64_t texture_size,
    nanobind::object origin,
    double voxel_size,
    int64_t decode_resolution,
    int64_t atlas_cols,
    int64_t atlas_rows,
    double tile_padding,
    int64_t max_texture_pixels);

}  // namespace mlx_spatialkit
