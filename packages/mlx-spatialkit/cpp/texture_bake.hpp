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
    int64_t atlas_faces_per_tile,
    double tile_padding,
    int64_t max_texture_pixels,
    nanobind::object source_vertices,
    nanobind::object source_faces);

}  // namespace mlx_spatialkit
