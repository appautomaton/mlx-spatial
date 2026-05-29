#pragma once

#include <nanobind/nanobind.h>

#include <string>

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
    nanobind::object source_faces,
    std::string source_projection_fallback_mode,
    int64_t source_projection_fallback_neighbors,
    double source_projection_fallback_max_distance_voxels,
    bool render_padding,
    bool surface_fill);

}  // namespace mlx_spatialkit
