#pragma once

#include <string>

#include <nanobind/nanobind.h>

namespace mlx_spatialkit {

nanobind::dict make_face_atlas_uvs(
    nanobind::object vertices,
    nanobind::object faces,
    double tile_padding);

nanobind::dict make_native_chart_uvs(
    nanobind::object vertices,
    nanobind::object faces,
    double chart_angle_degrees,
    double tile_padding);

nanobind::bytes textured_glb_payload(
    nanobind::object vertices,
    nanobind::object faces,
    nanobind::object uvs,
    nanobind::object base_color_rgba,
    nanobind::object metallic_roughness,
    const std::string &generator,
    const std::string &mesh_name,
    const std::string &material_name);

}  // namespace mlx_spatialkit
