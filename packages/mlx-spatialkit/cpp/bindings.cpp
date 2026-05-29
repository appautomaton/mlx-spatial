#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "flexi_dual_grid.hpp"
#include "glb_writer.hpp"
#include "metal_probe.hpp"
#include "mesh_processing.hpp"
#include "pixal3d_contracts.hpp"
#include "remesh.hpp"
#include "texture_bake.hpp"

namespace nb = nanobind;

NB_MODULE(_native, module) {
  module.doc() = "Native C++/Metal bindings for mlx-spatialkit";

  module.def("metal_device_available", &mlx_spatialkit::metal_device_available,
             "Return whether the default Metal device is available.");

  module.def("backend_info", []() {
    nb::dict info;
    info["native"] = true;
    info["metal_available"] = mlx_spatialkit::metal_device_available();
    info["metal_device"] = mlx_spatialkit::metal_device_name();
    return info;
  }, "Return native backend build and Metal availability information.");

  module.def("validate_pixal3d_shape_fields",
             &mlx_spatialkit::validate_pixal3d_shape_fields,
             "Validate Pixal3D decoded shape coordinate and field arrays.");

  module.def("validate_pixal3d_texture_attributes",
             &mlx_spatialkit::validate_pixal3d_texture_attributes,
             "Validate Pixal3D decoded texture coordinate and attribute arrays.");

  module.def("extract_flexi_dual_grid",
             &mlx_spatialkit::extract_flexi_dual_grid,
             nb::arg("coordinates"),
             nb::arg("fields"),
             nb::arg("grid_size"),
             "Extract a triangle mesh from Pixal3D FlexiDualGrid fields.");

  module.def("mesh_metrics",
             &mlx_spatialkit::mesh_metrics,
             nb::arg("vertices"),
             nb::arg("faces"),
             "Return native mesh metrics and export-blocking reasons.");

  module.def("clean_mesh",
             &mlx_spatialkit::clean_mesh,
             nb::arg("vertices"),
             nb::arg("faces"),
             nb::arg("min_component_faces") = 32,
             "Clean degenerate, duplicate, unreferenced, and tiny-component mesh data.");

  module.def("simplify_mesh",
             &mlx_spatialkit::simplify_mesh,
             nb::arg("vertices"),
             nb::arg("faces"),
             nb::arg("target_faces"),
             nb::arg("min_component_faces") = 32,
             nb::arg("backend") = "spatial-cluster",
             nb::arg("small_boundary_loop_fill_max_edges") = 8,
             nb::arg("small_boundary_loop_fill_max_perimeter") = 0.03,
             "Run the native-owned first-pass mesh simplification interface.");

  module.def("remesh_narrow_band",
             &mlx_spatialkit::remesh_narrow_band,
             nb::arg("vertices"),
             nb::arg("faces"),
             nb::arg("resolution"),
             nb::arg("band") = 1.0,
             nb::arg("project_back") = 0.0,
             nb::arg("repair_nonmanifold") = false,
             "Narrow-band dual-contour remesh into a watertight triangle mesh.");

  module.def("make_face_atlas_uvs",
             &mlx_spatialkit::make_face_atlas_uvs,
             nb::arg("vertices"),
             nb::arg("faces"),
             nb::arg("tile_padding") = 0.08,
             "Create a deterministic native face-atlas UV mesh.");

  module.def("make_native_chart_uvs",
             &mlx_spatialkit::make_native_chart_uvs,
             nb::arg("vertices"),
             nb::arg("faces"),
             nb::arg("chart_angle_degrees") = 45.0,
             nb::arg("tile_padding") = 0.04,
             "Create a deterministic native chart UV mesh.");

  module.def("textured_glb_payload",
             &mlx_spatialkit::textured_glb_payload,
             nb::arg("vertices"),
             nb::arg("faces"),
             nb::arg("uvs"),
             nb::arg("base_color_rgba"),
             nb::arg("metallic_roughness"),
             nb::arg("generator") = "mlx-spatialkit",
             nb::arg("mesh_name") = "TexturedMesh",
             nb::arg("material_name") = "PBRMaterial",
             "Build a self-contained GLB 2.0 payload with embedded PBR textures.");

  module.def("bake_pbr_texture_metal",
             &mlx_spatialkit::bake_pbr_texture_metal,
             nb::arg("vertices"),
             nb::arg("faces"),
             nb::arg("uvs"),
             nb::arg("texture_coordinates"),
             nb::arg("texture_attributes"),
             nb::arg("texture_size"),
             nb::arg("origin"),
             nb::arg("voxel_size"),
             nb::arg("decode_resolution"),
             nb::arg("atlas_cols") = 0,
             nb::arg("atlas_rows") = 0,
             nb::arg("atlas_faces_per_tile") = 0,
             nb::arg("tile_padding") = 0.08,
             nb::arg("max_texture_pixels") = 1048576,
             nb::arg("source_vertices") = nb::none(),
             nb::arg("source_faces") = nb::none(),
             nb::arg("source_projection_fallback_mode") = "knn",
             nb::arg("source_projection_fallback_neighbors") = 8,
             nb::arg("source_projection_fallback_max_distance_voxels") = 12.0,
             nb::arg("render_padding") = true,
             nb::arg("surface_fill") = true,
             "Bake PBR texture buffers with the Metal backend.");
}
