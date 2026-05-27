#include "mesh_processing.hpp"

#include <algorithm>

#include "mesh_common.hpp"

namespace nb = nanobind;

namespace mlx_spatialkit {

nb::dict simplify_mesh(
    nb::object vertices,
    nb::object faces,
    int64_t target_faces,
    int64_t min_component_faces) {
  if (target_faces <= 0) {
    throw nb::value_error("target_faces must be positive");
  }
  if (min_component_faces <= 0) {
    throw nb::value_error("min_component_faces must be positive");
  }
  mesh_common::MeshData input = mesh_common::load_mesh(vertices, faces);

  std::vector<std::array<int64_t, 3>> selected_faces;
  if (static_cast<int64_t>(input.faces.size()) <= target_faces) {
    selected_faces = input.faces;
  } else {
    selected_faces.reserve(static_cast<size_t>(target_faces));
    for (int64_t index = 0; index < target_faces; ++index) {
      const int64_t source_index = std::min<int64_t>(
          static_cast<int64_t>(input.faces.size()) - 1,
          (index * static_cast<int64_t>(input.faces.size())) / target_faces);
      selected_faces.push_back(input.faces[static_cast<size_t>(source_index)]);
    }
  }

  mesh_common::MeshData simplified{input.vertices, std::move(selected_faces)};
  int64_t unreferenced_removed = 0;
  simplified = mesh_common::compact_mesh(simplified, &unreferenced_removed);
  nb::dict result = mesh_common::mesh_result(simplified);
  nb::dict stats;
  stats["backend"] = "face-stride-preview";
  stats["algorithm"] = "deterministic_face_stride_compaction";
  stats["quality_tier"] = "preview";
  stats["production_ready"] = false;
  stats["target_faces"] = target_faces;
  stats["source_faces"] = static_cast<int64_t>(input.faces.size());
  stats["final_faces"] = static_cast<int64_t>(simplified.faces.size());
  stats["unreferenced_vertices_removed"] = unreferenced_removed;
  stats["simplified"] = static_cast<int64_t>(input.faces.size()) > target_faces;
  stats["min_component_faces"] = min_component_faces;
  result["stats"] = stats;
  return result;
}

}  // namespace mlx_spatialkit
