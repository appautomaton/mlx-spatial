#include "mesh_processing.hpp"

#include <set>
#include <unordered_map>

#include "mesh_common.hpp"

namespace nb = nanobind;

namespace mlx_spatialkit {
namespace {

mesh_common::MeshData clean_mesh_data(
    const mesh_common::MeshData &input,
    int64_t min_component_faces,
    nb::dict &stats) {
  if (min_component_faces <= 0) {
    throw nb::value_error("min_component_faces must be positive");
  }

  std::vector<std::array<int64_t, 3>> faces;
  faces.reserve(input.faces.size());
  int64_t degenerate_removed = 0;
  for (const auto &face : input.faces) {
    if (mesh_common::face_degenerate(input, face)) {
      degenerate_removed += 1;
    } else {
      faces.push_back(face);
    }
  }

  std::set<std::array<int64_t, 3>> seen;
  std::vector<std::array<int64_t, 3>> deduped_faces;
  deduped_faces.reserve(faces.size());
  int64_t duplicate_removed = 0;
  for (const auto &face : faces) {
    const auto canonical = mesh_common::canonical_face(face);
    if (seen.insert(canonical).second) {
      deduped_faces.push_back(face);
    } else {
      duplicate_removed += 1;
    }
  }

  mesh_common::MeshData cleaned{input.vertices, std::move(deduped_faces)};
  int64_t unreferenced_removed = 0;
  cleaned = mesh_common::compact_mesh(cleaned, &unreferenced_removed);

  int64_t components_removed = 0;
  int64_t component_faces_removed = 0;
  if (!cleaned.faces.empty()) {
    mesh_common::UnionFind uf(cleaned.vertices.size());
    for (const auto &face : cleaned.faces) {
      uf.unite(static_cast<size_t>(face[0]), static_cast<size_t>(face[1]));
      uf.unite(static_cast<size_t>(face[0]), static_cast<size_t>(face[2]));
    }

    std::unordered_map<size_t, int64_t> root_face_counts;
    std::vector<size_t> face_roots;
    face_roots.reserve(cleaned.faces.size());
    for (const auto &face : cleaned.faces) {
      const size_t root = uf.find(static_cast<size_t>(face[0]));
      face_roots.push_back(root);
      root_face_counts[root] += 1;
    }

    size_t largest_root = face_roots.empty() ? 0 : face_roots[0];
    int64_t largest_count = -1;
    for (const auto &[root, count] : root_face_counts) {
      if (count > largest_count) {
        largest_root = root;
        largest_count = count;
      }
    }

    std::set<size_t> keep_roots;
    for (const auto &[root, count] : root_face_counts) {
      if (count >= min_component_faces) {
        keep_roots.insert(root);
      }
    }
    keep_roots.insert(largest_root);
    components_removed = static_cast<int64_t>(root_face_counts.size() - keep_roots.size());

    std::vector<std::array<int64_t, 3>> component_faces;
    component_faces.reserve(cleaned.faces.size());
    for (size_t index = 0; index < cleaned.faces.size(); ++index) {
      if (keep_roots.contains(face_roots[index])) {
        component_faces.push_back(cleaned.faces[index]);
      } else {
        component_faces_removed += 1;
      }
    }
    cleaned.faces = std::move(component_faces);
    cleaned = mesh_common::compact_mesh(cleaned, &unreferenced_removed);
  }

  stats["degenerate_faces_removed"] = degenerate_removed;
  stats["duplicate_faces_removed"] = duplicate_removed;
  stats["unreferenced_vertices_removed"] = unreferenced_removed;
  stats["components_removed"] = components_removed;
  stats["component_faces_removed"] = component_faces_removed;
  stats["final_vertices"] = static_cast<int64_t>(cleaned.vertices.size());
  stats["final_faces"] = static_cast<int64_t>(cleaned.faces.size());
  return cleaned;
}

}  // namespace

nb::dict clean_mesh(nb::object vertices, nb::object faces, int64_t min_component_faces) {
  mesh_common::MeshData input = mesh_common::load_mesh(vertices, faces);
  nb::dict stats;
  mesh_common::MeshData cleaned = clean_mesh_data(input, min_component_faces, stats);
  nb::dict result = mesh_common::mesh_result(cleaned);
  result["stats"] = stats;
  return result;
}

}  // namespace mlx_spatialkit
