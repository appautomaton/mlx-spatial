#include "mesh_processing.hpp"

#include <set>
#include <unordered_map>

#include "mesh_common.hpp"

namespace nb = nanobind;

namespace mlx_spatialkit {
namespace {

struct CleanStats {
  int64_t degenerate_faces_removed = 0;
  int64_t duplicate_faces_removed = 0;
  int64_t unreferenced_vertices_removed = 0;
  int64_t components_removed = 0;
  int64_t component_faces_removed = 0;
  int64_t final_vertices = 0;
  int64_t final_faces = 0;
};

mesh_common::MeshData clean_mesh_data(
    const mesh_common::MeshData &input,
    int64_t min_component_faces,
    CleanStats &stats) {
  std::vector<std::array<int64_t, 3>> faces;
  faces.reserve(input.faces.size());
  for (const auto &face : input.faces) {
    if (mesh_common::face_degenerate(input, face)) {
      stats.degenerate_faces_removed += 1;
    } else {
      faces.push_back(face);
    }
  }

  std::set<std::array<int64_t, 3>> seen;
  std::vector<std::array<int64_t, 3>> deduped_faces;
  deduped_faces.reserve(faces.size());
  for (const auto &face : faces) {
    const auto canonical = mesh_common::canonical_face(face);
    if (seen.insert(canonical).second) {
      deduped_faces.push_back(face);
    } else {
      stats.duplicate_faces_removed += 1;
    }
  }

  mesh_common::MeshData cleaned{input.vertices, std::move(deduped_faces)};
  cleaned = mesh_common::compact_mesh(cleaned, &stats.unreferenced_vertices_removed);

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
    stats.components_removed = static_cast<int64_t>(root_face_counts.size() - keep_roots.size());

    std::vector<std::array<int64_t, 3>> component_faces;
    component_faces.reserve(cleaned.faces.size());
    for (size_t index = 0; index < cleaned.faces.size(); ++index) {
      if (keep_roots.contains(face_roots[index])) {
        component_faces.push_back(cleaned.faces[index]);
      } else {
        stats.component_faces_removed += 1;
      }
    }
    cleaned.faces = std::move(component_faces);
    cleaned = mesh_common::compact_mesh(cleaned, &stats.unreferenced_vertices_removed);
  }

  stats.final_vertices = static_cast<int64_t>(cleaned.vertices.size());
  stats.final_faces = static_cast<int64_t>(cleaned.faces.size());
  return cleaned;
}

}  // namespace

nb::dict clean_mesh(nb::object vertices, nb::object faces, int64_t min_component_faces) {
  if (min_component_faces <= 0) {
    throw nb::value_error("min_component_faces must be positive");
  }
  mesh_common::MeshData input = mesh_common::load_mesh(vertices, faces);
  CleanStats clean_stats;
  mesh_common::MeshData cleaned;
  {
    nb::gil_scoped_release release;
    cleaned = clean_mesh_data(input, min_component_faces, clean_stats);
  }

  nb::dict stats;
  stats["degenerate_faces_removed"] = clean_stats.degenerate_faces_removed;
  stats["duplicate_faces_removed"] = clean_stats.duplicate_faces_removed;
  stats["unreferenced_vertices_removed"] = clean_stats.unreferenced_vertices_removed;
  stats["components_removed"] = clean_stats.components_removed;
  stats["component_faces_removed"] = clean_stats.component_faces_removed;
  stats["final_vertices"] = clean_stats.final_vertices;
  stats["final_faces"] = clean_stats.final_faces;
  nb::dict result = mesh_common::mesh_result(cleaned);
  result["stats"] = stats;
  return result;
}

}  // namespace mlx_spatialkit
