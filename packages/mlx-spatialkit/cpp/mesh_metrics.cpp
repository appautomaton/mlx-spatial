#include "mesh_processing.hpp"

#include <set>
#include <unordered_map>

#include "mesh_common.hpp"

namespace nb = nanobind;

namespace mlx_spatialkit {

nb::dict mesh_metrics(nb::object vertices, nb::object faces) {
  mesh_common::MeshData mesh = mesh_common::load_mesh(vertices, faces);
  int64_t degenerate_faces = 0;
  std::set<std::array<int64_t, 3>> canonical_faces;
  int64_t duplicate_faces = 0;
  for (const auto &face : mesh.faces) {
    if (mesh_common::face_degenerate(mesh, face)) {
      degenerate_faces += 1;
    }
    const auto canonical = mesh_common::canonical_face(face);
    if (!canonical_faces.insert(canonical).second) {
      duplicate_faces += 1;
    }
  }

  int64_t boundary_edges = 0;
  int64_t nonmanifold_edges = 0;
  for (const auto &[edge, count] : mesh_common::edge_counts(mesh.faces)) {
    (void)edge;
    if (count == 1) {
      boundary_edges += 1;
    }
    if (count > 2) {
      nonmanifold_edges += 1;
    }
  }

  nb::list blockers;
  if (mesh.faces.empty()) {
    blockers.append("no_faces");
  }
  if (degenerate_faces > 0) {
    blockers.append("degenerate_faces_present");
  }
  if (duplicate_faces > 0) {
    blockers.append("duplicate_faces_present");
  }
  if (nonmanifold_edges > 0) {
    blockers.append("nonmanifold_edges_present");
  }

  nb::dict result;
  result["vertex_count"] = static_cast<int64_t>(mesh.vertices.size());
  result["face_count"] = static_cast<int64_t>(mesh.faces.size());
  result["degenerate_faces"] = degenerate_faces;
  result["duplicate_faces"] = duplicate_faces;
  result["boundary_edges"] = boundary_edges;
  result["nonmanifold_edges"] = nonmanifold_edges;
  result["connected_components"] = mesh_common::connected_component_count(mesh);
  result["export_blocking_reasons"] = blockers;
  return result;
}

}  // namespace mlx_spatialkit
