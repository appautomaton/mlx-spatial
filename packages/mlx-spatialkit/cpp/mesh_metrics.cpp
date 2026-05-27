#include "mesh_processing.hpp"

#include <algorithm>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mesh_common.hpp"

namespace nb = nanobind;

namespace mlx_spatialkit {
namespace {

constexpr int64_t kSmallBoundaryLoopThresholdEdges = 32;

struct BoundaryTopology {
  int64_t boundary_vertices = 0;
  int64_t loop_count = 0;
  int64_t open_chain_count = 0;
  int64_t small_loop_count = 0;
  int64_t small_loop_edge_count = 0;
  int64_t max_loop_edges = 0;
  int64_t max_component_edges = 0;
};

BoundaryTopology boundary_topology(const std::unordered_map<mesh_common::EdgeKey, int64_t, mesh_common::EdgeKeyHash> &edges) {
  std::unordered_map<int64_t, std::vector<int64_t>> adjacency;
  adjacency.reserve(edges.size() * 2);
  for (const auto &[edge, count] : edges) {
    if (count != 1) {
      continue;
    }
    adjacency[edge.a].push_back(edge.b);
    adjacency[edge.b].push_back(edge.a);
  }

  BoundaryTopology topology;
  topology.boundary_vertices = static_cast<int64_t>(adjacency.size());
  if (adjacency.empty()) {
    return topology;
  }

  std::unordered_set<int64_t> visited;
  visited.reserve(adjacency.size());
  std::vector<int64_t> stack;
  stack.reserve(adjacency.size());
  for (const auto &[seed, _] : adjacency) {
    (void)_;
    if (visited.contains(seed)) {
      continue;
    }

    stack.clear();
    stack.push_back(seed);
    visited.insert(seed);
    int64_t component_vertices = 0;
    int64_t edge_stubs = 0;
    bool every_vertex_degree_two = true;
    while (!stack.empty()) {
      const int64_t vertex = stack.back();
      stack.pop_back();
      component_vertices += 1;
      const auto found = adjacency.find(vertex);
      if (found == adjacency.end()) {
        every_vertex_degree_two = false;
        continue;
      }
      const auto &neighbors = found->second;
      edge_stubs += static_cast<int64_t>(neighbors.size());
      if (neighbors.size() != 2) {
        every_vertex_degree_two = false;
      }
      for (int64_t neighbor : neighbors) {
        if (!visited.contains(neighbor)) {
          visited.insert(neighbor);
          stack.push_back(neighbor);
        }
      }
    }

    const int64_t component_edges = edge_stubs / 2;
    topology.max_component_edges = std::max(topology.max_component_edges, component_edges);
    const bool closed_loop = every_vertex_degree_two && component_vertices >= 3 && component_edges >= 3;
    if (closed_loop) {
      topology.loop_count += 1;
      topology.max_loop_edges = std::max(topology.max_loop_edges, component_edges);
      if (component_edges <= kSmallBoundaryLoopThresholdEdges) {
        topology.small_loop_count += 1;
        topology.small_loop_edge_count += component_edges;
      }
    } else {
      topology.open_chain_count += 1;
    }
  }
  return topology;
}

}  // namespace

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
  const auto edges = mesh_common::edge_counts(mesh.faces);
  for (const auto &[edge, count] : edges) {
    (void)edge;
    if (count == 1) {
      boundary_edges += 1;
    }
    if (count > 2) {
      nonmanifold_edges += 1;
    }
  }
  const BoundaryTopology boundary = boundary_topology(edges);

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
  result["boundary_vertices"] = boundary.boundary_vertices;
  result["boundary_loop_count"] = boundary.loop_count;
  result["boundary_open_chain_count"] = boundary.open_chain_count;
  result["boundary_small_loop_count"] = boundary.small_loop_count;
  result["boundary_small_loop_edge_count"] = boundary.small_loop_edge_count;
  result["boundary_small_loop_threshold_edges"] = kSmallBoundaryLoopThresholdEdges;
  result["boundary_max_loop_edges"] = boundary.max_loop_edges;
  result["boundary_max_component_edges"] = boundary.max_component_edges;
  result["nonmanifold_edges"] = nonmanifold_edges;
  result["connected_components"] = mesh_common::connected_component_count(mesh);
  result["export_blocking_reasons"] = blockers;
  return result;
}

}  // namespace mlx_spatialkit
