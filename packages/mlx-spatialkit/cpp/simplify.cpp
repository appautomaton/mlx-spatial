#include "mesh_processing.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mesh_common.hpp"

namespace nb = nanobind;

namespace mlx_spatialkit {
namespace {

struct ClusterKey {
  int64_t x;
  int64_t y;
  int64_t z;

  bool operator==(const ClusterKey &other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct ClusterKeyHash {
  std::size_t operator()(const ClusterKey &key) const {
    std::size_t value = static_cast<std::size_t>(key.x * 73856093LL);
    value ^= static_cast<std::size_t>(key.y * 19349663LL);
    value ^= static_cast<std::size_t>(key.z * 83492791LL);
    return value;
  }
};

struct FaceKeyHash {
  std::size_t operator()(const std::array<int64_t, 3> &face) const {
    return static_cast<std::size_t>(face[0] * 73856093LL)
        ^ static_cast<std::size_t>(face[1] * 19349663LL)
        ^ static_cast<std::size_t>(face[2] * 83492791LL);
  }
};

struct ClusterAccum {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  int64_t count = 0;
};

struct ClusterResult {
  mesh_common::MeshData mesh;
  int64_t grid_resolution = 0;
  int64_t cluster_count = 0;
  int64_t degenerate_faces_removed = 0;
  int64_t duplicate_faces_removed = 0;
  int64_t nonmanifold_faces_removed = 0;
  int64_t representative_vertices_selected = 0;
};

struct SmallLoopFillResult {
  mesh_common::MeshData mesh;
  int64_t face_budget = 0;
  int64_t loops_considered = 0;
  int64_t loops_filled = 0;
  int64_t loops_rejected = 0;
  int64_t loops_budget_limited = 0;
  int64_t faces_added = 0;
};

struct BackendSelection {
  std::string requested;
  std::string backend;
  std::string algorithm;
  bool topology_aware = false;
};

std::array<float, 3> mesh_min_bounds(const mesh_common::MeshData &mesh) {
  std::array<float, 3> min_bounds{
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity(),
  };
  for (const auto &vertex : mesh.vertices) {
    for (int axis = 0; axis < 3; ++axis) {
      min_bounds[static_cast<size_t>(axis)] = std::min(min_bounds[static_cast<size_t>(axis)], vertex[static_cast<size_t>(axis)]);
    }
  }
  return min_bounds;
}

std::array<float, 3> mesh_max_bounds(const mesh_common::MeshData &mesh) {
  std::array<float, 3> max_bounds{
      -std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
  };
  for (const auto &vertex : mesh.vertices) {
    for (int axis = 0; axis < 3; ++axis) {
      max_bounds[static_cast<size_t>(axis)] = std::max(max_bounds[static_cast<size_t>(axis)], vertex[static_cast<size_t>(axis)]);
    }
  }
  return max_bounds;
}

int64_t cluster_axis(float value, float min_value, float max_value, int64_t grid_resolution) {
  const float extent = max_value - min_value;
  if (!std::isfinite(extent) || extent <= 1e-12f) {
    return 0;
  }
  const double normalized = static_cast<double>(value - min_value) / static_cast<double>(extent);
  const auto index = static_cast<int64_t>(std::floor(normalized * static_cast<double>(grid_resolution)));
  return std::clamp<int64_t>(index, 0, grid_resolution - 1);
}

ClusterResult cluster_mesh(const mesh_common::MeshData &input, int64_t grid_resolution, bool representative_vertices) {
  const std::array<float, 3> min_bounds = mesh_min_bounds(input);
  const std::array<float, 3> max_bounds = mesh_max_bounds(input);
  std::unordered_map<ClusterKey, int64_t, ClusterKeyHash> cluster_ids;
  cluster_ids.reserve(input.vertices.size());
  std::vector<ClusterAccum> accumulators;
  accumulators.reserve(input.vertices.size());
  std::vector<int64_t> vertex_to_cluster(input.vertices.size(), -1);

  for (size_t index = 0; index < input.vertices.size(); ++index) {
    const auto &vertex = input.vertices[index];
    ClusterKey key{
        cluster_axis(vertex[0], min_bounds[0], max_bounds[0], grid_resolution),
        cluster_axis(vertex[1], min_bounds[1], max_bounds[1], grid_resolution),
        cluster_axis(vertex[2], min_bounds[2], max_bounds[2], grid_resolution),
    };
    auto found = cluster_ids.find(key);
    if (found == cluster_ids.end()) {
      const int64_t cluster_id = static_cast<int64_t>(accumulators.size());
      found = cluster_ids.emplace(key, cluster_id).first;
      accumulators.push_back(ClusterAccum{});
    }
    const int64_t cluster_id = found->second;
    vertex_to_cluster[index] = cluster_id;
    ClusterAccum &accum = accumulators[static_cast<size_t>(cluster_id)];
    accum.x += vertex[0];
    accum.y += vertex[1];
    accum.z += vertex[2];
    accum.count += 1;
  }

  mesh_common::MeshData output;
  output.vertices.reserve(accumulators.size());
  if (representative_vertices) {
    std::vector<double> best_distances(accumulators.size(), std::numeric_limits<double>::infinity());
    std::vector<std::array<float, 3>> representatives(accumulators.size(), {0.0f, 0.0f, 0.0f});
    for (size_t index = 0; index < input.vertices.size(); ++index) {
      const int64_t cluster_id = vertex_to_cluster[index];
      const ClusterAccum &accum = accumulators[static_cast<size_t>(cluster_id)];
      const double denom = static_cast<double>(std::max<int64_t>(1, accum.count));
      const double cx = accum.x / denom;
      const double cy = accum.y / denom;
      const double cz = accum.z / denom;
      const auto &vertex = input.vertices[index];
      const double dx = static_cast<double>(vertex[0]) - cx;
      const double dy = static_cast<double>(vertex[1]) - cy;
      const double dz = static_cast<double>(vertex[2]) - cz;
      const double distance = dx * dx + dy * dy + dz * dz;
      if (distance < best_distances[static_cast<size_t>(cluster_id)]) {
        best_distances[static_cast<size_t>(cluster_id)] = distance;
        representatives[static_cast<size_t>(cluster_id)] = vertex;
      }
    }
    for (const auto &representative : representatives) {
      output.vertices.push_back(representative);
    }
  } else {
    for (const auto &accum : accumulators) {
      const double denom = static_cast<double>(std::max<int64_t>(1, accum.count));
      output.vertices.push_back({
          static_cast<float>(accum.x / denom),
          static_cast<float>(accum.y / denom),
          static_cast<float>(accum.z / denom),
      });
    }
  }

  std::unordered_set<std::array<int64_t, 3>, FaceKeyHash> seen_faces;
  seen_faces.reserve(input.faces.size());
  std::unordered_map<mesh_common::EdgeKey, int64_t, mesh_common::EdgeKeyHash> edge_usage;
  edge_usage.reserve(input.faces.size() * 3);
  output.faces.reserve(input.faces.size());
  ClusterResult result;
  for (const auto &face : input.faces) {
    std::array<int64_t, 3> remapped{
        vertex_to_cluster[static_cast<size_t>(face[0])],
        vertex_to_cluster[static_cast<size_t>(face[1])],
        vertex_to_cluster[static_cast<size_t>(face[2])],
    };
    if (remapped[0] == remapped[1] || remapped[1] == remapped[2] || remapped[0] == remapped[2]) {
      result.degenerate_faces_removed += 1;
      continue;
    }
    std::array<int64_t, 3> canonical = remapped;
    std::sort(canonical.begin(), canonical.end());
    if (!seen_faces.insert(canonical).second) {
      result.duplicate_faces_removed += 1;
      continue;
    }
    const std::array<mesh_common::EdgeKey, 3> edges{
        mesh_common::edge_key(remapped[0], remapped[1]),
        mesh_common::edge_key(remapped[1], remapped[2]),
        mesh_common::edge_key(remapped[2], remapped[0]),
    };
    const bool would_make_nonmanifold = std::any_of(edges.begin(), edges.end(), [&](const auto &edge) {
      const auto found = edge_usage.find(edge);
      return found != edge_usage.end() && found->second >= 2;
    });
    if (would_make_nonmanifold) {
      result.nonmanifold_faces_removed += 1;
      continue;
    }
    for (const auto &edge : edges) {
      edge_usage[edge] += 1;
    }
    output.faces.push_back(remapped);
  }
  int64_t unreferenced_removed = 0;
  output = mesh_common::compact_mesh(output, &unreferenced_removed);
  (void)unreferenced_removed;
  result.mesh = std::move(output);
  result.grid_resolution = grid_resolution;
  result.cluster_count = static_cast<int64_t>(accumulators.size());
  result.representative_vertices_selected = representative_vertices ? result.cluster_count : 0;
  return result;
}

std::vector<int64_t> ordered_closed_loop(
    const std::unordered_map<int64_t, std::vector<int64_t>> &adjacency,
    const std::vector<int64_t> &component_vertices) {
  if (component_vertices.size() < 3) {
    return {};
  }
  const int64_t seed = *std::min_element(component_vertices.begin(), component_vertices.end());
  auto found_seed = adjacency.find(seed);
  if (found_seed == adjacency.end() || found_seed->second.size() != 2) {
    return {};
  }

  std::vector<int64_t> ordered;
  ordered.reserve(component_vertices.size());
  int64_t previous = -1;
  int64_t current = seed;
  int64_t next = std::min(found_seed->second[0], found_seed->second[1]);
  for (size_t step = 0; step <= component_vertices.size(); ++step) {
    ordered.push_back(current);
    previous = current;
    current = next;
    if (current == seed) {
      break;
    }
    auto found = adjacency.find(current);
    if (found == adjacency.end() || found->second.size() != 2) {
      return {};
    }
    const auto &neighbors = found->second;
    next = neighbors[0] == previous ? neighbors[1] : neighbors[0];
  }
  if (current != seed || ordered.size() != component_vertices.size()) {
    return {};
  }
  return ordered;
}

SmallLoopFillResult fill_small_boundary_loops(
    const mesh_common::MeshData &input,
    int64_t max_loop_edges,
    int64_t face_budget) {
  SmallLoopFillResult result;
  result.mesh = input;
  result.face_budget = std::max<int64_t>(0, face_budget);
  if (result.face_budget <= 0 || result.mesh.faces.empty()) {
    return result;
  }

  std::unordered_map<mesh_common::EdgeKey, int64_t, mesh_common::EdgeKeyHash> edge_counts =
      mesh_common::edge_counts(result.mesh.faces);
  std::unordered_map<int64_t, std::vector<int64_t>> adjacency;
  adjacency.reserve(edge_counts.size() * 2);
  for (const auto &[edge, count] : edge_counts) {
    if (count != 1) {
      continue;
    }
    adjacency[edge.a].push_back(edge.b);
    adjacency[edge.b].push_back(edge.a);
  }
  if (adjacency.empty()) {
    return result;
  }

  std::unordered_set<std::array<int64_t, 3>, FaceKeyHash> seen_faces;
  seen_faces.reserve(result.mesh.faces.size());
  for (auto face : result.mesh.faces) {
    std::sort(face.begin(), face.end());
    seen_faces.insert(face);
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
    std::vector<int64_t> component_vertices;
    stack.clear();
    stack.push_back(seed);
    visited.insert(seed);
    int64_t edge_stubs = 0;
    bool closed = true;
    while (!stack.empty()) {
      const int64_t vertex = stack.back();
      stack.pop_back();
      component_vertices.push_back(vertex);
      const auto found = adjacency.find(vertex);
      if (found == adjacency.end()) {
        closed = false;
        continue;
      }
      const auto &neighbors = found->second;
      edge_stubs += static_cast<int64_t>(neighbors.size());
      if (neighbors.size() != 2) {
        closed = false;
      }
      for (int64_t neighbor : neighbors) {
        if (!visited.contains(neighbor)) {
          visited.insert(neighbor);
          stack.push_back(neighbor);
        }
      }
    }

    const int64_t component_edges = edge_stubs / 2;
    if (!closed || component_edges < 3 || component_edges > max_loop_edges) {
      continue;
    }
    result.loops_considered += 1;
    const std::vector<int64_t> loop = ordered_closed_loop(adjacency, component_vertices);
    if (loop.size() < 3 || static_cast<int64_t>(loop.size()) != component_edges) {
      result.loops_rejected += 1;
      continue;
    }

    std::vector<std::array<int64_t, 3>> patch_faces;
    patch_faces.reserve(loop.size() - 2);
    bool valid_patch = true;
    std::unordered_map<mesh_common::EdgeKey, int64_t, mesh_common::EdgeKeyHash> local_edge_adds;
    local_edge_adds.reserve((loop.size() - 2) * 3);
    std::set<std::array<int64_t, 3>> local_seen_faces;
    for (size_t index = 1; index + 1 < loop.size(); ++index) {
      std::array<int64_t, 3> face{loop[0], loop[index], loop[index + 1]};
      if (mesh_common::face_degenerate(result.mesh, face)) {
        valid_patch = false;
        break;
      }
      std::array<int64_t, 3> canonical = face;
      std::sort(canonical.begin(), canonical.end());
      if (seen_faces.contains(canonical) || !local_seen_faces.insert(canonical).second) {
        valid_patch = false;
        break;
      }
      const std::array<mesh_common::EdgeKey, 3> edges{
          mesh_common::edge_key(face[0], face[1]),
          mesh_common::edge_key(face[1], face[2]),
          mesh_common::edge_key(face[2], face[0]),
      };
      for (const auto &edge : edges) {
        const int64_t current_count = edge_counts[edge] + local_edge_adds[edge];
        if (current_count >= 2) {
          valid_patch = false;
          break;
        }
        local_edge_adds[edge] += 1;
      }
      if (!valid_patch) {
        break;
      }
      patch_faces.push_back(face);
    }

    if (!valid_patch || patch_faces.empty()) {
      result.loops_rejected += 1;
      continue;
    }
    if (static_cast<int64_t>(patch_faces.size()) > result.face_budget - result.faces_added) {
      result.loops_budget_limited += 1;
      continue;
    }
    for (const auto &face : patch_faces) {
      result.mesh.faces.push_back(face);
      std::array<int64_t, 3> canonical = face;
      std::sort(canonical.begin(), canonical.end());
      seen_faces.insert(canonical);
      edge_counts[mesh_common::edge_key(face[0], face[1])] += 1;
      edge_counts[mesh_common::edge_key(face[1], face[2])] += 1;
      edge_counts[mesh_common::edge_key(face[2], face[0])] += 1;
    }
    result.faces_added += static_cast<int64_t>(patch_faces.size());
    result.loops_filled += 1;
  }
  return result;
}

int64_t initial_grid_resolution(int64_t target_faces) {
  const double resolution = std::ceil(std::sqrt(std::max<double>(2.0, static_cast<double>(target_faces) * 0.5)));
  return std::max<int64_t>(2, static_cast<int64_t>(resolution));
}

BackendSelection resolve_backend(const std::string &backend) {
  if (backend.empty() || backend == "spatial-cluster" || backend == "preview") {
    return BackendSelection{
        "spatial-cluster",
        "spatial-cluster",
        "native_spatial_vertex_clustering",
        false,
    };
  }
  if (backend == "topology-aware") {
    return BackendSelection{
        "topology-aware",
        "topology-aware",
        "native_topology_aware_representative_clustering",
        true,
    };
  }
  throw nb::value_error("simplifier backend must be 'spatial-cluster' or 'topology-aware'");
}

nb::list production_blockers(const BackendSelection &selection, int64_t final_faces, bool target_reached) {
  nb::list blockers;
  if (!selection.topology_aware) {
    blockers.append("preview_backend_tier");
    return blockers;
  }
  if (final_faces <= 0) {
    blockers.append("no_faces");
  }
  if (!target_reached) {
    blockers.append("target_not_reached");
  }
  return blockers;
}

void add_backend_stats(
    nb::dict &stats,
    const BackendSelection &selection,
    int64_t final_faces,
    bool target_reached) {
  nb::list blockers = production_blockers(selection, final_faces, target_reached);
  const bool production_ready = selection.topology_aware && final_faces > 0 && target_reached;
  stats["requested_backend"] = selection.requested;
  stats["backend"] = selection.backend;
  stats["algorithm"] = selection.algorithm;
  stats["quality_tier"] = production_ready ? "production" : (selection.topology_aware ? "production_candidate_blocked" : "geometry_aware_preview");
  stats["production_ready"] = production_ready;
  stats["production_blockers"] = blockers;
  stats["backend_selection_status"] = "selected";
  stats["backend_selection_reason"] = selection.topology_aware ? "topology_aware_backend_requested" : "preview_backend_requested";
}

void add_small_loop_fill_stats(
    nb::dict &stats,
    bool enabled,
    int64_t max_loop_edges,
    const SmallLoopFillResult &fill) {
  stats["small_boundary_loop_fill_enabled"] = enabled;
  stats["small_boundary_loop_fill_max_edges"] = max_loop_edges;
  stats["small_boundary_loop_fill_face_budget"] = fill.face_budget;
  stats["small_boundary_loops_considered"] = fill.loops_considered;
  stats["small_boundary_loops_filled"] = fill.loops_filled;
  stats["small_boundary_loops_rejected"] = fill.loops_rejected;
  stats["small_boundary_loops_budget_limited"] = fill.loops_budget_limited;
  stats["small_boundary_loop_faces_added"] = fill.faces_added;
}

}  // namespace

nb::dict simplify_mesh(
    nb::object vertices,
    nb::object faces,
    int64_t target_faces,
    int64_t min_component_faces,
    const std::string &backend,
    int64_t small_boundary_loop_fill_max_edges) {
  if (target_faces <= 0) {
    throw nb::value_error("target_faces must be positive");
  }
  if (min_component_faces <= 0) {
    throw nb::value_error("min_component_faces must be positive");
  }
  if (small_boundary_loop_fill_max_edges < 0) {
    throw nb::value_error("small_boundary_loop_fill_max_edges must be non-negative");
  }
  const BackendSelection selection = resolve_backend(backend);
  mesh_common::MeshData input = mesh_common::load_mesh(vertices, faces);

  if (static_cast<int64_t>(input.faces.size()) <= target_faces) {
    int64_t unreferenced_removed = 0;
    mesh_common::MeshData compact = mesh_common::compact_mesh(input, &unreferenced_removed);
    SmallLoopFillResult fill;
    fill.mesh = compact;
    const bool small_loop_fill_enabled = selection.topology_aware && small_boundary_loop_fill_max_edges > 0;
    if (small_loop_fill_enabled) {
      fill = fill_small_boundary_loops(
          compact,
          small_boundary_loop_fill_max_edges,
          target_faces - static_cast<int64_t>(compact.faces.size()));
      compact = fill.mesh;
    }
    nb::dict result = mesh_common::mesh_result(compact);
    nb::dict stats;
    const bool target_reached = static_cast<int64_t>(compact.faces.size()) <= target_faces;
    add_backend_stats(stats, selection, static_cast<int64_t>(compact.faces.size()), target_reached);
    add_small_loop_fill_stats(stats, small_loop_fill_enabled, small_boundary_loop_fill_max_edges, fill);
    stats["target_faces"] = target_faces;
    stats["source_faces"] = static_cast<int64_t>(input.faces.size());
    stats["source_vertices"] = static_cast<int64_t>(input.vertices.size());
    stats["final_faces"] = static_cast<int64_t>(compact.faces.size());
    stats["final_vertices"] = static_cast<int64_t>(compact.vertices.size());
    stats["cluster_count"] = static_cast<int64_t>(compact.vertices.size());
    stats["grid_resolution"] = 0;
    stats["degenerate_faces_removed"] = 0;
    stats["duplicate_faces_removed"] = 0;
    stats["nonmanifold_faces_removed"] = 0;
    stats["unreferenced_vertices_removed"] = unreferenced_removed;
    stats["target_reached"] = target_reached;
    stats["simplified"] = false;
    stats["min_component_faces"] = min_component_faces;
    stats["candidate_faces_considered"] = static_cast<int64_t>(input.faces.size());
    stats["accepted_faces"] = static_cast<int64_t>(compact.faces.size());
    stats["representative_vertices_selected"] = selection.topology_aware ? static_cast<int64_t>(compact.vertices.size()) : 0;
    result["stats"] = stats;
    return result;
  }

  int64_t grid_resolution = initial_grid_resolution(target_faces);
  ClusterResult best = cluster_mesh(input, grid_resolution, selection.topology_aware);
  for (int attempt = 0; attempt < 4; ++attempt) {
    const int64_t final_faces = static_cast<int64_t>(best.mesh.faces.size());
    if (final_faces <= target_faces && final_faces >= std::max<int64_t>(1, target_faces * 6 / 10)) {
      break;
    }
    if (final_faces == 0) {
      grid_resolution = std::max<int64_t>(2, grid_resolution * 2);
    } else {
      const double scale = std::sqrt(static_cast<double>(target_faces) / static_cast<double>(final_faces));
      double adjusted = static_cast<double>(grid_resolution) * scale * (final_faces > target_faces ? 0.95 : 1.05);
      grid_resolution = std::max<int64_t>(2, static_cast<int64_t>(std::ceil(adjusted)));
    }
    ClusterResult candidate = cluster_mesh(input, grid_resolution, selection.topology_aware);
    if (candidate.mesh.faces.empty()) {
      continue;
    }
    const int64_t candidate_faces = static_cast<int64_t>(candidate.mesh.faces.size());
    const int64_t best_faces = static_cast<int64_t>(best.mesh.faces.size());
    if (best_faces == 0 || std::llabs(candidate_faces - target_faces) < std::llabs(best_faces - target_faces)
        || (candidate_faces <= target_faces && best_faces > target_faces)) {
      best = std::move(candidate);
    }
  }

  int64_t unreferenced_removed = 0;
  mesh_common::MeshData simplified = mesh_common::compact_mesh(best.mesh, &unreferenced_removed);
  SmallLoopFillResult fill;
  fill.mesh = simplified;
  const bool small_loop_fill_enabled = selection.topology_aware && small_boundary_loop_fill_max_edges > 0;
  if (small_loop_fill_enabled) {
    fill = fill_small_boundary_loops(
        simplified,
        small_boundary_loop_fill_max_edges,
        target_faces - static_cast<int64_t>(simplified.faces.size()));
    simplified = fill.mesh;
  }
  nb::dict result = mesh_common::mesh_result(simplified);
  nb::dict stats;
  const bool target_reached = static_cast<int64_t>(simplified.faces.size()) <= target_faces;
  add_backend_stats(stats, selection, static_cast<int64_t>(simplified.faces.size()), target_reached);
  add_small_loop_fill_stats(stats, small_loop_fill_enabled, small_boundary_loop_fill_max_edges, fill);
  stats["target_faces"] = target_faces;
  stats["source_faces"] = static_cast<int64_t>(input.faces.size());
  stats["source_vertices"] = static_cast<int64_t>(input.vertices.size());
  stats["final_faces"] = static_cast<int64_t>(simplified.faces.size());
  stats["final_vertices"] = static_cast<int64_t>(simplified.vertices.size());
  stats["cluster_count"] = best.cluster_count;
  stats["grid_resolution"] = best.grid_resolution;
  stats["degenerate_faces_removed"] = best.degenerate_faces_removed;
  stats["duplicate_faces_removed"] = best.duplicate_faces_removed;
  stats["nonmanifold_faces_removed"] = best.nonmanifold_faces_removed;
  stats["unreferenced_vertices_removed"] = unreferenced_removed;
  stats["target_reached"] = target_reached;
  stats["simplified"] = static_cast<int64_t>(input.faces.size()) > target_faces;
  stats["min_component_faces"] = min_component_faces;
  stats["candidate_faces_considered"] = static_cast<int64_t>(input.faces.size());
  stats["accepted_faces"] = static_cast<int64_t>(simplified.faces.size());
  stats["representative_vertices_selected"] = best.representative_vertices_selected;
  result["stats"] = stats;
  return result;
}

}  // namespace mlx_spatialkit
