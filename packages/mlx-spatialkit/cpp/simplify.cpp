#include "mesh_processing.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
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
  int64_t quadric_representative_candidates_evaluated = 0;
  int64_t quadric_representative_nonfinite_candidates = 0;
  double quadric_representative_error_sum = 0.0;
  double quadric_representative_error_max = 0.0;
  std::string representative_selection_strategy = "not_requested";
};

struct SmallLoopFillResult {
  mesh_common::MeshData mesh;
  int64_t face_budget = 0;
  int64_t repair_pass_count = 0;
  int64_t loops_considered = 0;
  int64_t loops_filled = 0;
  int64_t loops_filled_by_ear_clipping = 0;
  int64_t loops_alternative_triangulation_attempted = 0;
  int64_t loops_filled_by_alternative_triangulation = 0;
  int64_t loops_centroid_fan_attempted = 0;
  int64_t loops_filled_by_centroid_fan = 0;
  int64_t loops_rejected = 0;
  int64_t loops_rejected_ordering = 0;
  int64_t loops_rejected_triangulation = 0;
  int64_t loops_rejected_perimeter = 0;
  int64_t loops_rejected_edge_cap = 0;
  int64_t loops_rejected_fallback_cap = 0;
  int64_t loops_rejected_degenerate = 0;
  int64_t loops_rejected_duplicate = 0;
  int64_t loops_rejected_nonmanifold = 0;
  int64_t loops_budget_limited = 0;
  int64_t loops_edge_count_sum = 0;
  int64_t loops_edge_count_max = 0;
  double loops_perimeter_sum = 0.0;
  double loops_perimeter_max = 0.0;
  int64_t loops_rejected_perimeter_edge_count_sum = 0;
  int64_t loops_rejected_perimeter_edge_count_max = 0;
  double loops_rejected_perimeter_sum = 0.0;
  double loops_rejected_perimeter_min = std::numeric_limits<double>::infinity();
  double loops_rejected_perimeter_max = 0.0;
  int64_t branched_cycle_candidates = 0;
  int64_t branched_cycles_filled = 0;
  int64_t branched_cycles_rejected = 0;
  int64_t branched_cycles_budget_limited = 0;
  int64_t faces_added = 0;
};

struct ReferenceLoopFillResult {
  mesh_common::MeshData mesh;
  int64_t boundary_edges_before = 0;
  int64_t clean_boundary_loops = 0;
  int64_t filled_loops = 0;
  int64_t skipped_large_loops = 0;
  int64_t skipped_complex_components = 0;
  int64_t vertices_added = 0;
  int64_t faces_added = 0;
};

struct Point2 {
  double x = 0.0;
  double y = 0.0;
};

struct BackendSelection {
  std::string requested;
  std::string backend;
  std::string algorithm;
  bool topology_aware = false;
  bool is_qem = false;
};

struct Quadric {
  std::array<double, 16> values{};
};

constexpr const char *kSmallBoundaryLoopFillAlgorithm = "cumesh-perimeter-centroid-fan";
constexpr const char *kSmallBoundaryLoopFillFallbackAlgorithm = "disabled";
constexpr double kSmallBoundaryLoopFillMaxPerimeter = 3.0e-2;
constexpr int64_t kSmallBoundaryLoopFillFallbackMaxEdges = 0;
constexpr int64_t kSmallBoundaryBranchedCycleFillMaxEdges = 8;
constexpr int64_t kSmallBoundaryLoopRepairMaxPasses = 3;
constexpr int64_t kPreSimplifyCleanBoundaryLoopFillMaxEdges = 64;
constexpr size_t kSmallBoundaryAlternativeTriangulationMaxVariants = 256;

enum class PatchRejectReason {
  none,
  degenerate,
  duplicate,
  nonmanifold,
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

void add_plane_to_quadric(Quadric &quadric, const std::array<double, 4> &plane) {
  for (size_t row = 0; row < 4; ++row) {
    for (size_t col = 0; col < 4; ++col) {
      quadric.values[row * 4 + col] += plane[row] * plane[col];
    }
  }
}

void add_quadric(Quadric &left, const Quadric &right) {
  for (size_t index = 0; index < left.values.size(); ++index) {
    left.values[index] += right.values[index];
  }
}

double evaluate_quadric(const Quadric &quadric, const std::array<float, 3> &vertex) {
  const std::array<double, 4> value{
      static_cast<double>(vertex[0]),
      static_cast<double>(vertex[1]),
      static_cast<double>(vertex[2]),
      1.0,
  };
  double result = 0.0;
  for (size_t row = 0; row < 4; ++row) {
    double row_value = 0.0;
    for (size_t col = 0; col < 4; ++col) {
      row_value += quadric.values[row * 4 + col] * value[col];
    }
    result += value[row] * row_value;
  }
  return result;
}

std::vector<Quadric> vertex_plane_quadrics(const mesh_common::MeshData &input) {
  std::vector<Quadric> quadrics(input.vertices.size());
  for (const auto &face : input.faces) {
    const auto &a = input.vertices[static_cast<size_t>(face[0])];
    const auto &b = input.vertices[static_cast<size_t>(face[1])];
    const auto &c = input.vertices[static_cast<size_t>(face[2])];
    const std::array<double, 3> ab{
        static_cast<double>(b[0]) - static_cast<double>(a[0]),
        static_cast<double>(b[1]) - static_cast<double>(a[1]),
        static_cast<double>(b[2]) - static_cast<double>(a[2]),
    };
    const std::array<double, 3> ac{
        static_cast<double>(c[0]) - static_cast<double>(a[0]),
        static_cast<double>(c[1]) - static_cast<double>(a[1]),
        static_cast<double>(c[2]) - static_cast<double>(a[2]),
    };
    std::array<double, 3> normal{
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    };
    const double length = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    if (!std::isfinite(length) || length <= 1e-18) {
      continue;
    }
    normal[0] /= length;
    normal[1] /= length;
    normal[2] /= length;
    const double d =
        -(normal[0] * static_cast<double>(a[0]) + normal[1] * static_cast<double>(a[1]) +
          normal[2] * static_cast<double>(a[2]));
    const std::array<double, 4> plane{normal[0], normal[1], normal[2], d};
    for (int corner = 0; corner < 3; ++corner) {
      add_plane_to_quadric(quadrics[static_cast<size_t>(face[corner])], plane);
    }
  }
  return quadrics;
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

  ClusterResult result;
  mesh_common::MeshData output;
  output.vertices.reserve(accumulators.size());
  if (representative_vertices) {
    const std::vector<Quadric> vertex_quadrics = vertex_plane_quadrics(input);
    std::vector<Quadric> cluster_quadrics(accumulators.size());
    std::vector<std::vector<size_t>> cluster_members(accumulators.size());
    for (auto &members : cluster_members) {
      members.reserve(8);
    }
    for (size_t index = 0; index < input.vertices.size(); ++index) {
      const int64_t cluster_id = vertex_to_cluster[index];
      if (cluster_id < 0) {
        continue;
      }
      cluster_members[static_cast<size_t>(cluster_id)].push_back(index);
      add_quadric(cluster_quadrics[static_cast<size_t>(cluster_id)], vertex_quadrics[index]);
    }
    std::vector<std::array<float, 3>> representatives(accumulators.size(), {0.0f, 0.0f, 0.0f});
    for (size_t cluster_id = 0; cluster_id < cluster_members.size(); ++cluster_id) {
      const ClusterAccum &accum = accumulators[static_cast<size_t>(cluster_id)];
      const double denom = static_cast<double>(std::max<int64_t>(1, accum.count));
      const double cx = accum.x / denom;
      const double cy = accum.y / denom;
      const double cz = accum.z / denom;
      double best_error = std::numeric_limits<double>::infinity();
      double best_distance = std::numeric_limits<double>::infinity();
      size_t best_source_index = std::numeric_limits<size_t>::max();
      for (size_t source_index : cluster_members[cluster_id]) {
        const auto &vertex = input.vertices[source_index];
        double error = evaluate_quadric(cluster_quadrics[cluster_id], vertex);
        if (!std::isfinite(error)) {
          result.quadric_representative_nonfinite_candidates += 1;
          error = std::numeric_limits<double>::infinity();
        }
        const double dx = static_cast<double>(vertex[0]) - cx;
        const double dy = static_cast<double>(vertex[1]) - cy;
        const double dz = static_cast<double>(vertex[2]) - cz;
        const double distance = dx * dx + dy * dy + dz * dz;
        result.quadric_representative_candidates_evaluated += 1;
        if (error < best_error - 1e-18 ||
            (std::fabs(error - best_error) <= 1e-18 && distance < best_distance - 1e-18) ||
            (std::fabs(error - best_error) <= 1e-18 && std::fabs(distance - best_distance) <= 1e-18 &&
             source_index < best_source_index)) {
          best_error = error;
          best_distance = distance;
          best_source_index = source_index;
        }
      }
      if (best_source_index == std::numeric_limits<size_t>::max()) {
        representatives[cluster_id] = {
            static_cast<float>(cx),
            static_cast<float>(cy),
            static_cast<float>(cz),
        };
      } else {
        representatives[cluster_id] = input.vertices[best_source_index];
        if (std::isfinite(best_error)) {
          result.quadric_representative_error_sum += best_error;
          result.quadric_representative_error_max =
              std::max(result.quadric_representative_error_max, best_error);
        }
      }
    }
    for (const auto &representative : representatives) {
      output.vertices.push_back(representative);
    }
    result.representative_selection_strategy = "cluster_quadric_error_minimizer";
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

struct DirectedBoundaryEdge {
  int64_t start = 0;
  int64_t end = 0;
};

struct DirectedEdgeKey {
  int64_t start = 0;
  int64_t end = 0;

  bool operator==(const DirectedEdgeKey &other) const {
    return start == other.start && end == other.end;
  }
};

struct DirectedEdgeKeyHash {
  std::size_t operator()(const DirectedEdgeKey &edge) const {
    return static_cast<std::size_t>(edge.start * 1000003LL) ^ static_cast<std::size_t>(edge.end);
  }
};

std::vector<DirectedBoundaryEdge> reference_boundary_edges(const mesh_common::MeshData &mesh) {
  const std::unordered_map<mesh_common::EdgeKey, int64_t, mesh_common::EdgeKeyHash> counts =
      mesh_common::edge_counts(mesh.faces);
  std::vector<DirectedBoundaryEdge> boundary_edges;
  boundary_edges.reserve(counts.size());
  for (const auto &face : mesh.faces) {
    const std::array<DirectedBoundaryEdge, 3> directed_edges{
        DirectedBoundaryEdge{face[0], face[1]},
        DirectedBoundaryEdge{face[1], face[2]},
        DirectedBoundaryEdge{face[2], face[0]},
    };
    for (const DirectedBoundaryEdge &edge : directed_edges) {
      const mesh_common::EdgeKey key = mesh_common::edge_key(edge.start, edge.end);
      const auto found = counts.find(key);
      if (found != counts.end() && found->second == 1) {
        boundary_edges.push_back(edge);
      }
    }
  }
  return boundary_edges;
}

std::unordered_map<int64_t, std::vector<int64_t>> boundary_adjacency(
    const std::vector<DirectedBoundaryEdge> &boundary_edges) {
  std::unordered_map<int64_t, std::vector<int64_t>> adjacency;
  adjacency.reserve(boundary_edges.size() * 2);
  for (const DirectedBoundaryEdge &edge : boundary_edges) {
    adjacency[edge.start].push_back(edge.end);
    adjacency[edge.end].push_back(edge.start);
  }
  for (auto &[_, neighbors] : adjacency) {
    (void)_;
    std::sort(neighbors.begin(), neighbors.end());
  }
  return adjacency;
}

int64_t count_complex_boundary_components(
    const std::unordered_map<int64_t, std::vector<int64_t>> &adjacency,
    const std::vector<DirectedBoundaryEdge> &boundary_edges) {
  std::unordered_set<mesh_common::EdgeKey, mesh_common::EdgeKeyHash> visited_edges;
  visited_edges.reserve(boundary_edges.size());
  int64_t complex_components = 0;
  std::vector<int64_t> stack;
  for (const DirectedBoundaryEdge &raw_edge : boundary_edges) {
    const mesh_common::EdgeKey first_edge = mesh_common::edge_key(raw_edge.start, raw_edge.end);
    if (visited_edges.contains(first_edge)) {
      continue;
    }
    std::unordered_set<int64_t> component_vertices;
    std::unordered_set<mesh_common::EdgeKey, mesh_common::EdgeKeyHash> component_edges;
    stack.clear();
    stack.push_back(raw_edge.start);
    while (!stack.empty()) {
      const int64_t vertex = stack.back();
      stack.pop_back();
      if (!component_vertices.insert(vertex).second) {
        continue;
      }
      const auto found = adjacency.find(vertex);
      if (found == adjacency.end()) {
        continue;
      }
      for (int64_t neighbor : found->second) {
        const mesh_common::EdgeKey edge = mesh_common::edge_key(vertex, neighbor);
        component_edges.insert(edge);
        if (!visited_edges.contains(edge)) {
          visited_edges.insert(edge);
          stack.push_back(neighbor);
        }
      }
    }
    const bool complex =
        std::any_of(component_vertices.begin(), component_vertices.end(), [&](int64_t vertex) {
          const auto found = adjacency.find(vertex);
          return found == adjacency.end() || found->second.size() != 2;
        }) ||
        component_edges.size() != component_vertices.size();
    if (complex) {
      complex_components += 1;
    }
  }
  return complex_components;
}

std::vector<std::vector<int64_t>> order_reference_clean_boundary_loops(
    const std::unordered_map<int64_t, std::vector<int64_t>> &adjacency,
    const std::vector<DirectedBoundaryEdge> &boundary_edges) {
  std::vector<std::vector<int64_t>> loops;
  std::unordered_set<mesh_common::EdgeKey, mesh_common::EdgeKeyHash> visited_edges;
  visited_edges.reserve(boundary_edges.size());
  for (const DirectedBoundaryEdge &raw_edge : boundary_edges) {
    const mesh_common::EdgeKey first_edge = mesh_common::edge_key(raw_edge.start, raw_edge.end);
    if (visited_edges.contains(first_edge)) {
      continue;
    }

    const int64_t start = raw_edge.start;
    int64_t previous = -1;
    int64_t current = start;
    std::vector<int64_t> loop;
    std::unordered_set<int64_t> loop_seen;
    std::unordered_set<mesh_common::EdgeKey, mesh_common::EdgeKeyHash> component_edges;
    bool clean = true;
    while (true) {
      const auto found = adjacency.find(current);
      const bool current_seen = loop_seen.contains(current);
      if (found == adjacency.end() || found->second.size() != 2 || current_seen) {
        clean = current == start && loop.size() >= 3;
        break;
      }
      loop.push_back(current);
      loop_seen.insert(current);
      int64_t next = -1;
      for (int64_t candidate : found->second) {
        if (candidate != previous) {
          next = candidate;
          break;
        }
      }
      if (next < 0) {
        clean = false;
        break;
      }
      component_edges.insert(mesh_common::edge_key(current, next));
      previous = current;
      current = next;
      if (current == start) {
        clean = loop.size() >= 3;
        break;
      }
    }
    for (const mesh_common::EdgeKey &edge : component_edges) {
      visited_edges.insert(edge);
    }
    if (clean &&
        std::all_of(loop.begin(), loop.end(), [&](int64_t vertex) {
          const auto found = adjacency.find(vertex);
          return found != adjacency.end() && found->second.size() == 2;
        })) {
      loops.push_back(std::move(loop));
    }
  }
  return loops;
}

std::vector<std::array<int64_t, 3>> reference_centroid_fan_patch(
    const std::vector<int64_t> &loop,
    int64_t center_vertex_id,
    const std::unordered_set<DirectedEdgeKey, DirectedEdgeKeyHash> &directed_boundary) {
  int64_t forward_votes = 0;
  for (size_t index = 0; index < loop.size(); ++index) {
    const int64_t start = loop[index];
    const int64_t end = loop[(index + 1) % loop.size()];
    if (directed_boundary.contains(DirectedEdgeKey{start, end})) {
      forward_votes += 1;
    }
  }
  const bool reverse_boundary = forward_votes >= static_cast<int64_t>(loop.size()) - forward_votes;
  std::vector<std::array<int64_t, 3>> patch_faces;
  patch_faces.reserve(loop.size());
  for (size_t index = 0; index < loop.size(); ++index) {
    const int64_t start = loop[index];
    const int64_t end = loop[(index + 1) % loop.size()];
    if (reverse_boundary) {
      patch_faces.push_back({center_vertex_id, end, start});
    } else {
      patch_faces.push_back({center_vertex_id, start, end});
    }
  }
  return patch_faces;
}

double orient2d(const Point2 &a, const Point2 &b, const Point2 &c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

double polygon_signed_area(const std::vector<Point2> &points) {
  double area = 0.0;
  for (size_t index = 0; index < points.size(); ++index) {
    const Point2 &a = points[index];
    const Point2 &b = points[(index + 1) % points.size()];
    area += a.x * b.y - b.x * a.y;
  }
  return 0.5 * area;
}

bool point_in_or_on_oriented_triangle(
    const Point2 &a,
    const Point2 &b,
    const Point2 &c,
    const Point2 &point,
    double sign,
    double eps) {
  const double ab = orient2d(a, b, point) * sign;
  const double bc = orient2d(b, c, point) * sign;
  const double ca = orient2d(c, a, point) * sign;
  return ab >= -eps && bc >= -eps && ca >= -eps;
}

std::vector<Point2> project_loop_to_stable_plane(
    const mesh_common::MeshData &mesh,
    const std::vector<int64_t> &loop) {
  std::array<double, 3> normal{0.0, 0.0, 0.0};
  for (size_t index = 0; index < loop.size(); ++index) {
    const auto &current = mesh.vertices[static_cast<size_t>(loop[index])];
    const auto &next = mesh.vertices[static_cast<size_t>(loop[(index + 1) % loop.size()])];
    normal[0] += (static_cast<double>(current[1]) - static_cast<double>(next[1]))
        * (static_cast<double>(current[2]) + static_cast<double>(next[2]));
    normal[1] += (static_cast<double>(current[2]) - static_cast<double>(next[2]))
        * (static_cast<double>(current[0]) + static_cast<double>(next[0]));
    normal[2] += (static_cast<double>(current[0]) - static_cast<double>(next[0]))
        * (static_cast<double>(current[1]) + static_cast<double>(next[1]));
  }
  const double normal_length_sq = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
  if (!std::isfinite(normal_length_sq) || normal_length_sq <= 1e-24) {
    return {};
  }

  int drop_axis = 0;
  if (std::abs(normal[1]) > std::abs(normal[drop_axis])) {
    drop_axis = 1;
  }
  if (std::abs(normal[2]) > std::abs(normal[drop_axis])) {
    drop_axis = 2;
  }

  std::vector<Point2> projected;
  projected.reserve(loop.size());
  for (const int64_t vertex_id : loop) {
    const auto &vertex = mesh.vertices[static_cast<size_t>(vertex_id)];
    if (drop_axis == 0) {
      projected.push_back(Point2{static_cast<double>(vertex[1]), static_cast<double>(vertex[2])});
    } else if (drop_axis == 1) {
      projected.push_back(Point2{static_cast<double>(vertex[0]), static_cast<double>(vertex[2])});
    } else {
      projected.push_back(Point2{static_cast<double>(vertex[0]), static_cast<double>(vertex[1])});
    }
  }
  return projected;
}

std::vector<std::array<int64_t, 3>> triangulate_loop_patch(
    const mesh_common::MeshData &mesh,
    const std::vector<int64_t> &loop) {
  if (loop.size() < 3) {
    return {};
  }

  std::vector<Point2> projected = project_loop_to_stable_plane(mesh, loop);
  if (projected.size() != loop.size()) {
    return {};
  }
  double min_x = std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  for (const Point2 &point : projected) {
    min_x = std::min(min_x, point.x);
    min_y = std::min(min_y, point.y);
    max_x = std::max(max_x, point.x);
    max_y = std::max(max_y, point.y);
  }
  const double span = std::max(max_x - min_x, max_y - min_y);
  const double eps = std::max(1e-12, span * span * 1e-12);
  const double area = polygon_signed_area(projected);
  if (!std::isfinite(area) || std::abs(area) <= eps) {
    return {};
  }
  const double sign = area >= 0.0 ? 1.0 : -1.0;

  std::vector<size_t> remaining(loop.size());
  std::iota(remaining.begin(), remaining.end(), 0);
  std::vector<std::array<int64_t, 3>> patch_faces;
  patch_faces.reserve(loop.size() - 2);

  auto make_face = [&](size_t previous, size_t current, size_t next) {
    if (sign > 0.0) {
      return std::array<int64_t, 3>{loop[previous], loop[current], loop[next]};
    }
    return std::array<int64_t, 3>{loop[previous], loop[next], loop[current]};
  };

  size_t guard = loop.size() * loop.size();
  while (remaining.size() > 3 && guard > 0) {
    guard -= 1;
    bool clipped = false;
    for (size_t position = 0; position < remaining.size(); ++position) {
      const size_t previous = remaining[(position + remaining.size() - 1) % remaining.size()];
      const size_t current = remaining[position];
      const size_t next = remaining[(position + 1) % remaining.size()];
      const double turn = orient2d(projected[previous], projected[current], projected[next]) * sign;
      if (turn <= eps) {
        continue;
      }

      bool contains_other_point = false;
      for (const size_t candidate : remaining) {
        if (candidate == previous || candidate == current || candidate == next) {
          continue;
        }
        if (point_in_or_on_oriented_triangle(
                projected[previous],
                projected[current],
                projected[next],
                projected[candidate],
                sign,
                eps)) {
          contains_other_point = true;
          break;
        }
      }
      if (contains_other_point) {
        continue;
      }

      patch_faces.push_back(make_face(previous, current, next));
      remaining.erase(remaining.begin() + static_cast<std::ptrdiff_t>(position));
      clipped = true;
      break;
    }
    if (!clipped) {
      return {};
    }
  }
  if (remaining.size() != 3) {
    return {};
  }
  const double final_turn =
      orient2d(projected[remaining[0]], projected[remaining[1]], projected[remaining[2]]) * sign;
  if (final_turn <= eps) {
    return {};
  }
  patch_faces.push_back(make_face(remaining[0], remaining[1], remaining[2]));
  return patch_faces;
}

std::vector<std::vector<std::array<int64_t, 3>>> triangulate_loop_patch_variants(
    const mesh_common::MeshData &mesh,
    const std::vector<int64_t> &loop,
    size_t max_variants) {
  std::vector<std::vector<std::array<int64_t, 3>>> variants;
  if (loop.size() < 3 || max_variants == 0) {
    return variants;
  }

  std::vector<Point2> projected = project_loop_to_stable_plane(mesh, loop);
  if (projected.size() != loop.size()) {
    return variants;
  }
  double min_x = std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  for (const Point2 &point : projected) {
    min_x = std::min(min_x, point.x);
    min_y = std::min(min_y, point.y);
    max_x = std::max(max_x, point.x);
    max_y = std::max(max_y, point.y);
  }
  const double span = std::max(max_x - min_x, max_y - min_y);
  const double eps = std::max(1e-12, span * span * 1e-12);
  const double area = polygon_signed_area(projected);
  if (!std::isfinite(area) || std::abs(area) <= eps) {
    return variants;
  }
  const double sign = area >= 0.0 ? 1.0 : -1.0;

  auto make_face = [&](size_t previous, size_t current, size_t next) {
    if (sign > 0.0) {
      return std::array<int64_t, 3>{loop[previous], loop[current], loop[next]};
    }
    return std::array<int64_t, 3>{loop[previous], loop[next], loop[current]};
  };

  auto is_ear = [&](const std::vector<size_t> &remaining, size_t position) {
    const size_t previous = remaining[(position + remaining.size() - 1) % remaining.size()];
    const size_t current = remaining[position];
    const size_t next = remaining[(position + 1) % remaining.size()];
    const double turn = orient2d(projected[previous], projected[current], projected[next]) * sign;
    if (turn <= eps) {
      return false;
    }
    for (const size_t candidate : remaining) {
      if (candidate == previous || candidate == current || candidate == next) {
        continue;
      }
      if (point_in_or_on_oriented_triangle(
              projected[previous],
              projected[current],
              projected[next],
              projected[candidate],
              sign,
              eps)) {
        return false;
      }
    }
    return true;
  };

  std::set<std::vector<std::array<int64_t, 3>>> seen_variants;
  std::function<void(std::vector<size_t> &, std::vector<std::array<int64_t, 3>> &)> enumerate =
      [&](std::vector<size_t> &remaining, std::vector<std::array<int64_t, 3>> &current) {
        if (variants.size() >= max_variants) {
          return;
        }
        if (remaining.size() == 3) {
          const double final_turn =
              orient2d(projected[remaining[0]], projected[remaining[1]], projected[remaining[2]]) * sign;
          if (final_turn <= eps) {
            return;
          }
          current.push_back(make_face(remaining[0], remaining[1], remaining[2]));
          std::vector<std::array<int64_t, 3>> canonical = current;
          for (auto &face : canonical) {
            std::sort(face.begin(), face.end());
          }
          std::sort(canonical.begin(), canonical.end());
          if (seen_variants.insert(canonical).second) {
            variants.push_back(current);
          }
          current.pop_back();
          return;
        }

        for (size_t position = 0; position < remaining.size() && variants.size() < max_variants; ++position) {
          if (!is_ear(remaining, position)) {
            continue;
          }
          const size_t previous = remaining[(position + remaining.size() - 1) % remaining.size()];
          const size_t current_index = remaining[position];
          const size_t next = remaining[(position + 1) % remaining.size()];
          current.push_back(make_face(previous, current_index, next));
          const size_t removed = remaining[position];
          remaining.erase(remaining.begin() + static_cast<std::ptrdiff_t>(position));
          enumerate(remaining, current);
          remaining.insert(remaining.begin() + static_cast<std::ptrdiff_t>(position), removed);
          current.pop_back();
        }
      };

  std::vector<size_t> remaining(loop.size());
  std::iota(remaining.begin(), remaining.end(), 0);
  std::vector<std::array<int64_t, 3>> current;
  current.reserve(loop.size() - 2);
  enumerate(remaining, current);
  return variants;
}

std::vector<std::array<int64_t, 3>> centroid_fan_loop_patch(
    const mesh_common::MeshData &mesh,
    const std::vector<int64_t> &loop,
    int64_t center_vertex_id) {
  if (loop.size() < 3) {
    return {};
  }

  double sign = 1.0;
  const std::vector<Point2> projected = project_loop_to_stable_plane(mesh, loop);
  if (projected.size() == loop.size()) {
    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    for (const Point2 &point : projected) {
      min_x = std::min(min_x, point.x);
      min_y = std::min(min_y, point.y);
      max_x = std::max(max_x, point.x);
      max_y = std::max(max_y, point.y);
    }
    const double span = std::max(max_x - min_x, max_y - min_y);
    const double eps = std::max(1e-12, span * span * 1e-12);
    const double area = polygon_signed_area(projected);
    if (std::isfinite(area) && std::abs(area) > eps) {
      sign = area >= 0.0 ? 1.0 : -1.0;
    }
  }

  std::vector<std::array<int64_t, 3>> patch_faces;
  patch_faces.reserve(loop.size());
  for (size_t index = 0; index < loop.size(); ++index) {
    const int64_t current = loop[index];
    const int64_t next = loop[(index + 1) % loop.size()];
    if (sign >= 0.0) {
      patch_faces.push_back({current, next, center_vertex_id});
    } else {
      patch_faces.push_back({current, center_vertex_id, next});
    }
  }
  return patch_faces;
}

double loop_perimeter(const mesh_common::MeshData &mesh, const std::vector<int64_t> &loop) {
  if (loop.size() < 3) {
    return 0.0;
  }
  double perimeter = 0.0;
  for (size_t index = 0; index < loop.size(); ++index) {
    const auto &left = mesh.vertices[static_cast<size_t>(loop[index])];
    const auto &right = mesh.vertices[static_cast<size_t>(loop[(index + 1) % loop.size()])];
    const double dx = static_cast<double>(right[0]) - static_cast<double>(left[0]);
    const double dy = static_cast<double>(right[1]) - static_cast<double>(left[1]);
    const double dz = static_cast<double>(right[2]) - static_cast<double>(left[2]);
    perimeter += std::sqrt(dx * dx + dy * dy + dz * dz);
  }
  return perimeter;
}

std::array<float, 3> loop_boundary_midpoint_mean(const mesh_common::MeshData &mesh, const std::vector<int64_t> &loop) {
  std::array<double, 3> center{0.0, 0.0, 0.0};
  for (size_t index = 0; index < loop.size(); ++index) {
    const auto &left = mesh.vertices[static_cast<size_t>(loop[index])];
    const auto &right = mesh.vertices[static_cast<size_t>(loop[(index + 1) % loop.size()])];
    center[0] += (static_cast<double>(left[0]) + static_cast<double>(right[0])) * 0.5;
    center[1] += (static_cast<double>(left[1]) + static_cast<double>(right[1])) * 0.5;
    center[2] += (static_cast<double>(left[2]) + static_cast<double>(right[2])) * 0.5;
  }
  const double denom = static_cast<double>(std::max<size_t>(1, loop.size()));
  return {
      static_cast<float>(center[0] / denom),
      static_cast<float>(center[1] / denom),
      static_cast<float>(center[2] / denom),
  };
}

ReferenceLoopFillResult fill_reference_clean_boundary_loops(
    const mesh_common::MeshData &input,
    int64_t max_loop_edges,
    double max_perimeter) {
  ReferenceLoopFillResult result;
  result.mesh = input;
  if (input.faces.empty()) {
    return result;
  }
  const std::vector<DirectedBoundaryEdge> boundary_edges = reference_boundary_edges(input);
  result.boundary_edges_before = static_cast<int64_t>(boundary_edges.size());
  if (boundary_edges.empty()) {
    return result;
  }

  const std::unordered_map<int64_t, std::vector<int64_t>> adjacency = boundary_adjacency(boundary_edges);
  result.skipped_complex_components = count_complex_boundary_components(adjacency, boundary_edges);
  const std::vector<std::vector<int64_t>> loops =
      order_reference_clean_boundary_loops(adjacency, boundary_edges);
  result.clean_boundary_loops = static_cast<int64_t>(loops.size());

  std::unordered_set<DirectedEdgeKey, DirectedEdgeKeyHash> directed_boundary;
  directed_boundary.reserve(boundary_edges.size());
  for (const DirectedBoundaryEdge &edge : boundary_edges) {
    directed_boundary.insert(DirectedEdgeKey{edge.start, edge.end});
  }

  std::vector<std::array<float, 3>> new_vertices;
  std::vector<std::array<int64_t, 3>> new_faces;
  new_vertices.reserve(loops.size());
  new_faces.reserve(loops.size() * 4);
  for (const std::vector<int64_t> &loop : loops) {
    if (static_cast<int64_t>(loop.size()) > max_loop_edges) {
      result.skipped_large_loops += 1;
      continue;
    }
    const double perimeter = loop_perimeter(input, loop);
    if (!std::isfinite(perimeter) || perimeter > max_perimeter) {
      result.skipped_large_loops += 1;
      continue;
    }
    const int64_t center_vertex_id = static_cast<int64_t>(input.vertices.size() + new_vertices.size());
    const std::array<float, 3> center = loop_boundary_midpoint_mean(input, loop);
    if (!std::isfinite(center[0]) || !std::isfinite(center[1]) || !std::isfinite(center[2])) {
      result.skipped_large_loops += 1;
      continue;
    }
    std::vector<std::array<int64_t, 3>> patch_faces =
        reference_centroid_fan_patch(loop, center_vertex_id, directed_boundary);
    new_vertices.push_back(center);
    new_faces.insert(new_faces.end(), patch_faces.begin(), patch_faces.end());
  }

  if (!new_vertices.empty()) {
    result.mesh.vertices.reserve(input.vertices.size() + new_vertices.size());
    result.mesh.faces.reserve(input.faces.size() + new_faces.size());
    result.mesh.vertices.insert(result.mesh.vertices.end(), new_vertices.begin(), new_vertices.end());
    result.mesh.faces.insert(result.mesh.faces.end(), new_faces.begin(), new_faces.end());
  }
  result.filled_loops = static_cast<int64_t>(new_vertices.size());
  result.vertices_added = static_cast<int64_t>(new_vertices.size());
  result.faces_added = static_cast<int64_t>(new_faces.size());
  return result;
}

std::vector<std::array<int64_t, 3>> cumesh_centroid_fan_loop_patch(
    const std::vector<int64_t> &loop,
    int64_t center_vertex_id) {
  if (loop.size() < 3) {
    return {};
  }

  std::vector<std::array<int64_t, 3>> patch_faces;
  patch_faces.reserve(loop.size());
  for (size_t index = 0; index < loop.size(); ++index) {
    const mesh_common::EdgeKey edge = mesh_common::edge_key(loop[index], loop[(index + 1) % loop.size()]);
    patch_faces.push_back({edge.b, edge.a, center_vertex_id});
  }
  return patch_faces;
}

void record_patch_rejection(SmallLoopFillResult &result, PatchRejectReason reason) {
  result.loops_rejected += 1;
  if (reason == PatchRejectReason::degenerate) {
    result.loops_rejected_degenerate += 1;
  } else if (reason == PatchRejectReason::duplicate) {
    result.loops_rejected_duplicate += 1;
  } else if (reason == PatchRejectReason::nonmanifold) {
    result.loops_rejected_nonmanifold += 1;
  }
}

PatchRejectReason validate_patch_faces(
    const mesh_common::MeshData &mesh,
    const std::vector<std::array<int64_t, 3>> &patch_faces,
    const std::unordered_set<std::array<int64_t, 3>, FaceKeyHash> &seen_faces,
    const std::unordered_map<mesh_common::EdgeKey, int64_t, mesh_common::EdgeKeyHash> &edge_counts) {
  std::unordered_map<mesh_common::EdgeKey, int64_t, mesh_common::EdgeKeyHash> local_edge_adds;
  local_edge_adds.reserve(patch_faces.size() * 3);
  std::set<std::array<int64_t, 3>> local_seen_faces;
  for (const std::array<int64_t, 3> &face : patch_faces) {
    if (mesh_common::face_degenerate(mesh, face)) {
      return PatchRejectReason::degenerate;
    }
    std::array<int64_t, 3> canonical = face;
    std::sort(canonical.begin(), canonical.end());
    if (seen_faces.contains(canonical) || !local_seen_faces.insert(canonical).second) {
      return PatchRejectReason::duplicate;
    }
    const std::array<mesh_common::EdgeKey, 3> edges{
        mesh_common::edge_key(face[0], face[1]),
        mesh_common::edge_key(face[1], face[2]),
        mesh_common::edge_key(face[2], face[0]),
    };
    for (const auto &edge : edges) {
      const auto found_edge = edge_counts.find(edge);
      const int64_t mesh_edge_count = found_edge == edge_counts.end() ? 0 : found_edge->second;
      const auto found_local = local_edge_adds.find(edge);
      const int64_t local_edge_count = found_local == local_edge_adds.end() ? 0 : found_local->second;
      const int64_t current_count = mesh_edge_count + local_edge_count;
      if (current_count >= 2) {
        return PatchRejectReason::nonmanifold;
      }
      local_edge_adds[edge] += 1;
    }
  }
  return PatchRejectReason::none;
}

std::vector<int64_t> canonical_loop_key(const std::vector<int64_t> &loop) {
  if (loop.empty()) {
    return {};
  }
  auto rotate_min = [](const std::vector<int64_t> &values) {
    const auto min_iter = std::min_element(values.begin(), values.end());
    const size_t min_index = static_cast<size_t>(std::distance(values.begin(), min_iter));
    std::vector<int64_t> rotated;
    rotated.reserve(values.size());
    for (size_t offset = 0; offset < values.size(); ++offset) {
      rotated.push_back(values[(min_index + offset) % values.size()]);
    }
    return rotated;
  };
  std::vector<int64_t> forward = rotate_min(loop);
  std::vector<int64_t> reversed = loop;
  std::reverse(reversed.begin(), reversed.end());
  reversed = rotate_min(reversed);
  return std::lexicographical_compare(reversed.begin(), reversed.end(), forward.begin(), forward.end())
      ? reversed
      : forward;
}

std::vector<int64_t> bounded_path_without_edge(
    const std::unordered_map<int64_t, std::vector<int64_t>> &adjacency,
    int64_t start,
    int64_t goal,
    const mesh_common::EdgeKey &banned_edge,
    int64_t max_path_edges) {
  struct PathState {
    int64_t vertex = 0;
    std::vector<int64_t> path;
  };

  std::deque<PathState> queue;
  queue.push_back(PathState{start, {start}});
  while (!queue.empty()) {
    PathState state = std::move(queue.front());
    queue.pop_front();
    if (static_cast<int64_t>(state.path.size()) - 1 >= max_path_edges) {
      continue;
    }
    const auto found = adjacency.find(state.vertex);
    if (found == adjacency.end()) {
      continue;
    }
    for (const int64_t neighbor : found->second) {
      const mesh_common::EdgeKey edge = mesh_common::edge_key(state.vertex, neighbor);
      if (edge == banned_edge) {
        continue;
      }
      if (neighbor == goal) {
        std::vector<int64_t> result = state.path;
        result.push_back(neighbor);
        return result;
      }
      if (std::find(state.path.begin(), state.path.end(), neighbor) != state.path.end()) {
        continue;
      }
      std::vector<int64_t> next_path = state.path;
      next_path.push_back(neighbor);
      queue.push_back(PathState{neighbor, std::move(next_path)});
    }
  }
  return {};
}

std::vector<std::vector<int64_t>> branched_component_cycles(
    const std::unordered_map<int64_t, std::vector<int64_t>> &adjacency,
    const std::vector<int64_t> &component_vertices,
    int64_t max_loop_edges) {
  std::unordered_set<int64_t> component_set;
  component_set.reserve(component_vertices.size());
  for (const int64_t vertex : component_vertices) {
    component_set.insert(vertex);
  }

  std::vector<mesh_common::EdgeKey> component_edges;
  for (const int64_t vertex : component_vertices) {
    const auto found = adjacency.find(vertex);
    if (found == adjacency.end()) {
      continue;
    }
    for (const int64_t neighbor : found->second) {
      if (vertex < neighbor && component_set.contains(neighbor)) {
        component_edges.push_back(mesh_common::edge_key(vertex, neighbor));
      }
    }
  }
  std::sort(component_edges.begin(), component_edges.end(), [](const auto &left, const auto &right) {
    return left.a == right.a ? left.b < right.b : left.a < right.a;
  });

  std::set<std::vector<int64_t>> seen_cycles;
  std::vector<std::vector<int64_t>> cycles;
  for (const auto &edge : component_edges) {
    std::vector<int64_t> path =
        bounded_path_without_edge(adjacency, edge.b, edge.a, edge, std::max<int64_t>(1, max_loop_edges - 1));
    if (path.empty()) {
      continue;
    }
    std::vector<int64_t> loop;
    loop.reserve(path.size());
    loop.push_back(edge.a);
    for (size_t index = 0; index + 1 < path.size(); ++index) {
      loop.push_back(path[index]);
    }
    if (loop.size() < 3 || static_cast<int64_t>(loop.size()) > max_loop_edges) {
      continue;
    }
    std::vector<int64_t> key = canonical_loop_key(loop);
    if (seen_cycles.insert(key).second) {
      cycles.push_back(std::move(loop));
    }
  }
  std::sort(cycles.begin(), cycles.end(), [](const auto &left, const auto &right) {
    if (left.size() != right.size()) {
      return left.size() < right.size();
    }
    return canonical_loop_key(left) < canonical_loop_key(right);
  });
  return cycles;
}

int64_t effective_repair_cap(int64_t public_cap, int64_t policy_cap) {
  return std::min(std::max<int64_t>(0, public_cap), policy_cap);
}

SmallLoopFillResult fill_small_boundary_loops_single_pass(
    const mesh_common::MeshData &input,
    int64_t max_loop_edges,
    double max_perimeter,
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

  for (auto &[_, neighbors] : adjacency) {
    (void)_;
    std::sort(neighbors.begin(), neighbors.end());
  }

  auto apply_loop_patch = [&](const std::vector<int64_t> &loop) {
    result.loops_considered += 1;
    const int64_t loop_edges = static_cast<int64_t>(loop.size());
    result.loops_edge_count_sum += loop_edges;
    result.loops_edge_count_max = std::max(result.loops_edge_count_max, loop_edges);
    const double perimeter = loop_perimeter(result.mesh, loop);
    if (!std::isfinite(perimeter) || perimeter <= 0.0) {
      result.loops_rejected += 1;
      result.loops_rejected_degenerate += 1;
      return;
    }
    result.loops_perimeter_sum += perimeter;
    result.loops_perimeter_max = std::max(result.loops_perimeter_max, perimeter);
    if (perimeter >= max_perimeter) {
      result.loops_rejected += 1;
      result.loops_rejected_perimeter += 1;
      result.loops_rejected_perimeter_edge_count_sum += loop_edges;
      result.loops_rejected_perimeter_edge_count_max =
          std::max(result.loops_rejected_perimeter_edge_count_max, loop_edges);
      result.loops_rejected_perimeter_sum += perimeter;
      result.loops_rejected_perimeter_min = std::min(result.loops_rejected_perimeter_min, perimeter);
      result.loops_rejected_perimeter_max = std::max(result.loops_rejected_perimeter_max, perimeter);
      return;
    }
    if (loop_edges > max_loop_edges) {
      result.loops_rejected += 1;
      result.loops_rejected_edge_cap += 1;
      return;
    }
    result.loops_centroid_fan_attempted += 1;
    const int64_t center_vertex_id = static_cast<int64_t>(result.mesh.vertices.size());
    const std::array<float, 3> center = loop_boundary_midpoint_mean(result.mesh, loop);
    if (!std::isfinite(center[0]) || !std::isfinite(center[1]) || !std::isfinite(center[2])) {
      result.loops_rejected += 1;
      result.loops_rejected_triangulation += 1;
      return;
    }
    result.mesh.vertices.push_back(center);
    std::vector<std::array<int64_t, 3>> patch_faces = cumesh_centroid_fan_loop_patch(loop, center_vertex_id);
    if (patch_faces.empty()) {
      result.mesh.vertices.pop_back();
      result.loops_rejected += 1;
      result.loops_rejected_triangulation += 1;
      return;
    }
    PatchRejectReason reject_reason = validate_patch_faces(result.mesh, patch_faces, seen_faces, edge_counts);
    if (reject_reason != PatchRejectReason::none) {
      result.mesh.vertices.pop_back();
      record_patch_rejection(result, reject_reason);
      return;
    }
    if (static_cast<int64_t>(patch_faces.size()) > result.face_budget - result.faces_added) {
      result.mesh.vertices.pop_back();
      result.loops_budget_limited += 1;
      return;
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
    result.loops_filled_by_centroid_fan += 1;
  };

  std::unordered_set<int64_t> visited;
  visited.reserve(adjacency.size());
  std::vector<int64_t> stack;
  stack.reserve(adjacency.size());
  std::vector<int64_t> seeds;
  seeds.reserve(adjacency.size());
  for (const auto &[seed, _] : adjacency) {
    (void)_;
    seeds.push_back(seed);
  }
  std::sort(seeds.begin(), seeds.end());
  for (const int64_t seed : seeds) {
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
    if (component_edges < 3) {
      continue;
    }
    if (closed) {
      const std::vector<int64_t> loop = ordered_closed_loop(adjacency, component_vertices);
      if (loop.size() < 3 || static_cast<int64_t>(loop.size()) != component_edges) {
        result.loops_considered += 1;
        result.loops_rejected += 1;
        result.loops_rejected_ordering += 1;
        continue;
      }
      apply_loop_patch(loop);
      continue;
    }
    const std::vector<std::vector<int64_t>> cycles =
        branched_component_cycles(adjacency, component_vertices, max_loop_edges);
    result.branched_cycle_candidates += static_cast<int64_t>(cycles.size());
    for (const std::vector<int64_t> &loop : cycles) {
      const int64_t before_filled = result.loops_filled;
      const int64_t before_budget_limited = result.loops_budget_limited;
      const int64_t before_rejected = result.loops_rejected;
      apply_loop_patch(loop);
      if (result.loops_filled > before_filled) {
        result.branched_cycles_filled += 1;
      } else if (result.loops_budget_limited > before_budget_limited) {
        result.branched_cycles_budget_limited += 1;
      } else if (result.loops_rejected > before_rejected) {
        result.branched_cycles_rejected += 1;
      }
      if (result.faces_added >= result.face_budget) {
        break;
      }
    }
  }
  return result;
}

void accumulate_fill_pass(SmallLoopFillResult &total, SmallLoopFillResult &&pass) {
  total.repair_pass_count += 1;
  total.loops_considered += pass.loops_considered;
  total.loops_filled += pass.loops_filled;
  total.loops_filled_by_ear_clipping += pass.loops_filled_by_ear_clipping;
  total.loops_alternative_triangulation_attempted += pass.loops_alternative_triangulation_attempted;
  total.loops_filled_by_alternative_triangulation += pass.loops_filled_by_alternative_triangulation;
  total.loops_centroid_fan_attempted += pass.loops_centroid_fan_attempted;
  total.loops_filled_by_centroid_fan += pass.loops_filled_by_centroid_fan;
  total.loops_rejected += pass.loops_rejected;
  total.loops_rejected_ordering += pass.loops_rejected_ordering;
  total.loops_rejected_triangulation += pass.loops_rejected_triangulation;
  total.loops_rejected_perimeter += pass.loops_rejected_perimeter;
  total.loops_rejected_edge_cap += pass.loops_rejected_edge_cap;
  total.loops_rejected_fallback_cap += pass.loops_rejected_fallback_cap;
  total.loops_rejected_degenerate += pass.loops_rejected_degenerate;
  total.loops_rejected_duplicate += pass.loops_rejected_duplicate;
  total.loops_rejected_nonmanifold += pass.loops_rejected_nonmanifold;
  total.loops_budget_limited += pass.loops_budget_limited;
  total.loops_edge_count_sum += pass.loops_edge_count_sum;
  total.loops_edge_count_max = std::max(total.loops_edge_count_max, pass.loops_edge_count_max);
  total.loops_perimeter_sum += pass.loops_perimeter_sum;
  total.loops_perimeter_max = std::max(total.loops_perimeter_max, pass.loops_perimeter_max);
  total.loops_rejected_perimeter_edge_count_sum += pass.loops_rejected_perimeter_edge_count_sum;
  total.loops_rejected_perimeter_edge_count_max = std::max(
      total.loops_rejected_perimeter_edge_count_max,
      pass.loops_rejected_perimeter_edge_count_max);
  total.loops_rejected_perimeter_sum += pass.loops_rejected_perimeter_sum;
  total.loops_rejected_perimeter_min =
      std::min(total.loops_rejected_perimeter_min, pass.loops_rejected_perimeter_min);
  total.loops_rejected_perimeter_max =
      std::max(total.loops_rejected_perimeter_max, pass.loops_rejected_perimeter_max);
  total.branched_cycle_candidates += pass.branched_cycle_candidates;
  total.branched_cycles_filled += pass.branched_cycles_filled;
  total.branched_cycles_rejected += pass.branched_cycles_rejected;
  total.branched_cycles_budget_limited += pass.branched_cycles_budget_limited;
  total.faces_added += pass.faces_added;
  total.mesh = std::move(pass.mesh);
}

SmallLoopFillResult fill_small_boundary_loops(
    const mesh_common::MeshData &input,
    int64_t max_loop_edges,
    double max_perimeter,
    int64_t face_budget) {
  SmallLoopFillResult total;
  total.mesh = input;
  total.face_budget = std::max<int64_t>(0, face_budget);
  if (total.face_budget <= 0 || total.mesh.faces.empty()) {
    return total;
  }
  for (int64_t pass_index = 0; pass_index < kSmallBoundaryLoopRepairMaxPasses; ++pass_index) {
    const int64_t remaining_budget = total.face_budget - total.faces_added;
    if (remaining_budget <= 0) {
      break;
    }
    SmallLoopFillResult pass =
        fill_small_boundary_loops_single_pass(total.mesh, max_loop_edges, max_perimeter, remaining_budget);
    const int64_t pass_faces_added = pass.faces_added;
    accumulate_fill_pass(total, std::move(pass));
    if (pass_faces_added <= 0) {
      break;
    }
  }
  return total;
}

int64_t initial_grid_resolution(int64_t target_faces) {
  const double resolution = std::ceil(std::sqrt(std::max<double>(2.0, static_cast<double>(target_faces) * 0.5)));
  return std::max<int64_t>(2, static_cast<int64_t>(resolution));
}

// ---------------------------------------------------------------------------
// Native QEM (quadric-error-metric) edge-collapse simplifier.
//
// Decimates a closed manifold toward a target face count while preserving
// closed-manifold topology by construction: a collapse is applied only when
// every validity guard (boundary lock, link condition, vertex-fan/pinch,
// low-valence/fold, normal-flip, degeneracy) passes.  Determinism is mandatory,
// so every container whose iteration order can influence the collapse ORDER is
// ordered (sorted std::vector / std::map / std::set) — never std::unordered_*.
// See DESIGN.md for the authoritative mechanism.
// ---------------------------------------------------------------------------

// Tunables (fixed thresholds — deterministic across builds, no env reads).
constexpr double kQemLambdaEdgeLength = 1.0e-3;   // length regularizer (reorder only)
constexpr double kQemLambdaSkinny = 1.0e-3;       // sliver penalty (reorder only)
constexpr double kQemNormalFlipCosThreshold = 0.2;  // positive cos(theta_min) guard
constexpr double kQemMinTriangleArea2 = 1.0e-12;  // post-collapse area*2 hard reject
constexpr double kQemMinTriangleAspect = 1.0e-4;  // post-collapse aspect hard reject
constexpr int64_t kQemTetrahedronFloorFaces = 4;  // never decimate below a tetrahedron

// Ordered comparator for EdgeKey so it can key std::map / std::set (EdgeKey has
// no operator<). Lexicographic (a, b) — deterministic, ASLR-independent.
struct EdgeKeyLess {
  bool operator()(const mesh_common::EdgeKey &left, const mesh_common::EdgeKey &right) const {
    if (left.a != right.a) {
      return left.a < right.a;
    }
    return left.b < right.b;
  }
};

struct QemEdgeCost {
  double cost = 0.0;
  int64_t edge_id = 0;
  int64_t version = 0;
};

// Min-heap ordering: cost ASC, edge_id ASC, version DESC.  std::priority_queue
// is a MAX-heap on operator<, so this comparator returns true when `left` should
// pop AFTER `right` (i.e. when left is the "greater"/lower-priority element).
struct QemEdgeCostGreater {
  bool operator()(const QemEdgeCost &left, const QemEdgeCost &right) const {
    if (left.cost != right.cost) {
      return left.cost > right.cost;  // smaller cost pops first
    }
    if (left.edge_id != right.edge_id) {
      return left.edge_id > right.edge_id;  // smaller edge_id pops first
    }
    return left.version < right.version;  // newer version pops first
  }
};

struct QemEdge {
  int64_t a = 0;
  int64_t b = 0;
  int64_t version = 0;
  bool alive = false;
};

// Solve the 3x3 system A x = -b (with A = upper-left 3x3 of Q, b = Q[0..2][3])
// to find the optimal collapse position.  Returns false when ill-conditioned.
bool solve_optimal_vertex(const Quadric &quadric, std::array<float, 3> &out) {
  const double a00 = quadric.values[0];
  const double a01 = quadric.values[1];
  const double a02 = quadric.values[2];
  const double a11 = quadric.values[5];
  const double a12 = quadric.values[6];
  const double a22 = quadric.values[10];
  const double b0 = quadric.values[3];
  const double b1 = quadric.values[7];
  const double b2 = quadric.values[11];

  const double det = a00 * (a11 * a22 - a12 * a12)
      - a01 * (a01 * a22 - a12 * a02)
      + a02 * (a01 * a12 - a11 * a02);
  if (!std::isfinite(det) || std::fabs(det) <= 1e-12) {
    return false;
  }
  const double inv_det = 1.0 / det;
  // Cofactor inverse of symmetric A, applied to -b.
  const double c00 = (a11 * a22 - a12 * a12);
  const double c01 = (a02 * a12 - a01 * a22);
  const double c02 = (a01 * a12 - a02 * a11);
  const double c11 = (a00 * a22 - a02 * a02);
  const double c12 = (a02 * a01 - a00 * a12);
  const double c22 = (a00 * a11 - a01 * a01);
  const double x = -(c00 * b0 + c01 * b1 + c02 * b2) * inv_det;
  const double y = -(c01 * b0 + c11 * b1 + c12 * b2) * inv_det;
  const double z = -(c02 * b0 + c12 * b1 + c22 * b2) * inv_det;
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
    return false;
  }
  out = {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
  return true;
}

// Choose the collapse target position for edge (a,b) and return its quadric
// error.  Falls back to the best of {midpoint,a,b} when the system is singular,
// with explicit tiebreak (error -> distance-to-midpoint -> fixed candidate
// order) mirroring cluster_mesh's representative selection.
double qem_collapse_target(
    const Quadric &combined,
    const std::array<float, 3> &va,
    const std::array<float, 3> &vb,
    std::array<float, 3> &target) {
  std::array<float, 3> solved{};
  if (solve_optimal_vertex(combined, solved)) {
    target = solved;
    const double error = evaluate_quadric(combined, solved);
    if (std::isfinite(error)) {
      return error;
    }
  }
  const std::array<float, 3> midpoint{
      0.5f * (va[0] + vb[0]),
      0.5f * (va[1] + vb[1]),
      0.5f * (va[2] + vb[2]),
  };
  const std::array<std::array<float, 3>, 3> candidates{midpoint, va, vb};
  double best_error = std::numeric_limits<double>::infinity();
  double best_distance = std::numeric_limits<double>::infinity();
  int best_index = -1;
  for (int index = 0; index < 3; ++index) {
    const std::array<float, 3> &candidate = candidates[static_cast<size_t>(index)];
    double error = evaluate_quadric(combined, candidate);
    if (!std::isfinite(error)) {
      error = std::numeric_limits<double>::infinity();
    }
    const double dx = static_cast<double>(candidate[0]) - static_cast<double>(midpoint[0]);
    const double dy = static_cast<double>(candidate[1]) - static_cast<double>(midpoint[1]);
    const double dz = static_cast<double>(candidate[2]) - static_cast<double>(midpoint[2]);
    const double distance = dx * dx + dy * dy + dz * dz;
    if (error < best_error - 1e-18 ||
        (std::fabs(error - best_error) <= 1e-18 && distance < best_distance - 1e-18) ||
        (std::fabs(error - best_error) <= 1e-18 && std::fabs(distance - best_distance) <= 1e-18 &&
         best_index < 0)) {
      best_error = error;
      best_distance = distance;
      best_index = index;
    }
  }
  target = candidates[static_cast<size_t>(std::max(0, best_index))];
  return std::isfinite(best_error) ? best_error : 0.0;
}

struct QemSimplifyResult {
  mesh_common::MeshData mesh;
  int64_t collapses_applied = 0;
  int64_t collapses_rejected_by_guard = 0;
  double geometric_error_sum = 0.0;
  double geometric_error_max = 0.0;
  bool target_reached = false;
};

class QemSimplifier {
 public:
  explicit QemSimplifier(const mesh_common::MeshData &input)
      : vertices_(input.vertices), live_faces_(static_cast<int64_t>(input.faces.size())) {
    faces_.reserve(input.faces.size());
    face_alive_.reserve(input.faces.size());
    for (const auto &face : input.faces) {
      faces_.push_back(face);
      face_alive_.push_back(true);
    }
    quadrics_ = vertex_plane_quadrics(input);
    build_topology();
  }

  QemSimplifyResult run(int64_t target_faces) {
    QemSimplifyResult result;
    while (live_faces_ > target_faces && live_faces_ > kQemTetrahedronFloorFaces) {
      QemEdgeCost top;
      if (!pop_valid_edge(top)) {
        break;  // no valid collapse remains (stall / exhausted)
      }
      double applied_error = 0.0;
      if (collapse_edge(top.edge_id, applied_error)) {
        result.collapses_applied += 1;
        result.geometric_error_sum += applied_error;
        result.geometric_error_max = std::max(result.geometric_error_max, applied_error);
        live_faces_ -= 2;
      } else {
        result.collapses_rejected_by_guard += 1;
      }
      maybe_compact_heap();
    }
    result.mesh = emit_mesh();
    result.target_reached = static_cast<int64_t>(result.mesh.faces.size()) <= target_faces;
    return result;
  }

 private:
  std::vector<std::array<float, 3>> vertices_;
  std::vector<std::array<int64_t, 3>> faces_;
  std::vector<bool> face_alive_;
  std::vector<Quadric> quadrics_;
  int64_t live_faces_ = 0;  // O(1) maintained; decremented by 2 per successful collapse.

  // Ordered adjacency (determinism-critical):
  //  - vertex_faces_: per-vertex sorted incident-face indices.
  //  - edge_index_: EdgeKey -> edge_id in an ordered std::map.
  //  - edges_: edge records indexed by edge_id.
  std::vector<std::set<int64_t>> vertex_faces_;
  std::map<mesh_common::EdgeKey, int64_t, EdgeKeyLess> edge_index_;
  std::vector<QemEdge> edges_;
  std::vector<bool> boundary_vertex_;

  std::priority_queue<QemEdgeCost, std::vector<QemEdgeCost>, QemEdgeCostGreater> heap_;

  int64_t live_face_count() const {
    int64_t count = 0;
    for (const bool alive : face_alive_) {
      if (alive) {
        ++count;
      }
    }
    return count;
  }

  int64_t live_edge_count() const {
    int64_t count = 0;
    for (const QemEdge &edge : edges_) {
      if (edge.alive) {
        ++count;
      }
    }
    return count;
  }

  int64_t edge_id_for(int64_t u, int64_t v) {
    const mesh_common::EdgeKey key = mesh_common::edge_key(u, v);
    auto found = edge_index_.find(key);
    if (found != edge_index_.end()) {
      return found->second;
    }
    const int64_t id = static_cast<int64_t>(edges_.size());
    edges_.push_back(QemEdge{key.a, key.b, 0, true});
    edge_index_.emplace(key, id);
    return id;
  }

  void build_topology() {
    vertex_faces_.assign(vertices_.size(), {});
    for (int64_t fi = 0; fi < static_cast<int64_t>(faces_.size()); ++fi) {
      if (!face_alive_[static_cast<size_t>(fi)]) {
        continue;
      }
      const auto &face = faces_[static_cast<size_t>(fi)];
      for (int c = 0; c < 3; ++c) {
        vertex_faces_[static_cast<size_t>(face[c])].insert(fi);
      }
      edge_id_for(face[0], face[1]);
      edge_id_for(face[1], face[2]);
      edge_id_for(face[2], face[0]);
    }
    // Boundary vertices: any endpoint of an edge used by exactly one live face.
    boundary_vertex_.assign(vertices_.size(), false);
    std::map<mesh_common::EdgeKey, int64_t, EdgeKeyLess> counts;
    for (int64_t fi = 0; fi < static_cast<int64_t>(faces_.size()); ++fi) {
      if (!face_alive_[static_cast<size_t>(fi)]) {
        continue;
      }
      const auto &face = faces_[static_cast<size_t>(fi)];
      counts[mesh_common::edge_key(face[0], face[1])] += 1;
      counts[mesh_common::edge_key(face[1], face[2])] += 1;
      counts[mesh_common::edge_key(face[2], face[0])] += 1;
    }
    for (const auto &entry : counts) {
      if (entry.second != 2) {
        boundary_vertex_[static_cast<size_t>(entry.first.a)] = true;
        boundary_vertex_[static_cast<size_t>(entry.first.b)] = true;
      }
    }
    // Seed the heap from the live edges in edge_id order (deterministic).
    for (int64_t id = 0; id < static_cast<int64_t>(edges_.size()); ++id) {
      if (edges_[static_cast<size_t>(id)].alive) {
        push_edge_cost(id);
      }
    }
  }

  void push_edge_cost(int64_t edge_id) {
    const QemEdge &edge = edges_[static_cast<size_t>(edge_id)];
    if (!edge.alive) {
      return;
    }
    Quadric combined = quadrics_[static_cast<size_t>(edge.a)];
    add_quadric(combined, quadrics_[static_cast<size_t>(edge.b)]);
    const std::array<float, 3> &va = vertices_[static_cast<size_t>(edge.a)];
    const std::array<float, 3> &vb = vertices_[static_cast<size_t>(edge.b)];
    std::array<float, 3> target{};
    double cost = qem_collapse_target(combined, va, vb, target);
    if (!std::isfinite(cost)) {
      cost = std::numeric_limits<double>::infinity();
    }
    const double dx = static_cast<double>(va[0]) - static_cast<double>(vb[0]);
    const double dy = static_cast<double>(va[1]) - static_cast<double>(vb[1]);
    const double dz = static_cast<double>(va[2]) - static_cast<double>(vb[2]);
    const double len2 = dx * dx + dy * dy + dz * dz;
    cost += kQemLambdaEdgeLength * len2;
    cost += kQemLambdaSkinny * skinny_penalty(edge.a, edge.b);
    heap_.push(QemEdgeCost{cost, edge_id, edge.version});
  }

  // Sliver penalty proportional to inverse aspect of the worst incident face
  // (reorder only; never substitutes a guard).
  double skinny_penalty(int64_t u, int64_t v) const {
    double worst = 0.0;
    std::set<int64_t> incident = vertex_faces_[static_cast<size_t>(u)];
    for (const int64_t fi : vertex_faces_[static_cast<size_t>(v)]) {
      incident.insert(fi);
    }
    for (const int64_t fi : incident) {
      if (!face_alive_[static_cast<size_t>(fi)]) {
        continue;
      }
      const double aspect = triangle_inverse_aspect(faces_[static_cast<size_t>(fi)]);
      worst = std::max(worst, aspect);
    }
    return worst;
  }

  double triangle_inverse_aspect(const std::array<int64_t, 3> &face) const {
    const auto &a = vertices_[static_cast<size_t>(face[0])];
    const auto &b = vertices_[static_cast<size_t>(face[1])];
    const auto &c = vertices_[static_cast<size_t>(face[2])];
    auto dist2 = [](const std::array<float, 3> &p, const std::array<float, 3> &q) {
      const double dx = static_cast<double>(p[0]) - static_cast<double>(q[0]);
      const double dy = static_cast<double>(p[1]) - static_cast<double>(q[1]);
      const double dz = static_cast<double>(p[2]) - static_cast<double>(q[2]);
      return dx * dx + dy * dy + dz * dz;
    };
    const double longest = std::sqrt(std::max({dist2(a, b), dist2(b, c), dist2(c, a)}));
    const double area2 = triangle_area2_local(face);
    if (area2 <= 1e-18) {
      return 1.0e6;
    }
    return (longest * longest) / area2;
  }

  double triangle_area2_local(const std::array<int64_t, 3> &face) const {
    const auto &a = vertices_[static_cast<size_t>(face[0])];
    const auto &b = vertices_[static_cast<size_t>(face[1])];
    const auto &c = vertices_[static_cast<size_t>(face[2])];
    const double ab[3] = {
        static_cast<double>(b[0]) - static_cast<double>(a[0]),
        static_cast<double>(b[1]) - static_cast<double>(a[1]),
        static_cast<double>(b[2]) - static_cast<double>(a[2]),
    };
    const double ac[3] = {
        static_cast<double>(c[0]) - static_cast<double>(a[0]),
        static_cast<double>(c[1]) - static_cast<double>(a[1]),
        static_cast<double>(c[2]) - static_cast<double>(a[2]),
    };
    const double cross[3] = {
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    };
    return std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
  }

  std::array<double, 3> triangle_normal(
      const std::array<float, 3> &a,
      const std::array<float, 3> &b,
      const std::array<float, 3> &c) const {
    const double ab[3] = {
        static_cast<double>(b[0]) - static_cast<double>(a[0]),
        static_cast<double>(b[1]) - static_cast<double>(a[1]),
        static_cast<double>(b[2]) - static_cast<double>(a[2]),
    };
    const double ac[3] = {
        static_cast<double>(c[0]) - static_cast<double>(a[0]),
        static_cast<double>(c[1]) - static_cast<double>(a[1]),
        static_cast<double>(c[2]) - static_cast<double>(a[2]),
    };
    std::array<double, 3> normal{
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    };
    const double len = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    if (!std::isfinite(len) || len <= 1e-18) {
      return {0.0, 0.0, 0.0};
    }
    normal[0] /= len;
    normal[1] /= len;
    normal[2] /= len;
    return normal;
  }

  bool pop_valid_edge(QemEdgeCost &out) {
    while (!heap_.empty()) {
      QemEdgeCost candidate = heap_.top();
      heap_.pop();
      const QemEdge &edge = edges_[static_cast<size_t>(candidate.edge_id)];
      if (!edge.alive || edge.version != candidate.version) {
        continue;  // stale lazy entry
      }
      out = candidate;
      return true;
    }
    return false;
  }

  void maybe_compact_heap() {
    const int64_t live = live_edge_count();
    if (static_cast<int64_t>(heap_.size()) <= 3 * std::max<int64_t>(1, live)) {
      return;
    }
    decltype(heap_) rebuilt;
    for (int64_t id = 0; id < static_cast<int64_t>(edges_.size()); ++id) {
      if (edges_[static_cast<size_t>(id)].alive) {
        const QemEdge &edge = edges_[static_cast<size_t>(id)];
        Quadric combined = quadrics_[static_cast<size_t>(edge.a)];
        add_quadric(combined, quadrics_[static_cast<size_t>(edge.b)]);
        const std::array<float, 3> &va = vertices_[static_cast<size_t>(edge.a)];
        const std::array<float, 3> &vb = vertices_[static_cast<size_t>(edge.b)];
        std::array<float, 3> target{};
        double cost = qem_collapse_target(combined, va, vb, target);
        if (!std::isfinite(cost)) {
          cost = std::numeric_limits<double>::infinity();
        }
        const double dx = static_cast<double>(va[0]) - static_cast<double>(vb[0]);
        const double dy = static_cast<double>(va[1]) - static_cast<double>(vb[1]);
        const double dz = static_cast<double>(va[2]) - static_cast<double>(vb[2]);
        cost += kQemLambdaEdgeLength * (dx * dx + dy * dy + dz * dz);
        cost += kQemLambdaSkinny * skinny_penalty(edge.a, edge.b);
        rebuilt.push(QemEdgeCost{cost, id, edge.version});
      }
    }
    heap_ = std::move(rebuilt);
  }

  // Live faces incident to vertex v (those that are alive and reference v).
  std::vector<int64_t> live_incident_faces(int64_t v) const {
    std::vector<int64_t> out;
    for (const int64_t fi : vertex_faces_[static_cast<size_t>(v)]) {
      if (face_alive_[static_cast<size_t>(fi)]) {
        out.push_back(fi);
      }
    }
    return out;  // already ascending: std::set iteration
  }

  // Common one-ring neighbours of a and b (vertices adjacent to both via a live
  // face), returned sorted/ascending.
  std::vector<int64_t> common_neighbours(int64_t a, int64_t b) const {
    std::set<int64_t> na = one_ring(a);
    std::set<int64_t> nb = one_ring(b);
    std::vector<int64_t> shared;
    std::set_intersection(na.begin(), na.end(), nb.begin(), nb.end(), std::back_inserter(shared));
    return shared;
  }

  std::set<int64_t> one_ring(int64_t v) const {
    std::set<int64_t> ring;
    for (const int64_t fi : vertex_faces_[static_cast<size_t>(v)]) {
      if (!face_alive_[static_cast<size_t>(fi)]) {
        continue;
      }
      const auto &face = faces_[static_cast<size_t>(fi)];
      for (int c = 0; c < 3; ++c) {
        if (face[c] != v) {
          ring.insert(face[c]);
        }
      }
    }
    return ring;
  }

  // The two faces shared by edge (a,b): faces containing both a and b.
  std::vector<int64_t> faces_on_edge(int64_t a, int64_t b) const {
    std::vector<int64_t> out;
    for (const int64_t fi : vertex_faces_[static_cast<size_t>(a)]) {
      if (!face_alive_[static_cast<size_t>(fi)]) {
        continue;
      }
      const auto &face = faces_[static_cast<size_t>(fi)];
      if (face[0] == b || face[1] == b || face[2] == b) {
        out.push_back(fi);
      }
    }
    return out;
  }

  int64_t opposite_vertex(int64_t face_index, int64_t a, int64_t b) const {
    const auto &face = faces_[static_cast<size_t>(face_index)];
    for (int c = 0; c < 3; ++c) {
      if (face[c] != a && face[c] != b) {
        return face[c];
      }
    }
    return -1;
  }

  // Try to collapse edge (a,b) -> a.  Returns true and mutates the mesh when all
  // guards pass; otherwise returns false (no mutation, edge stays alive).
  bool collapse_edge(int64_t edge_id, double &applied_error) {
    const QemEdge edge = edges_[static_cast<size_t>(edge_id)];
    const int64_t a0 = edge.a;
    const int64_t b0 = edge.b;

    // Guard 1: boundary lock (interior-only collapse in v1).
    if (boundary_vertex_[static_cast<size_t>(a0)] || boundary_vertex_[static_cast<size_t>(b0)]) {
      return false;
    }

    const std::vector<int64_t> incident_faces = faces_on_edge(a0, b0);
    if (incident_faces.size() != 2) {
      return false;  // interior manifold edge must have exactly two faces
    }
    const int64_t opp0 = opposite_vertex(incident_faces[0], a0, b0);
    const int64_t opp1 = opposite_vertex(incident_faces[1], a0, b0);
    if (opp0 < 0 || opp1 < 0 || opp0 == opp1) {
      return false;
    }

    // Guard 2: link condition. Common one-ring neighbours of a and b must be
    // EXACTLY the two opposite vertices.
    {
      std::vector<int64_t> common = common_neighbours(a0, b0);
      std::vector<int64_t> expected{std::min(opp0, opp1), std::max(opp0, opp1)};
      if (common != expected) {
        return false;
      }
    }

    // Decide collapse target a := keep, b := removed.  Keep the lower index as
    // the survivor for deterministic output ordering.
    const int64_t keep = std::min(a0, b0);
    const int64_t drop = std::max(a0, b0);

    Quadric combined = quadrics_[static_cast<size_t>(a0)];
    add_quadric(combined, quadrics_[static_cast<size_t>(b0)]);
    std::array<float, 3> target{};
    const double error = qem_collapse_target(
        combined, vertices_[static_cast<size_t>(a0)], vertices_[static_cast<size_t>(b0)], target);

    // Build the candidate surviving faces around `keep` after the collapse.
    // These are all live faces incident to keep OR drop, excluding the two
    // collapsing faces, with `drop` retargeted to `keep`.
    std::set<int64_t> affected;
    for (const int64_t fi : vertex_faces_[static_cast<size_t>(a0)]) {
      if (face_alive_[static_cast<size_t>(fi)]) {
        affected.insert(fi);
      }
    }
    for (const int64_t fi : vertex_faces_[static_cast<size_t>(b0)]) {
      if (face_alive_[static_cast<size_t>(fi)]) {
        affected.insert(fi);
      }
    }
    affected.erase(incident_faces[0]);
    affected.erase(incident_faces[1]);

    std::vector<std::array<int64_t, 3>> new_faces;
    new_faces.reserve(affected.size());
    std::vector<int64_t> affected_ids(affected.begin(), affected.end());
    for (const int64_t fi : affected_ids) {
      std::array<int64_t, 3> face = faces_[static_cast<size_t>(fi)];
      for (int c = 0; c < 3; ++c) {
        if (face[c] == drop) {
          face[c] = keep;
        }
      }
      new_faces.push_back(face);
    }

    // Guard 4a: canonical-face duplicate (fold-over). No two surviving faces in
    // the merged 1-ring may share the same canonical vertex set.
    {
      std::set<std::array<int64_t, 3>> seen;
      for (const auto &face : new_faces) {
        if (face[0] == face[1] || face[1] == face[2] || face[0] == face[2]) {
          return false;  // degenerate retarget
        }
        std::array<int64_t, 3> canonical = mesh_common::canonical_face(face);
        if (!seen.insert(canonical).second) {
          return false;
        }
      }
    }

    // Guards 5 + 6: normal-flip and degeneracy over the merged 1-ring.
    for (size_t index = 0; index < new_faces.size(); ++index) {
      const std::array<int64_t, 3> &pre_face = faces_[static_cast<size_t>(affected_ids[index])];
      const std::array<int64_t, 3> &post_face = new_faces[index];
      const std::array<double, 3> normal_pre = triangle_normal(
          vertices_[static_cast<size_t>(pre_face[0])],
          vertices_[static_cast<size_t>(pre_face[1])],
          vertices_[static_cast<size_t>(pre_face[2])]);
      // Post position: vertex `keep` moves to `target`.
      auto pos = [&](int64_t vid) -> std::array<float, 3> {
        return vid == keep ? target : vertices_[static_cast<size_t>(vid)];
      };
      const std::array<double, 3> normal_post = triangle_normal(
          pos(post_face[0]), pos(post_face[1]), pos(post_face[2]));
      const double dot = normal_pre[0] * normal_post[0] + normal_pre[1] * normal_post[1]
          + normal_pre[2] * normal_post[2];
      // Guard 5: normal-flip — reject when alignment drops below positive thresh.
      if (dot <= kQemNormalFlipCosThreshold) {
        return false;
      }
      // Guard 6: degeneracy — hard reject on tiny area / bad aspect.
      const double area2 = triangle_area2_at(post_face, keep, target);
      if (!std::isfinite(area2) || area2 <= kQemMinTriangleArea2) {
        return false;
      }
      const double aspect_inv = 1.0 / std::max(1e-12, triangle_inverse_aspect_at(post_face, keep, target));
      if (aspect_inv < kQemMinTriangleAspect) {
        return false;
      }
    }

    // Guard 3: vertex-fan / pinch test. After the retarget, the surviving faces
    // incident to `keep` must form a SINGLE edge-connected fan (one umbrella),
    // mirroring the nonmanifold_vertices metric's union-find over shared edges.
    if (!single_fan_after_collapse(new_faces, keep)) {
      return false;
    }

    // All guards passed — apply the collapse.
    apply_collapse(keep, drop, target, incident_faces, affected_ids, new_faces, combined);
    applied_error = std::isfinite(error) ? error : 0.0;
    return true;
  }

  double triangle_area2_at(
      const std::array<int64_t, 3> &face,
      int64_t moved_vertex,
      const std::array<float, 3> &moved_position) const {
    auto pos = [&](int64_t vid) -> std::array<float, 3> {
      return vid == moved_vertex ? moved_position : vertices_[static_cast<size_t>(vid)];
    };
    const std::array<float, 3> a = pos(face[0]);
    const std::array<float, 3> b = pos(face[1]);
    const std::array<float, 3> c = pos(face[2]);
    const double ab[3] = {
        static_cast<double>(b[0]) - static_cast<double>(a[0]),
        static_cast<double>(b[1]) - static_cast<double>(a[1]),
        static_cast<double>(b[2]) - static_cast<double>(a[2]),
    };
    const double ac[3] = {
        static_cast<double>(c[0]) - static_cast<double>(a[0]),
        static_cast<double>(c[1]) - static_cast<double>(a[1]),
        static_cast<double>(c[2]) - static_cast<double>(a[2]),
    };
    const double cross[3] = {
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    };
    return std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
  }

  double triangle_inverse_aspect_at(
      const std::array<int64_t, 3> &face,
      int64_t moved_vertex,
      const std::array<float, 3> &moved_position) const {
    auto pos = [&](int64_t vid) -> std::array<float, 3> {
      return vid == moved_vertex ? moved_position : vertices_[static_cast<size_t>(vid)];
    };
    const std::array<float, 3> a = pos(face[0]);
    const std::array<float, 3> b = pos(face[1]);
    const std::array<float, 3> c = pos(face[2]);
    auto dist2 = [](const std::array<float, 3> &p, const std::array<float, 3> &q) {
      const double dx = static_cast<double>(p[0]) - static_cast<double>(q[0]);
      const double dy = static_cast<double>(p[1]) - static_cast<double>(q[1]);
      const double dz = static_cast<double>(p[2]) - static_cast<double>(q[2]);
      return dx * dx + dy * dy + dz * dz;
    };
    const double longest = std::sqrt(std::max({dist2(a, b), dist2(b, c), dist2(c, a)}));
    const double area2 = triangle_area2_at(face, moved_vertex, moved_position);
    if (area2 <= 1e-18) {
      return 1.0e6;
    }
    return (longest * longest) / area2;
  }

  // Union-find over the candidate surviving faces incident to `keep`, uniting
  // two faces when they share a non-keep vertex (an edge through keep). A single
  // root => one fan; more than one => a pinch (reject).
  bool single_fan_after_collapse(
      const std::vector<std::array<int64_t, 3>> &new_faces,
      int64_t keep) const {
    std::vector<size_t> incident;
    for (size_t index = 0; index < new_faces.size(); ++index) {
      const auto &face = new_faces[index];
      if (face[0] == keep || face[1] == keep || face[2] == keep) {
        incident.push_back(index);
      }
    }
    if (incident.size() < 2) {
      return true;  // 0 or 1 incident face is always a single fan
    }
    const size_t n = incident.size();
    mesh_common::UnionFind uf(n);
    for (size_t a = 0; a < n; ++a) {
      const auto &fa = new_faces[incident[a]];
      for (size_t b = a + 1; b < n; ++b) {
        const auto &fb = new_faces[incident[b]];
        int shared_non_keep = 0;
        for (int ca = 0; ca < 3; ++ca) {
          if (fa[ca] == keep) {
            continue;
          }
          for (int cb = 0; cb < 3; ++cb) {
            if (fb[cb] == keep) {
              continue;
            }
            if (fa[ca] == fb[cb]) {
              ++shared_non_keep;
            }
          }
        }
        if (shared_non_keep >= 1) {
          uf.unite(a, b);
        }
      }
    }
    std::set<size_t> roots;
    for (size_t a = 0; a < n; ++a) {
      roots.insert(uf.find(a));
    }
    return roots.size() == 1;
  }

  void apply_collapse(
      int64_t keep,
      int64_t drop,
      const std::array<float, 3> &target,
      const std::vector<int64_t> &incident_faces,
      const std::vector<int64_t> &affected_ids,
      const std::vector<std::array<int64_t, 3>> &new_faces,
      const Quadric &combined) {
    // Kill the two collapsing faces.
    for (const int64_t fi : incident_faces) {
      face_alive_[static_cast<size_t>(fi)] = false;
      const auto &face = faces_[static_cast<size_t>(fi)];
      for (int c = 0; c < 3; ++c) {
        vertex_faces_[static_cast<size_t>(face[c])].erase(fi);
      }
    }
    // Retarget affected faces drop->keep, maintaining adjacency.
    for (size_t index = 0; index < affected_ids.size(); ++index) {
      const int64_t fi = affected_ids[index];
      std::array<int64_t, 3> &face = faces_[static_cast<size_t>(fi)];
      for (int c = 0; c < 3; ++c) {
        if (face[c] == drop) {
          vertex_faces_[static_cast<size_t>(drop)].erase(fi);
          face[c] = keep;
          vertex_faces_[static_cast<size_t>(keep)].insert(fi);
        }
      }
    }
    // Move the survivor to the optimal position and accumulate quadrics.
    vertices_[static_cast<size_t>(keep)] = target;
    quadrics_[static_cast<size_t>(keep)] = combined;

    // Drop vertex is now isolated.
    vertex_faces_[static_cast<size_t>(drop)].clear();

    // Invalidate every edge touching keep or drop, register new edges around
    // keep, bump versions, and re-push fresh costs. Iterate the merged 1-ring in
    // sorted order for determinism.
    // Invalidate any existing edge whose endpoints touch keep/drop by bumping
    // version (lazy). Drop's incident edges become dead.
    for (auto &entry : edge_index_) {
      const mesh_common::EdgeKey &key = entry.first;
      QemEdge &e = edges_[static_cast<size_t>(entry.second)];
      if (!e.alive) {
        continue;
      }
      if (key.a == drop || key.b == drop) {
        e.alive = false;
        e.version += 1;
      } else if (key.a == keep || key.b == keep) {
        e.alive = false;  // will be re-registered below if still present
        e.version += 1;
      }
    }
    // Re-register live edges around keep from the surviving incident faces.
    std::set<mesh_common::EdgeKey, EdgeKeyLess> live_keep_edges;
    for (const int64_t fi : vertex_faces_[static_cast<size_t>(keep)]) {
      if (!face_alive_[static_cast<size_t>(fi)]) {
        continue;
      }
      const auto &face = faces_[static_cast<size_t>(fi)];
      for (int c = 0; c < 3; ++c) {
        const int64_t u = face[c];
        const int64_t v = face[(c + 1) % 3];
        if (u == keep || v == keep) {
          live_keep_edges.insert(mesh_common::edge_key(u, v));
        }
      }
    }
    for (const mesh_common::EdgeKey &key : live_keep_edges) {
      auto found = edge_index_.find(key);
      int64_t id = -1;
      if (found == edge_index_.end()) {
        id = static_cast<int64_t>(edges_.size());
        edges_.push_back(QemEdge{key.a, key.b, 0, true});
        edge_index_.emplace(key, id);
      } else {
        id = found->second;
        QemEdge &e = edges_[static_cast<size_t>(id)];
        e.alive = true;
        e.version += 1;
      }
      push_edge_cost(id);
    }
  }

  // Compact to a clean MeshData, dropping dead faces and isolated vertices.
  mesh_common::MeshData emit_mesh() const {
    mesh_common::MeshData out;
    out.faces.reserve(faces_.size());
    for (int64_t fi = 0; fi < static_cast<int64_t>(faces_.size()); ++fi) {
      if (face_alive_[static_cast<size_t>(fi)]) {
        out.faces.push_back(faces_[static_cast<size_t>(fi)]);
      }
    }
    out.vertices = vertices_;
    int64_t unreferenced_removed = 0;
    return mesh_common::compact_mesh(out, &unreferenced_removed);
  }
};

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
        "native_topology_aware_quadric_representative_clustering",
        true,
    };
  }
  if (backend == "qem") {
    return BackendSelection{
        "qem",
        "qem",
        "native_qem_edge_collapse",
        true,
        true,
    };
  }
  throw nb::value_error("simplifier backend must be 'spatial-cluster', 'topology-aware', or 'qem'");
}

std::vector<std::string> production_blocker_values(
    const BackendSelection &selection,
    int64_t final_faces,
    bool target_reached) {
  std::vector<std::string> blockers;
  if (!selection.topology_aware) {
    blockers.push_back("preview_backend_tier");
    return blockers;
  }
  if (final_faces <= 0) {
    blockers.push_back("no_faces");
  }
  if (!target_reached) {
    blockers.push_back("target_not_reached");
  }
  if (!selection.is_qem) {
    // qem already provides native edge-collapse simplification; only the
    // narrow-band DC remesh (cleared by export.py post-remesh) remains.
    blockers.push_back("missing_qem_edge_collapse_simplification");
  }
  blockers.push_back("missing_narrow_band_dc_remesh");
  return blockers;
}

nb::list string_list(const std::vector<std::string> &values) {
  nb::list result;
  for (const auto &value : values) {
    result.append(value.c_str());
  }
  return result;
}

void add_backend_stats(
    nb::dict &stats,
    const BackendSelection &selection,
    int64_t final_faces,
    bool target_reached) {
  const std::vector<std::string> blockers = production_blocker_values(selection, final_faces, target_reached);
  const bool production_ready = blockers.empty();
  stats["requested_backend"] = selection.requested;
  stats["backend"] = selection.backend;
  stats["algorithm"] = selection.algorithm;
  stats["quality_tier"] = production_ready ? "production" : (selection.topology_aware ? "production_candidate_blocked" : "geometry_aware_preview");
  stats["production_ready"] = production_ready;
  stats["production_blockers"] = string_list(blockers);
  stats["backend_selection_status"] = "selected";
  stats["backend_selection_reason"] = selection.topology_aware ? "topology_aware_backend_requested" : "preview_backend_requested";
  stats["remesh_backend"] = "not_implemented";
  stats["remesh_equivalence_status"] = "blocked_missing_narrow_band_dc";
  if (!selection.is_qem) {
    // For the qem path these fields are written once with correct final values
    // by build_qem_stats; skip the stale intermediate write here.
    stats["qem_simplification_backend"] = "not_implemented";
    stats["qem_equivalence_status"] = selection.topology_aware
        ? "qem_scored_not_edge_collapse"
        : "not_requested_preview_backend";
  }
  stats["reference_geometry_backend_status"] = production_ready
      ? "reference_ready"
      : "blocked_missing_reference_geometry";
  stats["reference_geometry_blockers"] = string_list(blockers);
}

void add_small_loop_fill_stats(
    nb::dict &stats,
    bool enabled,
    int64_t max_loop_edges,
    double max_perimeter,
    const SmallLoopFillResult &fill) {
  const int64_t effective_branched_cycle_cap =
      enabled ? effective_repair_cap(max_loop_edges, kSmallBoundaryBranchedCycleFillMaxEdges) : 0;
  stats["small_boundary_loop_fill_enabled"] = enabled;
  stats["small_boundary_loop_fill_algorithm"] = kSmallBoundaryLoopFillAlgorithm;
  stats["small_boundary_loop_fill_fallback_algorithm"] = kSmallBoundaryLoopFillFallbackAlgorithm;
  stats["small_boundary_loop_fill_fallback_enabled"] = false;
  stats["small_boundary_loop_fill_fallback_policy_max_edges"] = kSmallBoundaryLoopFillFallbackMaxEdges;
  stats["small_boundary_loop_fill_fallback_max_edges"] = kSmallBoundaryLoopFillFallbackMaxEdges;
  stats["small_boundary_loop_fill_fallback_effective_max_edges"] =
      enabled ? effective_repair_cap(max_loop_edges, kSmallBoundaryLoopFillFallbackMaxEdges) : 0;
  stats["small_boundary_loop_fill_max_perimeter"] = max_perimeter;
  stats["small_boundary_branched_cycle_fill_enabled"] = effective_branched_cycle_cap > 0;
  stats["small_boundary_branched_cycle_fill_policy_max_edges"] = kSmallBoundaryBranchedCycleFillMaxEdges;
  stats["small_boundary_branched_cycle_fill_max_edges"] = kSmallBoundaryBranchedCycleFillMaxEdges;
  stats["small_boundary_branched_cycle_fill_effective_max_edges"] = effective_branched_cycle_cap;
  stats["small_boundary_loop_repair_max_passes"] = kSmallBoundaryLoopRepairMaxPasses;
  stats["small_boundary_loop_repair_pass_count"] = fill.repair_pass_count;
  stats["small_boundary_loop_fill_max_edges"] = max_loop_edges;
  stats["small_boundary_loop_fill_face_budget"] = fill.face_budget;
  stats["small_boundary_loops_considered"] = fill.loops_considered;
  stats["small_boundary_loops_filled"] = fill.loops_filled;
  stats["small_boundary_loops_filled_by_ear_clipping"] = fill.loops_filled_by_ear_clipping;
  stats["small_boundary_loops_alternative_triangulation_attempted"] =
      fill.loops_alternative_triangulation_attempted;
  stats["small_boundary_loops_filled_by_alternative_triangulation"] =
      fill.loops_filled_by_alternative_triangulation;
  stats["small_boundary_loops_centroid_fan_attempted"] = fill.loops_centroid_fan_attempted;
  stats["small_boundary_loops_filled_by_centroid_fan"] = fill.loops_filled_by_centroid_fan;
  stats["small_boundary_loops_rejected"] = fill.loops_rejected;
  stats["small_boundary_loops_rejected_ordering"] = fill.loops_rejected_ordering;
  stats["small_boundary_loops_rejected_triangulation"] = fill.loops_rejected_triangulation;
  stats["small_boundary_loops_rejected_perimeter"] = fill.loops_rejected_perimeter;
  stats["small_boundary_loops_rejected_edge_cap"] = fill.loops_rejected_edge_cap;
  stats["small_boundary_loops_rejected_fallback_cap"] = fill.loops_rejected_fallback_cap;
  stats["small_boundary_loops_rejected_degenerate"] = fill.loops_rejected_degenerate;
  stats["small_boundary_loops_rejected_duplicate"] = fill.loops_rejected_duplicate;
  stats["small_boundary_loops_rejected_nonmanifold"] = fill.loops_rejected_nonmanifold;
  stats["small_boundary_loops_budget_limited"] = fill.loops_budget_limited;
  stats["small_boundary_loops_edge_count_avg"] = fill.loops_considered > 0
      ? static_cast<double>(fill.loops_edge_count_sum) / static_cast<double>(fill.loops_considered)
      : 0.0;
  stats["small_boundary_loops_edge_count_max"] = fill.loops_edge_count_max;
  stats["small_boundary_loops_perimeter_avg"] = fill.loops_considered > 0
      ? fill.loops_perimeter_sum / static_cast<double>(fill.loops_considered)
      : 0.0;
  stats["small_boundary_loops_perimeter_max"] = fill.loops_perimeter_max;
  stats["small_boundary_loops_rejected_perimeter_edge_count_avg"] = fill.loops_rejected_perimeter > 0
      ? static_cast<double>(fill.loops_rejected_perimeter_edge_count_sum) /
            static_cast<double>(fill.loops_rejected_perimeter)
      : 0.0;
  stats["small_boundary_loops_rejected_perimeter_edge_count_max"] =
      fill.loops_rejected_perimeter_edge_count_max;
  stats["small_boundary_loops_rejected_perimeter_avg"] = fill.loops_rejected_perimeter > 0
      ? fill.loops_rejected_perimeter_sum / static_cast<double>(fill.loops_rejected_perimeter)
      : 0.0;
  stats["small_boundary_loops_rejected_perimeter_min"] = fill.loops_rejected_perimeter > 0
      ? fill.loops_rejected_perimeter_min
      : 0.0;
  stats["small_boundary_loops_rejected_perimeter_max"] =
      fill.loops_rejected_perimeter_max;
  stats["small_boundary_branched_cycle_candidates"] = fill.branched_cycle_candidates;
  stats["small_boundary_branched_cycles_filled"] = fill.branched_cycles_filled;
  stats["small_boundary_branched_cycles_rejected"] = fill.branched_cycles_rejected;
  stats["small_boundary_branched_cycles_budget_limited"] = fill.branched_cycles_budget_limited;
  stats["small_boundary_loop_faces_added"] = fill.faces_added;
}

void add_pre_simplify_loop_fill_stats(
    nb::dict &stats,
    bool enabled,
    double max_perimeter,
    const ReferenceLoopFillResult &fill) {
  stats["pre_simplify_hole_fill_enabled"] = enabled;
  stats["pre_simplify_hole_fill_algorithm"] = "reference-clean-boundary-centroid-fan";
  stats["pre_simplify_hole_fill_max_edges"] = kPreSimplifyCleanBoundaryLoopFillMaxEdges;
  stats["pre_simplify_hole_fill_max_perimeter"] = max_perimeter;
  stats["pre_simplify_hole_fill_boundary_edges_before"] = fill.boundary_edges_before;
  stats["pre_simplify_hole_fill_clean_boundary_loops"] = fill.clean_boundary_loops;
  stats["pre_simplify_hole_fill_filled_loops"] = fill.filled_loops;
  stats["pre_simplify_hole_fill_skipped_large_loops"] = fill.skipped_large_loops;
  stats["pre_simplify_hole_fill_skipped_complex_components"] = fill.skipped_complex_components;
  stats["pre_simplify_hole_fill_vertices_added"] = fill.vertices_added;
  stats["pre_simplify_hole_fill_faces_added"] = fill.faces_added;
}

// Forked qem stat emission (M2/M4). The qem path has no clustering `best`, so it
// must NOT route through the shared stat block (which unconditionally reads an
// unpopulated `ClusterResult`). This builder emits the SAME keyset a clustering
// call produces on the same input (so S3 can widen to full keyset-equality),
// using 0/"n/a" sentinels for clustering-specific fields with no qem analogue,
// then overlays the qem-specific keys. `source_faces` is the raw input face
// count; `final_faces`/`final_vertices`/`pre_simplify_*` come from the qem run.
nb::dict build_qem_stats(
    const BackendSelection &selection,
    int64_t target_faces,
    int64_t source_faces,
    int64_t source_vertices,
    int64_t final_faces,
    int64_t final_vertices,
    int64_t unreferenced_removed,
    int64_t min_component_faces,
    bool target_reached,
    bool simplified,
    const QemSimplifyResult &qem) {
  nb::dict stats;
  // Backend identity / production gating (forks qem_* fields below).
  add_backend_stats(stats, selection, final_faces, target_reached);

  // Loop-fill families have no qem analogue: emit the disabled keyset so the
  // keyset matches a clustering call (sentinels, not arithmetic inputs).
  ReferenceLoopFillResult empty_pre_fill;
  SmallLoopFillResult empty_fill;
  add_pre_simplify_loop_fill_stats(stats, false, kSmallBoundaryLoopFillMaxPerimeter, empty_pre_fill);
  add_small_loop_fill_stats(
      stats, false, 0, kSmallBoundaryLoopFillMaxPerimeter, empty_fill);

  // Scalar block — same keys the clustering path emits.
  stats["target_faces"] = target_faces;
  stats["source_faces"] = source_faces;
  stats["source_vertices"] = source_vertices;
  stats["pre_simplify_faces"] = source_faces;
  stats["pre_simplify_vertices"] = source_vertices;
  stats["final_faces"] = final_faces;
  stats["final_vertices"] = final_vertices;
  stats["cluster_count"] = 0;
  stats["grid_resolution"] = 0;
  stats["degenerate_faces_removed"] = 0;
  stats["duplicate_faces_removed"] = 0;
  stats["nonmanifold_faces_removed"] = 0;
  stats["unreferenced_vertices_removed"] = unreferenced_removed;
  stats["target_reached"] = target_reached;
  stats["simplified"] = simplified;
  stats["min_component_faces"] = min_component_faces;
  stats["candidate_faces_considered"] = source_faces;
  stats["accepted_faces"] = final_faces;
  stats["representative_vertices_selected"] = 0;
  stats["representative_selection_strategy"] = "not_requested";
  stats["quadric_representative_candidates_evaluated"] = 0;
  stats["quadric_representative_nonfinite_candidates"] = 0;
  stats["quadric_representative_error_sum"] = 0.0;
  stats["quadric_representative_error_max"] = 0.0;

  // qem-specific overlays.
  stats["qem_simplification_backend"] = "native-qem-edge-collapse";
  stats["qem_equivalence_status"] = "edge-collapse";
  stats["qem_collapses_applied"] = qem.collapses_applied;
  stats["qem_collapses_rejected_by_guard"] = qem.collapses_rejected_by_guard;
  stats["qem_geometric_error_mean"] = qem.collapses_applied > 0
      ? qem.geometric_error_sum / static_cast<double>(qem.collapses_applied)
      : 0.0;
  stats["qem_geometric_error_max"] = qem.geometric_error_max;
  stats["qem_input_faces"] = source_faces;
  return stats;
}

}  // namespace

nb::dict simplify_mesh(
    nb::object vertices,
    nb::object faces,
    int64_t target_faces,
    int64_t min_component_faces,
    const std::string &backend,
    int64_t small_boundary_loop_fill_max_edges,
    double small_boundary_loop_fill_max_perimeter) {
  if (target_faces <= 0) {
    throw nb::value_error("target_faces must be positive");
  }
  if (min_component_faces <= 0) {
    throw nb::value_error("min_component_faces must be positive");
  }
  if (small_boundary_loop_fill_max_edges < 0) {
    throw nb::value_error("small_boundary_loop_fill_max_edges must be non-negative");
  }
  if (!std::isfinite(small_boundary_loop_fill_max_perimeter) || small_boundary_loop_fill_max_perimeter <= 0.0) {
    throw nb::value_error("small_boundary_loop_fill_max_perimeter must be positive");
  }
  const BackendSelection selection = resolve_backend(backend);
  mesh_common::MeshData input = mesh_common::load_mesh(vertices, faces);

  // QEM fork (M2): self-contained path that NEVER routes through the shared
  // clustering stat block (which reads an unpopulated `ClusterResult best`).
  // Handles both the backend-agnostic early-return (input already <= target)
  // and the edge-collapse simplification, each with its own stat emission.
  if (selection.is_qem) {
    const int64_t source_faces = static_cast<int64_t>(input.faces.size());
    const int64_t source_vertices = static_cast<int64_t>(input.vertices.size());
    if (source_faces <= target_faces) {
      int64_t unreferenced_removed = 0;
      mesh_common::MeshData compact = mesh_common::compact_mesh(input, &unreferenced_removed);
      nb::dict result = mesh_common::mesh_result(compact);
      QemSimplifyResult empty_run;
      empty_run.target_reached = true;
      result["stats"] = build_qem_stats(
          selection,
          target_faces,
          source_faces,
          source_vertices,
          static_cast<int64_t>(compact.faces.size()),
          static_cast<int64_t>(compact.vertices.size()),
          unreferenced_removed,
          min_component_faces,
          /*target_reached=*/true,
          /*simplified=*/false,
          empty_run);
      return result;
    }

    QemSimplifier simplifier(input);
    QemSimplifyResult qem = simplifier.run(target_faces);
    nb::dict result = mesh_common::mesh_result(qem.mesh);
    const int64_t final_faces = static_cast<int64_t>(qem.mesh.faces.size());
    const int64_t final_vertices = static_cast<int64_t>(qem.mesh.vertices.size());
    const int64_t unreferenced_removed = std::max<int64_t>(0, source_vertices - final_vertices);
    result["stats"] = build_qem_stats(
        selection,
        target_faces,
        source_faces,
        source_vertices,
        final_faces,
        final_vertices,
        unreferenced_removed,
        min_component_faces,
        qem.target_reached,
        /*simplified=*/qem.collapses_applied > 0,
        qem);
    return result;
  }

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
          small_boundary_loop_fill_max_perimeter,
          target_faces - static_cast<int64_t>(compact.faces.size()));
      compact = fill.mesh;
    }
    nb::dict result = mesh_common::mesh_result(compact);
    nb::dict stats;
    const bool target_reached = static_cast<int64_t>(compact.faces.size()) <= target_faces;
    ReferenceLoopFillResult pre_simplify_fill;
    add_backend_stats(stats, selection, static_cast<int64_t>(compact.faces.size()), target_reached);
    add_pre_simplify_loop_fill_stats(
        stats,
        false,
        small_boundary_loop_fill_max_perimeter,
        pre_simplify_fill);
    add_small_loop_fill_stats(
        stats,
        small_loop_fill_enabled,
        small_boundary_loop_fill_max_edges,
        small_boundary_loop_fill_max_perimeter,
        fill);
    stats["target_faces"] = target_faces;
    stats["source_faces"] = static_cast<int64_t>(input.faces.size());
    stats["source_vertices"] = static_cast<int64_t>(input.vertices.size());
    stats["pre_simplify_faces"] = static_cast<int64_t>(compact.faces.size());
    stats["pre_simplify_vertices"] = static_cast<int64_t>(compact.vertices.size());
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
    stats["representative_vertices_selected"] = 0;
    stats["representative_selection_strategy"] = selection.topology_aware ? "not_run_input_under_target" : "not_requested";
    stats["quadric_representative_candidates_evaluated"] = 0;
    stats["quadric_representative_nonfinite_candidates"] = 0;
    stats["quadric_representative_error_sum"] = 0.0;
    stats["quadric_representative_error_max"] = 0.0;
    result["stats"] = stats;
    return result;
  }

  ReferenceLoopFillResult pre_simplify_fill;
  const bool pre_simplify_fill_enabled = selection.topology_aware && small_boundary_loop_fill_max_edges > 0;
  const mesh_common::MeshData *simplification_input = &input;
  if (pre_simplify_fill_enabled) {
    pre_simplify_fill = fill_reference_clean_boundary_loops(
        input,
        kPreSimplifyCleanBoundaryLoopFillMaxEdges,
        small_boundary_loop_fill_max_perimeter);
    simplification_input = &pre_simplify_fill.mesh;
  }

  int64_t grid_resolution = initial_grid_resolution(target_faces);
  ClusterResult best = cluster_mesh(*simplification_input, grid_resolution, selection.topology_aware);
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
    ClusterResult candidate = cluster_mesh(*simplification_input, grid_resolution, selection.topology_aware);
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
        small_boundary_loop_fill_max_perimeter,
        target_faces - static_cast<int64_t>(simplified.faces.size()));
    simplified = fill.mesh;
  }
  nb::dict result = mesh_common::mesh_result(simplified);
  nb::dict stats;
  const bool target_reached = static_cast<int64_t>(simplified.faces.size()) <= target_faces;
  add_backend_stats(stats, selection, static_cast<int64_t>(simplified.faces.size()), target_reached);
  add_pre_simplify_loop_fill_stats(
      stats,
      pre_simplify_fill_enabled,
      small_boundary_loop_fill_max_perimeter,
      pre_simplify_fill);
  add_small_loop_fill_stats(
      stats,
      small_loop_fill_enabled,
      small_boundary_loop_fill_max_edges,
      small_boundary_loop_fill_max_perimeter,
      fill);
  stats["target_faces"] = target_faces;
  stats["source_faces"] = static_cast<int64_t>(input.faces.size());
  stats["source_vertices"] = static_cast<int64_t>(input.vertices.size());
  stats["pre_simplify_faces"] = static_cast<int64_t>(simplification_input->faces.size());
  stats["pre_simplify_vertices"] = static_cast<int64_t>(simplification_input->vertices.size());
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
  stats["candidate_faces_considered"] = static_cast<int64_t>(simplification_input->faces.size());
  stats["accepted_faces"] = static_cast<int64_t>(simplified.faces.size());
  stats["representative_vertices_selected"] = best.representative_vertices_selected;
  stats["representative_selection_strategy"] = best.representative_selection_strategy;
  stats["quadric_representative_candidates_evaluated"] = best.quadric_representative_candidates_evaluated;
  stats["quadric_representative_nonfinite_candidates"] = best.quadric_representative_nonfinite_candidates;
  stats["quadric_representative_error_sum"] = best.quadric_representative_error_sum;
  stats["quadric_representative_error_max"] = best.quadric_representative_error_max;
  result["stats"] = stats;
  return result;
}

}  // namespace mlx_spatialkit
