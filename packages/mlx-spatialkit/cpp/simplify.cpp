#include "mesh_processing.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <set>
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

ClusterResult cluster_mesh(const mesh_common::MeshData &input, int64_t grid_resolution) {
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
  for (const auto &accum : accumulators) {
    const double denom = static_cast<double>(std::max<int64_t>(1, accum.count));
    output.vertices.push_back({
        static_cast<float>(accum.x / denom),
        static_cast<float>(accum.y / denom),
        static_cast<float>(accum.z / denom),
    });
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
  return result;
}

int64_t initial_grid_resolution(int64_t target_faces) {
  const double resolution = std::ceil(std::sqrt(std::max<double>(2.0, static_cast<double>(target_faces) * 0.5)));
  return std::max<int64_t>(2, static_cast<int64_t>(resolution));
}

}  // namespace

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

  if (static_cast<int64_t>(input.faces.size()) <= target_faces) {
    int64_t unreferenced_removed = 0;
    mesh_common::MeshData compact = mesh_common::compact_mesh(input, &unreferenced_removed);
    nb::dict result = mesh_common::mesh_result(compact);
    nb::dict stats;
    stats["backend"] = "spatial-cluster";
    stats["algorithm"] = "native_spatial_vertex_clustering";
    stats["quality_tier"] = "geometry_aware_preview";
    stats["production_ready"] = false;
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
    stats["target_reached"] = static_cast<int64_t>(compact.faces.size()) <= target_faces;
    stats["simplified"] = false;
    stats["min_component_faces"] = min_component_faces;
    result["stats"] = stats;
    return result;
  }

  int64_t grid_resolution = initial_grid_resolution(target_faces);
  ClusterResult best = cluster_mesh(input, grid_resolution);
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
    ClusterResult candidate = cluster_mesh(input, grid_resolution);
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
  nb::dict result = mesh_common::mesh_result(simplified);
  nb::dict stats;
  stats["backend"] = "spatial-cluster";
  stats["algorithm"] = "native_spatial_vertex_clustering";
  stats["quality_tier"] = "geometry_aware_preview";
  stats["production_ready"] = false;
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
  stats["target_reached"] = static_cast<int64_t>(simplified.faces.size()) <= target_faces;
  stats["simplified"] = static_cast<int64_t>(input.faces.size()) > target_faces;
  stats["min_component_faces"] = min_component_faces;
  result["stats"] = stats;
  return result;
}

}  // namespace mlx_spatialkit
