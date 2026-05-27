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
#include <numeric>
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
  int64_t loops_rejected_fallback_cap = 0;
  int64_t loops_rejected_degenerate = 0;
  int64_t loops_rejected_duplicate = 0;
  int64_t loops_rejected_nonmanifold = 0;
  int64_t loops_budget_limited = 0;
  int64_t branched_cycle_candidates = 0;
  int64_t branched_cycles_filled = 0;
  int64_t branched_cycles_rejected = 0;
  int64_t branched_cycles_budget_limited = 0;
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
};

constexpr const char *kSmallBoundaryLoopFillAlgorithm = "projected-ear-clipping";
constexpr const char *kSmallBoundaryLoopFillFallbackAlgorithm = "centroid-fan";
constexpr int64_t kSmallBoundaryLoopFillFallbackMaxEdges = 8;
constexpr int64_t kSmallBoundaryBranchedCycleFillMaxEdges = 6;
constexpr int64_t kSmallBoundaryLoopRepairMaxPasses = 2;
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

std::array<float, 3> loop_centroid(const mesh_common::MeshData &mesh, const std::vector<int64_t> &loop) {
  std::array<double, 3> centroid{0.0, 0.0, 0.0};
  for (const int64_t vertex_id : loop) {
    const auto &vertex = mesh.vertices[static_cast<size_t>(vertex_id)];
    centroid[0] += vertex[0];
    centroid[1] += vertex[1];
    centroid[2] += vertex[2];
  }
  const double denom = static_cast<double>(std::max<size_t>(1, loop.size()));
  return {
      static_cast<float>(centroid[0] / denom),
      static_cast<float>(centroid[1] / denom),
      static_cast<float>(centroid[2] / denom),
  };
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
    int64_t face_budget) {
  SmallLoopFillResult result;
  result.mesh = input;
  result.face_budget = std::max<int64_t>(0, face_budget);
  if (result.face_budget <= 0 || result.mesh.faces.empty()) {
    return result;
  }
  const int64_t fallback_max_edges = effective_repair_cap(max_loop_edges, kSmallBoundaryLoopFillFallbackMaxEdges);
  const int64_t branched_cycle_max_edges =
      effective_repair_cap(max_loop_edges, kSmallBoundaryBranchedCycleFillMaxEdges);

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

  auto apply_loop_patch = [&](const std::vector<int64_t> &loop, bool branched_cycle) {
    result.loops_considered += 1;
    if (branched_cycle) {
      result.branched_cycle_candidates += 1;
    }
    const int64_t loop_edges = static_cast<int64_t>(loop.size());

    std::vector<std::array<int64_t, 3>> patch_faces = triangulate_loop_patch(result.mesh, loop);
    bool centroid_fan_patch = false;
    bool alternative_triangulation_patch = false;
    if (patch_faces.empty()) {
      if (loop_edges > fallback_max_edges) {
        result.loops_rejected += 1;
        result.loops_rejected_fallback_cap += 1;
        if (branched_cycle) {
          result.branched_cycles_rejected += 1;
        }
        return;
      }
      result.loops_centroid_fan_attempted += 1;
      const int64_t center_vertex_id = static_cast<int64_t>(result.mesh.vertices.size());
      const std::array<float, 3> center = loop_centroid(result.mesh, loop);
      if (!std::isfinite(center[0]) || !std::isfinite(center[1]) || !std::isfinite(center[2])) {
        result.loops_rejected += 1;
        result.loops_rejected_triangulation += 1;
        if (branched_cycle) {
          result.branched_cycles_rejected += 1;
        }
        return;
      }
      result.mesh.vertices.push_back(center);
      patch_faces = centroid_fan_loop_patch(result.mesh, loop, center_vertex_id);
      centroid_fan_patch = true;
      if (patch_faces.empty()) {
        result.mesh.vertices.pop_back();
        result.loops_rejected += 1;
        result.loops_rejected_triangulation += 1;
        if (branched_cycle) {
          result.branched_cycles_rejected += 1;
        }
        return;
      }
    }

    PatchRejectReason reject_reason = validate_patch_faces(result.mesh, patch_faces, seen_faces, edge_counts);
    if (!centroid_fan_patch
        && (reject_reason == PatchRejectReason::duplicate || reject_reason == PatchRejectReason::nonmanifold)) {
      std::vector<std::vector<std::array<int64_t, 3>>> variants =
          triangulate_loop_patch_variants(result.mesh, loop, kSmallBoundaryAlternativeTriangulationMaxVariants);
      if (variants.size() > 1) {
        result.loops_alternative_triangulation_attempted += 1;
      }
      for (const auto &variant : variants) {
        const PatchRejectReason variant_reject =
            validate_patch_faces(result.mesh, variant, seen_faces, edge_counts);
        if (variant_reject == PatchRejectReason::none) {
          patch_faces = variant;
          alternative_triangulation_patch = true;
          reject_reason = PatchRejectReason::none;
          break;
        }
      }
    }

    if (reject_reason != PatchRejectReason::none) {
      if (centroid_fan_patch) {
        result.mesh.vertices.pop_back();
      }
      record_patch_rejection(result, reject_reason);
      if (branched_cycle) {
        result.branched_cycles_rejected += 1;
      }
      return;
    }
    if (static_cast<int64_t>(patch_faces.size()) > result.face_budget - result.faces_added) {
      if (centroid_fan_patch) {
        result.mesh.vertices.pop_back();
      }
      result.loops_budget_limited += 1;
      if (branched_cycle) {
        result.branched_cycles_budget_limited += 1;
      }
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
    if (branched_cycle) {
      result.branched_cycles_filled += 1;
    }
    if (centroid_fan_patch) {
      result.loops_filled_by_centroid_fan += 1;
    } else {
      result.loops_filled_by_ear_clipping += 1;
      if (alternative_triangulation_patch) {
        result.loops_filled_by_alternative_triangulation += 1;
      }
    }
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
      if (component_edges > max_loop_edges) {
        continue;
      }
      const std::vector<int64_t> loop = ordered_closed_loop(adjacency, component_vertices);
      if (loop.size() < 3 || static_cast<int64_t>(loop.size()) != component_edges) {
        result.loops_considered += 1;
        result.loops_rejected += 1;
        result.loops_rejected_ordering += 1;
        continue;
      }
      apply_loop_patch(loop, false);
      continue;
    }

    if (branched_cycle_max_edges < 3) {
      continue;
    }
    std::vector<std::vector<int64_t>> cycles =
        branched_component_cycles(adjacency, component_vertices, branched_cycle_max_edges);
    for (const auto &cycle : cycles) {
      apply_loop_patch(cycle, true);
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
  total.loops_rejected_fallback_cap += pass.loops_rejected_fallback_cap;
  total.loops_rejected_degenerate += pass.loops_rejected_degenerate;
  total.loops_rejected_duplicate += pass.loops_rejected_duplicate;
  total.loops_rejected_nonmanifold += pass.loops_rejected_nonmanifold;
  total.loops_budget_limited += pass.loops_budget_limited;
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
    SmallLoopFillResult pass = fill_small_boundary_loops_single_pass(total.mesh, max_loop_edges, remaining_budget);
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
  stats["small_boundary_loop_fill_algorithm"] = kSmallBoundaryLoopFillAlgorithm;
  stats["small_boundary_loop_fill_fallback_algorithm"] = kSmallBoundaryLoopFillFallbackAlgorithm;
  stats["small_boundary_loop_fill_fallback_enabled"] = enabled;
  stats["small_boundary_loop_fill_fallback_policy_max_edges"] = kSmallBoundaryLoopFillFallbackMaxEdges;
  stats["small_boundary_loop_fill_fallback_max_edges"] = kSmallBoundaryLoopFillFallbackMaxEdges;
  stats["small_boundary_loop_fill_fallback_effective_max_edges"] =
      enabled ? effective_repair_cap(max_loop_edges, kSmallBoundaryLoopFillFallbackMaxEdges) : 0;
  stats["small_boundary_branched_cycle_fill_enabled"] = enabled;
  stats["small_boundary_branched_cycle_fill_policy_max_edges"] = kSmallBoundaryBranchedCycleFillMaxEdges;
  stats["small_boundary_branched_cycle_fill_max_edges"] = kSmallBoundaryBranchedCycleFillMaxEdges;
  stats["small_boundary_branched_cycle_fill_effective_max_edges"] =
      enabled ? effective_repair_cap(max_loop_edges, kSmallBoundaryBranchedCycleFillMaxEdges) : 0;
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
  stats["small_boundary_loops_rejected_fallback_cap"] = fill.loops_rejected_fallback_cap;
  stats["small_boundary_loops_rejected_degenerate"] = fill.loops_rejected_degenerate;
  stats["small_boundary_loops_rejected_duplicate"] = fill.loops_rejected_duplicate;
  stats["small_boundary_loops_rejected_nonmanifold"] = fill.loops_rejected_nonmanifold;
  stats["small_boundary_loops_budget_limited"] = fill.loops_budget_limited;
  stats["small_boundary_branched_cycle_candidates"] = fill.branched_cycle_candidates;
  stats["small_boundary_branched_cycles_filled"] = fill.branched_cycles_filled;
  stats["small_boundary_branched_cycles_rejected"] = fill.branched_cycles_rejected;
  stats["small_boundary_branched_cycles_budget_limited"] = fill.branched_cycles_budget_limited;
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
