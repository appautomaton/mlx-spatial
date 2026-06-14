#pragma once

// Shared point->triangle-mesh distance / closest-point BVH.
//
// Hoisted verbatim from metal/texture_bake.mm (no behavior change) so both the
// texture bake (texel -> source-mesh projection) and the narrow-band remesh
// (unsigned-distance field sampling) reuse the exact same closest-point
// primitive. Free functions are `inline` for safe inclusion in multiple
// translation units; the class member functions are implicitly inline.

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include "mesh_common.hpp"

namespace mlx_spatialkit {

struct BvhTriangle {
  int64_t face_index = 0;
  std::array<float, 3> min_bounds{};
  std::array<float, 3> max_bounds{};
  std::array<float, 3> centroid{};
};

struct BvhNode {
  std::array<float, 3> min_bounds{};
  std::array<float, 3> max_bounds{};
  int32_t left = -1;
  int32_t right = -1;
  int32_t start = 0;
  int32_t count = 0;
};

struct ClosestPointResult {
  std::array<float, 3> point{};
  std::array<float, 3> barycentric{};
  int64_t face_index = -1;
  double distance2 = std::numeric_limits<double>::infinity();
};

inline std::array<float, 3> min3(const std::array<float, 3> &left, const std::array<float, 3> &right) {
  return {
      std::min(left[0], right[0]),
      std::min(left[1], right[1]),
      std::min(left[2], right[2]),
  };
}

inline std::array<float, 3> max3(const std::array<float, 3> &left, const std::array<float, 3> &right) {
  return {
      std::max(left[0], right[0]),
      std::max(left[1], right[1]),
      std::max(left[2], right[2]),
  };
}

inline double distance2_aabb(
    const std::array<float, 3> &point,
    const std::array<float, 3> &min_bounds,
    const std::array<float, 3> &max_bounds) {
  double distance = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    const double value = point[static_cast<size_t>(axis)];
    const double lo = min_bounds[static_cast<size_t>(axis)];
    const double hi = max_bounds[static_cast<size_t>(axis)];
    if (value < lo) {
      const double delta = lo - value;
      distance += delta * delta;
    } else if (value > hi) {
      const double delta = value - hi;
      distance += delta * delta;
    }
  }
  return distance;
}

inline ClosestPointResult closest_point_on_triangle(
    const std::array<float, 3> &point,
    const std::array<float, 3> &a,
    const std::array<float, 3> &b,
    const std::array<float, 3> &c,
    int64_t face_index) {
  auto sub = [](const std::array<float, 3> &left, const std::array<float, 3> &right) {
    return std::array<double, 3>{
        static_cast<double>(left[0]) - static_cast<double>(right[0]),
        static_cast<double>(left[1]) - static_cast<double>(right[1]),
        static_cast<double>(left[2]) - static_cast<double>(right[2]),
    };
  };
  auto dot = [](const std::array<double, 3> &left, const std::array<double, 3> &right) {
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
  };
  auto make_result = [&](double wa, double wb, double wc) {
    std::array<float, 3> projected{
        static_cast<float>(wa * a[0] + wb * b[0] + wc * c[0]),
        static_cast<float>(wa * a[1] + wb * b[1] + wc * c[1]),
        static_cast<float>(wa * a[2] + wb * b[2] + wc * c[2]),
    };
    const double dx = static_cast<double>(point[0]) - projected[0];
    const double dy = static_cast<double>(point[1]) - projected[1];
    const double dz = static_cast<double>(point[2]) - projected[2];
    return ClosestPointResult{
        projected,
        {static_cast<float>(wa), static_cast<float>(wb), static_cast<float>(wc)},
        face_index,
        dx * dx + dy * dy + dz * dz,
    };
  };

  const std::array<double, 3> ab = sub(b, a);
  const std::array<double, 3> ac = sub(c, a);
  const std::array<double, 3> ap = sub(point, a);
  const double d1 = dot(ab, ap);
  const double d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) {
    return make_result(1.0, 0.0, 0.0);
  }

  const std::array<double, 3> bp = sub(point, b);
  const double d3 = dot(ab, bp);
  const double d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) {
    return make_result(0.0, 1.0, 0.0);
  }

  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    const double v = d1 / (d1 - d3);
    return make_result(1.0 - v, v, 0.0);
  }

  const std::array<double, 3> cp = sub(point, c);
  const double d5 = dot(ab, cp);
  const double d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) {
    return make_result(0.0, 0.0, 1.0);
  }

  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    const double w = d2 / (d2 - d6);
    return make_result(1.0 - w, 0.0, w);
  }

  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return make_result(0.0, 1.0 - w, w);
  }

  const double denom = 1.0 / (va + vb + vc);
  const double v = vb * denom;
  const double w = vc * denom;
  return make_result(1.0 - v - w, v, w);
}

class TriangleBvh {
 public:
  explicit TriangleBvh(const mesh_common::MeshData &mesh) : mesh_(mesh) {
    triangles_.reserve(mesh_.faces.size());
    for (size_t index = 0; index < mesh_.faces.size(); ++index) {
      const auto &face = mesh_.faces[index];
      const auto &a = mesh_.vertices[static_cast<size_t>(face[0])];
      const auto &b = mesh_.vertices[static_cast<size_t>(face[1])];
      const auto &c = mesh_.vertices[static_cast<size_t>(face[2])];
      BvhTriangle triangle;
      triangle.face_index = static_cast<int64_t>(index);
      triangle.min_bounds = min3(min3(a, b), c);
      triangle.max_bounds = max3(max3(a, b), c);
      triangle.centroid = {
          (a[0] + b[0] + c[0]) / 3.0f,
          (a[1] + b[1] + c[1]) / 3.0f,
          (a[2] + b[2] + c[2]) / 3.0f,
      };
      triangles_.push_back(triangle);
    }
    if (!triangles_.empty()) {
      build_node(0, static_cast<int32_t>(triangles_.size()));
    }
  }

  int64_t node_count() const {
    return static_cast<int64_t>(nodes_.size());
  }

  ClosestPointResult closest_point(const std::array<float, 3> &point) const {
    ClosestPointResult best;
    if (nodes_.empty()) {
      return best;
    }
    std::vector<int32_t> stack;
    stack.reserve(64);
    stack.push_back(0);
    while (!stack.empty()) {
      const int32_t node_index = stack.back();
      stack.pop_back();
      const BvhNode &node = nodes_[static_cast<size_t>(node_index)];
      if (distance2_aabb(point, node.min_bounds, node.max_bounds) > best.distance2) {
        continue;
      }
      if (node.left < 0 && node.right < 0) {
        for (int32_t offset = 0; offset < node.count; ++offset) {
          const BvhTriangle &triangle = triangles_[static_cast<size_t>(node.start + offset)];
          const auto &face = mesh_.faces[static_cast<size_t>(triangle.face_index)];
          const ClosestPointResult candidate = closest_point_on_triangle(
              point,
              mesh_.vertices[static_cast<size_t>(face[0])],
              mesh_.vertices[static_cast<size_t>(face[1])],
              mesh_.vertices[static_cast<size_t>(face[2])],
              triangle.face_index);
          if (candidate.distance2 < best.distance2) {
            best = candidate;
          }
        }
        continue;
      }
      const int32_t left = node.left;
      const int32_t right = node.right;
      if (left >= 0 && right >= 0) {
        const BvhNode &left_node = nodes_[static_cast<size_t>(left)];
        const BvhNode &right_node = nodes_[static_cast<size_t>(right)];
        const double left_distance = distance2_aabb(point, left_node.min_bounds, left_node.max_bounds);
        const double right_distance = distance2_aabb(point, right_node.min_bounds, right_node.max_bounds);
        if (left_distance < right_distance) {
          if (right_distance <= best.distance2) {
            stack.push_back(right);
          }
          if (left_distance <= best.distance2) {
            stack.push_back(left);
          }
        } else {
          if (left_distance <= best.distance2) {
            stack.push_back(left);
          }
          if (right_distance <= best.distance2) {
            stack.push_back(right);
          }
        }
      } else if (left >= 0) {
        stack.push_back(left);
      } else if (right >= 0) {
        stack.push_back(right);
      }
    }
    return best;
  }

 private:
  int32_t build_node(int32_t start, int32_t end) {
    BvhNode node;
    node.start = start;
    node.count = end - start;
    node.min_bounds = {
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
    };
    node.max_bounds = {
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    };
    std::array<float, 3> centroid_min = node.min_bounds;
    std::array<float, 3> centroid_max = node.max_bounds;
    for (int32_t index = start; index < end; ++index) {
      const BvhTriangle &triangle = triangles_[static_cast<size_t>(index)];
      node.min_bounds = min3(node.min_bounds, triangle.min_bounds);
      node.max_bounds = max3(node.max_bounds, triangle.max_bounds);
      centroid_min = min3(centroid_min, triangle.centroid);
      centroid_max = max3(centroid_max, triangle.centroid);
    }
    const int32_t node_index = static_cast<int32_t>(nodes_.size());
    nodes_.push_back(node);
    if (node.count <= 8) {
      return node_index;
    }

    int axis = 0;
    float best_extent = centroid_max[0] - centroid_min[0];
    for (int candidate_axis = 1; candidate_axis < 3; ++candidate_axis) {
      const float extent = centroid_max[static_cast<size_t>(candidate_axis)] - centroid_min[static_cast<size_t>(candidate_axis)];
      if (extent > best_extent) {
        axis = candidate_axis;
        best_extent = extent;
      }
    }
    if (best_extent <= 1e-12f) {
      return node_index;
    }
    const int32_t mid = start + node.count / 2;
    std::nth_element(
        triangles_.begin() + start,
        triangles_.begin() + mid,
        triangles_.begin() + end,
        [axis](const BvhTriangle &left, const BvhTriangle &right) {
          return left.centroid[static_cast<size_t>(axis)] < right.centroid[static_cast<size_t>(axis)];
        });
    nodes_[static_cast<size_t>(node_index)].left = build_node(start, mid);
    nodes_[static_cast<size_t>(node_index)].right = build_node(mid, end);
    nodes_[static_cast<size_t>(node_index)].start = 0;
    nodes_[static_cast<size_t>(node_index)].count = 0;
    return node_index;
  }

  const mesh_common::MeshData &mesh_;
  std::vector<BvhTriangle> triangles_;
  std::vector<BvhNode> nodes_;
};

}  // namespace mlx_spatialkit
