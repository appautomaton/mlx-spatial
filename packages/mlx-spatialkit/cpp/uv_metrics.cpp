#include "uv_metrics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "mesh_common.hpp"

namespace nb = nanobind;

namespace mlx_spatialkit {
namespace {

// Tolerance on signed UV areas and on separating-axis projection overlaps.
// Edge- or vertex-touching triangles (coordinates within this epsilon) must
// not register as overlapping.
constexpr double kUvEps = 1e-12;
constexpr int64_t kMaxGridCellsPerAxis = 1024;
// Triangles spanning more than this many grid cells on either axis are kept
// out of the uniform grid and checked against every candidate instead, so a
// few atlas-spanning triangles cannot blow grid memory up to
// kMaxGridCellsPerAxis^2 entries each.
constexpr int64_t kMaxCellSpanPerAxis = 64;

struct UvTriangle {
  std::array<double, 3> u{};
  std::array<double, 3> v{};
  double min_u = 0.0;
  double max_u = 0.0;
  double min_v = 0.0;
  double max_v = 0.0;
  double signed_area = 0.0;
};

struct StretchAccumulator {
  double weighted_l2_squared = 0.0;
  double area_3d = 0.0;
  double max_gamma = 0.0;

  // Stretch semantics: flipped faces are excluded from stretch accumulation
  // (they are still counted in uv_flipped_count), as are degenerate faces.
  // An accumulator whose faces were all degenerate or flipped therefore has
  // area_3d == 0 and reports stretch 0.0, meaning "no measurable faces",
  // not "zero distortion".  No sentinel is emitted; downstream consumers
  // gate on uv_flipped_count / uv_overlap_count before reading stretch.
  double l2() const {
    return area_3d > 0.0 ? std::sqrt(weighted_l2_squared / area_3d) : 0.0;
  }
};

std::vector<std::array<double, 2>> load_uv_coordinates(nb::object uvs_object, int64_t vertex_count) {
  mesh_common::validate_matrix(uvs_object, "UV coordinates", 2, "float32");
  if (mesh_common::dimension(uvs_object, "UV coordinates", 0) != vertex_count) {
    std::ostringstream message;
    message << "UV coordinates must have shape (" << vertex_count << ", 2)";
    throw nb::value_error(message.str().c_str());
  }
  mesh_common::BufferView uv_buffer(uvs_object.ptr(), "UV coordinates");
  const Py_buffer &view = uv_buffer.get();
  std::vector<std::array<double, 2>> uvs;
  uvs.reserve(static_cast<size_t>(vertex_count));
  for (int64_t row = 0; row < vertex_count; ++row) {
    const std::array<double, 2> uv{
        static_cast<double>(mesh_common::read_matrix_value<float>(view, row, 0)),
        static_cast<double>(mesh_common::read_matrix_value<float>(view, row, 1)),
    };
    if (!std::isfinite(uv[0]) || !std::isfinite(uv[1])) {
      throw nb::value_error("UV coordinates must contain only finite values");
    }
    uvs.push_back(uv);
  }
  return uvs;
}

// Rank-1 buffer reader.  mesh_common::read_matrix_value unconditionally
// dereferences strides[1], but NumPy allocates only ndim stride entries, so a
// rank-1 buffer must never reach it.  This reader honours strides[0] when the
// exporter provides strides and falls back to contiguous layout otherwise.
template <typename T>
T read_vector_value(const Py_buffer &view, int64_t index) {
  const auto *base = static_cast<const char *>(view.buf);
  const Py_ssize_t stride = view.strides != nullptr ? view.strides[0] : view.itemsize;
  T value{};
  std::memcpy(&value, base + index * stride, sizeof(T));
  return value;
}

std::vector<int64_t> load_chart_ids(nb::object chart_ids_object, int64_t face_count) {
  const auto ndim = nb::cast<int64_t>(nb::getattr(chart_ids_object, "ndim"));
  if (ndim != 1) {
    std::ostringstream message;
    message << "chart_ids must have rank 1, got rank " << ndim;
    throw nb::value_error(message.str().c_str());
  }
  const std::string dtype = mesh_common::dtype_name(chart_ids_object, "chart_ids");
  if (dtype != "int64") {
    std::ostringstream message;
    message << "chart_ids must have dtype int64, got " << dtype;
    throw nb::value_error(message.str().c_str());
  }
  if (mesh_common::dimension(chart_ids_object, "chart_ids", 0) != face_count) {
    std::ostringstream message;
    message << "chart_ids must have shape (" << face_count << ",)";
    throw nb::value_error(message.str().c_str());
  }
  mesh_common::BufferView chart_buffer(chart_ids_object.ptr(), "chart_ids");
  const Py_buffer &view = chart_buffer.get();
  std::vector<int64_t> chart_ids;
  chart_ids.reserve(static_cast<size_t>(face_count));
  for (int64_t row = 0; row < face_count; ++row) {
    chart_ids.push_back(read_vector_value<int64_t>(view, row));
  }
  return chart_ids;
}

// Returns true when the projections of both triangles onto the (normalized)
// axis overlap by strictly more than kUvEps.  A merely touching projection
// (shared edge or vertex) keeps the axis separating, so edge-adjacent
// triangles do not count as overlapping.
bool axis_keeps_positive_overlap(
    const UvTriangle &a,
    const UvTriangle &b,
    double axis_x,
    double axis_y) {
  const double length = std::sqrt(axis_x * axis_x + axis_y * axis_y);
  if (length <= kUvEps) {
    return true;  // Degenerate axis cannot separate anything.
  }
  axis_x /= length;
  axis_y /= length;
  double min_a = std::numeric_limits<double>::infinity();
  double max_a = -std::numeric_limits<double>::infinity();
  double min_b = std::numeric_limits<double>::infinity();
  double max_b = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < 3; ++i) {
    const double projection_a = axis_x * a.u[i] + axis_y * a.v[i];
    const double projection_b = axis_x * b.u[i] + axis_y * b.v[i];
    min_a = std::min(min_a, projection_a);
    max_a = std::max(max_a, projection_a);
    min_b = std::min(min_b, projection_b);
    max_b = std::max(max_b, projection_b);
  }
  return std::min(max_a, max_b) - std::max(min_a, min_b) > kUvEps;
}

// Exact 2D separating-axis test for positive-area interior overlap.  For
// convex polygons with disjoint interiors a separating (or merely supporting)
// line parallel to one of the edges always exists, so checking the six edge
// normals is sufficient.
bool interiors_overlap(const UvTriangle &a, const UvTriangle &b) {
  const std::array<const UvTriangle *, 2> triangles{&a, &b};
  for (const UvTriangle *triangle : triangles) {
    for (int i = 0; i < 3; ++i) {
      const int j = (i + 1) % 3;
      const double edge_x = triangle->u[j] - triangle->u[i];
      const double edge_y = triangle->v[j] - triangle->v[i];
      if (!axis_keeps_positive_overlap(a, b, -edge_y, edge_x)) {
        return false;
      }
    }
  }
  return true;
}

double triangle_area_3d(const mesh_common::MeshData &mesh, const std::array<int64_t, 3> &face) {
  std::array<double, 3> edge_ab{};
  std::array<double, 3> edge_ac{};
  for (int axis = 0; axis < 3; ++axis) {
    const double a = mesh.vertices[static_cast<size_t>(face[0])][axis];
    const double b = mesh.vertices[static_cast<size_t>(face[1])][axis];
    const double c = mesh.vertices[static_cast<size_t>(face[2])][axis];
    edge_ab[static_cast<size_t>(axis)] = b - a;
    edge_ac[static_cast<size_t>(axis)] = c - a;
  }
  const std::array<double, 3> cross{
      edge_ab[1] * edge_ac[2] - edge_ab[2] * edge_ac[1],
      edge_ab[2] * edge_ac[0] - edge_ab[0] * edge_ac[2],
      edge_ab[0] * edge_ac[1] - edge_ab[1] * edge_ac[0],
  };
  return std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]) / 2.0;
}

int64_t clamp_cell(double value, int64_t cell_count) {
  if (!(value > 0.0)) {
    return 0;
  }
  const auto cell = static_cast<int64_t>(value);
  return std::min(std::max<int64_t>(cell, 0), cell_count - 1);
}

}  // namespace

nb::dict uv_quality_metrics(
    nb::object vertices,
    nb::object faces,
    nb::object uvs,
    nb::object chart_ids) {
  const mesh_common::MeshData mesh = mesh_common::load_mesh(vertices, faces);
  const auto uv = load_uv_coordinates(uvs, static_cast<int64_t>(mesh.vertices.size()));
  const auto face_count = static_cast<int64_t>(mesh.faces.size());
  const bool has_chart_ids = chart_ids.ptr() != Py_None;
  std::vector<int64_t> chart_id_values;
  if (has_chart_ids) {
    chart_id_values = load_chart_ids(chart_ids, face_count);
  }

  std::vector<UvTriangle> triangles;
  triangles.reserve(static_cast<size_t>(face_count));
  int64_t flipped_count = 0;
  int64_t degenerate_count = 0;
  double total_uv_area = 0.0;
  double bbox_min_u = std::numeric_limits<double>::infinity();
  double bbox_max_u = -std::numeric_limits<double>::infinity();
  double bbox_min_v = std::numeric_limits<double>::infinity();
  double bbox_max_v = -std::numeric_limits<double>::infinity();

  StretchAccumulator global_stretch;
  // Ordered map keeps the per-chart output deterministic and sorted by id.
  std::map<int64_t, StretchAccumulator> chart_stretch;
  for (const int64_t chart_id : chart_id_values) {
    chart_stretch[chart_id];
  }

  for (int64_t face_index = 0; face_index < face_count; ++face_index) {
    const auto &face = mesh.faces[static_cast<size_t>(face_index)];
    UvTriangle triangle;
    for (int corner = 0; corner < 3; ++corner) {
      const auto &coordinates = uv[static_cast<size_t>(face[corner])];
      triangle.u[static_cast<size_t>(corner)] = coordinates[0];
      triangle.v[static_cast<size_t>(corner)] = coordinates[1];
    }
    triangle.min_u = std::min({triangle.u[0], triangle.u[1], triangle.u[2]});
    triangle.max_u = std::max({triangle.u[0], triangle.u[1], triangle.u[2]});
    triangle.min_v = std::min({triangle.v[0], triangle.v[1], triangle.v[2]});
    triangle.max_v = std::max({triangle.v[0], triangle.v[1], triangle.v[2]});
    triangle.signed_area =
        ((triangle.u[1] - triangle.u[0]) * (triangle.v[2] - triangle.v[0]) -
         (triangle.u[2] - triangle.u[0]) * (triangle.v[1] - triangle.v[0])) / 2.0;
    bbox_min_u = std::min(bbox_min_u, triangle.min_u);
    bbox_max_u = std::max(bbox_max_u, triangle.max_u);
    bbox_min_v = std::min(bbox_min_v, triangle.min_v);
    bbox_max_v = std::max(bbox_max_v, triangle.max_v);
    total_uv_area += std::abs(triangle.signed_area);

    const bool degenerate = std::abs(triangle.signed_area) <= kUvEps;
    const bool flipped = triangle.signed_area < -kUvEps;
    if (degenerate) {
      degenerate_count += 1;
    }
    if (flipped) {
      flipped_count += 1;
    }

    if (!degenerate && !flipped) {
      // Sander et al. 2001 texel stretch for the parameterization (s, t) -> q.
      const double area = triangle.signed_area;
      std::array<double, 3> tangent_s{};
      std::array<double, 3> tangent_t{};
      for (int axis = 0; axis < 3; ++axis) {
        const double q0 = mesh.vertices[static_cast<size_t>(face[0])][axis];
        const double q1 = mesh.vertices[static_cast<size_t>(face[1])][axis];
        const double q2 = mesh.vertices[static_cast<size_t>(face[2])][axis];
        tangent_s[static_cast<size_t>(axis)] =
            (q0 * (triangle.v[1] - triangle.v[2]) +
             q1 * (triangle.v[2] - triangle.v[0]) +
             q2 * (triangle.v[0] - triangle.v[1])) / (2.0 * area);
        tangent_t[static_cast<size_t>(axis)] =
            (q0 * (triangle.u[2] - triangle.u[1]) +
             q1 * (triangle.u[0] - triangle.u[2]) +
             q2 * (triangle.u[1] - triangle.u[0])) / (2.0 * area);
      }
      const double a = tangent_s[0] * tangent_s[0] + tangent_s[1] * tangent_s[1] + tangent_s[2] * tangent_s[2];
      const double b = tangent_s[0] * tangent_t[0] + tangent_s[1] * tangent_t[1] + tangent_s[2] * tangent_t[2];
      const double c = tangent_t[0] * tangent_t[0] + tangent_t[1] * tangent_t[1] + tangent_t[2] * tangent_t[2];
      const double discriminant = std::sqrt(std::max(0.0, (a - c) * (a - c) + 4.0 * b * b));
      const double gamma = std::sqrt(((a + c) + discriminant) / 2.0);
      const double l2_squared = (a + c) / 2.0;
      const double area_3d = triangle_area_3d(mesh, face);
      global_stretch.weighted_l2_squared += l2_squared * area_3d;
      global_stretch.area_3d += area_3d;
      global_stretch.max_gamma = std::max(global_stretch.max_gamma, gamma);
      if (has_chart_ids) {
        StretchAccumulator &chart = chart_stretch[chart_id_values[static_cast<size_t>(face_index)]];
        chart.weighted_l2_squared += l2_squared * area_3d;
        chart.area_3d += area_3d;
        chart.max_gamma = std::max(chart.max_gamma, gamma);
      }
    }

    triangles.push_back(triangle);
  }

  // Overlap counting: uniform grid over the UV bounding box for candidate
  // pairs, exact separating-axis test for positive-area interior overlap.
  // Cells, candidate lists, and pairs are all visited in ascending index
  // order so the result is deterministic.
  int64_t overlap_count = 0;
  int64_t overlap_checked_pairs = 0;
  std::vector<int64_t> overlap_candidates;
  overlap_candidates.reserve(static_cast<size_t>(face_count));
  for (int64_t face_index = 0; face_index < face_count; ++face_index) {
    if (std::abs(triangles[static_cast<size_t>(face_index)].signed_area) > kUvEps) {
      overlap_candidates.push_back(face_index);
    }
  }
  if (overlap_candidates.size() >= 2) {
    const double bbox_width = bbox_max_u - bbox_min_u;
    const double bbox_height = bbox_max_v - bbox_min_v;
    std::vector<double> extents;
    extents.reserve(overlap_candidates.size());
    for (const int64_t face_index : overlap_candidates) {
      const UvTriangle &triangle = triangles[static_cast<size_t>(face_index)];
      extents.push_back(std::max(triangle.max_u - triangle.min_u, triangle.max_v - triangle.min_v));
    }
    std::sort(extents.begin(), extents.end());
    double cell_size = extents[extents.size() / 2];
    if (cell_size <= 0.0) {
      cell_size = std::max(bbox_width, bbox_height) / 16.0;
    }
    if (cell_size <= 0.0) {
      cell_size = 1.0;
    }
    const int64_t cells_x = std::min(
        kMaxGridCellsPerAxis,
        std::max<int64_t>(1, static_cast<int64_t>(std::ceil(bbox_width / cell_size))));
    const int64_t cells_y = std::min(
        kMaxGridCellsPerAxis,
        std::max<int64_t>(1, static_cast<int64_t>(std::ceil(bbox_height / cell_size))));
    const double cell_width = bbox_width > 0.0 ? bbox_width / static_cast<double>(cells_x) : 1.0;
    const double cell_height = bbox_height > 0.0 ? bbox_height / static_cast<double>(cells_y) : 1.0;

    // Per-candidate cell ranges.  Triangles spanning more than
    // kMaxCellSpanPerAxis cells on either axis are kept out of the grid and
    // checked against every candidate instead, bounding grid memory at
    // O(candidates * kMaxCellSpanPerAxis^2) even when a few atlas-spanning
    // triangles meet median-extent cell sizing.
    struct CellRange {
      int64_t min_x = 0;
      int64_t max_x = 0;
      int64_t min_y = 0;
      int64_t max_y = 0;
    };
    std::vector<CellRange> cell_ranges;
    cell_ranges.reserve(overlap_candidates.size());
    std::vector<char> is_large(static_cast<size_t>(face_count), 0);
    std::vector<int64_t> large_triangles;
    for (const int64_t face_index : overlap_candidates) {
      const UvTriangle &triangle = triangles[static_cast<size_t>(face_index)];
      CellRange range;
      range.min_x = clamp_cell((triangle.min_u - bbox_min_u) / cell_width, cells_x);
      range.max_x = clamp_cell((triangle.max_u - bbox_min_u) / cell_width, cells_x);
      range.min_y = clamp_cell((triangle.min_v - bbox_min_v) / cell_height, cells_y);
      range.max_y = clamp_cell((triangle.max_v - bbox_min_v) / cell_height, cells_y);
      cell_ranges.push_back(range);
      if (range.max_x - range.min_x + 1 > kMaxCellSpanPerAxis ||
          range.max_y - range.min_y + 1 > kMaxCellSpanPerAxis) {
        is_large[static_cast<size_t>(face_index)] = 1;
        large_triangles.push_back(face_index);
      }
    }

    std::vector<std::vector<int64_t>> cells(static_cast<size_t>(cells_x * cells_y));
    for (size_t candidate = 0; candidate < overlap_candidates.size(); ++candidate) {
      const int64_t face_index = overlap_candidates[candidate];
      if (is_large[static_cast<size_t>(face_index)]) {
        continue;
      }
      const CellRange &range = cell_ranges[candidate];
      for (int64_t cell_y = range.min_y; cell_y <= range.max_y; ++cell_y) {
        for (int64_t cell_x = range.min_x; cell_x <= range.max_x; ++cell_x) {
          cells[static_cast<size_t>(cell_y * cells_x + cell_x)].push_back(face_index);
        }
      }
    }

    const auto check_pair = [&](int64_t face_a, int64_t face_b) {
      overlap_checked_pairs += 1;
      const UvTriangle &triangle_a = triangles[static_cast<size_t>(face_a)];
      const UvTriangle &triangle_b = triangles[static_cast<size_t>(face_b)];
      if (std::min(triangle_a.max_u, triangle_b.max_u) - std::max(triangle_a.min_u, triangle_b.min_u) <= kUvEps ||
          std::min(triangle_a.max_v, triangle_b.max_v) - std::max(triangle_a.min_v, triangle_b.min_v) <= kUvEps) {
        return;
      }
      if (interiors_overlap(triangle_a, triangle_b)) {
        overlap_count += 1;
      }
    };

    // Pair dedup uses a per-partner "last anchor" stamp instead of a pair
    // set: O(face_count) memory and no per-pair node allocations.  Anchors
    // are visited in ascending face order and partners are restricted to
    // higher indices, so every unordered pair that shares at least one grid
    // cell is checked exactly once -- the same unique-pair set (and counts)
    // as the previous set-based dedup.
    std::vector<int64_t> last_anchor(static_cast<size_t>(face_count), -1);
    for (size_t candidate = 0; candidate < overlap_candidates.size(); ++candidate) {
      const int64_t anchor = overlap_candidates[candidate];
      if (is_large[static_cast<size_t>(anchor)]) {
        continue;
      }
      const CellRange &range = cell_ranges[candidate];
      for (int64_t cell_y = range.min_y; cell_y <= range.max_y; ++cell_y) {
        for (int64_t cell_x = range.min_x; cell_x <= range.max_x; ++cell_x) {
          for (const int64_t partner : cells[static_cast<size_t>(cell_y * cells_x + cell_x)]) {
            if (partner <= anchor || last_anchor[static_cast<size_t>(partner)] == anchor) {
              continue;
            }
            last_anchor[static_cast<size_t>(partner)] = anchor;
            check_pair(anchor, partner);
          }
        }
      }
    }

    // Atlas-spanning triangles excluded from the grid are checked against
    // every candidate, and against each other exactly once per unordered
    // pair, in ascending face order.
    for (const int64_t large : large_triangles) {
      for (const int64_t partner : overlap_candidates) {
        if (partner == large ||
            (is_large[static_cast<size_t>(partner)] && partner < large)) {
          continue;
        }
        check_pair(large, partner);
      }
    }
  }

  const double bbox_area = face_count > 0
      ? (bbox_max_u - bbox_min_u) * (bbox_max_v - bbox_min_v)
      : 0.0;
  const double bbox_utilization = bbox_area > 0.0 ? total_uv_area / bbox_area : 0.0;

  nb::dict result;
  result["uv_flipped_count"] = flipped_count;
  result["uv_degenerate_count"] = degenerate_count;
  result["uv_overlap_count"] = overlap_count;
  result["uv_overlap_checked_pairs"] = overlap_checked_pairs;
  result["uv_stretch_l2"] = global_stretch.l2();
  result["uv_stretch_linf"] = global_stretch.max_gamma;
  result["uv_total_area"] = total_uv_area;
  result["uv_bbox_utilization"] = bbox_utilization;
  if (has_chart_ids) {
    nb::list chart_ids_present;
    nb::list chart_l2;
    nb::list chart_linf;
    for (const auto &[chart_id, accumulator] : chart_stretch) {
      chart_ids_present.append(chart_id);
      chart_l2.append(accumulator.l2());
      chart_linf.append(accumulator.max_gamma);
    }
    result["chart_ids_present"] = chart_ids_present;
    result["chart_stretch_l2"] = chart_l2;
    result["chart_stretch_linf"] = chart_linf;
  }
  return result;
}

}  // namespace mlx_spatialkit
