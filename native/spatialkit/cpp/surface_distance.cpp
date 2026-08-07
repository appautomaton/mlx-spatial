#include "surface_distance.hpp"

#include <Python.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <vector>

#include <nanobind/nanobind.h>

#include "mesh_common.hpp"
#include "parallel.hpp"
#include "triangle_bvh.hpp"

namespace mlx_spatialkit {

namespace nb = nanobind;

nb::dict point_to_mesh_distances(
    nb::object query_points_object,
    nb::object vertices_object,
    nb::object faces_object) {
  mesh_common::validate_matrix(query_points_object, "query points", 3, "float32");
  const int64_t query_count = mesh_common::dimension(query_points_object, "query points", 0);
  if (query_count <= 0) {
    throw nb::value_error("query points must contain at least one point");
  }

  const mesh_common::MeshData mesh = mesh_common::load_mesh(vertices_object, faces_object);
  if (mesh.faces.empty()) {
    throw nb::value_error("mesh faces must contain at least one triangle");
  }

  mesh_common::BufferView query_buffer(query_points_object.ptr(), "query points");
  const Py_buffer &query_view = query_buffer.get();
  std::vector<std::array<float, 3>> query_points(static_cast<size_t>(query_count));
  std::atomic<bool> invalid_query{false};
  parallel_for(static_cast<size_t>(query_count), [&](size_t begin, size_t end) {
    for (size_t row = begin; row < end; ++row) {
      const std::array<float, 3> point{
          mesh_common::read_matrix_value<float>(query_view, static_cast<int64_t>(row), 0),
          mesh_common::read_matrix_value<float>(query_view, static_cast<int64_t>(row), 1),
          mesh_common::read_matrix_value<float>(query_view, static_cast<int64_t>(row), 2),
      };
      if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2])) {
        invalid_query.store(true, std::memory_order_relaxed);
      }
      query_points[row] = point;
    }
  });
  if (invalid_query.load(std::memory_order_relaxed)) {
    throw nb::value_error("query points must contain only finite values");
  }

  std::vector<float> distances(static_cast<size_t>(query_count));
  std::vector<float> closest_points(static_cast<size_t>(query_count) * 3);
  std::vector<int64_t> closest_faces(static_cast<size_t>(query_count));
  const size_t worker_count = native_worker_count(static_cast<size_t>(query_count), 256);
  int64_t bvh_node_count = 0;
  {
    nb::gil_scoped_release release;
    const TriangleBvh bvh(mesh);
    bvh_node_count = bvh.node_count();
    parallel_for(
        static_cast<size_t>(query_count),
        [&](size_t begin, size_t end) {
          for (size_t row = begin; row < end; ++row) {
            const ClosestPointResult closest = bvh.closest_point(query_points[row]);
            distances[row] = static_cast<float>(std::sqrt(closest.distance2));
            closest_points[row * 3] = closest.point[0];
            closest_points[row * 3 + 1] = closest.point[1];
            closest_points[row * 3 + 2] = closest.point[2];
            closest_faces[row] = closest.face_index;
          }
        },
        256,
        64);
  }

  nb::dict stats;
  stats["backend"] = "native-cpu-triangle-bvh";
  stats["query_count"] = query_count;
  stats["mesh_vertices"] = static_cast<int64_t>(mesh.vertices.size());
  stats["mesh_faces"] = static_cast<int64_t>(mesh.faces.size());
  stats["bvh_nodes"] = bvh_node_count;
  stats["workers"] = static_cast<int64_t>(worker_count);
  stats["distance_kind"] = "exact-point-to-triangle-unsigned";

  nb::dict result;
  result["distances"] = mesh_common::make_float32_array(
      std::move(distances), static_cast<size_t>(query_count));
  result["closest_points"] = mesh_common::make_float32_array(
      std::move(closest_points), static_cast<size_t>(query_count), 3);
  result["closest_faces"] = mesh_common::make_int64_array(
      std::move(closest_faces), static_cast<size_t>(query_count));
  result["stats"] = std::move(stats);
  return result;
}

}  // namespace mlx_spatialkit
