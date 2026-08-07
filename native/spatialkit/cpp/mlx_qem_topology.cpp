#include "mlx_qem_topology.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mesh_common.hpp"
#include "parallel.hpp"

namespace mlx_spatialkit {
namespace {

namespace nb = nanobind;
using mesh_common::BufferView;
using mesh_common::EdgeKey;
using mesh_common::EdgeKeyHash;

struct EdgeRecord {
  int32_t a;
  int32_t b;
  int32_t face_count;
};

struct Csr {
  std::vector<int32_t> offsets;
  std::vector<int32_t> items;
};

bool edge_less(const EdgeRecord &left, const EdgeRecord &right) {
  return left.a < right.a || (left.a == right.a && left.b < right.b);
}

std::vector<EdgeRecord> merge_edge_runs(
    const std::vector<EdgeRecord> &left,
    const std::vector<EdgeRecord> &right) {
  std::vector<EdgeRecord> merged;
  merged.reserve(left.size() + right.size());
  size_t left_index = 0;
  size_t right_index = 0;
  while (left_index < left.size() || right_index < right.size()) {
    if (right_index >= right.size() ||
        (left_index < left.size() && edge_less(left[left_index], right[right_index]))) {
      merged.push_back(left[left_index++]);
      continue;
    }
    if (left_index >= left.size() || edge_less(right[right_index], left[left_index])) {
      merged.push_back(right[right_index++]);
      continue;
    }
    const int64_t count = static_cast<int64_t>(left[left_index].face_count) +
                          static_cast<int64_t>(right[right_index].face_count);
    if (count > std::numeric_limits<int32_t>::max()) {
      throw std::overflow_error("MLX QEM edge incidence count exceeds int32 capacity");
    }
    merged.push_back(EdgeRecord{
        left[left_index].a,
        left[left_index].b,
        static_cast<int32_t>(count),
    });
    ++left_index;
    ++right_index;
  }
  return merged;
}

std::vector<EdgeRecord> build_sorted_edges(
    const std::vector<std::array<int32_t, 3>> &faces,
    size_t worker_count) {
  if (faces.empty()) {
    return {};
  }
  const size_t shard_count = std::min(faces.size(), std::max<size_t>(1, worker_count * 4));
  std::vector<std::vector<EdgeRecord>> runs(shard_count);
  parallel_for(
      shard_count,
      [&](size_t shard_begin, size_t shard_end) {
        for (size_t shard = shard_begin; shard < shard_end; ++shard) {
          const size_t begin = faces.size() * shard / shard_count;
          const size_t end = faces.size() * (shard + 1) / shard_count;
          std::unordered_map<EdgeKey, int32_t, EdgeKeyHash> counts;
          counts.reserve(std::max<size_t>(16, (end - begin) * 2));
          for (size_t face_index = begin; face_index < end; ++face_index) {
            const auto &face = faces[face_index];
            const std::array<std::pair<int32_t, int32_t>, 3> pairs{{
                {face[0], face[1]},
                {face[1], face[2]},
                {face[2], face[0]},
            }};
            for (const auto &[first, second] : pairs) {
              const EdgeKey edge{
                  static_cast<int64_t>(std::min(first, second)),
                  static_cast<int64_t>(std::max(first, second)),
              };
              auto [iterator, inserted] = counts.try_emplace(edge, 0);
              if (iterator->second == std::numeric_limits<int32_t>::max()) {
                throw std::overflow_error("MLX QEM edge incidence count exceeds int32 capacity");
              }
              iterator->second += 1;
            }
          }
          auto &run = runs[shard];
          run.reserve(counts.size());
          for (const auto &[edge, count] : counts) {
            run.push_back(EdgeRecord{
                static_cast<int32_t>(edge.a),
                static_cast<int32_t>(edge.b),
                count,
            });
          }
          std::sort(run.begin(), run.end(), edge_less);
        }
      },
      1,
      1);

  while (runs.size() > 1) {
    const size_t next_count = (runs.size() + 1) / 2;
    std::vector<std::vector<EdgeRecord>> next(next_count);
    parallel_for(
        next_count,
        [&](size_t pair_begin, size_t pair_end) {
          for (size_t pair = pair_begin; pair < pair_end; ++pair) {
            const size_t left = pair * 2;
            const size_t right = left + 1;
            if (right >= runs.size()) {
              next[pair] = std::move(runs[left]);
            } else {
              next[pair] = merge_edge_runs(runs[left], runs[right]);
            }
          }
        },
        1,
        1);
    runs = std::move(next);
  }
  return std::move(runs.front());
}

template <typename Visit>
Csr build_sorted_csr(
    size_t vertex_count,
    size_t source_count,
    size_t incidence_count,
    const Visit &visit) {
  if (incidence_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw nb::value_error("MLX QEM CSR exceeds int32 indexing capacity");
  }
  std::vector<std::atomic<int32_t>> degrees(vertex_count);
  parallel_for(vertex_count, [&](size_t begin, size_t end) {
    for (size_t vertex = begin; vertex < end; ++vertex) {
      degrees[vertex].store(0, std::memory_order_relaxed);
    }
  });
  parallel_for(source_count, [&](size_t begin, size_t end) {
    for (size_t source = begin; source < end; ++source) {
      visit(source, [&](int32_t vertex, int32_t) {
        degrees[static_cast<size_t>(vertex)].fetch_add(1, std::memory_order_relaxed);
      });
    }
  });

  Csr csr;
  csr.offsets.resize(vertex_count + 1);
  csr.offsets[0] = 0;
  int64_t offset = 0;
  for (size_t vertex = 0; vertex < vertex_count; ++vertex) {
    offset += degrees[vertex].load(std::memory_order_relaxed);
    if (offset > std::numeric_limits<int32_t>::max()) {
      throw nb::value_error("MLX QEM CSR exceeds int32 indexing capacity");
    }
    csr.offsets[vertex + 1] = static_cast<int32_t>(offset);
  }
  if (static_cast<size_t>(offset) != incidence_count) {
    throw std::runtime_error("MLX QEM CSR incidence count mismatch");
  }

  csr.items.resize(incidence_count);
  std::vector<std::atomic<int32_t>> cursors(vertex_count);
  parallel_for(vertex_count, [&](size_t begin, size_t end) {
    for (size_t vertex = begin; vertex < end; ++vertex) {
      cursors[vertex].store(csr.offsets[vertex], std::memory_order_relaxed);
    }
  });
  parallel_for(source_count, [&](size_t begin, size_t end) {
    for (size_t source = begin; source < end; ++source) {
      visit(source, [&](int32_t vertex, int32_t item) {
        const int32_t destination =
            cursors[static_cast<size_t>(vertex)].fetch_add(1, std::memory_order_relaxed);
        csr.items[static_cast<size_t>(destination)] = item;
      });
    }
  });
  parallel_for(vertex_count, [&](size_t begin, size_t end) {
    for (size_t vertex = begin; vertex < end; ++vertex) {
      std::sort(
          csr.items.begin() + csr.offsets[vertex],
          csr.items.begin() + csr.offsets[vertex + 1]);
    }
  });
  return csr;
}

}  // namespace

nb::dict build_mlx_qem_topology(int64_t vertex_count, nb::object faces_object) {
  const auto started = std::chrono::steady_clock::now();
  if (vertex_count <= 0 || vertex_count > std::numeric_limits<int32_t>::max()) {
    throw nb::value_error("vertex_count must be in the int32 indexing range");
  }
  mesh_common::validate_matrix(faces_object, "MLX QEM faces", 3, "int32");
  const int64_t face_count_i64 = mesh_common::dimension(faces_object, "MLX QEM faces", 0);
  if (face_count_i64 < 0 ||
      face_count_i64 > static_cast<int64_t>(std::numeric_limits<int32_t>::max() / 3)) {
    throw nb::value_error("MLX QEM face adjacency exceeds int32 indexing capacity");
  }
  const size_t face_count = static_cast<size_t>(face_count_i64);
  const size_t vertex_count_size = static_cast<size_t>(vertex_count);
  const size_t worker_count = native_worker_count(std::max(face_count, vertex_count_size));

  BufferView face_buffer(faces_object.ptr(), "MLX QEM faces");
  const Py_buffer &face_view = face_buffer.get();
  std::vector<std::array<int32_t, 3>> faces(face_count);
  std::atomic<bool> invalid_face{false};
  parallel_for(face_count, [&](size_t begin, size_t end) {
    for (size_t row = begin; row < end; ++row) {
      auto &face = faces[row];
      for (int corner = 0; corner < 3; ++corner) {
        face[static_cast<size_t>(corner)] = mesh_common::read_matrix_value<int32_t>(
            face_view,
            static_cast<int64_t>(row),
            corner);
        if (face[static_cast<size_t>(corner)] < 0 ||
            face[static_cast<size_t>(corner)] >= vertex_count) {
          invalid_face.store(true, std::memory_order_relaxed);
        }
      }
    }
  });
  if (invalid_face.load(std::memory_order_relaxed)) {
    throw nb::value_error("MLX QEM faces contain indices outside the vertex array");
  }

  std::vector<EdgeRecord> edge_records = build_sorted_edges(faces, worker_count);
  Csr vertex_faces = build_sorted_csr(
      vertex_count_size,
      face_count,
      face_count * 3,
      [&](size_t face_index, const auto &emit) {
        const auto &face = faces[face_index];
        const int32_t item = static_cast<int32_t>(face_index);
        emit(face[0], item);
        emit(face[1], item);
        emit(face[2], item);
      });
  Csr vertex_edges = build_sorted_csr(
      vertex_count_size,
      edge_records.size(),
      edge_records.size() * 2,
      [&](size_t edge_index, const auto &emit) {
        const auto &edge = edge_records[edge_index];
        const int32_t item = static_cast<int32_t>(edge_index);
        emit(edge.a, item);
        emit(edge.b, item);
      });

  std::vector<int32_t> face_values(face_count * 3);
  parallel_for(face_count, [&](size_t begin, size_t end) {
    for (size_t face_index = begin; face_index < end; ++face_index) {
      face_values[face_index * 3] = faces[face_index][0];
      face_values[face_index * 3 + 1] = faces[face_index][1];
      face_values[face_index * 3 + 2] = faces[face_index][2];
    }
  });
  std::vector<int32_t> edge_values(edge_records.size() * 2);
  std::vector<int32_t> edge_face_counts(edge_records.size());
  std::vector<uint8_t> boundary_vertices(vertex_count_size, 0);
  for (size_t edge_index = 0; edge_index < edge_records.size(); ++edge_index) {
    const auto &edge = edge_records[edge_index];
    edge_values[edge_index * 2] = edge.a;
    edge_values[edge_index * 2 + 1] = edge.b;
    edge_face_counts[edge_index] = edge.face_count;
    if (edge.face_count != 2) {
      boundary_vertices[static_cast<size_t>(edge.a)] = 1;
      boundary_vertices[static_cast<size_t>(edge.b)] = 1;
    }
  }

  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  nb::dict stats;
  stats["backend"] = "native-cpp-sharded-sorted-csr";
  stats["framework"] = "native-c++";
  stats["execution_device"] = "cpu";
  stats["cpu_workers"] = static_cast<int64_t>(worker_count);
  stats["seconds"] = seconds;

  nb::dict result;
  result["faces"] = mesh_common::make_int32_array(std::move(face_values), face_count, 3);
  result["edges"] = mesh_common::make_int32_array(std::move(edge_values), edge_records.size(), 2);
  result["edge_face_counts"] = mesh_common::make_int32_array(
      std::move(edge_face_counts), edge_records.size());
  result["boundary_vertices"] = mesh_common::make_uint8_array(
      std::move(boundary_vertices), vertex_count_size);
  result["vertex_face_offsets"] = mesh_common::make_int32_array(
      std::move(vertex_faces.offsets), vertex_count_size + 1);
  result["vertex_faces"] = mesh_common::make_int32_array(
      std::move(vertex_faces.items), face_count * 3);
  result["vertex_edge_offsets"] = mesh_common::make_int32_array(
      std::move(vertex_edges.offsets), vertex_count_size + 1);
  result["vertex_edges"] = mesh_common::make_int32_array(
      std::move(vertex_edges.items), edge_records.size() * 2);
  result["stats"] = std::move(stats);
  return result;
}

nb::dict apply_mlx_qem_collapse_batch(
    nb::object vertices_object,
    nb::object faces_object,
    nb::object selected_edges_object,
    nb::object boundary_vertices_object) {
  const auto started = std::chrono::steady_clock::now();
  mesh_common::validate_matrix(vertices_object, "MLX QEM vertices", 3, "float32");
  mesh_common::validate_matrix(faces_object, "MLX QEM faces", 3, "int32");
  mesh_common::validate_matrix(selected_edges_object, "MLX QEM selected edges", 2, "int32");
  mesh_common::validate_vector(boundary_vertices_object, "MLX QEM boundary vertices", "uint8");
  const size_t vertex_count = static_cast<size_t>(
      mesh_common::dimension(vertices_object, "MLX QEM vertices", 0));
  const size_t face_count = static_cast<size_t>(
      mesh_common::dimension(faces_object, "MLX QEM faces", 0));
  const size_t collapse_count = static_cast<size_t>(
      mesh_common::dimension(selected_edges_object, "MLX QEM selected edges", 0));
  const size_t boundary_count = static_cast<size_t>(
      mesh_common::dimension(boundary_vertices_object, "MLX QEM boundary vertices", 0));
  if (vertex_count == 0) {
    throw nb::value_error("MLX QEM vertices must not be empty");
  }
  if (boundary_count != vertex_count) {
    throw nb::value_error("MLX QEM boundary vertex count must match vertices");
  }

  BufferView vertex_buffer(vertices_object.ptr(), "MLX QEM vertices");
  BufferView face_buffer(faces_object.ptr(), "MLX QEM faces");
  BufferView selected_edge_buffer(selected_edges_object.ptr(), "MLX QEM selected edges");
  BufferView boundary_buffer(boundary_vertices_object.ptr(), "MLX QEM boundary vertices");
  const Py_buffer &vertex_view = vertex_buffer.get();
  const Py_buffer &face_view = face_buffer.get();
  const Py_buffer &selected_edge_view = selected_edge_buffer.get();
  const Py_buffer &boundary_view = boundary_buffer.get();

  std::vector<std::array<float, 3>> vertices(vertex_count);
  std::vector<std::array<int32_t, 3>> faces(face_count);
  std::vector<std::array<int32_t, 2>> selected_edges(collapse_count);
  std::vector<uint8_t> boundary_vertices(vertex_count);
  std::atomic<bool> invalid_vertex{false};
  std::atomic<bool> invalid_face{false};
  std::atomic<bool> invalid_edge{false};
  parallel_for(vertex_count, [&](size_t begin, size_t end) {
    for (size_t row = begin; row < end; ++row) {
      auto &vertex = vertices[row];
      for (int axis = 0; axis < 3; ++axis) {
        vertex[static_cast<size_t>(axis)] = mesh_common::read_matrix_value<float>(
            vertex_view, static_cast<int64_t>(row), axis);
        if (!std::isfinite(vertex[static_cast<size_t>(axis)])) {
          invalid_vertex.store(true, std::memory_order_relaxed);
        }
      }
      boundary_vertices[row] = mesh_common::read_matrix_value<uint8_t>(
          boundary_view, static_cast<int64_t>(row), 0);
    }
  });
  parallel_for(face_count, [&](size_t begin, size_t end) {
    for (size_t row = begin; row < end; ++row) {
      auto &face = faces[row];
      for (int corner = 0; corner < 3; ++corner) {
        face[static_cast<size_t>(corner)] = mesh_common::read_matrix_value<int32_t>(
            face_view, static_cast<int64_t>(row), corner);
        if (face[static_cast<size_t>(corner)] < 0 ||
            static_cast<size_t>(face[static_cast<size_t>(corner)]) >= vertex_count) {
          invalid_face.store(true, std::memory_order_relaxed);
        }
      }
    }
  });
  parallel_for(collapse_count, [&](size_t begin, size_t end) {
    for (size_t row = begin; row < end; ++row) {
      auto &edge = selected_edges[row];
      for (int endpoint = 0; endpoint < 2; ++endpoint) {
        edge[static_cast<size_t>(endpoint)] = mesh_common::read_matrix_value<int32_t>(
            selected_edge_view, static_cast<int64_t>(row), endpoint);
      }
      if (edge[0] < 0 || edge[1] < 0 || edge[0] == edge[1] ||
          static_cast<size_t>(edge[0]) >= vertex_count ||
          static_cast<size_t>(edge[1]) >= vertex_count) {
        invalid_edge.store(true, std::memory_order_relaxed);
      }
    }
  });
  if (invalid_vertex.load(std::memory_order_relaxed)) {
    throw nb::value_error("MLX QEM vertices must contain only finite values");
  }
  if (invalid_face.load(std::memory_order_relaxed)) {
    throw nb::value_error("MLX QEM faces contain indices outside the vertex array");
  }
  if (invalid_edge.load(std::memory_order_relaxed)) {
    throw nb::value_error("MLX QEM selected edges must contain distinct valid endpoints");
  }
  std::vector<uint8_t> touched(vertex_count, 0);
  for (const auto &edge : selected_edges) {
    if (touched[static_cast<size_t>(edge[0])] != 0 ||
        touched[static_cast<size_t>(edge[1])] != 0) {
      throw nb::value_error("MLX QEM selected edges must be endpoint-disjoint");
    }
    touched[static_cast<size_t>(edge[0])] = 1;
    touched[static_cast<size_t>(edge[1])] = 1;
  }

  const size_t worker_count = native_worker_count(std::max(vertex_count, face_count));
  std::vector<std::array<float, 3>> moved_vertices;
  std::vector<int32_t> vertex_map;
  std::vector<std::array<int32_t, 3>> remapped_faces;
  std::vector<uint8_t> keep_faces;
  std::vector<int32_t> compact_vertex_map;
  std::vector<int64_t> compact_face_offsets;
  size_t final_vertex_count = 0;
  size_t final_face_count = 0;
  {
    nb::gil_scoped_release release;
    moved_vertices = vertices;
    parallel_for(collapse_count, [&](size_t begin, size_t end) {
      for (size_t index = begin; index < end; ++index) {
        const int32_t keep = selected_edges[index][0];
        const int32_t drop = selected_edges[index][1];
        const bool keep_boundary = boundary_vertices[static_cast<size_t>(keep)] != 0;
        const bool drop_boundary = boundary_vertices[static_cast<size_t>(drop)] != 0;
        auto &target = moved_vertices[static_cast<size_t>(keep)];
        if (keep_boundary && !drop_boundary) {
          target = vertices[static_cast<size_t>(keep)];
        } else if (!keep_boundary && drop_boundary) {
          target = vertices[static_cast<size_t>(drop)];
        } else {
          for (int axis = 0; axis < 3; ++axis) {
            target[static_cast<size_t>(axis)] = 0.5f * (
                vertices[static_cast<size_t>(keep)][static_cast<size_t>(axis)] +
                vertices[static_cast<size_t>(drop)][static_cast<size_t>(axis)]);
          }
        }
      }
    }, 256, 64);

    vertex_map.resize(vertex_count);
    parallel_for(vertex_count, [&](size_t begin, size_t end) {
      for (size_t vertex = begin; vertex < end; ++vertex) {
        vertex_map[vertex] = static_cast<int32_t>(vertex);
      }
    });
    parallel_for(collapse_count, [&](size_t begin, size_t end) {
      for (size_t index = begin; index < end; ++index) {
        vertex_map[static_cast<size_t>(selected_edges[index][1])] = selected_edges[index][0];
      }
    }, 256, 64);

    remapped_faces.resize(face_count);
    keep_faces.resize(face_count);
    parallel_for(face_count, [&](size_t begin, size_t end) {
      for (size_t face_index = begin; face_index < end; ++face_index) {
        auto &remapped = remapped_faces[face_index];
        const auto &face = faces[face_index];
        remapped = {
            vertex_map[static_cast<size_t>(face[0])],
            vertex_map[static_cast<size_t>(face[1])],
            vertex_map[static_cast<size_t>(face[2])],
        };
        keep_faces[face_index] = static_cast<uint8_t>(
            remapped[0] != remapped[1] &&
            remapped[1] != remapped[2] &&
            remapped[0] != remapped[2]);
      }
    });

    std::vector<std::atomic<uint8_t>> used_vertices(vertex_count);
    parallel_for(vertex_count, [&](size_t begin, size_t end) {
      for (size_t vertex = begin; vertex < end; ++vertex) {
        used_vertices[vertex].store(0, std::memory_order_relaxed);
      }
    });
    parallel_for(face_count, [&](size_t begin, size_t end) {
      for (size_t face_index = begin; face_index < end; ++face_index) {
        if (keep_faces[face_index] == 0) {
          continue;
        }
        for (int corner = 0; corner < 3; ++corner) {
          used_vertices[static_cast<size_t>(remapped_faces[face_index][static_cast<size_t>(corner)])]
              .store(1, std::memory_order_relaxed);
        }
      }
    });

    compact_vertex_map.resize(vertex_count);
    int64_t vertex_offset = 0;
    for (size_t vertex = 0; vertex < vertex_count; ++vertex) {
      if (used_vertices[vertex].load(std::memory_order_relaxed) != 0) {
        if (vertex_offset > std::numeric_limits<int32_t>::max()) {
          throw std::overflow_error("MLX QEM compact vertices exceed int32 capacity");
        }
        compact_vertex_map[vertex] = static_cast<int32_t>(vertex_offset++);
      } else {
        compact_vertex_map[vertex] = -1;
      }
    }
    final_vertex_count = static_cast<size_t>(vertex_offset);

    compact_face_offsets.resize(face_count + 1);
    compact_face_offsets[0] = 0;
    for (size_t face_index = 0; face_index < face_count; ++face_index) {
      compact_face_offsets[face_index + 1] = compact_face_offsets[face_index] + keep_faces[face_index];
    }
    final_face_count = static_cast<size_t>(compact_face_offsets.back());
  }

  std::vector<float> output_vertices(final_vertex_count * 3);
  parallel_for(vertex_count, [&](size_t begin, size_t end) {
    for (size_t vertex = begin; vertex < end; ++vertex) {
      const int32_t destination = compact_vertex_map[vertex];
      if (destination < 0) {
        continue;
      }
      for (int axis = 0; axis < 3; ++axis) {
        output_vertices[static_cast<size_t>(destination) * 3 + static_cast<size_t>(axis)] =
            moved_vertices[vertex][static_cast<size_t>(axis)];
      }
    }
  });
  std::vector<int32_t> output_faces(final_face_count * 3);
  parallel_for(face_count, [&](size_t begin, size_t end) {
    for (size_t face_index = begin; face_index < end; ++face_index) {
      if (keep_faces[face_index] == 0) {
        continue;
      }
      const size_t destination = static_cast<size_t>(compact_face_offsets[face_index]);
      for (int corner = 0; corner < 3; ++corner) {
        output_faces[destination * 3 + static_cast<size_t>(corner)] = compact_vertex_map[
            static_cast<size_t>(remapped_faces[face_index][static_cast<size_t>(corner)])];
      }
    }
  });

  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  nb::dict stats;
  stats["backend"] = "native-cpp-parallel-collapse-compact";
  stats["cpu_workers"] = static_cast<int64_t>(worker_count);
  stats["seconds"] = seconds;
  stats["source_vertices"] = static_cast<int64_t>(vertex_count);
  stats["source_faces"] = static_cast<int64_t>(face_count);
  stats["selected_edges"] = static_cast<int64_t>(collapse_count);
  stats["final_vertices"] = static_cast<int64_t>(final_vertex_count);
  stats["final_faces"] = static_cast<int64_t>(final_face_count);

  nb::dict result;
  result["vertices"] = mesh_common::make_float32_array(
      std::move(output_vertices), final_vertex_count, 3);
  result["faces"] = mesh_common::make_int32_array(
      std::move(output_faces), final_face_count, 3);
  result["stats"] = std::move(stats);
  return result;
}

}  // namespace mlx_spatialkit
