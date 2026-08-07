#include "flexi_dual_grid.hpp"

#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "parallel.hpp"
#include "pixal3d_contracts.hpp"

namespace nb = nanobind;

namespace mlx_spatialkit {
namespace {

struct Coord3 {
  int32_t z;
  int32_t y;
  int32_t x;

  bool operator==(const Coord3 &other) const {
    return z == other.z && y == other.y && x == other.x;
  }
};

struct Coord3Hash {
  std::size_t operator()(const Coord3 &coord) const {
    const std::size_t h0 = static_cast<std::size_t>(coord.z) * 73856093u;
    const std::size_t h1 = static_cast<std::size_t>(coord.y) * 19349663u;
    const std::size_t h2 = static_cast<std::size_t>(coord.x) * 83492791u;
    return h0 ^ h1 ^ h2;
  }
};

using CoordIndex = std::unordered_map<Coord3, int64_t, Coord3Hash>;

class BufferView {
 public:
  explicit BufferView(PyObject *object, const char *name) : view_{} {
    if (PyObject_GetBuffer(object, &view_, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
      PyErr_Clear();
      std::ostringstream message;
      message << name << " must expose the Python buffer protocol";
      throw nb::type_error(message.str().c_str());
    }
  }

  BufferView(const BufferView &) = delete;
  BufferView &operator=(const BufferView &) = delete;

  ~BufferView() {
    PyBuffer_Release(&view_);
  }

  const Py_buffer &get() const {
    return view_;
  }

 private:
  Py_buffer view_;
};

template <typename T>
T read_matrix_value(const Py_buffer &view, int64_t row, int64_t col) {
  const auto *base = static_cast<const char *>(view.buf);
  const Py_ssize_t col_count = view.shape != nullptr && view.ndim >= 2 ? view.shape[1] : 0;
  const Py_ssize_t row_stride = view.strides != nullptr ? view.strides[0] : view.itemsize * col_count;
  const Py_ssize_t col_stride = view.strides != nullptr ? view.strides[1] : view.itemsize;
  T value{};
  std::memcpy(&value, base + row * row_stride + col * col_stride, sizeof(T));
  return value;
}

float sigmoid(float value) {
  return 1.0f / (1.0f + std::exp(-value));
}

float softplus(float value) {
  return std::log1p(std::exp(-std::fabs(value))) + std::max(value, 0.0f);
}

nb::object make_float32_array(std::vector<float> values, size_t rows, size_t cols) {
  auto owner = new std::vector<float>(std::move(values));
  nb::capsule capsule(owner, [](void *ptr) noexcept {
    delete static_cast<std::vector<float> *>(ptr);
  });
  return nb::ndarray<nb::numpy, float>(owner->data(), {rows, cols}, capsule).cast();
}

nb::object make_int64_array(std::vector<int64_t> values, size_t rows, size_t cols) {
  auto owner = new std::vector<int64_t>(std::move(values));
  nb::capsule capsule(owner, [](void *ptr) noexcept {
    delete static_cast<std::vector<int64_t> *>(ptr);
  });
  return nb::ndarray<nb::numpy, int64_t>(owner->data(), {rows, cols}, capsule).cast();
}

constexpr int32_t kEdgeNeighborVoxelOffset[3][4][3] = {
    {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}},
    {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}},
    {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}},
};
constexpr int kQuadSplit1[6] = {0, 1, 2, 0, 2, 3};
constexpr int kQuadSplit2[6] = {0, 1, 3, 3, 1, 2};

}  // namespace

nb::dict extract_flexi_dual_grid(nb::object coordinates, nb::object fields, int64_t grid_size) {
  if (grid_size <= 0) {
    throw nb::value_error("grid_size must be positive");
  }

  nb::dict contract = validate_pixal3d_shape_fields(coordinates, fields);
  const auto token_count = nb::cast<int64_t>(contract["token_count"]);

  BufferView coordinate_buffer(coordinates.ptr(), "shape coordinates");
  BufferView field_buffer(fields.ptr(), "shape fields");
  const Py_buffer &coordinate_view = coordinate_buffer.get();
  const Py_buffer &field_view = field_buffer.get();

  std::vector<Coord3> coords;
  coords.resize(static_cast<size_t>(token_count));
  std::vector<std::array<bool, 3>> intersected;
  intersected.resize(static_cast<size_t>(token_count));
  std::vector<float> split_weight;
  split_weight.resize(static_cast<size_t>(token_count));
  std::vector<std::array<float, 3>> vertices;
  vertices.resize(static_cast<size_t>(token_count));
  constexpr float bounds_min[3] = {-0.5f, -0.5f, -0.5f};
  constexpr float bounds_max[3] = {0.5f, 0.5f, 0.5f};
  const float voxel_size[3] = {
      (bounds_max[0] - bounds_min[0]) / static_cast<float>(grid_size),
      (bounds_max[1] - bounds_min[1]) / static_cast<float>(grid_size),
      (bounds_max[2] - bounds_min[2]) / static_cast<float>(grid_size),
  };

  std::vector<std::array<int64_t, 4>> quads;
  std::vector<float> vertex_values;
  std::vector<int64_t> face_values;
  {
    nb::gil_scoped_release release;
    parallel_for(static_cast<size_t>(token_count), [&](size_t begin, size_t end) {
      for (size_t row = begin; row < end; ++row) {
        Coord3 coord{
            read_matrix_value<int32_t>(coordinate_view, static_cast<int64_t>(row), 1),
            read_matrix_value<int32_t>(coordinate_view, static_cast<int64_t>(row), 2),
            read_matrix_value<int32_t>(coordinate_view, static_cast<int64_t>(row), 3),
        };
        coords[row] = coord;

        std::array<float, 3> vertex{};
        const int32_t coord_values[3] = {coord.z, coord.y, coord.x};
        for (int axis = 0; axis < 3; ++axis) {
          const float field_value =
              read_matrix_value<float>(field_view, static_cast<int64_t>(row), axis);
          const float dual_vertex = 2.0f * sigmoid(field_value) - 0.5f;
          vertex[axis] =
              (static_cast<float>(coord_values[axis]) + dual_vertex) * voxel_size[axis] + bounds_min[axis];
        }
        vertices[row] = vertex;

        intersected[row] = {
            read_matrix_value<float>(field_view, static_cast<int64_t>(row), 3) > 0.0f,
            read_matrix_value<float>(field_view, static_cast<int64_t>(row), 4) > 0.0f,
            read_matrix_value<float>(field_view, static_cast<int64_t>(row), 5) > 0.0f,
        };
        split_weight[row] = softplus(read_matrix_value<float>(field_view, static_cast<int64_t>(row), 6));
      }
    });

    const size_t worker_count = native_worker_count(static_cast<size_t>(token_count));
    const size_t shard_count = std::max<size_t>(1, worker_count * 4);
    std::vector<std::vector<int64_t>> shard_rows(shard_count);
    const size_t rows_per_shard =
        (static_cast<size_t>(token_count) + shard_count - 1) / shard_count;
    for (auto &rows : shard_rows) {
      rows.reserve(rows_per_shard);
    }
    for (int64_t row = 0; row < token_count; ++row) {
      const Coord3 &coord = coords[static_cast<size_t>(row)];
      const size_t shard = Coord3Hash{}(coord) % shard_count;
      shard_rows[shard].push_back(row);
    }

    std::vector<CoordIndex> coordinate_shards(shard_count);
    parallel_for(
        shard_count,
        [&](size_t begin, size_t end) {
          for (size_t shard = begin; shard < end; ++shard) {
            auto &index = coordinate_shards[shard];
            index.reserve(shard_rows[shard].size());
            for (const int64_t row : shard_rows[shard]) {
              index[coords[static_cast<size_t>(row)]] = row;
            }
          }
        },
        1,
        1);
    std::vector<std::vector<int64_t>>().swap(shard_rows);

    constexpr size_t kQuadChunkSize = 16384;
    const size_t chunk_count =
        (static_cast<size_t>(token_count) + kQuadChunkSize - 1) / kQuadChunkSize;
    std::vector<std::vector<std::array<int64_t, 4>>> chunk_quads(chunk_count);
    const auto &coordinate_index = coordinate_shards;
    parallel_for(
        static_cast<size_t>(token_count),
        [&](size_t begin, size_t end) {
          auto &local_quads = chunk_quads[begin / kQuadChunkSize];
          for (size_t row = begin; row < end; ++row) {
            const Coord3 coord = coords[row];
            for (int axis = 0; axis < 3; ++axis) {
              if (!intersected[row][static_cast<size_t>(axis)]) {
                continue;
              }

              std::array<int64_t, 4> quad{};
              bool valid_quad = true;
              for (int corner = 0; corner < 4; ++corner) {
                Coord3 neighbor{
                    coord.z + kEdgeNeighborVoxelOffset[axis][corner][0],
                    coord.y + kEdgeNeighborVoxelOffset[axis][corner][1],
                    coord.x + kEdgeNeighborVoxelOffset[axis][corner][2],
                };
                const size_t neighbor_shard = Coord3Hash{}(neighbor) % shard_count;
                const auto &shard_index = coordinate_index[neighbor_shard];
                auto found = shard_index.find(neighbor);
                if (found == shard_index.end()) {
                  valid_quad = false;
                  break;
                }
                quad[static_cast<size_t>(corner)] = found->second;
              }
              if (valid_quad) {
                local_quads.push_back(quad);
              }
            }
          }
        },
        2048,
        kQuadChunkSize);

    size_t quad_count = 0;
    for (const auto &local_quads : chunk_quads) {
      quad_count += local_quads.size();
    }
    quads.reserve(quad_count);
    for (auto &local_quads : chunk_quads) {
      quads.insert(
          quads.end(),
          std::make_move_iterator(local_quads.begin()),
          std::make_move_iterator(local_quads.end()));
    }

    std::vector<Coord3>().swap(coords);
    std::vector<std::array<bool, 3>>().swap(intersected);
    std::vector<CoordIndex>().swap(coordinate_shards);

    if (!quads.empty()) {
      vertex_values.resize(static_cast<size_t>(token_count) * 3);
      parallel_for(static_cast<size_t>(token_count), [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) {
          const std::array<float, 3> &vertex = vertices[row];
          vertex_values[row * 3] = vertex[0];
          vertex_values[row * 3 + 1] = vertex[1];
          vertex_values[row * 3 + 2] = vertex[2];
        }
      });

      face_values.resize(quads.size() * 6);
      parallel_for(quads.size(), [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) {
          const std::array<int64_t, 4> &quad = quads[row];
          const float split_02 =
              split_weight[static_cast<size_t>(quad[0])] * split_weight[static_cast<size_t>(quad[2])];
          const float split_13 =
              split_weight[static_cast<size_t>(quad[1])] * split_weight[static_cast<size_t>(quad[3])];
          const int *split = split_02 > split_13 ? kQuadSplit1 : kQuadSplit2;
          for (int index = 0; index < 6; ++index) {
            face_values[row * 6 + static_cast<size_t>(index)] = quad[static_cast<size_t>(split[index])];
          }
        }
      });
    }
  }

  nb::dict result;
  const size_t vertex_count = quads.empty() ? 0 : static_cast<size_t>(token_count);
  result["vertices"] = make_float32_array(std::move(vertex_values), vertex_count, 3);
  result["faces"] = make_int64_array(std::move(face_values), quads.size() * 2, 3);
  return result;
}

}  // namespace mlx_spatialkit
