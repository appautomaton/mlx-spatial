#include "flexi_dual_grid.hpp"

#include <Python.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

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
  coords.reserve(static_cast<size_t>(token_count));
  std::vector<std::array<bool, 3>> intersected;
  intersected.reserve(static_cast<size_t>(token_count));
  std::vector<float> split_weight;
  split_weight.reserve(static_cast<size_t>(token_count));
  std::vector<std::array<float, 3>> vertices;
  vertices.reserve(static_cast<size_t>(token_count));
  std::unordered_map<Coord3, int64_t, Coord3Hash> coord_to_index;
  coord_to_index.reserve(static_cast<size_t>(token_count));

  constexpr float bounds_min[3] = {-0.5f, -0.5f, -0.5f};
  constexpr float bounds_max[3] = {0.5f, 0.5f, 0.5f};
  const float voxel_size[3] = {
      (bounds_max[0] - bounds_min[0]) / static_cast<float>(grid_size),
      (bounds_max[1] - bounds_min[1]) / static_cast<float>(grid_size),
      (bounds_max[2] - bounds_min[2]) / static_cast<float>(grid_size),
  };

  for (int64_t row = 0; row < token_count; ++row) {
    Coord3 coord{
        read_matrix_value<int32_t>(coordinate_view, row, 1),
        read_matrix_value<int32_t>(coordinate_view, row, 2),
        read_matrix_value<int32_t>(coordinate_view, row, 3),
    };
    coords.push_back(coord);
    coord_to_index[coord] = row;

    std::array<float, 3> vertex{};
    const int32_t coord_values[3] = {coord.z, coord.y, coord.x};
    for (int axis = 0; axis < 3; ++axis) {
      const float field_value = read_matrix_value<float>(field_view, row, axis);
      const float dual_vertex = 2.0f * sigmoid(field_value) - 0.5f;
      vertex[axis] = (static_cast<float>(coord_values[axis]) + dual_vertex) * voxel_size[axis] + bounds_min[axis];
    }
    vertices.push_back(vertex);

    intersected.push_back({
        read_matrix_value<float>(field_view, row, 3) > 0.0f,
        read_matrix_value<float>(field_view, row, 4) > 0.0f,
        read_matrix_value<float>(field_view, row, 5) > 0.0f,
    });
    split_weight.push_back(softplus(read_matrix_value<float>(field_view, row, 6)));
  }

  std::vector<std::array<int64_t, 4>> quads;
  for (int64_t row = 0; row < token_count; ++row) {
    const Coord3 coord = coords[static_cast<size_t>(row)];
    for (int axis = 0; axis < 3; ++axis) {
      if (!intersected[static_cast<size_t>(row)][static_cast<size_t>(axis)]) {
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
        auto found = coord_to_index.find(neighbor);
        if (found == coord_to_index.end()) {
          valid_quad = false;
          break;
        }
        quad[static_cast<size_t>(corner)] = found->second;
      }
      if (valid_quad) {
        quads.push_back(quad);
      }
    }
  }

  if (quads.empty()) {
    nb::dict result;
    result["vertices"] = make_float32_array({}, 0, 3);
    result["faces"] = make_int64_array({}, 0, 3);
    return result;
  }

  std::vector<float> vertex_values;
  vertex_values.reserve(static_cast<size_t>(token_count) * 3);
  for (const std::array<float, 3> &vertex : vertices) {
    vertex_values.push_back(vertex[0]);
    vertex_values.push_back(vertex[1]);
    vertex_values.push_back(vertex[2]);
  }

  std::vector<int64_t> face_values;
  face_values.reserve(quads.size() * 6);
  for (const std::array<int64_t, 4> &quad : quads) {
    const float split_02 =
        split_weight[static_cast<size_t>(quad[0])] * split_weight[static_cast<size_t>(quad[2])];
    const float split_13 =
        split_weight[static_cast<size_t>(quad[1])] * split_weight[static_cast<size_t>(quad[3])];
    const int *split = split_02 > split_13 ? kQuadSplit1 : kQuadSplit2;
    for (int index = 0; index < 6; ++index) {
      face_values.push_back(quad[static_cast<size_t>(split[index])]);
    }
  }

  nb::dict result;
  result["vertices"] = make_float32_array(std::move(vertex_values), static_cast<size_t>(token_count), 3);
  result["faces"] = make_int64_array(std::move(face_values), quads.size() * 2, 3);
  return result;
}

}  // namespace mlx_spatialkit
