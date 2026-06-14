#pragma once

#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

namespace mlx_spatialkit::mesh_common {

namespace nb = nanobind;

struct MeshData {
  std::vector<std::array<float, 3>> vertices;
  std::vector<std::array<int64_t, 3>> faces;
};

struct EdgeKey {
  int64_t a;
  int64_t b;

  bool operator==(const EdgeKey &other) const {
    return a == other.a && b == other.b;
  }
};

struct EdgeKeyHash {
  std::size_t operator()(const EdgeKey &edge) const {
    return static_cast<std::size_t>(edge.a * 1000003LL) ^ static_cast<std::size_t>(edge.b);
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

inline std::string dtype_name(nb::object array, const char *name) {
  if (!PyObject_HasAttrString(array.ptr(), "dtype")) {
    std::ostringstream message;
    message << name << " must expose a dtype";
    throw nb::type_error(message.str().c_str());
  }
  return nb::cast<std::string>(nb::str(nb::getattr(array, "dtype")));
}

inline int64_t dimension(nb::object array, const char *name, int axis) {
  nb::tuple shape = nb::cast<nb::tuple>(nb::getattr(array, "shape"));
  if (shape.size() <= static_cast<size_t>(axis)) {
    std::ostringstream message;
    message << name << " shape is missing axis " << axis;
    throw nb::value_error(message.str().c_str());
  }
  return nb::cast<int64_t>(shape[axis]);
}

inline void validate_matrix(nb::object array, const char *name, int64_t cols, const char *dtype) {
  const auto ndim = nb::cast<int64_t>(nb::getattr(array, "ndim"));
  if (ndim != 2) {
    std::ostringstream message;
    message << name << " must have rank 2, got rank " << ndim;
    throw nb::value_error(message.str().c_str());
  }
  if (dimension(array, name, 1) != cols) {
    std::ostringstream message;
    message << name << " must have shape (n, " << cols << ")";
    throw nb::value_error(message.str().c_str());
  }
  const std::string actual_dtype = dtype_name(array, name);
  if (actual_dtype != dtype) {
    std::ostringstream message;
    message << name << " must have dtype " << dtype << ", got " << actual_dtype;
    throw nb::value_error(message.str().c_str());
  }
}

inline MeshData load_mesh(nb::object vertices_object, nb::object faces_object) {
  validate_matrix(vertices_object, "mesh vertices", 3, "float32");
  validate_matrix(faces_object, "mesh faces", 3, "int64");
  const int64_t vertex_count = dimension(vertices_object, "mesh vertices", 0);
  const int64_t face_count = dimension(faces_object, "mesh faces", 0);
  if (vertex_count <= 0) {
    throw nb::value_error("mesh vertices must contain at least one vertex");
  }

  BufferView vertex_buffer(vertices_object.ptr(), "mesh vertices");
  BufferView face_buffer(faces_object.ptr(), "mesh faces");
  MeshData mesh;
  mesh.vertices.reserve(static_cast<size_t>(vertex_count));
  mesh.faces.reserve(static_cast<size_t>(face_count));
  const Py_buffer &vertex_view = vertex_buffer.get();
  const Py_buffer &face_view = face_buffer.get();
  for (int64_t row = 0; row < vertex_count; ++row) {
    std::array<float, 3> vertex{
        read_matrix_value<float>(vertex_view, row, 0),
        read_matrix_value<float>(vertex_view, row, 1),
        read_matrix_value<float>(vertex_view, row, 2),
    };
    if (!std::isfinite(vertex[0]) || !std::isfinite(vertex[1]) || !std::isfinite(vertex[2])) {
      throw nb::value_error("mesh vertices must contain only finite values");
    }
    mesh.vertices.push_back(vertex);
  }
  for (int64_t row = 0; row < face_count; ++row) {
    std::array<int64_t, 3> face{
        read_matrix_value<int64_t>(face_view, row, 0),
        read_matrix_value<int64_t>(face_view, row, 1),
        read_matrix_value<int64_t>(face_view, row, 2),
    };
    for (int i = 0; i < 3; ++i) {
      if (face[i] < 0 || face[i] >= vertex_count) {
        throw nb::value_error("mesh faces contain vertex indices outside the vertex array");
      }
    }
    mesh.faces.push_back(face);
  }
  return mesh;
}

inline float triangle_area2(const MeshData &mesh, const std::array<int64_t, 3> &face) {
  const auto &a = mesh.vertices[static_cast<size_t>(face[0])];
  const auto &b = mesh.vertices[static_cast<size_t>(face[1])];
  const auto &c = mesh.vertices[static_cast<size_t>(face[2])];
  const float ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const float ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  const float cross[3] = {
      ab[1] * ac[2] - ab[2] * ac[1],
      ab[2] * ac[0] - ab[0] * ac[2],
      ab[0] * ac[1] - ab[1] * ac[0],
  };
  return std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
}

inline bool face_degenerate(const MeshData &mesh, const std::array<int64_t, 3> &face) {
  if (face[0] == face[1] || face[1] == face[2] || face[0] == face[2]) {
    return true;
  }
  const float area2 = triangle_area2(mesh, face);
  return !std::isfinite(area2) || area2 <= 1e-14f;
}

inline std::array<int64_t, 3> canonical_face(std::array<int64_t, 3> face) {
  std::sort(face.begin(), face.end());
  return face;
}

inline EdgeKey edge_key(int64_t left, int64_t right) {
  return EdgeKey{std::min(left, right), std::max(left, right)};
}

inline std::unordered_map<EdgeKey, int64_t, EdgeKeyHash> edge_counts(const std::vector<std::array<int64_t, 3>> &faces) {
  std::unordered_map<EdgeKey, int64_t, EdgeKeyHash> counts;
  counts.reserve(faces.size() * 3);
  for (const auto &face : faces) {
    counts[edge_key(face[0], face[1])] += 1;
    counts[edge_key(face[1], face[2])] += 1;
    counts[edge_key(face[2], face[0])] += 1;
  }
  return counts;
}

inline nb::object make_float32_array(std::vector<float> values, size_t rows, size_t cols) {
  auto owner = new std::vector<float>(std::move(values));
  nb::capsule capsule(owner, [](void *ptr) noexcept {
    delete static_cast<std::vector<float> *>(ptr);
  });
  return nb::ndarray<nb::numpy, float>(owner->data(), {rows, cols}, capsule).cast();
}

inline nb::object make_int64_array(std::vector<int64_t> values, size_t rows, size_t cols) {
  auto owner = new std::vector<int64_t>(std::move(values));
  nb::capsule capsule(owner, [](void *ptr) noexcept {
    delete static_cast<std::vector<int64_t> *>(ptr);
  });
  return nb::ndarray<nb::numpy, int64_t>(owner->data(), {rows, cols}, capsule).cast();
}

inline nb::object make_uint8_array(std::vector<uint8_t> values, size_t rows, size_t cols) {
  auto owner = new std::vector<uint8_t>(std::move(values));
  nb::capsule capsule(owner, [](void *ptr) noexcept {
    delete static_cast<std::vector<uint8_t> *>(ptr);
  });
  return nb::ndarray<nb::numpy, uint8_t>(owner->data(), {rows, cols}, capsule).cast();
}

inline nb::object make_uint8_array(std::vector<uint8_t> values, size_t rows, size_t cols, size_t channels) {
  auto owner = new std::vector<uint8_t>(std::move(values));
  nb::capsule capsule(owner, [](void *ptr) noexcept {
    delete static_cast<std::vector<uint8_t> *>(ptr);
  });
  return nb::ndarray<nb::numpy, uint8_t>(owner->data(), {rows, cols, channels}, capsule).cast();
}

class UnionFind {
 public:
  explicit UnionFind(size_t count) : parent_(count), rank_(count, 0) {
    std::iota(parent_.begin(), parent_.end(), 0);
  }

  size_t find(size_t value) {
    size_t root = value;
    while (parent_[root] != root) {
      root = parent_[root];
    }
    while (parent_[value] != value) {
      const size_t next = parent_[value];
      parent_[value] = root;
      value = next;
    }
    return root;
  }

  void unite(size_t left, size_t right) {
    size_t left_root = find(left);
    size_t right_root = find(right);
    if (left_root == right_root) {
      return;
    }
    if (rank_[left_root] < rank_[right_root]) {
      parent_[left_root] = right_root;
    } else if (rank_[left_root] > rank_[right_root]) {
      parent_[right_root] = left_root;
    } else {
      parent_[right_root] = left_root;
      rank_[left_root] += 1;
    }
  }

 private:
  std::vector<size_t> parent_;
  std::vector<uint8_t> rank_;
};

inline int64_t connected_component_count(const MeshData &mesh) {
  if (mesh.faces.empty()) {
    return 0;
  }
  UnionFind uf(mesh.vertices.size());
  for (const auto &face : mesh.faces) {
    uf.unite(static_cast<size_t>(face[0]), static_cast<size_t>(face[1]));
    uf.unite(static_cast<size_t>(face[0]), static_cast<size_t>(face[2]));
  }
  std::set<size_t> roots;
  for (const auto &face : mesh.faces) {
    roots.insert(uf.find(static_cast<size_t>(face[0])));
  }
  return static_cast<int64_t>(roots.size());
}

inline MeshData compact_mesh(const MeshData &mesh, int64_t *unreferenced_removed) {
  std::vector<int64_t> remap(mesh.vertices.size(), -1);
  std::vector<std::array<float, 3>> compact_vertices;
  compact_vertices.reserve(mesh.vertices.size());
  std::vector<std::array<int64_t, 3>> compact_faces;
  compact_faces.reserve(mesh.faces.size());
  for (const auto &face : mesh.faces) {
    std::array<int64_t, 3> compact_face{};
    for (int i = 0; i < 3; ++i) {
      const size_t index = static_cast<size_t>(face[i]);
      if (remap[index] < 0) {
        remap[index] = static_cast<int64_t>(compact_vertices.size());
        compact_vertices.push_back(mesh.vertices[index]);
      }
      compact_face[static_cast<size_t>(i)] = remap[index];
    }
    compact_faces.push_back(compact_face);
  }
  if (unreferenced_removed != nullptr) {
    *unreferenced_removed += static_cast<int64_t>(mesh.vertices.size() - compact_vertices.size());
  }
  return MeshData{std::move(compact_vertices), std::move(compact_faces)};
}

inline nb::dict mesh_result(const MeshData &mesh) {
  std::vector<float> vertices;
  vertices.reserve(mesh.vertices.size() * 3);
  for (const auto &vertex : mesh.vertices) {
    vertices.push_back(vertex[0]);
    vertices.push_back(vertex[1]);
    vertices.push_back(vertex[2]);
  }
  std::vector<int64_t> faces;
  faces.reserve(mesh.faces.size() * 3);
  for (const auto &face : mesh.faces) {
    faces.push_back(face[0]);
    faces.push_back(face[1]);
    faces.push_back(face[2]);
  }
  nb::dict result;
  result["vertices"] = make_float32_array(std::move(vertices), mesh.vertices.size(), 3);
  result["faces"] = make_int64_array(std::move(faces), mesh.faces.size(), 3);
  return result;
}

}  // namespace mlx_spatialkit::mesh_common
