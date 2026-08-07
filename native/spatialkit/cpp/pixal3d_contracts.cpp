#include "pixal3d_contracts.hpp"

#include <Python.h>

#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>

#include <nanobind/stl/string.h>

namespace nb = nanobind;

namespace mlx_spatialkit {
namespace {

struct MatrixInfo {
  int64_t rows;
  int64_t cols;
  std::string dtype;
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

int64_t cast_int64(nb::handle value, const char *what) {
  try {
    return nb::cast<int64_t>(value);
  } catch (const nb::cast_error &) {
    std::ostringstream message;
    message << what << " must be an integer";
    throw nb::value_error(message.str().c_str());
  }
}

std::string dtype_name(nb::object array, const char *name) {
  if (!PyObject_HasAttrString(array.ptr(), "dtype")) {
    std::ostringstream message;
    message << name << " must expose a dtype";
    throw nb::type_error(message.str().c_str());
  }
  return nb::cast<std::string>(nb::str(nb::getattr(array, "dtype")));
}

int64_t ndim_value(nb::object array, const char *name) {
  if (!PyObject_HasAttrString(array.ptr(), "ndim")) {
    std::ostringstream message;
    message << name << " must expose ndim";
    throw nb::type_error(message.str().c_str());
  }
  return cast_int64(nb::getattr(array, "ndim"), "ndim");
}

nb::tuple shape_tuple(nb::object array, const char *name) {
  if (!PyObject_HasAttrString(array.ptr(), "shape")) {
    std::ostringstream message;
    message << name << " must expose shape";
    throw nb::type_error(message.str().c_str());
  }
  return nb::cast<nb::tuple>(nb::getattr(array, "shape"));
}

MatrixInfo validate_matrix(
    nb::object array,
    const char *name,
    int64_t expected_cols,
    const char *expected_dtype) {
  const int64_t ndim = ndim_value(array, name);
  if (ndim != 2) {
    std::ostringstream message;
    message << name << " must have rank 2, got rank " << ndim;
    throw nb::value_error(message.str().c_str());
  }

  nb::tuple shape = shape_tuple(array, name);
  if (shape.size() != 2) {
    std::ostringstream message;
    message << name << " shape must contain 2 dims, got " << shape.size();
    throw nb::value_error(message.str().c_str());
  }

  const int64_t rows = cast_int64(shape[0], "shape[0]");
  const int64_t cols = cast_int64(shape[1], "shape[1]");
  if (rows <= 0) {
    std::ostringstream message;
    message << name << " must contain at least one token";
    throw nb::value_error(message.str().c_str());
  }
  if (cols != expected_cols) {
    std::ostringstream message;
    message << name << " must have shape (n, " << expected_cols
            << "), got (" << rows << ", " << cols << ")";
    throw nb::value_error(message.str().c_str());
  }

  const std::string dtype = dtype_name(array, name);
  if (dtype != expected_dtype) {
    std::ostringstream message;
    message << name << " must have dtype " << expected_dtype
            << ", got " << dtype;
    throw nb::value_error(message.str().c_str());
  }

  return MatrixInfo{rows, cols, dtype};
}

void validate_same_rows(
    const MatrixInfo &left,
    const MatrixInfo &right,
    const char *left_name,
    const char *right_name) {
  if (left.rows != right.rows) {
    std::ostringstream message;
    message << left_name << "/" << right_name << " token mismatch: "
            << left_name << "=" << left.rows << " "
            << right_name << "=" << right.rows;
    throw nb::value_error(message.str().c_str());
  }
}

void validate_batch_zero(nb::object coordinates, int64_t rows, const char *name) {
  BufferView buffer(coordinates.ptr(), name);
  const Py_buffer &view = buffer.get();
  if (view.ndim != 2 || view.itemsize != static_cast<Py_ssize_t>(sizeof(int32_t))) {
    std::ostringstream message;
    message << name << " buffer must be a 2D int32 array";
    throw nb::value_error(message.str().c_str());
  }

  const auto *base = static_cast<const char *>(view.buf);
  const Py_ssize_t row_stride = view.strides != nullptr ? view.strides[0] : view.itemsize * 4;
  for (int64_t row = 0; row < rows; ++row) {
    int32_t batch = 0;
    std::memcpy(&batch, base + row * row_stride, sizeof(batch));
    if (batch != 0) {
      std::ostringstream message;
      message << name << " currently supports batch index 0 only; found "
              << batch << " at row " << row;
      throw nb::value_error(message.str().c_str());
    }
  }
}

nb::tuple shape_pair(int64_t rows, int64_t cols) {
  return nb::make_tuple(rows, cols);
}

}  // namespace

nb::dict validate_pixal3d_shape_fields(nb::object coordinates, nb::object fields) {
  MatrixInfo coordinate_info = validate_matrix(coordinates, "shape coordinates", 4, "int32");
  MatrixInfo field_info = validate_matrix(fields, "shape fields", 7, "float32");
  validate_same_rows(coordinate_info, field_info, "coordinates", "fields");
  validate_batch_zero(coordinates, coordinate_info.rows, "shape coordinates");

  nb::dict result;
  result["token_count"] = coordinate_info.rows;
  result["coordinates_shape"] = shape_pair(coordinate_info.rows, coordinate_info.cols);
  result["fields_shape"] = shape_pair(field_info.rows, field_info.cols);
  result["coordinates_dtype"] = coordinate_info.dtype;
  result["fields_dtype"] = field_info.dtype;
  return result;
}

nb::dict validate_pixal3d_texture_attributes(nb::object coordinates, nb::object attributes) {
  MatrixInfo coordinate_info = validate_matrix(coordinates, "texture coordinates", 4, "int32");
  MatrixInfo attribute_info = validate_matrix(attributes, "texture attributes", 6, "float32");
  validate_same_rows(coordinate_info, attribute_info, "coordinates", "attributes");
  validate_batch_zero(coordinates, coordinate_info.rows, "texture coordinates");

  nb::dict result;
  result["token_count"] = coordinate_info.rows;
  result["coordinates_shape"] = shape_pair(coordinate_info.rows, coordinate_info.cols);
  result["attributes_shape"] = shape_pair(attribute_info.rows, attribute_info.cols);
  result["coordinates_dtype"] = coordinate_info.dtype;
  result["attributes_dtype"] = attribute_info.dtype;
  return result;
}

}  // namespace mlx_spatialkit
