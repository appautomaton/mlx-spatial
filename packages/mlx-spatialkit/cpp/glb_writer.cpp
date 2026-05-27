#include "glb_writer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mesh_common.hpp"

namespace nb = nanobind;

namespace mlx_spatialkit {
namespace {

constexpr uint32_t kGlbMagic = 0x46546C67;
constexpr uint32_t kArrayBufferTarget = 34962;
constexpr uint32_t kElementArrayBufferTarget = 34963;
constexpr uint32_t kComponentFloat32 = 5126;
constexpr uint32_t kComponentUint16 = 5123;
constexpr int64_t kMaxTextureDimension = 8192;
constexpr uint64_t kMaxTexturePixels = static_cast<uint64_t>(kMaxTextureDimension) * kMaxTextureDimension;
constexpr uint64_t kMaxTextureBytes = 512ull * 1024ull * 1024ull;
constexpr uint32_t kMaxUint16Index = std::numeric_limits<uint16_t>::max();

struct TextureImage {
  int64_t height;
  int64_t width;
  int64_t channels;
  std::vector<uint8_t> pixels;
};

struct BufferView {
  uint32_t offset;
  uint32_t length;
  int target;
};

struct EdgeKey {
  int64_t a;
  int64_t b;

  bool operator==(const EdgeKey &other) const {
    return a == other.a && b == other.b;
  }
};

struct EdgeKeyHash {
  size_t operator()(const EdgeKey &edge) const {
    const auto left = static_cast<uint64_t>(edge.a);
    const auto right = static_cast<uint64_t>(edge.b);
    return static_cast<size_t>((left * 11400714819323198485ull) ^ (right + 0x9e3779b97f4a7c15ull + (left << 6) + (left >> 2)));
  }
};

void append_u16_le(std::vector<uint8_t> &out, uint16_t value) {
  out.push_back(static_cast<uint8_t>(value & 0xff));
  out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
}

void append_u32_le(std::vector<uint8_t> &out, uint32_t value) {
  out.push_back(static_cast<uint8_t>(value & 0xff));
  out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
  out.push_back(static_cast<uint8_t>((value >> 16) & 0xff));
  out.push_back(static_cast<uint8_t>((value >> 24) & 0xff));
}

void append_u32_be(std::vector<uint8_t> &out, uint32_t value) {
  out.push_back(static_cast<uint8_t>((value >> 24) & 0xff));
  out.push_back(static_cast<uint8_t>((value >> 16) & 0xff));
  out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
  out.push_back(static_cast<uint8_t>(value & 0xff));
}

uint32_t checked_u32(uint64_t value, const char *name) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    std::ostringstream message;
    message << name << " exceeds GLB/PNG 32-bit length limits";
    throw nb::value_error(message.str().c_str());
  }
  return static_cast<uint32_t>(value);
}

uint64_t checked_mul(uint64_t left, uint64_t right, const char *name) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
    std::ostringstream message;
    message << name << " size overflows 64-bit arithmetic";
    throw nb::value_error(message.str().c_str());
  }
  return left * right;
}

void append_float32_le(std::vector<uint8_t> &out, float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(float));
  append_u32_le(out, bits);
}

void pad4(std::vector<uint8_t> &out, uint8_t value) {
  const size_t padding = (4 - (out.size() % 4)) % 4;
  out.insert(out.end(), padding, value);
}

uint32_t crc32_png(const uint8_t *data, size_t length) {
  uint32_t crc = 0xffffffffu;
  for (size_t i = 0; i < length; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
  }
  return crc ^ 0xffffffffu;
}

uint32_t adler32(const std::vector<uint8_t> &data) {
  uint32_t a = 1;
  uint32_t b = 0;
  for (uint8_t byte : data) {
    a = (a + byte) % 65521u;
    b = (b + a) % 65521u;
  }
  return (b << 16) | a;
}

void append_png_chunk(std::vector<uint8_t> &out, const char type[4], const std::vector<uint8_t> &data) {
  append_u32_be(out, checked_u32(data.size(), "PNG chunk"));
  const size_t crc_start = out.size();
  out.insert(out.end(), type, type + 4);
  out.insert(out.end(), data.begin(), data.end());
  append_u32_be(out, crc32_png(out.data() + crc_start, out.size() - crc_start));
}

std::vector<uint8_t> zlib_store(const std::vector<uint8_t> &raw) {
  const uint64_t max_encoded_size = static_cast<uint64_t>(raw.size()) + raw.size() / 65535 + 16;
  checked_u32(max_encoded_size, "PNG IDAT");
  std::vector<uint8_t> out;
  out.reserve(raw.size() + raw.size() / 65535 + 16);
  out.push_back(0x78);
  out.push_back(0x01);
  size_t offset = 0;
  while (offset < raw.size()) {
    const size_t block_length = std::min<size_t>(65535, raw.size() - offset);
    const bool final_block = offset + block_length == raw.size();
    out.push_back(final_block ? 0x01 : 0x00);
    append_u16_le(out, static_cast<uint16_t>(block_length));
    append_u16_le(out, static_cast<uint16_t>(~static_cast<uint16_t>(block_length)));
    out.insert(out.end(), raw.begin() + static_cast<std::ptrdiff_t>(offset),
               raw.begin() + static_cast<std::ptrdiff_t>(offset + block_length));
    offset += block_length;
  }
  append_u32_be(out, adler32(raw));
  return out;
}

std::vector<uint8_t> png_payload(const TextureImage &image) {
  const bool rgba = image.channels == 4;
  const uint8_t color_type = rgba ? 6 : 2;
  const uint64_t row_bytes64 =
      checked_mul(static_cast<uint64_t>(image.width), static_cast<uint64_t>(image.channels), "PNG row");
  checked_u32(row_bytes64, "PNG row");
  const size_t row_bytes = static_cast<size_t>(row_bytes64);
  const uint64_t filtered_size =
      checked_mul(row_bytes64 + 1, static_cast<uint64_t>(image.height), "PNG filtered texture");
  checked_u32(filtered_size + filtered_size / 65535 + 16, "PNG encoded texture");
  std::vector<uint8_t> filtered;
  filtered.reserve(static_cast<size_t>(filtered_size));
  for (int64_t row = 0; row < image.height; ++row) {
    filtered.push_back(0);
    const size_t start = static_cast<size_t>(row) * row_bytes;
    filtered.insert(filtered.end(), image.pixels.begin() + static_cast<std::ptrdiff_t>(start),
                    image.pixels.begin() + static_cast<std::ptrdiff_t>(start + row_bytes));
  }

  std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
  std::vector<uint8_t> ihdr;
  append_u32_be(ihdr, checked_u32(static_cast<uint64_t>(image.width), "PNG width"));
  append_u32_be(ihdr, checked_u32(static_cast<uint64_t>(image.height), "PNG height"));
  ihdr.push_back(8);
  ihdr.push_back(color_type);
  ihdr.push_back(0);
  ihdr.push_back(0);
  ihdr.push_back(0);
  append_png_chunk(png, "IHDR", ihdr);
  append_png_chunk(png, "IDAT", zlib_store(filtered));
  append_png_chunk(png, "IEND", {});
  return png;
}

std::string json_escape(const std::string &value) {
  std::ostringstream out;
  for (unsigned char ch : value) {
    switch (ch) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (ch < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch);
        } else {
          out << static_cast<char>(ch);
        }
    }
  }
  return out.str();
}

std::string quoted(const std::string &value) {
  return "\"" + json_escape(value) + "\"";
}

std::string float_array_json(const std::array<float, 3> &values, size_t count) {
  std::ostringstream out;
  out << std::setprecision(9) << "[";
  for (size_t i = 0; i < count; ++i) {
    if (i != 0) {
      out << ",";
    }
    out << values[i];
  }
  out << "]";
  return out.str();
}

TextureImage load_texture_image(nb::object image_object, const char *name, int64_t channels) {
  const auto ndim = nb::cast<int64_t>(nb::getattr(image_object, "ndim"));
  if (ndim != 3) {
    std::ostringstream message;
    message << name << " must have rank 3";
    throw nb::value_error(message.str().c_str());
  }
  if (mesh_common::dimension(image_object, name, 2) != channels) {
    std::ostringstream message;
    message << name << " must have shape (height, width, " << channels << ")";
    throw nb::value_error(message.str().c_str());
  }
  if (mesh_common::dtype_name(image_object, name) != "uint8") {
    std::ostringstream message;
    message << name << " must have dtype uint8";
    throw nb::value_error(message.str().c_str());
  }
  const int64_t height = mesh_common::dimension(image_object, name, 0);
  const int64_t width = mesh_common::dimension(image_object, name, 1);
  if (height <= 0 || width <= 0) {
    std::ostringstream message;
    message << name << " must not be empty";
    throw nb::value_error(message.str().c_str());
  }
  if (height > kMaxTextureDimension || width > kMaxTextureDimension) {
    std::ostringstream message;
    message << name << " dimensions must be <= " << kMaxTextureDimension << " pixels per side";
    throw nb::value_error(message.str().c_str());
  }
  const uint64_t pixel_count =
      checked_mul(static_cast<uint64_t>(height), static_cast<uint64_t>(width), "texture pixel");
  if (pixel_count > kMaxTexturePixels) {
    std::ostringstream message;
    message << name << " exceeds maximum texture pixel count " << kMaxTexturePixels;
    throw nb::value_error(message.str().c_str());
  }
  const uint64_t byte_count = checked_mul(pixel_count, static_cast<uint64_t>(channels), "texture byte");
  if (byte_count > kMaxTextureBytes) {
    std::ostringstream message;
    message << name << " exceeds maximum texture byte count " << kMaxTextureBytes;
    throw nb::value_error(message.str().c_str());
  }

  mesh_common::BufferView image_buffer(image_object.ptr(), name);
  const Py_buffer &view = image_buffer.get();
  const auto *base = static_cast<const char *>(view.buf);
  const Py_ssize_t row_stride = view.strides != nullptr ? view.strides[0] : view.itemsize * width * channels;
  const Py_ssize_t col_stride = view.strides != nullptr ? view.strides[1] : view.itemsize * channels;
  const Py_ssize_t channel_stride = view.strides != nullptr ? view.strides[2] : view.itemsize;
  TextureImage image{height, width, channels, {}};
  image.pixels.reserve(static_cast<size_t>(byte_count));
  for (int64_t row = 0; row < height; ++row) {
    for (int64_t col = 0; col < width; ++col) {
      for (int64_t channel = 0; channel < channels; ++channel) {
        image.pixels.push_back(static_cast<uint8_t>(
            *(base + row * row_stride + col * col_stride + channel * channel_stride)));
      }
    }
  }
  return image;
}

std::vector<std::array<float, 2>> load_uvs(nb::object uvs_object, int64_t vertex_count) {
  mesh_common::validate_matrix(uvs_object, "GLB UVs", 2, "float32");
  if (mesh_common::dimension(uvs_object, "GLB UVs", 0) != vertex_count) {
    std::ostringstream message;
    message << "GLB UVs must have shape (" << vertex_count << ", 2)";
    throw nb::value_error(message.str().c_str());
  }
  mesh_common::BufferView uv_buffer(uvs_object.ptr(), "GLB UVs");
  const Py_buffer &view = uv_buffer.get();
  std::vector<std::array<float, 2>> uvs;
  uvs.reserve(static_cast<size_t>(vertex_count));
  for (int64_t row = 0; row < vertex_count; ++row) {
    std::array<float, 2> uv{
        mesh_common::read_matrix_value<float>(view, row, 0),
        mesh_common::read_matrix_value<float>(view, row, 1),
    };
    if (!std::isfinite(uv[0]) || !std::isfinite(uv[1])) {
      throw nb::value_error("GLB UVs must contain only finite values");
    }
    if (uv[0] < 0.0f || uv[0] > 1.0f || uv[1] < 0.0f || uv[1] > 1.0f) {
      throw nb::value_error("GLB UVs must stay in [0, 1]");
    }
    uvs.push_back(uv);
  }
  return uvs;
}

std::array<float, 3> vertex_min(const std::vector<std::array<float, 3>> &vertices) {
  std::array<float, 3> result = vertices[0];
  for (const auto &vertex : vertices) {
    for (size_t axis = 0; axis < 3; ++axis) {
      result[axis] = std::min(result[axis], vertex[axis]);
    }
  }
  return result;
}

std::array<float, 3> vertex_max(const std::vector<std::array<float, 3>> &vertices) {
  std::array<float, 3> result = vertices[0];
  for (const auto &vertex : vertices) {
    for (size_t axis = 0; axis < 3; ++axis) {
      result[axis] = std::max(result[axis], vertex[axis]);
    }
  }
  return result;
}

std::array<float, 3> uv_min_padded(const std::vector<std::array<float, 2>> &uvs) {
  std::array<float, 3> result{uvs[0][0], uvs[0][1], 0.0f};
  for (const auto &uv : uvs) {
    result[0] = std::min(result[0], uv[0]);
    result[1] = std::min(result[1], uv[1]);
  }
  return result;
}

std::array<float, 3> uv_max_padded(const std::vector<std::array<float, 2>> &uvs) {
  std::array<float, 3> result{uvs[0][0], uvs[0][1], 0.0f};
  for (const auto &uv : uvs) {
    result[0] = std::max(result[0], uv[0]);
    result[1] = std::max(result[1], uv[1]);
  }
  return result;
}

std::array<float, 3> normalized_face_normal(const mesh_common::MeshData &mesh, const std::array<int64_t, 3> &face) {
  const auto &a = mesh.vertices[static_cast<size_t>(face[0])];
  const auto &b = mesh.vertices[static_cast<size_t>(face[1])];
  const auto &c = mesh.vertices[static_cast<size_t>(face[2])];
  const std::array<float, 3> ab{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const std::array<float, 3> ac{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  std::array<float, 3> normal{
      ab[1] * ac[2] - ab[2] * ac[1],
      ab[2] * ac[0] - ab[0] * ac[2],
      ab[0] * ac[1] - ab[1] * ac[0],
  };
  const float length = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
  if (std::isfinite(length) && length > 1e-12f) {
    normal[0] /= length;
    normal[1] /= length;
    normal[2] /= length;
    return normal;
  }
  return {0.0f, 0.0f, 1.0f};
}

float dot3(const std::array<float, 3> &left, const std::array<float, 3> &right) {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

std::array<float, 3> cross3(const std::array<float, 3> &left, const std::array<float, 3> &right) {
  return {
      left[1] * right[2] - left[2] * right[1],
      left[2] * right[0] - left[0] * right[2],
      left[0] * right[1] - left[1] * right[0],
  };
}

std::array<float, 3> normalize3(const std::array<float, 3> &value, const std::array<float, 3> &fallback) {
  const float length = std::sqrt(dot3(value, value));
  if (std::isfinite(length) && length > 1e-12f) {
    return {value[0] / length, value[1] / length, value[2] / length};
  }
  return fallback;
}

EdgeKey edge_key(int64_t left, int64_t right) {
  return left < right ? EdgeKey{left, right} : EdgeKey{right, left};
}

std::vector<std::array<float, 3>> compute_vertex_normals(const mesh_common::MeshData &mesh) {
  std::vector<std::array<float, 3>> normals(mesh.vertices.size(), std::array<float, 3>{0.0f, 0.0f, 0.0f});
  for (const auto &face : mesh.faces) {
    const auto &a = mesh.vertices[static_cast<size_t>(face[0])];
    const auto &b = mesh.vertices[static_cast<size_t>(face[1])];
    const auto &c = mesh.vertices[static_cast<size_t>(face[2])];
    const std::array<float, 3> ab{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const std::array<float, 3> ac{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    const std::array<float, 3> normal{
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    };
    for (int corner = 0; corner < 3; ++corner) {
      auto &accum = normals[static_cast<size_t>(face[corner])];
      accum[0] += normal[0];
      accum[1] += normal[1];
      accum[2] += normal[2];
    }
  }
  for (auto &normal : normals) {
    const float length = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    if (std::isfinite(length) && length > 1e-12f) {
      normal[0] /= length;
      normal[1] /= length;
      normal[2] /= length;
    } else {
      normal = {0.0f, 0.0f, 1.0f};
    }
  }
  return normals;
}

std::vector<uint8_t> vec3_payload(const std::vector<std::array<float, 3>> &values) {
  std::vector<uint8_t> payload;
  payload.reserve(values.size() * 12);
  for (const auto &value : values) {
    append_float32_le(payload, value[0]);
    append_float32_le(payload, value[1]);
    append_float32_le(payload, value[2]);
  }
  return payload;
}

std::vector<uint8_t> vec2_payload(const std::vector<std::array<float, 2>> &values) {
  std::vector<uint8_t> payload;
  payload.reserve(values.size() * 8);
  for (const auto &value : values) {
    append_float32_le(payload, value[0]);
    append_float32_le(payload, value[1]);
  }
  return payload;
}

std::vector<uint8_t> uint16_payload(const std::vector<uint16_t> &values) {
  std::vector<uint8_t> payload;
  payload.reserve(values.size() * 2);
  for (uint16_t value : values) {
    append_u16_le(payload, value);
  }
  return payload;
}

uint32_t add_buffer_view(
    std::vector<uint8_t> &bin_blob,
    std::vector<BufferView> &views,
    const std::vector<uint8_t> &payload,
    int target = 0) {
  pad4(bin_blob, 0);
  if (bin_blob.size() > std::numeric_limits<uint32_t>::max() || payload.size() > std::numeric_limits<uint32_t>::max()) {
    throw nb::value_error("GLB payload is too large for 32-bit buffer offsets");
  }
  const auto offset = static_cast<uint32_t>(bin_blob.size());
  const uint64_t end_offset = static_cast<uint64_t>(bin_blob.size()) + payload.size();
  checked_u32(end_offset, "GLB buffer view end offset");
  bin_blob.insert(bin_blob.end(), payload.begin(), payload.end());
  views.push_back(BufferView{offset, checked_u32(payload.size(), "GLB buffer view"), target});
  return static_cast<uint32_t>(views.size() - 1);
}

std::string buffer_views_json(const std::vector<BufferView> &views) {
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < views.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << "{\"buffer\":0,\"byteOffset\":" << views[i].offset << ",\"byteLength\":" << views[i].length;
    if (views[i].target != 0) {
      out << ",\"target\":" << views[i].target;
    }
    out << "}";
  }
  out << "]";
  return out.str();
}

std::vector<uint8_t> bytes_from_string(const std::string &value) {
  return {value.begin(), value.end()};
}

}  // namespace

nb::dict make_face_atlas_uvs(nb::object vertices, nb::object faces, double tile_padding) {
  if (tile_padding < 0.0 || tile_padding >= 0.45) {
    throw nb::value_error("tile_padding must be in [0, 0.45)");
  }
  mesh_common::MeshData mesh = mesh_common::load_mesh(vertices, faces);
  if (mesh.faces.empty()) {
    throw nb::value_error("UV atlas generation requires at least one face");
  }

  const int64_t face_count = static_cast<int64_t>(mesh.faces.size());
  constexpr int64_t faces_per_tile = 2;
  const int64_t atlas_tiles = (face_count + faces_per_tile - 1) / faces_per_tile;
  const int64_t cols = static_cast<int64_t>(std::ceil(std::sqrt(static_cast<double>(atlas_tiles))));
  const int64_t rows = static_cast<int64_t>(std::ceil(static_cast<double>(atlas_tiles) / static_cast<double>(cols)));
  const std::array<std::array<float, 2>, 3> lower_left_uv{{
      {static_cast<float>(tile_padding), static_cast<float>(tile_padding)},
      {static_cast<float>(1.0 - tile_padding), static_cast<float>(tile_padding)},
      {static_cast<float>(tile_padding), static_cast<float>(1.0 - tile_padding)},
  }};
  const std::array<std::array<float, 2>, 3> upper_right_uv{{
      {static_cast<float>(1.0 - tile_padding), static_cast<float>(1.0 - tile_padding)},
      {static_cast<float>(tile_padding), static_cast<float>(1.0 - tile_padding)},
      {static_cast<float>(1.0 - tile_padding), static_cast<float>(tile_padding)},
  }};

  std::vector<float> atlas_vertices;
  std::vector<float> atlas_uvs;
  std::vector<int64_t> atlas_faces;
  atlas_vertices.reserve(static_cast<size_t>(face_count * 9));
  atlas_uvs.reserve(static_cast<size_t>(face_count * 6));
  atlas_faces.reserve(static_cast<size_t>(face_count * 3));
  for (int64_t face_index = 0; face_index < face_count; ++face_index) {
    const auto &face = mesh.faces[static_cast<size_t>(face_index)];
    const int64_t tile_index = face_index / faces_per_tile;
    const int64_t col = tile_index % cols;
    const int64_t row = tile_index / cols;
    const auto &local_uv = face_index % faces_per_tile == 0 ? lower_left_uv : upper_right_uv;
    for (int corner = 0; corner < 3; ++corner) {
      const auto &vertex = mesh.vertices[static_cast<size_t>(face[corner])];
      atlas_vertices.push_back(vertex[0]);
      atlas_vertices.push_back(vertex[1]);
      atlas_vertices.push_back(vertex[2]);
      atlas_uvs.push_back((static_cast<float>(col) + local_uv[static_cast<size_t>(corner)][0]) / static_cast<float>(cols));
      atlas_uvs.push_back((static_cast<float>(row) + local_uv[static_cast<size_t>(corner)][1]) / static_cast<float>(rows));
      atlas_faces.push_back(face_index * 3 + corner);
    }
  }

  nb::dict result;
  result["vertices"] = mesh_common::make_float32_array(std::move(atlas_vertices), static_cast<size_t>(face_count * 3), 3);
  result["faces"] = mesh_common::make_int64_array(std::move(atlas_faces), static_cast<size_t>(face_count), 3);
  result["uvs"] = mesh_common::make_float32_array(std::move(atlas_uvs), static_cast<size_t>(face_count * 3), 2);
  nb::dict stats;
  stats["backend"] = "face-atlas";
  stats["source_vertices"] = static_cast<int64_t>(mesh.vertices.size());
  stats["source_faces"] = face_count;
  stats["output_vertices"] = face_count * 3;
  stats["output_faces"] = face_count;
  stats["packing"] = "paired-triangles";
  stats["faces_per_tile"] = faces_per_tile;
  stats["atlas_tiles"] = atlas_tiles;
  stats["atlas_cols"] = cols;
  stats["atlas_rows"] = rows;
  stats["tile_padding"] = tile_padding;
  stats["estimated_tile_utilization"] = face_count > 1 ? std::pow(1.0 - 2.0 * tile_padding, 2.0) : 0.5 * std::pow(1.0 - 2.0 * tile_padding, 2.0);
  result["stats"] = stats;
  return result;
}

nb::dict make_native_chart_uvs(nb::object vertices, nb::object faces, double chart_angle_degrees, double tile_padding) {
  if (!std::isfinite(chart_angle_degrees) || chart_angle_degrees < 0.0 || chart_angle_degrees > 180.0) {
    throw nb::value_error("chart_angle_degrees must be finite and in [0, 180]");
  }
  if (!std::isfinite(tile_padding) || tile_padding < 0.0 || tile_padding >= 0.45) {
    throw nb::value_error("tile_padding must be finite and in [0, 0.45)");
  }
  mesh_common::MeshData mesh = mesh_common::load_mesh(vertices, faces);
  if (mesh.faces.empty()) {
    throw nb::value_error("native chart UV generation requires at least one face");
  }

  const int64_t face_count = static_cast<int64_t>(mesh.faces.size());
  std::vector<std::array<float, 3>> face_normals;
  face_normals.reserve(mesh.faces.size());
  for (const auto &face : mesh.faces) {
    face_normals.push_back(normalized_face_normal(mesh, face));
  }

  std::unordered_map<EdgeKey, std::vector<int64_t>, EdgeKeyHash> edge_faces;
  edge_faces.reserve(mesh.faces.size() * 3);
  for (int64_t face_index = 0; face_index < face_count; ++face_index) {
    const auto &face = mesh.faces[static_cast<size_t>(face_index)];
    edge_faces[edge_key(face[0], face[1])].push_back(face_index);
    edge_faces[edge_key(face[1], face[2])].push_back(face_index);
    edge_faces[edge_key(face[2], face[0])].push_back(face_index);
  }

  std::vector<std::vector<int64_t>> neighbors(mesh.faces.size());
  for (const auto &item : edge_faces) {
    const auto &edge_face_list = item.second;
    for (size_t left = 0; left < edge_face_list.size(); ++left) {
      for (size_t right = left + 1; right < edge_face_list.size(); ++right) {
        neighbors[static_cast<size_t>(edge_face_list[left])].push_back(edge_face_list[right]);
        neighbors[static_cast<size_t>(edge_face_list[right])].push_back(edge_face_list[left]);
      }
    }
  }

  const double radians = chart_angle_degrees * 3.14159265358979323846 / 180.0;
  const float cos_threshold = static_cast<float>(std::cos(radians));
  std::vector<int32_t> chart_labels(mesh.faces.size(), -1);
  std::vector<std::vector<int64_t>> charts;
  std::vector<int64_t> stack;
  for (int64_t seed = 0; seed < face_count; ++seed) {
    if (chart_labels[static_cast<size_t>(seed)] >= 0) {
      continue;
    }
    const int32_t chart_id = static_cast<int32_t>(charts.size());
    charts.push_back({});
    chart_labels[static_cast<size_t>(seed)] = chart_id;
    stack = {seed};
    while (!stack.empty()) {
      const int64_t current = stack.back();
      stack.pop_back();
      charts.back().push_back(current);
      for (int64_t neighbor : neighbors[static_cast<size_t>(current)]) {
        if (chart_labels[static_cast<size_t>(neighbor)] >= 0) {
          continue;
        }
        if (dot3(face_normals[static_cast<size_t>(current)], face_normals[static_cast<size_t>(neighbor)]) < cos_threshold) {
          continue;
        }
        chart_labels[static_cast<size_t>(neighbor)] = chart_id;
        stack.push_back(neighbor);
      }
    }
  }

  constexpr int64_t chart_split_max_faces = 512;
  const int64_t source_chart_count = static_cast<int64_t>(charts.size());
  int64_t oversized_source_chart_count = 0;
  std::vector<std::vector<int64_t>> split_charts;
  split_charts.reserve(charts.size());
  auto face_centroid_axis = [&mesh](int64_t face_index, int axis) {
    const auto &face = mesh.faces[static_cast<size_t>(face_index)];
    return (static_cast<double>(mesh.vertices[static_cast<size_t>(face[0])][static_cast<size_t>(axis)]) +
            static_cast<double>(mesh.vertices[static_cast<size_t>(face[1])][static_cast<size_t>(axis)]) +
            static_cast<double>(mesh.vertices[static_cast<size_t>(face[2])][static_cast<size_t>(axis)])) /
           3.0;
  };
  for (const auto &chart : charts) {
    if (static_cast<int64_t>(chart.size()) <= chart_split_max_faces) {
      split_charts.push_back(chart);
      continue;
    }
    oversized_source_chart_count += 1;
    std::array<double, 3> min_centroid{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    std::array<double, 3> max_centroid{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    for (int64_t face_index : chart) {
      for (int axis = 0; axis < 3; ++axis) {
        const double value = face_centroid_axis(face_index, axis);
        min_centroid[static_cast<size_t>(axis)] = std::min(min_centroid[static_cast<size_t>(axis)], value);
        max_centroid[static_cast<size_t>(axis)] = std::max(max_centroid[static_cast<size_t>(axis)], value);
      }
    }
    int split_axis = 0;
    double best_extent = max_centroid[0] - min_centroid[0];
    for (int axis = 1; axis < 3; ++axis) {
      const double extent = max_centroid[static_cast<size_t>(axis)] - min_centroid[static_cast<size_t>(axis)];
      if (extent > best_extent + 1e-12) {
        split_axis = axis;
        best_extent = extent;
      }
    }
    std::vector<int64_t> sorted_chart = chart;
    std::stable_sort(sorted_chart.begin(), sorted_chart.end(), [&](int64_t left, int64_t right) {
      const double left_value = face_centroid_axis(left, split_axis);
      const double right_value = face_centroid_axis(right, split_axis);
      if (std::fabs(left_value - right_value) > 1e-12) {
        return left_value < right_value;
      }
      return left < right;
    });
    for (size_t begin = 0; begin < sorted_chart.size(); begin += static_cast<size_t>(chart_split_max_faces)) {
      const size_t end = std::min(sorted_chart.size(), begin + static_cast<size_t>(chart_split_max_faces));
      split_charts.emplace_back(sorted_chart.begin() + static_cast<std::ptrdiff_t>(begin),
                                sorted_chart.begin() + static_cast<std::ptrdiff_t>(end));
    }
  }
  charts = std::move(split_charts);

  constexpr int64_t projection_rotation_candidates = 19;
  constexpr double projection_rotation_step_degrees = 5.0;
  constexpr double pi = 3.14159265358979323846;
  constexpr double low_fill_rect_fill_threshold = 0.65;
  constexpr double low_fill_split_min_improvement = 0.02;
  constexpr int64_t low_fill_split_min_faces = 6;
  constexpr int64_t low_fill_split_min_child_faces = 3;
  constexpr int64_t low_fill_split_max_depth = 2;

  struct ChartFillEvaluation {
    double triangle_area = 0.0;
    double rect_area = 1.0;
    double rect_fill = 0.0;
    int split_axis = 0;
    std::vector<std::array<double, 2>> face_centroids;
  };

  auto evaluate_chart_fill = [&](const std::vector<int64_t> &chart) {
    ChartFillEvaluation evaluation;
    if (chart.empty()) {
      return evaluation;
    }

    std::unordered_map<int64_t, int64_t> local_index;
    local_index.reserve(chart.size() * 3);
    std::vector<int64_t> source_vertices;
    std::vector<std::array<int64_t, 3>> local_faces;
    source_vertices.reserve(chart.size() * 3);
    local_faces.reserve(chart.size());
    for (int64_t face_index : chart) {
      const auto &face = mesh.faces[static_cast<size_t>(face_index)];
      std::array<int64_t, 3> local_face{};
      for (int corner = 0; corner < 3; ++corner) {
        const int64_t source_index = face[corner];
        auto inserted = local_index.emplace(source_index, static_cast<int64_t>(source_vertices.size()));
        if (inserted.second) {
          source_vertices.push_back(source_index);
        }
        local_face[static_cast<size_t>(corner)] = inserted.first->second;
      }
      local_faces.push_back(local_face);
    }
    if (source_vertices.empty()) {
      return evaluation;
    }

    std::array<float, 3> average_normal{0.0f, 0.0f, 0.0f};
    for (int64_t face_index : chart) {
      const auto &normal = face_normals[static_cast<size_t>(face_index)];
      average_normal[0] += normal[0];
      average_normal[1] += normal[1];
      average_normal[2] += normal[2];
    }
    const std::array<float, 3> chart_normal = normalize3(average_normal, {0.0f, 0.0f, 1.0f});
    const float nx = std::fabs(chart_normal[0]);
    const float ny = std::fabs(chart_normal[1]);
    const float nz = std::fabs(chart_normal[2]);
    std::array<float, 3> seed_axis{1.0f, 0.0f, 0.0f};
    if (ny <= nx && ny <= nz) {
      seed_axis = {0.0f, 1.0f, 0.0f};
    } else if (nz <= nx && nz <= ny) {
      seed_axis = {0.0f, 0.0f, 1.0f};
    }
    const std::array<float, 3> tangent_u = normalize3(cross3(seed_axis, chart_normal), {1.0f, 0.0f, 0.0f});
    const std::array<float, 3> tangent_v = normalize3(cross3(chart_normal, tangent_u), {0.0f, 1.0f, 0.0f});

    std::vector<std::array<double, 2>> projected_coords;
    projected_coords.reserve(source_vertices.size());
    double mean_u = 0.0;
    double mean_v = 0.0;
    for (int64_t source_index : source_vertices) {
      const auto &vertex = mesh.vertices[static_cast<size_t>(source_index)];
      const double u = static_cast<double>(vertex[0]) * tangent_u[0] +
                       static_cast<double>(vertex[1]) * tangent_u[1] +
                       static_cast<double>(vertex[2]) * tangent_u[2];
      const double v = static_cast<double>(vertex[0]) * tangent_v[0] +
                       static_cast<double>(vertex[1]) * tangent_v[1] +
                       static_cast<double>(vertex[2]) * tangent_v[2];
      projected_coords.push_back({u, v});
      mean_u += u;
      mean_v += v;
    }
    mean_u /= static_cast<double>(projected_coords.size());
    mean_v /= static_cast<double>(projected_coords.size());

    double cov_uu = 0.0;
    double cov_vv = 0.0;
    double cov_uv = 0.0;
    for (const auto &coord : projected_coords) {
      const double du = coord[0] - mean_u;
      const double dv = coord[1] - mean_v;
      cov_uu += du * du;
      cov_vv += dv * dv;
      cov_uv += du * dv;
    }
    const double pca_angle = std::isfinite(cov_uu) && std::isfinite(cov_vv) && std::isfinite(cov_uv)
                                 ? 0.5 * std::atan2(2.0 * cov_uv, cov_uu - cov_vv)
                                 : 0.0;
    std::array<double, projection_rotation_candidates> candidate_angles{};
    const double step_radians = projection_rotation_step_degrees * pi / 180.0;
    const int64_t midpoint = projection_rotation_candidates / 2;
    for (int64_t candidate = 0; candidate < projection_rotation_candidates; ++candidate) {
      candidate_angles[static_cast<size_t>(candidate)] =
          pca_angle + static_cast<double>(candidate - midpoint) * step_radians;
    }

    double best_angle = 0.0;
    double best_min_u = 0.0;
    double best_min_v = 0.0;
    double best_width = std::numeric_limits<double>::infinity();
    double best_height = std::numeric_limits<double>::infinity();
    double best_area = std::numeric_limits<double>::infinity();
    for (double angle : candidate_angles) {
      const double cos_angle = std::cos(angle);
      const double sin_angle = std::sin(angle);
      double min_u = std::numeric_limits<double>::infinity();
      double min_v = std::numeric_limits<double>::infinity();
      double max_u = -std::numeric_limits<double>::infinity();
      double max_v = -std::numeric_limits<double>::infinity();
      for (const auto &coord : projected_coords) {
        const double centered_u = coord[0] - mean_u;
        const double centered_v = coord[1] - mean_v;
        const double rotated_u = cos_angle * centered_u + sin_angle * centered_v;
        const double rotated_v = -sin_angle * centered_u + cos_angle * centered_v;
        min_u = std::min(min_u, rotated_u);
        min_v = std::min(min_v, rotated_v);
        max_u = std::max(max_u, rotated_u);
        max_v = std::max(max_v, rotated_v);
      }
      const double width = std::max(max_u - min_u, 1e-12);
      const double height = std::max(max_v - min_v, 1e-12);
      const double area = width * height;
      if (area < best_area - 1e-12) {
        best_angle = angle;
        best_min_u = min_u;
        best_min_v = min_v;
        best_width = width;
        best_height = height;
        best_area = area;
      }
    }

    const double max_extent = std::max(best_width, best_height);
    if (!std::isfinite(max_extent) || max_extent <= 0.0) {
      return evaluation;
    }
    const double cos_angle = std::cos(best_angle);
    const double sin_angle = std::sin(best_angle);
    std::vector<std::array<double, 2>> local_uvs;
    local_uvs.reserve(projected_coords.size());
    for (const auto &coord : projected_coords) {
      const double centered_u = coord[0] - mean_u;
      const double centered_v = coord[1] - mean_v;
      const double rotated_u = cos_angle * centered_u + sin_angle * centered_v;
      const double rotated_v = -sin_angle * centered_u + cos_angle * centered_v;
      local_uvs.push_back({
          (rotated_u - best_min_u) / max_extent,
          (rotated_v - best_min_v) / max_extent,
      });
    }

    evaluation.split_axis = best_width >= best_height ? 0 : 1;
    evaluation.rect_area = (best_width / max_extent) * (best_height / max_extent);
    evaluation.face_centroids.reserve(local_faces.size());
    for (const auto &face : local_faces) {
      const auto &a = local_uvs[static_cast<size_t>(face[0])];
      const auto &b = local_uvs[static_cast<size_t>(face[1])];
      const auto &c = local_uvs[static_cast<size_t>(face[2])];
      const double area = 0.5 * std::fabs(
          (b[0] - a[0]) * (c[1] - a[1]) -
          (b[1] - a[1]) * (c[0] - a[0]));
      evaluation.triangle_area += area;
      evaluation.face_centroids.push_back({
          (a[0] + b[0] + c[0]) / 3.0,
          (a[1] + b[1] + c[1]) / 3.0,
      });
    }
    evaluation.rect_fill = evaluation.rect_area > 0.0 ? evaluation.triangle_area / evaluation.rect_area : 0.0;
    return evaluation;
  };

  const int64_t pre_low_fill_chart_count = static_cast<int64_t>(charts.size());
  const int64_t oversized_chart_split_count = pre_low_fill_chart_count - source_chart_count;
  double pre_low_fill_triangle_area = 0.0;
  double pre_low_fill_rect_area = 0.0;
  for (const auto &chart : charts) {
    const ChartFillEvaluation evaluation = evaluate_chart_fill(chart);
    pre_low_fill_triangle_area += evaluation.triangle_area;
    pre_low_fill_rect_area += evaluation.rect_area;
  }
  int64_t low_fill_split_candidate_count = 0;
  int64_t low_fill_source_chart_count = 0;
  int64_t low_fill_split_accepted_count = 0;
  int64_t low_fill_split_rejected_count = 0;
  std::vector<std::vector<int64_t>> low_fill_charts;
  low_fill_charts.reserve(charts.size());
  struct LowFillWorkItem {
    std::vector<int64_t> faces;
    int64_t depth = 0;
    bool source_counted = false;
  };
  for (const auto &chart : charts) {
    std::vector<LowFillWorkItem> pending;
    pending.push_back(LowFillWorkItem{chart, 0, false});
    while (!pending.empty()) {
      LowFillWorkItem item = std::move(pending.back());
      pending.pop_back();
      const ChartFillEvaluation parent = evaluate_chart_fill(item.faces);
      const bool can_split = static_cast<int64_t>(item.faces.size()) >= low_fill_split_min_faces &&
                             item.depth < low_fill_split_max_depth &&
                             parent.rect_fill > 0.0 &&
                             parent.rect_fill < low_fill_rect_fill_threshold;
      if (can_split) {
        low_fill_split_candidate_count += 1;
        std::vector<std::pair<double, int64_t>> sortable;
        sortable.reserve(item.faces.size());
        for (size_t index = 0; index < item.faces.size(); ++index) {
          const auto &centroid = parent.face_centroids[index];
          sortable.push_back({centroid[static_cast<size_t>(parent.split_axis)], item.faces[index]});
        }
        std::stable_sort(sortable.begin(), sortable.end(), [](const auto &left, const auto &right) {
          if (std::fabs(left.first - right.first) > 1e-12) {
            return left.first < right.first;
          }
          return left.second < right.second;
        });
        const size_t midpoint_index = sortable.size() / 2;
        std::vector<int64_t> left_faces;
        std::vector<int64_t> right_faces;
        left_faces.reserve(midpoint_index);
        right_faces.reserve(sortable.size() - midpoint_index);
        for (size_t index = 0; index < sortable.size(); ++index) {
          const int64_t face_index = sortable[index].second;
          if (index < midpoint_index) {
            left_faces.push_back(face_index);
          } else {
            right_faces.push_back(face_index);
          }
        }
        if (static_cast<int64_t>(left_faces.size()) >= low_fill_split_min_child_faces &&
            static_cast<int64_t>(right_faces.size()) >= low_fill_split_min_child_faces) {
          const ChartFillEvaluation left = evaluate_chart_fill(left_faces);
          const ChartFillEvaluation right = evaluate_chart_fill(right_faces);
          const double child_rect_area = left.rect_area + right.rect_area;
          const double child_fill = child_rect_area > 0.0 ? (left.triangle_area + right.triangle_area) / child_rect_area : 0.0;
          if (child_fill > parent.rect_fill + low_fill_split_min_improvement) {
            if (!item.source_counted) {
              low_fill_source_chart_count += 1;
            }
            low_fill_split_accepted_count += 1;
            pending.push_back(LowFillWorkItem{std::move(right_faces), item.depth + 1, true});
            pending.push_back(LowFillWorkItem{std::move(left_faces), item.depth + 1, true});
            continue;
          }
        }
        low_fill_split_rejected_count += 1;
      }
      low_fill_charts.push_back(std::move(item.faces));
    }
  }
  charts = std::move(low_fill_charts);

  struct ChartData {
    int64_t original_index = 0;
    std::vector<int64_t> source_vertices;
    std::vector<std::array<int64_t, 3>> local_faces;
    std::vector<std::array<float, 2>> local_uvs;
    float width_ratio = 1.0f;
    float height_ratio = 1.0f;
    float packed_x = 0.0f;
    float packed_y = 0.0f;
    float packed_width = 1.0f;
    float packed_height = 1.0f;
    double local_triangle_area = 0.0;
    double local_rect_area = 1.0;
    double projection_rotation_radians = 0.0;
  };

  const int64_t chart_count = static_cast<int64_t>(charts.size());
  const int64_t chart_split_count = chart_count - source_chart_count;
  std::vector<ChartData> chart_data;
  chart_data.reserve(charts.size());
  std::vector<float> chart_vertices;
  std::vector<float> chart_uvs;
  std::vector<int64_t> chart_faces;
  chart_faces.reserve(mesh.faces.size() * 3);
  int64_t max_chart_faces = 0;
  std::vector<int64_t> local_index_lookup(mesh.vertices.size(), -1);
  std::vector<int64_t> touched_vertices;

  for (int64_t chart_index = 0; chart_index < chart_count; ++chart_index) {
    const auto &chart = charts[static_cast<size_t>(chart_index)];
    max_chart_faces = std::max<int64_t>(max_chart_faces, static_cast<int64_t>(chart.size()));
    std::array<float, 3> average_normal{0.0f, 0.0f, 0.0f};
    for (int64_t face_index : chart) {
      const auto &normal = face_normals[static_cast<size_t>(face_index)];
      average_normal[0] += normal[0];
      average_normal[1] += normal[1];
      average_normal[2] += normal[2];
    }
    const std::array<float, 3> chart_normal = normalize3(average_normal, {0.0f, 0.0f, 1.0f});
    const float nx = std::fabs(chart_normal[0]);
    const float ny = std::fabs(chart_normal[1]);
    const float nz = std::fabs(chart_normal[2]);
    std::array<float, 3> seed_axis{1.0f, 0.0f, 0.0f};
    if (ny <= nx && ny <= nz) {
      seed_axis = {0.0f, 1.0f, 0.0f};
    } else if (nz <= nx && nz <= ny) {
      seed_axis = {0.0f, 0.0f, 1.0f};
    }
    const std::array<float, 3> tangent_u = normalize3(cross3(seed_axis, chart_normal), {1.0f, 0.0f, 0.0f});
    const std::array<float, 3> tangent_v = normalize3(cross3(chart_normal, tangent_u), {0.0f, 1.0f, 0.0f});

    std::vector<int64_t> source_vertices;
    std::vector<std::array<int64_t, 3>> local_faces;
    local_faces.reserve(chart.size());
    for (int64_t face_index : chart) {
      const auto &face = mesh.faces[static_cast<size_t>(face_index)];
      for (int corner = 0; corner < 3; ++corner) {
        const int64_t source_index = face[corner];
        if (local_index_lookup[static_cast<size_t>(source_index)] < 0) {
          local_index_lookup[static_cast<size_t>(source_index)] = static_cast<int64_t>(source_vertices.size());
          source_vertices.push_back(source_index);
          touched_vertices.push_back(source_index);
        }
      }
      local_faces.push_back({
          local_index_lookup[static_cast<size_t>(face[0])],
          local_index_lookup[static_cast<size_t>(face[1])],
          local_index_lookup[static_cast<size_t>(face[2])],
      });
    }

    std::vector<std::array<double, 2>> projected_coords;
    projected_coords.reserve(source_vertices.size());
    double mean_u = 0.0;
    double mean_v = 0.0;
    for (int64_t source_index : source_vertices) {
      const auto &vertex = mesh.vertices[static_cast<size_t>(source_index)];
      const double u = static_cast<double>(vertex[0]) * tangent_u[0] +
                       static_cast<double>(vertex[1]) * tangent_u[1] +
                       static_cast<double>(vertex[2]) * tangent_u[2];
      const double v = static_cast<double>(vertex[0]) * tangent_v[0] +
                       static_cast<double>(vertex[1]) * tangent_v[1] +
                       static_cast<double>(vertex[2]) * tangent_v[2];
      projected_coords.push_back({u, v});
      mean_u += u;
      mean_v += v;
    }
    mean_u /= static_cast<double>(projected_coords.size());
    mean_v /= static_cast<double>(projected_coords.size());
    double cov_uu = 0.0;
    double cov_vv = 0.0;
    double cov_uv = 0.0;
    for (const auto &coord : projected_coords) {
      const double du = coord[0] - mean_u;
      const double dv = coord[1] - mean_v;
      cov_uu += du * du;
      cov_vv += dv * dv;
      cov_uv += du * dv;
    }
    const double pca_angle = std::isfinite(cov_uu) && std::isfinite(cov_vv) && std::isfinite(cov_uv)
                                 ? 0.5 * std::atan2(2.0 * cov_uv, cov_uu - cov_vv)
                                 : 0.0;
    std::array<double, projection_rotation_candidates> candidate_angles{};
    const double step_radians = projection_rotation_step_degrees * pi / 180.0;
    const int64_t midpoint = projection_rotation_candidates / 2;
    for (int64_t candidate = 0; candidate < projection_rotation_candidates; ++candidate) {
      candidate_angles[static_cast<size_t>(candidate)] =
          pca_angle + static_cast<double>(candidate - midpoint) * step_radians;
    }

    double best_angle = 0.0;
    double best_min_u = 0.0;
    double best_min_v = 0.0;
    double best_width = std::numeric_limits<double>::infinity();
    double best_height = std::numeric_limits<double>::infinity();
    double best_area = std::numeric_limits<double>::infinity();
    for (double angle : candidate_angles) {
      const double cos_angle = std::cos(angle);
      const double sin_angle = std::sin(angle);
      double min_u = std::numeric_limits<double>::infinity();
      double min_v = std::numeric_limits<double>::infinity();
      double max_u = -std::numeric_limits<double>::infinity();
      double max_v = -std::numeric_limits<double>::infinity();
      for (const auto &coord : projected_coords) {
        const double centered_u = coord[0] - mean_u;
        const double centered_v = coord[1] - mean_v;
        const double rotated_u = cos_angle * centered_u + sin_angle * centered_v;
        const double rotated_v = -sin_angle * centered_u + cos_angle * centered_v;
        min_u = std::min(min_u, rotated_u);
        min_v = std::min(min_v, rotated_v);
        max_u = std::max(max_u, rotated_u);
        max_v = std::max(max_v, rotated_v);
      }
      const double width = std::max(max_u - min_u, 1e-12);
      const double height = std::max(max_v - min_v, 1e-12);
      const double area = width * height;
      if (area < best_area - 1e-12) {
        best_angle = angle;
        best_min_u = min_u;
        best_min_v = min_v;
        best_width = width;
        best_height = height;
        best_area = area;
      }
    }

    const double max_extent = std::max(best_width, best_height);
    const double cos_angle = std::cos(best_angle);
    const double sin_angle = std::sin(best_angle);
    ChartData data;
    data.original_index = chart_index;
    data.source_vertices = std::move(source_vertices);
    data.local_faces = std::move(local_faces);
    data.width_ratio = static_cast<float>(best_width / max_extent);
    data.height_ratio = static_cast<float>(best_height / max_extent);
    data.local_rect_area = static_cast<double>(data.width_ratio) * static_cast<double>(data.height_ratio);
    data.projection_rotation_radians = best_angle;
    data.local_uvs.reserve(data.source_vertices.size());
    for (const auto &coord : projected_coords) {
      const double centered_u = coord[0] - mean_u;
      const double centered_v = coord[1] - mean_v;
      const double rotated_u = cos_angle * centered_u + sin_angle * centered_v;
      const double rotated_v = -sin_angle * centered_u + cos_angle * centered_v;
      data.local_uvs.push_back({
          static_cast<float>((rotated_u - best_min_u) / max_extent),
          static_cast<float>((rotated_v - best_min_v) / max_extent),
      });
    }
    for (const auto &face : data.local_faces) {
      const auto &a = data.local_uvs[static_cast<size_t>(face[0])];
      const auto &b = data.local_uvs[static_cast<size_t>(face[1])];
      const auto &c = data.local_uvs[static_cast<size_t>(face[2])];
      const double area = 0.5 * std::fabs(
          (static_cast<double>(b[0]) - a[0]) * (static_cast<double>(c[1]) - a[1]) -
          (static_cast<double>(b[1]) - a[1]) * (static_cast<double>(c[0]) - a[0]));
      data.local_triangle_area += area;
    }
    chart_data.push_back(std::move(data));
    for (int64_t source_index : touched_vertices) {
      local_index_lookup[static_cast<size_t>(source_index)] = -1;
    }
    touched_vertices.clear();
  }

  std::vector<int64_t> order;
  order.reserve(chart_data.size());
  for (int64_t index = 0; index < chart_count; ++index) {
    order.push_back(index);
  }
  std::sort(order.begin(), order.end(), [&chart_data](int64_t left_index, int64_t right_index) {
    const auto &left = chart_data[static_cast<size_t>(left_index)];
    const auto &right = chart_data[static_cast<size_t>(right_index)];
    if (std::fabs(left.height_ratio - right.height_ratio) > 1e-9f) {
      return left.height_ratio > right.height_ratio;
    }
    if (std::fabs(left.width_ratio - right.width_ratio) > 1e-9f) {
      return left.width_ratio > right.width_ratio;
    }
    return left.original_index < right.original_index;
  });

  int64_t shelf_rows = 1;
  float packed_width = 0.0f;
  float packed_height = 0.0f;
  auto pack_charts = [&](float scale, bool write) {
    float cursor_x = 0.0f;
    float cursor_y = 0.0f;
    float row_height = 0.0f;
    int64_t rows_used = 1;
    float max_x = 0.0f;
    for (int64_t chart_index : order) {
      auto &chart = chart_data[static_cast<size_t>(chart_index)];
      const float rect_width = std::max(chart.width_ratio * scale, 1e-12f);
      const float rect_height = std::max(chart.height_ratio * scale, 1e-12f);
      if (rect_width > 1.0f || rect_height > 1.0f) {
        return false;
      }
      if (cursor_x > 0.0f && cursor_x + rect_width > 1.0f + 1e-7f) {
        cursor_y += row_height;
        cursor_x = 0.0f;
        row_height = 0.0f;
        rows_used += 1;
      }
      if (cursor_y + rect_height > 1.0f + 1e-7f) {
        return false;
      }
      if (write) {
        chart.packed_x = cursor_x;
        chart.packed_y = cursor_y;
        chart.packed_width = rect_width;
        chart.packed_height = rect_height;
      }
      cursor_x += rect_width;
      row_height = std::max(row_height, rect_height);
      max_x = std::max(max_x, cursor_x);
    }
    if (write) {
      shelf_rows = rows_used;
      packed_width = max_x;
      packed_height = cursor_y + row_height;
    }
    return true;
  };

  float low_scale = 0.0f;
  float high_scale = 1.0f;
  for (int iteration = 0; iteration < 40; ++iteration) {
    const float mid = (low_scale + high_scale) * 0.5f;
    if (pack_charts(mid, false)) {
      low_scale = mid;
    } else {
      high_scale = mid;
    }
  }
  if (low_scale <= 0.0f) {
    throw nb::value_error("native chart UV shelf packing failed to fit charts");
  }
  pack_charts(low_scale, true);

  double packed_rect_area = 0.0;
  double local_triangle_area = 0.0;
  double local_rect_area = 0.0;
  double rotation_abs_sum = 0.0;
  for (const auto &chart : chart_data) {
    packed_rect_area += static_cast<double>(chart.packed_width) * static_cast<double>(chart.packed_height);
    local_triangle_area += chart.local_triangle_area;
    local_rect_area += chart.local_rect_area;
    rotation_abs_sum += std::fabs(chart.projection_rotation_radians);
  }
  const double packed_bounds_area = static_cast<double>(packed_width) * static_cast<double>(packed_height);
  const float content_scale = static_cast<float>(1.0 - 2.0 * tile_padding);

  for (int64_t chart_index : order) {
    const auto &chart = chart_data[static_cast<size_t>(chart_index)];
    const int64_t output_offset = static_cast<int64_t>(chart_vertices.size() / 3);
    for (int64_t source_index : chart.source_vertices) {
      const auto &vertex = mesh.vertices[static_cast<size_t>(source_index)];
      chart_vertices.push_back(vertex[0]);
      chart_vertices.push_back(vertex[1]);
      chart_vertices.push_back(vertex[2]);
    }
    for (const auto &local_uv : chart.local_uvs) {
      const float u = chart.packed_x + chart.packed_width * static_cast<float>(tile_padding) + local_uv[0] * low_scale * content_scale;
      const float v = chart.packed_y + chart.packed_height * static_cast<float>(tile_padding) + local_uv[1] * low_scale * content_scale;
      chart_uvs.push_back(std::clamp(u, 0.0f, 1.0f));
      chart_uvs.push_back(std::clamp(v, 0.0f, 1.0f));
    }
    for (const auto &face : chart.local_faces) {
      chart_faces.push_back(output_offset + face[0]);
      chart_faces.push_back(output_offset + face[1]);
      chart_faces.push_back(output_offset + face[2]);
    }
  }

  nb::dict result;
  const size_t output_vertices = chart_vertices.size() / 3;
  result["vertices"] = mesh_common::make_float32_array(std::move(chart_vertices), output_vertices, 3);
  result["faces"] = mesh_common::make_int64_array(std::move(chart_faces), mesh.faces.size(), 3);
  result["uvs"] = mesh_common::make_float32_array(std::move(chart_uvs), output_vertices, 2);
  nb::dict stats;
  stats["backend"] = "native-chart-atlas";
  stats["source_vertices"] = static_cast<int64_t>(mesh.vertices.size());
  stats["source_faces"] = face_count;
  stats["output_vertices"] = static_cast<int64_t>(output_vertices);
  stats["output_faces"] = face_count;
  stats["source_chart_count"] = source_chart_count;
  stats["chart_count"] = chart_count;
  stats["chart_split_max_faces"] = chart_split_max_faces;
  stats["chart_split_count"] = chart_split_count;
  stats["oversized_source_chart_count"] = oversized_source_chart_count;
  stats["oversized_chart_split_count"] = oversized_chart_split_count;
  stats["pre_low_fill_chart_count"] = pre_low_fill_chart_count;
  stats["pre_low_fill_chart_rect_fill_ratio"] =
      pre_low_fill_rect_area > 0.0 ? pre_low_fill_triangle_area / pre_low_fill_rect_area : 0.0;
  stats["low_fill_rect_fill_threshold"] = low_fill_rect_fill_threshold;
  stats["low_fill_split_min_improvement"] = low_fill_split_min_improvement;
  stats["low_fill_split_min_faces"] = low_fill_split_min_faces;
  stats["low_fill_split_min_child_faces"] = low_fill_split_min_child_faces;
  stats["low_fill_split_max_depth"] = low_fill_split_max_depth;
  stats["low_fill_split_candidate_count"] = low_fill_split_candidate_count;
  stats["low_fill_source_chart_count"] = low_fill_source_chart_count;
  stats["low_fill_split_accepted_count"] = low_fill_split_accepted_count;
  stats["low_fill_split_rejected_count"] = low_fill_split_rejected_count;
  stats["low_fill_chart_split_count"] = chart_count - pre_low_fill_chart_count;
  stats["max_chart_faces"] = max_chart_faces;
  stats["average_chart_faces"] = static_cast<double>(face_count) / static_cast<double>(chart_count);
  stats["chart_angle_degrees"] = chart_angle_degrees;
  stats["chart_normal_cos_threshold"] = cos_threshold;
  stats["projection"] = "local-frame-pca";
  stats["projection_rotation_candidates"] = projection_rotation_candidates;
  stats["projection_rotation_step_degrees"] = projection_rotation_step_degrees;
  stats["average_abs_projection_rotation_radians"] = rotation_abs_sum / static_cast<double>(chart_count);
  stats["chart_rect_fill_ratio"] = local_rect_area > 0.0 ? local_triangle_area / local_rect_area : 0.0;
  stats["packing"] = "aspect-shelf-charts";
  stats["shelf_rows"] = shelf_rows;
  stats["shelf_scale"] = low_scale;
  stats["packed_width"] = packed_width;
  stats["packed_height"] = packed_height;
  stats["packed_bounds_area"] = packed_bounds_area;
  stats["packed_chart_rect_area"] = packed_rect_area;
  stats["shelf_packing_efficiency"] = packed_bounds_area > 0.0 ? packed_rect_area / packed_bounds_area : 0.0;
  stats["atlas_rect_coverage_ratio"] = packed_rect_area;
  stats["tile_padding"] = tile_padding;
  stats["duplicated_vertex_ratio"] = static_cast<double>(output_vertices) / static_cast<double>(mesh.vertices.size());
  result["stats"] = stats;
  return result;
}

nb::bytes textured_glb_payload(
    nb::object vertices_object,
    nb::object faces_object,
    nb::object uvs_object,
    nb::object base_color_object,
    nb::object metallic_roughness_object,
    const std::string &generator,
    const std::string &mesh_name,
    const std::string &material_name) {
  mesh_common::MeshData mesh = mesh_common::load_mesh(vertices_object, faces_object);
  if (mesh.faces.empty()) {
    throw nb::value_error("GLB faces must not be empty");
  }
  const std::vector<std::array<float, 2>> uvs = load_uvs(uvs_object, static_cast<int64_t>(mesh.vertices.size()));
  const TextureImage base_color = load_texture_image(base_color_object, "base color texture", 4);
  const TextureImage metallic_roughness = load_texture_image(metallic_roughness_object, "metallic-roughness texture", 3);
  const std::vector<std::array<float, 3>> normals = compute_vertex_normals(mesh);

  std::vector<uint8_t> bin_blob;
  std::vector<BufferView> views;
  std::vector<std::string> accessor_jsons;
  std::vector<std::string> primitive_jsons;
  std::vector<std::array<float, 3>> local_positions;
  std::vector<std::array<float, 3>> local_normals;
  std::vector<std::array<float, 2>> local_uvs;
  std::vector<uint16_t> local_indices;
  std::unordered_map<int64_t, uint16_t> local_vertex_map;

  auto add_accessor_json = [&accessor_jsons](std::string accessor) {
    const uint32_t index = checked_u32(accessor_jsons.size(), "GLB accessor index");
    accessor_jsons.push_back(std::move(accessor));
    return index;
  };

  auto flush_primitive = [&]() {
    if (local_indices.empty()) {
      return;
    }
    const uint32_t position_view = add_buffer_view(bin_blob, views, vec3_payload(local_positions), kArrayBufferTarget);
    const uint32_t normal_view = add_buffer_view(bin_blob, views, vec3_payload(local_normals), kArrayBufferTarget);
    const uint32_t uv_view = add_buffer_view(bin_blob, views, vec2_payload(local_uvs), kArrayBufferTarget);
    const uint32_t index_view = add_buffer_view(bin_blob, views, uint16_payload(local_indices), kElementArrayBufferTarget);

    const auto position_min = vertex_min(local_positions);
    const auto position_max = vertex_max(local_positions);
    const auto normal_min = vertex_min(local_normals);
    const auto normal_max = vertex_max(local_normals);
    const auto uv_min = uv_min_padded(local_uvs);
    const auto uv_max = uv_max_padded(local_uvs);
    const uint16_t max_local_index = static_cast<uint16_t>(local_positions.size() - 1);

    std::ostringstream position_accessor;
    position_accessor << "{\"bufferView\":" << position_view << ",\"byteOffset\":0,\"componentType\":"
                      << kComponentFloat32 << ",\"count\":" << local_positions.size()
                      << ",\"type\":\"VEC3\",\"min\":" << float_array_json(position_min, 3)
                      << ",\"max\":" << float_array_json(position_max, 3) << "}";
    const uint32_t position_accessor_index = add_accessor_json(position_accessor.str());

    std::ostringstream normal_accessor;
    normal_accessor << "{\"bufferView\":" << normal_view << ",\"byteOffset\":0,\"componentType\":"
                    << kComponentFloat32 << ",\"count\":" << local_normals.size()
                    << ",\"type\":\"VEC3\",\"min\":" << float_array_json(normal_min, 3)
                    << ",\"max\":" << float_array_json(normal_max, 3) << "}";
    const uint32_t normal_accessor_index = add_accessor_json(normal_accessor.str());

    std::ostringstream uv_accessor;
    uv_accessor << "{\"bufferView\":" << uv_view << ",\"byteOffset\":0,\"componentType\":" << kComponentFloat32
                << ",\"count\":" << local_uvs.size() << ",\"type\":\"VEC2\",\"min\":"
                << float_array_json(uv_min, 2) << ",\"max\":" << float_array_json(uv_max, 2) << "}";
    const uint32_t uv_accessor_index = add_accessor_json(uv_accessor.str());

    std::ostringstream index_accessor;
    index_accessor << "{\"bufferView\":" << index_view << ",\"byteOffset\":0,\"componentType\":"
                   << kComponentUint16 << ",\"count\":" << local_indices.size()
                   << ",\"type\":\"SCALAR\",\"min\":[0],\"max\":[" << max_local_index << "]}";
    const uint32_t index_accessor_index = add_accessor_json(index_accessor.str());

    std::ostringstream primitive;
    primitive << "{\"attributes\":{\"POSITION\":" << position_accessor_index << ",\"NORMAL\":"
              << normal_accessor_index << ",\"TEXCOORD_0\":" << uv_accessor_index << "},\"indices\":"
              << index_accessor_index << ",\"material\":0}";
    primitive_jsons.push_back(primitive.str());

    local_positions.clear();
    local_normals.clear();
    local_uvs.clear();
    local_indices.clear();
    local_vertex_map.clear();
  };

  local_positions.reserve(std::min<size_t>(mesh.vertices.size(), static_cast<size_t>(kMaxUint16Index) + 1));
  local_normals.reserve(local_positions.capacity());
  local_uvs.reserve(local_positions.capacity());
  local_indices.reserve(std::min<size_t>(mesh.faces.size() * 3, static_cast<size_t>(kMaxUint16Index) + 1));
  for (const auto &face : mesh.faces) {
    size_t new_vertex_count = 0;
    for (int corner = 0; corner < 3; ++corner) {
      if (local_vertex_map.find(face[corner]) == local_vertex_map.end()) {
        ++new_vertex_count;
      }
    }
    if (!local_indices.empty() &&
        local_positions.size() + new_vertex_count > static_cast<size_t>(kMaxUint16Index) + 1) {
      flush_primitive();
    }
    for (int corner = 0; corner < 3; ++corner) {
      const int64_t source_index = face[corner];
      auto found = local_vertex_map.find(source_index);
      if (found == local_vertex_map.end()) {
        const uint16_t local_index = static_cast<uint16_t>(local_positions.size());
        local_vertex_map.emplace(source_index, local_index);
        local_positions.push_back(mesh.vertices[static_cast<size_t>(source_index)]);
        local_normals.push_back(normals[static_cast<size_t>(source_index)]);
        local_uvs.push_back(uvs[static_cast<size_t>(source_index)]);
        local_indices.push_back(local_index);
      } else {
        local_indices.push_back(found->second);
      }
    }
  }
  flush_primitive();

  const uint32_t base_color_view = add_buffer_view(bin_blob, views, png_payload(base_color));
  const uint32_t metallic_roughness_view = add_buffer_view(bin_blob, views, png_payload(metallic_roughness));
  pad4(bin_blob, 0);

  std::ostringstream json;
  json << std::setprecision(9);
  json << "{\"asset\":{\"version\":\"2.0\",\"generator\":" << quoted(generator) << "},";
  json << "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],";
  json << "\"nodes\":[{\"mesh\":0,\"name\":" << quoted(mesh_name) << "}],";
  json << "\"meshes\":[{\"name\":" << quoted(mesh_name) << ",\"primitives\":[";
  for (size_t index = 0; index < primitive_jsons.size(); ++index) {
    if (index != 0) {
      json << ",";
    }
    json << primitive_jsons[index];
  }
  json << "]}],";
  json << "\"materials\":[{\"name\":" << quoted(material_name)
       << ",\"doubleSided\":true,\"alphaMode\":\"OPAQUE\",\"pbrMetallicRoughness\":{"
       << "\"baseColorTexture\":{\"index\":0},\"metallicRoughnessTexture\":{\"index\":1},"
       << "\"metallicFactor\":1,\"roughnessFactor\":1}}],";
  json << "\"samplers\":[{\"magFilter\":9729,\"minFilter\":9729,\"wrapS\":33071,\"wrapT\":33071}],";
  json << "\"textures\":[{\"sampler\":0,\"source\":0},{\"sampler\":0,\"source\":1}],";
  json << "\"images\":[{\"bufferView\":" << base_color_view
       << ",\"mimeType\":\"image/png\",\"name\":\"baseColorTexture\"},{\"bufferView\":" << metallic_roughness_view
       << ",\"mimeType\":\"image/png\",\"name\":\"metallicRoughnessTexture\"}],";
  json << "\"buffers\":[{\"byteLength\":" << bin_blob.size() << "}],";
  json << "\"bufferViews\":" << buffer_views_json(views) << ",";
  json << "\"accessors\":[";
  for (size_t index = 0; index < accessor_jsons.size(); ++index) {
    if (index != 0) {
      json << ",";
    }
    json << accessor_jsons[index];
  }
  json << "]}";

  std::vector<uint8_t> json_payload = bytes_from_string(json.str());
  pad4(json_payload, static_cast<uint8_t>(' '));
  const uint64_t total_length64 =
      12ull + 8ull + static_cast<uint64_t>(json_payload.size()) + 8ull + static_cast<uint64_t>(bin_blob.size());
  const uint32_t total_length = checked_u32(total_length64, "GLB total length");
  std::vector<uint8_t> payload;
  payload.reserve(total_length);
  append_u32_le(payload, kGlbMagic);
  append_u32_le(payload, 2);
  append_u32_le(payload, total_length);
  append_u32_le(payload, checked_u32(json_payload.size(), "GLB JSON chunk"));
  payload.insert(payload.end(), {'J', 'S', 'O', 'N'});
  payload.insert(payload.end(), json_payload.begin(), json_payload.end());
  append_u32_le(payload, checked_u32(bin_blob.size(), "GLB BIN chunk"));
  payload.insert(payload.end(), {'B', 'I', 'N', '\0'});
  payload.insert(payload.end(), bin_blob.begin(), bin_blob.end());
  return nb::bytes(reinterpret_cast<const char *>(payload.data()), payload.size());
}

}  // namespace mlx_spatialkit
