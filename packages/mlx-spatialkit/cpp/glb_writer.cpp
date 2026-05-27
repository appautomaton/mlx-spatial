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
