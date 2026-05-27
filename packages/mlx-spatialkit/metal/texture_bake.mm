#include "metal_probe.hpp"
#include "texture_bake.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mesh_common.hpp"

namespace nb = nanobind;

namespace mlx_spatialkit {
namespace {

constexpr int64_t kMaxTextureDimension = 8192;
constexpr uint64_t kMaxUvBinFaceReferences = 64ull * 1024ull * 1024ull;

struct BakeConfig {
  uint32_t texture_size;
  uint32_t face_count;
  uint32_t voxel_count;
  uint32_t atlas_cols;
  uint32_t atlas_rows;
  uint32_t atlas_faces_per_tile;
  float tile_padding;
  float origin_x;
  float origin_y;
  float origin_z;
  float voxel_size;
  uint64_t stride_z;
  uint64_t stride_y;
  uint64_t stride_x;
  int32_t grid_z;
  int32_t grid_y;
  int32_t grid_x;
  uint32_t fallback_radius;
  uint32_t uv_bin_cols;
  uint32_t uv_bin_rows;
};

struct VoxelRecord {
  uint64_t key;
  std::array<float, 6> attributes;
};

struct UvBins {
  uint32_t cols = 0;
  uint32_t rows = 0;
  std::vector<uint32_t> offsets;
  std::vector<int32_t> faces;
  uint64_t reference_count = 0;
  uint32_t max_candidates = 0;
  uint32_t non_empty_bins = 0;
};

NSString *texture_bake_metallib_path() {
  Dl_info info;
  if (dladdr(reinterpret_cast<const void *>(&metal_device_available), &info) == 0 || info.dli_fname == nullptr) {
    throw std::runtime_error("failed to locate mlx-spatialkit native module for Metal library loading");
  }
  NSString *module_path = [NSString stringWithUTF8String:info.dli_fname];
  if (module_path == nil) {
    throw std::runtime_error("failed to convert mlx-spatialkit native module path for Metal library loading");
  }
  return [[module_path stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"texture_bake.metallib"];
}

std::runtime_error metal_error(const std::string &message, NSError *error = nil) {
  if (error == nil) {
    return std::runtime_error(message);
  }
  NSString *detail = [error localizedDescription];
  if (detail == nil) {
    return std::runtime_error(message);
  }
  return std::runtime_error(message + ": " + std::string([detail UTF8String]));
}

uint64_t checked_mul(uint64_t left, uint64_t right, const char *name) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
    std::ostringstream message;
    message << name << " size overflows 64-bit arithmetic";
    throw nb::value_error(message.str().c_str());
  }
  return left * right;
}

uint32_t checked_u32(uint64_t value, const char *name) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    std::ostringstream message;
    message << name << " exceeds Metal buffer 32-bit indexing limits";
    throw nb::value_error(message.str().c_str());
  }
  return static_cast<uint32_t>(value);
}

size_t checked_size(uint64_t value, const char *name) {
  if (value > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    std::ostringstream message;
    message << name << " exceeds platform size limits";
    throw nb::value_error(message.str().c_str());
  }
  return static_cast<size_t>(value);
}

std::array<float, 3> parse_origin(nb::object origin_object) {
  nb::tuple origin = nb::cast<nb::tuple>(origin_object);
  if (origin.size() != 3) {
    throw nb::value_error("origin must contain exactly three values");
  }
  return {
      static_cast<float>(nb::cast<double>(origin[0])),
      static_cast<float>(nb::cast<double>(origin[1])),
      static_cast<float>(nb::cast<double>(origin[2])),
  };
}

std::vector<float> flatten_vertices(const mesh_common::MeshData &mesh) {
  std::vector<float> values;
  values.reserve(mesh.vertices.size() * 3);
  for (const auto &vertex : mesh.vertices) {
    values.push_back(vertex[0]);
    values.push_back(vertex[1]);
    values.push_back(vertex[2]);
  }
  return values;
}

std::vector<int32_t> flatten_faces_i32(const mesh_common::MeshData &mesh) {
  std::vector<int32_t> values;
  values.reserve(mesh.faces.size() * 3);
  for (const auto &face : mesh.faces) {
    for (int corner = 0; corner < 3; ++corner) {
      if (face[corner] > std::numeric_limits<int32_t>::max()) {
        throw nb::value_error("texture bake faces exceed int32 Metal index limits");
      }
      values.push_back(static_cast<int32_t>(face[corner]));
    }
  }
  return values;
}

std::vector<float> load_uvs_flat(nb::object uvs_object, int64_t vertex_count) {
  mesh_common::validate_matrix(uvs_object, "texture bake UVs", 2, "float32");
  if (mesh_common::dimension(uvs_object, "texture bake UVs", 0) != vertex_count) {
    std::ostringstream message;
    message << "texture bake UVs must have shape (" << vertex_count << ", 2)";
    throw nb::value_error(message.str().c_str());
  }
  mesh_common::BufferView uv_buffer(uvs_object.ptr(), "texture bake UVs");
  const Py_buffer &view = uv_buffer.get();
  std::vector<float> values;
  values.reserve(static_cast<size_t>(vertex_count * 2));
  for (int64_t row = 0; row < vertex_count; ++row) {
    const float u = mesh_common::read_matrix_value<float>(view, row, 0);
    const float v = mesh_common::read_matrix_value<float>(view, row, 1);
    if (!std::isfinite(u) || !std::isfinite(v)) {
      throw nb::value_error("texture bake UVs must contain only finite values");
    }
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
      throw nb::value_error("texture bake UVs must stay in [0, 1]");
    }
    values.push_back(u);
    values.push_back(v);
  }
  return values;
}

int64_t resolve_uv_bin_side(int64_t texture_size, int64_t face_count) {
  const double face_side = std::ceil(std::sqrt(static_cast<double>(std::max<int64_t>(1, face_count)) / 4.0));
  const int64_t resolved = std::max<int64_t>(4, static_cast<int64_t>(face_side));
  return std::min<int64_t>({resolved, std::max<int64_t>(4, texture_size), 256});
}

UvBins build_uv_bins(
    const mesh_common::MeshData &mesh,
    const std::vector<float> &uvs,
    int64_t texture_size,
    bool enabled) {
  UvBins bins;
  if (!enabled) {
    bins.offsets = {0, 0};
    bins.faces = {0};
    return bins;
  }
  const int64_t side = resolve_uv_bin_side(texture_size, static_cast<int64_t>(mesh.faces.size()));
  bins.cols = checked_u32(static_cast<uint64_t>(side), "UV bin columns");
  bins.rows = checked_u32(static_cast<uint64_t>(side), "UV bin rows");
  const uint64_t bin_count64 = checked_mul(static_cast<uint64_t>(bins.cols), static_cast<uint64_t>(bins.rows), "UV bin count");
  const size_t bin_count = checked_size(bin_count64, "UV bin count");
  std::vector<uint32_t> counts(bin_count, 0);

  auto bin_range = [](float min_value, float max_value, uint32_t bins_per_axis) {
    const float clamped_min = std::clamp(min_value, 0.0f, 1.0f);
    const float clamped_max = std::clamp(max_value, 0.0f, 1.0f);
    const float lo = std::min(clamped_min, clamped_max);
    const float hi = std::max(clamped_min, clamped_max);
    uint32_t first = std::min<uint32_t>(static_cast<uint32_t>(std::floor(lo * static_cast<float>(bins_per_axis))), bins_per_axis - 1);
    uint32_t last = std::min<uint32_t>(static_cast<uint32_t>(std::floor(hi * static_cast<float>(bins_per_axis))), bins_per_axis - 1);
    return std::array<uint32_t, 2>{first, last};
  };

  for (const auto &face : mesh.faces) {
    const float u0 = uvs[static_cast<size_t>(face[0]) * 2 + 0];
    const float v0 = uvs[static_cast<size_t>(face[0]) * 2 + 1];
    const float u1 = uvs[static_cast<size_t>(face[1]) * 2 + 0];
    const float v1 = uvs[static_cast<size_t>(face[1]) * 2 + 1];
    const float u2 = uvs[static_cast<size_t>(face[2]) * 2 + 0];
    const float v2 = uvs[static_cast<size_t>(face[2]) * 2 + 1];
    const auto x_range = bin_range(std::min({u0, u1, u2}), std::max({u0, u1, u2}), bins.cols);
    const auto y_range = bin_range(std::min({v0, v1, v2}), std::max({v0, v1, v2}), bins.rows);
    const uint64_t refs_for_face = checked_mul(
        static_cast<uint64_t>(x_range[1] - x_range[0] + 1),
        static_cast<uint64_t>(y_range[1] - y_range[0] + 1),
        "UV bin face reference");
    bins.reference_count += refs_for_face;
    if (bins.reference_count > kMaxUvBinFaceReferences) {
      std::ostringstream message;
      message << "UV bin face references exceed guard " << kMaxUvBinFaceReferences;
      throw nb::value_error(message.str().c_str());
    }
    for (uint32_t y = y_range[0]; y <= y_range[1]; ++y) {
      for (uint32_t x = x_range[0]; x <= x_range[1]; ++x) {
        const size_t bin = static_cast<size_t>(y) * bins.cols + x;
        counts[bin] += 1;
      }
    }
  }

  bins.offsets.resize(bin_count + 1, 0);
  for (size_t index = 0; index < bin_count; ++index) {
    bins.offsets[index + 1] = bins.offsets[index] + counts[index];
    bins.max_candidates = std::max(bins.max_candidates, counts[index]);
    if (counts[index] > 0) {
      bins.non_empty_bins += 1;
    }
  }
  bins.faces.assign(checked_size(bins.reference_count, "UV bin face references"), 0);
  std::vector<uint32_t> cursors = bins.offsets;
  for (size_t face_index = 0; face_index < mesh.faces.size(); ++face_index) {
    const auto &face = mesh.faces[face_index];
    const float u0 = uvs[static_cast<size_t>(face[0]) * 2 + 0];
    const float v0 = uvs[static_cast<size_t>(face[0]) * 2 + 1];
    const float u1 = uvs[static_cast<size_t>(face[1]) * 2 + 0];
    const float v1 = uvs[static_cast<size_t>(face[1]) * 2 + 1];
    const float u2 = uvs[static_cast<size_t>(face[2]) * 2 + 0];
    const float v2 = uvs[static_cast<size_t>(face[2]) * 2 + 1];
    const auto x_range = bin_range(std::min({u0, u1, u2}), std::max({u0, u1, u2}), bins.cols);
    const auto y_range = bin_range(std::min({v0, v1, v2}), std::max({v0, v1, v2}), bins.rows);
    for (uint32_t y = y_range[0]; y <= y_range[1]; ++y) {
      for (uint32_t x = x_range[0]; x <= x_range[1]; ++x) {
        const size_t bin = static_cast<size_t>(y) * bins.cols + x;
        bins.faces[static_cast<size_t>(cursors[bin]++)] = static_cast<int32_t>(face_index);
      }
    }
  }
  if (bins.faces.empty()) {
    bins.faces = {0};
  }
  return bins;
}

std::vector<VoxelRecord> load_voxels(
    nb::object coordinates_object,
    nb::object attributes_object,
    int64_t decode_resolution,
    float *resolved_voxel_size,
    int32_t *grid_z,
    int32_t *grid_y,
    int32_t *grid_x) {
  mesh_common::validate_matrix(coordinates_object, "texture coordinates", 4, "int32");
  mesh_common::validate_matrix(attributes_object, "texture attributes", 6, "float32");
  const int64_t rows = mesh_common::dimension(coordinates_object, "texture coordinates", 0);
  if (rows <= 0) {
    throw nb::value_error("texture coordinates must contain at least one voxel");
  }
  if (mesh_common::dimension(attributes_object, "texture attributes", 0) != rows) {
    throw nb::value_error("texture coordinate/attribute token mismatch");
  }
  if (decode_resolution == 0 || decode_resolution < -1) {
    throw nb::value_error("decode_resolution must be positive or None");
  }

  mesh_common::BufferView coord_buffer(coordinates_object.ptr(), "texture coordinates");
  mesh_common::BufferView attr_buffer(attributes_object.ptr(), "texture attributes");
  const Py_buffer &coord_view = coord_buffer.get();
  const Py_buffer &attr_view = attr_buffer.get();

  int32_t max_z = 0;
  int32_t max_y = 0;
  int32_t max_x = 0;
  std::vector<std::array<int32_t, 3>> spatial;
  spatial.reserve(static_cast<size_t>(rows));
  std::vector<std::array<float, 6>> attrs;
  attrs.reserve(static_cast<size_t>(rows));
  for (int64_t row = 0; row < rows; ++row) {
    const int32_t batch = mesh_common::read_matrix_value<int32_t>(coord_view, row, 0);
    const int32_t z = mesh_common::read_matrix_value<int32_t>(coord_view, row, 1);
    const int32_t y = mesh_common::read_matrix_value<int32_t>(coord_view, row, 2);
    const int32_t x = mesh_common::read_matrix_value<int32_t>(coord_view, row, 3);
    if (batch != 0) {
      throw nb::value_error("texture baking currently supports only batch index 0");
    }
    if (z < 0 || y < 0 || x < 0) {
      throw nb::value_error("texture spatial coordinates must be non-negative");
    }
    if (decode_resolution > 0 && (z >= decode_resolution || y >= decode_resolution || x >= decode_resolution)) {
      throw nb::value_error("texture spatial coordinates must be < decode_resolution");
    }
    max_z = std::max(max_z, z);
    max_y = std::max(max_y, y);
    max_x = std::max(max_x, x);
    spatial.push_back({z, y, x});
    std::array<float, 6> attr{};
    for (int channel = 0; channel < 6; ++channel) {
      attr[static_cast<size_t>(channel)] = mesh_common::read_matrix_value<float>(attr_view, row, channel);
      if (!std::isfinite(attr[static_cast<size_t>(channel)])) {
        throw nb::value_error("texture attributes must contain only finite values");
      }
    }
    attrs.push_back(attr);
  }

  *grid_z = decode_resolution > 0 ? static_cast<int32_t>(decode_resolution) : max_z + 1;
  *grid_y = decode_resolution > 0 ? static_cast<int32_t>(decode_resolution) : max_y + 1;
  *grid_x = decode_resolution > 0 ? static_cast<int32_t>(decode_resolution) : max_x + 1;
  if (*grid_z <= 0 || *grid_y <= 0 || *grid_x <= 0) {
    throw nb::value_error("texture grid shape must be positive");
  }
  if (!std::isfinite(*resolved_voxel_size)) {
    *resolved_voxel_size = 1.0f / static_cast<float>(decode_resolution > 0 ? decode_resolution : std::max({*grid_z, *grid_y, *grid_x}));
  }
  if (*resolved_voxel_size <= 0.0f) {
    throw nb::value_error("voxel_size must be positive");
  }

  const uint64_t stride_y = static_cast<uint64_t>(*grid_x);
  const uint64_t stride_z = checked_mul(static_cast<uint64_t>(*grid_y), stride_y, "texture grid stride");
  std::vector<VoxelRecord> records;
  records.reserve(static_cast<size_t>(rows));
  for (size_t row = 0; row < spatial.size(); ++row) {
    const auto coord = spatial[row];
    const uint64_t key = static_cast<uint64_t>(coord[0]) * stride_z
        + static_cast<uint64_t>(coord[1]) * stride_y
        + static_cast<uint64_t>(coord[2]);
    records.push_back(VoxelRecord{key, attrs[row]});
  }
  std::sort(records.begin(), records.end(), [](const VoxelRecord &left, const VoxelRecord &right) {
    return left.key < right.key;
  });
  for (size_t row = 1; row < records.size(); ++row) {
    if (records[row - 1].key == records[row].key) {
      throw nb::value_error("texture coordinates must be unique");
    }
  }
  return records;
}

id<MTLBuffer> make_buffer(id<MTLDevice> device, const void *data, size_t bytes, NSString *label) {
  id<MTLBuffer> buffer = [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    throw std::runtime_error("Metal buffer allocation failed for " + std::string([label UTF8String]));
  }
  [buffer setLabel:label];
  return buffer;
}

id<MTLBuffer> make_output_buffer(id<MTLDevice> device, size_t bytes, NSString *label) {
  id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    throw std::runtime_error("Metal output buffer allocation failed for " + std::string([label UTF8String]));
  }
  std::memset([buffer contents], 0, bytes);
  [buffer setLabel:label];
  return buffer;
}

int64_t dilate_missing_surface_texels(
    std::vector<uint8_t> &base_color,
    std::vector<uint8_t> &metallic_roughness,
    std::vector<uint8_t> &coverage,
    int64_t texture_size,
    int64_t max_passes,
    int64_t *passes_run) {
  const size_t side = checked_size(static_cast<uint64_t>(texture_size), "texture dilation side");
  int64_t total_filled = 0;
  *passes_run = 0;
  for (int64_t pass = 0; pass < max_passes; ++pass) {
    std::vector<uint8_t> next_base = base_color;
    std::vector<uint8_t> next_mr = metallic_roughness;
    std::vector<uint8_t> next_coverage = coverage;
    int64_t pass_filled = 0;
    for (size_t y = 0; y < side; ++y) {
      for (size_t x = 0; x < side; ++x) {
        const size_t texel = y * side + x;
        if (coverage[texel] != 2 && coverage[texel] != 3) {
          continue;
        }
        bool found = false;
        size_t source = 0;
        for (int dy = -1; dy <= 1 && !found; ++dy) {
          const int64_t sy = static_cast<int64_t>(y) + dy;
          if (sy < 0 || sy >= texture_size) {
            continue;
          }
          for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) {
              continue;
            }
            const int64_t sx = static_cast<int64_t>(x) + dx;
            if (sx < 0 || sx >= texture_size) {
              continue;
            }
            const size_t neighbor = static_cast<size_t>(sy) * side + static_cast<size_t>(sx);
            if (coverage[neighbor] == 1 || coverage[neighbor] == 4) {
              source = neighbor;
              found = true;
              break;
            }
          }
        }
        if (!found) {
          continue;
        }
        const size_t dst_base = texel * 4;
        const size_t src_base = source * 4;
        next_base[dst_base + 0] = base_color[src_base + 0];
        next_base[dst_base + 1] = base_color[src_base + 1];
        next_base[dst_base + 2] = base_color[src_base + 2];
        next_base[dst_base + 3] = base_color[src_base + 3];
        const size_t dst_mr = texel * 3;
        const size_t src_mr = source * 3;
        next_mr[dst_mr + 0] = metallic_roughness[src_mr + 0];
        next_mr[dst_mr + 1] = metallic_roughness[src_mr + 1];
        next_mr[dst_mr + 2] = metallic_roughness[src_mr + 2];
        next_coverage[texel] = 4;
        pass_filled += 1;
      }
    }
    if (pass_filled == 0) {
      break;
    }
    base_color.swap(next_base);
    metallic_roughness.swap(next_mr);
    coverage.swap(next_coverage);
    total_filled += pass_filled;
    *passes_run = pass + 1;
  }
  return total_filled;
}

int64_t fill_remaining_surface_texels(
    std::vector<uint8_t> &base_color,
    std::vector<uint8_t> &metallic_roughness,
    std::vector<uint8_t> &coverage,
    int64_t texture_size,
    int64_t *seed_count) {
  const size_t side = checked_size(static_cast<uint64_t>(texture_size), "texture surface-fill side");
  const size_t pixel_count = checked_size(
      checked_mul(static_cast<uint64_t>(texture_size), static_cast<uint64_t>(texture_size), "texture surface-fill pixels"),
      "texture surface-fill pixels");
  constexpr uint32_t unvisited = std::numeric_limits<uint32_t>::max();
  std::vector<uint32_t> nearest_source(pixel_count, unvisited);
  std::vector<uint32_t> queue;
  queue.reserve(pixel_count);
  *seed_count = 0;

  for (size_t texel = 0; texel < pixel_count; ++texel) {
    if (coverage[texel] != 1 && coverage[texel] != 4) {
      continue;
    }
    if (base_color[texel * 4 + 3] == 0) {
      continue;
    }
    const uint32_t texel_index = checked_u32(static_cast<uint64_t>(texel), "texture surface-fill queue index");
    nearest_source[texel] = texel_index;
    queue.push_back(texel_index);
  }
  *seed_count = static_cast<int64_t>(queue.size());

  int64_t filled = 0;
  size_t head = 0;
  while (head < queue.size()) {
    const size_t source = static_cast<size_t>(queue[head++]);
    const size_t source_x = source % side;
    const size_t source_y = source / side;
    for (int dy = -1; dy <= 1; ++dy) {
      const int64_t y = static_cast<int64_t>(source_y) + dy;
      if (y < 0 || y >= texture_size) {
        continue;
      }
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0) {
          continue;
        }
        const int64_t x = static_cast<int64_t>(source_x) + dx;
        if (x < 0 || x >= texture_size) {
          continue;
        }
        const size_t target = static_cast<size_t>(y) * side + static_cast<size_t>(x);
        if (nearest_source[target] != unvisited) {
          continue;
        }
        const uint32_t fill_source = nearest_source[source];
        nearest_source[target] = fill_source;
        if (coverage[target] == 2 || coverage[target] == 3) {
          const size_t target_base = target * 4;
          const size_t source_base = static_cast<size_t>(fill_source) * 4;
          base_color[target_base + 0] = base_color[source_base + 0];
          base_color[target_base + 1] = base_color[source_base + 1];
          base_color[target_base + 2] = base_color[source_base + 2];
          base_color[target_base + 3] = base_color[source_base + 3];
          const size_t target_mr = target * 3;
          const size_t source_mr = static_cast<size_t>(fill_source) * 3;
          metallic_roughness[target_mr + 0] = metallic_roughness[source_mr + 0];
          metallic_roughness[target_mr + 1] = metallic_roughness[source_mr + 1];
          metallic_roughness[target_mr + 2] = metallic_roughness[source_mr + 2];
          coverage[target] = 5;
          filled += 1;
        }
        queue.push_back(checked_u32(static_cast<uint64_t>(target), "texture surface-fill queue index"));
      }
    }
  }
  return filled;
}

int64_t fill_no_face_gutter_texels(
    std::vector<uint8_t> &base_color,
    std::vector<uint8_t> &metallic_roughness,
    const std::vector<uint8_t> &coverage,
    int64_t texture_size,
    int64_t max_passes,
    int64_t *passes_run) {
  const size_t side = checked_size(static_cast<uint64_t>(texture_size), "texture gutter-fill side");
  int64_t total_filled = 0;
  *passes_run = 0;
  for (int64_t pass = 0; pass < max_passes; ++pass) {
    std::vector<uint8_t> next_base = base_color;
    std::vector<uint8_t> next_mr = metallic_roughness;
    int64_t pass_filled = 0;
    for (size_t y = 0; y < side; ++y) {
      for (size_t x = 0; x < side; ++x) {
        const size_t texel = y * side + x;
        if (coverage[texel] != 0) {
          continue;
        }
        const size_t dst_base = texel * 4;
        if (base_color[dst_base + 0] != 0 || base_color[dst_base + 1] != 0 || base_color[dst_base + 2] != 0) {
          continue;
        }
        bool found = false;
        size_t source = 0;
        for (int dy = -1; dy <= 1 && !found; ++dy) {
          const int64_t sy = static_cast<int64_t>(y) + dy;
          if (sy < 0 || sy >= texture_size) {
            continue;
          }
          for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) {
              continue;
            }
            const int64_t sx = static_cast<int64_t>(x) + dx;
            if (sx < 0 || sx >= texture_size) {
              continue;
            }
            const size_t neighbor = static_cast<size_t>(sy) * side + static_cast<size_t>(sx);
            const size_t src_base = neighbor * 4;
            if (base_color[src_base + 0] == 0 && base_color[src_base + 1] == 0 &&
                base_color[src_base + 2] == 0 && base_color[src_base + 3] == 0) {
              continue;
            }
            source = neighbor;
            found = true;
            break;
          }
        }
        if (!found) {
          continue;
        }
        const size_t src_base = source * 4;
        next_base[dst_base + 0] = base_color[src_base + 0];
        next_base[dst_base + 1] = base_color[src_base + 1];
        next_base[dst_base + 2] = base_color[src_base + 2];
        next_base[dst_base + 3] = base_color[dst_base + 3];
        const size_t dst_mr = texel * 3;
        const size_t src_mr = source * 3;
        next_mr[dst_mr + 0] = metallic_roughness[src_mr + 0];
        next_mr[dst_mr + 1] = metallic_roughness[src_mr + 1];
        next_mr[dst_mr + 2] = metallic_roughness[src_mr + 2];
        pass_filled += 1;
      }
    }
    if (pass_filled == 0) {
      break;
    }
    base_color.swap(next_base);
    metallic_roughness.swap(next_mr);
    total_filled += pass_filled;
    *passes_run = pass + 1;
  }
  return total_filled;
}

int64_t resolve_dilation_max_passes(int64_t texture_size, int64_t atlas_cols, int64_t atlas_rows) {
  if (atlas_cols <= 0 || atlas_rows <= 0) {
    return 8;
  }
  const int64_t atlas_side = std::max<int64_t>(atlas_cols, atlas_rows);
  const double tile_pixels = static_cast<double>(texture_size) / static_cast<double>(std::max<int64_t>(1, atlas_side));
  const int64_t high_res_floor = static_cast<int64_t>(std::ceil(static_cast<double>(texture_size) / 160.0));
  const int64_t passes = static_cast<int64_t>(std::ceil(tile_pixels * 2.0));
  return std::clamp<int64_t>(std::max<int64_t>(passes, high_res_floor), 8, 64);
}

int64_t resolve_fallback_radius(int64_t texture_size, int64_t atlas_cols, int64_t atlas_rows) {
  if (atlas_cols <= 0 || atlas_rows <= 0) {
    return 12;
  }
  const int64_t atlas_side = std::max<int64_t>(atlas_cols, atlas_rows);
  const double tile_pixels = static_cast<double>(texture_size) / static_cast<double>(std::max<int64_t>(1, atlas_side));
  const int64_t high_res_floor = static_cast<int64_t>(std::ceil(static_cast<double>(texture_size) / 171.0));
  const int64_t radius = static_cast<int64_t>(std::ceil(tile_pixels * 2.0));
  return std::clamp<int64_t>(std::max<int64_t>(radius, high_res_floor), 12, 24);
}

}  // namespace

bool metal_device_available() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device != nil;
  }
}

std::string metal_device_name() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      return "unavailable";
    }
    NSString *name = [device name];
    if (name == nil) {
      return "unknown";
    }
    return std::string([name UTF8String]);
  }
}

nb::dict bake_pbr_texture_metal(
    nb::object vertices_object,
    nb::object faces_object,
    nb::object uvs_object,
    nb::object texture_coordinates,
    nb::object texture_attributes,
    int64_t texture_size,
    nb::object origin_object,
    double voxel_size,
    int64_t decode_resolution,
    int64_t atlas_cols,
    int64_t atlas_rows,
    int64_t atlas_faces_per_tile,
    double tile_padding,
    int64_t max_texture_pixels) {
  if (texture_size <= 0) {
    throw nb::value_error("texture_size must be positive");
  }
  if (texture_size > kMaxTextureDimension) {
    throw nb::value_error("texture_size exceeds mlx-spatialkit Metal texture dimension guard");
  }
  if (max_texture_pixels <= 0) {
    throw nb::value_error("max_texture_pixels must be positive");
  }
  const uint64_t pixel_count = checked_mul(static_cast<uint64_t>(texture_size), static_cast<uint64_t>(texture_size), "texture");
  const uint64_t pixel_guard = static_cast<uint64_t>(max_texture_pixels);
  if (pixel_count > pixel_guard) {
    std::ostringstream message;
    message << "texture bake would allocate " << pixel_count << " pixels, above guard " << pixel_guard;
    throw nb::value_error(message.str().c_str());
  }
  if (atlas_cols < 0 || atlas_rows < 0) {
    throw nb::value_error("atlas_cols and atlas_rows must be non-negative");
  }
  if (atlas_faces_per_tile < 0) {
    throw nb::value_error("atlas_faces_per_tile must be non-negative");
  }
  if ((atlas_cols > 0 || atlas_rows > 0) && atlas_faces_per_tile <= 0) {
    throw nb::value_error("atlas_faces_per_tile must be positive when atlas dimensions are provided");
  }
  if (tile_padding < 0.0 || tile_padding >= 0.45) {
    throw nb::value_error("tile_padding must be in [0, 0.45)");
  }

  mesh_common::MeshData mesh = mesh_common::load_mesh(vertices_object, faces_object);
  if (mesh.faces.empty()) {
    throw nb::value_error("texture bake requires at least one face");
  }
  std::vector<float> vertices = flatten_vertices(mesh);
  std::vector<int32_t> faces = flatten_faces_i32(mesh);
  std::vector<float> uvs = load_uvs_flat(uvs_object, static_cast<int64_t>(mesh.vertices.size()));
  const bool use_uv_bins = atlas_cols == 0 && atlas_rows == 0;
  UvBins uv_bins = build_uv_bins(mesh, uvs, texture_size, use_uv_bins);
  std::array<float, 3> origin = parse_origin(origin_object);
  float resolved_voxel_size = std::isfinite(voxel_size) ? static_cast<float>(voxel_size) : std::numeric_limits<float>::quiet_NaN();
  int32_t grid_z = 0;
  int32_t grid_y = 0;
  int32_t grid_x = 0;
  std::vector<VoxelRecord> records = load_voxels(
      texture_coordinates,
      texture_attributes,
      decode_resolution,
      &resolved_voxel_size,
      &grid_z,
      &grid_y,
      &grid_x);

  const uint64_t stride_x = 1;
  const uint64_t stride_y = static_cast<uint64_t>(grid_x);
  const uint64_t stride_z = checked_mul(static_cast<uint64_t>(grid_y), stride_y, "texture grid stride");
  std::vector<uint64_t> keys;
  std::vector<float> attributes;
  keys.reserve(records.size());
  attributes.reserve(records.size() * 6);
  for (const auto &record : records) {
    keys.push_back(record.key);
    for (float value : record.attributes) {
      attributes.push_back(value);
    }
  }

  checked_u32(vertices.size(), "vertex buffer");
  checked_u32(faces.size(), "face buffer");
  checked_u32(uvs.size(), "uv buffer");
  checked_u32(uv_bins.offsets.size(), "UV bin offset buffer");
  checked_u32(uv_bins.faces.size(), "UV bin face buffer");
  checked_u32(keys.size(), "voxel key buffer");
  checked_u32(attributes.size(), "voxel attribute buffer");
  checked_u32(pixel_count, "texture pixel count");
  const size_t base_bytes = checked_size(checked_mul(pixel_count, 4, "base color output"), "base color output");
  const size_t mr_bytes = checked_size(checked_mul(pixel_count, 3, "metallic roughness output"), "metallic roughness output");
  const size_t coverage_bytes = checked_size(pixel_count, "coverage output");

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      throw std::runtime_error("Metal device unavailable for mlx-spatialkit texture bake");
    }

    NSError *error = nil;
    NSString *library_path = texture_bake_metallib_path();
    NSURL *library_url = [NSURL fileURLWithPath:library_path];
    id<MTLLibrary> library = [device newLibraryWithURL:library_url error:&error];
    if (library == nil) {
      throw metal_error("failed to load mlx-spatialkit texture_bake.metallib", error);
    }
    id<MTLFunction> function = [library newFunctionWithName:@"mlx_spatialkit_bake_pbr_texture"];
    if (function == nil) {
      throw std::runtime_error("failed to load mlx_spatialkit_bake_pbr_texture Metal function");
    }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
      throw metal_error("failed to create mlx-spatialkit texture bake Metal pipeline", error);
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      throw std::runtime_error("failed to create mlx-spatialkit Metal command queue");
    }

    const int64_t fallback_radius = resolve_fallback_radius(texture_size, atlas_cols, atlas_rows);
    BakeConfig config{
        checked_u32(static_cast<uint64_t>(texture_size), "texture_size"),
        checked_u32(mesh.faces.size(), "face count"),
        checked_u32(records.size(), "voxel count"),
        checked_u32(static_cast<uint64_t>(atlas_cols), "atlas_cols"),
        checked_u32(static_cast<uint64_t>(atlas_rows), "atlas_rows"),
        checked_u32(static_cast<uint64_t>(atlas_faces_per_tile), "atlas_faces_per_tile"),
        static_cast<float>(tile_padding),
        origin[0],
        origin[1],
        origin[2],
        resolved_voxel_size,
        stride_z,
        stride_y,
        stride_x,
        grid_z,
        grid_y,
        grid_x,
        checked_u32(static_cast<uint64_t>(fallback_radius), "fallback_radius"),
        uv_bins.cols,
        uv_bins.rows,
    };

    id<MTLBuffer> vertex_buffer = make_buffer(device, vertices.data(), vertices.size() * sizeof(float), @"mlx-spatialkit vertices");
    id<MTLBuffer> face_buffer = make_buffer(device, faces.data(), faces.size() * sizeof(int32_t), @"mlx-spatialkit faces");
    id<MTLBuffer> uv_buffer = make_buffer(device, uvs.data(), uvs.size() * sizeof(float), @"mlx-spatialkit uvs");
    id<MTLBuffer> key_buffer = make_buffer(device, keys.data(), keys.size() * sizeof(uint64_t), @"mlx-spatialkit voxel keys");
    id<MTLBuffer> attr_buffer = make_buffer(device, attributes.data(), attributes.size() * sizeof(float), @"mlx-spatialkit voxel attributes");
    id<MTLBuffer> config_buffer = make_buffer(device, &config, sizeof(BakeConfig), @"mlx-spatialkit bake config");
    id<MTLBuffer> base_buffer = make_output_buffer(device, base_bytes, @"mlx-spatialkit base color");
    id<MTLBuffer> mr_buffer = make_output_buffer(device, mr_bytes, @"mlx-spatialkit metallic roughness");
    id<MTLBuffer> coverage_buffer = make_output_buffer(device, coverage_bytes, @"mlx-spatialkit coverage");
    id<MTLBuffer> uv_bin_offsets_buffer = make_buffer(
        device,
        uv_bins.offsets.data(),
        uv_bins.offsets.size() * sizeof(uint32_t),
        @"mlx-spatialkit UV bin offsets");
    id<MTLBuffer> uv_bin_faces_buffer = make_buffer(
        device,
        uv_bins.faces.data(),
        uv_bins.faces.size() * sizeof(int32_t),
        @"mlx-spatialkit UV bin faces");

    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    if (command_buffer == nil) {
      throw std::runtime_error("failed to create mlx-spatialkit Metal command buffer");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      throw std::runtime_error("failed to create mlx-spatialkit Metal command encoder");
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:vertex_buffer offset:0 atIndex:0];
    [encoder setBuffer:face_buffer offset:0 atIndex:1];
    [encoder setBuffer:uv_buffer offset:0 atIndex:2];
    [encoder setBuffer:key_buffer offset:0 atIndex:3];
    [encoder setBuffer:attr_buffer offset:0 atIndex:4];
    [encoder setBuffer:config_buffer offset:0 atIndex:5];
    [encoder setBuffer:base_buffer offset:0 atIndex:6];
    [encoder setBuffer:mr_buffer offset:0 atIndex:7];
    [encoder setBuffer:coverage_buffer offset:0 atIndex:8];
    [encoder setBuffer:uv_bin_offsets_buffer offset:0 atIndex:9];
    [encoder setBuffer:uv_bin_faces_buffer offset:0 atIndex:10];
    const NSUInteger thread_width = std::min<NSUInteger>(16, [pipeline threadExecutionWidth]);
    const NSUInteger thread_height = std::max<NSUInteger>(1, std::min<NSUInteger>(16, [pipeline maxTotalThreadsPerThreadgroup] / thread_width));
    MTLSize threads_per_group = MTLSizeMake(thread_width, thread_height, 1);
    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(texture_size), static_cast<NSUInteger>(texture_size), 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads_per_group];
    [encoder endEncoding];
    {
      nb::gil_scoped_release release;
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      if ([command_buffer error] != nil) {
        throw metal_error("mlx-spatialkit Metal texture bake command failed", [command_buffer error]);
      }
    }

    const auto *base_ptr = static_cast<const uint8_t *>([base_buffer contents]);
    const auto *mr_ptr = static_cast<const uint8_t *>([mr_buffer contents]);
    const auto *coverage_ptr = static_cast<const uint8_t *>([coverage_buffer contents]);
    std::vector<uint8_t> base_color(base_ptr, base_ptr + base_bytes);
    std::vector<uint8_t> metallic_roughness(mr_ptr, mr_ptr + mr_bytes);
    std::vector<uint8_t> coverage(coverage_ptr, coverage_ptr + coverage_bytes);

    int64_t exact_missing_before_fill = 0;
    for (uint8_t value : coverage) {
      if (value == 2 || value == 4) {
        exact_missing_before_fill += 1;
      }
    }
    int64_t dilation_passes = 0;
    const int64_t dilation_max_passes = resolve_dilation_max_passes(texture_size, atlas_cols, atlas_rows);
    const int64_t dilation_filled = dilate_missing_surface_texels(
        base_color,
        metallic_roughness,
        coverage,
        texture_size,
        dilation_max_passes,
        &dilation_passes);
    int64_t surface_fill_seed_texels = 0;
    const int64_t surface_filled = fill_remaining_surface_texels(
        base_color,
        metallic_roughness,
        coverage,
        texture_size,
        &surface_fill_seed_texels);
    constexpr int64_t gutter_fill_max_passes = 4;
    int64_t gutter_fill_passes = 0;
    const int64_t gutter_filled = fill_no_face_gutter_texels(
        base_color,
        metallic_roughness,
        coverage,
        texture_size,
        gutter_fill_max_passes,
        &gutter_fill_passes);

    int64_t no_face = 0;
    int64_t sampled = 0;
    int64_t fallback_filled = 0;
    int64_t surface_filled_count = 0;
    int64_t missing = 0;
    int64_t out_of_grid = 0;
    for (uint8_t value : coverage) {
      if (value == 0) {
        no_face += 1;
      } else if (value == 1) {
        sampled += 1;
      } else if (value == 2) {
        missing += 1;
      } else if (value == 3) {
        out_of_grid += 1;
      } else if (value == 4) {
        fallback_filled += 1;
      } else if (value == 5) {
        surface_filled_count += 1;
      }
    }
    int64_t visible_base_color = 0;
    int64_t nonzero_rgb = 0;
    for (uint64_t pixel = 0; pixel < pixel_count; ++pixel) {
      const size_t offset = static_cast<size_t>(pixel) * 4;
      if (base_color[offset + 3] != 0) {
        visible_base_color += 1;
      }
      if (base_color[offset + 0] != 0 || base_color[offset + 1] != 0 || base_color[offset + 2] != 0) {
        nonzero_rgb += 1;
      }
    }
    const int64_t surface = sampled + missing + out_of_grid + fallback_filled + surface_filled_count;
    const int64_t exact_missing = exact_missing_before_fill;
    const double pixel_denominator = static_cast<double>(pixel_count);
    const double surface_denominator = surface > 0 ? static_cast<double>(surface) : 1.0;

    nb::dict stats;
    stats["backend"] = atlas_cols > 0 && atlas_rows > 0 ? "metal-face-atlas-nearest" : "metal-uv-binned-nearest";
    stats["metal_device"] = metal_device_name();
    stats["texture_size"] = texture_size;
    stats["texture_pixel_count"] = static_cast<int64_t>(pixel_count);
    stats["voxel_count"] = static_cast<int64_t>(records.size());
    stats["no_face_texel_count"] = no_face;
    stats["uv_surface_texel_count"] = surface;
    stats["exact_sampled_texel_count"] = sampled;
    stats["sampled_texel_count"] = sampled;
    stats["fallback_filled_texel_count"] = fallback_filled;
    stats["surface_fill_enabled"] = true;
    stats["surface_fill_seed_texel_count"] = surface_fill_seed_texels;
    stats["surface_filled_texel_count"] = surface_filled_count;
    stats["surface_fill_filled_texel_count"] = surface_filled;
    stats["surface_unfilled_texel_count"] = missing + out_of_grid;
    stats["dilation_filled_texel_count"] = dilation_filled;
    stats["dilation_pass_count"] = dilation_passes;
    stats["gutter_fill_enabled"] = true;
    stats["gutter_fill_max_passes"] = gutter_fill_max_passes;
    stats["gutter_fill_pass_count"] = gutter_fill_passes;
    stats["gutter_filled_texel_count"] = gutter_filled;
    stats["exact_missing_texel_count"] = exact_missing;
    stats["missing_texel_count"] = missing;
    stats["out_of_grid_texel_count"] = out_of_grid;
    stats["visible_base_color_texel_count"] = visible_base_color;
    stats["nonzero_rgb_texel_count"] = nonzero_rgb;
    stats["raw_coverage_ratio"] = static_cast<double>(sampled) / pixel_denominator;
    stats["final_visible_coverage_ratio"] = static_cast<double>(visible_base_color) / pixel_denominator;
    stats["uv_surface_exact_coverage_ratio"] = static_cast<double>(sampled) / surface_denominator;
    stats["uv_surface_final_visible_coverage_ratio"] = static_cast<double>(visible_base_color) / surface_denominator;
    stats["coverage_ratio"] = static_cast<double>(visible_base_color) / pixel_denominator;
    stats["origin"] = nb::make_tuple(origin[0], origin[1], origin[2]);
    stats["voxel_size"] = resolved_voxel_size;
    stats["atlas_cols"] = atlas_cols;
    stats["atlas_rows"] = atlas_rows;
    stats["atlas_faces_per_tile"] = atlas_faces_per_tile;
    stats["uv_bin_cols"] = uv_bins.cols;
    stats["uv_bin_rows"] = uv_bins.rows;
    stats["uv_bin_count"] = static_cast<int64_t>(static_cast<uint64_t>(uv_bins.cols) * static_cast<uint64_t>(uv_bins.rows));
    stats["uv_bin_face_reference_count"] = static_cast<int64_t>(uv_bins.reference_count);
    stats["uv_bin_max_candidate_faces"] = static_cast<int64_t>(uv_bins.max_candidates);
    stats["uv_bin_non_empty_count"] = static_cast<int64_t>(uv_bins.non_empty_bins);
    stats["uv_bin_average_candidate_faces"] = uv_bins.cols > 0 && uv_bins.rows > 0
        ? static_cast<double>(uv_bins.reference_count) / static_cast<double>(static_cast<uint64_t>(uv_bins.cols) * static_cast<uint64_t>(uv_bins.rows))
        : 0.0;
    stats["uv_bin_face_reference_guard"] = static_cast<int64_t>(kMaxUvBinFaceReferences);
    stats["uv_bin_guard_passed"] = true;
    stats["fallback_radius"] = config.fallback_radius;
    stats["dilation_max_passes"] = dilation_max_passes;

    nb::dict result;
    result["base_color_rgba"] = mesh_common::make_uint8_array(std::move(base_color), static_cast<size_t>(texture_size), static_cast<size_t>(texture_size), 4);
    result["metallic_roughness"] = mesh_common::make_uint8_array(std::move(metallic_roughness), static_cast<size_t>(texture_size), static_cast<size_t>(texture_size), 3);
    result["coverage_mask"] = mesh_common::make_uint8_array(std::move(coverage), static_cast<size_t>(texture_size), static_cast<size_t>(texture_size));
    result["stats"] = stats;
    return result;
  }
}

}  // namespace mlx_spatialkit
