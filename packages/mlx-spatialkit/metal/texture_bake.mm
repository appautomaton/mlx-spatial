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
};

struct VoxelRecord {
  uint64_t key;
  std::array<float, 6> attributes;
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
    const NSUInteger thread_width = std::min<NSUInteger>(16, [pipeline threadExecutionWidth]);
    const NSUInteger thread_height = std::max<NSUInteger>(1, std::min<NSUInteger>(16, [pipeline maxTotalThreadsPerThreadgroup] / thread_width));
    MTLSize threads_per_group = MTLSizeMake(thread_width, thread_height, 1);
    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(texture_size), static_cast<NSUInteger>(texture_size), 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads_per_group];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if ([command_buffer error] != nil) {
      throw metal_error("mlx-spatialkit Metal texture bake command failed", [command_buffer error]);
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

    int64_t no_face = 0;
    int64_t sampled = 0;
    int64_t fallback_filled = 0;
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
    const int64_t surface = sampled + missing + out_of_grid + fallback_filled;
    const int64_t exact_missing = exact_missing_before_fill;
    const double pixel_denominator = static_cast<double>(pixel_count);
    const double surface_denominator = surface > 0 ? static_cast<double>(surface) : 1.0;

    nb::dict stats;
    stats["backend"] = atlas_cols > 0 && atlas_rows > 0 ? "metal-face-atlas-nearest" : "metal-uv-nearest";
    stats["metal_device"] = metal_device_name();
    stats["texture_size"] = texture_size;
    stats["texture_pixel_count"] = static_cast<int64_t>(pixel_count);
    stats["voxel_count"] = static_cast<int64_t>(records.size());
    stats["no_face_texel_count"] = no_face;
    stats["uv_surface_texel_count"] = surface;
    stats["exact_sampled_texel_count"] = sampled;
    stats["sampled_texel_count"] = sampled;
    stats["fallback_filled_texel_count"] = fallback_filled;
    stats["dilation_filled_texel_count"] = dilation_filled;
    stats["dilation_pass_count"] = dilation_passes;
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
