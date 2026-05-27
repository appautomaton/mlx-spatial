#include <metal_stdlib>

using namespace metal;

struct BakeConfig {
  uint texture_size;
  uint face_count;
  uint voxel_count;
  uint atlas_cols;
  uint atlas_rows;
  float tile_padding;
  float origin_x;
  float origin_y;
  float origin_z;
  float voxel_size;
  ulong stride_z;
  ulong stride_y;
  ulong stride_x;
  int grid_z;
  int grid_y;
  int grid_x;
};

static inline bool barycentric_uv(float2 p, float2 a, float2 b, float2 c, thread float3 &weights) {
  float2 v0 = b - a;
  float2 v1 = c - a;
  float2 v2 = p - a;
  float denom = v0.x * v1.y - v1.x * v0.y;
  if (fabs(denom) < 1e-12f) {
    return false;
  }
  float w1 = (v2.x * v1.y - v1.x * v2.y) / denom;
  float w2 = (v0.x * v2.y - v2.x * v0.y) / denom;
  float w0 = 1.0f - w1 - w2;
  weights = float3(w0, w1, w2);
  return w0 >= -1e-5f && w1 >= -1e-5f && w2 >= -1e-5f;
}

static inline int find_voxel_key(device const ulong *keys, uint count, ulong key) {
  uint left = 0;
  uint right = count;
  while (left < right) {
    uint mid = left + ((right - left) >> 1);
    ulong value = keys[mid];
    if (value < key) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  if (left < count && keys[left] == key) {
    return int(left);
  }
  return -1;
}

kernel void mlx_spatialkit_bake_pbr_texture(
    device const float *vertices [[buffer(0)]],
    device const int *faces [[buffer(1)]],
    device const float *uvs [[buffer(2)]],
    device const ulong *voxel_keys [[buffer(3)]],
    device const float *voxel_attributes [[buffer(4)]],
    constant BakeConfig &config [[buffer(5)]],
    device uchar *base_color_rgba [[buffer(6)]],
    device uchar *metallic_roughness [[buffer(7)]],
    device uchar *coverage [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= config.texture_size || gid.y >= config.texture_size) {
    return;
  }

  uint texel = gid.y * config.texture_size + gid.x;
  float2 uv = (float2(gid) + 0.5f) / float(config.texture_size);
  int face_index = -1;
  float3 weights = float3(0.0f);

  if (config.atlas_cols > 0 && config.atlas_rows > 0) {
    uint col = min(uint(floor(uv.x * float(config.atlas_cols))), config.atlas_cols - 1);
    uint row = min(uint(floor(uv.y * float(config.atlas_rows))), config.atlas_rows - 1);
    uint candidate = row * config.atlas_cols + col;
    if (candidate < config.face_count) {
      float2 local_uv = float2(uv.x * float(config.atlas_cols) - float(col),
                               uv.y * float(config.atlas_rows) - float(row));
      float scale = 1.0f - 2.0f * config.tile_padding;
      if (scale > 0.0f) {
        float w1 = (local_uv.x - config.tile_padding) / scale;
        float w2 = (local_uv.y - config.tile_padding) / scale;
        float w0 = 1.0f - w1 - w2;
        if (w0 >= -1e-5f && w1 >= -1e-5f && w2 >= -1e-5f) {
          face_index = int(candidate);
          weights = float3(w0, w1, w2);
        }
      }
    }
  } else {
    for (uint i = 0; i < config.face_count; ++i) {
      int ia = faces[i * 3 + 0];
      int ib = faces[i * 3 + 1];
      int ic = faces[i * 3 + 2];
      float2 a = float2(uvs[ia * 2 + 0], uvs[ia * 2 + 1]);
      float2 b = float2(uvs[ib * 2 + 0], uvs[ib * 2 + 1]);
      float2 c = float2(uvs[ic * 2 + 0], uvs[ic * 2 + 1]);
      if (barycentric_uv(uv, a, b, c, weights)) {
        face_index = int(i);
        break;
      }
    }
  }

  if (face_index < 0) {
    coverage[texel] = 0;
    base_color_rgba[texel * 4 + 0] = 0;
    base_color_rgba[texel * 4 + 1] = 0;
    base_color_rgba[texel * 4 + 2] = 0;
    base_color_rgba[texel * 4 + 3] = 0;
    metallic_roughness[texel * 3 + 0] = 0;
    metallic_roughness[texel * 3 + 1] = 255;
    metallic_roughness[texel * 3 + 2] = 0;
    return;
  }

  uint face_offset = uint(face_index) * 3;
  int ia = faces[face_offset + 0];
  int ib = faces[face_offset + 1];
  int ic = faces[face_offset + 2];
  float3 a = float3(vertices[ia * 3 + 0], vertices[ia * 3 + 1], vertices[ia * 3 + 2]);
  float3 b = float3(vertices[ib * 3 + 0], vertices[ib * 3 + 1], vertices[ib * 3 + 2]);
  float3 c = float3(vertices[ic * 3 + 0], vertices[ic * 3 + 1], vertices[ic * 3 + 2]);
  float3 position = weights.x * a + weights.y * b + weights.z * c;

  int x = int(floor((position.x - config.origin_x) / config.voxel_size + 0.5f));
  int y = int(floor((position.y - config.origin_y) / config.voxel_size + 0.5f));
  int z = int(floor((position.z - config.origin_z) / config.voxel_size + 0.5f));
  if (x < 0 || y < 0 || z < 0 || x >= config.grid_x || y >= config.grid_y || z >= config.grid_z) {
    coverage[texel] = 3;
    return;
  }
  ulong key = ulong(z) * config.stride_z + ulong(y) * config.stride_y + ulong(x) * config.stride_x;
  int voxel_index = find_voxel_key(voxel_keys, config.voxel_count, key);
  if (voxel_index < 0) {
    coverage[texel] = 2;
    return;
  }

  device const float *attr = voxel_attributes + uint(voxel_index) * 6;
  float r = clamp(attr[0], 0.0f, 1.0f);
  float g = clamp(attr[1], 0.0f, 1.0f);
  float bch = clamp(attr[2], 0.0f, 1.0f);
  float metallic = clamp(attr[3], 0.0f, 1.0f);
  float roughness = clamp(attr[4], 0.0f, 1.0f);
  float alpha = clamp(attr[5], 0.0f, 1.0f);

  coverage[texel] = 1;
  base_color_rgba[texel * 4 + 0] = uchar(r * 255.0f + 0.5f);
  base_color_rgba[texel * 4 + 1] = uchar(g * 255.0f + 0.5f);
  base_color_rgba[texel * 4 + 2] = uchar(bch * 255.0f + 0.5f);
  base_color_rgba[texel * 4 + 3] = uchar(alpha * 255.0f + 0.5f);
  metallic_roughness[texel * 3 + 0] = 0;
  metallic_roughness[texel * 3 + 1] = uchar(roughness * 255.0f + 0.5f);
  metallic_roughness[texel * 3 + 2] = uchar(metallic * 255.0f + 0.5f);
}
