#include <metal_stdlib>

using namespace metal;

kernel void mlx_spatialkit_texture_probe(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  output[index] = input[index];
}
