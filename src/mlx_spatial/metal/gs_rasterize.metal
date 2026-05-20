// Body source for mlx.core.fast.metal_kernel.
//
// One thread renders one output pixel. Inputs are already projected, clipped,
// and sorted front-to-back by the Python side. This intentionally avoids float
// atomics so it works on the M1 baseline targeted by mlx-spatial; it is a
// correctness-first kernel for small/test-size renders before tile binning.

uint pixel_index = thread_position_in_grid.x;
uint total_pixels = uint(width * height);
if (pixel_index >= total_pixels) {
    return;
}

uint image_width = uint(width);
uint x = pixel_index % image_width;
uint y = pixel_index / image_width;
float px = float(x) + 0.5f;
float py = float(y) + 0.5f;

float transmittance = 1.0f;
float out_r = 0.0f;
float out_g = 0.0f;
float out_b = 0.0f;
float out_a = 0.0f;
float depth_weighted = 0.0f;

for (uint i = 0; i < uint(gaussian_count); ++i) {
    float radius = radii[i];
    if (radius <= 0.0f || transmittance <= 0.0039215689f) {
        continue;
    }

    float dx = px - centers_xy[i * 2];
    float dy = py - centers_xy[i * 2 + 1];
    float dist2 = dx * dx + dy * dy;
    if (dist2 > radius * radius) {
        continue;
    }

    float a = conics[i * 3];
    float b = conics[i * 3 + 1];
    float c = conics[i * 3 + 2];
    float mahalanobis = a * dx * dx + 2.0f * b * dx * dy + c * dy * dy;
    if (mahalanobis > mahalanobis_clip_sq) {
        continue;
    }

    float alpha = opacities[i] * metal::exp(-0.5f * mahalanobis);
    if (alpha <= min_alpha) {
        continue;
    }
    alpha = metal::min(alpha, 0.999f);

    float contribution = transmittance * alpha;
    out_r += contribution * colors[i * 3];
    out_g += contribution * colors[i * 3 + 1];
    out_b += contribution * colors[i * 3 + 2];
    out_a += contribution;
    depth_weighted += contribution * depths[i];
    transmittance *= 1.0f - alpha;
}

float inv_alpha = out_a > 1e-6f ? 1.0f / out_a : 0.0f;
rgba[pixel_index * 4] = out_r * inv_alpha;
rgba[pixel_index * 4 + 1] = out_g * inv_alpha;
rgba[pixel_index * 4 + 2] = out_b * inv_alpha;
rgba[pixel_index * 4 + 3] = out_a;
depth[pixel_index] = out_a > 1e-6f ? depth_weighted / out_a : 0.0f;
