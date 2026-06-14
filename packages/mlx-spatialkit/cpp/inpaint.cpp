#include "inpaint.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

#include "mesh_common.hpp"

namespace nb = nanobind;

namespace mlx_spatialkit {
namespace {

// Fast-marching pixel state.  KNOWN: value is final (outside the hole or
// already marched through).  BAND: on the advancing front (value computed,
// awaiting its turn to become KNOWN).  INSIDE: still to be inpainted.
enum Flag : uint8_t { KNOWN = 0, BAND = 1, INSIDE = 2 };

constexpr float kLargeT = 1.0e6f;

struct HeapNode {
  float t;
  int y;
  int x;
};

// Min-heap on T with a named (T, y, x) tie-break.  std::priority_queue is a
// max-heap, so the comparator reports "lower priority" (further from the top)
// for the lexicographically *larger* node — leaving the smallest at the top.
struct HeapCompare {
  bool operator()(const HeapNode &a, const HeapNode &b) const {
    if (a.t != b.t) {
      return a.t > b.t;
    }
    if (a.y != b.y) {
      return a.y > b.y;
    }
    return a.x > b.x;
  }
};

// Strided uint8 image/mask loaded into a dense row-major buffer.
struct Uint8Image {
  std::vector<uint8_t> data;
  int height = 0;
  int width = 0;
  int channels = 0;
  int ndim = 0;  // 2 for (H,W); 3 for (H,W,C)
};

Uint8Image load_uint8(nb::object array, const char *name, bool allow_channels) {
  const std::string dtype = mesh_common::dtype_name(array, name);
  if (dtype != "uint8") {
    std::ostringstream message;
    message << name << " must have dtype uint8, got " << dtype;
    throw nb::value_error(message.str().c_str());
  }
  const auto ndim = nb::cast<int64_t>(nb::getattr(array, "ndim"));
  if (ndim != 2 && !(allow_channels && ndim == 3)) {
    std::ostringstream message;
    if (allow_channels) {
      message << name << " must have rank 2 (H, W) or 3 (H, W, C), got rank " << ndim;
    } else {
      message << name << " must have rank 2 (H, W), got rank " << ndim;
    }
    throw nb::value_error(message.str().c_str());
  }

  Uint8Image image;
  image.ndim = static_cast<int>(ndim);
  image.height = static_cast<int>(mesh_common::dimension(array, name, 0));
  image.width = static_cast<int>(mesh_common::dimension(array, name, 1));
  image.channels = ndim == 3 ? static_cast<int>(mesh_common::dimension(array, name, 2)) : 1;
  if (image.height <= 0 || image.width <= 0) {
    std::ostringstream message;
    message << name << " must have positive height and width";
    throw nb::value_error(message.str().c_str());
  }
  if (allow_channels && (image.channels < 1 || image.channels > 4 || image.channels == 2)) {
    std::ostringstream message;
    message << name << " channel count must be 1, 3, or 4, got " << image.channels;
    throw nb::value_error(message.str().c_str());
  }

  mesh_common::BufferView view_holder(array.ptr(), name);
  const Py_buffer &view = view_holder.get();
  const auto *base = static_cast<const char *>(view.buf);
  const Py_ssize_t s0 = view.strides != nullptr ? view.strides[0] : 0;
  const Py_ssize_t s1 = view.strides != nullptr ? view.strides[1] : 0;
  const Py_ssize_t s2 = (image.ndim == 3 && view.strides != nullptr) ? view.strides[2] : 0;

  image.data.resize(static_cast<size_t>(image.height) * image.width * image.channels);
  size_t out = 0;
  for (int y = 0; y < image.height; ++y) {
    for (int x = 0; x < image.width; ++x) {
      for (int c = 0; c < image.channels; ++c) {
        const char *ptr = base + y * s0 + x * s1 + (image.ndim == 3 ? c * s2 : 0);
        image.data[out++] = static_cast<uint8_t>(*reinterpret_cast<const unsigned char *>(ptr));
      }
    }
  }
  return image;
}

class TeleaInpainter {
 public:
  TeleaInpainter(Uint8Image image, const Uint8Image &mask, int radius)
      : img_(std::move(image)),
        radius_(radius),
        h_(img_.height),
        w_(img_.width),
        c_(img_.channels) {
    flag_.assign(static_cast<size_t>(h_) * w_, KNOWN);
    t_.assign(static_cast<size_t>(h_) * w_, 0.0f);
    for (int y = 0; y < h_; ++y) {
      for (int x = 0; x < w_; ++x) {
        if (mask.data[idx(y, x)] != 0) {
          flag_[idx(y, x)] = INSIDE;
          t_[idx(y, x)] = kLargeT;
        }
      }
    }
  }

  void run() {
    std::priority_queue<HeapNode, std::vector<HeapNode>, HeapCompare> heap;
    // Initial band: KNOWN pixels 8-adjacent to the hole (dilate(mask) - mask),
    // matching OpenCV's 3x3 dilation seed, with T == 0.
    for (int y = 0; y < h_; ++y) {
      for (int x = 0; x < w_; ++x) {
        if (flag_[idx(y, x)] != KNOWN) {
          continue;
        }
        if (touches_inside(y, x)) {
          flag_[idx(y, x)] = BAND;
          t_[idx(y, x)] = 0.0f;
          heap.push(HeapNode{0.0f, y, x});
        }
      }
    }

    static const int dy[4] = {-1, 1, 0, 0};
    static const int dx[4] = {0, 0, -1, 1};
    while (!heap.empty()) {
      const HeapNode node = heap.top();
      heap.pop();
      const int i = node.y;
      const int j = node.x;
      if (flag_[idx(i, j)] == KNOWN) {
        continue;  // already finalized via an earlier (lower) key
      }
      flag_[idx(i, j)] = KNOWN;
      for (int d = 0; d < 4; ++d) {
        const int ii = i + dy[d];
        const int jj = j + dx[d];
        if (ii < 0 || ii >= h_ || jj < 0 || jj >= w_) {
          continue;
        }
        if (flag_[idx(ii, jj)] == KNOWN) {
          continue;
        }
        const float candidate = std::min(
            std::min(solve(ii - 1, jj, ii, jj - 1), solve(ii + 1, jj, ii, jj - 1)),
            std::min(solve(ii - 1, jj, ii, jj + 1), solve(ii + 1, jj, ii, jj + 1)));
        t_[idx(ii, jj)] = candidate;
        if (flag_[idx(ii, jj)] == INSIDE) {
          flag_[idx(ii, jj)] = BAND;
          inpaint_pixel(ii, jj);
          heap.push(HeapNode{candidate, ii, jj});
        }
      }
    }
  }

  Uint8Image &image() { return img_; }

 private:
  size_t idx(int y, int x) const { return static_cast<size_t>(y) * w_ + x; }

  bool in_bounds(int y, int x) const { return y >= 0 && y < h_ && x >= 0 && x < w_; }

  bool touches_inside(int y, int x) const {
    for (int dyi = -1; dyi <= 1; ++dyi) {
      for (int dxi = -1; dxi <= 1; ++dxi) {
        if (dyi == 0 && dxi == 0) {
          continue;
        }
        const int ny = y + dyi;
        const int nx = x + dxi;
        if (in_bounds(ny, nx) && flag_[idx(ny, nx)] == INSIDE) {
          return true;
        }
      }
    }
    return false;
  }

  // Eikonal |grad T| = 1 update from two orthogonal KNOWN neighbors; matches
  // OpenCV's FastMarching_solve.  Out-of-bounds or non-KNOWN neighbors do not
  // contribute.
  float solve(int y1, int x1, int y2, int x2) const {
    float sol = kLargeT;
    const bool k1 = in_bounds(y1, x1) && flag_[idx(y1, x1)] == KNOWN;
    const bool k2 = in_bounds(y2, x2) && flag_[idx(y2, x2)] == KNOWN;
    if (k1) {
      const float t1 = t_[idx(y1, x1)];
      if (k2) {
        const float t2 = t_[idx(y2, x2)];
        const float d = 2.0f - (t1 - t2) * (t1 - t2);
        if (d > 0.0f) {
          const float r = std::sqrt(d);
          float s = (t1 + t2 - r) * 0.5f;
          if (s >= t1 && s >= t2) {
            sol = s;
          } else {
            s += r;
            if (s >= t1 && s >= t2) {
              sol = s;
            }
          }
        }
      } else {
        sol = 1.0f + t1;
      }
    } else if (k2) {
      sol = 1.0f + t_[idx(y2, x2)];
    }
    return sol;
  }

  // Whether (y,x) is in bounds and not still-unknown (KNOWN or BAND): such
  // pixels carry usable values and may enter gradient/weight computations.
  bool usable(int y, int x) const {
    return in_bounds(y, x) && flag_[idx(y, x)] != INSIDE;
  }

  // Gradient of the marching field T at (y,x) using only non-INSIDE neighbors
  // (central where both sides are available, one-sided otherwise, 0 if neither).
  void grad_t(int y, int x, float &gx, float &gy) const {
    const bool xp = usable(y, x + 1);
    const bool xm = usable(y, x - 1);
    if (xp && xm) {
      gx = (t_[idx(y, x + 1)] - t_[idx(y, x - 1)]) * 0.5f;
    } else if (xp) {
      gx = t_[idx(y, x + 1)] - t_[idx(y, x)];
    } else if (xm) {
      gx = t_[idx(y, x)] - t_[idx(y, x - 1)];
    } else {
      gx = 0.0f;
    }
    const bool yp = usable(y + 1, x);
    const bool ym = usable(y - 1, x);
    if (yp && ym) {
      gy = (t_[idx(y + 1, x)] - t_[idx(y - 1, x)]) * 0.5f;
    } else if (yp) {
      gy = t_[idx(y + 1, x)] - t_[idx(y, x)];
    } else if (ym) {
      gy = t_[idx(y, x)] - t_[idx(y - 1, x)];
    } else {
      gy = 0.0f;
    }
  }

  // Gradient of image channel c at (y,x), same non-INSIDE neighbor rule.
  void grad_channel(int y, int x, int c, float &gx, float &gy) const {
    const bool xp = usable(y, x + 1);
    const bool xm = usable(y, x - 1);
    if (xp && xm) {
      gx = (static_cast<float>(pixel(y, x + 1, c)) - static_cast<float>(pixel(y, x - 1, c))) * 0.5f;
    } else if (xp) {
      gx = static_cast<float>(pixel(y, x + 1, c)) - static_cast<float>(pixel(y, x, c));
    } else if (xm) {
      gx = static_cast<float>(pixel(y, x, c)) - static_cast<float>(pixel(y, x - 1, c));
    } else {
      gx = 0.0f;
    }
    const bool yp = usable(y + 1, x);
    const bool ym = usable(y - 1, x);
    if (yp && ym) {
      gy = (static_cast<float>(pixel(y + 1, x, c)) - static_cast<float>(pixel(y - 1, x, c))) * 0.5f;
    } else if (yp) {
      gy = static_cast<float>(pixel(y + 1, x, c)) - static_cast<float>(pixel(y, x, c));
    } else if (ym) {
      gy = static_cast<float>(pixel(y, x, c)) - static_cast<float>(pixel(y - 1, x, c));
    } else {
      gy = 0.0f;
    }
  }

  void inpaint_pixel(int i, int j) {
    // Front normal = gradient of the marching field T at (i, j).
    float gtx = 0.0f;
    float gty = 0.0f;
    grad_t(i, j, gtx, gty);

    std::vector<double> ia(static_cast<size_t>(c_), 0.0);
    double weight_sum = 0.0;
    const int r = radius_;
    const int r2 = r * r;
    for (int k = i - r; k <= i + r; ++k) {
      if (k < 0 || k >= h_) {
        continue;
      }
      for (int l = j - r; l <= j + r; ++l) {
        if (l < 0 || l >= w_) {
          continue;
        }
        if (flag_[idx(k, l)] == INSIDE) {
          continue;
        }
        const float ry = static_cast<float>(i - k);
        const float rx = static_cast<float>(j - l);
        const float len2 = rx * rx + ry * ry;
        if (len2 == 0.0f || len2 > static_cast<float>(r2)) {
          continue;
        }
        const float len = std::sqrt(len2);
        const float dst = 1.0f / (len2 * len);
        const float lev = 1.0f / (1.0f + std::fabs(t_[idx(k, l)] - t_[idx(i, j)]));
        float dir = (rx * gtx + ry * gty) / len;
        if (std::fabs(dir) <= 0.01f) {
          dir = 1.0e-6f;
        }
        const float w = std::fabs(dir * dst * lev);
        for (int c = 0; c < c_; ++c) {
          float gix = 0.0f;
          float giy = 0.0f;
          grad_channel(k, l, c, gix, giy);
          const float extrapolated = static_cast<float>(pixel(k, l, c)) + gix * rx + giy * ry;
          ia[static_cast<size_t>(c)] += static_cast<double>(w) * extrapolated;
        }
        weight_sum += static_cast<double>(w);
      }
    }
    if (weight_sum <= 0.0) {
      return;  // no usable neighbor; leave the (rare) pixel as initialized
    }
    for (int c = 0; c < c_; ++c) {
      const double value = ia[static_cast<size_t>(c)] / weight_sum;
      int rounded = static_cast<int>(std::floor(value + 0.5));  // round-half-up
      rounded = std::clamp(rounded, 0, 255);
      set_pixel(i, j, c, static_cast<uint8_t>(rounded));
    }
  }

  uint8_t pixel(int y, int x, int c) const {
    return img_.data[(static_cast<size_t>(y) * w_ + x) * c_ + c];
  }
  void set_pixel(int y, int x, int c, uint8_t value) {
    img_.data[(static_cast<size_t>(y) * w_ + x) * c_ + c] = value;
  }

  Uint8Image img_;
  int radius_;
  int h_;
  int w_;
  int c_;
  std::vector<uint8_t> flag_;
  std::vector<float> t_;
};

}  // namespace

nb::object telea_inpaint(nb::object image, nb::object mask, int radius) {
  if (radius < 1) {
    throw nb::value_error("radius must be a positive integer");
  }
  Uint8Image img = load_uint8(image, "image", /*allow_channels=*/true);
  Uint8Image msk = load_uint8(mask, "mask", /*allow_channels=*/false);
  if (msk.height != img.height || msk.width != img.width) {
    std::ostringstream message;
    message << "mask shape (" << msk.height << ", " << msk.width
            << ") must match image height/width (" << img.height << ", " << img.width << ")";
    throw nb::value_error(message.str().c_str());
  }

  const int out_ndim = img.ndim;
  const size_t rows = static_cast<size_t>(img.height);
  const size_t cols = static_cast<size_t>(img.width);
  const size_t channels = static_cast<size_t>(img.channels);

  TeleaInpainter inpainter(std::move(img), msk, radius);
  inpainter.run();
  std::vector<uint8_t> out = std::move(inpainter.image().data);

  if (out_ndim == 2) {
    return mesh_common::make_uint8_array(std::move(out), rows, cols);
  }
  return mesh_common::make_uint8_array(std::move(out), rows, cols, channels);
}

}  // namespace mlx_spatialkit
