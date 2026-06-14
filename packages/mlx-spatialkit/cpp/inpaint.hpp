#pragma once

#include <nanobind/nanobind.h>

namespace mlx_spatialkit {

// Native Telea (2004) fast-marching-method inpaint, the dependency-light
// equivalent of cv2.inpaint(..., cv2.INPAINT_TELEA).  Given a uint8 image
// (H,W) or (H,W,C) with C in {1,3,4} and a uint8 mask (H,W) whose nonzero
// entries mark pixels to inpaint, returns a uint8 image of the same shape in
// which ONLY masked pixels are overwritten; unmasked bytes are bit-identical
// to the input.  `radius` is the inpaint neighborhood radius (OpenCV
// inpaintRadius), a positive integer.
//
// The fast-marching order uses a named (T, y, x) tie-break so output is fully
// deterministic and cross-process stable; bit-exactness against OpenCV is a
// non-goal (parity is bounded per-pixel error, anchored in Slice 2).
nanobind::object telea_inpaint(
    nanobind::object image,
    nanobind::object mask,
    int radius);

}  // namespace mlx_spatialkit
