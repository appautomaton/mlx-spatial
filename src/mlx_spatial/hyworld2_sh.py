"""Spherical harmonics evaluation for HY-World 2.0 inference.

Gap ID: HW-15. Matches
``vendors/HY-World-2.0/hyworld2/worldrecon/hyworldmirror/models/utils/sh_utils.py``.

Provides SH evaluation up to degree 4, plus RGB↔SH conversion.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


_C0 = 0.28209479177387814
_C1 = 0.4886025119029199
_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
_C3 = [
    -0.5900436355331605,
    2.89061142604197,
    -0.4570250667930509,
    0.3731763875965469,
    -0.4570250667930509,
    1.445305713020985,
    -0.5900436355331605,
]
_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg: int, sh: mx.array, dirs: mx.array) -> mx.array:
    """Evaluate spherical harmonics up to the given degree.

    Args:
        deg: Maximum degree (0-4).
        sh: ``(..., C, (deg+1)**2)`` SH coefficients per color channel.
            C is typically 1 (grayscale) or 3 (RGB).
        dirs: ``(..., 3)`` unit direction vectors [x, y, z].

    Returns:
        ``(..., C)`` evaluated color values.
    """
    assert deg <= 4, f"SH degree must be 0-4, got {deg}"
    x = dirs[..., None, 0:1]
    y = dirs[..., None, 1:2]
    z = dirs[..., None, 2:3]
    result = _C0 * sh[..., 0:1]

    if deg >= 1:
        result = result + _C1 * (-y * sh[..., 1:2] + z * sh[..., 2:3] - x * sh[..., 3:4])
    if deg >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = result + (_C2[0] * xy * sh[..., 4:5]
                           + _C2[1] * yz * sh[..., 5:6]
                           + _C2[2] * (2.0 * zz - xx - yy) * sh[..., 6:7]
                           + _C2[3] * xz * sh[..., 7:8]
                           + _C2[4] * (xx - yy) * sh[..., 8:9])
    if deg >= 3:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = result + (_C3[0] * y * (3.0 * xx - yy) * sh[..., 9:10]
                           + _C3[1] * xy * z * sh[..., 10:11]
                           + _C3[2] * y * (4.0 * zz - xx - yy) * sh[..., 11:12]
                           + _C3[3] * z * (2.0 * zz - 3.0 * (xx + yy)) * sh[..., 12:13]
                           + _C3[4] * x * (4.0 * zz - xx - yy) * sh[..., 13:14]
                           + _C3[5] * z * (xx - yy) * sh[..., 14:15]
                           + _C3[6] * x * (xx - 3.0 * yy) * sh[..., 15:16])
    if deg >= 4:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = result + (_C4[0] * xy * (xx - yy) * sh[..., 16:17]
                           + _C4[1] * yz * (3.0 * xx - yy) * sh[..., 17:18]
                           + _C4[2] * xy * (7.0 * zz - 1.0) * sh[..., 18:19]
                           + _C4[3] * yz * (7.0 * zz - 3.0) * sh[..., 19:20]
                           + _C4[4] * (zz * (35.0 * zz - 30.0) + 3.0) * sh[..., 20:21]
                           + _C4[5] * xz * (7.0 * zz - 3.0) * sh[..., 21:22]
                           + _C4[6] * (xx - yy) * (7.0 * zz - 1.0) * sh[..., 22:23]
                           + _C4[7] * xz * (xx - 3.0 * yy) * sh[..., 23:24]
                           + _C4[8] * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy)) * sh[..., 24:25])
    return result


def eval_sh_numpy(deg: int, sh: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Evaluate spherical harmonics using numpy (for parity testing).

    Same as ``eval_sh`` but operates on numpy arrays.
    """
    result = _C0 * sh[..., 0:1]
    x, y, z = dirs[..., None, 0:1], dirs[..., None, 1:2], dirs[..., None, 2:3]

    if deg >= 1 and sh.shape[-1] >= 4:
        result = result + _C1 * (-y * sh[..., 1:2] + z * sh[..., 2:3]
                                  - x * sh[..., 3:4])
    if deg >= 2 and sh.shape[-1] >= 9:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = result + (_C2[0] * xy * sh[..., 4:5]
                           + _C2[1] * yz * sh[..., 5:6]
                           + _C2[2] * (2.0 * zz - xx - yy) * sh[..., 6:7]
                           + _C2[3] * xz * sh[..., 7:8]
                           + _C2[4] * (xx - yy) * sh[..., 8:9])
    if deg >= 3 and sh.shape[-1] >= 16:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = result + (_C3[0] * y * (3.0 * xx - yy) * sh[..., 9:10]
                           + _C3[1] * xy * z * sh[..., 10:11]
                           + _C3[2] * y * (4.0 * zz - xx - yy) * sh[..., 11:12]
                           + _C3[3] * z * (2.0 * zz - 3.0 * (xx + yy)) * sh[..., 12:13]
                           + _C3[4] * x * (4.0 * zz - xx - yy) * sh[..., 13:14]
                           + _C3[5] * z * (xx - yy) * sh[..., 14:15]
                           + _C3[6] * x * (xx - 3.0 * yy) * sh[..., 15:16])
    if deg >= 4 and sh.shape[-1] >= 25:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = result + (_C4[0] * xy * (xx - yy) * sh[..., 16:17]
                           + _C4[1] * yz * (3.0 * xx - yy) * sh[..., 17:18]
                           + _C4[2] * xy * (7.0 * zz - 1.0) * sh[..., 18:19]
                           + _C4[3] * yz * (7.0 * zz - 3.0) * sh[..., 19:20]
                           + _C4[4] * (zz * (35.0 * zz - 30.0) + 3.0) * sh[..., 20:21]
                           + _C4[5] * xz * (7.0 * zz - 3.0) * sh[..., 21:22]
                           + _C4[6] * (xx - yy) * (7.0 * zz - 1.0) * sh[..., 22:23]
                           + _C4[7] * xz * (xx - 3.0 * yy) * sh[..., 23:24]
                           + _C4[8] * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy)) * sh[..., 24:25])
    return result


def rgb_to_sh(rgb: mx.array) -> mx.array:
    """Convert RGB colors to SH degree-0 coefficients.

    ``sh = (rgb - 0.5) / C0``

    Args:
        rgb: ``(..., 3)`` RGB colors in [0, 1].

    Returns:
        ``(..., 3)`` SH coefficients.
    """
    return (rgb - 0.5) / _C0


def sh_to_rgb(sh: mx.array) -> mx.array:
    """Convert SH degree-0 coefficients back to RGB colors.

    ``rgb = sh * C0 + 0.5``

    Args:
        sh: ``(..., 3)`` SH coefficients.

    Returns:
        ``(..., 3)`` RGB colors in [0, 1] (may clip).
    """
    return sh * _C0 + 0.5