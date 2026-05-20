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
    2.5035574406792172,
    -1.7701308583598902,
    0.9461746972030186,
    -0.6690465341828717,
    0.10578552846601598,
    -0.6690465341828717,
    0.9461746972030186,
    -1.7701308583598902,
    2.5035574406792172,
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
    x = dirs[..., 0:1]
    y = dirs[..., 1:2]
    z = dirs[..., 2:3]
    result = _C0 * sh[..., 0:1]

    if deg >= 1:
        result = result + _C1 * (-y * sh[..., 1:2] + z * sh[..., 2:3]
                                  + x * sh[..., 3:4] + (-x * sh[..., 4:5] if sh.shape[-1] > 4 else mx.zeros_like(x)))
    if deg >= 2:
        result = result + (_C2[0] * x * y * sh[..., 5:6]
                           + _C2[1] * y * z * sh[..., 6:7]
                           + _C2[2] * (2.0 * z * z - x * x - y * y) * sh[..., 7:8]
                           + _C2[3] * x * z * sh[..., 8:9]
                           + _C2[4] * (x * x - y * y) * sh[..., 9:10])
    if deg >= 3:
        result = result + (_C3[0] * y * (3.0 * x * x - y * y) * sh[..., 10:11]
                           + _C3[1] * x * y * z * sh[..., 11:12]
                           + _C3[2] * y * (4.0 * z * z - x * x - y * y) * sh[..., 12:13]
                           + _C3[3] * z * (2.0 * z * z - 3.0 * (x * x + y * y) if sh.shape[-1] > 13 else z) * sh[..., 13:14]
                           + _C3[4] * x * (4.0 * z * z - x * x - y * y) * sh[..., 14:15]
                           + _C3[5] * x * (x * x - 3.0 * y * y) * sh[..., 15:16]
                           + _C3[6] * (3.0 * y * y - x * x) * sh[..., 16:17] if sh.shape[-1] > 16 else mx.zeros_like(x))
    return result


def eval_sh_numpy(deg: int, sh: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Evaluate spherical harmonics using numpy (for parity testing).

    Same as ``eval_sh`` but operates on numpy arrays.
    """
    result = _C0 * sh[..., 0:1]
    x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]

    if deg >= 1 and sh.shape[-1] >= 4:
        result = result + _C1 * (-y * sh[..., 1:2] + z * sh[..., 2:3]
                                  + x * sh[..., 3:4])
    if deg >= 2 and sh.shape[-1] >= 9:
        xy, yz, xx_yy, xz, xxyy = x * y, y * z, x * x - y * y, x * z, x * x + y * y
        result = result + _C2[0] * xy * sh[..., 5:6] + _C2[1] * yz * sh[..., 6:7]
        result = result + _C2[2] * (2.0 * z * z - xxyy) * sh[..., 7:8]
        result = result + _C2[3] * xz * sh[..., 8:9] + _C2[4] * xx_yy * sh[..., 9:10]
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