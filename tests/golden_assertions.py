"""Compact numerical summaries and tolerant golden comparisons for MLX fixtures."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


def summarize_array(value: Any) -> dict[str, Any]:
    """Summarize an array using stable structure and distribution statistics."""

    if isinstance(value, mx.array):
        mx.eval(value)
        dtype = str(value.dtype).removeprefix("mlx.core.")
        array = np.asarray(value.astype(mx.float32) if dtype == "bfloat16" else value)
    else:
        array = np.asarray(value)
        dtype = str(array.dtype)

    summary: dict[str, Any] = {"shape": list(array.shape), "dtype": dtype}
    flat = array.reshape(-1)
    if np.issubdtype(array.dtype, np.floating):
        finite = np.asarray(flat, dtype=np.float64)
        if not np.all(np.isfinite(finite)):
            raise ValueError("golden fixture arrays must contain only finite values")
        summary["statistics"] = {
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "l2": float(np.linalg.norm(finite)),
        }
        return summary

    canonical = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    summary["sha256"] = hashlib.sha256(canonical.tobytes(order="C")).hexdigest()
    return summary


def assert_golden_close(
    actual: Any,
    expected: Any,
    *,
    path: str = "golden",
    rtol: float = 5e-4,
    atol: float = 1e-4,
) -> None:
    """Compare nested golden data exactly except for floating-point leaves."""

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise AssertionError(f"{path}: expected a mapping, got {type(actual).__name__}")
        if set(actual) != set(expected):
            raise AssertionError(
                f"{path}: keys differ; actual={sorted(actual)} expected={sorted(expected)}"
            )
        for key in expected:
            assert_golden_close(
                actual[key], expected[key], path=f"{path}.{key}", rtol=rtol, atol=atol
            )
        return

    if isinstance(expected, Sequence) and not isinstance(
        expected, (str, bytes, bytearray)
    ):
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes, bytearray)):
            raise AssertionError(
                f"{path}: expected a sequence, got {type(actual).__name__}"
            )
        if len(actual) != len(expected):
            raise AssertionError(
                f"{path}: length differs; actual={len(actual)} expected={len(expected)}"
            )
        for index, expected_value in enumerate(expected):
            assert_golden_close(
                actual[index], expected_value, path=f"{path}[{index}]", rtol=rtol, atol=atol
            )
        return

    if isinstance(expected, float):
        if not np.isclose(float(actual), expected, rtol=rtol, atol=atol):
            raise AssertionError(f"{path}: actual={actual!r} expected={expected!r}")
        return

    if actual != expected:
        raise AssertionError(f"{path}: actual={actual!r} expected={expected!r}")


def summarize_glb(path: Path) -> dict[str, Any]:
    """Read a GLB and summarize stable structural fields without third-party parsers."""

    payload = path.read_bytes()
    if len(payload) < 20:
        raise ValueError("GLB payload is too short")
    magic, version, declared_length = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(payload):
        raise ValueError("invalid GLB header")
    json_length, json_type = struct.unpack_from("<I4s", payload, 12)
    if json_type != b"JSON":
        raise ValueError("GLB first chunk is not JSON")
    document = json.loads(
        payload[20 : 20 + json_length].decode("utf-8").rstrip(" \x00")
    )
    accessors = document.get("accessors", [])
    primitives = []
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            position_accessor = primitive.get("attributes", {}).get("POSITION")
            index_accessor = primitive.get("indices")
            primitives.append(
                {
                    "positions": (
                        accessors[position_accessor]["count"]
                        if position_accessor is not None
                        else 0
                    ),
                    "triangles": (
                        accessors[index_accessor]["count"] // 3
                        if index_accessor is not None
                        else 0
                    ),
                    "has_normal": "NORMAL" in primitive.get("attributes", {}),
                    "has_texcoord_0": "TEXCOORD_0" in primitive.get("attributes", {}),
                    "material": primitive.get("material"),
                }
            )
    return {
        "meshes": len(document.get("meshes", [])),
        "primitives": primitives,
        "materials": len(document.get("materials", [])),
        "textures": len(document.get("textures", [])),
        "images": len(document.get("images", [])),
    }


__all__ = ["assert_golden_close", "summarize_array", "summarize_glb"]
