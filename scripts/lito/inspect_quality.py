#!/usr/bin/env python3
"""Inspect LiTo Gaussian PLY quality signals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData


_CHECKPOINT_COMMENT = "mlx-spatial LiTo checkpoint-backed 3DGS export"
_SMOKE_COMMENT = "mlx-spatial LiTo source-contract smoke 3DGS export"


def inspect_lito_ply(path: str | Path, *, baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return schema, distribution, and obvious-failure stats for a LiTo PLY."""

    ply_path = Path(path)
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]
    data = vertex.data
    properties = list(data.dtype.names or ())
    comments = [str(comment) for comment in getattr(ply, "comments", [])]
    arrays = {name: np.asarray(data[name], dtype=np.float64) for name in properties}

    report: dict[str, Any] = {
        "path": str(ply_path),
        "vertex_count": int(vertex.count),
        "property_count": len(properties),
        "properties": properties,
        "comments": comments,
        "checkpoint_backed": any(_CHECKPOINT_COMMENT in comment for comment in comments),
        "source_contract_smoke": any(_SMOKE_COMMENT in comment for comment in comments),
    }
    report["xyz"] = _xyz_stats(arrays)
    report["opacity_logit"] = _field_stats(_stack_fields(arrays, ["opacity"]))
    report["opacity_probability"] = _field_stats(_sigmoid(_stack_fields(arrays, ["opacity"])))
    report["scale_log"] = _field_stats(_stack_fields(arrays, [f"scale_{index}" for index in range(3)]))
    report["scale_exp"] = _field_stats(np.exp(_stack_fields(arrays, [f"scale_{index}" for index in range(3)])))
    quaternion = _stack_fields(arrays, [f"rot_{index}" for index in range(4)])
    report["quaternion_norm"] = _field_stats(np.linalg.norm(quaternion, axis=1) if quaternion.size else np.array([]))
    report["sh"] = _sh_stats(arrays)
    report["flags"] = _quality_flags(report)
    report["failure_classification"] = _classify(report["flags"])
    if baseline is not None:
        report["comparison"] = _compare_reports(baseline, report)
    return report


def _xyz_stats(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    xyz = _stack_fields(arrays, ["x", "y", "z"])
    stats = _field_stats(xyz)
    if xyz.size == 0:
        stats.update({"bbox_min": [], "bbox_max": [], "bbox_span": []})
        return stats
    stats["bbox_min"] = _float_list(np.nanmin(xyz, axis=0))
    stats["bbox_max"] = _float_list(np.nanmax(xyz, axis=0))
    stats["bbox_span"] = _float_list(np.nanmax(xyz, axis=0) - np.nanmin(xyz, axis=0))
    return stats


def _sh_stats(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    f_dc = _stack_prefix(arrays, "f_dc_")
    f_rest = _stack_prefix(arrays, "f_rest_")
    rgb = _stack_fields(arrays, ["red", "green", "blue"])
    return {
        "f_dc": _field_stats(f_dc),
        "f_rest": _field_stats(f_rest),
        "rgb": _field_stats(rgb),
        "f_dc_count": int(f_dc.shape[1]) if f_dc.ndim == 2 else 0,
        "f_rest_count": int(f_rest.shape[1]) if f_rest.ndim == 2 else 0,
    }


def _field_stats(values: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    stats: dict[str, Any] = {
        "count": int(flat.size),
        "finite_count": int(finite.size),
        "nan_count": int(np.isnan(flat).sum()),
        "inf_count": int(np.isinf(flat).sum()),
    }
    if finite.size == 0:
        stats.update({"min": None, "max": None, "mean": None, "median": None, "p01": None, "p99": None})
        return stats
    stats.update(
        {
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "p01": float(np.percentile(finite, 1)),
            "p99": float(np.percentile(finite, 99)),
        }
    )
    return stats


def _quality_flags(report: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if not report["checkpoint_backed"]:
        flags.append("not_checkpoint_backed")
    if report["source_contract_smoke"]:
        flags.append("source_contract_smoke")
    if report["vertex_count"] <= 0:
        flags.append("empty_vertices")
    if not _all_finite(report["xyz"]):
        flags.append("xyz_non_finite")
    span = report["xyz"].get("bbox_span", [])
    if span:
        max_span = max(float(value) for value in span)
        if max_span < 1e-5:
            flags.append("bbox_collapsed")
        if max_span > 20.0:
            flags.append("bbox_extreme")
    if not _all_finite(report["opacity_logit"]):
        flags.append("opacity_non_finite")
    if not _all_finite(report["scale_log"]):
        flags.append("scale_non_finite")
    scale_exp = report["scale_exp"]
    if scale_exp["finite_count"]:
        median = float(scale_exp["median"])
        if median < 1e-6:
            flags.append("scale_tiny")
        if median > 1.0:
            flags.append("scale_large")
    q_norm = report["quaternion_norm"]
    if q_norm["finite_count"]:
        max_norm_error = max(abs(float(q_norm["min"]) - 1.0), abs(float(q_norm["max"]) - 1.0))
        if max_norm_error > 1e-2:
            flags.append("quaternion_norm_drift")
    elif report["vertex_count"] > 0:
        flags.append("quaternion_missing")
    return flags


def _classify(flags: list[str]) -> str:
    severe = {
        "not_checkpoint_backed",
        "source_contract_smoke",
        "empty_vertices",
        "xyz_non_finite",
        "bbox_collapsed",
        "bbox_extreme",
        "opacity_non_finite",
        "scale_non_finite",
        "quaternion_missing",
    }
    if any(flag in severe for flag in flags):
        return "invalid_or_schema_only"
    if flags:
        return "inspect_visual_quality"
    return "stats_sane_visual_review_required"


def _compare_reports(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    return {
        "vertex_count_delta": int(current.get("vertex_count", 0)) - int(baseline.get("vertex_count", 0)),
        "bbox_span_delta": _list_delta(
            baseline.get("xyz", {}).get("bbox_span", []),
            current.get("xyz", {}).get("bbox_span", []),
        ),
        "opacity_probability_median_delta": _stat_delta(baseline, current, "opacity_probability", "median"),
        "scale_exp_median_delta": _stat_delta(baseline, current, "scale_exp", "median"),
        "quaternion_norm_median_delta": _stat_delta(baseline, current, "quaternion_norm", "median"),
        "new_flags": sorted(set(current.get("flags", [])) - set(baseline.get("flags", []))),
        "resolved_flags": sorted(set(baseline.get("flags", [])) - set(current.get("flags", []))),
    }


def _stack_fields(arrays: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    if not all(name in arrays for name in names):
        return np.array([], dtype=np.float64)
    return np.stack([arrays[name] for name in names], axis=1)


def _stack_prefix(arrays: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    names = sorted((name for name in arrays if name.startswith(prefix)), key=lambda value: int(value.rsplit("_", 1)[1]))
    return _stack_fields(arrays, names) if names else np.array([], dtype=np.float64)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def _all_finite(stats: dict[str, Any]) -> bool:
    return bool(stats["count"] == stats["finite_count"] and stats["count"] > 0)


def _stat_delta(baseline: dict[str, Any], current: dict[str, Any], section: str, field: str) -> float | None:
    left = baseline.get(section, {}).get(field)
    right = current.get(section, {}).get(field)
    if left is None or right is None:
        return None
    return float(right) - float(left)


def _list_delta(left: list[Any], right: list[Any]) -> list[float]:
    if len(left) != len(right):
        return []
    return [float(r) - float(l) for l, r in zip(left, right, strict=True)]


def _float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in values.tolist()]


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ply", type=Path)
    parser.add_argument("--json", type=Path, help="write the full report to this JSON path")
    parser.add_argument("--compare", type=Path, help="compare against a prior inspector JSON report")
    args = parser.parse_args(argv)

    report = inspect_lito_ply(args.ply, baseline=_load_json(args.compare))
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
