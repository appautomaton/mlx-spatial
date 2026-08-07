#!/usr/bin/env python3
"""Measure cached GLBs against the original decoded high-resolution mesh."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


_COMPONENT_DTYPES = {
    5120: np.dtype("i1"),
    5121: np.dtype("u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decoded_dir", type=Path)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="LABEL=MODEL.GLB",
        help="Cached candidate GLB; repeat for convergence comparisons.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--min-component-faces", type=int, default=32)
    parser.add_argument("--major-source-min-faces", type=int, default=10_000)
    parser.add_argument("--max-samples-per-mesh", type=int, default=200_000)
    parser.add_argument("--poll-interval-sec", type=float, default=0.25)
    parser.add_argument("--max-rss-gib", type=float, default=16.0)
    parser.add_argument("--max-swap-growth-gib", type=float, default=0.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.worker:
        return _run_watchdog(args)
    return _run_measurement(args)


def _run_measurement(args: argparse.Namespace) -> int:
    candidates = _parse_candidates(args.candidate)
    if args.min_component_faces <= 0:
        raise ValueError("--min-component-faces must be positive")
    if args.max_samples_per_mesh <= 0:
        raise ValueError("--max-samples-per-mesh must be positive")
    if args.major_source_min_faces <= 0:
        raise ValueError("--major-source-min-faces must be positive")

    from mlx_spatial.spatialkit import (
        bidirectional_surface_distance_metrics,
        clean_mesh,
        extract_flexi_dual_grid,
        point_to_mesh_distances,
        sampled_surface_to_mesh_distance_metrics,
        validate_pixal3d_shape_fields,
    )

    shape_path = args.decoded_dir / "shape_decoder_fields.npz"
    if not shape_path.is_file():
        raise ValueError(f"missing decoded shape artifact: {shape_path}")
    with np.load(shape_path) as payload:
        coordinates = np.ascontiguousarray(payload["coordinates"])
        fields = np.ascontiguousarray(payload["fields"])
        metadata = _npz_metadata(payload)
    validate_pixal3d_shape_fields(coordinates, fields)
    grid_size = _resolve_grid_size(args.grid_size, metadata)
    source_raw = extract_flexi_dual_grid(coordinates, fields, grid_size=grid_size)
    del coordinates, fields
    gc.collect()
    source, clean_stats = clean_mesh(
        source_raw.vertices,
        source_raw.faces,
        min_component_faces=args.min_component_faces,
    )
    del source_raw
    gc.collect()

    major_source, major_clean_stats = clean_mesh(
        source.vertices,
        source.faces,
        min_component_faces=args.major_source_min_faces,
    )

    voxel_size = 1.0 / float(grid_size)
    report: dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "decoded_dir": str(args.decoded_dir),
            "shape_path": str(shape_path),
            "grid_size": grid_size,
            "voxel_size": voxel_size,
            "vertices": int(source.vertices.shape[0]),
            "faces": int(source.faces.shape[0]),
            "clean_stats": clean_stats,
            "major_component_probe": {
                "min_component_faces": args.major_source_min_faces,
                "vertices": int(major_source.vertices.shape[0]),
                "faces": int(major_source.faces.shape[0]),
                "clean_stats": major_clean_stats,
            },
        },
        "settings": {
            "max_samples_per_mesh": args.max_samples_per_mesh,
            "sample_policy": "deterministic-even-index-vertices-and-face-centroids",
            "distance_backend": "native-cpu-triangle-bvh",
        },
        "candidates": {},
    }
    worst_source_points: dict[str, list[float]] = {}
    for label, path in candidates:
        candidate_vertices, candidate_faces = _load_glb_scene_mesh(path)
        metrics = bidirectional_surface_distance_metrics(
            source.vertices,
            source.faces,
            candidate_vertices,
            candidate_faces,
            max_samples_per_mesh=args.max_samples_per_mesh,
            voxel_size=voxel_size,
        )
        metrics["major_source_to_candidate"] = sampled_surface_to_mesh_distance_metrics(
            major_source.vertices,
            major_source.faces,
            candidate_vertices,
            candidate_faces,
            max_samples=args.max_samples_per_mesh,
            normalization_vertices=source.vertices,
            voxel_size=voxel_size,
        )
        report["candidates"][label] = {
            "path": str(path),
            "vertices": int(candidate_vertices.shape[0]),
            "faces": int(candidate_faces.shape[0]),
            "metrics": metrics,
        }
        symmetric = metrics["symmetric"]
        worst_source_points[label] = list(metrics["source_to_candidate"]["max_query_point"])
        print(
            json.dumps(
                {
                    "label": label,
                    "sampled_chamfer_l1_voxels": symmetric["sampled_chamfer_l1_voxels"],
                    "sampled_p95_max_voxels": symmetric["sampled_p95_max_voxels"],
                    "sampled_p99_max_voxels": symmetric["sampled_p99_max_voxels"],
                    "sampled_hausdorff_voxels": symmetric["sampled_hausdorff_voxels"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del candidate_vertices, candidate_faces
        gc.collect()

    labels = [label for label, _ in candidates]
    major_queries = np.ascontiguousarray(
        np.asarray([worst_source_points[label] for label in labels], dtype=np.float32)
    )
    distances_to_major, major_query_stats = point_to_mesh_distances(
        major_queries,
        major_source.vertices,
        major_source.faces,
    )
    major_tolerance = max(voxel_size * 1.0e-3, 1.0e-7)
    report["source"]["major_component_probe"].update(
        {
            "on_surface_tolerance": major_tolerance,
            "native": major_query_stats,
        }
    )
    for index, label in enumerate(labels):
        distance = float(distances_to_major[index])
        report["candidates"][label]["metrics"]["source_to_candidate"].update(
            {
                "max_query_distance_to_source_major_surface": distance,
                "max_query_on_source_major_surface": distance <= major_tolerance,
            }
        )
    del major_source, major_queries, distances_to_major
    gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    return 0


def _run_watchdog(args: argparse.Namespace) -> int:
    if args.poll_interval_sec <= 0:
        raise ValueError("--poll-interval-sec must be positive")
    if args.max_rss_gib <= 0:
        raise ValueError("--max-rss-gib must be positive")
    if args.max_swap_growth_gib < 0:
        raise ValueError("--max-swap-growth-gib must be non-negative")

    from export_cached_ovoxel import (
        _apple_gpu_usage,
        _cpu_and_process_usage,
        _summarize_values,
        _swap_usage_bytes,
        _terminate,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(args.output.suffix + ".log")
    samples_path = args.output.with_suffix(args.output.suffix + ".resource-samples.jsonl")
    summary_path = args.output.with_suffix(args.output.suffix + ".watchdog.json")
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    baseline_swap = _swap_usage_bytes().get("swap_used_bytes")
    max_rss_bytes = int(args.max_rss_gib * 1024**3)
    max_swap_growth_bytes = int(args.max_swap_growth_gib * 1024**3)
    logical_cpu_count = os.cpu_count() or 1
    started = time.monotonic()
    peak_rss = 0
    peak_swap_growth = 0
    abort_reason: str | None = None
    previous_cpu_time: float | None = None
    previous_elapsed: float | None = None
    utilization_values: dict[str, list[float]] = {
        "process_cpu_percent": [],
        "gpu_device_utilization_percent": [],
    }

    with log_path.open("w", encoding="utf-8") as log, samples_path.open("w", encoding="utf-8") as samples:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            elapsed = time.monotonic() - started
            process_usage = _cpu_and_process_usage(process.pid, logical_cpu_count)
            cpu_time = process_usage.get("process_cpu_time_sec")
            process_cpu_percent: float | None = None
            if cpu_time is not None and previous_cpu_time is not None and previous_elapsed is not None:
                elapsed_delta = elapsed - previous_elapsed
                if elapsed_delta > 0:
                    process_cpu_percent = max(0.0, (float(cpu_time) - previous_cpu_time) / elapsed_delta * 100.0)
            if cpu_time is not None:
                previous_cpu_time = float(cpu_time)
                previous_elapsed = elapsed
            gpu_usage = _apple_gpu_usage()
            swap_usage = _swap_usage_bytes()
            swap_used = swap_usage.get("swap_used_bytes")
            swap_growth = (
                max(0, int(swap_used) - int(baseline_swap))
                if swap_used is not None and baseline_swap is not None
                else None
            )
            sample = {
                "elapsed_sec": elapsed,
                "pid": process.pid,
                **process_usage,
                "process_cpu_percent": process_cpu_percent,
                **gpu_usage,
                **swap_usage,
                "swap_growth_bytes": swap_growth,
            }
            samples.write(json.dumps(sample, sort_keys=True) + "\n")
            samples.flush()
            rss = sample.get("rss_bytes")
            if rss is not None:
                peak_rss = max(peak_rss, int(rss))
                if int(rss) > max_rss_bytes:
                    abort_reason = f"process RSS {int(rss)} exceeded limit {max_rss_bytes}"
            if swap_growth is not None:
                peak_swap_growth = max(peak_swap_growth, int(swap_growth))
                if int(swap_growth) > max_swap_growth_bytes:
                    abort_reason = f"system swap growth {int(swap_growth)} exceeded limit {max_swap_growth_bytes}"
            if process_cpu_percent is not None:
                utilization_values["process_cpu_percent"].append(process_cpu_percent)
            gpu_device = gpu_usage.get("gpu_device_utilization_percent")
            if gpu_device is not None:
                utilization_values["gpu_device_utilization_percent"].append(float(gpu_device))
            if abort_reason is not None:
                _terminate(process, 10.0)
                break
            time.sleep(args.poll_interval_sec)
        returncode = process.wait()

    summary = {
        "status": "aborted" if abort_reason is not None else ("completed" if returncode == 0 else "failed"),
        "returncode": returncode,
        "abort_reason": abort_reason,
        "elapsed_sec": time.monotonic() - started,
        "peak_rss_bytes": peak_rss,
        "baseline_swap_used_bytes": baseline_swap,
        "peak_swap_growth_bytes": peak_swap_growth,
        "logical_cpu_count": logical_cpu_count,
        "utilization": {
            key: _summarize_values(values) for key, values in utilization_values.items()
        },
        "limits": {
            "max_rss_bytes": max_rss_bytes,
            "max_swap_growth_bytes": max_swap_growth_bytes,
        },
        "samples_path": str(samples_path),
        "log_path": str(log_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if log_path.exists():
        print(log_path.read_text(encoding="utf-8"), end="")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 1 if abort_reason is not None else returncode


def _parse_candidates(values: list[str]) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"--candidate must be LABEL=MODEL.GLB, got {value!r}")
        if label in labels:
            raise ValueError(f"duplicate candidate label: {label}")
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"candidate GLB does not exist: {path}")
        labels.add(label)
        candidates.append((label, path))
    return candidates


def _npz_metadata(payload: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata_json" not in payload.files:
        return {}
    raw = payload["metadata_json"]
    if raw.shape != ():
        raise ValueError("shape metadata_json must be a scalar")
    decoded = json.loads(str(raw.item()))
    if not isinstance(decoded, dict):
        raise ValueError("shape metadata_json must decode to an object")
    return decoded


def _resolve_grid_size(explicit: int | None, metadata: dict[str, Any]) -> int:
    candidates = (explicit, metadata.get("actual_hr_resolution"), metadata.get("decode_resolution"), 1024)
    for value in candidates:
        if value is not None:
            resolved = int(value)
            if resolved <= 0:
                raise ValueError("grid size must be positive")
            return resolved
    raise AssertionError("unreachable")


def _load_glb_scene_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from mlx_spatial.spatialkit.glb_compare import parse_glb

    payload = parse_glb(path.read_bytes())
    document = payload.document
    instances = _scene_mesh_instances(document)
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    for mesh_index, transform in instances:
        meshes = document.get("meshes", [])
        if mesh_index < 0 or mesh_index >= len(meshes):
            raise ValueError(f"GLB node references invalid mesh {mesh_index}: {path}")
        for primitive in meshes[mesh_index].get("primitives", []):
            if int(primitive.get("mode", 4)) != 4:
                raise ValueError(f"only GLB TRIANGLES primitives are supported: {path}")
            attributes = primitive.get("attributes", {})
            if "POSITION" not in attributes:
                raise ValueError(f"GLB primitive is missing POSITION: {path}")
            positions = _read_accessor(payload, int(attributes["POSITION"]))
            if positions.shape[1:] != (3,) or positions.dtype != np.dtype("<f4"):
                raise ValueError(f"GLB POSITION must be float32 VEC3: {path}")
            positions = _transform_positions(positions, transform)
            if "indices" in primitive:
                primitive_indices = _read_accessor(payload, int(primitive["indices"]))
                if primitive_indices.shape[1:] != (1,) or primitive_indices.dtype.kind not in "ui":
                    raise ValueError(f"GLB indices must be an integer SCALAR accessor: {path}")
                indices = primitive_indices[:, 0].astype(np.int64, copy=False)
            else:
                indices = np.arange(positions.shape[0], dtype=np.int64)
            if indices.size % 3 != 0:
                raise ValueError(f"GLB triangle index count must be divisible by three: {path}")
            primitive_faces = indices.reshape((-1, 3))
            if primitive_faces.size and (
                int(primitive_faces.min()) < 0 or int(primitive_faces.max()) >= positions.shape[0]
            ):
                raise ValueError(f"GLB indices are outside POSITION bounds: {path}")
            vertex_offset = sum(item.shape[0] for item in vertices)
            vertices.append(np.ascontiguousarray(positions, dtype=np.float32))
            faces.append(np.ascontiguousarray(primitive_faces + vertex_offset, dtype=np.int64))
    if not vertices or not faces:
        raise ValueError(f"GLB scene contains no triangle meshes: {path}")
    return (
        np.ascontiguousarray(np.concatenate(vertices, axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(faces, axis=0), dtype=np.int64),
    )


def _read_accessor(payload: Any, accessor_index: int) -> np.ndarray:
    document = payload.document
    accessors = document.get("accessors", [])
    if accessor_index < 0 or accessor_index >= len(accessors):
        raise ValueError(f"GLB accessor index out of range: {accessor_index}")
    accessor = accessors[accessor_index]
    if "sparse" in accessor:
        raise ValueError("sparse GLB accessors are not supported")
    view_index = accessor.get("bufferView")
    views = document.get("bufferViews", [])
    if not isinstance(view_index, int) or view_index < 0 or view_index >= len(views):
        raise ValueError("GLB accessor bufferView is invalid")
    view = views[view_index]
    if int(view.get("buffer", 0)) != 0:
        raise ValueError("only the embedded GLB BIN buffer is supported")
    component_type = int(accessor["componentType"])
    dtype = _COMPONENT_DTYPES.get(component_type)
    component_count = _TYPE_COMPONENTS.get(str(accessor["type"]))
    if dtype is None or component_count is None:
        raise ValueError("unsupported GLB accessor component or type")
    count = int(accessor["count"])
    if count < 0:
        raise ValueError("GLB accessor count must be non-negative")
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    row_bytes = dtype.itemsize * component_count
    stride = int(view.get("byteStride", row_bytes))
    if stride < row_bytes:
        raise ValueError("GLB accessor byteStride is smaller than one element")
    required = offset if count == 0 else offset + (count - 1) * stride + row_bytes
    if offset < 0 or required > len(payload.bin_blob):
        raise ValueError("GLB accessor extends past the BIN chunk")
    array = np.ndarray(
        shape=(count, component_count),
        dtype=dtype,
        buffer=payload.bin_blob,
        offset=offset,
        strides=(stride, dtype.itemsize),
    )
    return np.array(array, copy=True, order="C")


def _scene_mesh_instances(document: dict[str, Any]) -> list[tuple[int, np.ndarray]]:
    nodes = document.get("nodes", [])
    scenes = document.get("scenes", [])
    if not nodes or not scenes:
        return [(index, np.eye(4, dtype=np.float64)) for index in range(len(document.get("meshes", [])))]
    scene_index = int(document.get("scene", 0))
    if scene_index < 0 or scene_index >= len(scenes):
        raise ValueError("GLB default scene index is invalid")
    instances: list[tuple[int, np.ndarray]] = []

    def visit(node_index: int, parent: np.ndarray, ancestry: frozenset[int]) -> None:
        if node_index in ancestry:
            raise ValueError("GLB node graph contains a cycle")
        if node_index < 0 or node_index >= len(nodes):
            raise ValueError("GLB scene references an invalid node")
        node = nodes[node_index]
        world = parent @ _node_transform(node)
        if "mesh" in node:
            instances.append((int(node["mesh"]), world))
        next_ancestry = ancestry | {node_index}
        for child in node.get("children", []):
            visit(int(child), world, next_ancestry)

    for root in scenes[scene_index].get("nodes", []):
        visit(int(root), np.eye(4, dtype=np.float64), frozenset())
    return instances


def _node_transform(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        values = np.asarray(node["matrix"], dtype=np.float64)
        if values.shape != (16,) or not np.all(np.isfinite(values)):
            raise ValueError("GLB node matrix must contain 16 finite values")
        return values.reshape((4, 4), order="F")
    translation = np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    rotation = np.asarray(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
    scale = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    if translation.shape != (3,) or rotation.shape != (4,) or scale.shape != (3,):
        raise ValueError("GLB node TRS shapes are invalid")
    if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(scale)):
        raise ValueError("GLB node TRS values must be finite")
    rotation_norm = float(np.linalg.norm(rotation))
    if rotation_norm <= 0:
        raise ValueError("GLB node quaternion must have non-zero length")
    x, y, z, w = rotation / rotation_norm
    rotation_matrix = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    translation_matrix = np.eye(4, dtype=np.float64)
    translation_matrix[:3, 3] = translation
    scale_matrix = np.diag([scale[0], scale[1], scale[2], 1.0])
    return translation_matrix @ rotation_matrix @ scale_matrix


def _transform_positions(positions: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        (positions.astype(np.float64), np.ones((positions.shape[0], 1), dtype=np.float64)),
        axis=1,
    )
    transformed = homogeneous @ transform.T
    w = transformed[:, 3]
    if not np.all(np.isfinite(transformed)) or np.any(np.abs(w) <= 1.0e-12):
        raise ValueError("GLB node transform produced invalid homogeneous positions")
    return np.ascontiguousarray(transformed[:, :3] / w[:, None], dtype=np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
