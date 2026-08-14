#!/usr/bin/env python3
"""Write a compact real-weight-derived Pixal3D decoder/export fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_REPOSITORY = "TencentARC/Pixal3D"
SHAPE_FILENAME = "shape_decoder_fields.npz"
TEXTURE_FILENAME = "texture_decoder_pbr.npz"
MANIFEST_FILENAME = "golden.json"


def main() -> int:
    args = _parser().parse_args()
    # Keep --help and argument validation usable without a Metal device.
    from mlx_spatial.spatialkit import export_decoded_ovoxel_glb, inspect_glb

    source_dir = args.source_decoded_dir
    output_dir = args.output_dir
    trace = json.loads(args.source_trace.read_text(encoding="utf-8"))

    with np.load(source_dir / SHAPE_FILENAME, allow_pickle=False) as payload:
        source_coordinates = np.asarray(payload["coordinates"])
        source_fields = np.asarray(payload["fields"])
    with np.load(source_dir / TEXTURE_FILENAME, allow_pickle=False) as payload:
        texture_coordinates = np.asarray(payload["coordinates"])
        source_attributes = np.asarray(payload["attributes"])
        source_grid_size = int(payload["decode_resolution"].item())

    if not np.array_equal(source_coordinates, texture_coordinates):
        raise ValueError("source shape and texture coordinates differ")

    origin = np.asarray(args.origin, dtype=np.int32)
    local_grid_size = int(args.local_grid_size)
    spatial = source_coordinates[:, 1:]
    mask = np.all((spatial >= origin) & (spatial < origin + local_grid_size), axis=1)
    if not np.any(mask):
        raise ValueError("selected local patch is empty")

    coordinates = np.ascontiguousarray(source_coordinates[mask])
    coordinates[:, 1:] -= origin
    fields = np.ascontiguousarray(source_fields[mask])
    attributes = np.ascontiguousarray(source_attributes[mask])
    metadata = {
        "fixture_kind": "real-weight-derived-decoder-patch",
        "local_grid_size": local_grid_size,
        "local_origin": origin.tolist(),
        "model_family": "pixal3d",
        "source_coordinate_system": "gltf-y-up",
        "source_grid_size": source_grid_size,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": args.source_revision,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    shape_path = output_dir / SHAPE_FILENAME
    texture_path = output_dir / TEXTURE_FILENAME
    metadata_json = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(
        shape_path,
        coordinates=coordinates,
        fields=fields,
        metadata_json=metadata_json,
    )
    np.savez_compressed(
        texture_path,
        coordinates=coordinates,
        attributes=attributes,
        spatial_shape=np.asarray([local_grid_size] * 3, dtype=np.int32),
        batch_size=np.asarray(1, dtype=np.int32),
        decode_resolution=np.asarray(local_grid_size, dtype=np.int32),
        voxel_size=np.asarray(1.0 / local_grid_size, dtype=np.float32),
        metadata_json=metadata_json,
    )

    export_settings = {
        "grid_size": local_grid_size,
        "quality_preset": "reference-target",
        "remesh": True,
        "remesh_resolution": local_grid_size,
        "simplify_backend": "mlx-qem",
        "target_faces": int(args.target_faces),
        "texture_postprocess": "telea",
        "texture_size": int(args.texture_size),
        "uv_backend": "xatlas-equivalent-native",
    }
    with tempfile.TemporaryDirectory(prefix="pixal3d-derived-golden.") as directory:
        result = export_decoded_ovoxel_glb(
            output_dir,
            Path(directory) / "model.glb",
            **export_settings,
        )
        glb = inspect_glb(result.glb.path)

    source_input = Path(str(trace.get("image_path", "")))
    manifest = {
        "schema_version": 1,
        "fixture_kind": "real-weight-derived-decoder-patch",
        "scope": {
            "covered": "decoded O-Voxel contract through native textured GLB export",
            "not_covered": (
                "Pixal3D checkpoint loading, conditioning, flow sampling, "
                "and decoder execution"
            ),
        },
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": args.source_revision,
            "precision": "unquantized mixed BF16 flow and FP16 decoder checkpoints",
            "quantization": "none",
            "pipeline_type": trace.get("pipeline_type"),
            "seed": trace.get("seed"),
            "grid_size": source_grid_size,
            "input_sha256": _sha256(source_input) if source_input.is_file() else None,
            "trace_sha256": _sha256(args.source_trace),
            "completed_stages": trace.get("completed_stages", []),
        },
        "selection": {
            "origin": origin.tolist(),
            "grid_size": local_grid_size,
            "source_token_count": int(source_coordinates.shape[0]),
            "selected_token_count": int(coordinates.shape[0]),
        },
        "files": {
            SHAPE_FILENAME: {
                "bytes": shape_path.stat().st_size,
                "sha256": _sha256(shape_path),
            },
            TEXTURE_FILENAME: {
                "bytes": texture_path.stat().st_size,
                "sha256": _sha256(texture_path),
            },
        },
        "arrays": {
            "coordinates": _array_summary(coordinates),
            "fields": _array_summary(fields),
            "attributes": _array_summary(attributes),
        },
        "expected_export": {
            "settings": export_settings,
            "source_vertices": int(
                result.diagnostics["stages"]["extract_mesh"]["source_vertices"]
            ),
            "source_faces": int(
                result.diagnostics["stages"]["extract_mesh"]["source_faces"]
            ),
            "final_faces": int(
                result.diagnostics["stages"]["simplify_mesh"]["stats"]["final_faces"]
            ),
            "glb": {
                "meshes": int(glb["mesh_count"]),
                "primitives": int(glb["primitive_count"]),
                "materials": int(glb["material_count"]),
                "textures": int(glb["texture_count"]),
                "images": int(glb["image_count"]),
                "vertices": int(glb["total_vertices"]),
                "faces": int(glb["total_faces"]),
            },
        },
    }
    (output_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    summary: dict[str, Any] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if np.issubdtype(array.dtype, np.integer):
        summary["sha256"] = hashlib.sha256(
            np.ascontiguousarray(array).tobytes()
        ).hexdigest()
    else:
        values = np.asarray(array, dtype=np.float64)
        summary["statistics"] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "l2": float(np.linalg.norm(values)),
        }
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_decoded_dir", type=Path)
    parser.add_argument("source_trace", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--origin",
        type=int,
        nargs=3,
        default=(224, 496, 688),
        metavar=("Z", "Y", "X"),
    )
    parser.add_argument("--local-grid-size", type=int, default=16)
    parser.add_argument("--target-faces", type=int, default=512)
    parser.add_argument("--texture-size", type=int, default=16)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
