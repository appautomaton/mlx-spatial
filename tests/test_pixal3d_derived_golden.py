"""Regression coverage for the compact real-weight-derived Pixal3D fixture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mlx_spatial.spatialkit import (
    export_decoded_ovoxel_glb,
    inspect_glb,
    load_decoded_ovoxel_npz,
)
from tests.golden_assertions import assert_golden_close, summarize_array


FIXTURE_ROOT = Path(__file__).parent / "data/pixal3d_derived_golden"
MANIFEST_PATH = FIXTURE_ROOT / "golden.json"


def test_pixal3d_derived_golden_manifest_matches_committed_decoder_patch():
    manifest = _manifest()

    assert manifest["fixture_kind"] == "real-weight-derived-decoder-patch"
    assert manifest["source"]["repository"] == "TencentARC/Pixal3D"
    assert manifest["source"]["quantization"] == "none"
    assert manifest["scope"]["not_covered"].startswith("Pixal3D checkpoint loading")
    for filename, expected in manifest["files"].items():
        path = FIXTURE_ROOT / filename
        assert path.stat().st_size == expected["bytes"]
        assert _sha256(path) == expected["sha256"]

    decoded = load_decoded_ovoxel_npz(
        FIXTURE_ROOT / "shape_decoder_fields.npz",
        FIXTURE_ROOT / "texture_decoder_pbr.npz",
    )
    assert decoded.shape_metadata == decoded.texture_metadata
    assert (
        decoded.shape_metadata["source_revision"]
        == manifest["source"]["revision"]
    )
    assert decoded.texture_decode_resolution == manifest["selection"]["grid_size"]
    assert_golden_close(
        {
            "coordinates": summarize_array(decoded.shape_coordinates),
            "fields": summarize_array(decoded.shape_fields),
            "attributes": summarize_array(decoded.texture_attributes),
        },
        manifest["arrays"],
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.heavy
def test_pixal3d_derived_golden_replays_native_textured_export(tmp_path):
    expected = _manifest()["expected_export"]
    result = export_decoded_ovoxel_glb(
        FIXTURE_ROOT,
        tmp_path / "model.glb",
        **expected["settings"],
    )
    glb = inspect_glb(result.glb.path)

    actual = {
        "settings": expected["settings"],
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
    }
    assert actual == expected


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
