from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mlx_spatialkit import export_pixal3d_glb, metal_device_available


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.mark.heavy
def test_export_pixal3d_glb_real_decoded_fixture_writes_glb_and_diagnostics() -> None:
    if not metal_device_available():
        pytest.skip("Metal device unavailable for mlx-spatialkit real Pixal3D export")
    fixture = _repo_root() / "inputs" / "mlx-spatialkit" / "pixal3d-1024-cascade-decoded-pbr"
    if not fixture.exists():
        pytest.skip(f"real Pixal3D decoded fixture not present: {fixture}")

    output_dir = Path("/tmp") / f"mlx-spatialkit-real-pixal3d-export-{os.getpid()}"
    result = export_pixal3d_glb(
        fixture,
        output_dir,
        texture_size=1024,
        target_faces=50_000,
        min_component_faces=32,
    )

    assert result.glb.path == output_dir / "model.glb"
    assert result.diagnostics_path == output_dir / "diagnostics.json"
    assert result.glb.path.read_bytes()[:4] == b"glTF"
    assert result.glb.bytes_written == result.glb.path.stat().st_size
    assert result.glb.bytes_written > 1_000_000

    diagnostics = json.loads(result.diagnostics_path.read_text())
    assert diagnostics["result"]["ready"] is True
    assert diagnostics["settings"]["texture_size"] == 1024
    assert diagnostics["settings"]["target_faces"] == 50_000
    assert diagnostics["settings"]["grid_size"] == 1024
    assert diagnostics["contracts"]["shape"]["token_count"] == 4_150_336
    assert diagnostics["contracts"]["texture"]["token_count"] == 4_150_336
    assert diagnostics["stages"]["extract_mesh"]["source_faces"] == 8_304_022
    assert diagnostics["stages"]["clean_mesh"]["cleaned_faces"] > 8_000_000
    assert diagnostics["stages"]["simplify_mesh"]["simplified_faces"] == 50_000
    assert diagnostics["stages"]["uv"]["stats"]["backend"] == "face-atlas"
    assert diagnostics["stages"]["texture_bake"]["stats"]["backend"] == "metal-face-atlas-nearest"
    assert diagnostics["stages"]["texture_bake"]["stats"]["sampled_texel_count"] > 0
    assert diagnostics["stages"]["write_glb"]["artifact"]["mesh_name"] == "Pixal3D_TexturedMesh"
    assert "after_write_glb" in diagnostics["memory_samples"]


def test_export_pixal3d_glb_rejects_invalid_public_guards(tmp_path) -> None:
    decoded_dir = tmp_path / "decoded"
    decoded_dir.mkdir()

    with pytest.raises(ValueError, match="grid_size must be positive"):
        export_pixal3d_glb(decoded_dir, tmp_path / "out", grid_size=0)

    with pytest.raises(ValueError, match="max_texture_pixels must be positive"):
        export_pixal3d_glb(decoded_dir, tmp_path / "out", max_texture_pixels=0)
