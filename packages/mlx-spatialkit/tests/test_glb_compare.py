from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from mlx_spatialkit import compare_textured_glbs, inspect_glb, make_face_atlas_uvs, png_coverage, textured_glb_payload


def test_png_coverage_supports_standard_row_filters() -> None:
    rows = [
        bytes([0, 0, 0, 0, 10, 0, 0, 255]),
        bytes([0, 20, 0, 255, 0, 0, 0, 0]),
        bytes([0, 0, 30, 255, 0, 0, 0, 255]),
        bytes([0, 0, 0, 0, 40, 40, 0, 255]),
        bytes([50, 0, 50, 255, 0, 0, 0, 0]),
    ]
    png = _filtered_png(width=2, rows=rows, filters=[0, 1, 2, 3, 4])

    coverage = png_coverage(png)

    assert coverage.width == 2
    assert coverage.height == 5
    assert coverage.channels == 4
    assert coverage.pixel_count == 10
    assert coverage.alpha_nonzero_count == 6
    assert coverage.rgb_nonzero_count == 5
    assert coverage.alpha_coverage_ratio == pytest.approx(0.6)
    assert coverage.rgb_coverage_ratio == pytest.approx(0.5)


def test_inspect_glb_reports_mesh_counts_and_embedded_texture_coverage(tmp_path: Path) -> None:
    path = _write_fixture_glb(tmp_path)

    summary = inspect_glb(path)

    assert summary["path"] == str(path)
    assert summary["mesh_count"] == 1
    assert summary["material_count"] == 1
    assert summary["image_count"] == 2
    assert summary["total_faces"] == 2
    assert summary["total_vertices"] == 6
    assert summary["primitives"][0]["attributes"] == ["POSITION", "TEXCOORD_0"]
    base_image = summary["images"][0]
    assert base_image["name"] == "baseColorTexture"
    assert base_image["coverage"]["width"] == 4
    assert base_image["coverage"]["height"] == 4
    assert base_image["coverage"]["alpha_coverage_ratio"] == pytest.approx(0.25)
    assert base_image["coverage"]["rgb_coverage_ratio"] == pytest.approx(0.25)


def test_compare_textured_glbs_writes_report_and_preview_artifacts(tmp_path: Path) -> None:
    candidate = _write_fixture_glb(tmp_path / "candidate")
    reference = _write_fixture_glb(tmp_path / "reference")
    output_dir = tmp_path / "visual_parity"

    report = compare_textured_glbs(candidate, reference, output_dir=output_dir)

    assert report["summary"]["all_passed"] is True
    assert report["summary"]["face_count_ratio"] == pytest.approx(1.0)
    assert report["summary"]["base_color_alpha_coverage_ratio"] == pytest.approx(1.0)
    assert report["summary"]["base_color_rgb_coverage_ratio"] == pytest.approx(1.0)
    assert report["summary"]["texture_resolution_match"] is True
    assert report["deferred_parity_boundaries"] == [
        "not_xatlas_chart_parity",
        "not_1m_face_export_setting_parity",
    ]
    assert Path(report["artifacts"]["report_json"]) == output_dir / "visual_parity.json"
    assert (output_dir / "visual_parity.json").exists()
    assert (output_dir / "index.html").exists()
    assert (output_dir / "candidate_base_color.png").read_bytes().startswith(b"\x89PNG")
    assert (output_dir / "reference_base_color.png").read_bytes().startswith(b"\x89PNG")


def _write_fixture_glb(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    mesh = make_face_atlas_uvs(vertices, faces, tile_padding=0.0)
    base_color = np.zeros((4, 4, 4), dtype=np.uint8)
    base_color[:2, :2] = np.array([255, 0, 0, 255], dtype=np.uint8)
    metallic_roughness = np.zeros((4, 4, 3), dtype=np.uint8)
    payload = textured_glb_payload(
        mesh,
        base_color_rgba=base_color,
        metallic_roughness=metallic_roughness,
        mesh_name="FixtureMesh",
        material_name="FixtureMaterial",
    )
    path = tmp_path / "fixture.glb"
    path.write_bytes(payload)
    return path


def _filtered_png(*, width: int, rows: list[bytes], filters: list[int]) -> bytes:
    if len(rows) != len(filters):
        raise ValueError("rows and filters length mismatch")
    channels = 4
    height = len(rows)
    row_bytes = width * channels
    encoded = bytearray()
    prior = bytes(row_bytes)
    for row, filter_type in zip(rows, filters, strict=True):
        if len(row) != row_bytes:
            raise ValueError("row width mismatch")
        encoded.append(filter_type)
        encoded.extend(_filter_row(row, prior, channels, filter_type))
        prior = row
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(bytes(encoded)))
        + _chunk(b"IEND", b"")
    )


def _filter_row(row: bytes, prior: bytes, bpp: int, filter_type: int) -> bytes:
    filtered = bytearray()
    for idx, value in enumerate(row):
        left = row[idx - bpp] if idx >= bpp else 0
        up = prior[idx]
        up_left = prior[idx - bpp] if idx >= bpp else 0
        if filter_type == 0:
            predictor = 0
        elif filter_type == 1:
            predictor = left
        elif filter_type == 2:
            predictor = up
        elif filter_type == 3:
            predictor = (left + up) >> 1
        elif filter_type == 4:
            predictor = _paeth(left, up, up_left)
        else:
            raise ValueError(f"bad filter type {filter_type}")
        filtered.append((value - predictor) & 0xFF)
    return bytes(filtered)


def _paeth(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    left_distance = abs(estimate - left)
    up_distance = abs(estimate - up)
    up_left_distance = abs(estimate - up_left)
    if left_distance <= up_distance and left_distance <= up_left_distance:
        return left
    if up_distance <= up_left_distance:
        return up
    return up_left


def _chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(payload, crc)
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc & 0xFFFFFFFF)
