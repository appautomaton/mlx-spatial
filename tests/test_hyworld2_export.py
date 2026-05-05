import json
import struct

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_spatial.hyworld2_export import (
    export_hyworld2_gaussian_attributes,
    export_hyworld2_cameras,
    export_hyworld2_depth,
    export_hyworld2_normals,
    export_hyworld2_points_ply,
)


def test_depth_export_writes_arrays_and_png_preview(tmp_path):
    records = export_hyworld2_depth(
        tmp_path,
        mx.array([[[[[0.0], [1.0]], [[2.0], [3.0]]]]], dtype=mx.float32),
        mx.array([[[[0.5, 0.6], [0.7, 0.8]]]], dtype=mx.float32),
    )

    names = {record["name"] for record in records}
    assert {"depth", "depth-confidence", "depth-preview-000"} <= names
    np.testing.assert_allclose(
        np.load(tmp_path / "depth" / "depth.npy"),
        np.array([[[[[0.0], [1.0]], [[2.0], [3.0]]]]], dtype=np.float32),
    )
    assert Image.open(tmp_path / "depth" / "depth_b00_f000.png").size == (2, 2)


def test_normal_export_maps_unit_vectors_to_rgb_preview(tmp_path):
    export_hyworld2_normals(
        tmp_path,
        mx.array([[[[[-1.0, 0.0, 1.0]]]]], dtype=mx.float32),
        mx.array([[[[1.0]]]], dtype=mx.float32),
    )

    preview = np.array(Image.open(tmp_path / "normal" / "normal_b00_f000.png"))
    assert preview.tolist() == [[[0, 127, 255]]]
    assert (tmp_path / "normal" / "normal.npy").stat().st_size > 0
    assert (tmp_path / "normal" / "confidence.npy").stat().st_size > 0


def test_camera_export_writes_deterministic_json(tmp_path):
    export_hyworld2_cameras(
        tmp_path,
        mx.array([[[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, np.pi / 2, np.pi / 2]]], dtype=mx.float32),
        image_size=(2, 4),
        image_paths=("a.png",),
    )

    payload = json.loads((tmp_path / "camera_params.json").read_text())
    assert payload["num_cameras"] == 1
    assert payload["image_paths"] == ["a.png"]
    np.testing.assert_allclose(np.array(payload["extrinsics"][0]["matrix"])[:3, :3], np.eye(3), atol=1e-5)
    np.testing.assert_allclose(np.array(payload["extrinsics"][0]["matrix"])[:3, 3], [-1.0, -2.0, -3.0], atol=1e-5)
    np.testing.assert_allclose(
        np.array(payload["intrinsics"][0]["matrix"]),
        np.array([[2.0, 0.0, 2.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        atol=1e-5,
    )


def test_points_ply_export_is_row_major_with_deterministic_formatting(tmp_path):
    points = mx.array(
        [[[[[1.0, 2.0, 3.0], [4.5, 5.25, 6.125]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]]],
        dtype=mx.float32,
    )
    image_tensor = mx.array(
        [[[[[0.0, 1.0], [0.5, 0.25]], [[0.0, 0.5], [1.0, 0.25]], [[1.0, 0.0], [0.0, 0.25]]]]],
        dtype=mx.float32,
    )

    records = export_hyworld2_points_ply(tmp_path, points, image_tensor)

    assert records == ({"name": "points", "path": tmp_path / "points" / "points.ply", "kind": "ply"},)
    lines = (tmp_path / "points" / "points.ply").read_text().splitlines()
    assert lines[:10] == [
        "ply",
        "format ascii 1.0",
        "element vertex 4",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    assert lines[10:] == [
        "1.000000 2.000000 3.000000 0 0 255",
        "4.500000 5.250000 6.125000 255 128 0",
        "7.000000 8.000000 9.000000 128 255 0",
        "10.000000 11.000000 12.000000 64 64 64",
    ]


def test_gaussian_export_writes_official_3dgs_ply_schema(tmp_path):
    raw = np.zeros((1, 1, 12, 1, 2), dtype=np.float32)
    raw[:, :, 3, :, :] = 1.0
    raw[:, :, 4:7, :, :] = -7.0
    raw[:, :, 7, :, :] = -2.0
    raw[:, :, 11, :, :] = 2.0
    records = export_hyworld2_gaussian_attributes(
        tmp_path,
        features=mx.zeros((1, 1, 128, 1, 2), dtype=mx.float32),
        depth=mx.ones((1, 1, 1, 2, 1), dtype=mx.float32),
        confidence=mx.ones((1, 1, 1, 2), dtype=mx.float32),
        raw_params=mx.array(raw),
        image_tensor=mx.ones((1, 1, 3, 1, 2), dtype=mx.float32) * 0.5,
    )

    names = {record["name"] for record in records}
    assert {"gaussian-attributes", "gaussians", "gaussian-metadata"} <= names
    ply_bytes = (tmp_path / "gaussians.ply").read_bytes()
    header_bytes, body = ply_bytes.split(b"end_header\n", 1)
    lines = header_bytes.decode("ascii").splitlines() + ["end_header"]
    assert lines == [
        "ply",
        "format binary_little_endian 1.0",
        "element vertex 2",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ]
    assert len(body) == 2 * 17 * 4
    first = struct.unpack("<17f", body[: 17 * 4])
    np.testing.assert_allclose(
        first,
        np.array([-1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11920292, -7.0, -7.0, -7.0, 0.0, 0.0, 0.0, 1.0]),
        atol=1e-6,
    )
    metadata = json.loads((tmp_path / "gaussian" / "metadata.json").read_text())
    assert metadata["gaussians_ply"]["format"] == "3dgs-ply"
    assert metadata["gaussians_ply"]["ply_encoding"] == "binary_little_endian"
    assert metadata["gaussians_ply"]["means_source"] == "gsdepth+pixel-grid"
