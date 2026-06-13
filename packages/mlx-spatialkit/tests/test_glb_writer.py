from __future__ import annotations

import json
import struct
import zlib

import numpy as np
import pytest

from mlx_spatialkit import (
    NativeUvMesh,
    make_face_atlas_uvs,
    make_native_chart_uvs,
    textured_glb_payload,
    write_textured_glb,
)
from mlx_spatialkit._native import uv_quality_metrics
from glb_texture_utils import glb_image_payload, png_coverage


def _fixture_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64)
    return vertices, faces


def _textures() -> tuple[np.ndarray, np.ndarray]:
    base_color = np.array(
        [
            [[255, 0, 0, 255], [0, 255, 0, 255]],
            [[0, 0, 255, 255], [255, 255, 0, 255]],
        ],
        dtype=np.uint8,
    )
    metallic_roughness = np.array(
        [
            [[0, 64, 128], [0, 96, 160]],
            [[0, 128, 192], [0, 255, 255]],
        ],
        dtype=np.uint8,
    )
    return base_color, metallic_roughness


def _glb_document_and_bin(payload: bytes) -> tuple[dict, bytes]:
    magic, version, total_length = struct.unpack_from("<III", payload, 0)
    assert magic == 0x46546C67
    assert version == 2
    assert total_length == len(payload)
    json_length, json_type = struct.unpack_from("<I4s", payload, 12)
    assert json_type == b"JSON"
    json_start = 20
    json_end = json_start + json_length
    document = json.loads(payload[json_start:json_end].rstrip(b" ").decode("utf-8"))
    bin_length, bin_type = struct.unpack_from("<I4s", payload, json_end)
    assert bin_type == b"BIN\x00"
    bin_start = json_end + 8
    assert bin_start + bin_length == len(payload)
    return document, payload[bin_start:]


def _accessor_vec3(document: dict, bin_blob: bytes, accessor_index: int) -> np.ndarray:
    accessor = document["accessors"][accessor_index]
    view = document["bufferViews"][accessor["bufferView"]]
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    count = int(accessor["count"])
    values = struct.unpack_from("<" + "f" * count * 3, bin_blob, start)
    return np.asarray(values, dtype=np.float32).reshape(count, 3)


def _png_pixels(png: bytes) -> np.ndarray:
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pos = 8
    width = 0
    height = 0
    channels = 0
    idat = bytearray()
    while pos < len(png):
        length = struct.unpack(">I", png[pos : pos + 4])[0]
        pos += 4
        chunk_type = png[pos : pos + 4]
        pos += 4
        payload = png[pos : pos + length]
        pos += length + 4
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            assert bit_depth == 8
            assert color_type in (2, 6)
            assert compression == 0
            assert filter_method == 0
            assert interlace == 0
            channels = 4 if color_type == 6 else 3
        elif chunk_type == b"IDAT":
            idat.extend(payload)
        elif chunk_type == b"IEND":
            break
    assert width > 0 and height > 0 and channels > 0
    raw = zlib.decompress(bytes(idat))
    row_bytes = width * channels
    rows = []
    offset = 0
    for _ in range(height):
        filter_type = raw[offset]
        offset += 1
        assert filter_type == 0
        rows.append(raw[offset : offset + row_bytes])
        offset += row_bytes
    return np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(height, width, channels)


def test_make_face_atlas_uvs_duplicates_vertices_and_returns_stats() -> None:
    vertices, faces = _fixture_mesh()

    mesh = make_face_atlas_uvs(vertices, faces, tile_padding=0.08)

    assert mesh.vertices.shape == (6, 3)
    assert mesh.faces.shape == (2, 3)
    assert mesh.uvs.shape == (6, 2)
    assert mesh.stats["backend"] == "face-atlas"
    assert mesh.stats["packing"] == "paired-triangles"
    assert mesh.stats["faces_per_tile"] == 2
    assert mesh.stats["atlas_tiles"] == 1
    assert mesh.stats["atlas_cols"] == 1
    assert mesh.stats["atlas_rows"] == 1
    assert mesh.stats["estimated_tile_utilization"] == pytest.approx((1.0 - 2.0 * 0.08) ** 2)
    assert mesh.stats["source_vertices"] == 4
    assert mesh.stats["output_vertices"] == 6
    assert np.all(mesh.uvs >= 0.0)
    assert np.all(mesh.uvs <= 1.0)
    np.testing.assert_array_equal(mesh.faces, np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64))
    np.testing.assert_allclose(
        mesh.uvs,
        np.array(
            [
                [0.08, 0.08],
                [0.92, 0.08],
                [0.08, 0.92],
                [0.92, 0.92],
                [0.08, 0.92],
                [0.92, 0.08],
            ],
            dtype=np.float32,
        ),
    )


def test_make_native_chart_uvs_groups_coplanar_faces_and_reuses_vertices() -> None:
    vertices, faces = _fixture_mesh()

    mesh = make_native_chart_uvs(vertices, faces, chart_angle_degrees=1.0, tile_padding=0.05)

    assert mesh.vertices.shape == (4, 3)
    assert mesh.faces.shape == (2, 3)
    assert mesh.uvs.shape == (4, 2)
    assert mesh.stats["backend"] == "native-chart-atlas"
    assert mesh.stats["packing"] == "aspect-shelf-charts"
    assert mesh.stats["source_vertices"] == 4
    assert mesh.stats["source_faces"] == 2
    assert mesh.stats["output_vertices"] == 4
    assert mesh.stats["output_faces"] == 2
    assert mesh.stats["source_chart_count"] == 1
    assert mesh.stats["chart_count"] == 1
    assert mesh.stats["chart_split_count"] == 0
    assert mesh.stats["oversized_source_chart_count"] == 0
    assert mesh.stats["low_fill_chart_split_count"] == 0
    assert mesh.stats["low_fill_split_accepted_count"] == 0
    assert mesh.stats["pre_low_fill_chart_count"] == 1
    assert mesh.stats["pre_low_fill_chart_rect_fill_ratio"] == pytest.approx(1.0)
    assert mesh.stats["max_chart_faces"] == 2
    assert mesh.stats["projection"] == "local-frame-pca"
    assert mesh.stats["projection_rotation_candidates"] == 19
    assert mesh.stats["projection_rotation_step_degrees"] == 5.0
    assert mesh.stats["chart_rect_fill_ratio"] == pytest.approx(1.0)
    assert mesh.stats["shelf_rows"] == 1
    assert mesh.stats["packed_width"] == pytest.approx(1.0)
    assert mesh.stats["packed_height"] == pytest.approx(1.0)
    assert mesh.stats["shelf_packing_efficiency"] == pytest.approx(1.0)
    assert mesh.stats["atlas_rect_coverage_ratio"] == pytest.approx(1.0)
    assert mesh.stats["chart_angle_degrees"] == 1.0
    assert mesh.stats["duplicated_vertex_ratio"] == pytest.approx(1.0)
    assert np.all(mesh.uvs >= 0.0)
    assert np.all(mesh.uvs <= 1.0)
    np.testing.assert_array_equal(mesh.faces, faces)


def test_make_native_chart_uvs_splits_hard_crease() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [2, 1, 3],
            [0, 4, 1],
            [1, 4, 5],
        ],
        dtype=np.int64,
    )

    mesh = make_native_chart_uvs(vertices, faces, chart_angle_degrees=30.0, tile_padding=0.05)

    assert mesh.stats["backend"] == "native-chart-atlas"
    assert mesh.stats["packing"] == "aspect-shelf-charts"
    assert mesh.stats["source_chart_count"] == 2
    assert mesh.stats["chart_count"] == 2
    assert mesh.stats["chart_split_count"] == 0
    assert mesh.stats["source_vertices"] == 6
    assert mesh.stats["output_vertices"] == 8
    assert mesh.stats["output_faces"] == 4
    assert mesh.stats["max_chart_faces"] == 2
    assert mesh.stats["duplicated_vertex_ratio"] == pytest.approx(8 / 6)
    assert mesh.stats["chart_normal_cos_threshold"] == pytest.approx(np.cos(np.deg2rad(30.0)))
    assert mesh.stats["projection"] == "local-frame-pca"
    assert mesh.stats["projection_rotation_candidates"] == 19
    assert mesh.stats["projection_rotation_step_degrees"] == 5.0
    assert mesh.stats["low_fill_chart_split_count"] == 0
    assert mesh.stats["chart_rect_fill_ratio"] > 0.0
    assert mesh.stats["shelf_rows"] >= 1
    assert mesh.stats["shelf_packing_efficiency"] > 0.0
    assert mesh.stats["atlas_rect_coverage_ratio"] > 0.0
    assert mesh.faces.shape == (4, 3)
    assert mesh.uvs.shape == (8, 2)
    assert np.all(mesh.uvs >= 0.0)
    assert np.all(mesh.uvs <= 1.0)


def test_make_native_chart_uvs_limits_curved_surface_normal_drift() -> None:
    vertices: list[list[float]] = []
    for angle_degrees in range(0, 121, 20):
        angle = np.deg2rad(float(angle_degrees))
        x = float(np.sin(angle))
        z = float(np.cos(angle))
        vertices.append([x, 0.0, z])
        vertices.append([x, 1.0, z])

    faces: list[list[int]] = []
    for segment in range((len(vertices) // 2) - 1):
        lower_left = segment * 2
        upper_left = lower_left + 1
        lower_right = lower_left + 2
        upper_right = lower_left + 3
        faces.append([lower_left, lower_right, upper_left])
        faces.append([upper_left, lower_right, upper_right])

    mesh = make_native_chart_uvs(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.int64),
        chart_angle_degrees=30.0,
        tile_padding=0.02,
    )

    assert mesh.stats["backend"] == "native-chart-atlas"
    assert mesh.stats["chart_cluster_normal_policy"] == "edge-and-seed-cone"
    assert mesh.stats["source_chart_count"] > 1
    assert mesh.stats["chart_cone_rejected_adjacency_count"] > 0
    assert mesh.stats["chart_normal_cos_threshold"] == pytest.approx(np.cos(np.deg2rad(30.0)))
    assert mesh.stats["output_vertices"] > mesh.stats["source_vertices"]
    assert np.all(mesh.uvs >= 0.0)
    assert np.all(mesh.uvs <= 1.0)


def test_make_native_chart_uvs_local_projection_fills_rotated_rectangle() -> None:
    width = 4.0
    height = 1.0
    angle = np.deg2rad(37.0)
    base = np.array(
        [
            [-width / 2.0, -height / 2.0, 0.0],
            [width / 2.0, -height / 2.0, 0.0],
            [-width / 2.0, height / 2.0, 0.0],
            [width / 2.0, height / 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    vertices = base @ rotation.T
    faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64)

    mesh = make_native_chart_uvs(vertices, faces, chart_angle_degrees=1.0, tile_padding=0.04)

    assert mesh.stats["backend"] == "native-chart-atlas"
    assert mesh.stats["projection"] == "local-frame-pca"
    assert mesh.stats["packing"] == "aspect-shelf-charts"
    assert mesh.stats["chart_count"] == 1
    assert mesh.stats["chart_rect_fill_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert mesh.stats["packed_width"] == pytest.approx(1.0)
    assert mesh.stats["packed_height"] == pytest.approx(0.25)
    assert np.ptp(mesh.uvs[:, 0]) == pytest.approx(0.92, abs=1e-6)
    assert np.ptp(mesh.uvs[:, 1]) == pytest.approx(0.23, abs=1e-6)


def test_make_native_chart_uvs_splits_low_fill_l_shape_deterministically() -> None:
    cells = 5
    vertices = []
    for y in range(cells + 1):
        for x in range(cells + 1):
            vertices.append([float(x), float(y), 0.0])
    faces = []
    for y in range(cells):
        for x in range(cells):
            if x != 0 and y != 0:
                continue
            v0 = y * (cells + 1) + x
            v1 = v0 + 1
            v2 = v0 + cells + 1
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v2, v1, v3])
    vertices_array = np.asarray(vertices, dtype=np.float32)
    faces_array = np.asarray(faces, dtype=np.int64)

    mesh = make_native_chart_uvs(vertices_array, faces_array, chart_angle_degrees=1.0, tile_padding=0.02)
    repeat = make_native_chart_uvs(vertices_array, faces_array, chart_angle_degrees=1.0, tile_padding=0.02)

    assert mesh.stats["backend"] == "native-chart-atlas"
    assert mesh.stats["source_chart_count"] == 1
    assert mesh.stats["pre_low_fill_chart_count"] == 1
    assert mesh.stats["low_fill_rect_fill_threshold"] == pytest.approx(0.70)
    assert mesh.stats["low_fill_split_min_faces"] == 4
    assert mesh.stats["low_fill_split_min_child_faces"] == 2
    assert mesh.stats["low_fill_split_max_depth"] == 3
    assert mesh.stats["low_fill_split_axis_candidates"] == 2
    assert mesh.stats["low_fill_split_position_candidates"] == 5
    assert mesh.stats["pre_low_fill_chart_rect_fill_ratio"] < mesh.stats["low_fill_rect_fill_threshold"]
    assert mesh.stats["low_fill_split_candidate_count"] > 0
    assert mesh.stats["low_fill_split_axis_candidate_count"] == (
        mesh.stats["low_fill_split_candidate_count"] * mesh.stats["low_fill_split_axis_candidates"]
    )
    assert mesh.stats["low_fill_split_partition_candidate_count"] == (
        mesh.stats["low_fill_split_axis_candidate_count"] * mesh.stats["low_fill_split_position_candidates"]
    )
    assert mesh.stats["low_fill_split_partition_evaluated_count"] > mesh.stats["low_fill_split_axis_candidate_count"]
    assert mesh.stats["low_fill_split_partition_evaluated_count"] <= mesh.stats["low_fill_split_partition_candidate_count"]
    assert mesh.stats["low_fill_source_chart_count"] == 1
    assert mesh.stats["low_fill_split_accepted_count"] > 0
    assert mesh.stats["low_fill_chart_split_count"] == mesh.stats["chart_count"] - mesh.stats["pre_low_fill_chart_count"]
    assert mesh.stats["chart_rect_fill_ratio"] > mesh.stats["pre_low_fill_chart_rect_fill_ratio"]
    assert mesh.stats["chart_count"] > mesh.stats["pre_low_fill_chart_count"]
    assert mesh.faces.shape == faces_array.shape
    assert mesh.uvs.shape[0] == mesh.vertices.shape[0]
    np.testing.assert_array_equal(mesh.faces, repeat.faces)
    np.testing.assert_allclose(mesh.vertices, repeat.vertices)
    np.testing.assert_allclose(mesh.uvs, repeat.uvs)
    for key in (
        "chart_count",
        "chart_rect_fill_ratio",
        "low_fill_split_candidate_count",
        "low_fill_split_partition_candidate_count",
        "low_fill_split_partition_evaluated_count",
        "low_fill_split_accepted_count",
        "low_fill_chart_split_count",
    ):
        assert repeat.stats[key] == pytest.approx(mesh.stats[key])


def test_make_native_chart_uvs_splits_oversized_coplanar_chart() -> None:
    columns = 34
    rows = 18
    vertices = []
    for y in range(rows):
        for x in range(columns):
            vertices.append([float(x), float(y), 0.0])
    faces = []
    for y in range(rows - 1):
        for x in range(columns - 1):
            v0 = y * columns + x
            v1 = v0 + 1
            v2 = v0 + columns
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v2, v1, v3])
    vertices_array = np.asarray(vertices, dtype=np.float32)
    faces_array = np.asarray(faces, dtype=np.int64)

    mesh = make_native_chart_uvs(vertices_array, faces_array, chart_angle_degrees=1.0, tile_padding=0.04)

    assert mesh.stats["backend"] == "native-chart-atlas"
    assert mesh.stats["source_chart_count"] == 1
    assert mesh.stats["chart_count"] > 1
    assert mesh.stats["chart_split_count"] == mesh.stats["chart_count"] - mesh.stats["source_chart_count"]
    assert mesh.stats["low_fill_chart_split_count"] == 0
    assert mesh.stats["oversized_chart_split_count"] == mesh.stats["chart_count"] - mesh.stats["source_chart_count"]
    assert mesh.stats["oversized_source_chart_count"] == 1
    assert mesh.stats["chart_split_max_faces"] == 512
    assert mesh.stats["max_chart_faces"] <= mesh.stats["chart_split_max_faces"]
    assert mesh.stats["output_faces"] == faces_array.shape[0]
    assert mesh.faces.shape == faces_array.shape
    assert mesh.uvs.shape[0] == mesh.vertices.shape[0]
    assert np.all(mesh.uvs >= 0.0)
    assert np.all(mesh.uvs <= 1.0)


def test_make_native_chart_uvs_rejects_non_finite_parameters() -> None:
    vertices, faces = _fixture_mesh()

    with pytest.raises(ValueError, match="chart_angle_degrees"):
        make_native_chart_uvs(vertices, faces, chart_angle_degrees=float("nan"))

    with pytest.raises(ValueError, match="tile_padding"):
        make_native_chart_uvs(vertices, faces, tile_padding=float("nan"))


def test_textured_glb_payload_contains_mesh_material_textures_and_metadata() -> None:
    vertices, faces = _fixture_mesh()
    mesh = make_face_atlas_uvs(vertices, faces)
    base_color, metallic_roughness = _textures()

    payload = textured_glb_payload(
        mesh,
        base_color_rgba=base_color,
        metallic_roughness=metallic_roughness,
        generator="mlx-spatial Pixal3D",
        mesh_name="Pixal3D_TexturedMesh",
        material_name="Pixal3D_PBR",
    )
    document, bin_blob = _glb_document_and_bin(payload)

    assert payload.startswith(b"glTF")
    assert document["asset"] == {"version": "2.0", "generator": "mlx-spatial Pixal3D"}
    assert document["nodes"][0]["name"] == "Pixal3D_TexturedMesh"
    assert document["meshes"][0]["name"] == "Pixal3D_TexturedMesh"
    primitive = document["meshes"][0]["primitives"][0]
    assert primitive["attributes"] == {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}
    assert primitive["indices"] == 3
    assert primitive["material"] == 0
    material = document["materials"][0]
    assert material["name"] == "Pixal3D_PBR"
    assert material["doubleSided"] is True
    assert material["alphaMode"] == "OPAQUE"
    pbr = material["pbrMetallicRoughness"]
    assert pbr["baseColorTexture"] == {"index": 0}
    assert pbr["metallicRoughnessTexture"] == {"index": 1}
    assert document["images"][0]["mimeType"] == "image/png"
    assert document["images"][1]["mimeType"] == "image/png"
    assert len(document["bufferViews"]) == 6
    assert len(document["accessors"]) == 4
    assert document["accessors"][0]["count"] == 6
    assert document["accessors"][1]["count"] == 6
    assert document["accessors"][1]["type"] == "VEC3"
    assert document["accessors"][2]["count"] == 6
    assert document["accessors"][2]["type"] == "VEC2"
    assert document["accessors"][3]["count"] == 6
    assert document["accessors"][3]["componentType"] == 5123
    assert document["buffers"][0]["byteLength"] == len(bin_blob)
    assert all(view["byteOffset"] % 4 == 0 for view in document["bufferViews"])
    for image in document["images"]:
        view = document["bufferViews"][image["bufferView"]]
        start = view["byteOffset"]
        end = start + view["byteLength"]
        assert bin_blob[start:end].startswith(b"\x89PNG\r\n\x1a\n")
    coverage = png_coverage(glb_image_payload(payload, "baseColorTexture"))
    assert coverage.width == 2
    assert coverage.height == 2
    assert coverage.channels == 4
    assert coverage.alpha_nonzero_count == 4
    assert coverage.rgb_nonzero_count == 4
    assert coverage.alpha_coverage_ratio == 1.0


def test_textured_glb_payload_preserves_material_channels_separately_from_normals() -> None:
    vertices, faces = _fixture_mesh()
    mesh = make_face_atlas_uvs(vertices, faces)
    base_color, metallic_roughness = _textures()

    payload = textured_glb_payload(
        mesh,
        base_color_rgba=base_color,
        metallic_roughness=metallic_roughness,
        mesh_name="MaterialNormalFixture",
        material_name="UnequalPBR",
    )
    document, bin_blob = _glb_document_and_bin(payload)

    material = document["materials"][0]
    assert material["doubleSided"] is True
    assert material["alphaMode"] == "OPAQUE"
    pbr = material["pbrMetallicRoughness"]
    assert pbr["baseColorTexture"] == {"index": 0}
    assert pbr["metallicRoughnessTexture"] == {"index": 1}
    assert pbr["metallicFactor"] == 1
    assert pbr["roughnessFactor"] == 1
    np.testing.assert_array_equal(
        _png_pixels(glb_image_payload(payload, "metallicRoughnessTexture")),
        metallic_roughness,
    )

    primitive = document["meshes"][0]["primitives"][0]
    normals = _accessor_vec3(document, bin_blob, primitive["attributes"]["NORMAL"])
    assert normals.shape == (6, 3)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), np.ones(6), atol=1e-6)


def test_textured_glb_payload_smooths_normals_across_duplicate_position_seams() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    uvs = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 0.5],
            [0.5, 1.0],
        ],
        dtype=np.float32,
    )
    mesh = NativeUvMesh(vertices=vertices, faces=faces, uvs=uvs, stats={})
    base_color, metallic_roughness = _textures()

    document, bin_blob = _glb_document_and_bin(
        textured_glb_payload(mesh, base_color_rgba=base_color, metallic_roughness=metallic_roughness)
    )

    primitive = document["meshes"][0]["primitives"][0]
    normals = _accessor_vec3(document, bin_blob, primitive["attributes"]["NORMAL"])
    np.testing.assert_allclose(normals[0], normals[3], atol=1e-6)
    np.testing.assert_allclose(normals[1], normals[5], atol=1e-6)


def test_textured_glb_payload_splits_large_mesh_into_uint16_primitives() -> None:
    vertex_count = 66_000
    vertices = np.zeros((vertex_count, 3), dtype=np.float32)
    vertices[:, 0] = np.arange(vertex_count, dtype=np.float32) / float(vertex_count)
    vertices[:, 1] = (np.arange(vertex_count, dtype=np.float32) % 3.0) / 3.0
    faces = np.arange(vertex_count, dtype=np.int64).reshape(-1, 3)
    uvs = np.zeros((vertex_count, 2), dtype=np.float32)
    mesh = NativeUvMesh(vertices=vertices, faces=faces, uvs=uvs, stats={})
    base_color = np.full((1, 1, 4), 255, dtype=np.uint8)
    metallic_roughness = np.full((1, 1, 3), 127, dtype=np.uint8)

    payload = textured_glb_payload(
        mesh,
        base_color_rgba=base_color,
        metallic_roughness=metallic_roughness,
    )
    document, _ = _glb_document_and_bin(payload)

    primitives = document["meshes"][0]["primitives"]
    assert len(primitives) == 2
    for primitive in primitives:
        assert set(primitive["attributes"]) == {"POSITION", "NORMAL", "TEXCOORD_0"}
        position = document["accessors"][primitive["attributes"]["POSITION"]]
        normal = document["accessors"][primitive["attributes"]["NORMAL"]]
        texcoord = document["accessors"][primitive["attributes"]["TEXCOORD_0"]]
        indices = document["accessors"][primitive["indices"]]
        assert position["count"] <= 65_536
        assert normal["count"] == position["count"]
        assert texcoord["count"] == position["count"]
        assert indices["componentType"] == 5123
        assert indices["max"][0] <= 65_535


def test_write_textured_glb_writes_payload_and_metadata(tmp_path) -> None:
    vertices, faces = _fixture_mesh()
    mesh = make_face_atlas_uvs(vertices, faces)
    base_color, metallic_roughness = _textures()

    artifact = write_textured_glb(
        tmp_path / "model.glb",
        mesh,
        base_color_rgba=base_color,
        metallic_roughness=metallic_roughness,
        metadata={"pipeline_type": "1024_cascade"},
    )

    assert artifact.path == tmp_path / "model.glb"
    assert artifact.path.read_bytes().startswith(b"glTF")
    assert artifact.bytes_written == artifact.path.stat().st_size
    assert artifact.metadata["stage"] == "textured_glb"
    assert artifact.metadata["pipeline_type"] == "1024_cascade"


def test_textured_glb_payload_rejects_invalid_geometry_and_textures() -> None:
    vertices, faces = _fixture_mesh()
    mesh = make_face_atlas_uvs(vertices, faces)
    base_color, metallic_roughness = _textures()

    bad_vertices = mesh.vertices.copy()
    bad_vertices[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        textured_glb_payload(
            NativeUvMesh(vertices=bad_vertices, faces=mesh.faces, uvs=mesh.uvs, stats={}),
            base_color_rgba=base_color,
            metallic_roughness=metallic_roughness,
        )

    bad_faces = mesh.faces.copy()
    bad_faces[0, 0] = mesh.vertices.shape[0]
    with pytest.raises(ValueError, match="outside the vertex array"):
        textured_glb_payload(
            NativeUvMesh(vertices=mesh.vertices, faces=bad_faces, uvs=mesh.uvs, stats={}),
            base_color_rgba=base_color,
            metallic_roughness=metallic_roughness,
        )

    bad_uvs = mesh.uvs.copy()
    bad_uvs[0, 0] = 1.25
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        textured_glb_payload(
            NativeUvMesh(vertices=mesh.vertices, faces=mesh.faces, uvs=bad_uvs, stats={}),
            base_color_rgba=base_color,
            metallic_roughness=metallic_roughness,
        )

    with pytest.raises(ValueError, match="uint8"):
        textured_glb_payload(
            mesh,
            base_color_rgba=base_color.astype(np.float32),
            metallic_roughness=metallic_roughness,
        )


def test_textured_glb_payload_rejects_huge_logical_texture_before_copying() -> None:
    vertices, faces = _fixture_mesh()
    mesh = make_face_atlas_uvs(vertices, faces)
    _, metallic_roughness = _textures()
    huge_base_color = np.broadcast_to(np.zeros((1, 1, 4), dtype=np.uint8), (8193, 1, 4))

    with pytest.raises(ValueError, match="dimensions"):
        textured_glb_payload(
            mesh,
            base_color_rgba=huge_base_color,
            metallic_roughness=metallic_roughness,
        )


def _uv_metrics_grid_mesh(n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, n, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    vertices = np.stack(
        [grid_x.ravel(), grid_y.ravel(), np.zeros(n * n, dtype=np.float32)],
        axis=1,
    ).astype(np.float32)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = (i + 1) * n + j
            c = i * n + j + 1
            d = (i + 1) * n + j + 1
            faces.append([a, b, c])
            faces.append([c, b, d])
    return vertices, np.array(faces, dtype=np.int64)


def _uv_metrics_overlap_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    uvs = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.25, 0.25],
            [1.25, 0.25],
            [0.25, 1.25],
        ],
        dtype=np.float32,
    )
    return vertices, faces, uvs


def test_uv_metrics_clean_grid_has_no_overlaps_or_flips() -> None:
    vertices, faces = _uv_metrics_grid_mesh()
    uvs = vertices[:, :2].copy()

    metrics = uv_quality_metrics(vertices, faces, uvs)

    assert metrics["uv_overlap_count"] == 0
    assert metrics["uv_flipped_count"] == 0
    assert metrics["uv_degenerate_count"] == 0
    assert metrics["uv_overlap_checked_pairs"] >= 0
    assert metrics["uv_total_area"] == pytest.approx(1.0)
    assert metrics["uv_bbox_utilization"] == pytest.approx(1.0)
    assert metrics["uv_stretch_l2"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["uv_stretch_linf"] == pytest.approx(1.0, rel=1e-6)


def test_uv_metrics_counts_overlapping_uv_triangles() -> None:
    vertices, faces, uvs = _uv_metrics_overlap_fixture()

    metrics = uv_quality_metrics(vertices, faces, uvs)

    assert metrics["uv_overlap_count"] == 1
    assert metrics["uv_overlap_checked_pairs"] >= 1
    assert metrics["uv_flipped_count"] == 0
    assert metrics["uv_degenerate_count"] == 0


def test_uv_metrics_overlap_dedup_counts_each_pair_once_across_shared_cells() -> None:
    # Three identical atlas-spanning triangles plus four tiny well-separated
    # triangles inside them.  The tiny extents drag the median-based cell size
    # down, so each big triangle occupies every grid cell and every big-big /
    # big-tiny pair co-occupies many cells.  The dedup must still check each
    # unordered pair exactly once: 3 big-big + 3*4 big-tiny = 15 pairs, all of
    # which genuinely overlap; the tiny triangles share no cells (and do not
    # overlap) with each other.
    big_uv = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    tiny_offsets = [(0.02, 0.02), (0.32, 0.02), (0.02, 0.32), (0.32, 0.32)]
    uv_rows: list[tuple[float, float]] = []
    for _ in range(3):
        uv_rows.extend(big_uv)
    for x, y in tiny_offsets:
        uv_rows.extend([(x, y), (x + 0.06, y), (x, y + 0.06)])
    uvs = np.array(uv_rows, dtype=np.float32)
    vertices = np.array(
        [[u, v, float(i // 3)] for i, (u, v) in enumerate(uv_rows)],
        dtype=np.float32,
    )
    faces = np.arange(len(uv_rows), dtype=np.int64).reshape(-1, 3)

    metrics = uv_quality_metrics(vertices, faces, uvs)

    assert metrics["uv_overlap_checked_pairs"] == 15
    assert metrics["uv_overlap_count"] == 15
    assert metrics["uv_flipped_count"] == 0
    assert metrics["uv_degenerate_count"] == 0

    repeat = uv_quality_metrics(vertices, faces, uvs)
    assert repeat["uv_overlap_checked_pairs"] == 15
    assert repeat["uv_overlap_count"] == 15


def test_uv_metrics_counts_mirrored_uv_triangle_as_flipped() -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    mirrored_uvs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    metrics = uv_quality_metrics(vertices, faces, mirrored_uvs)

    assert metrics["uv_flipped_count"] == 1
    assert metrics["uv_overlap_count"] == 0
    assert metrics["uv_total_area"] == pytest.approx(0.5)
    assert metrics["uv_bbox_utilization"] == pytest.approx(0.5)

    clean_uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    clean = uv_quality_metrics(vertices, faces, clean_uvs)
    assert clean["uv_flipped_count"] == 0


def test_uv_metrics_isotropic_stretch_matches_scale() -> None:
    vertices, faces = _uv_metrics_grid_mesh()
    scale = 2.0
    uvs = (vertices[:, :2] / scale).astype(np.float32)

    metrics = uv_quality_metrics(vertices, faces, uvs)

    assert metrics["uv_stretch_l2"] == pytest.approx(scale, rel=1e-6)
    assert metrics["uv_stretch_linf"] == pytest.approx(scale, rel=1e-6)


def test_uv_metrics_anisotropic_stretch_reports_max_singular_value() -> None:
    vertices, faces = _uv_metrics_grid_mesh()
    squash = 4.0
    uvs = vertices[:, :2].copy()
    uvs[:, 0] /= squash

    metrics = uv_quality_metrics(vertices, faces, uvs)

    assert metrics["uv_stretch_linf"] == pytest.approx(squash, rel=1e-6)
    assert metrics["uv_stretch_l2"] == pytest.approx(
        np.sqrt((squash**2 + 1.0) / 2.0), rel=1e-6
    )


def _uv_metrics_two_chart_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [2, 1, 3], [4, 5, 6], [6, 5, 7]], dtype=np.int64)
    uvs = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
            [0.6, 0.0],
            [0.85, 0.0],
            [0.6, 0.25],
            [0.85, 0.25],
        ],
        dtype=np.float32,
    )
    chart_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    return vertices, faces, uvs, chart_ids


def test_uv_metrics_per_chart_stretch_breakdown() -> None:
    vertices, faces, uvs, chart_ids = _uv_metrics_two_chart_fixture()

    metrics = uv_quality_metrics(vertices, faces, uvs, chart_ids=chart_ids)

    assert metrics["chart_ids_present"] == [0, 1]
    assert metrics["chart_stretch_l2"] == pytest.approx([2.0, 4.0], rel=1e-6)
    assert metrics["chart_stretch_linf"] == pytest.approx([2.0, 4.0], rel=1e-6)
    assert metrics["uv_stretch_linf"] == pytest.approx(4.0, rel=1e-6)
    assert metrics["uv_flipped_count"] == 0
    assert metrics["uv_degenerate_count"] == 0
    assert metrics["uv_overlap_count"] == 0


def test_uv_metrics_repeat_calls_are_deterministic() -> None:
    vertices, faces, uvs = _uv_metrics_overlap_fixture()
    first = uv_quality_metrics(vertices, faces, uvs)
    second = uv_quality_metrics(vertices, faces, uvs)
    assert first == second

    chart_vertices, chart_faces, chart_uvs, chart_ids = _uv_metrics_two_chart_fixture()
    chart_first = uv_quality_metrics(chart_vertices, chart_faces, chart_uvs, chart_ids=chart_ids)
    chart_second = uv_quality_metrics(chart_vertices, chart_faces, chart_uvs, chart_ids=chart_ids)
    assert chart_first == chart_second


# ---------------------------------------------------------------------------
# UV-parity oracle anchors (tests/data/uv_oracle_anchors.json)
#
# Version-pinned reference values produced by pip xatlas 0.0.11 (the CuMesh
# uv_unwrap reference composition) on both cached fixtures' QEM-decimated 50k
# meshes, via tests/tools/gen_uv_oracle_anchors.py.  These tests read the
# committed anchors only; xatlas itself must NOT be importable here.
# ---------------------------------------------------------------------------

import importlib.util
import math as _math
from pathlib import Path as _Path

_UV_ORACLE_ANCHORS_PATH = _Path(__file__).resolve().parent / "data" / "uv_oracle_anchors.json"
_UV_ORACLE_FIXTURE_NAMES = ("main", "violin_bow")
_UV_ORACLE_BLOCK_KEYS = (
    "chart_count",
    "atlas_utilization",
    "uv_overlap_count",
    "uv_flipped_count",
    "uv_stretch_l2",
    "uv_stretch_linf",
    "uv_bbox_utilization",
    "uv_total_area",
    "output_vertices",
    "duplicated_vertex_ratio",
)


def _load_uv_oracle_anchors() -> dict:
    return json.loads(_UV_ORACLE_ANCHORS_PATH.read_text())


def test_uv_oracle_anchors_file_exists_and_parses() -> None:
    assert _UV_ORACLE_ANCHORS_PATH.exists(), (
        f"missing committed oracle anchors: {_UV_ORACLE_ANCHORS_PATH} "
        "(regenerate with .venv/bin/python tests/tools/gen_uv_oracle_anchors.py)"
    )
    anchors = _load_uv_oracle_anchors()
    assert isinstance(anchors, dict)
    assert "generated_with" in anchors
    assert "fixtures" in anchors


def test_uv_oracle_anchors_pinned_xatlas_version_and_option_mapping() -> None:
    generated_with = _load_uv_oracle_anchors()["generated_with"]
    assert generated_with["xatlas_version"] == "0.0.11"
    option_mapping = generated_with["option_mapping"]
    # Every CuMesh reference option must be mapped to a pip attribute (or "n/a").
    chart_options = option_mapping["chart_options"]
    for cumesh_name in (
        "max_cost",
        "normal_deviation_weight",
        "roundness",
        "straightness",
        "normal_seam",
        "texture_seam",
        "max_iterations",
    ):
        assert chart_options[cumesh_name]["pip_attr"]
    pack_options = option_mapping["pack_options"]
    for cumesh_name in ("padding", "bilinear", "rotate_charts", "brute_force"):
        assert pack_options[cumesh_name]["pip_attr"]


def test_uv_oracle_anchors_have_both_fixtures_with_required_keys() -> None:
    fixtures = _load_uv_oracle_anchors()["fixtures"]
    for name in _UV_ORACLE_FIXTURE_NAMES:
        record = fixtures[name]
        assert record["source_faces"] > 0
        assert record["source_vertices"] > 0
        assert record["stage_a_cluster_count"] >= 1
        for block_name in ("per_cluster_composition", "whole_mesh"):
            block = record[block_name]
            for key in _UV_ORACLE_BLOCK_KEYS:
                assert key in block, f"{name}.{block_name} missing {key}"


def test_uv_oracle_anchors_values_sane() -> None:
    fixtures = _load_uv_oracle_anchors()["fixtures"]
    for name in _UV_ORACLE_FIXTURE_NAMES:
        record = fixtures[name]
        cluster_count = record["stage_a_cluster_count"]
        assert cluster_count >= 1
        composition = record["per_cluster_composition"]
        # xatlas re-splits clusters but never merges across add_mesh calls,
        # so the composition has at least one chart per stage-A cluster.
        assert composition["chart_count"] >= cluster_count
        for block_name in ("per_cluster_composition", "whole_mesh"):
            block = record[block_name]
            label = f"{name}.{block_name}"
            assert block["chart_count"] >= 1, label
            assert 0.0 < block["atlas_utilization"] <= 1.0, label
            assert 0.0 < block["uv_bbox_utilization"] <= 1.0, label
            for stretch_key in ("uv_stretch_l2", "uv_stretch_linf"):
                stretch = block[stretch_key]
                assert _math.isfinite(stretch), f"{label}.{stretch_key}"
                assert stretch >= 0.5, f"{label}.{stretch_key}"
            assert block["uv_stretch_linf"] >= block["uv_stretch_l2"], label
            # Overlap/flip counts from real xatlas are RECORDED, not judged:
            # padding=0 packing and xatlas's mirrored charts make small
            # overlap counts and large flip counts genuine reference output.
            assert isinstance(block["uv_overlap_count"], int), label
            assert block["uv_overlap_count"] >= 0, label
            assert isinstance(block["uv_flipped_count"], int), label
            assert block["uv_flipped_count"] >= 0, label
            assert 0.0 < block["uv_total_area"] <= 1.0, label
            assert block["output_vertices"] >= record["source_vertices"], label
            assert block["duplicated_vertex_ratio"] >= 1.0, label
            # Per-chart stretch summaries exclude the 0.0 "no measurable
            # faces" sentinel (fully mirrored charts, ~half of real xatlas
            # output); measured vs total chart counts record the split.
            for summary_key in ("chart_stretch_l2_summary", "chart_stretch_linf_summary"):
                summary = block[summary_key]
                slabel = f"{label}.{summary_key}"
                assert 1 <= summary["measured_chart_count"] <= summary["total_chart_count"], slabel
                assert summary["total_chart_count"] == block["chart_count"], slabel
                # Heavy-tailed distributions can put the mean above p95
                # (linf outliers reach thousands), so bound each against
                # max only.
                assert 0.0 < summary["mean"] <= summary["max"], slabel
                assert 0.0 < summary["p95"] <= summary["max"], slabel
        assert record["whole_mesh"]["parametrize_matches_atlas"] is True


def test_uv_oracle_xatlas_not_importable_in_project_venv() -> None:
    # The oracle dependency lives ONLY in the throwaway oracle venv; the
    # project venv (and therefore src/ and tests/) must never import xatlas.
    assert importlib.util.find_spec("xatlas") is None, (
        "xatlas must NOT be installed in the project venv; anchors are "
        "regenerated via the oracle venv (see tests/tools/gen_uv_oracle_anchors.py)"
    )


# ---------------------------------------------------------------------------
# Stage-B packing + reference-backend assembly (slice 6): pack_uv_charts
# (xatlas PackOptions semantics: texel gaps, rotate-to-axis, deterministic
# shelf placement) and make_reference_uvs (full cluster->grow->parameterize->
# pack pipeline emitting the NativeUvMesh contract).
# ---------------------------------------------------------------------------

from mlx_spatialkit._native import (
    grow_uv_charts as _grow_uv_charts,
    pack_uv_charts as _pack_uv_charts,
    parameterize_uv_charts as _parameterize_uv_charts,
)
from mlx_spatialkit.export import make_reference_uvs


def _cube_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], dtype=np.float32)
    faces = np.array(
        [[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4],
         [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]],
        dtype=np.int64)
    return vertices, faces


def _cube_parameterized() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices, faces = _cube_mesh()
    grown = _grow_uv_charts(vertices, faces)
    param = _parameterize_uv_charts(
        vertices, faces, np.ascontiguousarray(np.asarray(grown["chart_ids"])))
    chart_ids = np.ascontiguousarray(np.asarray(param["chart_ids"]), dtype=np.int64)
    corner_uvs = np.ascontiguousarray(np.asarray(param["corner_uvs"]), dtype=np.float64)
    return vertices, faces, chart_ids, corner_uvs


def test_pack_uv_charts_padding_and_unit_square() -> None:
    _, faces, chart_ids, corner_uvs = _cube_parameterized()
    padding = 2.0
    packed = _pack_uv_charts(
        faces, chart_ids, corner_uvs, resolution=128, padding=padding, bilinear=True)
    assert packed["gap_texels"] == padding + 1.0
    uvs = np.asarray(packed["corner_uvs"])
    assert (uvs >= 0.0).all() and (uvs <= 1.0).all()
    # Padding honored: pairwise chart-rect gaps >= gap_texels on at least one
    # axis (rects either share a shelf or sit on different shelves).
    rects = np.asarray(packed["chart_rects_texels"])  # [C, 4] x, y, w, h
    chart_count = rects.shape[0]
    assert chart_count == int(chart_ids.max()) + 1
    for i in range(chart_count):
        for j in range(i + 1, chart_count):
            xi, yi, wi, hi = rects[i]
            xj, yj, wj, hj = rects[j]
            gap_x = max(xj - (xi + wi), xi - (xj + wj))
            gap_y = max(yj - (yi + hi), yi - (yj + hj))
            assert max(gap_x, gap_y) >= packed["gap_texels"] - 1e-6, (i, j)


def test_pack_uv_charts_deterministic_repeat() -> None:
    _, faces, chart_ids, corner_uvs = _cube_parameterized()
    first = _pack_uv_charts(faces, chart_ids, corner_uvs)
    second = _pack_uv_charts(faces, chart_ids, corner_uvs)
    np.testing.assert_array_equal(
        np.asarray(first["corner_uvs"]), np.asarray(second["corner_uvs"]))
    assert first["texels_per_unit"] == second["texels_per_unit"]


def test_pack_uv_charts_rejects_invalid_arguments() -> None:
    _, faces, chart_ids, corner_uvs = _cube_parameterized()
    with pytest.raises(ValueError):
        _pack_uv_charts(faces, chart_ids, corner_uvs, resolution=0)
    with pytest.raises(ValueError):
        _pack_uv_charts(faces, chart_ids, corner_uvs, padding=-1.0)
    with pytest.raises(ValueError):
        _pack_uv_charts(faces, chart_ids[:3], corner_uvs)
    with pytest.raises(ValueError):
        _pack_uv_charts(faces, chart_ids, corner_uvs[:5])


def test_make_reference_uvs_contract_and_invariants() -> None:
    vertices, faces = _cube_mesh()
    mesh = make_reference_uvs(vertices, faces, texture_resolution=256)
    # NativeUvMesh contract: corner positions preserved through duplication.
    assert mesh.faces.shape == faces.shape
    np.testing.assert_allclose(mesh.vertices[mesh.faces], vertices[faces])
    assert mesh.uvs.shape == (mesh.vertices.shape[0], 2)
    assert (mesh.uvs >= 0.0).all() and (mesh.uvs <= 1.0).all()
    stats = mesh.stats
    assert stats["backend"] == "xatlas-equivalent-native"
    assert stats["chart_count"] == 6
    assert stats["uv_overlap_count"] == 0
    assert stats["uv_flipped_count"] == 0
    assert stats["output_vertices"] == 24  # 4 per planar 2-face chart
    assert stats["gap_texels"] == 1.0  # padding 0 + bilinear gutter
    for key in (
        "stage_a_cluster_count", "growth_chart_count", "projected_chart_count",
        "lscm_chart_count", "shattered_face_chart_count", "texels_per_unit",
        "uv_stretch_l2", "uv_bbox_utilization", "duplicated_vertex_ratio",
    ):
        assert key in stats, key


def test_make_reference_uvs_deterministic() -> None:
    vertices, faces = _cube_mesh()
    first = make_reference_uvs(vertices, faces, texture_resolution=256)
    second = make_reference_uvs(vertices, faces, texture_resolution=256)
    np.testing.assert_array_equal(first.uvs, second.uvs)
    np.testing.assert_array_equal(first.faces, second.faces)
    np.testing.assert_array_equal(first.vertices, second.vertices)
