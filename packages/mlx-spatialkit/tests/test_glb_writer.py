from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from mlx_spatialkit import NativeUvMesh, make_face_atlas_uvs, textured_glb_payload, write_textured_glb
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
