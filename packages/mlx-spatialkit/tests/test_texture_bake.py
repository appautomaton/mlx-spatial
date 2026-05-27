from __future__ import annotations

import numpy as np
import pytest

from mlx_spatialkit import NativeUvMesh, bake_pbr_texture, make_face_atlas_uvs, metal_device_available


def _uv_mesh():
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
    return make_face_atlas_uvs(vertices, faces, tile_padding=0.0)


def _texture_fields() -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.int32,
    )
    attributes = np.array(
        [
            [1.0, 0.0, 0.0, 0.1, 0.2, 1.0],
            [0.0, 1.0, 0.0, 0.3, 0.4, 0.8],
            [0.0, 0.0, 1.0, 0.5, 0.6, 0.6],
            [1.0, 1.0, 0.0, 0.7, 0.8, 0.4],
        ],
        dtype=np.float32,
    )
    return coordinates, attributes


def test_bake_pbr_texture_metal_returns_deterministic_buffers_and_diagnostics() -> None:
    mesh = _uv_mesh()
    coordinates, attributes = _texture_fields()
    if not metal_device_available():
        with pytest.raises(RuntimeError, match="Metal device unavailable"):
            bake_pbr_texture(
                mesh,
                coordinates,
                attributes,
                texture_size=4,
                origin=(0.0, 0.0, 0.0),
                voxel_size=1.0,
                decode_resolution=2,
            )
        return

    baked = bake_pbr_texture(
        mesh,
        coordinates,
        attributes,
        texture_size=4,
        origin=(0.0, 0.0, 0.0),
        voxel_size=1.0,
        decode_resolution=2,
    )

    assert baked.base_color_rgba.shape == (4, 4, 4)
    assert baked.base_color_rgba.dtype == np.uint8
    assert baked.metallic_roughness.shape == (4, 4, 3)
    assert baked.metallic_roughness.dtype == np.uint8
    assert baked.coverage_mask.shape == (4, 4)
    assert baked.coverage_status.dtype == np.uint8
    assert baked.stats["backend"] == "metal-face-atlas-nearest"
    assert baked.stats["voxel_count"] == 4
    assert baked.stats["sampled_texel_count"] > 0
    assert baked.stats["coverage_ratio"] == pytest.approx(float(np.count_nonzero(baked.coverage_mask)) / 16.0)
    np.testing.assert_array_equal(baked.base_color_rgba[0, 0], np.array([255, 0, 0, 255], dtype=np.uint8))
    np.testing.assert_array_equal(baked.metallic_roughness[0, 0], np.array([0, 51, 26], dtype=np.uint8))
    np.testing.assert_array_equal(baked.base_color_rgba[0, 1], np.array([0, 255, 0, 204], dtype=np.uint8))
    np.testing.assert_array_equal(baked.metallic_roughness[0, 1], np.array([0, 102, 77], dtype=np.uint8))


def test_bake_pbr_texture_metal_supports_provided_uv_scan_path() -> None:
    coordinates, attributes = _texture_fields()
    mesh = NativeUvMesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        uvs=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        stats={"backend": "provided"},
    )
    if not metal_device_available():
        pytest.skip("Metal device unavailable")

    baked = bake_pbr_texture(
        mesh,
        coordinates,
        attributes,
        texture_size=2,
        origin=(0.0, 0.0, 0.0),
        voxel_size=1.0,
        decode_resolution=2,
    )

    assert baked.stats["backend"] == "metal-uv-nearest"
    assert baked.stats["sampled_texel_count"] >= 1
    np.testing.assert_array_equal(baked.base_color_rgba[0, 0], np.array([255, 0, 0, 255], dtype=np.uint8))


def test_bake_pbr_texture_rejects_unsafe_texture_size_before_metal_allocation() -> None:
    mesh = _uv_mesh()
    coordinates, attributes = _texture_fields()

    with pytest.raises(ValueError, match="above guard"):
        bake_pbr_texture(
            mesh,
            coordinates,
            attributes,
            texture_size=8,
            max_texture_pixels=16,
        )


def test_bake_pbr_texture_rejects_invalid_texture_contracts() -> None:
    mesh = _uv_mesh()
    coordinates, attributes = _texture_fields()

    with pytest.raises(ValueError, match="dtype int32"):
        bake_pbr_texture(
            mesh,
            coordinates.astype(np.int64),
            attributes,
            texture_size=4,
        )

    duplicate_coordinates = coordinates.copy()
    duplicate_coordinates[1] = duplicate_coordinates[0]
    with pytest.raises(ValueError, match="unique"):
        bake_pbr_texture(
            mesh,
            duplicate_coordinates,
            attributes,
            texture_size=4,
        )
