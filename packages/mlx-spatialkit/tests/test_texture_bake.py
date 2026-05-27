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
    assert mesh.stats["packing"] == "paired-triangles"
    assert baked.stats["voxel_count"] == 4
    assert baked.stats["texture_pixel_count"] == 16
    assert baked.stats["exact_sampled_texel_count"] == baked.stats["sampled_texel_count"]
    assert baked.stats["sampled_texel_count"] > 0
    assert baked.stats["fallback_filled_texel_count"] >= 0
    assert baked.stats["uv_surface_texel_count"] == (
        baked.stats["exact_sampled_texel_count"]
        + baked.stats["fallback_filled_texel_count"]
        + baked.stats["missing_texel_count"]
        + baked.stats["out_of_grid_texel_count"]
    )
    assert baked.stats["exact_missing_texel_count"] == (
        baked.stats["fallback_filled_texel_count"] + baked.stats["missing_texel_count"]
    )
    assert baked.stats["no_face_texel_count"] + baked.stats["uv_surface_texel_count"] == 16
    assert baked.stats["visible_base_color_texel_count"] == int(np.count_nonzero(baked.base_color_rgba[:, :, 3]))
    assert baked.stats["nonzero_rgb_texel_count"] == int(np.count_nonzero(np.any(baked.base_color_rgba[:, :, :3] != 0, axis=2)))
    assert baked.stats["raw_coverage_ratio"] == pytest.approx(float(np.count_nonzero(baked.coverage_mask)) / 16.0)
    assert baked.stats["final_visible_coverage_ratio"] == pytest.approx(
        float(baked.stats["visible_base_color_texel_count"]) / 16.0
    )
    assert baked.stats["coverage_ratio"] == pytest.approx(baked.stats["final_visible_coverage_ratio"])
    assert baked.stats["uv_surface_exact_coverage_ratio"] == pytest.approx(
        baked.stats["exact_sampled_texel_count"] / baked.stats["uv_surface_texel_count"]
    )
    assert baked.stats["uv_surface_final_visible_coverage_ratio"] == pytest.approx(
        baked.stats["visible_base_color_texel_count"] / baked.stats["uv_surface_texel_count"]
    )
    assert baked.stats["uv_surface_texel_count"] >= 10
    assert baked.stats["final_visible_coverage_ratio"] >= 0.625
    assert baked.stats["dilation_max_passes"] >= 8
    assert baked.stats["dilation_pass_count"] <= baked.stats["dilation_max_passes"]
    assert baked.stats["fallback_radius"] >= 12
    np.testing.assert_array_equal(baked.base_color_rgba[0, 0], np.array([255, 0, 0, 255], dtype=np.uint8))
    np.testing.assert_array_equal(baked.metallic_roughness[0, 0], np.array([0, 51, 26], dtype=np.uint8))
    np.testing.assert_array_equal(baked.base_color_rgba[0, 2], np.array([0, 255, 0, 204], dtype=np.uint8))
    np.testing.assert_array_equal(baked.metallic_roughness[0, 2], np.array([0, 102, 77], dtype=np.uint8))


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

    assert baked.stats["backend"] == "metal-uv-binned-nearest"
    assert baked.stats["uv_bin_cols"] >= 4
    assert baked.stats["uv_bin_rows"] >= 4
    assert baked.stats["uv_bin_count"] == baked.stats["uv_bin_cols"] * baked.stats["uv_bin_rows"]
    assert baked.stats["uv_bin_face_reference_count"] > 0
    assert baked.stats["uv_bin_max_candidate_faces"] >= 1
    assert baked.stats["uv_bin_max_candidate_faces"] < baked.stats["uv_bin_face_reference_count"]
    assert baked.stats["uv_bin_guard_passed"] is True
    assert baked.stats["sampled_texel_count"] >= 1
    np.testing.assert_array_equal(baked.base_color_rgba[0, 0], np.array([255, 0, 0, 255], dtype=np.uint8))


def test_bake_pbr_texture_diagnostics_separate_missing_surface_and_no_face_texels() -> None:
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
    coordinates = np.array([[0, 0, 0, 0]], dtype=np.int32)
    attributes = np.array([[1.0, 0.25, 0.0, 0.0, 0.5, 1.0]], dtype=np.float32)
    if not metal_device_available():
        pytest.skip("Metal device unavailable")

    baked = bake_pbr_texture(
        mesh,
        coordinates,
        attributes,
        texture_size=4,
        origin=(0.0, 0.0, 0.0),
        voxel_size=1.0,
        decode_resolution=2,
    )

    assert baked.stats["no_face_texel_count"] > 0
    assert baked.stats["uv_surface_texel_count"] > 0
    assert baked.stats["exact_sampled_texel_count"] > 0
    assert baked.stats["missing_texel_count"] >= 0
    assert baked.stats["fallback_filled_texel_count"] > 0
    assert baked.stats["exact_missing_texel_count"] > 0
    fallback_texels = baked.base_color_rgba[baked.coverage_status == 4]
    assert fallback_texels.shape[0] == baked.stats["fallback_filled_texel_count"]
    assert np.all(fallback_texels[:, 3] > 0)
    assert baked.stats["visible_base_color_texel_count"] == (
        baked.stats["exact_sampled_texel_count"] + baked.stats["fallback_filled_texel_count"]
    )
    assert baked.stats["exact_missing_texel_count"] == (
        baked.stats["missing_texel_count"] + baked.stats["fallback_filled_texel_count"]
    )
    assert baked.stats["uv_surface_exact_coverage_ratio"] < 1.0
    assert baked.stats["uv_surface_final_visible_coverage_ratio"] > baked.stats["uv_surface_exact_coverage_ratio"]
    assert baked.stats["fallback_radius"] == 12
    assert baked.stats["dilation_max_passes"] == 8
    assert baked.stats["backend"] == "metal-uv-binned-nearest"
    assert baked.stats["uv_bin_max_candidate_faces"] < baked.stats["uv_bin_face_reference_count"]


def test_bake_pbr_texture_uses_adaptive_dilation_budget_for_atlas_textures() -> None:
    mesh = _uv_mesh()
    coordinates = np.array([[0, 0, 0, 0]], dtype=np.int32)
    attributes = np.array([[1.0, 0.25, 0.0, 0.0, 0.5, 1.0]], dtype=np.float32)
    if not metal_device_available():
        pytest.skip("Metal device unavailable")

    baked = bake_pbr_texture(
        mesh,
        coordinates,
        attributes,
        texture_size=32,
        origin=(0.0, 0.0, 0.0),
        voxel_size=1.0,
        decode_resolution=2,
    )

    assert baked.stats["backend"] == "metal-face-atlas-nearest"
    assert baked.stats["atlas_cols"] == 1
    assert baked.stats["atlas_rows"] == 1
    assert baked.stats["fallback_radius"] > 12
    assert baked.stats["fallback_radius"] <= 24
    assert baked.stats["dilation_max_passes"] > 8
    assert baked.stats["dilation_max_passes"] <= 64
    assert baked.stats["dilation_pass_count"] <= baked.stats["dilation_max_passes"]
    assert baked.stats["fallback_radius"] == 24
    assert baked.stats["dilation_max_passes"] == 64
    assert baked.stats["uv_bin_count"] == 0
    assert baked.stats["uv_bin_face_reference_count"] == 0


def test_bake_pbr_texture_binned_uv_path_bounds_large_candidate_sets() -> None:
    side = 34
    coords = np.stack(
        np.meshgrid(
            np.linspace(0.0, 1.0, side, dtype=np.float32),
            np.linspace(0.0, 1.0, side, dtype=np.float32),
            indexing="xy",
        ),
        axis=-1,
    ).reshape(-1, 2)
    vertices = np.column_stack([coords[:, 0], coords[:, 1], np.zeros(coords.shape[0], dtype=np.float32)]).astype(np.float32)
    faces = []
    for y in range(side - 1):
        for x in range(side - 1):
            a = y * side + x
            b = a + 1
            c = a + side
            d = c + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    mesh = NativeUvMesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        uvs=coords.astype(np.float32),
        stats={"backend": "provided"},
    )
    coordinates, attributes = _texture_fields()
    if not metal_device_available():
        pytest.skip("Metal device unavailable")

    baked = bake_pbr_texture(
        mesh,
        coordinates,
        attributes,
        texture_size=16,
        origin=(0.0, 0.0, 0.0),
        voxel_size=1.0,
        decode_resolution=2,
    )

    total_faces = int(mesh.faces.shape[0])
    assert baked.stats["backend"] == "metal-uv-binned-nearest"
    assert baked.stats["uv_bin_cols"] == 16
    assert baked.stats["uv_bin_rows"] == 16
    assert baked.stats["uv_bin_max_candidate_faces"] < total_faces // 10
    assert baked.stats["uv_bin_face_reference_count"] < total_faces * 8
    assert baked.stats["sampled_texel_count"] > 0


def test_bake_pbr_texture_binned_uv_path_rejects_reference_explosion() -> None:
    face_count = 20_000
    vertices = np.zeros((face_count * 3, 3), dtype=np.float32)
    vertices[1::3, 0] = 1.0
    vertices[2::3, 1] = 1.0
    uvs = np.zeros((face_count * 3, 2), dtype=np.float32)
    uvs[1::3, 0] = 1.0
    uvs[2::3, 1] = 1.0
    mesh = NativeUvMesh(
        vertices=vertices,
        faces=np.arange(face_count * 3, dtype=np.int64).reshape(face_count, 3),
        uvs=uvs,
        stats={"backend": "provided"},
    )
    coordinates, attributes = _texture_fields()

    with pytest.raises(ValueError, match="UV bin face references exceed guard"):
        bake_pbr_texture(
            mesh,
            coordinates,
            attributes,
            texture_size=64,
            origin=(0.0, 0.0, 0.0),
            voxel_size=1.0,
            decode_resolution=2,
        )


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
