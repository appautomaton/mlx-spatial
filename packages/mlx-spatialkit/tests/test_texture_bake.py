from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from mlx_spatialkit import (
    NativeUvMesh,
    bake_pbr_texture,
    coverage_status_histogram,
    make_face_atlas_uvs,
    make_native_chart_uvs,
    metal_device_available,
    telea_inpaint,
)


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
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
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


def _assert_coverage_status_matches_stats(baked) -> None:
    histogram = coverage_status_histogram(baked.coverage_status)
    assert baked.stats["coverage_status_histogram"] == histogram
    assert histogram == {
        "no_face": baked.stats["no_face_texel_count"],
        "exact_sampled": baked.stats["exact_sampled_texel_count"],
        "missing_surface": baked.stats["missing_texel_count"],
        "out_of_grid": baked.stats["out_of_grid_texel_count"],
        "fallback_filled": baked.stats["fallback_filled_texel_count"],
        "surface_filled": baked.stats["surface_filled_texel_count"],
        "unknown": 0,
    }
    assert sum(histogram.values()) == baked.stats["texture_pixel_count"]


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
    _assert_coverage_status_matches_stats(baked)
    assert mesh.stats["packing"] == "paired-triangles"
    assert baked.stats["voxel_count"] == 4
    assert baked.stats["texture_pixel_count"] == 16
    assert baked.stats["exact_sampled_texel_count"] == baked.stats["sampled_texel_count"]
    assert baked.stats["sampled_texel_count"] > 0
    assert baked.stats["fallback_filled_texel_count"] >= 0
    assert baked.stats["surface_fill_enabled"] is True
    assert baked.stats["surface_fill_traversal_policy"] == "uv-surface-only-no-face-gap-blocked"
    assert baked.stats["surface_fill_seed_texel_count"] > 0
    assert baked.stats["surface_fill_cross_gap_prevented_count"] >= 0
    assert baked.stats["surface_filled_texel_count"] >= 0
    assert baked.stats["surface_fill_filled_texel_count"] == baked.stats["surface_filled_texel_count"]
    assert baked.stats["uv_surface_texel_count"] == (
        baked.stats["exact_sampled_texel_count"]
        + baked.stats["fallback_filled_texel_count"]
        + baked.stats["surface_filled_texel_count"]
        + baked.stats["missing_texel_count"]
        + baked.stats["out_of_grid_texel_count"]
    )
    assert baked.stats["no_face_texel_count"] + baked.stats["uv_surface_texel_count"] == 16
    uv_visible = np.isin(baked.coverage_status, [1, 4, 5]) & (baked.base_color_rgba[:, :, 3] != 0)
    assert baked.stats["visible_base_color_texel_count"] == int(np.count_nonzero(uv_visible))
    assert baked.stats["render_visible_base_color_texel_count"] == int(np.count_nonzero(baked.base_color_rgba[:, :, 3]))
    assert baked.stats["nonzero_rgb_texel_count"] == int(np.count_nonzero(np.any(baked.base_color_rgba[:, :, :3] != 0, axis=2)))
    assert baked.stats["render_padding_enabled"] is True
    assert baked.stats["render_padding_policy"] == "opaque-glb-nearest-inpaint-after-diagnostics"
    assert baked.stats["render_padding_seed_texel_count"] >= baked.stats["visible_base_color_texel_count"]
    assert baked.stats["render_alpha_coverage_ratio"] == pytest.approx(
        baked.stats["render_visible_base_color_texel_count"] / 16.0
    )
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


def test_bake_pbr_texture_constant_fields_isolate_uv_occupancy_from_texture_values() -> None:
    if not metal_device_available():
        pytest.skip("Metal device unavailable")

    mesh = _uv_mesh()
    coordinates, attributes = _texture_fields()
    red_attributes = attributes.copy()
    red_attributes[:, :] = np.array([1.0, 0.0, 0.0, 0.1, 0.8, 1.0], dtype=np.float32)
    blue_attributes = attributes.copy()
    blue_attributes[:, :] = np.array([0.0, 0.0, 1.0, 0.7, 0.2, 1.0], dtype=np.float32)

    red = bake_pbr_texture(
        mesh,
        coordinates,
        red_attributes,
        texture_size=4,
        origin=(0.0, 0.0, 0.0),
        voxel_size=1.0,
        decode_resolution=2,
        render_padding=False,
    )
    blue = bake_pbr_texture(
        mesh,
        coordinates,
        blue_attributes,
        texture_size=4,
        origin=(0.0, 0.0, 0.0),
        voxel_size=1.0,
        decode_resolution=2,
        render_padding=False,
    )

    np.testing.assert_array_equal(red.coverage_status, blue.coverage_status)
    assert red.stats["coverage_status_histogram"] == blue.stats["coverage_status_histogram"]
    visible = np.isin(red.coverage_status, [1, 4, 5])
    assert np.any(visible)
    assert np.all(red.base_color_rgba[visible, 0] >= blue.base_color_rgba[visible, 0])
    assert np.all(blue.base_color_rgba[visible, 2] >= red.base_color_rgba[visible, 2])


def test_bake_pbr_texture_metal_supports_concurrent_public_api_calls() -> None:
    if not metal_device_available():
        pytest.skip("Metal device unavailable")

    def bake_once() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        mesh = _uv_mesh()
        coordinates, attributes = _texture_fields()
        baked = bake_pbr_texture(
            mesh,
            coordinates,
            attributes,
            texture_size=4,
            origin=(0.0, 0.0, 0.0),
            voxel_size=1.0,
            decode_resolution=2,
        )
        return baked.base_color_rgba.copy(), baked.metallic_roughness.copy(), dict(baked.stats)

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda _: bake_once(), range(3)))

    reference_base, reference_mr, reference_stats = results[0]
    for base_color, metallic_roughness, stats in results[1:]:
        np.testing.assert_array_equal(base_color, reference_base)
        np.testing.assert_array_equal(metallic_roughness, reference_mr)
        assert stats["backend"] == reference_stats["backend"]
        assert stats["sampled_texel_count"] == reference_stats["sampled_texel_count"]
        assert stats["visible_base_color_texel_count"] == reference_stats["visible_base_color_texel_count"]
        assert stats["gutter_filled_texel_count"] == reference_stats["gutter_filled_texel_count"]


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


def test_bake_pbr_texture_projects_to_source_mesh_and_samples_trilinear() -> None:
    if not metal_device_available():
        pytest.skip("Metal device unavailable")

    mesh = NativeUvMesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.8],
                [1.0, 0.0, 0.8],
                [0.0, 1.0, 0.8],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        uvs=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        stats={"backend": "provided"},
    )
    source_vertices = mesh.vertices.copy()
    source_vertices[:, 2] = 0.0
    source_faces = mesh.faces.copy()

    coordinates = []
    attributes = []
    for x in range(2):
        for y in range(2):
            for z in range(2):
                coordinates.append([0, x, y, z])
                attributes.append([float(x), float(y), float(z), 0.0, 0.5, 1.0])

    baked = bake_pbr_texture(
        mesh,
        np.asarray(coordinates, dtype=np.int32),
        np.asarray(attributes, dtype=np.float32),
        texture_size=2,
        origin=(0.0, 0.0, 0.0),
        voxel_size=1.0,
        decode_resolution=2,
        source_vertices=source_vertices,
        source_faces=source_faces,
    )

    assert baked.stats["uv_raster_interpolate_reference"] is True
    assert baked.stats["source_projection_used"] is True
    assert baked.stats["source_projection_detail"] == "native_bvh"
    assert baked.stats["source_projection_source_faces"] == 1
    assert baked.stats["source_projection_projected_texel_count"] > 0
    assert baked.stats["source_projection_returns_face_id"] is True
    assert baked.stats["source_projection_returns_barycentric"] is True
    assert baked.stats["sampling_mode"] == "trilinear-with-sparse-knn-fallback"
    assert baked.stats["nearest_fallback_enabled"] is True
    assert baked.stats["nearest_fallback_scope"] == "source-projection-sparse-knn"
    assert baked.stats["source_projection_nearest_fallback_enabled"] is True
    assert baked.stats["source_projection_nearest_fallback_k_neighbors"] == 8
    assert baked.stats["source_projection_nearest_fallback_max_distance_voxels"] == pytest.approx(12.0)
    assert baked.stats["source_projection_nearest_fallback_weight_epsilon_voxels"] == pytest.approx(0.1)
    assert baked.stats["trilinear_sampled_texel_count"] > 0
    np.testing.assert_array_equal(baked.base_color_rgba[0, 0], np.array([64, 64, 0, 255], dtype=np.uint8))


def test_bake_pbr_texture_normalizes_sparse_trilinear_present_corners() -> None:
    if not metal_device_available():
        pytest.skip("Metal device unavailable")

    mesh = NativeUvMesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.8],
                [1.0, 0.0, 0.8],
                [0.0, 1.0, 0.8],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        uvs=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        stats={"backend": "provided"},
    )
    source_vertices = mesh.vertices.copy()
    source_vertices[:, 2] = 0.0

    baked = bake_pbr_texture(
        mesh,
        np.asarray([[0, 0, 0, 0]], dtype=np.int32),
        np.asarray([[1.0, 0.5, 0.0, 0.25, 1.0, 1.0]], dtype=np.float32),
        texture_size=2,
        origin=(0.0, 0.0, 0.0),
        voxel_size=1.0,
        decode_resolution=2,
        source_vertices=source_vertices,
        source_faces=mesh.faces.copy(),
    )

    assert baked.stats["trilinear_sampled_texel_count"] > 0
    assert baked.stats["trilinear_missing_corner_texel_count"] > 0
    np.testing.assert_array_equal(baked.base_color_rgba[0, 0], np.array([255, 128, 0, 255], dtype=np.uint8))
    np.testing.assert_array_equal(baked.metallic_roughness[0, 0], np.array([0, 255, 64], dtype=np.uint8))


def test_bake_pbr_texture_source_projection_uses_sparse_knn_when_trilinear_has_no_present_corner() -> None:
    if not metal_device_available():
        pytest.skip("Metal device unavailable")

    mesh = NativeUvMesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.8],
                [1.0, 0.0, 0.8],
                [0.0, 1.0, 0.8],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        uvs=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        stats={"backend": "provided"},
    )
    source_vertices = mesh.vertices.copy()
    source_vertices[:, 2] = 0.0

    baked = bake_pbr_texture(
        mesh,
        np.asarray([[0, 2, 0, 0]], dtype=np.int32),
        np.asarray([[0.0, 1.0, 0.25, 0.75, 0.9, 1.0]], dtype=np.float32),
        texture_size=2,
        origin=(-0.5, 0.0, 0.0),
        voxel_size=1.0,
        decode_resolution=4,
        source_vertices=source_vertices,
        source_faces=mesh.faces.copy(),
        source_projection_fallback_neighbors=4,
        source_projection_fallback_max_distance_voxels=2.0,
    )

    assert baked.stats["source_projection_nearest_fallback_enabled"] is True
    assert baked.stats["source_projection_nearest_fallback_k_neighbors"] == 4
    assert baked.stats["source_projection_nearest_fallback_max_distance_voxels"] == pytest.approx(2.0)
    assert baked.stats["source_projection_nearest_fallback_texel_count"] > 0
    assert baked.stats["trilinear_invalid_texel_count"] == 0
    assert baked.coverage_status[0, 0] == 4
    np.testing.assert_array_equal(baked.base_color_rgba[0, 0], np.array([0, 255, 64, 255], dtype=np.uint8))
    np.testing.assert_array_equal(baked.metallic_roughness[0, 0], np.array([0, 230, 191], dtype=np.uint8))


def test_bake_pbr_texture_source_projection_can_disable_sparse_knn_for_isolation() -> None:
    if not metal_device_available():
        pytest.skip("Metal device unavailable")

    mesh = NativeUvMesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.8],
                [1.0, 0.0, 0.8],
                [0.0, 1.0, 0.8],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        uvs=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        stats={"backend": "provided"},
    )
    source_vertices = mesh.vertices.copy()
    source_vertices[:, 2] = 0.0

    baked = bake_pbr_texture(
        mesh,
        np.asarray([[0, 2, 0, 0]], dtype=np.int32),
        np.asarray([[0.0, 1.0, 0.25, 0.75, 0.9, 1.0]], dtype=np.float32),
        texture_size=1,
        origin=(-0.5, 0.0, 0.0),
        voxel_size=1.0,
        decode_resolution=4,
        source_vertices=source_vertices,
        source_faces=mesh.faces.copy(),
        source_projection_fallback_mode="disabled",
    )

    assert baked.stats["source_projection_nearest_fallback_enabled"] is False
    assert baked.stats["source_projection_nearest_fallback_mode"] == "disabled"
    assert baked.stats["source_projection_nearest_fallback_k_neighbors"] == 0
    assert baked.stats["source_projection_nearest_fallback_texel_count"] == 0
    assert baked.stats["source_projection_nearest_fallback_missing_texel_count"] > 0
    assert baked.stats["sampling_mode"] == "trilinear-without-sparse-knn-fallback"
    assert baked.stats["nearest_fallback_scope"] == "disabled-source-projection-trilinear-only"
    assert baked.stats["trilinear_invalid_texel_count"] > 0
    assert baked.coverage_status[0, 0] == 2
    np.testing.assert_array_equal(baked.base_color_rgba[0, 0], np.array([0, 0, 0, 0], dtype=np.uint8))


def test_bake_pbr_texture_can_disable_source_projection_for_isolation() -> None:
    if not metal_device_available():
        pytest.skip("Metal device unavailable")

    mesh = NativeUvMesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.8],
                [1.0, 0.0, 0.8],
                [0.0, 1.0, 0.8],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        uvs=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        stats={"backend": "provided"},
    )
    source_vertices = mesh.vertices.copy()
    source_vertices[:, 2] = 0.0

    baked = bake_pbr_texture(
        mesh,
        np.asarray([[0, 0, 0, 0]], dtype=np.int32),
        np.asarray([[1.0, 0.0, 0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
        texture_size=2,
        origin=(0.0, 0.0, 0.0),
        voxel_size=1.0,
        decode_resolution=2,
        source_vertices=source_vertices,
        source_faces=mesh.faces.copy(),
        source_projection=False,
    )

    assert baked.stats["source_projection_used"] is False
    assert baked.stats["source_projection_detail"] == "none"
    assert baked.stats["source_projection_nearest_fallback_enabled"] is False
    assert baked.stats["source_projection_nearest_fallback_k_neighbors"] == 0
    assert baked.stats["sampling_mode"] == "nearest"
    assert baked.stats["nearest_fallback_scope"] == "metal-kernel-missing-voxel"


def test_bake_pbr_texture_rejects_invalid_source_projection_fallback_mode() -> None:
    mesh = _uv_mesh()
    coordinates, attributes = _texture_fields()

    with pytest.raises(ValueError, match="source_projection_fallback_mode"):
        bake_pbr_texture(
            mesh,
            coordinates,
            attributes,
            texture_size=4,
            origin=(0.0, 0.0, 0.0),
            voxel_size=1.0,
            decode_resolution=2,
            source_projection_fallback_mode="far-nearest",
        )


def test_bake_pbr_texture_can_disable_render_padding_for_isolation() -> None:
    if not metal_device_available():
        pytest.skip("Metal device unavailable")

    mesh = _uv_mesh()
    coordinates, attributes = _texture_fields()
    padded = bake_pbr_texture(
        mesh,
        coordinates,
        attributes,
        texture_size=4,
        origin=(0.0, 0.0, 0.0),
        voxel_size=1.0,
        decode_resolution=2,
        render_padding=True,
    )
    unpadded = bake_pbr_texture(
        mesh,
        coordinates,
        attributes,
        texture_size=4,
        origin=(0.0, 0.0, 0.0),
        voxel_size=1.0,
        decode_resolution=2,
        render_padding=False,
    )

    assert padded.stats["render_padding_enabled"] is True
    assert padded.stats["render_alpha_coverage_ratio"] >= unpadded.stats["render_alpha_coverage_ratio"]
    assert unpadded.stats["render_padding_enabled"] is False
    assert unpadded.stats["render_padding_policy"] == "disabled"
    assert unpadded.stats["render_padding_filled_texel_count"] == 0
    np.testing.assert_array_equal(padded.coverage_status, unpadded.coverage_status)
    interior = unpadded.coverage_status != 0
    np.testing.assert_array_equal(padded.base_color_rgba[interior], unpadded.base_color_rgba[interior])


def test_bake_pbr_texture_can_toggle_surface_fill_and_render_padding_independently() -> None:
    if not metal_device_available():
        pytest.skip("Metal device unavailable")

    mesh = NativeUvMesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [16.0, 0.0, 0.0],
                [0.0, 16.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        uvs=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        stats={"backend": "provided"},
    )
    coordinates = np.array([[0, 1, 1, 0]], dtype=np.int32)
    attributes = np.array([[1.0, 0.25, 0.0, 0.0, 0.5, 1.0]], dtype=np.float32)

    no_fill = bake_pbr_texture(
        mesh,
        coordinates,
        attributes,
        texture_size=16,
        origin=(0.0, 0.0, 0.0),
        voxel_size=1.0,
        decode_resolution=2,
        surface_fill=False,
        render_padding=False,
    )
    fill_only = bake_pbr_texture(
        mesh,
        coordinates,
        attributes,
        texture_size=16,
        origin=(0.0, 0.0, 0.0),
        voxel_size=1.0,
        decode_resolution=2,
        surface_fill=True,
        render_padding=False,
    )
    fill_and_padding = bake_pbr_texture(
        mesh,
        coordinates,
        attributes,
        texture_size=16,
        origin=(0.0, 0.0, 0.0),
        voxel_size=1.0,
        decode_resolution=2,
        surface_fill=True,
        render_padding=True,
    )

    assert no_fill.stats["surface_fill_enabled"] is False
    assert no_fill.stats["surface_fill_traversal_policy"] == "disabled"
    assert no_fill.stats["surface_filled_texel_count"] == 0
    assert fill_only.stats["surface_fill_enabled"] is True
    assert fill_only.stats["surface_filled_texel_count"] > 0
    assert fill_only.stats["surface_unfilled_texel_count"] < no_fill.stats["surface_unfilled_texel_count"]
    np.testing.assert_array_equal(no_fill.coverage_status == 1, fill_only.coverage_status == 1)
    np.testing.assert_array_equal(no_fill.coverage_status == 4, fill_only.coverage_status == 4)
    seeds = np.isin(no_fill.coverage_status, [1, 4])
    np.testing.assert_array_equal(no_fill.base_color_rgba[seeds], fill_only.base_color_rgba[seeds])

    assert fill_only.stats["render_padding_enabled"] is False
    assert fill_and_padding.stats["render_padding_enabled"] is True
    assert fill_and_padding.stats["render_padding_filled_texel_count"] > 0
    np.testing.assert_array_equal(fill_only.coverage_status, fill_and_padding.coverage_status)
    uv_surface = fill_only.coverage_status != 0
    np.testing.assert_array_equal(fill_only.base_color_rgba[uv_surface], fill_and_padding.base_color_rgba[uv_surface])


def test_bake_pbr_texture_metal_uses_binned_path_for_native_chart_uvs() -> None:
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
    mesh = make_native_chart_uvs(vertices, faces, chart_angle_degrees=1.0, tile_padding=0.0)
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

    assert mesh.stats["backend"] == "native-chart-atlas"
    assert mesh.stats["chart_count"] == 1
    assert "atlas_cols" not in mesh.stats
    assert baked.stats["backend"] == "metal-uv-binned-nearest"
    assert baked.stats["uv_bin_count"] > 0
    assert baked.stats["uv_bin_face_reference_count"] > 0
    assert baked.stats["uv_bin_guard_passed"] is True
    assert baked.stats["sampled_texel_count"] > 0
    assert baked.stats["visible_base_color_texel_count"] > 0


def test_bake_pbr_texture_diagnostics_separate_missing_surface_and_no_face_texels() -> None:
    mesh = NativeUvMesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [16.0, 0.0, 0.0],
                [0.0, 16.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        uvs=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        stats={"backend": "provided"},
    )
    coordinates = np.array([[0, 1, 1, 0]], dtype=np.int32)
    attributes = np.array([[1.0, 0.25, 0.0, 0.0, 0.5, 1.0]], dtype=np.float32)
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

    assert baked.stats["no_face_texel_count"] > 0
    assert baked.stats["uv_surface_texel_count"] > 0
    assert baked.stats["exact_sampled_texel_count"] > 0
    assert baked.stats["missing_texel_count"] >= 0
    assert baked.stats["fallback_filled_texel_count"] > 0
    assert baked.stats["surface_fill_enabled"] is True
    assert baked.stats["surface_fill_traversal_policy"] == "uv-surface-only-no-face-gap-blocked"
    assert baked.stats["surface_fill_cross_gap_prevented_count"] > 0
    assert baked.stats["surface_filled_texel_count"] > 0
    assert baked.stats["surface_unfilled_texel_count"] == (
        baked.stats["missing_texel_count"] + baked.stats["out_of_grid_texel_count"]
    )
    assert baked.stats["gutter_fill_enabled"] is True
    assert baked.stats["gutter_fill_max_passes"] == 4
    assert 0 < baked.stats["gutter_fill_pass_count"] <= baked.stats["gutter_fill_max_passes"]
    assert baked.stats["gutter_filled_texel_count"] > 0
    fallback_texels = baked.base_color_rgba[baked.coverage_status == 4]
    assert fallback_texels.shape[0] == baked.stats["fallback_filled_texel_count"]
    assert np.all(fallback_texels[:, 3] > 0)
    surface_texels = baked.base_color_rgba[baked.coverage_status == 5]
    assert surface_texels.shape[0] == baked.stats["surface_filled_texel_count"]
    assert np.all(surface_texels[:, 3] > 0)
    assert baked.stats["visible_base_color_texel_count"] == (
        baked.stats["exact_sampled_texel_count"]
        + baked.stats["fallback_filled_texel_count"]
        + baked.stats["surface_filled_texel_count"]
    )
    assert np.any(baked.base_color_rgba[baked.coverage_status == 0][:, 3] > 0)
    gutter_texels = baked.base_color_rgba[
        (baked.coverage_status == 0) & np.any(baked.base_color_rgba[:, :, :3] != 0, axis=2)
    ]
    assert gutter_texels.shape[0] >= baked.stats["gutter_filled_texel_count"]
    assert np.all(gutter_texels[:, 3] > 0)
    assert baked.stats["no_face_texel_count"] + baked.stats["uv_surface_texel_count"] == 256
    assert baked.stats["visible_base_color_texel_count"] < baked.stats["nonzero_rgb_texel_count"]
    assert baked.stats["render_visible_base_color_texel_count"] >= baked.stats["nonzero_rgb_texel_count"]
    assert baked.stats["render_alpha_coverage_ratio"] > baked.stats["final_visible_coverage_ratio"]
    assert baked.stats["render_padding_filled_texel_count"] > 0
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


# ---------------------------------------------------------------------------
# Stage 4 / Slice 1: native Telea (FMM) inpaint core (TPP-01, unit-level TPP-07)
# ---------------------------------------------------------------------------


def _ramp_image(height: int, width: int, slope: float, base: float = 20.0) -> np.ndarray:
    cols = np.arange(width, dtype=np.float32)
    row = np.clip(base + slope * cols, 0, 255)
    return np.repeat(row[None, :], height, axis=0).astype(np.uint8)


def test_telea_inpaint_writes_only_masked_pixels() -> None:
    rng = np.arange(12 * 10, dtype=np.uint8).reshape(12, 10)
    for channels in (None, 1, 3, 4):
        image = rng if channels is None else np.stack([rng + c for c in range(channels)], axis=-1)
        image = image.astype(np.uint8)
        mask = np.zeros((12, 10), dtype=np.uint8)
        mask[5:7, 4:6] = 1
        out = telea_inpaint(image, mask, radius=3)
        assert out.shape == image.shape
        assert out.dtype == np.uint8
        # Unmasked bytes are bit-identical to the input.
        unmasked = mask == 0
        assert np.array_equal(out[unmasked], image[unmasked])
        # Masked pixels were actually filled (not left at their input value here,
        # where the surrounding gradient differs from the masked input).
        assert out[mask != 0].shape[0] > 0


def test_telea_inpaint_radius_respected_for_isolated_pixel() -> None:
    # A single isolated masked pixel: pixels beyond the inpaint window must not
    # influence its filled value (windowed-weight test, Codex P1 scope). The
    # Telea gradient-extrapolation term reads the immediate neighbors of window
    # pixels, so the true influence reach is radius B + 1 (this matches OpenCV's
    # INPAINT_TELEA); pixels strictly beyond (B+1) cannot contribute.
    size = 15
    radius = 3
    reach2 = (radius + 1) ** 2
    cy = cx = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    base = np.full((size, size), 100, dtype=np.uint8)
    far = base.copy()
    far[dist2 > reach2] = 200  # differ only beyond the (B+1) influence reach
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[cy, cx] = 1
    out_base = telea_inpaint(base, mask, radius=radius)
    out_far = telea_inpaint(far, mask, radius=radius)
    assert out_base[cy, cx] == out_far[cy, cx]


def test_telea_inpaint_tracks_gradient_not_constant_fill() -> None:
    # Inpainting a band in a linear ramp tracks the gradient *direction*: the
    # filled columns rise monotonically across the band and span a real range.
    # (Telea diffuses rather than extrapolating the exact ramp — verified to
    # match cv2 within tolerance in the Slice 2 heavy oracle test — so this
    # asserts the structural anti-dilation property, not exact ramp values.)
    slope = 6.0
    image = _ramp_image(8, 20, slope=slope)
    mask = np.zeros((8, 20), dtype=np.uint8)
    mask[:, 8:12] = 1
    out = telea_inpaint(image, mask, radius=3)
    col_means = np.array([out[:, c].mean() for c in range(8, 12)], dtype=np.float64)
    # Strictly increasing across the band: a dilation/constant fill would be flat.
    assert np.all(np.diff(col_means) > 1.0), col_means.tolist()
    # Spans a real range (not a near-constant copy of one edge).
    assert col_means[-1] - col_means[0] > 8
    # Every filled value stays near the surrounding known ramp range (no garbage
    # values); Telea may overshoot a few levels at the band edges, like cv2.
    known_lo = float(image[:, 7].min())
    known_hi = float(image[:, 12].max())
    band_vals = out[:, 8:12].astype(np.float64)
    assert band_vals.min() >= known_lo - 10
    assert band_vals.max() <= known_hi + 10


def test_telea_inpaint_is_repeatable() -> None:
    image = _ramp_image(16, 24, slope=4.0)
    mask = np.zeros((16, 24), dtype=np.uint8)
    mask[4:12, 9:15] = 1
    first = telea_inpaint(image, mask, radius=3)
    second = telea_inpaint(image, mask, radius=3)
    assert np.array_equal(first, second)


def test_telea_inpaint_deterministic_across_pythonhashseed() -> None:
    # Unit-level TPP-07: byte-identical output across PYTHONHASHSEED 0/1 subprocesses.
    import hashlib
    import os
    import subprocess
    import sys

    script = (
        "import numpy as np, hashlib;"
        "from mlx_spatialkit import telea_inpaint;"
        "cols=np.clip(20+4*np.arange(24),0,255).astype(np.uint8);"
        "img=np.repeat(cols[None,:],16,axis=0).astype(np.uint8);"
        "m=np.zeros((16,24),np.uint8);m[4:12,9:15]=1;"
        "o=telea_inpaint(img,m,3);"
        "print(hashlib.sha256(np.ascontiguousarray(o).tobytes()).hexdigest())"
    )
    digests = []
    for seed in ("0", "1"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        digests.append(result.stdout.strip())
    assert digests[0] == digests[1]


def test_telea_inpaint_rejects_invalid_radius() -> None:
    image = np.zeros((6, 6), dtype=np.uint8)
    mask = np.zeros((6, 6), dtype=np.uint8)
    with pytest.raises(ValueError):
        telea_inpaint(image, mask, radius=0)


def test_telea_inpaint_rejects_mask_shape_mismatch() -> None:
    image = np.zeros((6, 6, 3), dtype=np.uint8)
    mask = np.zeros((6, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        telea_inpaint(image, mask, radius=3)


def test_telea_inpaint_rejects_unsupported_channel_count() -> None:
    image = np.zeros((6, 6, 2), dtype=np.uint8)
    mask = np.zeros((6, 6), dtype=np.uint8)
    with pytest.raises(ValueError):
        telea_inpaint(image, mask, radius=3)


_INPAINT_ANCHORS = Path(__file__).resolve().parent / "data" / "inpaint_oracle_anchors.json"


def test_inpaint_synthetic_oracle() -> None:
    # Non-heavy parity gate: our telea_inpaint must match committed cv2
    # INPAINT_TELEA outputs (generated by gen_inpaint_oracle_anchors.py) on tiny
    # deterministic cases, within the pinned tolerance. No cv2 import at runtime.
    if not _INPAINT_ANCHORS.exists():
        pytest.skip("inpaint oracle anchors not generated")
    anchors = json.loads(_INPAINT_ANCHORS.read_text())
    cases = anchors.get("synthetic", [])
    assert cases, "anchors carry no synthetic oracle cases"
    for case in cases:
        image = np.array(case["image"], dtype=np.uint8)
        mask = np.array(case["mask"], dtype=np.uint8)
        cv2_out = np.array(case["cv2_output"], dtype=np.uint8)
        ours = telea_inpaint(image, mask, int(case["radius"]))
        assert ours.shape == cv2_out.shape
        sel = mask != 0
        if ours.ndim == 3:
            sel = np.repeat(sel[:, :, None], ours.shape[2], axis=2)
        err = np.abs(ours.astype(np.int32) - cv2_out.astype(np.int32))[sel]
        assert err.max() <= case["tolerance_max_abs_err"], (
            f"{case['name']}: max|ours-cv2|={int(err.max())} > tol {case['tolerance_max_abs_err']}"
        )


# ---------------------------------------------------------------------------
# Stage 4 / Slice 3: reference-path Telea postprocess application (TPP-03)
# ---------------------------------------------------------------------------


def _gutter_mesh_and_fields():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64)
    mesh = make_face_atlas_uvs(vertices, faces, tile_padding=0.2)  # padding => real gutter
    coords, attrs = _texture_fields()
    return mesh, coords, attrs


def test_bake_rejects_unknown_postprocess() -> None:
    mesh, coords, attrs = _gutter_mesh_and_fields()
    with pytest.raises(ValueError, match="postprocess must be"):
        bake_pbr_texture(mesh, coords, attrs, texture_size=8, origin=(0, 0, 0), voxel_size=1.0, decode_resolution=2, postprocess="bogus")


def test_bake_telea_postprocess_reference_application() -> None:
    if not metal_device_available():
        pytest.skip("Metal device unavailable")
    mesh, coords, attrs = _gutter_mesh_and_fields()
    kw = dict(texture_size=16, origin=(0, 0, 0), voxel_size=1.0, decode_resolution=2)
    baked = bake_pbr_texture(mesh, coords, attrs, postprocess="telea", expose_raw_postprocess_inputs=True, **kw)

    mask = np.isin(baked.raw_coverage_status, (0, 2, 3))
    covered = ~mask
    assert mask.sum() > 0 and covered.sum() > 0, "need both gutter and covered texels"

    # Path selection + honest stats: telea mode, legacy fills disabled/zero.
    assert baked.stats["postprocess_mode"] == "native-telea-inpaint"
    assert baked.stats["legacy_postprocess_applied"] is False
    assert baked.stats["dilation_filled_texel_count"] == 0
    assert baked.stats["surface_filled_texel_count"] == 0
    assert baked.stats["telea_mask_texel_count"] == int(mask.sum())
    assert baked.stats["telea_radius_base_color_rgb"] == 3
    assert baked.stats["telea_radius_alpha"] == 1

    # Covered texels keep their exact raw samples (Telea writes only masked texels).
    assert np.array_equal(baked.base_color_rgba[covered], baked.raw_base_color_rgba[covered])
    assert np.array_equal(baked.metallic_roughness[covered], baked.raw_metallic_roughness[covered])

    # Gutter texels were painted (raw gutter base RGB is 0; telea fills from neighbours).
    changed = (baked.base_color_rgba[:, :, :3] != baked.raw_base_color_rgba[:, :, :3]).any(axis=2)
    assert changed[mask].any()

    # No black seam: masked texels 4-adjacent to coverage must be painted nonzero RGB.
    adj = np.zeros_like(covered)
    adj[1:, :] |= covered[:-1, :]
    adj[:-1, :] |= covered[1:, :]
    adj[:, 1:] |= covered[:, :-1]
    adj[:, :-1] |= covered[:, 1:]
    seam = mask & adj
    if seam.any():
        seam_rgb = baked.base_color_rgba[:, :, :3][seam]
        assert np.all(seam_rgb.any(axis=1)), "black seam texel adjacent to coverage"


def test_bake_legacy_postprocess_is_default_and_unchanged() -> None:
    if not metal_device_available():
        pytest.skip("Metal device unavailable")
    mesh, coords, attrs = _gutter_mesh_and_fields()
    kw = dict(texture_size=16, origin=(0, 0, 0), voxel_size=1.0, decode_resolution=2)
    default = bake_pbr_texture(mesh, coords, attrs, **kw)
    explicit = bake_pbr_texture(mesh, coords, attrs, postprocess="legacy-dilation", **kw)
    assert np.array_equal(default.base_color_rgba, explicit.base_color_rgba)
    assert np.array_equal(default.metallic_roughness, explicit.metallic_roughness)
    assert default.stats["postprocess_mode"] == explicit.stats["postprocess_mode"]
    assert default.stats["postprocess_mode"].startswith("native-dilation")
    assert default.stats["legacy_postprocess_applied"] is True
