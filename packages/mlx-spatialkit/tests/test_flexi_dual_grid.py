from __future__ import annotations

import numpy as np
import pytest

from mlx_spatialkit import extract_flexi_dual_grid


def _cube_coordinates() -> np.ndarray:
    return np.array(
        [[0, z, y, x] for z in (0, 1) for y in (0, 1) for x in (0, 1)],
        dtype=np.int32,
    )


def _cube_fields() -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0] for _ in range(8)], dtype=np.float32)


def test_extract_flexi_dual_grid_matches_python_tiny_cube_reference() -> None:
    mesh = extract_flexi_dual_grid(_cube_coordinates(), _cube_fields(), grid_size=2)

    np.testing.assert_allclose(
        mesh.vertices,
        np.array(
            [
                [-0.25, -0.25, -0.25],
                [-0.25, -0.25, 0.25],
                [-0.25, 0.25, -0.25],
                [-0.25, 0.25, 0.25],
                [0.25, -0.25, -0.25],
                [0.25, -0.25, 0.25],
                [0.25, 0.25, -0.25],
                [0.25, 0.25, 0.25],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        mesh.faces,
        np.array(
            [
                [0, 1, 2],
                [2, 1, 3],
                [0, 4, 1],
                [1, 4, 5],
                [0, 2, 4],
                [4, 2, 6],
                [1, 3, 5],
                [5, 3, 7],
                [2, 6, 3],
                [3, 6, 7],
                [4, 5, 6],
                [6, 5, 7],
            ],
            dtype=np.int64,
        ),
    )


def test_extract_flexi_dual_grid_returns_empty_mesh_when_no_valid_quad_exists() -> None:
    coordinates = np.array([[0, 0, 0, 0]], dtype=np.int32)
    fields = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]], dtype=np.float32)

    mesh = extract_flexi_dual_grid(coordinates, fields, grid_size=2)

    assert mesh.vertices.shape == (0, 3)
    assert mesh.faces.shape == (0, 3)
    assert mesh.vertices.dtype == np.float32
    assert mesh.faces.dtype == np.int64


def test_extract_flexi_dual_grid_uses_split_weight_to_choose_quad_diagonal() -> None:
    coordinates = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.int32,
    )
    fields = np.zeros((4, 7), dtype=np.float32)
    fields[:, 3] = 1.0
    fields[0, 6] = 10.0
    fields[3, 6] = 10.0

    mesh = extract_flexi_dual_grid(coordinates, fields, grid_size=2)

    np.testing.assert_array_equal(mesh.faces, np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int64))


def test_extract_flexi_dual_grid_uses_strict_positive_intersection_threshold() -> None:
    coordinates = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.int32,
    )
    fields = np.zeros((4, 7), dtype=np.float32)

    mesh = extract_flexi_dual_grid(coordinates, fields, grid_size=2)
    assert mesh.faces.shape == (0, 3)

    fields[:, 3] = np.finfo(np.float32).eps
    mesh = extract_flexi_dual_grid(coordinates, fields, grid_size=2)
    np.testing.assert_array_equal(mesh.faces, np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64))


def test_extract_flexi_dual_grid_rejects_invalid_grid_size() -> None:
    with pytest.raises(ValueError, match="grid_size must be positive"):
        extract_flexi_dual_grid(_cube_coordinates(), _cube_fields(), grid_size=0)


def test_extract_flexi_dual_grid_reuses_native_shape_contract_validation() -> None:
    bad_coordinates = _cube_coordinates()
    bad_coordinates[0, 0] = 1

    with pytest.raises(ValueError, match="batch index 0 only"):
        extract_flexi_dual_grid(bad_coordinates, _cube_fields(), grid_size=2)
