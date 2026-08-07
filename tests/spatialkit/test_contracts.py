from __future__ import annotations

import numpy as np
import pytest

from mlx_spatial.coordinate_systems import GLTF_Y_UP, TRELLIS_Z_UP, vertices_to_gltf_y_up
from mlx_spatial.spatialkit import (
    load_pixal3d_decoded_npz,
    validate_pixal3d_decoded,
    validate_pixal3d_shape_fields,
    validate_pixal3d_texture_attributes,
)
from mlx_spatial.spatialkit.export import _resolve_source_coordinate_system


def _shape_coordinates() -> np.ndarray:
    return np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.int32)


def _shape_fields() -> np.ndarray:
    return np.ones((2, 7), dtype=np.float32)


def _texture_attributes() -> np.ndarray:
    return np.ones((2, 6), dtype=np.float32)


def test_shape_contract_accepts_pixal3d_decoded_shape_arrays() -> None:
    contract = validate_pixal3d_shape_fields(_shape_coordinates(), _shape_fields())

    assert contract["token_count"] == 2
    assert contract["coordinates_shape"] == (2, 4)
    assert contract["fields_shape"] == (2, 7)
    assert contract["coordinates_dtype"] == "int32"
    assert contract["fields_dtype"] == "float32"


def test_texture_contract_accepts_pixal3d_decoded_texture_arrays() -> None:
    contract = validate_pixal3d_texture_attributes(_shape_coordinates(), _texture_attributes())

    assert contract["token_count"] == 2
    assert contract["coordinates_shape"] == (2, 4)
    assert contract["attributes_shape"] == (2, 6)
    assert contract["coordinates_dtype"] == "int32"
    assert contract["attributes_dtype"] == "float32"


def test_contract_rejects_wrong_coordinate_dtype() -> None:
    with pytest.raises(ValueError, match="shape coordinates must have dtype int32"):
        validate_pixal3d_shape_fields(_shape_coordinates().astype(np.int64), _shape_fields())


def test_contract_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError, match="shape coordinates must have rank 2"):
        validate_pixal3d_shape_fields(np.zeros((1, 2, 4), dtype=np.int32), _shape_fields())


def test_contract_rejects_wrong_channel_count() -> None:
    with pytest.raises(ValueError, match=r"shape fields must have shape \(n, 7\)"):
        validate_pixal3d_shape_fields(_shape_coordinates(), np.ones((2, 6), dtype=np.float32))


def test_contract_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="shape coordinates must contain at least one token"):
        validate_pixal3d_shape_fields(np.zeros((0, 4), dtype=np.int32), np.zeros((0, 7), dtype=np.float32))


def test_contract_rejects_token_mismatch() -> None:
    with pytest.raises(ValueError, match="coordinates/fields token mismatch"):
        validate_pixal3d_shape_fields(_shape_coordinates(), np.ones((3, 7), dtype=np.float32))


def test_contract_rejects_unsupported_batch_index() -> None:
    coordinates = _shape_coordinates()
    coordinates[1, 0] = 1

    with pytest.raises(ValueError, match="batch index 0 only"):
        validate_pixal3d_shape_fields(coordinates, _shape_fields())


def test_texture_contract_rejects_wrong_attribute_width() -> None:
    with pytest.raises(ValueError, match=r"texture attributes must have shape \(n, 6\)"):
        validate_pixal3d_texture_attributes(_shape_coordinates(), np.ones((2, 5), dtype=np.float32))


def test_validate_pixal3d_decoded_returns_shape_and_texture_contracts() -> None:
    contracts = validate_pixal3d_decoded(
        _shape_coordinates(),
        _shape_fields(),
        _shape_coordinates(),
        _texture_attributes(),
    )

    assert contracts["shape"]["fields_shape"] == (2, 7)
    assert contracts["texture"]["attributes_shape"] == (2, 6)


def test_load_pixal3d_decoded_npz_loads_and_validates_arrays(tmp_path) -> None:
    shape_path = tmp_path / "shape_decoder_fields.npz"
    texture_path = tmp_path / "texture_decoder_pbr.npz"
    np.savez(shape_path, coordinates=_shape_coordinates(), fields=_shape_fields())
    np.savez(texture_path, coordinates=_shape_coordinates(), attributes=_texture_attributes())

    decoded = load_pixal3d_decoded_npz(shape_path, texture_path)

    assert decoded.shape_coordinates.shape == (2, 4)
    assert decoded.shape_fields.shape == (2, 7)
    assert decoded.texture_coordinates.shape == (2, 4)
    assert decoded.texture_attributes.shape == (2, 6)
    assert decoded.contracts["shape"]["token_count"] == 2


def test_load_pixal3d_decoded_npz_reports_missing_required_array(tmp_path) -> None:
    shape_path = tmp_path / "shape_decoder_fields.npz"
    texture_path = tmp_path / "texture_decoder_pbr.npz"
    np.savez(shape_path, coordinates=_shape_coordinates())
    np.savez(texture_path, coordinates=_shape_coordinates(), attributes=_texture_attributes())

    with pytest.raises(ValueError, match="missing required array 'fields'"):
        load_pixal3d_decoded_npz(shape_path, texture_path)


def test_vertices_to_gltf_y_up_applies_trellis_reference_rotation() -> None:
    vertices = np.array([[1.0, 2.0, 3.0], [-4.0, -5.0, -6.0]], dtype=np.float32)

    converted = vertices_to_gltf_y_up(vertices, source_coordinate_system=TRELLIS_Z_UP)

    np.testing.assert_array_equal(
        converted,
        np.array([[1.0, 3.0, -2.0], [-4.0, -6.0, 5.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        vertices_to_gltf_y_up(vertices, source_coordinate_system=GLTF_Y_UP),
        vertices,
    )


def test_source_coordinate_system_auto_detects_trellis_metadata() -> None:
    metadata = {"source_model": "trellis.2"}

    assert _resolve_source_coordinate_system("auto", metadata, metadata) == TRELLIS_Z_UP
    assert _resolve_source_coordinate_system("auto", {}, {}) == GLTF_Y_UP
    assert _resolve_source_coordinate_system(GLTF_Y_UP, metadata, metadata) == GLTF_Y_UP
