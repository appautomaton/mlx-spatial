import numpy as np

from mlx_spatial.sam3d_export import pack_sam3d_gaussian_rows
from mlx_spatial.sam3d_gaussian import (
    Sam3dGaussianDecoderConfig,
    decode_sam3d_gaussian_fields,
    sam3d_hammersley_perturbation,
)


def test_sam3d_hammersley_perturbation_matches_first_official_samples():
    config = Sam3dGaussianDecoderConfig(num_gaussians=4, voxel_size=2.0)

    perturbation = sam3d_hammersley_perturbation(config)

    expected_first = np.arctanh(np.array([-0.5, -0.5, -0.5], dtype=np.float32))
    expected_second = np.arctanh(np.array([-0.25, 0.0, -1.0 / 6.0], dtype=np.float32))
    assert perturbation.shape == (4, 3)
    assert np.allclose(perturbation[0], expected_first)
    assert np.allclose(perturbation[1], expected_second)


def test_decode_sam3d_gaussian_fields_packs_official_ply_ready_values():
    config = Sam3dGaussianDecoderConfig(
        resolution=4,
        num_gaussians=2,
        voxel_size=2.0,
        perturb_offset=False,
        minimum_kernel_size=0.0,
        scaling_bias=0.004,
        opacity_bias=0.1,
    )
    coords = np.array([[0, 1, 2, 3]], dtype=np.int32)
    raw = np.zeros((1, config.output_channels), dtype=np.float32)
    # features_dc starts after xyz: 2 * 3 channels.
    raw[:, 6:12] = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    # rotation starts after xyz, features, scaling: 18 channels.
    raw[:, 18:26] = np.array([[0.0, 0.1, 0.2, 0.3, 0.0, -0.1, -0.2, -0.3]], dtype=np.float32)

    fields = decode_sam3d_gaussian_fields(coords, raw, config=config)

    assert fields.xyz.shape == (2, 3)
    assert np.allclose(fields.xyz[0], np.array([-0.125, 0.125, 0.375], dtype=np.float32))
    assert fields.features_dc.shape == (2, 1, 3)
    assert np.allclose(fields.features_dc[0, 0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(fields.rotation[0], np.array([1.0, 0.01, 0.02, 0.03], dtype=np.float32))
    assert fields.metadata["gaussian_count"] == 2
    rows = pack_sam3d_gaussian_rows(
        xyz=fields.xyz,
        features_dc=fields.features_dc,
        opacity=fields.opacity,
        scale=fields.scale,
        rotation=fields.rotation,
    )
    assert rows.shape == (2, 17)


def test_decode_sam3d_gaussian_fields_rejects_wrong_feature_width():
    config = Sam3dGaussianDecoderConfig(num_gaussians=2)

    try:
        decode_sam3d_gaussian_fields(np.zeros((1, 4), dtype=np.int32), np.zeros((1, 4), dtype=np.float32), config=config)
    except ValueError as error:
        assert "raw feature width" in str(error)
    else:
        raise AssertionError("expected gaussian raw feature width mismatch to fail")
