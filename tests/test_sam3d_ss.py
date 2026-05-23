import numpy as np
import mlx.core as mx
from safetensors.mlx import save_file

from mlx_spatial.sam3d_ss import (
    Sam3dSSDecoderConfig,
    downsample_sam3d_sparse_structure,
    extract_sam3d_ss_coords,
    load_sam3d_ss_decoder_tensors,
    prune_sam3d_sparse_structure,
    read_sam3d_ss_decoder_config,
    run_sam3d_ss_decoder,
)


def test_extract_sam3d_ss_coords_matches_official_argwhere_order():
    occupancy = np.zeros((1, 1, 2, 2, 2), dtype=np.float32)
    occupancy[0, 0, 0, 1, 1] = 1.0
    occupancy[0, 0, 1, 0, 1] = 2.0

    coords = extract_sam3d_ss_coords(occupancy)

    assert coords.tolist() == [[0, 0, 1, 1], [0, 1, 0, 1]]


def test_prune_sam3d_sparse_structure_removes_interior_voxel():
    coords = np.array([[0, x, y, z] for x in range(3) for y in range(3) for z in range(3)], dtype=np.int32)

    pruned = prune_sam3d_sparse_structure(coords, max_neighbor_axes_dist=1)

    assert pruned.shape == (26, 4)
    assert [0, 1, 1, 1] not in pruned.tolist()


def test_downsample_sam3d_sparse_structure_dedups_when_over_guard():
    coords = np.array([[0, i, 0, 0] for i in range(10)], dtype=np.int32)

    downsampled, factor = downsample_sam3d_sparse_structure(coords, max_coords=4, seed=7)

    assert factor == 2
    assert downsampled.shape == (4, 4)
    assert len({tuple(row) for row in downsampled.tolist()}) == 4


def test_read_sam3d_ss_decoder_config_accepts_active_yaml(tmp_path):
    path = tmp_path / "ss_decoder.yaml"
    path.write_text(
        """
_target_: sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae.SparseStructureDecoderTdfyWrapper
out_channels: 1
latent_channels: 8
num_res_blocks: 2
num_res_blocks_middle: 2
channels: [512, 128, 32]
reshape_input_to_cube: false
""".strip(),
        encoding="utf-8",
    )

    config = read_sam3d_ss_decoder_config(path)

    assert config.latent_channels == 8
    assert config.channels == (512, 128, 32)
    assert config.num_res_blocks == 2


def test_run_sam3d_ss_decoder_fixture_returns_occupancy_and_coords(tmp_path):
    config = Sam3dSSDecoderConfig(
        out_channels=1,
        latent_channels=1,
        num_res_blocks=0,
        channels=(2,),
        num_res_blocks_middle=0,
    )
    ckpt = tmp_path / "ss_decoder.safetensors"
    tensors = {
        "input_layer.weight": mx.zeros((2, 1, 3, 3, 3), dtype=mx.float32),
        "input_layer.bias": mx.array([0.0, 1.0], dtype=mx.float32),
        "out_layer.0.weight": mx.ones((2,), dtype=mx.float32),
        "out_layer.0.bias": mx.zeros((2,), dtype=mx.float32),
        "out_layer.2.weight": mx.zeros((1, 2, 3, 3, 3), dtype=mx.float32),
        "out_layer.2.bias": mx.ones((1,), dtype=mx.float32),
    }
    save_file(tensors, ckpt)

    output = run_sam3d_ss_decoder(
        np.zeros((1, 1, 2, 2, 2), dtype=np.float32),
        load_sam3d_ss_decoder_tensors(ckpt),
        config,
        prune_neighbor_axes_dist=0,
    )

    assert output.occupancy.shape == (1, 1, 2, 2, 2)
    assert output.coords_original.shape == (8, 4)
    assert output.coords.shape == (8, 4)
    assert output.downsample_factor == 1
    assert output.metadata["occupancy_min"] == 1.0
    assert output.metadata["occupancy_max"] == 1.0
    assert output.metadata["occupancy_positive_fraction"] == 1.0
