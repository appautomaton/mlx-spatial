from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from safetensors.numpy import save_file

from mlx_spatial.lito import LitoInferencePipeline
from mlx_spatial.lito_inference import LITO_REAL_TENSOR_SENTINELS


def test_lito_real_backend_import_has_no_optional_runtime_side_effects():
    code = """
import importlib
import json
import sys

for name in list(sys.modules):
    if name == "torch" or name.startswith(("torch.", "lito.", "vendors", "mlx_spatial.vendors")):
        sys.modules.pop(name, None)

module = importlib.import_module("mlx_spatial.lito_real_backend")
print(json.dumps({
    "module": module.__name__,
    "forbidden": sorted(
        name for name in sys.modules
        if name == "torch" or name.startswith(("torch.", "lito.", "vendors", "mlx_spatial.vendors"))
    ),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload == {"module": "mlx_spatial.lito_real_backend", "forbidden": []}


def test_create_lito_real_backend_rejects_cuda_request_before_architecture_load(tmp_path):
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    weights_root = _write_fake_lito_weights(tmp_path / "weights")
    config = backend.LitoRealBackendConfig(
        weights_root=weights_root,
        asset_summary=_asset_summary(weights_root),
        memory_profile="safe",
        allow_cuda=True,
    )

    with pytest.raises(backend.LitoBackendUnavailable, match="CUDA"):
        backend.create_lito_real_backend(config)


def test_inspect_lito_real_architecture_uses_safetensor_headers(tmp_path):
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    weights_root = _write_fake_lito_architecture_weights(tmp_path / "weights")

    inventory = backend.inspect_lito_real_architecture(weights_root)

    assert inventory.checkpoint_key_counts == {
        "image_to_3d/lito_dit_rgba.safetensors": 11,
        "tokenizer/lito_new.safetensors": 19,
    }
    assert inventory.prefix_counts["image_to_3d/lito_dit_rgba.safetensors"] == {"velocity_estimator_ema": 11}
    assert inventory.dit["num_blocks"] == 2
    assert inventory.dit["dim_latent"] == 2
    assert inventory.dit["dim_hidden"] == 8
    assert inventory.dit["dim_cond_token"] == 6
    assert inventory.dit["remaps"]["t_proj.0"] == "t_proj_linear1"
    assert inventory.gaussian_decoder["num_blocks"] == 2
    assert inventory.gaussian_decoder["num_self_attn"] == 2
    assert inventory.gaussian_decoder["gs_expansion_ratio"] == 64
    assert inventory.gaussian_decoder["rgb_sh_degree"] == 3
    assert inventory.gaussian_decoder["shape_outputs"] == ("xyz_w", "quaternion_prenorm", "scaling_logit")
    assert inventory.voxel_decoder["num_blocks"] == 2
    assert inventory.voxel_decoder["init_query_shape"] == (2, 2, 2, 8)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "weights/lito-mlx/tokenizer/lito_new.safetensors").is_file(),
    reason="LiTo weights absent",
)
def test_inspect_lito_real_architecture_reports_expected_real_lito_shapes():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    root = Path(__file__).resolve().parents[1] / "weights/lito-mlx"

    inventory = backend.inspect_lito_real_architecture(root)

    assert inventory.checkpoint_key_counts == {
        "image_to_3d/lito_dit_rgba.safetensors": 2793,
        "tokenizer/lito_new.safetensors": 1108,
    }
    assert inventory.prefix_counts["image_to_3d/lito_dit_rgba.safetensors"] == {
        "pretrained_tokenizer": 1108,
        "velocity_estimator": 669,
        "velocity_estimator_ema": 669,
        "patch_encoder": 347,
    }
    assert inventory.prefix_counts["tokenizer/lito_new.safetensors"]["gs_decoder"] == 273
    assert inventory.dit["num_latent"] == 8192
    assert inventory.dit["dim_latent"] == 32
    assert inventory.dit["dim_hidden"] == 1152
    assert inventory.dit["dim_cond_token"] == 2048
    assert inventory.dit["num_blocks"] == 28
    assert inventory.dit["num_heads"] == 16
    assert inventory.gaussian_decoder["perceiver_dim"] == 512
    assert inventory.gaussian_decoder["dim_latent"] == 32
    assert inventory.gaussian_decoder["num_blocks"] == 6
    assert inventory.gaussian_decoder["num_self_attn"] == 2
    assert inventory.gaussian_decoder["rgb_sh_degree"] == 3
    assert inventory.voxel_decoder["init_query_shape"] == (16, 16, 16, 512)


def test_load_lito_dit_weight_arrays_strips_prefix_and_remaps(tmp_path):
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    root = _write_fake_lito_weight_loader_tensors(tmp_path / "weights")

    arrays = backend.load_lito_dit_weight_arrays(
        root,
        names=(
            "z_proj.weight",
            "t_proj_linear1.weight",
            "t_proj_linear2.bias",
            "t0_proj_linear.weight",
            "final_layer.adaLN_linear1.weight",
            "final_layer.adaLN_linear2.bias",
        ),
        dtype=np.float16,
    )

    assert set(arrays) == {
        "z_proj.weight",
        "t_proj_linear1.weight",
        "t_proj_linear2.bias",
        "t0_proj_linear.weight",
        "final_layer.adaLN_linear1.weight",
        "final_layer.adaLN_linear2.bias",
    }
    assert arrays["z_proj.weight"].dtype == np.float16
    assert arrays["z_proj.weight"].tolist() == [[1.0, 2.0]]
    assert arrays["t_proj_linear1.weight"].tolist() == [[3.0, 4.0]]
    assert arrays["t_proj_linear2.bias"].tolist() == [5.0, 6.0]
    assert arrays["t0_proj_linear.weight"].tolist() == [[7.0, 8.0]]
    assert arrays["final_layer.adaLN_linear1.weight"].tolist() == [[9.0, 10.0]]
    assert arrays["final_layer.adaLN_linear2.bias"].tolist() == [11.0, 12.0]


def test_load_lito_patch_encoder_weight_arrays_strips_prefix(tmp_path):
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    root = _write_fake_lito_weight_loader_tensors(tmp_path / "weights")

    arrays = backend.load_lito_patch_encoder_weight_arrays(
        root,
        names=(
            "dinov2_model.model.cls_token",
            "dinov2_model.model.patch_embed.proj.weight",
            "learnable_model.weight",
            "learnable_paddings",
        ),
        dtype=np.float16,
    )

    assert set(arrays) == {
        "dinov2_model.model.cls_token",
        "dinov2_model.model.patch_embed.proj.weight",
        "learnable_model.weight",
        "learnable_paddings",
    }
    assert arrays["dinov2_model.model.cls_token"].shape == (1, 1, 4)
    assert arrays["learnable_model.weight"].shape == (4, 4, 14, 14)
    assert arrays["learnable_paddings"].dtype == np.float16


def test_run_lito_patch_encoder_condition_tokens_runs_fake_weights():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    weights = _fake_patch_encoder_weights()
    config = backend.LitoPatchEncoderConfig(
        input_size=28,
        patch_size=14,
        embed_dim=4,
        num_heads=2,
        num_blocks=1,
        register_count=1,
        attention_chunk_size=3,
    )
    rgba = np.ones((28, 28, 4), dtype=np.float32)
    rgba[..., :3] = np.array([0.25, 0.5, 0.75], dtype=np.float32)

    tokens = backend.run_lito_patch_encoder_condition_tokens(rgba, weights, block_indices=[0], config=config)

    assert tuple(tokens.shape) == (1, 6, 8)
    assert np.isfinite(np.asarray(tokens)).all()


def test_load_lito_gaussian_decoder_weight_arrays_remaps_and_splits(tmp_path):
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    root = _write_fake_lito_weight_loader_tensors(tmp_path / "weights")

    arrays = backend.load_lito_gaussian_decoder_weight_arrays(root)

    assert "point_linear.weight" in arrays
    assert "point_mlp.norms.0.weight" in arrays
    assert "point_mlp.mlps.0.w1.weight" in arrays
    assert "point_mlp.mlps.0.w2.weight" in arrays
    assert "point_mlp.mlps.0.w3.weight" in arrays
    assert "point_mlp.final_layer.linear.bias" in arrays
    assert "perceiver.blocks.0.ca_mlp.w1.weight" in arrays
    assert "perceiver.blocks.0.ca_mlp.w2.weight" in arrays
    assert "gs_output_shape_mlp.final_layer.linear.weight" in arrays
    assert "gs_output_color_mlp.final_layer.linear.weight" in arrays
    assert "voxel_decoder.input_linear.weight" not in arrays
    assert arrays["point_mlp.mlps.0.w1.weight"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert arrays["point_mlp.mlps.0.w2.weight"].tolist() == [[5.0, 6.0], [7.0, 8.0]]
    assert arrays["perceiver.blocks.0.ca_mlp.w1.weight"].shape == (2, 2)
    assert arrays["perceiver.blocks.0.ca_mlp.w2.weight"].shape == (2, 2)


def test_direct_backend_exposes_local_weight_loader_and_decode_helpers(tmp_path):
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    root = _write_fake_lito_weight_loader_tensors(tmp_path / "weights")
    instance = backend.DirectMlxLitoBackend(
        backend.LitoRealBackendConfig(
            weights_root=root,
            asset_summary=None,
            memory_profile="safe",
        ),
        architecture=None,
    )

    dit = instance.load_dit_weight_arrays(names=("z_proj.weight",))
    gs = instance.load_gaussian_decoder_weight_arrays(names=("point_linear.weight",))
    decoded = instance.decode_gaussian_outputs(
        np.zeros((1, 64 * 10), dtype=np.float32),
        np.zeros((1, 64 * 49), dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
    )

    assert dit["z_proj.weight"].shape == (1, 2)
    assert gs["point_linear.weight"].shape == (2, 2)
    assert decoded["xyz_w"].shape == (1, 64, 3)


def test_run_lito_gaussian_output_heads_and_decode_with_explicit_init_coords():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    weights = _fake_output_head_weights()
    query_latent = np.zeros((2, 512), dtype=np.float32)
    init_coord = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)

    shape_out, color_out = backend.run_lito_gaussian_output_heads(query_latent, weights)
    decoded = backend.decode_lito_gaussian_query_latents(query_latent, init_coord, weights)

    assert tuple(shape_out.shape) == (2, 640)
    assert tuple(color_out.shape) == (2, 3136)
    assert decoded["xyz_w"].shape == (2, 64, 3)
    assert decoded["opacity"].shape == (2, 64, 1)
    assert np.allclose(decoded["xyz_w"], init_coord[:, None, :])
    assert np.allclose(decoded["opacity"], 1.0 / (1.0 + np.exp(-0.1)))


def test_encode_lito_gaussian_query_points_and_decode_with_explicit_coords():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    weights = _fake_query_point_weights()
    init_coord = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)

    query = backend.encode_lito_gaussian_query_points(init_coord, weights)
    decoded = backend.decode_lito_gaussian_query_points(init_coord, weights)

    assert tuple(query.shape) == (2, 512)
    assert decoded["xyz_w"].shape == (2, 64, 3)
    assert np.allclose(decoded["xyz_w"], init_coord[:, None, :])


def test_run_lito_gaussian_perceiver_block0_cross_only_follows_cross_mlp_order():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    weights = _fake_perceiver_block0_cross_only_weights()
    query = np.array([[1.0, 2.0, 4.0, 8.0], [-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
    latent = np.array([[0.5, -0.5], [1.0, 2.0], [0.0, 3.0]], dtype=np.float32)

    result = backend.run_lito_gaussian_perceiver_block0_cross_only(
        query,
        latent,
        weights,
        q_seq_lens=[2],
        kv_seq_lens=[3],
        num_heads=2,
    )

    cross_bias = weights["perceiver.blocks.0.ca_layer.linear_out.bias"]
    after_cross = query + cross_bias[None, :]
    centered = after_cross - after_cross.mean(axis=-1, keepdims=True)
    normalized = centered / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + 1e-6)
    expected = after_cross + (normalized * (1.0 / (1.0 + np.exp(-normalized)))) * normalized
    assert np.allclose(np.asarray(result), expected, atol=5e-3)


def test_run_lito_gaussian_perceiver_block0_cross_only_validates_seq_lens_and_self_attn_boundary():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    weights = _fake_perceiver_block0_cross_only_weights()
    query = np.zeros((2, 4), dtype=np.float32)
    latent = np.zeros((3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="q_seq_lens must sum"):
        backend.run_lito_gaussian_perceiver_block0_cross_only(
            query,
            latent,
            weights,
            q_seq_lens=[1],
            kv_seq_lens=[3],
            num_heads=2,
        )

    with pytest.raises(backend.LitoBackendUnavailable, match="localized_voxel self-attention"):
        backend.run_lito_gaussian_perceiver_block0_cross_only(
            query,
            latent,
            weights,
            q_seq_lens=[2],
            kv_seq_lens=[3],
            num_heads=2,
            include_localized_self_attention=True,
        )


def test_build_lito_local_voxel_info_matches_packed_point_contract():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    coords = np.array(
        [
            [0.30, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.10, 0.00, 0.00],
            [0.00, 0.00, 0.00],
        ],
        dtype=np.float32,
    )

    info = backend.build_lito_local_voxel_info(coords, [3, 1], cell_width=0.25)

    assert info["total_cells"] == 3
    assert np.asarray(info["cell_counts"]).tolist() == [2, 1, 1]
    assert np.asarray(info["forward_idxs"]).tolist() == [1, 2, 0, 3]
    assert np.asarray(info["backward_idxs"]).tolist() == [2, 0, 1, 3]
    assert np.asarray(info["cu_seq_lens"][0]).tolist() == [0, 2, 3, 4]
    assert info["max_seq_lens"] == [2]
    assert info["chunk_start_idxs"] == [0, 4]


def test_run_lito_gaussian_perceiver_block0_with_local_voxel_self_attention_runs_fake_weights():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    weights = _fake_perceiver_block0_full_weights()
    query = np.array([[1.0, 2.0, 4.0, 8.0], [-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
    latent = np.array([[0.5, -0.5], [1.0, 2.0], [0.0, 3.0]], dtype=np.float32)
    init_coord = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)

    result = backend.run_lito_gaussian_perceiver_block0_with_local_voxel_self_attention(
        query,
        latent,
        init_coord,
        weights,
        q_seq_lens=[2],
        kv_seq_lens=[3],
        num_heads=2,
        self_cell_width=0.25,
    )

    assert tuple(result.shape) == (2, 4)
    assert np.isfinite(np.asarray(result)).all()
    assert not np.allclose(np.asarray(result), query)


def test_run_lito_gaussian_perceiver_all_blocks_with_local_voxel_self_attention_runs_fake_weights():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    weights = _fake_perceiver_blocks_full_weights((0, 1))
    query = np.array([[1.0, 2.0, 4.0, 8.0], [-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
    latent = np.array([[0.5, -0.5], [1.0, 2.0], [0.0, 3.0]], dtype=np.float32)
    init_coord = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)

    result = backend.run_lito_gaussian_perceiver_all_blocks_with_local_voxel_self_attention(
        query,
        latent,
        init_coord,
        weights,
        q_seq_lens=[2],
        kv_seq_lens=[3],
        block_indices=[0, 1],
        num_heads=2,
        self_cell_width=0.25,
    )

    assert tuple(result.shape) == (2, 4)
    assert np.isfinite(np.asarray(result)).all()
    assert not np.allclose(np.asarray(result), query)


def test_run_lito_dit_velocity_runs_fake_weights():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    weights = _fake_dit_weights()
    latent = np.zeros((1, 2, 32), dtype=np.float32)
    cond = np.zeros((1, 3, 2048), dtype=np.float32)

    velocity = backend.run_lito_dit_velocity(
        latent,
        np.array([0.5], dtype=np.float32),
        cond,
        weights,
        block_indices=[0],
        num_heads=2,
    )

    assert tuple(velocity.shape) == (1, 2, 32)
    assert np.isfinite(np.asarray(velocity)).all()


def test_sample_lito_dit_latents_runs_fake_weights_with_initial_latent():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    weights = _fake_dit_weights()
    latent = np.zeros((1, 2, 32), dtype=np.float32)
    cond = np.zeros((1, 3, 2048), dtype=np.float32)

    sampled = backend.sample_lito_dit_latents(
        cond,
        weights,
        initial_latent=latent,
        num_steps=2,
        method="euler",
        block_indices=[0],
        num_heads=2,
        latent_mean=0.0,
        latent_std=1.0,
    )

    assert tuple(sampled.shape) == (1, 2, 32)
    assert np.isfinite(np.asarray(sampled)).all()


def test_occ_grid_to_lito_init_coord_matches_upstream_axis_contract():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    occ = np.zeros((2, 1, 4, 4, 4), dtype=np.float32)
    occ[0, 0, 3, 1, 2] = 0.75
    occ[1, 0, 0, 0, 1] = 0.5

    converted = backend.occ_grid_to_lito_init_coord(occ, threshold=0.5)

    assert converted["q_seq_lens"] == [1, 1]
    assert converted["init_coord"].shape == (2, 3)
    assert np.allclose(converted["init_coord"][0], [0.25, -0.25, 0.75])
    assert np.allclose(converted["init_coord"][1], [-0.25, -0.75, -0.75])
    assert converted["occ_bool_grid"].dtype == np.bool_


def test_occ_grid_to_lito_init_coord_can_cap_cells_by_score_per_batch():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    occ = np.zeros((1, 1, 4, 4, 4), dtype=np.float32)
    occ[0, 0, 0, 0, 0] = 0.1
    occ[0, 0, 1, 0, 0] = 0.9
    occ[0, 0, 2, 0, 0] = 0.5

    converted = backend.occ_grid_to_lito_init_coord(occ, threshold=0.0, max_cells_per_batch=2)

    assert converted["q_seq_lens"] == [2]
    assert converted["init_coord"].shape == (2, 3)
    assert np.allclose(converted["init_coord"][0], [-0.75, -0.75, -0.25])
    assert np.allclose(converted["init_coord"][1], [-0.75, -0.75, 0.25])


def test_occ_grid_to_lito_init_coord_no_cap_preserves_all_occupied_cells():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    occ = np.zeros((1, 1, 4, 4, 4), dtype=np.float32)
    occ[0, 0, 0, 0, 0] = 0.6
    occ[0, 0, 1, 0, 0] = 0.7
    occ[0, 0, 2, 0, 0] = 0.8

    converted = backend.occ_grid_to_lito_init_coord(occ, threshold=0.5, max_cells_per_batch=None)

    assert converted["q_seq_lens"] == [3]
    assert converted["init_coord"].shape == (3, 3)


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ("profile", 512),
        (None, None),
        ("none", None),
        (7, 7),
    ],
)
def test_decode_sampled_latents_resolves_init_coord_cap_override(tmp_path, monkeypatch, override, expected):
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    observed: dict[str, object] = {}
    config = backend.LitoRealBackendConfig(
        weights_root=tmp_path / "weights",
        asset_summary=None,
        memory_profile="safe",
        max_init_coords_per_batch=override,
    )
    instance = backend.DirectMlxLitoBackend(config, architecture=None)

    monkeypatch.setattr(backend, "_resolve_lito_trellis_root", lambda config: tmp_path / "trellis")

    def fake_init_coords(latent_tokens, *, trellis_root, max_cells_per_batch, **kwargs):
        observed["trellis_root"] = trellis_root
        observed["max_cells_per_batch"] = max_cells_per_batch
        return {
            "init_coord": np.zeros((2, 3), dtype=np.float32),
            "q_seq_lens": [2],
        }

    def fake_gaussian_decode(latent_packed, init_coord, *, q_seq_lens, kv_seq_lens):
        observed["kv_seq_lens"] = kv_seq_lens
        return {"xyz_w": init_coord}

    monkeypatch.setattr(instance, "decode_init_coords_from_latents", fake_init_coords)
    monkeypatch.setattr(instance, "decode_gaussian_perceiver_all_blocks", fake_gaussian_decode)

    result = instance.decode_sampled_latents_to_gaussians(np.zeros((1, 3, 32), dtype=np.float32))

    assert observed["trellis_root"] == tmp_path / "trellis"
    assert observed["max_cells_per_batch"] == expected
    assert observed["kv_seq_lens"] == [3]
    assert result["xyz_w"].shape == (2, 3)


def test_run_lito_voxel_decoder_lowres_latent_runs_fake_weights():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    weights = _fake_voxel_decoder_weights()
    latent_tokens = np.zeros((1, 3, 32), dtype=np.float32)

    ss_latent = backend.run_lito_voxel_decoder_lowres_latent(
        latent_tokens,
        weights,
        block_indices=[0],
        num_heads=2,
    )

    assert tuple(ss_latent.shape) == (1, 8, 2, 2, 2)
    assert np.isfinite(np.asarray(ss_latent)).all()
    assert np.allclose(np.asarray(ss_latent)[:, :, 0, 0, 0], np.arange(8, dtype=np.float32)[None, :])


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "weights/lito-mlx/tokenizer/lito_new.safetensors").is_file(),
    reason="LiTo weights absent",
)
def test_real_gaussian_output_heads_run_from_loaded_checkpoint_weights():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    root = Path(__file__).resolve().parents[1] / "weights/lito-mlx"
    names = tuple(
        f"{prefix}.{suffix}"
        for prefix in ("gs_output_shape_mlp", "gs_output_color_mlp")
        for suffix in (
            "norms.0.weight",
            "norms.0.bias",
            "mlps.0.w1.weight",
            "mlps.0.w2.weight",
            "mlps.0.w3.weight",
            "final_layer.norm_final.weight",
            "final_layer.norm_final.bias",
            "final_layer.linear.weight",
            "final_layer.linear.bias",
        )
    )
    weights = backend.load_lito_gaussian_decoder_weight_arrays(root, names=names)
    query_latent = np.zeros((1, 512), dtype=np.float32)
    init_coord = np.zeros((1, 3), dtype=np.float32)

    decoded = backend.decode_lito_gaussian_query_latents(query_latent, init_coord, weights)

    assert decoded["xyz_w"].shape == (1, 64, 3)
    assert decoded["scaling"].shape == (1, 64, 3)
    assert decoded["quaternion"].shape == (1, 64, 4)
    assert decoded["opacity"].shape == (1, 64, 1)
    assert decoded["rgb_sh"].shape == (1, 64, 16, 3)
    assert np.isfinite(decoded["xyz_w"]).all()
    assert np.isfinite(decoded["rgb_sh"]).all()


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "weights/lito-mlx/image_to_3d/lito_dit_rgba.safetensors").is_file(),
    reason="LiTo weights absent",
)
def test_real_dit_velocity_block0_runs_from_loaded_checkpoint_weights():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    root = Path(__file__).resolve().parents[1] / "weights/lito-mlx"
    weights = backend.load_lito_dit_weight_arrays(root)
    latent = np.zeros((1, 2, 32), dtype=np.float32)
    cond = np.zeros((1, 3, 2048), dtype=np.float32)

    velocity = backend.run_lito_dit_velocity(
        latent,
        np.array([0.5], dtype=np.float32),
        cond,
        weights,
        block_indices=[0],
    )

    assert tuple(velocity.shape) == (1, 2, 32)
    assert np.isfinite(np.asarray(velocity)).all()


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "weights/lito-mlx/image_to_3d/lito_dit_rgba.safetensors").is_file(),
    reason="LiTo weights absent",
)
def test_real_dit_sampler_block0_runs_from_loaded_checkpoint_weights():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    root = Path(__file__).resolve().parents[1] / "weights/lito-mlx"
    weights = backend.load_lito_dit_weight_arrays(root)
    latent = np.zeros((1, 2, 32), dtype=np.float32)
    cond = np.zeros((1, 3, 2048), dtype=np.float32)

    sampled = backend.sample_lito_dit_latents(
        cond,
        weights,
        initial_latent=latent,
        num_steps=2,
        method="euler",
        block_indices=[0],
    )

    assert tuple(sampled.shape) == (1, 2, 32)
    assert np.isfinite(np.asarray(sampled)).all()


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "weights/lito-mlx/tokenizer/lito_new.safetensors").is_file(),
    reason="LiTo weights absent",
)
def test_real_gaussian_query_point_stem_runs_from_loaded_checkpoint_weights():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    root = Path(__file__).resolve().parents[1] / "weights/lito-mlx"
    weights = backend.load_lito_gaussian_decoder_weight_arrays(root)
    init_coord = np.zeros((1, 3), dtype=np.float32)

    query = backend.encode_lito_gaussian_query_points(init_coord, weights)
    decoded = backend.decode_lito_gaussian_query_points(init_coord, weights)

    assert tuple(query.shape) == (1, 512)
    assert decoded["xyz_w"].shape == (1, 64, 3)
    assert decoded["rgb_sh"].shape == (1, 64, 16, 3)
    assert np.isfinite(np.asarray(query)).all()
    assert np.isfinite(decoded["xyz_w"]).all()


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "weights/lito-mlx/tokenizer/lito_new.safetensors").is_file(),
    reason="LiTo weights absent",
)
def test_real_gaussian_perceiver_block0_cross_only_runs_from_loaded_checkpoint_weights():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    root = Path(__file__).resolve().parents[1] / "weights/lito-mlx"
    weights = backend.load_lito_gaussian_decoder_weight_arrays(root)
    init_coord = np.zeros((2, 3), dtype=np.float32)
    latent_tokens = np.zeros((4, 32), dtype=np.float32)

    query = backend.encode_lito_gaussian_query_points(init_coord, weights)
    cross_query = backend.run_lito_gaussian_perceiver_block0_cross_only(
        query,
        latent_tokens,
        weights,
        q_seq_lens=[2],
        kv_seq_lens=[4],
    )
    decoded = backend.decode_lito_gaussian_query_latents(cross_query, init_coord, weights)

    assert tuple(cross_query.shape) == (2, 512)
    assert decoded["xyz_w"].shape == (2, 64, 3)
    assert decoded["rgb_sh"].shape == (2, 64, 16, 3)
    assert np.isfinite(np.asarray(cross_query)).all()
    assert np.isfinite(decoded["xyz_w"]).all()


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "weights/lito-mlx/tokenizer/lito_new.safetensors").is_file(),
    reason="LiTo weights absent",
)
def test_real_gaussian_perceiver_block0_local_voxel_self_attention_runs_from_loaded_checkpoint_weights():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    root = Path(__file__).resolve().parents[1] / "weights/lito-mlx"
    weights = backend.load_lito_gaussian_decoder_weight_arrays(root)
    init_coord = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)
    latent_tokens = np.zeros((4, 32), dtype=np.float32)

    query = backend.encode_lito_gaussian_query_points(init_coord, weights)
    decoded_query = backend.run_lito_gaussian_perceiver_block0_with_local_voxel_self_attention(
        query,
        latent_tokens,
        init_coord,
        weights,
        q_seq_lens=[2],
        kv_seq_lens=[4],
    )
    decoded = backend.decode_lito_gaussian_query_latents(decoded_query, init_coord, weights)

    assert tuple(decoded_query.shape) == (2, 512)
    assert decoded["xyz_w"].shape == (2, 64, 3)
    assert decoded["rgb_sh"].shape == (2, 64, 16, 3)
    assert np.isfinite(np.asarray(decoded_query)).all()
    assert np.isfinite(decoded["xyz_w"]).all()


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "weights/lito-mlx/tokenizer/lito_new.safetensors").is_file(),
    reason="LiTo weights absent",
)
def test_real_gaussian_perceiver_all_blocks_local_voxel_self_attention_runs_from_loaded_checkpoint_weights():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    root = Path(__file__).resolve().parents[1] / "weights/lito-mlx"
    weights = backend.load_lito_gaussian_decoder_weight_arrays(root)
    init_coord = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)
    latent_tokens = np.zeros((4, 32), dtype=np.float32)

    decoded = backend.decode_lito_gaussian_perceiver_all_blocks(
        latent_tokens,
        init_coord,
        weights,
        q_seq_lens=[2],
        kv_seq_lens=[4],
    )

    assert decoded["xyz_w"].shape == (2, 64, 3)
    assert decoded["scaling"].shape == (2, 64, 3)
    assert decoded["quaternion"].shape == (2, 64, 4)
    assert decoded["opacity"].shape == (2, 64, 1)
    assert decoded["rgb_sh"].shape == (2, 64, 16, 3)
    assert np.isfinite(decoded["xyz_w"]).all()
    assert np.isfinite(decoded["scaling"]).all()
    assert np.isfinite(decoded["rgb_sh"]).all()


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "weights/lito-mlx/tokenizer/lito_new.safetensors").is_file(),
    reason="LiTo weights absent",
)
def test_real_voxel_decoder_lowres_latent_runs_from_loaded_checkpoint_weights():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    root = Path(__file__).resolve().parents[1] / "weights/lito-mlx"
    weights = backend.load_lito_voxel_decoder_weight_arrays(root)
    latent_tokens = np.zeros((1, 2, 32), dtype=np.float32)

    ss_latent = backend.run_lito_voxel_decoder_lowres_latent(
        latent_tokens,
        weights,
        block_indices=[0],
    )

    assert tuple(ss_latent.shape) == (1, 8, 16, 16, 16)
    assert np.isfinite(np.asarray(ss_latent)).all()


@pytest.mark.skipif(
    not (
        Path(__file__).resolve().parents[1]
        / "weights/trellis2/microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"
    ).is_file(),
    reason="TRELLIS sparse-structure decoder weights absent",
)
def test_real_trellis_sparse_structure_decoder_logits_run_from_local_mlx_weights():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    root = Path(__file__).resolve().parents[1] / "weights/trellis2/microsoft/TRELLIS-image-large"
    ss_latent = np.zeros((1, 8, 16, 16, 16), dtype=np.float32)

    logits = backend.decode_lito_trellis_sparse_structure_logits(ss_latent, trellis_root=root)
    converted = backend.occ_grid_to_lito_init_coord(logits, threshold=0.0)

    assert tuple(logits.shape) == (1, 1, 64, 64, 64)
    assert np.isfinite(np.asarray(logits)).all()
    assert converted["occ_bool_grid"].shape == (1, 1, 64, 64, 64)
    assert len(converted["q_seq_lens"]) == 1
    assert converted["init_coord"].shape[1] == 3


@pytest.mark.skipif(
    not (
        Path(__file__).resolve().parents[1] / "weights/lito-mlx/tokenizer/lito_new.safetensors"
    ).is_file()
    or not (
        Path(__file__).resolve().parents[1]
        / "weights/trellis2/microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"
    ).is_file(),
    reason="LiTo or TRELLIS sparse-structure decoder weights absent",
)
def test_real_init_coord_generation_from_latents_runs_with_local_mlx_weights():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    repo = Path(__file__).resolve().parents[1]
    lito_root = repo / "weights/lito-mlx"
    trellis_root = repo / "weights/trellis2/microsoft/TRELLIS-image-large"
    voxel_weights = backend.load_lito_voxel_decoder_weight_arrays(lito_root)
    latent_tokens = np.zeros((1, 2, 32), dtype=np.float32)

    result = backend.decode_lito_init_coords_from_latents(
        latent_tokens,
        voxel_weights,
        trellis_root=trellis_root,
        block_indices=[0],
    )

    assert tuple(result["ss_latent"].shape) == (1, 8, 16, 16, 16)
    assert tuple(result["occ_logits"].shape) == (1, 1, 64, 64, 64)
    assert result["occ_bool_grid"].shape == (1, 1, 64, 64, 64)
    assert len(result["q_seq_lens"]) == 1
    assert result["init_coord"].shape[1] == 3


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "weights/lito-mlx/tokenizer/lito_new.safetensors").is_file(),
    reason="LiTo weights absent",
)
def test_load_lito_gaussian_decoder_weight_arrays_reads_real_safetensors_subset():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    root = Path(__file__).resolve().parents[1] / "weights/lito-mlx"

    arrays = backend.load_lito_gaussian_decoder_weight_arrays(
        root,
        names=(
            "point_linear.weight",
            "point_mlp.mlps.0.w1.weight",
            "point_mlp.mlps.0.w2.weight",
            "perceiver.blocks.0.ca_mlp.w1.weight",
            "perceiver.blocks.0.ca_mlp.w2.weight",
            "gs_output_shape_mlp.final_layer.linear.weight",
            "gs_output_color_mlp.final_layer.linear.weight",
        ),
    )

    assert arrays["point_linear.weight"].shape == (512, 195)
    assert arrays["point_mlp.mlps.0.w1.weight"].shape == (512, 512)
    assert arrays["point_mlp.mlps.0.w2.weight"].shape == (512, 512)
    assert arrays["perceiver.blocks.0.ca_mlp.w1.weight"].shape == (2048, 512)
    assert arrays["perceiver.blocks.0.ca_mlp.w2.weight"].shape == (2048, 512)
    assert arrays["gs_output_shape_mlp.final_layer.linear.weight"].shape == (640, 512)
    assert arrays["gs_output_color_mlp.final_layer.linear.weight"].shape == (3136, 512)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "weights/lito-mlx/tokenizer/lito_new.safetensors").is_file(),
    reason="LiTo weights absent",
)
def test_load_lito_gaussian_decoder_weight_arrays_reads_real_cross_attention_subset():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    root = Path(__file__).resolve().parents[1] / "weights/lito-mlx"

    arrays = backend.load_lito_gaussian_decoder_weight_arrays(
        root,
        names=(
            "perceiver.blocks.0.kv_linear.weight",
            "perceiver.blocks.0.ca_layer.layernorm_q.weight",
            "perceiver.blocks.0.ca_layer.layernorm_q.bias",
            "perceiver.blocks.0.ca_layer.layernorm_kv.weight",
            "perceiver.blocks.0.ca_layer.layernorm_kv.bias",
            "perceiver.blocks.0.ca_layer.linear_q.weight",
            "perceiver.blocks.0.ca_layer.linear_q.bias",
            "perceiver.blocks.0.ca_layer.linear_kv.weight",
            "perceiver.blocks.0.ca_layer.linear_kv.bias",
            "perceiver.blocks.0.ca_layer.linear_out.weight",
            "perceiver.blocks.0.ca_layer.linear_out.bias",
            "perceiver.blocks.0.ca_layer.rmsnorm_q.scale",
            "perceiver.blocks.0.ca_layer.rmsnorm_k.scale",
            "perceiver.blocks.0.ca_ln.weight",
            "perceiver.blocks.0.ca_ln.bias",
            "perceiver.blocks.0.ca_mlp.w1.weight",
            "perceiver.blocks.0.ca_mlp.w2.weight",
            "perceiver.blocks.0.ca_mlp.w3.weight",
        ),
    )

    assert arrays["perceiver.blocks.0.kv_linear.weight"].shape == (32, 32)
    assert arrays["perceiver.blocks.0.ca_layer.layernorm_q.weight"].shape == (512,)
    assert arrays["perceiver.blocks.0.ca_layer.layernorm_kv.weight"].shape == (32,)
    assert arrays["perceiver.blocks.0.ca_layer.linear_q.weight"].shape == (512, 512)
    assert arrays["perceiver.blocks.0.ca_layer.linear_kv.weight"].shape == (1024, 32)
    assert arrays["perceiver.blocks.0.ca_layer.linear_out.weight"].shape == (512, 512)
    assert arrays["perceiver.blocks.0.ca_layer.rmsnorm_q.scale"].shape == (512,)
    assert arrays["perceiver.blocks.0.ca_layer.rmsnorm_k.scale"].shape == (512,)
    assert arrays["perceiver.blocks.0.ca_ln.weight"].shape == (512,)
    assert arrays["perceiver.blocks.0.ca_mlp.w1.weight"].shape == (2048, 512)
    assert arrays["perceiver.blocks.0.ca_mlp.w2.weight"].shape == (2048, 512)
    assert arrays["perceiver.blocks.0.ca_mlp.w3.weight"].shape == (512, 2048)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "weights/lito-mlx/tokenizer/lito_new.safetensors").is_file(),
    reason="LiTo weights absent",
)
def test_load_lito_voxel_decoder_weight_arrays_reads_real_safetensors_subset():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    root = Path(__file__).resolve().parents[1] / "weights/lito-mlx"

    arrays = backend.load_lito_voxel_decoder_weight_arrays(
        root,
        names=(
            "input_linear.weight",
            "net.init_query",
            "net.zyx_pos_encoder.freq_bands",
            "net.init_query_linear.weight",
            "net.encoder.blocks.0.ca_layer.linear_q.weight",
            "net.encoder.blocks.0.sa_layers.0.linear_qkv.weight",
            "final_layer.linear.weight",
        ),
    )

    assert arrays["input_linear.weight"].shape == (512, 32)
    assert arrays["net.init_query"].shape == (16, 16, 16, 512)
    assert arrays["net.zyx_pos_encoder.freq_bands"].shape == (128,)
    assert arrays["net.init_query_linear.weight"].shape == (512, 771)
    assert arrays["net.encoder.blocks.0.ca_layer.linear_q.weight"].shape == (1024, 512)
    assert arrays["net.encoder.blocks.0.sa_layers.0.linear_qkv.weight"].shape == (3072, 512)
    assert arrays["final_layer.linear.weight"].shape == (8, 512)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "weights/lito-mlx/image_to_3d/lito_dit_rgba.safetensors").is_file(),
    reason="LiTo weights absent",
)
def test_load_lito_dit_weight_arrays_reads_real_safetensors_subset():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    root = Path(__file__).resolve().parents[1] / "weights/lito-mlx"

    arrays = backend.load_lito_dit_weight_arrays(
        root,
        names=(
            "z_proj.weight",
            "pos_mtx",
            "cond_embedder.y_proj.fc1.weight",
            "t_proj_linear1.weight",
            "final_layer.adaLN_linear1.weight",
        ),
    )

    assert arrays["z_proj.weight"].shape == (1152, 32)
    assert arrays["pos_mtx"].shape == (8192, 1152)
    assert arrays["cond_embedder.y_proj.fc1.weight"].shape == (1152, 2048)
    assert arrays["t_proj_linear1.weight"].shape == (1152, 64)
    assert arrays["final_layer.adaLN_linear1.weight"].shape == (1152, 1152)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "weights/lito-mlx/image_to_3d/lito_dit_rgba.safetensors").is_file(),
    reason="LiTo weights absent",
)
def test_load_lito_patch_encoder_weight_arrays_reads_real_safetensors_subset():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    root = Path(__file__).resolve().parents[1] / "weights/lito-mlx"

    arrays = backend.load_lito_patch_encoder_weight_arrays(
        root,
        names=(
            "dinov2_model.model.patch_embed.proj.weight",
            "dinov2_model.model.blocks.0.attn.qkv.weight",
            "dinov2_model.model.register_tokens",
            "learnable_model.weight",
            "learnable_paddings",
        ),
    )

    assert arrays["dinov2_model.model.patch_embed.proj.weight"].shape == (1024, 3, 14, 14)
    assert arrays["dinov2_model.model.blocks.0.attn.qkv.weight"].shape == (3072, 1024)
    assert arrays["dinov2_model.model.register_tokens"].shape == (1, 4, 1024)
    assert arrays["learnable_model.weight"].shape == (1024, 4, 14, 14)
    assert arrays["learnable_paddings"].shape == (5, 1024)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "weights/lito-mlx/image_to_3d/lito_dit_rgba.safetensors").is_file(),
    reason="LiTo weights absent",
)
def test_real_patch_encoder_condition_tokens_block0_runs_from_loaded_checkpoint_weights():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    root = Path(__file__).resolve().parents[1] / "weights/lito-mlx"
    weights = backend.load_lito_patch_encoder_weight_arrays(root)
    config = backend.LitoPatchEncoderConfig(input_size=28, attention_chunk_size=4)
    rgba = np.ones((28, 28, 4), dtype=np.float32)
    rgba[..., :3] = np.array([0.2, 0.4, 0.6], dtype=np.float32)

    tokens = backend.run_lito_patch_encoder_condition_tokens(rgba, weights, block_indices=[0], config=config)

    assert tuple(tokens.shape) == (1, 9, 2048)
    assert np.isfinite(np.asarray(tokens)).all()


def test_decode_lito_gaussian_outputs_matches_upstream_equations():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    init_coord = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    shape_out = np.zeros((1, 64 * 10), dtype=np.float32)
    color_out = np.zeros((1, 64 * 49), dtype=np.float32)
    raw_rgb = np.linspace(-0.5, 0.5, 64 * 48, dtype=np.float32).reshape(1, 64, 48)
    color_out.reshape(1, 64, 49)[..., 1:] = raw_rgb

    decoded = backend.decode_lito_gaussian_outputs(shape_out, color_out, init_coord)

    expected_scaling = np.sqrt((0.5 * 0.01) ** 2 + 0.001**2)
    expected_opacity = 1.0 / (1.0 + np.exp(-0.1))
    assert decoded["xyz_w"].shape == (1, 64, 3)
    assert decoded["scaling"].shape == (1, 64, 3)
    assert decoded["quaternion"].shape == (1, 64, 4)
    assert decoded["opacity"].shape == (1, 64, 1)
    assert decoded["rgb_sh"].shape == (1, 64, 16, 3)
    assert np.allclose(decoded["xyz_w"], init_coord[:, None, :])
    assert np.allclose(decoded["scaling"], expected_scaling)
    assert np.allclose(decoded["quaternion"][..., :3], 0.0)
    assert np.allclose(decoded["quaternion"][..., 3], 1.0)
    assert np.allclose(decoded["opacity"], expected_opacity)
    assert np.allclose(decoded["rgb_sh"].reshape(1, 64, 48), raw_rgb)
    normalized = backend.normalize_lito_gs_dict(decoded)
    assert normalized["xyz_w"].shape == (64, 3)


def test_normalize_lito_gs_dict_returns_export_ready_numpy_arrays():
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    gs_dict = {
        "xyz_w": _ArrayLike(np.arange(6, dtype=np.float16).reshape(1, 2, 3)),
        "scaling": _ArrayLike(np.full((1, 2, 3), 0.25, dtype=np.float16)),
        "quaternion": _ArrayLike(
            np.array([[[0.0, 0.0, 0.0, 2.0], [0.0, 0.0, 1.0, 1.0]]], dtype=np.float16)
        ),
        "opacity": _ArrayLike(np.array([[[0.8], [0.4]]], dtype=np.float16)),
        "rgb_sh": _ArrayLike(np.arange(18, dtype=np.float16).reshape(1, 2, 3, 3)),
        "lf": _ArrayLike(np.arange(8, dtype=np.float16).reshape(1, 2, 4)),
    }

    normalized = backend.normalize_lito_gs_dict(gs_dict)

    assert set(normalized) >= {"xyz_w", "scaling", "quaternion", "opacity", "rgb_sh", "lf"}
    assert normalized["xyz_w"].shape == (2, 3)
    assert normalized["scaling"].shape == (2, 3)
    assert normalized["quaternion"].shape == (2, 4)
    assert normalized["opacity"].shape == (2, 1)
    assert normalized["rgb_sh"].shape == (2, 3, 3)
    assert normalized["lf"].shape == (2, 4)
    for key in ("xyz_w", "scaling", "quaternion", "opacity", "rgb_sh", "lf"):
        assert isinstance(normalized[key], np.ndarray)
        assert normalized[key].dtype == np.float32
    assert np.allclose(np.linalg.norm(normalized["quaternion"], axis=1), 1.0)


def test_write_lito_gaussians_ply_uses_checkpoint_schema(tmp_path):
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    output = tmp_path / "real-schema.ply"
    gs_dict = {
        "xyz_w": np.array([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=np.float32),
        "scaling": np.full((2, 3), 0.02, dtype=np.float32),
        "quaternion": np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
        "opacity": np.full((2, 1), 0.8, dtype=np.float32),
        "rgb_sh": np.arange(96, dtype=np.float32).reshape(2, 16, 3) / 100.0,
    }

    backend.write_lito_gaussians_ply(output, gs_dict, ply_storage="ascii")

    lines = output.read_text(encoding="ascii").splitlines()
    assert lines[2] == "comment mlx-spatial LiTo checkpoint-backed 3DGS export"
    assert "comment mlx-spatial LiTo source-contract smoke 3DGS export" not in lines
    assert "element vertex 2" in lines
    assert "property float f_dc_2" in lines
    assert "property float f_rest_44" in lines
    header_end = lines.index("end_header")
    assert len(lines[header_end + 1 :]) == 2
    assert len(lines[header_end + 1].split()) == 62


def test_write_lito_gaussians_ply_defaults_to_binary_little_endian(tmp_path):
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    output = tmp_path / "real-schema-binary.ply"
    gs_dict = {
        "xyz_w": np.array([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=np.float32),
        "scaling": np.full((2, 3), 0.02, dtype=np.float32),
        "quaternion": np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
        "opacity": np.full((2, 1), 0.8, dtype=np.float32),
        "rgb_sh": np.arange(96, dtype=np.float32).reshape(2, 16, 3) / 100.0,
    }

    backend.write_lito_gaussians_ply(output, gs_dict)

    data = output.read_bytes()
    assert b"format binary_little_endian 1.0\n" in data[:128]
    assert b"element vertex 2\n" in data
    _, body = data.split(b"end_header\n", maxsplit=1)
    values = np.frombuffer(body, dtype="<f4").reshape(2, 62)
    assert values.shape == (2, 62)
    assert np.allclose(values[0, :3], [0.0, 0.1, 0.2])


def test_checkpoint_generate_failure_leaves_no_output_after_backend_creation_fails(tmp_path, monkeypatch):
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    weights_root = _write_fake_lito_weights(tmp_path / "weights")
    image = _write_synthetic_image(tmp_path / "input.png")
    output = tmp_path / "result.ply"
    observed: dict[str, object] = {}

    def unavailable(config):
        observed["weights_root"] = config.weights_root
        observed["memory_profile"] = config.memory_profile
        observed["max_init_coords_per_batch"] = config.max_init_coords_per_batch
        observed["allow_cuda"] = config.allow_cuda
        observed["asset_summary"] = config.asset_summary
        raise backend.LitoBackendUnavailable("optional LiTo backend dependency unavailable")

    monkeypatch.setattr("mlx_spatial.lito_inference.create_lito_real_backend", unavailable)

    pipeline = LitoInferencePipeline(weights_root, memory_profile="safe")
    with pytest.raises(backend.LitoBackendUnavailable, match="optional LiTo backend dependency unavailable"):
        pipeline.generate(
            image,
            output_path=output,
            output_format="ply",
            num_steps=1,
            cfg_scale=1.0,
            seed=3,
            resolution=16,
            render_size=8,
        )

    assert observed["weights_root"] == weights_root
    assert observed["memory_profile"] == "safe"
    assert observed["max_init_coords_per_batch"] == "profile"
    assert observed["allow_cuda"] is False
    assert not output.exists()
    assert not output.with_suffix(".safetensors").exists()


def test_checkpoint_pipeline_passes_explicit_init_coord_cap_to_backend_config(tmp_path, monkeypatch):
    backend = importlib.import_module("mlx_spatial.lito_real_backend")
    weights_root = _write_fake_lito_weights(tmp_path / "weights")
    image = _write_synthetic_image(tmp_path / "input.png")
    observed: dict[str, object] = {}

    def unavailable(config):
        observed["max_init_coords_per_batch"] = config.max_init_coords_per_batch
        raise backend.LitoBackendUnavailable("backend stopped before decode")

    monkeypatch.setattr("mlx_spatial.lito_inference.create_lito_real_backend", unavailable)

    pipeline = LitoInferencePipeline(weights_root, memory_profile="safe", max_init_coords_per_batch=13)
    with pytest.raises(backend.LitoBackendUnavailable, match="backend stopped before decode"):
        pipeline.generate(
            image,
            output_path=tmp_path / "result.ply",
            output_format="ply",
            num_steps=1,
            cfg_scale=1.0,
            seed=3,
            resolution=16,
            render_size=8,
        )

    assert observed["max_init_coords_per_batch"] == 13


class _ArrayLike:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def detach(self) -> _ArrayLike:
        return self

    def cpu(self) -> _ArrayLike:
        return self

    def numpy(self) -> np.ndarray:
        return self._array


def _write_fake_lito_weights(root: Path) -> Path:
    for relative_path, required_keys in LITO_REAL_TENSOR_SENTINELS.items():
        tensors = {key: np.ones((1,), dtype=np.float32) for key in required_keys}
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, path)
    return root


def _write_fake_lito_architecture_weights(root: Path) -> Path:
    image_to_3d = {
        "velocity_estimator_ema.module.z_proj.weight": np.ones((8, 2), dtype=np.float32),
        "velocity_estimator_ema.module.pos_mtx": np.ones((4, 8), dtype=np.float32),
        "velocity_estimator_ema.module.cond_embedder.y_proj.fc1.weight": np.ones((8, 6), dtype=np.float32),
        "velocity_estimator_ema.module.final_layer.linear.weight": np.ones((2, 8), dtype=np.float32),
        "velocity_estimator_ema.module.final_layer.linear.bias": np.ones((2,), dtype=np.float32),
        "velocity_estimator_ema.module.t_proj.0.weight": np.ones((8, 4), dtype=np.float32),
        "velocity_estimator_ema.module.t_embedder.freq_bands": np.ones((2,), dtype=np.float32),
        "velocity_estimator_ema.module.pos_proj.weight": np.ones((8, 8), dtype=np.float32),
        "velocity_estimator_ema.module.blocks.0.mlp.w1.weight": np.ones((16, 8), dtype=np.float32),
        "velocity_estimator_ema.module.blocks.0.attn.rmsnorm_q.scale": np.ones((8,), dtype=np.float32),
        "velocity_estimator_ema.module.blocks.1.mlp.w1.weight": np.ones((16, 8), dtype=np.float32),
    }
    tokenizer = {
        "gs_decoder.point_linear.weight": np.ones((8, 195), dtype=np.float32),
        "gs_decoder.perceiver.blocks.0.ca_layer.linear_kv.weight": np.ones((16, 2), dtype=np.float32),
        "gs_decoder.perceiver.blocks.0.sa_layers.0.linear_qkv.weight": np.ones((24, 8), dtype=np.float32),
        "gs_decoder.perceiver.blocks.0.sa_layers.1.linear_qkv.weight": np.ones((24, 8), dtype=np.float32),
        "gs_decoder.perceiver.blocks.1.sa_layers.0.linear_qkv.weight": np.ones((24, 8), dtype=np.float32),
        "gs_decoder.perceiver.blocks.0.ca_layer.rmsnorm_q.scale": np.ones((8,), dtype=np.float32),
        "gs_decoder.perceiver.blocks.0.ca_mlp.w12.weight": np.ones((64, 8), dtype=np.float32),
        "gs_decoder.gs_output_shape_mlp.2.linear.weight": np.ones((640, 8), dtype=np.float32),
        "gs_decoder.gs_output_shape_mlp.2.linear.bias": np.ones((640,), dtype=np.float32),
        "gs_decoder.gs_output_color_mlp.2.linear.weight": np.ones((3136, 8), dtype=np.float32),
        "gs_decoder.xyz_encoding.freq_bands": np.ones((32,), dtype=np.float32),
        "voxel_decoder.input_linear.weight": np.ones((8, 2), dtype=np.float32),
        "voxel_decoder.net.init_query": np.ones((2, 2, 2, 8), dtype=np.float32),
        "voxel_decoder.net.init_query_linear.weight": np.ones((8, 771), dtype=np.float32),
        "voxel_decoder.net.encoder.blocks.0.sa_layers.0.linear_qkv.weight": np.ones((24, 8), dtype=np.float32),
        "voxel_decoder.net.encoder.blocks.0.sa_layers.1.linear_qkv.weight": np.ones((24, 8), dtype=np.float32),
        "voxel_decoder.net.encoder.blocks.1.sa_layers.0.linear_qkv.weight": np.ones((24, 8), dtype=np.float32),
        "voxel_decoder.final_layer.linear.weight": np.ones((8, 8), dtype=np.float32),
        "voxel_decoder.net.zyx_pos_encoder.freq_bands": np.ones((128,), dtype=np.float32),
    }
    image_path = root / "image_to_3d" / "lito_dit_rgba.safetensors"
    tokenizer_path = root / "tokenizer" / "lito_new.safetensors"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(image_to_3d, image_path)
    save_file(tokenizer, tokenizer_path)
    return root


def _write_fake_lito_weight_loader_tensors(root: Path) -> Path:
    image_to_3d = {
        "velocity_estimator_ema.module.z_proj.weight": np.array([[1.0, 2.0]], dtype=np.float32),
        "velocity_estimator.z_proj.weight": np.array([[100.0, 200.0]], dtype=np.float32),
        "velocity_estimator_ema.module.t_proj.0.weight": np.array([[3.0, 4.0]], dtype=np.float32),
        "velocity_estimator_ema.module.t_proj.2.bias": np.array([5.0, 6.0], dtype=np.float32),
        "velocity_estimator_ema.module.t0_proj.1.weight": np.array([[7.0, 8.0]], dtype=np.float32),
        "velocity_estimator_ema.module.final_layer.adaLN_modulation.0.weight": np.array(
            [[9.0, 10.0]], dtype=np.float32
        ),
        "velocity_estimator_ema.module.final_layer.adaLN_modulation.2.bias": np.array(
            [11.0, 12.0], dtype=np.float32
        ),
        "patch_encoder.dinov2_model.model.cls_token": np.ones((1, 1, 4), dtype=np.float32),
        "patch_encoder.dinov2_model.model.patch_embed.proj.weight": np.ones((4, 3, 14, 14), dtype=np.float32),
        "patch_encoder.learnable_model.weight": np.ones((4, 4, 14, 14), dtype=np.float32),
        "patch_encoder.learnable_paddings": np.ones((2, 4), dtype=np.float32),
    }
    tokenizer = {
        "gs_decoder.point_linear.weight": np.ones((2, 2), dtype=np.float32),
        "gs_decoder.point_mlp.0.weight": np.array([1.0, 2.0], dtype=np.float32),
        "gs_decoder.point_mlp.1.w12.weight": np.arange(1, 9, dtype=np.float32).reshape(4, 2),
        "gs_decoder.point_mlp.1.w3.weight": np.ones((2, 2), dtype=np.float32) * 3,
        "gs_decoder.point_mlp.2.linear.bias": np.array([4.0, 5.0], dtype=np.float32),
        "gs_decoder.perceiver.blocks.0.ca_mlp.w12.weight": np.arange(8, dtype=np.float32).reshape(4, 2),
        "gs_decoder.gs_output_shape_mlp.2.linear.weight": np.ones((4, 2), dtype=np.float32),
        "gs_decoder.gs_output_color_mlp.2.linear.weight": np.ones((4, 2), dtype=np.float32),
        "voxel_decoder.input_linear.weight": np.ones((2, 2), dtype=np.float32),
    }
    image_path = root / "image_to_3d" / "lito_dit_rgba.safetensors"
    tokenizer_path = root / "tokenizer" / "lito_new.safetensors"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(image_to_3d, image_path)
    save_file(tokenizer, tokenizer_path)
    return root


def _fake_output_head_weights() -> dict[str, np.ndarray]:
    weights: dict[str, np.ndarray] = {}
    for prefix, output_dim in (("gs_output_shape_mlp", 640), ("gs_output_color_mlp", 3136)):
        weights[f"{prefix}.norms.0.weight"] = np.ones((512,), dtype=np.float32)
        weights[f"{prefix}.norms.0.bias"] = np.zeros((512,), dtype=np.float32)
        weights[f"{prefix}.mlps.0.w1.weight"] = np.zeros((512, 512), dtype=np.float32)
        weights[f"{prefix}.mlps.0.w2.weight"] = np.zeros((512, 512), dtype=np.float32)
        weights[f"{prefix}.mlps.0.w3.weight"] = np.zeros((512, 512), dtype=np.float32)
        weights[f"{prefix}.final_layer.norm_final.weight"] = np.ones((512,), dtype=np.float32)
        weights[f"{prefix}.final_layer.norm_final.bias"] = np.zeros((512,), dtype=np.float32)
        weights[f"{prefix}.final_layer.linear.weight"] = np.zeros((output_dim, 512), dtype=np.float32)
        weights[f"{prefix}.final_layer.linear.bias"] = np.zeros((output_dim,), dtype=np.float32)
    return weights


def _fake_query_point_weights() -> dict[str, np.ndarray]:
    weights = _fake_output_head_weights()
    weights["xyz_encoding.freq_bands"] = np.ones((32,), dtype=np.float32)
    weights["point_linear.weight"] = np.zeros((512, 195), dtype=np.float32)
    weights["point_linear.bias"] = np.zeros((512,), dtype=np.float32)
    prefix = "point_mlp"
    weights[f"{prefix}.norms.0.weight"] = np.ones((512,), dtype=np.float32)
    weights[f"{prefix}.norms.0.bias"] = np.zeros((512,), dtype=np.float32)
    weights[f"{prefix}.mlps.0.w1.weight"] = np.zeros((512, 512), dtype=np.float32)
    weights[f"{prefix}.mlps.0.w2.weight"] = np.zeros((512, 512), dtype=np.float32)
    weights[f"{prefix}.mlps.0.w3.weight"] = np.zeros((512, 512), dtype=np.float32)
    weights[f"{prefix}.final_layer.norm_final.weight"] = np.ones((512,), dtype=np.float32)
    weights[f"{prefix}.final_layer.norm_final.bias"] = np.zeros((512,), dtype=np.float32)
    weights[f"{prefix}.final_layer.linear.weight"] = np.zeros((512, 512), dtype=np.float32)
    weights[f"{prefix}.final_layer.linear.bias"] = np.zeros((512,), dtype=np.float32)
    return weights


def _fake_perceiver_block0_cross_only_weights() -> dict[str, np.ndarray]:
    prefix = "perceiver.blocks.0"
    weights: dict[str, np.ndarray] = {
        f"{prefix}.kv_linear.weight": np.eye(2, dtype=np.float32),
        f"{prefix}.ca_layer.layernorm_q.weight": np.ones((4,), dtype=np.float32),
        f"{prefix}.ca_layer.layernorm_q.bias": np.zeros((4,), dtype=np.float32),
        f"{prefix}.ca_layer.layernorm_kv.weight": np.ones((2,), dtype=np.float32),
        f"{prefix}.ca_layer.layernorm_kv.bias": np.zeros((2,), dtype=np.float32),
        f"{prefix}.ca_layer.linear_q.weight": np.zeros((4, 4), dtype=np.float32),
        f"{prefix}.ca_layer.linear_q.bias": np.zeros((4,), dtype=np.float32),
        f"{prefix}.ca_layer.linear_kv.weight": np.zeros((8, 2), dtype=np.float32),
        f"{prefix}.ca_layer.linear_kv.bias": np.zeros((8,), dtype=np.float32),
        f"{prefix}.ca_layer.linear_out.weight": np.zeros((4, 4), dtype=np.float32),
        f"{prefix}.ca_layer.linear_out.bias": np.array([0.25, -0.25, 0.5, -0.5], dtype=np.float32),
        f"{prefix}.ca_layer.rmsnorm_q.scale": np.ones((4,), dtype=np.float32),
        f"{prefix}.ca_layer.rmsnorm_k.scale": np.ones((4,), dtype=np.float32),
        f"{prefix}.ca_ln.weight": np.ones((4,), dtype=np.float32),
        f"{prefix}.ca_ln.bias": np.zeros((4,), dtype=np.float32),
        f"{prefix}.ca_mlp.w1.weight": np.eye(4, dtype=np.float32),
        f"{prefix}.ca_mlp.w2.weight": np.eye(4, dtype=np.float32),
        f"{prefix}.ca_mlp.w3.weight": np.eye(4, dtype=np.float32),
    }
    return weights


def _fake_perceiver_block0_full_weights() -> dict[str, np.ndarray]:
    weights = _fake_perceiver_block0_cross_only_weights()
    prefix = "perceiver.blocks.0"
    for layer_index, bias_value in ((0, 0.125), (1, -0.0625)):
        weights[f"{prefix}.ln1_layers.{layer_index}.weight"] = np.ones((4,), dtype=np.float32)
        weights[f"{prefix}.ln1_layers.{layer_index}.bias"] = np.zeros((4,), dtype=np.float32)
        weights[f"{prefix}.ln2_layers.{layer_index}.weight"] = np.ones((4,), dtype=np.float32)
        weights[f"{prefix}.ln2_layers.{layer_index}.bias"] = np.zeros((4,), dtype=np.float32)
        weights[f"{prefix}.sa_layers.{layer_index}.linear_qkv.weight"] = np.zeros((12, 4), dtype=np.float32)
        weights[f"{prefix}.sa_layers.{layer_index}.linear_qkv.bias"] = np.zeros((12,), dtype=np.float32)
        weights[f"{prefix}.sa_layers.{layer_index}.linear_out.weight"] = np.zeros((4, 4), dtype=np.float32)
        weights[f"{prefix}.sa_layers.{layer_index}.linear_out.bias"] = np.full((4,), bias_value, dtype=np.float32)
        weights[f"{prefix}.sa_layers.{layer_index}.rmsnorm_q.scale"] = np.ones((4,), dtype=np.float32)
        weights[f"{prefix}.sa_layers.{layer_index}.rmsnorm_k.scale"] = np.ones((4,), dtype=np.float32)
        weights[f"{prefix}.mlp_layers.{layer_index}.w1.weight"] = np.eye(4, dtype=np.float32)
        weights[f"{prefix}.mlp_layers.{layer_index}.w2.weight"] = np.eye(4, dtype=np.float32)
        weights[f"{prefix}.mlp_layers.{layer_index}.w3.weight"] = np.eye(4, dtype=np.float32)
    return weights


def _fake_perceiver_blocks_full_weights(block_indices: tuple[int, ...]) -> dict[str, np.ndarray]:
    weights: dict[str, np.ndarray] = {}
    template = _fake_perceiver_block0_full_weights()
    for block_index in block_indices:
        for name, value in template.items():
            remapped = name.replace("perceiver.blocks.0.", f"perceiver.blocks.{block_index}.", 1)
            weights[remapped] = value.copy()
        weights[f"perceiver.blocks.{block_index}.ca_layer.linear_out.bias"] = np.array(
            [0.25, -0.25, 0.5, -0.5],
            dtype=np.float32,
        ) * (block_index + 1)
    return weights


def _fake_dit_weights() -> dict[str, np.ndarray]:
    dim = 4
    weights: dict[str, np.ndarray] = {
        "t_embedder.freq_bands": np.ones((1,), dtype=np.float32),
        "t_proj_linear1.weight": np.zeros((dim, 2), dtype=np.float32),
        "t_proj_linear1.bias": np.zeros((dim,), dtype=np.float32),
        "t_proj_linear2.weight": np.eye(dim, dtype=np.float32),
        "t_proj_linear2.bias": np.zeros((dim,), dtype=np.float32),
        "t0_proj_linear.weight": np.zeros((6 * dim, dim), dtype=np.float32),
        "t0_proj_linear.bias": np.zeros((6 * dim,), dtype=np.float32),
        "cond_embedder.y_embedding": np.zeros((2048,), dtype=np.float32),
        "cond_embedder.y_proj.fc1.weight": np.zeros((dim, 2048), dtype=np.float32),
        "cond_embedder.y_proj.fc1.bias": np.zeros((dim,), dtype=np.float32),
        "cond_embedder.y_proj.fc2.weight": np.eye(dim, dtype=np.float32),
        "cond_embedder.y_proj.fc2.bias": np.zeros((dim,), dtype=np.float32),
        "z_proj.weight": np.zeros((dim, 32), dtype=np.float32),
        "z_proj.bias": np.zeros((dim,), dtype=np.float32),
        "z_proj_ln.weight": np.ones((dim,), dtype=np.float32),
        "z_proj_ln.bias": np.zeros((dim,), dtype=np.float32),
        "pos_mtx": np.zeros((2, dim), dtype=np.float32),
        "pos_proj.weight": np.eye(dim, dtype=np.float32),
        "pos_proj.bias": np.zeros((dim,), dtype=np.float32),
        "final_layer.adaLN_linear1.weight": np.zeros((dim, dim), dtype=np.float32),
        "final_layer.adaLN_linear1.bias": np.zeros((dim,), dtype=np.float32),
        "final_layer.adaLN_linear2.weight": np.zeros((2 * dim, dim), dtype=np.float32),
        "final_layer.adaLN_linear2.bias": np.zeros((2 * dim,), dtype=np.float32),
        "final_layer.linear.weight": np.zeros((32, dim), dtype=np.float32),
        "final_layer.linear.bias": np.full((32,), 0.25, dtype=np.float32),
    }
    prefix = "blocks.0"
    weights.update(
        {
            f"{prefix}.scale_shift_table": np.zeros((6, dim), dtype=np.float32),
            f"{prefix}.attn.linear_qkv.weight": np.zeros((3 * dim, dim), dtype=np.float32),
            f"{prefix}.attn.linear_qkv.bias": np.zeros((3 * dim,), dtype=np.float32),
            f"{prefix}.attn.rmsnorm_q.scale": np.ones((dim,), dtype=np.float32),
            f"{prefix}.attn.rmsnorm_k.scale": np.ones((dim,), dtype=np.float32),
            f"{prefix}.attn.linear_out.weight": np.zeros((dim, dim), dtype=np.float32),
            f"{prefix}.attn.linear_out.bias": np.full((dim,), 0.125, dtype=np.float32),
            f"{prefix}.cross_attn.layernorm_q.weight": np.ones((dim,), dtype=np.float32),
            f"{prefix}.cross_attn.layernorm_q.bias": np.zeros((dim,), dtype=np.float32),
            f"{prefix}.cross_attn.layernorm_kv.weight": np.ones((dim,), dtype=np.float32),
            f"{prefix}.cross_attn.layernorm_kv.bias": np.zeros((dim,), dtype=np.float32),
            f"{prefix}.cross_attn.linear_q.weight": np.zeros((dim, dim), dtype=np.float32),
            f"{prefix}.cross_attn.linear_q.bias": np.zeros((dim,), dtype=np.float32),
            f"{prefix}.cross_attn.linear_kv.weight": np.zeros((2 * dim, dim), dtype=np.float32),
            f"{prefix}.cross_attn.linear_kv.bias": np.zeros((2 * dim,), dtype=np.float32),
            f"{prefix}.cross_attn.rmsnorm_q.scale": np.ones((dim,), dtype=np.float32),
            f"{prefix}.cross_attn.rmsnorm_k.scale": np.ones((dim,), dtype=np.float32),
            f"{prefix}.cross_attn.linear_out.weight": np.zeros((dim, dim), dtype=np.float32),
            f"{prefix}.cross_attn.linear_out.bias": np.zeros((dim,), dtype=np.float32),
            f"{prefix}.mlp.w1.weight": np.zeros((dim, dim), dtype=np.float32),
            f"{prefix}.mlp.w2.weight": np.zeros((dim, dim), dtype=np.float32),
            f"{prefix}.mlp.w2.bias": np.zeros((dim,), dtype=np.float32),
            f"{prefix}.mlp.w3.weight": np.zeros((dim, dim), dtype=np.float32),
        }
    )
    return weights


def _fake_patch_encoder_weights() -> dict[str, np.ndarray]:
    dim = 4
    weights: dict[str, np.ndarray] = {
        "dinov2_model.model.cls_token": np.zeros((1, 1, dim), dtype=np.float32),
        "dinov2_model.model.pos_embed": np.zeros((1, 5, dim), dtype=np.float32),
        "dinov2_model.model.register_tokens": np.zeros((1, 1, dim), dtype=np.float32),
        "dinov2_model.model.patch_embed.proj.weight": np.zeros((dim, 3, 14, 14), dtype=np.float32),
        "dinov2_model.model.patch_embed.proj.bias": np.linspace(-0.1, 0.2, dim, dtype=np.float32),
        "learnable_model.weight": np.zeros((dim, 4, 14, 14), dtype=np.float32),
        "learnable_model.bias": np.linspace(0.1, 0.4, dim, dtype=np.float32),
        "learnable_paddings": np.zeros((2, dim), dtype=np.float32),
    }
    prefix = "dinov2_model.model.blocks.0"
    weights.update(
        {
            f"{prefix}.norm1.weight": np.ones((dim,), dtype=np.float32),
            f"{prefix}.norm1.bias": np.zeros((dim,), dtype=np.float32),
            f"{prefix}.attn.qkv.weight": np.zeros((3 * dim, dim), dtype=np.float32),
            f"{prefix}.attn.qkv.bias": np.zeros((3 * dim,), dtype=np.float32),
            f"{prefix}.attn.proj.weight": np.zeros((dim, dim), dtype=np.float32),
            f"{prefix}.attn.proj.bias": np.full((dim,), 0.05, dtype=np.float32),
            f"{prefix}.ls1.gamma": np.ones((dim,), dtype=np.float32),
            f"{prefix}.norm2.weight": np.ones((dim,), dtype=np.float32),
            f"{prefix}.norm2.bias": np.zeros((dim,), dtype=np.float32),
            f"{prefix}.mlp.fc1.weight": np.zeros((2 * dim, dim), dtype=np.float32),
            f"{prefix}.mlp.fc1.bias": np.zeros((2 * dim,), dtype=np.float32),
            f"{prefix}.mlp.fc2.weight": np.zeros((dim, 2 * dim), dtype=np.float32),
            f"{prefix}.mlp.fc2.bias": np.zeros((dim,), dtype=np.float32),
            f"{prefix}.ls2.gamma": np.ones((dim,), dtype=np.float32),
        }
    )
    return weights


def _fake_voxel_decoder_weights() -> dict[str, np.ndarray]:
    weights: dict[str, np.ndarray] = {
        "input_linear.weight": np.zeros((4, 32), dtype=np.float32),
        "input_linear.bias": np.zeros((4,), dtype=np.float32),
        "net.init_query": np.zeros((2, 2, 2, 4), dtype=np.float32),
        "net.zyx_pos_encoder.freq_bands": np.ones((1,), dtype=np.float32),
        "net.init_query_linear.weight": np.zeros((4, 9), dtype=np.float32),
        "net.init_query_linear.bias": np.array([0.25, -0.25, 0.5, -0.5], dtype=np.float32),
        "final_layer.norm_final.weight": np.ones((4,), dtype=np.float32),
        "final_layer.norm_final.bias": np.zeros((4,), dtype=np.float32),
        "final_layer.linear.weight": np.zeros((8, 4), dtype=np.float32),
        "final_layer.linear.bias": np.arange(8, dtype=np.float32),
    }
    prefix = "net.encoder.blocks.0"
    weights.update(
        {
            f"{prefix}.ca_layer.layernorm_q.weight": np.ones((4,), dtype=np.float32),
            f"{prefix}.ca_layer.layernorm_q.bias": np.zeros((4,), dtype=np.float32),
            f"{prefix}.ca_layer.layernorm_kv.weight": np.ones((4,), dtype=np.float32),
            f"{prefix}.ca_layer.layernorm_kv.bias": np.zeros((4,), dtype=np.float32),
            f"{prefix}.ca_layer.linear_q.weight": np.zeros((4, 4), dtype=np.float32),
            f"{prefix}.ca_layer.linear_q.bias": np.zeros((4,), dtype=np.float32),
            f"{prefix}.ca_layer.linear_kv.weight": np.zeros((8, 4), dtype=np.float32),
            f"{prefix}.ca_layer.linear_kv.bias": np.zeros((8,), dtype=np.float32),
            f"{prefix}.ca_layer.linear_out.weight": np.zeros((4, 4), dtype=np.float32),
            f"{prefix}.ca_layer.linear_out.bias": np.full((4,), 0.125, dtype=np.float32),
            f"{prefix}.ca_layer.rmsnorm_q.scale": np.ones((4,), dtype=np.float32),
            f"{prefix}.ca_layer.rmsnorm_k.scale": np.ones((4,), dtype=np.float32),
            f"{prefix}.ca_ln.weight": np.ones((4,), dtype=np.float32),
            f"{prefix}.ca_ln.bias": np.zeros((4,), dtype=np.float32),
            f"{prefix}.ca_mlp.fc1.weight": np.eye(4, dtype=np.float32),
            f"{prefix}.ca_mlp.fc1.bias": np.zeros((4,), dtype=np.float32),
            f"{prefix}.ca_mlp.fc2.weight": np.eye(4, dtype=np.float32),
            f"{prefix}.ca_mlp.fc2.bias": np.zeros((4,), dtype=np.float32),
        }
    )
    for layer_index, bias_value in ((0, 0.0625), (1, -0.03125)):
        weights[f"{prefix}.ln1_layers.{layer_index}.weight"] = np.ones((4,), dtype=np.float32)
        weights[f"{prefix}.ln1_layers.{layer_index}.bias"] = np.zeros((4,), dtype=np.float32)
        weights[f"{prefix}.ln2_layers.{layer_index}.weight"] = np.ones((4,), dtype=np.float32)
        weights[f"{prefix}.ln2_layers.{layer_index}.bias"] = np.zeros((4,), dtype=np.float32)
        weights[f"{prefix}.sa_layers.{layer_index}.linear_qkv.weight"] = np.zeros((12, 4), dtype=np.float32)
        weights[f"{prefix}.sa_layers.{layer_index}.linear_qkv.bias"] = np.zeros((12,), dtype=np.float32)
        weights[f"{prefix}.sa_layers.{layer_index}.linear_out.weight"] = np.zeros((4, 4), dtype=np.float32)
        weights[f"{prefix}.sa_layers.{layer_index}.linear_out.bias"] = np.full((4,), bias_value, dtype=np.float32)
        weights[f"{prefix}.sa_layers.{layer_index}.rmsnorm_q.scale"] = np.ones((4,), dtype=np.float32)
        weights[f"{prefix}.sa_layers.{layer_index}.rmsnorm_k.scale"] = np.ones((4,), dtype=np.float32)
        weights[f"{prefix}.mlp_layers.{layer_index}.fc1.weight"] = np.eye(4, dtype=np.float32)
        weights[f"{prefix}.mlp_layers.{layer_index}.fc1.bias"] = np.zeros((4,), dtype=np.float32)
        weights[f"{prefix}.mlp_layers.{layer_index}.fc2.weight"] = np.eye(4, dtype=np.float32)
        weights[f"{prefix}.mlp_layers.{layer_index}.fc2.bias"] = np.zeros((4,), dtype=np.float32)
    return weights


def _asset_summary(weights_root: Path):
    pipeline = LitoInferencePipeline(weights_root, memory_profile="safe")
    assert pipeline.real_asset_summary is not None
    return pipeline.real_asset_summary


def _write_synthetic_image(path: Path) -> Path:
    rgba = np.zeros((24, 24, 4), dtype=np.uint8)
    rgba[..., 0] = 80
    rgba[..., 1] = 120
    rgba[..., 2] = 200
    rgba[4:20, 4:20, 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(path)
    return path
