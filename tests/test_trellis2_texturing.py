import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image
from safetensors.mlx import save_file

import mlx_spatial
from mlx_spatial.trellis2_texturing import (
    Trellis2TexturingBlocker,
    Trellis2TexturingPipeline,
    Trellis2TexturingResult,
    _load_obj_mesh,
    TRELLIS2_TEXTURING_DEFAULT_SEED,
    TRELLIS2_TEXTURING_DEFAULT_TEXTURE_SIZE,
)
from mlx_spatial.model_assets import TRELLIS2_ASSETS
from mlx_spatial.ovoxel import FlexibleDualGridMesh
from mlx_spatial.trellis2_export import Trellis2ExportArtifact


def _write_texturing_root(root: Path, *, skip_encoder_config: bool = False):
    """Write a minimal TRELLIS.2 asset root with configs and checkpoints needed for texturing."""
    for asset_path in TRELLIS2_ASSETS.required_paths:
        path = root / asset_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if asset_path.endswith(".safetensors"):
            save_file(
                {
                    "blocks.0.0.norm.weight": mx.array([7.0, 8.0], dtype=mx.float32),
                    "blocks.0.norm2.weight": mx.array([9.0, 10.0], dtype=mx.float32),
                },
                path,
            )
        else:
            path.write_text("{}")

    root.mkdir(parents=True, exist_ok=True)
    pipeline = {
        "name": "Trellis2ImageTo3DPipeline",
        "args": {
            "models": {
                "sparse_structure_decoder": "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
                "sparse_structure_flow_model": "ckpts/ss_flow_img_dit_1_3B_64_bf16",
                "shape_slat_decoder": "ckpts/shape_dec_next_dc_f16c32_fp16",
                "shape_slat_flow_model_512": "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16",
                "shape_slat_flow_model_1024": "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16",
                "tex_slat_decoder": "ckpts/tex_dec_next_dc_f16c32_fp16",
                "tex_slat_flow_model_512": "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16",
                "tex_slat_flow_model_1024": "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
            },
            "sparse_structure_sampler": _sampler_config(),
            "shape_slat_sampler": _sampler_config(),
            "tex_slat_sampler": _sampler_config(guidance_strength=1.0, guidance_rescale=0.0),
            "shape_slat_normalization": _normalization_config(),
            "tex_slat_normalization": _normalization_config(offset=1.0),
            "image_cond_model": {
                "name": "DinoV3FeatureExtractor",
                "args": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m", "image_size": 512},
            },
            "default_pipeline_type": "1024",
        },
    }
    (root / "pipeline.json").write_text(json.dumps(pipeline))

    _write_slat_config(root / "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json")
    _write_slat_checkpoint(root / "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors")

    _write_slat_config(root / "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json")
    _write_slat_checkpoint(root / "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors")

    _write_slat_config(root / "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json")
    _write_slat_checkpoint(root / "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors")

    _write_slat_config(root / "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json")
    _write_slat_checkpoint(root / "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors")

    _write_slat_config(root / "ckpts/ss_flow_img_dit_1_3B_64_bf16.json", in_channels=2, out_channels=2, cond_channels=1024)
    _write_slat_checkpoint(
        root / "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
        in_channels=2, out_channels=2, cond_channels=1024,
    )

    _write_ss_decoder_config(root / "ckpts/ss_dec_conv3d_16l8_fp16.json")
    _write_ss_decoder_checkpoint(root / "ckpts/ss_dec_conv3d_16l8_fp16.safetensors")

    _write_shape_decoder_config(root / "ckpts/shape_dec_next_dc_f16c32_fp16.json")
    _write_combined_shape_vae_checkpoint(
        root / "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        skip_encoder=skip_encoder_config,
    )

    _write_texture_decoder_config(root / "ckpts/tex_dec_next_dc_f16c32_fp16.json")
    _write_texture_decoder_checkpoint(root / "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors")

    if not skip_encoder_config:
        _write_encoder_config(root / "shape_encoder.json")


def _sampler_config(*, guidance_strength=1.0, guidance_rescale=0.0, interval=(0.6, 0.9)):
    return {
        "name": "FlowEulerScheduler",
        "args": {"sigma_min": 1e-5},
        "params": {
            "steps": 1,
            "guidance_strength": guidance_strength,
            "guidance_rescale": guidance_rescale,
            "guidance_interval": list(interval),
            "rescale_t": 3.0,
        },
    }


def _normalization_config(*, offset=0.0):
    return {"mean": [offset] * 32, "std": [1.0] * 32}


def _write_slat_config(path, *, in_channels=64, out_channels=32, cond_channels=1024):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "name": "SLatFlowModel",
        "args": {
            "resolution": 2,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "model_channels": 128,
            "cond_channels": cond_channels,
            "num_blocks": 1,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "pe_mode": "rope",
            "share_mod": True,
            "initialization": "standard",
            "qk_rms_norm": True,
            "qk_rms_norm_cross": True,
            "dtype": "float32",
        },
    }))


def _write_slat_checkpoint(path, *, in_channels=64, out_channels=32, cond_channels=1024):
    model_channels = 128
    num_heads = 4
    head_dim = model_channels // num_heads
    intermediate = int(model_channels * 4.0)
    tensors = {
        "input_layer.weight": mx.zeros((model_channels, in_channels), dtype=mx.float32),
        "input_layer.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "out_layer.weight": mx.zeros((out_channels, model_channels), dtype=mx.float32),
        "out_layer.bias": mx.zeros((out_channels,), dtype=mx.float32),
        "t_embedder.mlp.0.weight": mx.zeros((256, 256), dtype=mx.float32),
        "t_embedder.mlp.0.bias": mx.zeros((256,), dtype=mx.float32),
        "t_embedder.mlp.2.weight": mx.zeros((model_channels, 256), dtype=mx.float32),
        "t_embedder.mlp.2.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "adaLN_modulation.1.weight": mx.zeros((model_channels * 6, model_channels), dtype=mx.float32),
        "adaLN_modulation.1.bias": mx.zeros((model_channels * 6,), dtype=mx.float32),
        "blocks.0.modulation": mx.zeros((model_channels * 6,), dtype=mx.float32),
        "blocks.0.norm2.weight": mx.ones((model_channels,), dtype=mx.float32),
        "blocks.0.norm2.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "blocks.0.self_attn.to_qkv.weight": mx.zeros((model_channels * 3, model_channels), dtype=mx.float32),
        "blocks.0.self_attn.to_qkv.bias": mx.zeros((model_channels * 3,), dtype=mx.float32),
        "blocks.0.self_attn.q_rms_norm.gamma": mx.ones((num_heads, head_dim), dtype=mx.float32),
        "blocks.0.self_attn.k_rms_norm.gamma": mx.ones((num_heads, head_dim), dtype=mx.float32),
        "blocks.0.self_attn.to_out.weight": mx.zeros((model_channels, model_channels), dtype=mx.float32),
        "blocks.0.self_attn.to_out.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "blocks.0.cross_attn.to_q.weight": mx.zeros((model_channels, model_channels), dtype=mx.float32),
        "blocks.0.cross_attn.to_q.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "blocks.0.cross_attn.to_kv.weight": mx.zeros((model_channels * 2, cond_channels), dtype=mx.float32),
        "blocks.0.cross_attn.to_kv.bias": mx.zeros((model_channels * 2,), dtype=mx.float32),
        "blocks.0.cross_attn.q_rms_norm.gamma": mx.ones((num_heads, head_dim), dtype=mx.float32),
        "blocks.0.cross_attn.k_rms_norm.gamma": mx.ones((num_heads, head_dim), dtype=mx.float32),
        "blocks.0.cross_attn.to_out.weight": mx.zeros((model_channels, model_channels), dtype=mx.float32),
        "blocks.0.cross_attn.to_out.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "blocks.0.mlp.mlp.0.weight": mx.zeros((intermediate, model_channels), dtype=mx.float32),
        "blocks.0.mlp.mlp.0.bias": mx.zeros((intermediate,), dtype=mx.float32),
        "blocks.0.mlp.mlp.2.weight": mx.zeros((model_channels, intermediate), dtype=mx.float32),
        "blocks.0.mlp.mlp.2.bias": mx.zeros((model_channels,), dtype=mx.float32),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path)


def _write_ss_decoder_config(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "name": "SparseStructureDecoder",
        "args": {
            "in_channels": 2,
            "out_channels": 2,
            "model_channels": [4],
            "num_blocks": [0],
            "block_type": ["SparseConvNeXtBlock3d"],
            "up_block_type": [],
            "use_fp16": False,
            "resolution": 2,
        },
    }))


def _write_ss_decoder_checkpoint(path):
    tensors = {
        "input_layer.weight": mx.zeros((4, 2), dtype=mx.float32),
        "input_layer.bias": mx.zeros((4,), dtype=mx.float32),
        "output_layer.weight": mx.zeros((2, 4), dtype=mx.float32),
        "output_layer.bias": mx.zeros((2,), dtype=mx.float32),
        "from_latent.weight": mx.zeros((4, 2), dtype=mx.float32),
        "from_latent.bias": mx.zeros((4,), dtype=mx.float32),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path)


def _write_shape_decoder_config(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "name": "FlexiDualGridVaeDecoder",
        "args": {
            "model_channels": [16, 8],
            "latent_channels": 32,
            "num_blocks": [1, 0],
            "block_type": ["SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d"],
            "up_block_type": ["SparseResBlockC2S3d"],
            "use_fp16": False,
            "resolution": 512,
            "pred_subdiv": True,
        },
    }))


def _write_shape_decoder_checkpoint(path):
    model_channels = (16, 8)
    latent_channels = 32
    out_channels = 7
    tensors = {
        "from_latent.weight": mx.ones((model_channels[0], latent_channels), dtype=mx.float32),
        "from_latent.bias": mx.zeros((model_channels[0],), dtype=mx.float32),
        "output_layer.weight": mx.zeros((out_channels, model_channels[-1]), dtype=mx.float32),
        "output_layer.bias": mx.zeros((out_channels,), dtype=mx.float32),
        "blocks.0.0.conv.weight": mx.zeros((model_channels[0], 3, 3, 3, model_channels[0]), dtype=mx.float32),
        "blocks.0.0.conv.bias": mx.zeros((model_channels[0],), dtype=mx.float32),
        "blocks.0.0.norm.weight": mx.ones((model_channels[0],), dtype=mx.float32),
        "blocks.0.0.norm.bias": mx.zeros((model_channels[0],), dtype=mx.float32),
        "blocks.0.0.mlp.0.weight": mx.zeros((model_channels[0] * 4, model_channels[0]), dtype=mx.float32),
        "blocks.0.0.mlp.0.bias": mx.zeros((model_channels[0] * 4,), dtype=mx.float32),
        "blocks.0.0.mlp.2.weight": mx.zeros((model_channels[0], model_channels[0] * 4), dtype=mx.float32),
        "blocks.0.0.mlp.2.bias": mx.zeros((model_channels[0],), dtype=mx.float32),
        "blocks.0.1.norm1.weight": mx.ones((model_channels[0],), dtype=mx.float32),
        "blocks.0.1.norm1.bias": mx.zeros((model_channels[0],), dtype=mx.float32),
        "blocks.0.1.conv1.weight": mx.zeros((model_channels[1] * 8, 3, 3, 3, model_channels[0]), dtype=mx.float32),
        "blocks.0.1.conv1.bias": mx.zeros((model_channels[1] * 8,), dtype=mx.float32),
        "blocks.0.1.conv2.weight": mx.zeros((model_channels[1], 3, 3, 3, model_channels[1]), dtype=mx.float32),
        "blocks.0.1.conv2.bias": mx.zeros((model_channels[1],), dtype=mx.float32),
        "blocks.0.1.to_subdiv.weight": mx.zeros((8, model_channels[0]), dtype=mx.float32),
        "blocks.0.1.to_subdiv.bias": mx.ones((8,), dtype=mx.float32),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path)


def _write_combined_shape_vae_checkpoint(path, *, skip_encoder: bool = False):
    """Write shape decoder + encoder tensors into the shared VAE checkpoint."""
    model_channels = (16, 8)
    latent_channels = 32
    out_channels = 7
    tensors = {
        "from_latent.weight": mx.ones((model_channels[0], latent_channels), dtype=mx.float32),
        "from_latent.bias": mx.zeros((model_channels[0],), dtype=mx.float32),
        "output_layer.weight": mx.zeros((out_channels, model_channels[-1]), dtype=mx.float32),
        "output_layer.bias": mx.zeros((out_channels,), dtype=mx.float32),
        "blocks.0.0.conv.weight": mx.zeros((model_channels[0], 3, 3, 3, model_channels[0]), dtype=mx.float32),
        "blocks.0.0.conv.bias": mx.zeros((model_channels[0],), dtype=mx.float32),
        "blocks.0.0.norm.weight": mx.ones((model_channels[0],), dtype=mx.float32),
        "blocks.0.0.norm.bias": mx.zeros((model_channels[0],), dtype=mx.float32),
        "blocks.0.0.mlp.0.weight": mx.zeros((model_channels[0] * 4, model_channels[0]), dtype=mx.float32),
        "blocks.0.0.mlp.0.bias": mx.zeros((model_channels[0] * 4,), dtype=mx.float32),
        "blocks.0.0.mlp.2.weight": mx.zeros((model_channels[0], model_channels[0] * 4), dtype=mx.float32),
        "blocks.0.0.mlp.2.bias": mx.zeros((model_channels[0],), dtype=mx.float32),
        "blocks.0.1.norm1.weight": mx.ones((model_channels[0],), dtype=mx.float32),
        "blocks.0.1.norm1.bias": mx.zeros((model_channels[0],), dtype=mx.float32),
        "blocks.0.1.conv1.weight": mx.zeros((model_channels[1] * 8, 3, 3, 3, model_channels[0]), dtype=mx.float32),
        "blocks.0.1.conv1.bias": mx.zeros((model_channels[1] * 8,), dtype=mx.float32),
        "blocks.0.1.conv2.weight": mx.zeros((model_channels[1], 3, 3, 3, model_channels[1]), dtype=mx.float32),
        "blocks.0.1.conv2.bias": mx.zeros((model_channels[1],), dtype=mx.float32),
        "blocks.0.1.to_subdiv.weight": mx.zeros((8, model_channels[0]), dtype=mx.float32),
        "blocks.0.1.to_subdiv.bias": mx.ones((8,), dtype=mx.float32),
    }
    if not skip_encoder:
        encoder_latent_channels = 32
        to_latent_weight = np.zeros((2 * encoder_latent_channels, 6), dtype=np.float32)
        to_latent_weight[:6, :6] = np.eye(6, dtype=np.float32)
        tensors["input_layer.weight"] = mx.eye(6, dtype=mx.float32)
        tensors["input_layer.bias"] = mx.zeros((6,), dtype=mx.float32)
        tensors["to_latent.weight"] = mx.array(to_latent_weight)
        tensors["to_latent.bias"] = mx.zeros((2 * encoder_latent_channels,), dtype=mx.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path)


def _write_texture_decoder_config(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "name": "SparseUnetVaeDecoder",
        "args": {
            "model_channels": [16, 8],
            "latent_channels": 32,
            "num_blocks": [1, 0],
            "block_type": ["SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d"],
            "up_block_type": ["SparseResBlockC2S3d"],
            "use_fp16": False,
            "out_channels": 6,
            "pred_subdiv": False,
        },
    }))


def _write_texture_decoder_checkpoint(path):
    model_channels = (16, 8)
    latent_channels = 32
    out_channels = 6
    tensors = {
        "from_latent.weight": mx.ones((model_channels[0], latent_channels), dtype=mx.float32),
        "from_latent.bias": mx.zeros((model_channels[0],), dtype=mx.float32),
        "output_layer.weight": mx.zeros((out_channels, model_channels[-1]), dtype=mx.float32),
        "output_layer.bias": mx.zeros((out_channels,), dtype=mx.float32),
        "blocks.0.0.conv.weight": mx.zeros((model_channels[0], 3, 3, 3, model_channels[0]), dtype=mx.float32),
        "blocks.0.0.conv.bias": mx.zeros((model_channels[0],), dtype=mx.float32),
        "blocks.0.0.norm.weight": mx.ones((model_channels[0],), dtype=mx.float32),
        "blocks.0.0.norm.bias": mx.zeros((model_channels[0],), dtype=mx.float32),
        "blocks.0.0.mlp.0.weight": mx.zeros((model_channels[0] * 4, model_channels[0]), dtype=mx.float32),
        "blocks.0.0.mlp.0.bias": mx.zeros((model_channels[0] * 4,), dtype=mx.float32),
        "blocks.0.0.mlp.2.weight": mx.zeros((model_channels[0], model_channels[0] * 4), dtype=mx.float32),
        "blocks.0.0.mlp.2.bias": mx.zeros((model_channels[0],), dtype=mx.float32),
        "blocks.0.1.norm1.weight": mx.ones((model_channels[0],), dtype=mx.float32),
        "blocks.0.1.norm1.bias": mx.zeros((model_channels[0],), dtype=mx.float32),
        "blocks.0.1.conv1.weight": mx.zeros((model_channels[1] * 8, 3, 3, 3, model_channels[0]), dtype=mx.float32),
        "blocks.0.1.conv1.bias": mx.zeros((model_channels[1] * 8,), dtype=mx.float32),
        "blocks.0.1.conv2.weight": mx.zeros((model_channels[1], 3, 3, 3, model_channels[1]), dtype=mx.float32),
        "blocks.0.1.conv2.bias": mx.zeros((model_channels[1],), dtype=mx.float32),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path)


def _write_encoder_config(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "name": "FlexiDualGridVaeEncoder",
        "args": {
            "in_channels": 6,
            "latent_channels": 32,
            "model_channels": [6],
            "num_blocks": [0],
            "block_type": ["SparseConvNeXtBlock3d"],
            "down_block_type": [],
            "use_fp16": False,
        },
    }))


def _write_dinov3_root(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "mlx_spatial_fake_dinov3_vitl16",
        "image_size": 512,
        "patch_size": 16,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "layer_norm_eps": 1e-6,
        "use_swiglu_ffn": True,
        "num_register_tokens": 4,
        "mlx_spatial_fake_conditioning": True,
    }
    (root / "config.json").write_text(json.dumps(config))
    tensors = {
        "embeddings.patch_embeddings.weight": mx.zeros((1024, 3, 16, 16), dtype=mx.float32),
        "layer.0.norm.weight": mx.ones((1024,), dtype=mx.float32),
    }
    save_file(tensors, root / "model.safetensors")


def _write_rgb_image(path: Path):
    img = Image.new("RGBA", (256, 256), (128, 64, 200, 220))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _write_obj_mesh(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "v -0.3 -0.3 -0.3\nv 0.3 -0.3 -0.3\nv -0.3 0.3 -0.3\n"
        "v 0.3 0.3 -0.3\nv -0.3 -0.3 0.3\nv 0.3 -0.3 0.3\n"
        "v -0.3 0.3 0.3\nv 0.3 0.3 0.3\n"
        "f 1 3 2\nf 2 3 4\nf 5 6 7\nf 6 8 7\n"
        "f 1 5 3\nf 3 5 7\nf 2 4 6\nf 4 8 6\n"
        "f 1 2 5\nf 2 6 5\nf 3 7 4\nf 4 7 8\n"
    )


def _write_fixture_outputs_root(tmp_path: Path):
    """Write minimal assets needed by the texturing pipeline (no real compute)."""
    root = tmp_path / "fixture_weights/trellis2"
    dinov3_root = tmp_path / "fixture_weights/dinov3"
    rmbg_root = tmp_path / "fixture_weights/rmbg2"
    img_path = tmp_path / "fixture_inputs/demo.png"
    mesh_path = tmp_path / "fixture_inputs/cube.obj"
    outputs_dir = Path("outputs/fixture_textured")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / "fixture_textured.glb"

    _write_texturing_root(root)
    _write_dinov3_root(dinov3_root)
    _write_rgb_image(img_path)
    _write_obj_mesh(mesh_path)

    return root, dinov3_root, rmbg_root, img_path, mesh_path, output_path


class TestTrellis2TexturingPipeline:
    def test_pipeline_instantiation_defaults(self):
        pipeline = Trellis2TexturingPipeline()
        assert pipeline.root == Path("weights/trellis2")
        assert pipeline.rmbg_root is None
        assert pipeline.dino_root is None
        assert pipeline.encoder_config_path is None
        assert pipeline.encoder_checkpoint_path is None

    def test_pipeline_instantiation_with_paths(self, tmp_path):
        root = tmp_path / "custom_weights"
        dino = tmp_path / "custom_dino"
        rmbg = tmp_path / "custom_rmbg"
        pipeline = Trellis2TexturingPipeline(
            root=root, dino_root=dino, rmbg_root=rmbg,
            encoder_config_path="my_encoder.json",
            encoder_checkpoint_path="ckpts/my_encoder.safetensors",
        )
        assert pipeline.root == root
        assert pipeline.dino_root == dino
        assert pipeline.rmbg_root == rmbg
        assert pipeline.encoder_config_path == "my_encoder.json"
        assert pipeline.encoder_checkpoint_path == "ckpts/my_encoder.safetensors"

    def test_run_rejects_non_glb_output(self, tmp_path):
        pipeline = Trellis2TexturingPipeline(root=tmp_path / "weights/trellis2")
        result = pipeline.run(
            tmp_path / "img.png",
            tmp_path / "mesh.obj",
            output_path=tmp_path / "output.txt",
        )
        assert not result.ready
        assert result.blocker is not None
        assert result.blocker.stage == "mesh-export"
        assert "only writes .glb" in result.blocker.reason

    def test_run_rejects_export_path_outside_outputs(self, tmp_path):
        pipeline = Trellis2TexturingPipeline(root=tmp_path / "weights/trellis2")
        _write_rgb_image(tmp_path / "img.png")
        _write_obj_mesh(tmp_path / "mesh.obj")
        result = pipeline.run(
            tmp_path / "img.png",
            tmp_path / "mesh.obj",
            output_path=tmp_path / "outside.glb",
        )
        assert not result.ready
        assert result.blocker is not None
        assert result.blocker.stage == "mesh-export"

    def test_run_rejects_missing_image(self, tmp_path):
        pipeline = Trellis2TexturingPipeline(root=tmp_path / "weights/trellis2")
        output = tmp_path / "outputs/test.glb"
        output.parent.mkdir(parents=True, exist_ok=True)
        result = pipeline.run(
            tmp_path / "nonexistent.png",
            tmp_path / "mesh.obj",
            output_path=output,
        )
        assert not result.ready
        assert result.blocker is not None
        assert result.blocker.stage == "input-image"

    def test_run_rejects_missing_mesh(self, tmp_path):
        pipeline = Trellis2TexturingPipeline(root=tmp_path / "weights/trellis2")
        _write_rgb_image(tmp_path / "test.png")
        output = tmp_path / "outputs/test.glb"
        output.parent.mkdir(parents=True, exist_ok=True)
        result = pipeline.run(
            tmp_path / "test.png",
            tmp_path / "nonexistent.obj",
            output_path=output,
        )
        assert not result.ready
        assert result.blocker is not None
        assert result.blocker.stage == "mesh-load"

    def test_run_rejects_bad_grid_size(self, tmp_path):
        pipeline = Trellis2TexturingPipeline(root=tmp_path / "weights/trellis2")
        _write_rgb_image(tmp_path / "test.png")
        _write_obj_mesh(tmp_path / "cube.obj")
        output = tmp_path / "outputs/test.glb"
        output.parent.mkdir(parents=True, exist_ok=True)
        result = pipeline.run(
            tmp_path / "test.png",
            tmp_path / "cube.obj",
            output_path=output,
            grid_size=0,
        )
        assert not result.ready
        assert result.blocker is not None
        assert result.blocker.stage == "mesh-preprocess"

    def test_run_with_fixture_assets_produces_textured_glb(self, tmp_path):
        root, dinov3_root, rmbg_root, img_path, mesh_path, output_path = _write_fixture_outputs_root(tmp_path)

        pipeline = Trellis2TexturingPipeline(
            root=root,
            dino_root=dinov3_root,
            rmbg_root=rmbg_root,
        )
        result = pipeline.run(
            img_path,
            mesh_path,
            output_path=output_path,
            pipeline_type="1024",
            seed=42,
            grid_size=16,
            slat_steps=1,
            glb_target_faces=100,
        )

        assert isinstance(result, Trellis2TexturingResult)
        if result.ready:
            assert result.artifact is not None
            assert result.artifact.format == "glb"
            assert output_path.is_file()
            payload = output_path.read_bytes()
            assert payload[:4] == b"glTF"
            _verify_textured_glb_channels(payload)
        else:
            assert result.blocker is not None
            assert result.blocker.stage in {
                "mesh-export", "image-conditioning", "fdg-encoder",
                "shape-decoder", "texture-slat", "texture-decoder",
            }

    def test_run_with_fixture_assets_512_pipeline_type(self, tmp_path):
        root, dinov3_root, rmbg_root, img_path, mesh_path, output_path = _write_fixture_outputs_root(tmp_path)

        pipeline = Trellis2TexturingPipeline(
            root=root,
            dino_root=dinov3_root,
            rmbg_root=rmbg_root,
        )
        result = pipeline.run(
            img_path,
            mesh_path,
            output_path=output_path,
            pipeline_type="512",
            seed=42,
            grid_size=16,
            slat_steps=1,
            glb_target_faces=100,
        )

        assert isinstance(result, Trellis2TexturingResult)
        if result.ready:
            assert result.artifact is not None
            assert result.artifact.format == "glb"
            assert output_path.is_file()
            _verify_textured_glb_channels(output_path.read_bytes())
        else:
            assert result.blocker is not None
            assert result.blocker.stage in {
                "mesh-export", "image-conditioning", "fdg-encoder",
                "shape-decoder", "texture-slat", "texture-decoder",
            }

    def test_run_missing_encoder_config_is_blocked(self, tmp_path):
        root = tmp_path / "weights/trellis2"
        _write_texturing_root(root, skip_encoder_config=True)
        _write_dinov3_root(tmp_path / "weights/dinov3")
        _write_rgb_image(tmp_path / "test.png")
        _write_obj_mesh(tmp_path / "cube.obj")
        output = tmp_path / "outputs/test.glb"
        output.parent.mkdir(parents=True, exist_ok=True)

        pipeline = Trellis2TexturingPipeline(
            root=root,
            dino_root=tmp_path / "weights/dinov3",
        )
        result = pipeline.run(
            tmp_path / "test.png",
            tmp_path / "cube.obj",
            output_path=output,
            pipeline_type="1024",
            grid_size=16,
            slat_steps=1,
        )
        assert not result.ready
        assert result.blocker is not None

    def test_run_missing_pipeline_config_is_blocked(self, tmp_path):
        img = tmp_path / "test.png"
        mesh = tmp_path / "cube.obj"
        _write_rgb_image(img)
        _write_obj_mesh(mesh)
        output = tmp_path / "outputs/test.glb"
        output.parent.mkdir(parents=True, exist_ok=True)

        pipeline = Trellis2TexturingPipeline(root=tmp_path / "nonexistent_weights")
        result = pipeline.run(img, mesh, output_path=output)
        assert not result.ready
        assert result.blocker is not None

    def test_run_bad_slat_steps_is_blocked(self, tmp_path):
        root = tmp_path / "weights/trellis2"
        _write_texturing_root(root)
        _write_dinov3_root(tmp_path / "weights/dinov3")
        _write_rgb_image(tmp_path / "test.png")
        _write_obj_mesh(tmp_path / "cube.obj")
        output = tmp_path / "outputs/test.glb"
        output.parent.mkdir(parents=True, exist_ok=True)

        pipeline = Trellis2TexturingPipeline(
            root=root,
            dino_root=tmp_path / "weights/dinov3",
        )
        result = pipeline.run(
            tmp_path / "test.png",
            tmp_path / "cube.obj",
            output_path=output,
            slat_steps=0,
        )
        assert not result.ready
        assert result.blocker is not None

    def test_run_bad_pipeline_type_is_blocked(self, tmp_path):
        root = tmp_path / "weights/trellis2"
        _write_texturing_root(root)
        _write_dinov3_root(tmp_path / "weights/dinov3")
        _write_rgb_image(tmp_path / "test.png")
        _write_obj_mesh(tmp_path / "cube.obj")
        output = tmp_path / "outputs/test.glb"
        output.parent.mkdir(parents=True, exist_ok=True)

        pipeline = Trellis2TexturingPipeline(
            root=root,
            dino_root=tmp_path / "weights/dinov3",
        )
        result = pipeline.run(
            tmp_path / "test.png",
            tmp_path / "cube.obj",
            output_path=output,
            pipeline_type="invalid",
        )
        assert not result.ready
        assert result.blocker is not None


class TestLoadObjMesh:
    def test_loads_simple_cube(self, tmp_path):
        path = tmp_path / "cube.obj"
        _write_obj_mesh(path)
        verts, faces = _load_obj_mesh(path)
        assert verts.shape == (8, 3)
        assert faces.shape[0] > 0
        assert faces.shape[1] == 3

    def test_rejects_empty_file(self, tmp_path):
        path = tmp_path / "empty.obj"
        path.write_text("")
        with pytest.raises(ValueError, match="contains no vertices"):
            _load_obj_mesh(path)

    def test_rejects_no_faces(self, tmp_path):
        path = tmp_path / "verts_only.obj"
        path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\n")
        with pytest.raises(ValueError, match="contains no triangular faces"):
            _load_obj_mesh(path)

    def test_handles_quad_faces(self, tmp_path):
        path = tmp_path / "quad.obj"
        path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
        verts, faces = _load_obj_mesh(path)
        assert faces.shape == (2, 3)
        assert verts.shape == (4, 3)

    def test_handles_negative_indices(self, tmp_path):
        path = tmp_path / "relative.obj"
        path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n")
        verts, faces = _load_obj_mesh(path)
        assert faces.shape == (1, 3)


def _glb_json(payload: bytes) -> dict:
    magic, version, total_length = struct.unpack_from("<III", payload, 0)
    assert magic == 0x46546C67
    assert version == 2
    json_length, json_type = struct.unpack_from("<I4s", payload, 12)
    assert json_type == b"JSON"
    document = payload[20 : 20 + json_length].rstrip(b" ")
    return json.loads(document.decode("utf-8"))


def _verify_textured_glb_channels(payload: bytes):
    """Verify GLB contains baseColor and metallicRoughness textures."""
    doc = _glb_json(payload)
    assert "materials" in doc
    material = doc["materials"][0]
    pbr = material["pbrMetallicRoughness"]
    assert "baseColorTexture" in pbr
    assert "metallicRoughnessTexture" in pbr
    images = {img["name"]: img for img in doc["images"]}
    assert "baseColorTexture" in images
    assert "metallicRoughnessTexture" in images
