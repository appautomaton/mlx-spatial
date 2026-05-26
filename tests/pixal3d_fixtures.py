import json
from pathlib import Path

import mlx.core as mx
from safetensors.mlx import save_file

from mlx_spatial.model_assets import PIXAL3D_ASSETS


def write_fake_pixal3d_root(root: Path, *, sparse_steps: int = 12, shape_steps: int = 12, texture_steps: int = 12) -> Path:
    for relative_path in PIXAL3D_ASSETS.required_paths:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            if path.name == "pipeline.json":
                path.write_text(
                    json.dumps(
                        minimal_pixal3d_pipeline(
                            sparse_steps=sparse_steps,
                            shape_steps=shape_steps,
                            texture_steps=texture_steps,
                        )
                    ),
                    encoding="utf-8",
                )
            else:
                path.write_text(json.dumps({"name": path.stem, "args": {}}), encoding="utf-8")
        elif path.suffix == ".safetensors":
            save_file(minimal_pixal3d_checkpoint_tensors(), path)
    return root


def write_fake_pixal3d_dinov3_root(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mlx_spatial_fake_dinov3",
                "image_size": 64,
                "patch_size": 16,
                "hidden_size": 1024,
                "num_hidden_layers": 1,
                "num_attention_heads": 16,
                "intermediate_size": 4096,
                "layer_norm_eps": 0.000001,
                "use_swiglu_ffn": True,
                "num_register_tokens": 4,
                "rope_theta": 100.0,
                "pos_embed_rescale": 2.0,
                "mlx_spatial_fake_conditioning": True,
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "embeddings.patch_embeddings.weight": mx.ones((1024, 3, 16, 16), dtype=mx.float32),
            "layer.0.norm1.weight": mx.ones((1024,), dtype=mx.float32),
            "layer.0.attention.query.weight": mx.ones((1024, 1024), dtype=mx.float32),
        },
        root / "model.safetensors",
    )
    return root


def write_fake_pixal3d_sparse_flow_root(
    root: Path,
    *,
    proj_in_channels: int = 3,
    sparse_steps: int = 1,
) -> Path:
    write_fake_pixal3d_root(root, sparse_steps=sparse_steps)
    (root / "ckpts/ss_flow_img_dit_1_3B_64_bf16.json").write_text(
        json.dumps(
            {
                "name": "SparseStructureFlowModel",
                "args": {
                    "resolution": 16,
                    "in_channels": 2,
                    "out_channels": 2,
                    "model_channels": 6,
                    "cond_channels": proj_in_channels,
                    "num_blocks": 1,
                    "num_heads": 1,
                    "mlp_ratio": 2.0,
                    "pe_mode": "rope",
                    "share_mod": True,
                    "initialization": "scaled",
                    "qk_rms_norm": True,
                    "qk_rms_norm_cross": True,
                    "image_attn_mode": "proj",
                    "proj_in_channels": proj_in_channels,
                    "dtype": "bfloat16",
                },
            }
        ),
        encoding="utf-8",
    )
    save_file(
        _tiny_sparse_flow_checkpoint(
            model_channels=6,
            in_channels=2,
            out_channels=2,
            cond_channels=proj_in_channels,
            proj_in_channels=proj_in_channels,
        ),
        root / "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
    )
    return root


def write_fake_pixal3d_sparse_decoder_root(
    root: Path,
    *,
    proj_in_channels: int = 3,
    sparse_steps: int = 1,
) -> Path:
    write_fake_pixal3d_sparse_flow_root(root, proj_in_channels=proj_in_channels, sparse_steps=sparse_steps)
    (root / "ckpts/ss_dec_conv3d_16l8_fp16.json").write_text(
        json.dumps(
            {
                "name": "SparseStructureDecoder",
                "args": {
                    "out_channels": 1,
                    "latent_channels": 2,
                    "num_res_blocks": 0,
                    "channels": [2],
                    "num_res_blocks_middle": 0,
                    "norm_type": "layer",
                    "use_fp16": False,
                },
            }
        ),
        encoding="utf-8",
    )
    save_file(
        _tiny_sparse_decoder_checkpoint(channels=2, latent_channels=2, out_channels=1),
        root / "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
    )
    return root


def write_fake_pixal3d_shape_slat_root(
    root: Path,
    *,
    proj_in_channels: int = 3,
    sparse_steps: int = 1,
    shape_steps: int = 1,
) -> Path:
    write_fake_pixal3d_sparse_decoder_root(root, proj_in_channels=proj_in_channels, sparse_steps=sparse_steps)
    (root / "pipeline.json").write_text(
        json.dumps(minimal_pixal3d_pipeline(sparse_steps=sparse_steps, shape_steps=shape_steps)),
        encoding="utf-8",
    )
    (root / "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json").write_text(
        json.dumps(
            {
                "name": "ElasticSLatFlowModel",
                "args": {
                    "resolution": 32,
                    "in_channels": 32,
                    "out_channels": 32,
                    "model_channels": 6,
                    "cond_channels": proj_in_channels,
                    "num_blocks": 1,
                    "num_heads": 1,
                    "mlp_ratio": 2.0,
                    "pe_mode": "rope",
                    "share_mod": True,
                    "initialization": "scaled",
                    "qk_rms_norm": True,
                    "qk_rms_norm_cross": True,
                    "image_attn_mode": "proj",
                    "proj_in_channels": proj_in_channels * 2,
                    "dtype": "bfloat16",
                },
            }
        ),
        encoding="utf-8",
    )
    save_file(
        _tiny_slat_checkpoint(
            model_channels=6,
            in_channels=32,
            out_channels=32,
            cond_channels=proj_in_channels,
            proj_in_channels=proj_in_channels * 2,
        ),
        root / "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
    )
    return root


def write_fake_pixal3d_shape_hr_root(
    root: Path,
    *,
    proj_in_channels: int = 3,
    sparse_steps: int = 1,
    shape_steps: int = 1,
) -> Path:
    write_fake_pixal3d_shape_slat_root(
        root,
        proj_in_channels=proj_in_channels,
        sparse_steps=sparse_steps,
        shape_steps=shape_steps,
    )
    (root / "ckpts/shape_dec_next_dc_f16c32_fp16.json").write_text(
        json.dumps(_tiny_shape_decoder_config()),
        encoding="utf-8",
    )
    save_file(
        _tiny_shape_decoder_checkpoint(),
        root / "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
    )
    (root / "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json").write_text(
        json.dumps(
            {
                "name": "ElasticSLatFlowModel",
                "args": {
                    "resolution": 64,
                    "in_channels": 32,
                    "out_channels": 32,
                    "model_channels": 6,
                    "cond_channels": proj_in_channels,
                    "num_blocks": 1,
                    "num_heads": 1,
                    "mlp_ratio": 2.0,
                    "pe_mode": "rope",
                    "share_mod": True,
                    "initialization": "scaled",
                    "qk_rms_norm": True,
                    "qk_rms_norm_cross": True,
                    "image_attn_mode": "proj",
                    "proj_in_channels": proj_in_channels * 2,
                    "dtype": "bfloat16",
                },
            }
        ),
        encoding="utf-8",
    )
    save_file(
        _tiny_slat_checkpoint(
            model_channels=6,
            in_channels=32,
            out_channels=32,
            cond_channels=proj_in_channels,
            proj_in_channels=proj_in_channels * 2,
        ),
        root / "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
    )
    return root


def write_fake_pixal3d_texture_slat_root(
    root: Path,
    *,
    proj_in_channels: int = 3,
    sparse_steps: int = 1,
    shape_steps: int = 1,
    texture_steps: int = 1,
) -> Path:
    write_fake_pixal3d_shape_hr_root(
        root,
        proj_in_channels=proj_in_channels,
        sparse_steps=sparse_steps,
        shape_steps=shape_steps,
    )
    (root / "pipeline.json").write_text(
        json.dumps(
            minimal_pixal3d_pipeline(
                sparse_steps=sparse_steps,
                shape_steps=shape_steps,
                texture_steps=texture_steps,
            )
        ),
        encoding="utf-8",
    )
    (root / "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json").write_text(
        json.dumps(
            {
                "name": "ElasticSLatFlowModel",
                "args": {
                    "resolution": 64,
                    "in_channels": 64,
                    "out_channels": 32,
                    "model_channels": 6,
                    "cond_channels": proj_in_channels,
                    "num_blocks": 1,
                    "num_heads": 1,
                    "mlp_ratio": 2.0,
                    "pe_mode": "rope",
                    "share_mod": True,
                    "initialization": "scaled",
                    "qk_rms_norm": True,
                    "qk_rms_norm_cross": True,
                    "image_attn_mode": "proj",
                    "proj_in_channels": proj_in_channels * 2,
                    "dtype": "bfloat16",
                },
            }
        ),
        encoding="utf-8",
    )
    save_file(
        _tiny_slat_checkpoint(
            model_channels=6,
            in_channels=64,
            out_channels=32,
            cond_channels=proj_in_channels,
            proj_in_channels=proj_in_channels * 2,
        ),
        root / "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
    )
    return root


def minimal_pixal3d_pipeline(*, sparse_steps: int = 12, shape_steps: int = 12, texture_steps: int = 12):
    return {
        "args": {
            "models": {
                "sparse_structure_decoder": "ckpts/ss_dec_conv3d_16l8_fp16",
                "sparse_structure_flow_model": "ckpts/ss_flow_img_dit_1_3B_64_bf16",
                "shape_slat_decoder": "ckpts/shape_dec_next_dc_f16c32_fp16",
                "shape_slat_flow_model_512": "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16",
                "shape_slat_flow_model_1024": "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16",
                "tex_slat_decoder": "ckpts/tex_dec_next_dc_f16c32_fp16",
                "tex_slat_flow_model_1024": "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
            },
            "default_pipeline_type": "1536_cascade",
            "sparse_structure_sampler": _sampler(7.5, 0.7, [0.6, 1.0], 5.0, steps=sparse_steps),
            "shape_slat_sampler": _sampler(7.5, 0.5, [0.6, 1.0], 3.0, steps=shape_steps),
            "tex_slat_sampler": _sampler(1.0, 0.0, [0.6, 0.9], 3.0, steps=texture_steps),
            "shape_slat_normalization": {"mean": [0.0] * 32, "std": [1.0] * 32},
            "tex_slat_normalization": {"mean": [0.0] * 32, "std": [1.0] * 32},
        }
    }


def minimal_pixal3d_checkpoint_tensors():
    return {
        "blocks.0.norm2.weight": mx.array([1.0], dtype=mx.float32),
        "blocks.0.cross_attn.proj_linear.weight": mx.array([2.0], dtype=mx.float32),
        "layers.0.weight": mx.array([3.0], dtype=mx.float32),
        "out_layer.weight": mx.array([4.0], dtype=mx.float32),
        "blocks.0.0.norm.weight": mx.array([5.0], dtype=mx.float32),
    }


def _sampler(guidance_strength, guidance_rescale, interval, rescale_t, *, steps=12):
    return {
        "name": "FlowEulerGuidanceIntervalSampler",
        "args": {"sigma_min": 1e-5},
        "params": {
            "steps": steps,
            "guidance_strength": guidance_strength,
            "guidance_rescale": guidance_rescale,
            "guidance_interval": interval,
            "rescale_t": rescale_t,
        },
    }


def _tiny_sparse_flow_checkpoint(
    *,
    model_channels: int,
    in_channels: int,
    out_channels: int,
    cond_channels: int,
    proj_in_channels: int,
) -> dict[str, mx.array]:
    prefix = "blocks.0"
    head_dim = model_channels
    return {
        "input_layer.weight": mx.ones((model_channels, in_channels), dtype=mx.float32),
        "input_layer.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "out_layer.weight": mx.zeros((out_channels, model_channels), dtype=mx.float32),
        "out_layer.bias": mx.zeros((out_channels,), dtype=mx.float32),
        "t_embedder.mlp.0.weight": mx.zeros((model_channels, 256), dtype=mx.float32),
        "t_embedder.mlp.0.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "t_embedder.mlp.2.weight": mx.zeros((model_channels, model_channels), dtype=mx.float32),
        "t_embedder.mlp.2.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "adaLN_modulation.1.weight": mx.zeros((model_channels * 6, model_channels), dtype=mx.float32),
        "adaLN_modulation.1.bias": mx.zeros((model_channels * 6,), dtype=mx.float32),
        f"{prefix}.modulation": mx.zeros((model_channels * 6,), dtype=mx.float32),
        f"{prefix}.norm2.weight": mx.ones((model_channels,), dtype=mx.float32),
        f"{prefix}.norm2.bias": mx.zeros((model_channels,), dtype=mx.float32),
        f"{prefix}.self_attn.to_qkv.weight": mx.zeros((model_channels * 3, model_channels), dtype=mx.float32),
        f"{prefix}.self_attn.to_qkv.bias": mx.zeros((model_channels * 3,), dtype=mx.float32),
        f"{prefix}.self_attn.q_rms_norm.gamma": mx.ones((1, head_dim), dtype=mx.float32),
        f"{prefix}.self_attn.k_rms_norm.gamma": mx.ones((1, head_dim), dtype=mx.float32),
        f"{prefix}.self_attn.to_out.weight": mx.zeros((model_channels, model_channels), dtype=mx.float32),
        f"{prefix}.self_attn.to_out.bias": mx.zeros((model_channels,), dtype=mx.float32),
        f"{prefix}.cross_attn.cross_attn_block.to_q.weight": mx.zeros((model_channels, model_channels), dtype=mx.float32),
        f"{prefix}.cross_attn.cross_attn_block.to_q.bias": mx.zeros((model_channels,), dtype=mx.float32),
        f"{prefix}.cross_attn.cross_attn_block.to_kv.weight": mx.zeros((model_channels * 2, cond_channels), dtype=mx.float32),
        f"{prefix}.cross_attn.cross_attn_block.to_kv.bias": mx.zeros((model_channels * 2,), dtype=mx.float32),
        f"{prefix}.cross_attn.cross_attn_block.q_rms_norm.gamma": mx.ones((1, head_dim), dtype=mx.float32),
        f"{prefix}.cross_attn.cross_attn_block.k_rms_norm.gamma": mx.ones((1, head_dim), dtype=mx.float32),
        f"{prefix}.cross_attn.cross_attn_block.to_out.weight": mx.zeros((model_channels, model_channels), dtype=mx.float32),
        f"{prefix}.cross_attn.cross_attn_block.to_out.bias": mx.zeros((model_channels,), dtype=mx.float32),
        f"{prefix}.cross_attn.proj_linear.weight": mx.zeros((model_channels, proj_in_channels), dtype=mx.float32),
        f"{prefix}.cross_attn.proj_linear.bias": mx.zeros((model_channels,), dtype=mx.float32),
        f"{prefix}.mlp.mlp.0.weight": mx.zeros((model_channels * 2, model_channels), dtype=mx.float32),
        f"{prefix}.mlp.mlp.0.bias": mx.zeros((model_channels * 2,), dtype=mx.float32),
        f"{prefix}.mlp.mlp.2.weight": mx.zeros((model_channels, model_channels * 2), dtype=mx.float32),
        f"{prefix}.mlp.mlp.2.bias": mx.zeros((model_channels,), dtype=mx.float32),
    }


def _tiny_sparse_decoder_checkpoint(
    *,
    channels: int,
    latent_channels: int,
    out_channels: int,
) -> dict[str, mx.array]:
    return {
        "input_layer.weight": mx.zeros((channels, latent_channels, 3, 3, 3), dtype=mx.float32),
        "input_layer.bias": mx.zeros((channels,), dtype=mx.float32),
        "out_layer.0.weight": mx.ones((channels,), dtype=mx.float32),
        "out_layer.0.bias": mx.zeros((channels,), dtype=mx.float32),
        "out_layer.2.weight": mx.zeros((out_channels, channels, 3, 3, 3), dtype=mx.float32),
        "out_layer.2.bias": mx.ones((out_channels,), dtype=mx.float32),
    }


def _tiny_slat_checkpoint(
    *,
    model_channels: int,
    in_channels: int,
    out_channels: int,
    cond_channels: int,
    proj_in_channels: int,
) -> dict[str, mx.array]:
    prefix = "blocks.0"
    head_dim = model_channels
    return {
        "input_layer.weight": mx.ones((model_channels, in_channels), dtype=mx.float32),
        "input_layer.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "out_layer.weight": mx.zeros((out_channels, model_channels), dtype=mx.float32),
        "out_layer.bias": mx.zeros((out_channels,), dtype=mx.float32),
        "t_embedder.mlp.0.weight": mx.zeros((model_channels, 256), dtype=mx.float32),
        "t_embedder.mlp.0.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "t_embedder.mlp.2.weight": mx.zeros((model_channels, model_channels), dtype=mx.float32),
        "t_embedder.mlp.2.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "adaLN_modulation.1.weight": mx.zeros((model_channels * 6, model_channels), dtype=mx.float32),
        "adaLN_modulation.1.bias": mx.zeros((model_channels * 6,), dtype=mx.float32),
        f"{prefix}.modulation": mx.zeros((model_channels * 6,), dtype=mx.float32),
        f"{prefix}.norm2.weight": mx.ones((model_channels,), dtype=mx.float32),
        f"{prefix}.norm2.bias": mx.zeros((model_channels,), dtype=mx.float32),
        f"{prefix}.self_attn.to_qkv.weight": mx.zeros((model_channels * 3, model_channels), dtype=mx.float32),
        f"{prefix}.self_attn.to_qkv.bias": mx.zeros((model_channels * 3,), dtype=mx.float32),
        f"{prefix}.self_attn.q_rms_norm.gamma": mx.ones((1, head_dim), dtype=mx.float32),
        f"{prefix}.self_attn.k_rms_norm.gamma": mx.ones((1, head_dim), dtype=mx.float32),
        f"{prefix}.self_attn.to_out.weight": mx.zeros((model_channels, model_channels), dtype=mx.float32),
        f"{prefix}.self_attn.to_out.bias": mx.zeros((model_channels,), dtype=mx.float32),
        f"{prefix}.cross_attn.cross_attn_block.to_q.weight": mx.zeros((model_channels, model_channels), dtype=mx.float32),
        f"{prefix}.cross_attn.cross_attn_block.to_q.bias": mx.zeros((model_channels,), dtype=mx.float32),
        f"{prefix}.cross_attn.cross_attn_block.to_kv.weight": mx.zeros((model_channels * 2, cond_channels), dtype=mx.float32),
        f"{prefix}.cross_attn.cross_attn_block.to_kv.bias": mx.zeros((model_channels * 2,), dtype=mx.float32),
        f"{prefix}.cross_attn.cross_attn_block.q_rms_norm.gamma": mx.ones((1, head_dim), dtype=mx.float32),
        f"{prefix}.cross_attn.cross_attn_block.k_rms_norm.gamma": mx.ones((1, head_dim), dtype=mx.float32),
        f"{prefix}.cross_attn.cross_attn_block.to_out.weight": mx.zeros((model_channels, model_channels), dtype=mx.float32),
        f"{prefix}.cross_attn.cross_attn_block.to_out.bias": mx.zeros((model_channels,), dtype=mx.float32),
        f"{prefix}.cross_attn.proj_linear.weight": mx.zeros((model_channels, proj_in_channels), dtype=mx.float32),
        f"{prefix}.cross_attn.proj_linear.bias": mx.zeros((model_channels,), dtype=mx.float32),
        f"{prefix}.mlp.mlp.0.weight": mx.zeros((model_channels * 2, model_channels), dtype=mx.float32),
        f"{prefix}.mlp.mlp.0.bias": mx.zeros((model_channels * 2,), dtype=mx.float32),
        f"{prefix}.mlp.mlp.2.weight": mx.zeros((model_channels, model_channels * 2), dtype=mx.float32),
        f"{prefix}.mlp.mlp.2.bias": mx.zeros((model_channels,), dtype=mx.float32),
    }


def _tiny_shape_decoder_config() -> dict:
    return {
        "name": "FlexiDualGridVaeDecoder",
        "args": {
            "model_channels": [16, 8, 8, 8, 8],
            "latent_channels": 32,
            "num_blocks": [0, 0, 0, 0, 0],
            "block_type": ["SparseConvNeXtBlock3d"] * 5,
            "up_block_type": ["SparseResBlockC2S3d"] * 4,
            "block_args": [{}, {}, {}, {}, {}],
            "use_fp16": False,
            "resolution": 512,
            "pred_subdiv": True,
        },
    }


def _tiny_shape_decoder_checkpoint() -> dict[str, mx.array]:
    model_channels = (16, 8, 8, 8, 8)
    tensors: dict[str, mx.array] = {
        "from_latent.weight": mx.ones((model_channels[0], 32), dtype=mx.float32),
        "from_latent.bias": mx.zeros((model_channels[0],), dtype=mx.float32),
        "output_layer.weight": mx.zeros((7, model_channels[-1]), dtype=mx.float32),
        "output_layer.bias": mx.zeros((7,), dtype=mx.float32),
    }
    for level, (in_channels, out_channels) in enumerate(zip(model_channels[:-1], model_channels[1:])):
        prefix = f"blocks.{level}.0"
        subdiv_bias = [-1.0] * 8
        subdiv_bias[0] = 1.0
        tensors.update(
            {
                f"{prefix}.norm1.weight": mx.ones((in_channels,), dtype=mx.float32),
                f"{prefix}.norm1.bias": mx.zeros((in_channels,), dtype=mx.float32),
                f"{prefix}.conv1.weight": mx.zeros((out_channels * 8, 3, 3, 3, in_channels), dtype=mx.float32),
                f"{prefix}.conv1.bias": mx.zeros((out_channels * 8,), dtype=mx.float32),
                f"{prefix}.conv2.weight": mx.zeros((out_channels, 3, 3, 3, out_channels), dtype=mx.float32),
                f"{prefix}.conv2.bias": mx.zeros((out_channels,), dtype=mx.float32),
                f"{prefix}.to_subdiv.weight": mx.zeros((8, in_channels), dtype=mx.float32),
                f"{prefix}.to_subdiv.bias": mx.array(subdiv_bias, dtype=mx.float32),
            }
        )
    return tensors
