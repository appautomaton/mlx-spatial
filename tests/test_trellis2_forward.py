import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image
from safetensors.mlx import save_file

import mlx_spatial
from mlx_spatial.model_assets import TRELLIS2_ASSETS
from mlx_spatial.trellis2_forward import (
    Trellis2ConditioningConfig,
    Trellis2ForwardBlocker,
    Trellis2ForwardConfigResult,
    Trellis2ForwardTraceResult,
    Trellis2StageOutput,
    assess_dinov3_conditioning,
    default_dinov3_root,
    discover_trellis2_conditioning_config,
    dispatch_decode_latents_boundary,
    dispatch_shape_slat_boundary,
    dispatch_sparse_structure_decoder_boundary,
    dispatch_sparse_structure_boundary,
    dispatch_texture_slat_boundary,
    prepare_dinov3_image_tensor,
)
from mlx_spatial.trellis2_slat import select_texture_slat_route


def _write_trellis2_root(root: Path, *, conditioning_resolution: int | None = None):
    sparse_model_channels = 6
    sparse_in_channels = 2
    sparse_out_channels = 2
    sparse_cond_channels = 1024
    sparse_resolution = 2
    slat_model_channels = 6
    slat_in_channels = 32
    slat_out_channels = 32
    slat_cond_channels = 1024
    for asset_path in TRELLIS2_ASSETS.required_paths:
        path = root / asset_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if asset_path.endswith(".safetensors"):
            if "slat_flow_img2shape" in asset_path or "slat_flow_imgshape2tex" in asset_path:
                input_channels = 64 if "slat_flow_imgshape2tex" in asset_path else slat_in_channels
                tensors = {
                    "blocks.0.norm2.weight": mx.ones((slat_model_channels,), dtype=mx.float32),
                    "blocks.0.cross_attn.to_kv.weight": mx.zeros(
                        (slat_model_channels * 2, slat_cond_channels),
                        dtype=mx.float32,
                    ),
                    "input_layer.weight": mx.ones((slat_model_channels, input_channels), dtype=mx.float32),
                    "input_layer.bias": mx.zeros((slat_model_channels,), dtype=mx.float32),
                    "out_layer.weight": mx.zeros((slat_out_channels, slat_model_channels), dtype=mx.float32),
                    "out_layer.bias": mx.zeros((slat_out_channels,), dtype=mx.float32),
                    "t_embedder.mlp.0.weight": mx.zeros((slat_model_channels, 256), dtype=mx.float32),
                    "t_embedder.mlp.0.bias": mx.zeros((slat_model_channels,), dtype=mx.float32),
                    "t_embedder.mlp.2.weight": mx.zeros((slat_model_channels, slat_model_channels), dtype=mx.float32),
                    "t_embedder.mlp.2.bias": mx.zeros((slat_model_channels,), dtype=mx.float32),
                    "adaLN_modulation.1.weight": mx.zeros((slat_model_channels * 6, slat_model_channels), dtype=mx.float32),
                    "adaLN_modulation.1.bias": mx.zeros((slat_model_channels * 6,), dtype=mx.float32),
                    "blocks.0.modulation": mx.zeros((slat_model_channels * 6,), dtype=mx.float32),
                    "blocks.0.norm2.bias": mx.zeros((slat_model_channels,), dtype=mx.float32),
                    "blocks.0.self_attn.to_qkv.weight": mx.zeros(
                        (slat_model_channels * 3, slat_model_channels),
                        dtype=mx.float32,
                    ),
                    "blocks.0.self_attn.to_qkv.bias": mx.zeros((slat_model_channels * 3,), dtype=mx.float32),
                    "blocks.0.self_attn.q_rms_norm.gamma": mx.ones((1, slat_model_channels), dtype=mx.float32),
                    "blocks.0.self_attn.k_rms_norm.gamma": mx.ones((1, slat_model_channels), dtype=mx.float32),
                    "blocks.0.self_attn.to_out.weight": mx.zeros((slat_model_channels, slat_model_channels), dtype=mx.float32),
                    "blocks.0.self_attn.to_out.bias": mx.zeros((slat_model_channels,), dtype=mx.float32),
                    "blocks.0.cross_attn.to_q.weight": mx.zeros(
                        (slat_model_channels, slat_model_channels),
                        dtype=mx.float32,
                    ),
                    "blocks.0.cross_attn.to_q.bias": mx.zeros((slat_model_channels,), dtype=mx.float32),
                    "blocks.0.cross_attn.to_kv.bias": mx.zeros((slat_model_channels * 2,), dtype=mx.float32),
                    "blocks.0.cross_attn.q_rms_norm.gamma": mx.ones((1, slat_model_channels), dtype=mx.float32),
                    "blocks.0.cross_attn.k_rms_norm.gamma": mx.ones((1, slat_model_channels), dtype=mx.float32),
                    "blocks.0.cross_attn.to_out.weight": mx.zeros((slat_model_channels, slat_model_channels), dtype=mx.float32),
                    "blocks.0.cross_attn.to_out.bias": mx.zeros((slat_model_channels,), dtype=mx.float32),
                    "blocks.0.mlp.mlp.0.weight": mx.zeros(
                        (int(slat_model_channels * 2.0), slat_model_channels), dtype=mx.float32
                    ),
                    "blocks.0.mlp.mlp.0.bias": mx.zeros((int(slat_model_channels * 2.0),), dtype=mx.float32),
                    "blocks.0.mlp.mlp.2.weight": mx.zeros(
                        (slat_model_channels, int(slat_model_channels * 2.0)), dtype=mx.float32
                    ),
                    "blocks.0.mlp.mlp.2.bias": mx.zeros((slat_model_channels,), dtype=mx.float32),
                }
            else:
                tensors = {
                    "blocks.0.0.norm.weight": mx.array([0.0, 1.0], dtype=mx.float32),
                    "blocks.0.norm2.weight": mx.ones((sparse_model_channels,), dtype=mx.float32),
                    "blocks.0.cross_attn.to_kv.weight": mx.zeros(
                        (sparse_model_channels * 2, sparse_cond_channels),
                        dtype=mx.float32,
                    ),
                    "input_layer.weight": mx.ones((sparse_model_channels, sparse_in_channels), dtype=mx.float32),
                    "input_layer.bias": mx.zeros((sparse_model_channels,), dtype=mx.float32),
                    "t_embedder.mlp.0.weight": mx.zeros((sparse_model_channels, 256), dtype=mx.float32),
                    "t_embedder.mlp.0.bias": mx.zeros((sparse_model_channels,), dtype=mx.float32),
                    "t_embedder.mlp.2.weight": mx.zeros((sparse_model_channels, sparse_model_channels), dtype=mx.float32),
                    "t_embedder.mlp.2.bias": mx.zeros((sparse_model_channels,), dtype=mx.float32),
                    "adaLN_modulation.1.weight": mx.zeros((sparse_model_channels * 6, sparse_model_channels), dtype=mx.float32),
                    "adaLN_modulation.1.bias": mx.zeros((sparse_model_channels * 6,), dtype=mx.float32),
                    "blocks.0.modulation": mx.zeros((sparse_model_channels * 6,), dtype=mx.float32),
                    "blocks.0.norm2.bias": mx.zeros((sparse_model_channels,), dtype=mx.float32),
                    "blocks.0.self_attn.to_qkv.weight": mx.zeros(
                        (sparse_model_channels * 3, sparse_model_channels),
                        dtype=mx.float32,
                    ),
                    "blocks.0.self_attn.to_qkv.bias": mx.zeros((sparse_model_channels * 3,), dtype=mx.float32),
                    "blocks.0.self_attn.q_rms_norm.gamma": mx.ones((1, sparse_model_channels), dtype=mx.float32),
                    "blocks.0.self_attn.k_rms_norm.gamma": mx.ones((1, sparse_model_channels), dtype=mx.float32),
                    "blocks.0.self_attn.to_out.weight": mx.zeros(
                        (sparse_model_channels, sparse_model_channels),
                        dtype=mx.float32,
                    ),
                    "blocks.0.self_attn.to_out.bias": mx.zeros((sparse_model_channels,), dtype=mx.float32),
                    "blocks.0.cross_attn.to_q.weight": mx.zeros(
                        (sparse_model_channels, sparse_model_channels),
                        dtype=mx.float32,
                    ),
                    "blocks.0.cross_attn.to_q.bias": mx.zeros((sparse_model_channels,), dtype=mx.float32),
                    "blocks.0.cross_attn.to_kv.bias": mx.zeros((sparse_model_channels * 2,), dtype=mx.float32),
                    "out_layer.weight": mx.zeros((sparse_out_channels, sparse_model_channels), dtype=mx.float32),
                    "out_layer.bias": mx.zeros((sparse_out_channels,), dtype=mx.float32),
                    "blocks.0.cross_attn.q_rms_norm.gamma": mx.ones((1, sparse_model_channels), dtype=mx.float32),
                    "blocks.0.cross_attn.k_rms_norm.gamma": mx.ones((1, sparse_model_channels), dtype=mx.float32),
                    "blocks.0.cross_attn.to_out.weight": mx.zeros(
                        (sparse_model_channels, sparse_model_channels),
                        dtype=mx.float32,
                    ),
                    "blocks.0.cross_attn.to_out.bias": mx.zeros((sparse_model_channels,), dtype=mx.float32),
                    "blocks.0.mlp.mlp.0.weight": mx.zeros(
                        (int(sparse_model_channels * 2.0), sparse_model_channels),
                        dtype=mx.float32,
                    ),
                    "blocks.0.mlp.mlp.0.bias": mx.zeros((int(sparse_model_channels * 2.0),), dtype=mx.float32),
                    "blocks.0.mlp.mlp.2.weight": mx.zeros(
                        (sparse_model_channels, int(sparse_model_channels * 2.0)),
                        dtype=mx.float32,
                    ),
                    "blocks.0.mlp.mlp.2.bias": mx.zeros((sparse_model_channels,), dtype=mx.float32),
                }
            save_file(tensors, path)
        else:
            path.write_text("{}")
    image_args = {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"}
    if conditioning_resolution is not None:
        image_args["image_size"] = conditioning_resolution
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
            "sparse_structure_sampler": _sampler_config(guidance_rescale=0.7, rescale_t=5.0, interval=(0.6, 1.0)),
            "shape_slat_sampler": _sampler_config(guidance_rescale=0.5, rescale_t=3.0, interval=(0.6, 1.0)),
            "tex_slat_sampler": _sampler_config(
                guidance_strength=1.0,
                guidance_rescale=0.0,
                rescale_t=3.0,
                interval=(0.6, 0.9),
            ),
            "shape_slat_normalization": _normalization_config(),
            "tex_slat_normalization": _normalization_config(offset=1.0),
            "image_cond_model": {
                "name": "DinoV3FeatureExtractor",
                "args": image_args,
            },
            "default_pipeline_type": "1024_cascade",
        },
    }
    (root / "pipeline.json").write_text(json.dumps(pipeline))
    (root / "ckpts/ss_flow_img_dit_1_3B_64_bf16.json").write_text(
        json.dumps(
            {
                "name": "SparseStructureFlowModel",
                "args": {
                    "resolution": sparse_resolution,
                    "in_channels": sparse_in_channels,
                    "out_channels": sparse_out_channels,
                    "model_channels": sparse_model_channels,
                    "cond_channels": sparse_cond_channels,
                    "num_blocks": 1,
                    "num_heads": 1,
                    "mlp_ratio": 2.0,
                    "pe_mode": "rope",
                    "share_mod": True,
                    "initialization": "scaled",
                    "qk_rms_norm": True,
                    "qk_rms_norm_cross": True,
                    "dtype": "bfloat16",
                },
            }
        )
    )
    _write_slat_flow_config(
        root / "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json",
        resolution=32,
        in_channels=slat_in_channels,
        out_channels=slat_out_channels,
        model_channels=slat_model_channels,
        cond_channels=slat_cond_channels,
    )
    _write_slat_flow_config(
        root / "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json",
        resolution=64,
        in_channels=slat_in_channels,
        out_channels=slat_out_channels,
        model_channels=slat_model_channels,
        cond_channels=slat_cond_channels,
    )
    _write_slat_flow_config(
        root / "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json",
        resolution=32,
        in_channels=64,
        out_channels=slat_out_channels,
        model_channels=slat_model_channels,
        cond_channels=slat_cond_channels,
    )
    _write_slat_flow_config(
        root / "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json",
        resolution=64,
        in_channels=64,
        out_channels=slat_out_channels,
        model_channels=slat_model_channels,
        cond_channels=slat_cond_channels,
    )
    save_file(
        _slat_checkpoint_tensors(
            model_channels=slat_model_channels,
            in_channels=64,
            out_channels=slat_out_channels,
            cond_channels=slat_cond_channels,
        ),
        root / "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
    )
    _write_structured_decoder_config(
        root / "ckpts/shape_dec_next_dc_f16c32_fp16.json",
        name="FlexiDualGridVaeDecoder",
        out_channels=7,
    )
    _write_structured_decoder_config(
        root / "ckpts/tex_dec_next_dc_f16c32_fp16.json",
        name="SparseUnetVaeDecoder",
        out_channels=6,
        pred_subdiv=False,
    )
    save_file(
        _structured_decoder_tensors(model_channels=(16, 8), latent_channels=32, out_channels=7, pred_subdiv=True),
        root / "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
    )
    save_file(
        _structured_decoder_tensors(model_channels=(16, 8), latent_channels=32, out_channels=6, pred_subdiv=False),
        root / "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
    )


def _write_structured_decoder_config(
    path: Path,
    *,
    name: str,
    out_channels: int,
    pred_subdiv: bool | None = None,
):
    args = {
        "model_channels": [16, 8],
        "latent_channels": 32,
        "num_blocks": [1, 0],
        "block_type": ["SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d"],
        "up_block_type": ["SparseResBlockC2S3d"],
        "block_args": [{}, {}],
        "use_fp16": False,
    }
    if name == "FlexiDualGridVaeDecoder":
        args["resolution"] = 256
    else:
        args["out_channels"] = out_channels
        args["pred_subdiv"] = bool(pred_subdiv)
    path.write_text(json.dumps({"name": name, "args": args}))


def _structured_decoder_tensors(
    *,
    model_channels: tuple[int, int],
    latent_channels: int,
    out_channels: int,
    pred_subdiv: bool,
):
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
    if pred_subdiv:
        tensors["blocks.0.1.to_subdiv.weight"] = mx.zeros((8, model_channels[0]), dtype=mx.float32)
        tensors["blocks.0.1.to_subdiv.bias"] = mx.ones((8,), dtype=mx.float32)
    return tensors


def _slat_checkpoint_tensors(*, model_channels: int, in_channels: int, out_channels: int, cond_channels: int):
    intermediate_channels = int(model_channels * 2.0)
    return {
        "blocks.0.norm2.weight": mx.ones((model_channels,), dtype=mx.float32),
        "blocks.0.cross_attn.to_kv.weight": mx.zeros((model_channels * 2, cond_channels), dtype=mx.float32),
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
        "blocks.0.modulation": mx.zeros((model_channels * 6,), dtype=mx.float32),
        "blocks.0.norm2.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "blocks.0.self_attn.to_qkv.weight": mx.zeros((model_channels * 3, model_channels), dtype=mx.float32),
        "blocks.0.self_attn.to_qkv.bias": mx.zeros((model_channels * 3,), dtype=mx.float32),
        "blocks.0.self_attn.q_rms_norm.gamma": mx.ones((1, model_channels), dtype=mx.float32),
        "blocks.0.self_attn.k_rms_norm.gamma": mx.ones((1, model_channels), dtype=mx.float32),
        "blocks.0.self_attn.to_out.weight": mx.zeros((model_channels, model_channels), dtype=mx.float32),
        "blocks.0.self_attn.to_out.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "blocks.0.cross_attn.to_q.weight": mx.zeros((model_channels, model_channels), dtype=mx.float32),
        "blocks.0.cross_attn.to_q.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "blocks.0.cross_attn.to_kv.bias": mx.zeros((model_channels * 2,), dtype=mx.float32),
        "blocks.0.cross_attn.q_rms_norm.gamma": mx.ones((1, model_channels), dtype=mx.float32),
        "blocks.0.cross_attn.k_rms_norm.gamma": mx.ones((1, model_channels), dtype=mx.float32),
        "blocks.0.cross_attn.to_out.weight": mx.zeros((model_channels, model_channels), dtype=mx.float32),
        "blocks.0.cross_attn.to_out.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "blocks.0.mlp.mlp.0.weight": mx.zeros((intermediate_channels, model_channels), dtype=mx.float32),
        "blocks.0.mlp.mlp.0.bias": mx.zeros((intermediate_channels,), dtype=mx.float32),
        "blocks.0.mlp.mlp.2.weight": mx.zeros((model_channels, intermediate_channels), dtype=mx.float32),
        "blocks.0.mlp.mlp.2.bias": mx.zeros((model_channels,), dtype=mx.float32),
    }


def _write_slat_flow_config(
    path: Path,
    *,
    resolution: int,
    in_channels: int,
    out_channels: int,
    model_channels: int,
    cond_channels: int,
):
    path.write_text(
        json.dumps(
            {
                "name": "SLatFlowModel",
                "args": {
                    "resolution": resolution,
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "model_channels": model_channels,
                    "cond_channels": cond_channels,
                    "num_blocks": 1,
                    "num_heads": 1,
                    "mlp_ratio": 2.0,
                    "pe_mode": "rope",
                    "share_mod": True,
                    "initialization": "scaled",
                    "qk_rms_norm": True,
                    "qk_rms_norm_cross": True,
                    "dtype": "bfloat16",
                },
            }
        )
    )


def _sampler_config(
    *,
    guidance_strength: float = 7.5,
    guidance_rescale: float,
    rescale_t: float,
    interval: tuple[float, float],
):
    return {
        "name": "FlowEulerGuidanceIntervalSampler",
        "args": {"sigma_min": 1e-5},
        "params": {
            "steps": 12,
            "guidance_strength": guidance_strength,
            "guidance_rescale": guidance_rescale,
            "guidance_interval": list(interval),
            "rescale_t": rescale_t,
        },
    }


def _normalization_config(*, offset: float = 0.0):
    return {
        "mean": [offset + index for index in range(32)],
        "std": [offset + index + 0.5 for index in range(32)],
    }


def _write_alpha_image(path: Path):
    image = Image.new("RGBA", (8, 8), (10, 20, 30, 0))
    pixels = image.load()
    for y in range(2, 6):
        for x in range(2, 6):
            pixels[x, y] = (200, 100, 50, 255)
    image.save(path)


def _write_sparse_decoder_assets(root: Path, *, checkpoint: bool = True):
    base = root / "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
    base.parent.mkdir(parents=True, exist_ok=True)
    (base.with_suffix(".json")).write_text(
        json.dumps(
            {
                "name": "SparseStructureDecoder",
                "args": {
                    "out_channels": 1,
                    "latent_channels": 2,
                    "num_res_blocks": 1,
                    "channels": [4],
                    "num_res_blocks_middle": 1,
                    "norm_type": "layer",
                    "use_fp16": False,
                },
            }
        )
    )
    if checkpoint:
        save_file(
            {
                "input_layer.weight": mx.ones((4, 2, 3, 3, 3), dtype=mx.float32),
                "input_layer.bias": mx.zeros((4,), dtype=mx.float32),
                "middle_block.0.norm1.weight": mx.ones((4,), dtype=mx.float32),
                "middle_block.0.norm1.bias": mx.zeros((4,), dtype=mx.float32),
                "middle_block.0.conv1.weight": mx.zeros((4, 4, 3, 3, 3), dtype=mx.float32),
                "middle_block.0.conv1.bias": mx.zeros((4,), dtype=mx.float32),
                "middle_block.0.norm2.weight": mx.ones((4,), dtype=mx.float32),
                "middle_block.0.norm2.bias": mx.zeros((4,), dtype=mx.float32),
                "middle_block.0.conv2.weight": mx.zeros((4, 4, 3, 3, 3), dtype=mx.float32),
                "middle_block.0.conv2.bias": mx.zeros((4,), dtype=mx.float32),
                "blocks.0.norm1.weight": mx.ones((4,), dtype=mx.float32),
                "blocks.0.norm1.bias": mx.zeros((4,), dtype=mx.float32),
                "blocks.0.conv1.weight": mx.zeros((4, 4, 3, 3, 3), dtype=mx.float32),
                "blocks.0.conv1.bias": mx.zeros((4,), dtype=mx.float32),
                "blocks.0.norm2.weight": mx.ones((4,), dtype=mx.float32),
                "blocks.0.norm2.bias": mx.zeros((4,), dtype=mx.float32),
                "blocks.0.conv2.weight": mx.zeros((4, 4, 3, 3, 3), dtype=mx.float32),
                "blocks.0.conv2.bias": mx.zeros((4,), dtype=mx.float32),
                "out_layer.0.weight": mx.ones((4,), dtype=mx.float32),
                "out_layer.0.bias": mx.zeros((4,), dtype=mx.float32),
                "out_layer.2.weight": mx.zeros((1, 4, 3, 3, 3), dtype=mx.float32),
                "out_layer.2.bias": mx.zeros((1,), dtype=mx.float32),
            },
            base.with_suffix(".safetensors"),
        )


def _write_fake_dinov3_root(root: Path, *, fake: bool):
    root.mkdir(parents=True, exist_ok=True)
    model_type = "mlx_spatial_fake_dinov3" if fake else "dinov3_vit"
    (root / "config.json").write_text(
        f"""{{
            "model_type": "{model_type}",
            "image_size": 2,
            "patch_size": 1,
            "hidden_size": 1024,
            "num_hidden_layers": 1,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "layer_norm_eps": 0.000001,
            "use_swiglu_ffn": true,
            "num_register_tokens": 0,
            "rope_theta": 100.0,
            "pos_embed_rescale": 2.0,
            "mlx_spatial_fake_conditioning": {str(fake).lower()}
        }}"""
    )
    if fake:
        tensors = {
            "embeddings.patch_embeddings.weight": mx.ones((1024, 3, 1, 1), dtype=mx.float32),
            "layer.0.norm1.weight": mx.ones((1024,), dtype=mx.float32),
            "layer.0.attention.query.weight": mx.ones((1024, 1024), dtype=mx.float32),
        }
    else:
        tensors = {
            "embeddings.cls_token": mx.zeros((1, 1, 1024), dtype=mx.float32),
            "embeddings.patch_embeddings.bias": mx.zeros((1024,), dtype=mx.float32),
            "embeddings.patch_embeddings.weight": mx.ones((1024, 3, 1, 1), dtype=mx.float32),
            "layer.0.attention.k_proj.weight": mx.ones((1024, 1024), dtype=mx.float32),
            "layer.0.attention.o_proj.bias": mx.zeros((1024,), dtype=mx.float32),
            "layer.0.attention.o_proj.weight": mx.ones((1024, 1024), dtype=mx.float32),
            "layer.0.attention.q_proj.bias": mx.zeros((1024,), dtype=mx.float32),
            "layer.0.attention.q_proj.weight": mx.ones((1024, 1024), dtype=mx.float32),
            "layer.0.attention.v_proj.bias": mx.zeros((1024,), dtype=mx.float32),
            "layer.0.attention.v_proj.weight": mx.ones((1024, 1024), dtype=mx.float32),
            "layer.0.layer_scale1.lambda1": mx.ones((1024,), dtype=mx.float32),
            "layer.0.layer_scale2.lambda1": mx.ones((1024,), dtype=mx.float32),
            "layer.0.mlp.down_proj.bias": mx.zeros((1024,), dtype=mx.float32),
            "layer.0.mlp.down_proj.weight": mx.ones((1024, 4096), dtype=mx.float32),
            "layer.0.mlp.up_proj.bias": mx.zeros((4096,), dtype=mx.float32),
            "layer.0.mlp.up_proj.weight": mx.ones((4096, 1024), dtype=mx.float32),
            "layer.0.norm1.bias": mx.zeros((1024,), dtype=mx.float32),
            "layer.0.norm1.weight": mx.ones((1024,), dtype=mx.float32),
            "layer.0.norm2.bias": mx.zeros((1024,), dtype=mx.float32),
            "layer.0.norm2.weight": mx.ones((1024,), dtype=mx.float32),
            "norm.bias": mx.zeros((1024,), dtype=mx.float32),
            "norm.weight": mx.ones((1024,), dtype=mx.float32),
        }
    save_file(tensors, root / "model.safetensors")


def test_discover_conditioning_config_from_fake_trellis2_root(tmp_path):
    _write_trellis2_root(tmp_path)

    result = discover_trellis2_conditioning_config(tmp_path)

    assert result.ready
    assert result.config is not None
    assert result.config.image_model_family == "DinoV3FeatureExtractor"
    assert result.config.image_model_name == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert result.config.conditioning_resolution == 512
    assert result.config.expected_feature_width == 1024
    assert result.config.sparse_flow_checkpoint_path == "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"
    assert result.config.sparse_decoder_checkpoint_path == "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"
    assert result.config.shape_slat_512_checkpoint_path == "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors"
    assert result.config.shape_slat_1024_checkpoint_path == "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors"
    assert result.config.texture_slat_512_checkpoint_path == "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors"
    assert result.config.texture_slat_1024_checkpoint_path == "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors"
    assert result.config.shape_decoder_checkpoint_path == "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors"
    assert result.config.texture_decoder_checkpoint_path == "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors"
    assert tuple(asset.key for asset in result.config.model_assets) == (
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
        "shape_slat_flow_model_512",
        "shape_slat_flow_model_1024",
        "shape_slat_decoder",
        "tex_slat_flow_model_512",
        "tex_slat_flow_model_1024",
        "tex_slat_decoder",
    )
    assert result.config.sparse_structure_sampler.steps == 12
    assert result.config.sparse_structure_sampler.guidance_interval == (0.6, 1.0)
    assert result.config.shape_slat_sampler.guidance_rescale == 0.5
    assert result.config.texture_slat_sampler.guidance_strength == 1.0
    assert result.config.texture_slat_sampler.guidance_interval == (0.6, 0.9)
    assert len(result.config.shape_slat_normalization.mean) == 32
    assert len(result.config.texture_slat_normalization.std) == 32


def test_discover_conditioning_config_maps_texture_routes_for_all_pipeline_types(tmp_path):
    _write_trellis2_root(tmp_path)

    result = discover_trellis2_conditioning_config(tmp_path)

    assert result.ready
    assert result.config is not None
    expected = {
        "512": (
            "tex_slat_flow_model_512",
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json",
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
            512,
        ),
        "1024": (
            "tex_slat_flow_model_1024",
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json",
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
            1024,
        ),
        "1024_cascade": (
            "tex_slat_flow_model_1024",
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json",
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
            1024,
        ),
        "1536_cascade": (
            "tex_slat_flow_model_1024",
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json",
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
            1024,
        ),
    }

    for pipeline_type, (model_key, config_path, checkpoint_path, resolution) in expected.items():
        route = select_texture_slat_route(pipeline_type)

        assert route.model_key == model_key
        assert route.output_resolution == resolution
        if route.model_key == "tex_slat_flow_model_512":
            assert result.config.texture_slat_512_config_path == config_path
            assert result.config.texture_slat_512_checkpoint_path == checkpoint_path
        else:
            assert result.config.texture_slat_1024_config_path == config_path
            assert result.config.texture_slat_1024_checkpoint_path == checkpoint_path
        assert result.config.texture_decoder_config_path == "ckpts/tex_dec_next_dc_f16c32_fp16.json"
        assert result.config.texture_decoder_checkpoint_path == "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors"


def test_discover_conditioning_config_reports_missing_texture_1024_model_key(tmp_path):
    _write_trellis2_root(tmp_path)
    pipeline_path = tmp_path / "pipeline.json"
    pipeline = json.loads(pipeline_path.read_text())
    del pipeline["args"]["models"]["tex_slat_flow_model_1024"]
    pipeline_path.write_text(json.dumps(pipeline))

    result = discover_trellis2_conditioning_config(tmp_path)

    assert not result.ready
    assert result.blocker is not None
    assert result.blocker.stage == "texture-slat-sampling"
    assert result.blocker.operation == "TRELLIS.2 model contract discovery"
    assert "tex_slat_flow_model_1024" in result.blocker.reason


def test_discover_conditioning_config_reports_malformed_config(tmp_path):
    (tmp_path / "pipeline.json").write_text("{}")

    result = discover_trellis2_conditioning_config(tmp_path)

    assert not result.ready
    assert result.blocker is not None
    assert result.blocker.stage == "image-conditioning"
    assert result.blocker.operation == "TRELLIS.2 pipeline config discovery"
    assert "conditioning fields" in result.blocker.reason


def test_discover_conditioning_config_reports_missing_shape_model_key(tmp_path):
    _write_trellis2_root(tmp_path)
    pipeline_path = tmp_path / "pipeline.json"
    pipeline = json.loads(pipeline_path.read_text())
    del pipeline["args"]["models"]["shape_slat_flow_model_1024"]
    pipeline_path.write_text(json.dumps(pipeline))

    result = discover_trellis2_conditioning_config(tmp_path)

    assert not result.ready
    assert result.blocker is not None
    assert result.blocker.stage == "shape-slat-sampling"
    assert result.blocker.operation == "TRELLIS.2 model contract discovery"
    assert "shape_slat_flow_model_1024" in result.blocker.reason


def test_discover_conditioning_config_reports_invalid_texture_sampler(tmp_path):
    _write_trellis2_root(tmp_path)
    pipeline_path = tmp_path / "pipeline.json"
    pipeline = json.loads(pipeline_path.read_text())
    pipeline["args"]["tex_slat_sampler"]["params"]["guidance_interval"] = [0.6]
    pipeline_path.write_text(json.dumps(pipeline))

    result = discover_trellis2_conditioning_config(tmp_path)

    assert not result.ready
    assert result.blocker is not None
    assert result.blocker.stage == "texture-slat-sampling"
    assert result.blocker.operation == "TRELLIS.2 sampler config discovery"
    assert "tex_slat_sampler" in result.blocker.reason


def test_discover_conditioning_config_reports_invalid_shape_normalization(tmp_path):
    _write_trellis2_root(tmp_path)
    pipeline_path = tmp_path / "pipeline.json"
    pipeline = json.loads(pipeline_path.read_text())
    pipeline["args"]["shape_slat_normalization"]["std"] = [1.0]
    pipeline_path.write_text(json.dumps(pipeline))

    result = discover_trellis2_conditioning_config(tmp_path)

    assert not result.ready
    assert result.blocker is not None
    assert result.blocker.stage == "shape-slat-sampling"
    assert result.blocker.operation == "TRELLIS.2 normalization config discovery"


def test_prepare_dinov3_image_tensor_uses_reference_layout_and_normalization():
    image = Image.new("RGB", (2, 2), (255, 0, 0))

    tensor = prepare_dinov3_image_tensor(image, image_size=2)

    assert tuple(tensor.shape) == (1, 3, 2, 2)
    assert str(tensor.dtype).removeprefix("mlx.core.") == "float32"
    values = np.array(tensor)[0, :, 0, 0].tolist()
    np.testing.assert_allclose(
        values,
        [
            (1.0 - 0.485) / 0.229,
            (0.0 - 0.456) / 0.224,
            (0.0 - 0.406) / 0.225,
        ],
        rtol=1e-6,
    )


def test_assess_dinov3_conditioning_reports_missing_local_assets(tmp_path):
    _write_trellis2_root(tmp_path / "trellis")
    config = discover_trellis2_conditioning_config(tmp_path / "trellis").config

    output, blocker = assess_dinov3_conditioning(tmp_path / "trellis", config)

    assert output is None
    assert blocker is not None
    assert blocker.stage == "image-conditioning"
    assert blocker.operation == "local DINOv3 asset validation"
    assert "facebook/dinov3-vitl16-pretrain-lvd1689m" in blocker.reason
    assert str(default_dinov3_root(tmp_path / "trellis", config.image_model_name)) in blocker.reason


def test_assess_dinov3_conditioning_reports_present_asset_config_blocker(tmp_path):
    _write_trellis2_root(tmp_path / "trellis")
    config = discover_trellis2_conditioning_config(tmp_path / "trellis").config
    dino_root = tmp_path / "dinov3"
    dino_root.mkdir()
    (dino_root / "config.json").write_text("{}")
    save_file({"embeddings.patch_embeddings.weight": mx.ones((1,), dtype=mx.float32)}, dino_root / "model.safetensors")

    output, blocker = assess_dinov3_conditioning(tmp_path / "trellis", config, dino_root=dino_root)

    assert output is None
    assert blocker is not None
    assert blocker.operation == "DINOv3 config field validation"
    assert "model_type" in blocker.reason


@pytest.mark.heavy
def test_assess_dinov3_conditioning_reports_precise_transformer_blocker(tmp_path):
    _write_trellis2_root(tmp_path / "trellis")
    config = discover_trellis2_conditioning_config(tmp_path / "trellis").config
    dino_root = tmp_path / "dinov3"
    _write_fake_dinov3_root(dino_root, fake=False)

    output, blocker = assess_dinov3_conditioning(tmp_path / "trellis", config, dino_root=dino_root)

    assert output is None
    assert blocker is not None
    assert blocker.operation == "MLX DINOv3 attention block forward"
    assert "RoPE geometry" in blocker.reason


def test_assess_dinov3_conditioning_can_report_simulated_output(tmp_path):
    _write_trellis2_root(tmp_path / "trellis")
    config = discover_trellis2_conditioning_config(tmp_path / "trellis").config
    conditioning = mx.zeros((1, 4, 1024), dtype=mx.float32)

    output, blocker = assess_dinov3_conditioning(tmp_path / "trellis", config, conditioning=conditioning)

    assert blocker is None
    assert output == Trellis2StageOutput(
        stage="image-conditioning",
        name="cond",
        shape=(1, 4, 1024),
        dtype="float32",
        detail="simulated conditioning output for facebook/dinov3-vitl16-pretrain-lvd1689m",
    )


def test_sparse_structure_boundary_reports_conditioning_width_mismatch(tmp_path):
    _write_trellis2_root(tmp_path / "trellis")
    config = discover_trellis2_conditioning_config(tmp_path / "trellis").config
    output = Trellis2StageOutput(
        stage="image-conditioning",
        name="cond",
        shape=(1, 4, 768),
        dtype="float32",
        detail="test conditioning",
    )

    blocker = dispatch_sparse_structure_boundary(tmp_path / "trellis", config, output)

    assert blocker.stage == "sparse-structure-sampling"
    assert blocker.operation == "conditioning feature width validation"
    assert "expected 1024, got 768" in blocker.reason


def test_sparse_structure_boundary_advances_to_decoder_config_blocker(tmp_path):
    _write_trellis2_root(tmp_path / "trellis")
    config = discover_trellis2_conditioning_config(tmp_path / "trellis").config
    output = Trellis2StageOutput(
        stage="image-conditioning",
        name="cond",
        shape=(1, 4, 1024),
        dtype="float32",
        detail="test conditioning",
    )

    blocker = dispatch_sparse_structure_boundary(tmp_path / "trellis", config, output)

    assert blocker.stage == "sparse-structure-decoding"
    assert blocker.operation == "sparse structure decoder config/checkpoint probe"
    assert "ss_dec_conv3d_16l8_fp16.json" in blocker.reason


def test_sparse_structure_decoder_boundary_reports_missing_local_checkpoint(tmp_path):
    _write_trellis2_root(tmp_path / "trellis")
    _write_sparse_decoder_assets(tmp_path / "trellis", checkpoint=False)
    config = discover_trellis2_conditioning_config(tmp_path / "trellis").config

    blocker = dispatch_sparse_structure_decoder_boundary(tmp_path / "trellis", config)

    assert blocker.stage == "sparse-structure-decoding"
    assert blocker.operation == "sparse structure decoder config/checkpoint probe"
    assert "ss_dec_conv3d_16l8_fp16.safetensors" in blocker.reason


def test_sparse_structure_decoder_boundary_reports_upstream_latent_blocker(tmp_path):
    _write_trellis2_root(tmp_path / "trellis")
    _write_sparse_decoder_assets(tmp_path / "trellis")
    config = discover_trellis2_conditioning_config(tmp_path / "trellis").config

    blocker = dispatch_sparse_structure_decoder_boundary(tmp_path / "trellis", config)

    assert blocker.stage == "sparse-structure-decoding"
    assert blocker.operation == "sparse structure decoder upstream latent availability"
    assert "no sampled sparse latent" in blocker.reason


def test_sparse_structure_decoder_boundary_reports_conv_stack_blocker_with_latent(tmp_path):
    _write_trellis2_root(tmp_path / "trellis")
    _write_sparse_decoder_assets(tmp_path / "trellis")
    config = discover_trellis2_conditioning_config(tmp_path / "trellis").config
    latent = mx.zeros((1, 2, 2, 2, 2), dtype=mx.float32)

    blocker = dispatch_sparse_structure_decoder_boundary(tmp_path / "trellis", config, sparse_latent=latent)

    assert blocker.stage == "sparse-structure-decoding"
    assert blocker.operation == "sparse structure decoder coordinate extraction"
    assert "decoded to logits shape (1, 1, 2, 2, 2)" in blocker.reason


def test_shape_slat_boundary_reports_upstream_sparse_coordinate_blocker(tmp_path):
    _write_trellis2_root(tmp_path / "trellis")
    config = discover_trellis2_conditioning_config(tmp_path / "trellis").config

    blocker = dispatch_shape_slat_boundary(tmp_path / "trellis", config)

    assert blocker.stage == "shape-slat-sampling"
    assert blocker.operation == "shape SLat upstream sparse coordinate availability"
    assert "1024_cascade" in blocker.reason
    assert "shape_slat_flow_model_512" in blocker.reason


def test_shape_slat_boundary_reports_sparse_transformer_blocker_with_coordinates(tmp_path):
    _write_trellis2_root(tmp_path / "trellis")
    config = discover_trellis2_conditioning_config(tmp_path / "trellis").config
    coordinates = mx.array([[0, 0, 0, 0], [0, 0, 1, 1]], dtype=mx.int32)

    blocker = dispatch_shape_slat_boundary(tmp_path / "trellis", config, sparse_coordinates=coordinates)

    assert blocker.stage == "texture-slat-sampling"
    assert blocker.operation == "shape SLat texture handoff"
    assert "shape SLat sampling completed" in blocker.reason


def test_texture_slat_boundary_reports_upstream_shape_slat_blocker(tmp_path):
    _write_trellis2_root(tmp_path / "trellis")
    config = discover_trellis2_conditioning_config(tmp_path / "trellis").config

    blocker = dispatch_texture_slat_boundary(tmp_path / "trellis", config)

    assert blocker.stage == "texture-slat-sampling"
    assert blocker.operation == "texture SLat upstream shape_slat availability"
    assert "1024_cascade" in blocker.reason
    assert "tex_slat_flow_model_1024" in blocker.reason


def test_texture_slat_boundary_reports_sparse_transformer_blocker_with_shape_slat(tmp_path):
    _write_trellis2_root(tmp_path / "trellis")
    config = discover_trellis2_conditioning_config(tmp_path / "trellis").config
    coordinates = mx.array([[0, 0, 0, 0], [0, 0, 1, 1]], dtype=mx.int32)
    features = mx.zeros((2, 32), dtype=mx.float32)

    blocker = dispatch_texture_slat_boundary(
        tmp_path / "trellis",
        config,
        shape_slat_coordinates=coordinates,
        shape_slat_features=features,
    )

    assert blocker.stage == "latent-decoding"
    assert blocker.operation == "MLX shape latent decoder SparseConvNeXt/FlexiDualGrid forward"
    assert "shape_slat coordinate shape (2, 4)" in blocker.reason
    assert "texture_slat coordinate shape (2, 4)" in blocker.reason


def test_decode_latents_boundary_reports_upstream_shape_slat_blocker(tmp_path):
    _write_trellis2_root(tmp_path / "trellis")
    config = discover_trellis2_conditioning_config(tmp_path / "trellis").config

    blocker = dispatch_decode_latents_boundary(tmp_path / "trellis", config)

    assert blocker.stage == "latent-decoding"
    assert blocker.operation == "decode_latent upstream shape_slat availability"
    assert "shape_slat" in blocker.reason


def test_decode_latents_boundary_reports_upstream_texture_slat_blocker(tmp_path):
    _write_trellis2_root(tmp_path / "trellis")
    config = discover_trellis2_conditioning_config(tmp_path / "trellis").config
    coords = mx.array([[0, 0, 0, 0]], dtype=mx.int32)
    feats = mx.zeros((1, 32), dtype=mx.float32)

    blocker = dispatch_decode_latents_boundary(
        tmp_path / "trellis",
        config,
        shape_slat_coordinates=coords,
        shape_slat_features=feats,
    )

    assert blocker.stage == "latent-decoding"
    assert blocker.operation == "decode_latent upstream texture_slat availability"
    assert "texture_slat" in blocker.reason


def test_decode_latents_boundary_reports_shape_decoder_stack_blocker(tmp_path):
    _write_trellis2_root(tmp_path / "trellis")
    config = discover_trellis2_conditioning_config(tmp_path / "trellis").config
    coords = mx.array([[0, 0, 0, 0], [0, 1, 2, 3]], dtype=mx.int32)
    feats = mx.zeros((2, 32), dtype=mx.float32)

    blocker = dispatch_decode_latents_boundary(
        tmp_path / "trellis",
        config,
        shape_slat_coordinates=coords,
        shape_slat_features=feats,
        texture_slat_coordinates=coords,
        texture_slat_features=feats,
        resolution=1024,
    )

    assert blocker.stage == "latent-decoding"
    assert blocker.operation == "MLX shape latent decoder SparseConvNeXt/FlexiDualGrid forward"
    assert "shape decoder from_latent projection executed with output shape (2, 16)" in blocker.reason
    assert "first C2S up-block produced" in blocker.reason
    assert "texture decoder from_latent projection validates with output shape (2, 16)" in blocker.reason


def test_attempt_forward_trace_enters_image_conditioning_and_reports_dino_blocker(tmp_path):
    _write_trellis2_root(tmp_path / "trellis")
    image = tmp_path / "alpha.png"
    _write_alpha_image(image)

    report = mlx_spatial.Trellis2InferencePipeline(tmp_path / "trellis").attempt_forward_trace(image)

    assert isinstance(report, Trellis2ForwardTraceResult)
    assert report.completed_stages == (
        "input-image",
        "asset-config-validation",
        "checkpoint-probe-readiness",
        "image-preprocessing-background",
    )
    assert report.blocker is not None
    assert report.blocker.stage == "image-conditioning"
    assert report.blocker.operation == "local DINOv3 asset validation"


def test_attempt_forward_trace_with_simulated_conditioning_reaches_sparse_boundary(tmp_path):
    _write_trellis2_root(tmp_path / "trellis")
    image = tmp_path / "alpha.png"
    _write_alpha_image(image)
    conditioning = mx.zeros((1, 4, 1024), dtype=mx.float32)

    report = mlx_spatial.Trellis2InferencePipeline(tmp_path / "trellis").attempt_forward_trace(
        image,
        conditioning=conditioning,
    )

    assert report.completed_stages == (
        "input-image",
        "asset-config-validation",
        "checkpoint-probe-readiness",
        "image-preprocessing-background",
        "image-conditioning",
        "sparse-structure-sampling",
    )
    assert len(report.outputs) == 2
    assert report.outputs[0] == Trellis2StageOutput(
        stage="image-conditioning",
        name="cond",
        shape=(1, 4, 1024),
        dtype="float32",
        detail="simulated conditioning output for facebook/dinov3-vitl16-pretrain-lvd1689m",
    )
    assert report.outputs[1].stage == "sparse-structure-sampling"
    assert report.outputs[1].name == "sparse_latent"
    assert report.outputs[1].shape == (1, 2, 2, 2, 2)
    assert report.blocker is not None
    assert report.blocker.stage == "sparse-structure-decoding"
    assert report.blocker.operation == "sparse structure decoder config/checkpoint probe"


def test_attempt_forward_trace_with_fake_dinov3_assets_reaches_sparse_boundary(tmp_path):
    _write_trellis2_root(tmp_path / "trellis")
    dino_root = tmp_path / "dinov3"
    _write_fake_dinov3_root(dino_root, fake=True)
    image = tmp_path / "alpha.png"
    _write_alpha_image(image)

    report = mlx_spatial.Trellis2InferencePipeline(tmp_path / "trellis").attempt_forward_trace(
        image,
        dino_root=dino_root,
    )

    assert report.completed_stages == (
        "input-image",
        "asset-config-validation",
        "checkpoint-probe-readiness",
        "image-preprocessing-background",
        "image-conditioning",
        "sparse-structure-sampling",
    )
    assert report.outputs[0] == Trellis2StageOutput(
        stage="image-conditioning",
        name="cond",
        shape=(1, 4, 1024),
        dtype="float32",
        detail=f"fake DINOv3 conditioning from {dino_root} using embeddings.patch_embeddings.weight",
    )
    assert report.outputs[1].stage == "sparse-structure-sampling"
    assert report.blocker is not None
    assert report.blocker.stage == "sparse-structure-decoding"


@pytest.mark.heavy
def test_attempt_forward_trace_with_executable_dinov3_assets_reaches_sparse_boundary(tmp_path):
    _write_trellis2_root(tmp_path / "trellis", conditioning_resolution=2)
    dino_root = tmp_path / "dinov3"
    _write_fake_dinov3_root(dino_root, fake=False)
    image = tmp_path / "alpha.png"
    _write_alpha_image(image)

    report = mlx_spatial.Trellis2InferencePipeline(tmp_path / "trellis").attempt_forward_trace(
        image,
        dino_root=dino_root,
    )

    assert report.completed_stages == (
        "input-image",
        "asset-config-validation",
        "checkpoint-probe-readiness",
        "image-preprocessing-background",
        "image-conditioning",
        "sparse-structure-sampling",
    )
    assert len(report.outputs) == 2
    assert report.outputs[0].stage == "image-conditioning"
    assert report.outputs[0].shape == (1, 5, 1024)
    assert report.outputs[0].dtype == "float32"
    assert report.outputs[1].stage == "sparse-structure-sampling"
    assert report.outputs[1].shape == (1, 2, 2, 2, 2)
    assert report.blocker is not None
    assert report.blocker.stage == "sparse-structure-decoding"
    assert report.blocker.operation == "sparse structure decoder config/checkpoint probe"


def test_forward_exports_are_public():
    assert mlx_spatial.Trellis2ConditioningConfig is Trellis2ConditioningConfig
    assert mlx_spatial.Trellis2ForwardBlocker is Trellis2ForwardBlocker
    assert mlx_spatial.Trellis2ForwardConfigResult is Trellis2ForwardConfigResult
    assert mlx_spatial.Trellis2ForwardTraceResult is Trellis2ForwardTraceResult
    assert mlx_spatial.Trellis2StageOutput is Trellis2StageOutput
    assert mlx_spatial.discover_trellis2_conditioning_config is discover_trellis2_conditioning_config
    assert mlx_spatial.dispatch_decode_latents_boundary is dispatch_decode_latents_boundary
    assert mlx_spatial.dispatch_shape_slat_boundary is dispatch_shape_slat_boundary
    assert mlx_spatial.dispatch_sparse_structure_decoder_boundary is dispatch_sparse_structure_decoder_boundary
    assert mlx_spatial.dispatch_sparse_structure_boundary is dispatch_sparse_structure_boundary
    assert mlx_spatial.dispatch_texture_slat_boundary is dispatch_texture_slat_boundary
    assert mlx_spatial.prepare_dinov3_image_tensor is prepare_dinov3_image_tensor
    assert mlx_spatial.assess_dinov3_conditioning is assess_dinov3_conditioning
