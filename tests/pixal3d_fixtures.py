import json
from pathlib import Path

import mlx.core as mx
from safetensors.mlx import save_file

from mlx_spatial.model_assets import PIXAL3D_ASSETS


def write_fake_pixal3d_root(root: Path) -> Path:
    for relative_path in PIXAL3D_ASSETS.required_paths:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            if path.name == "pipeline.json":
                path.write_text(json.dumps(minimal_pixal3d_pipeline()), encoding="utf-8")
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


def minimal_pixal3d_pipeline():
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
            "sparse_structure_sampler": _sampler(7.5, 0.7, [0.6, 1.0], 5.0),
            "shape_slat_sampler": _sampler(7.5, 0.5, [0.6, 1.0], 3.0),
            "tex_slat_sampler": _sampler(1.0, 0.0, [0.6, 0.9], 3.0),
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


def _sampler(guidance_strength, guidance_rescale, interval, rescale_t):
    return {
        "name": "FlowEulerGuidanceIntervalSampler",
        "args": {"sigma_min": 1e-5},
        "params": {
            "steps": 12,
            "guidance_strength": guidance_strength,
            "guidance_rescale": guidance_rescale,
            "guidance_interval": interval,
            "rescale_t": rescale_t,
        },
    }
