"""Self-contained miniature TRELLIS.2 golden fixture construction."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_spatial.safetensors_io import save_safetensors
from mlx_spatial.spatialkit import export_decoded_ovoxel_glb
from mlx_spatial.trellis2_decode import StructuredLatentDecoderConfig
from mlx_spatial.trellis2_dinov3 import DinoV3ModelConfig
from mlx_spatial.trellis2_forward import Trellis2ForwardTraceResult
from mlx_spatial.trellis2_quantization import quantize_trellis2_weights
from mlx_spatial.trellis2_slat import SLatFlowConfig
from mlx_spatial.trellis2_sparse_structure import SparseStructureDecoderConfig, SparseStructureFlowConfig
from tests.golden_assertions import summarize_array


TRELLIS2_MINIATURE_EXPORT_GRID_SIZE = 32
TRELLIS2_MINIATURE_EXPORT_TARGET_FACES = 64
TRELLIS2_MINIATURE_EXPORT_TEXTURE_SIZE = 16


@dataclass(frozen=True)
class Trellis2MiniatureGoldenFixture:
    """Paths for one generated miniature affine-INT8 TRELLIS.2 fixture."""

    source_root: Path
    quantized_root: Path
    dino_root: Path
    image_path: Path
    output_path: Path


@dataclass
class Trellis2MiniatureSpatialKitExporter:
    """Run the real SpatialKit exporter with a miniature remesh policy."""

    requested_options: dict[str, Any] | None = None

    def __call__(self, decoded_dir: str | Path, output_path: str | Path, **options: Any):
        self.requested_options = dict(options)
        return export_decoded_ovoxel_glb(
            decoded_dir,
            output_path,
            texture_size=TRELLIS2_MINIATURE_EXPORT_TEXTURE_SIZE,
            target_faces=TRELLIS2_MINIATURE_EXPORT_TARGET_FACES,
            quality_preset="reference-target",
            grid_size=TRELLIS2_MINIATURE_EXPORT_GRID_SIZE,
            uv_backend="xatlas-equivalent-native",
            remesh=True,
            remesh_resolution=TRELLIS2_MINIATURE_EXPORT_GRID_SIZE,
            simplify_backend="mlx-qem",
            texture_postprocess="telea",
            diagnostics_path=options.get("diagnostics_path"),
        )


def build_trellis2_miniature_golden_fixture(root: Path) -> Trellis2MiniatureGoldenFixture:
    """Create tiny source checkpoints, quantize them, and return runnable fixture paths."""

    source_root = root / "trellis2-source"
    quantized_root = root / "trellis2-mlx-8bit"
    dino_root = root / "dinov3"
    image_path = root / "input.png"
    output_path = root / "output" / "model.glb"

    sparse_flow = SparseStructureFlowConfig(
        name="SparseStructureFlowModel",
        resolution=2,
        in_channels=2,
        out_channels=2,
        model_channels=64,
        cond_channels=64,
        num_blocks=1,
        num_heads=4,
        mlp_ratio=2.0,
        pe_mode="rope",
        share_mod=True,
        initialization="scaled",
        qk_rms_norm=True,
        qk_rms_norm_cross=True,
        dtype="float32",
    )
    sparse_decoder = SparseStructureDecoderConfig(
        name="SparseStructureDecoder",
        out_channels=1,
        latent_channels=2,
        num_res_blocks=0,
        channels=(4, 4),
        num_res_blocks_middle=0,
        norm_type="layer",
        use_fp16=False,
    )
    shape_slat = _slat_config(in_channels=32)
    texture_slat = _slat_config(in_channels=64)
    shape_decoder = _decoder_config(name="FlexiDualGridVaeDecoder", out_channels=7, pred_subdiv=True)
    texture_decoder = _decoder_config(name="SparseUnetVaeDecoder", out_channels=6, pred_subdiv=False)
    dino_config = DinoV3ModelConfig(
        model_type="dinov3_vit",
        image_size=512,
        patch_size=64,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=128,
        layer_norm_eps=1e-5,
        use_swiglu_ffn=False,
        num_register_tokens=0,
        expected_feature_width=64,
        rope_theta=100.0,
        pos_embed_rescale=2.0,
    )

    _write_pipeline(source_root)
    _write_json(
        source_root / "ckpts/ss_flow_img_dit_1_3B_64_bf16.json",
        _sparse_flow_config_payload(sparse_flow),
    )
    _write_flow_checkpoint(
        source_root / "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
        in_channels=sparse_flow.in_channels,
        out_channels=sparse_flow.out_channels,
        model_channels=sparse_flow.model_channels,
        cond_channels=sparse_flow.cond_channels,
        num_heads=sparse_flow.num_heads,
        mlp_ratio=sparse_flow.mlp_ratio,
    )

    sparse_decoder_base = source_root / "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
    _write_json(sparse_decoder_base.with_suffix(".json"), _sparse_decoder_config_payload(sparse_decoder))
    _write_sparse_decoder_checkpoint(sparse_decoder_base.with_suffix(".safetensors"), sparse_decoder)

    for resolution in (512, 1024):
        shape_base = source_root / f"ckpts/slat_flow_img2shape_dit_1_3B_{resolution}_bf16"
        _write_json(shape_base.with_suffix(".json"), _slat_config_payload(shape_slat, resolution=32))
        _write_flow_checkpoint(
            shape_base.with_suffix(".safetensors"),
            in_channels=shape_slat.in_channels,
            out_channels=shape_slat.out_channels,
            model_channels=shape_slat.model_channels,
            cond_channels=shape_slat.cond_channels,
            num_heads=shape_slat.num_heads,
            mlp_ratio=shape_slat.mlp_ratio,
        )
        texture_base = source_root / f"ckpts/slat_flow_imgshape2tex_dit_1_3B_{resolution}_bf16"
        _write_json(texture_base.with_suffix(".json"), _slat_config_payload(texture_slat, resolution=32))
        _write_flow_checkpoint(
            texture_base.with_suffix(".safetensors"),
            in_channels=texture_slat.in_channels,
            out_channels=texture_slat.out_channels,
            model_channels=texture_slat.model_channels,
            cond_channels=texture_slat.cond_channels,
            num_heads=texture_slat.num_heads,
            mlp_ratio=texture_slat.mlp_ratio,
        )

    shape_decoder_base = source_root / "ckpts/shape_dec_next_dc_f16c32_fp16"
    _write_json(shape_decoder_base.with_suffix(".json"), _decoder_config_payload(shape_decoder))
    _write_decoder_checkpoint(shape_decoder_base.with_suffix(".safetensors"), shape_decoder)
    texture_decoder_base = source_root / "ckpts/tex_dec_next_dc_f16c32_fp16"
    _write_json(texture_decoder_base.with_suffix(".json"), _decoder_config_payload(texture_decoder))
    _write_decoder_checkpoint(texture_decoder_base.with_suffix(".safetensors"), texture_decoder)

    for name in ("shape_enc_next_dc_f16c32_fp16", "tex_enc_next_dc_f16c32_fp16"):
        base = source_root / "ckpts" / name
        _write_json(base.with_suffix(".json"), {"name": "MiniatureUnusedEncoder", "args": {}})
        save_safetensors(
            base.with_suffix(".safetensors"),
            {"blocks.0.0.mlp.0.weight": _pattern((64, 64), scale=0.01)},
        )

    _write_dinov3(dino_root, dino_config)
    quantize_trellis2_weights(
        source_root,
        quantized_root,
        dinov3_root=dino_root,
        bits=8,
        group_size=64,
    )
    _write_rgba_fixture(image_path)
    return Trellis2MiniatureGoldenFixture(
        source_root=source_root,
        quantized_root=quantized_root,
        dino_root=quantized_root / "dinov3",
        image_path=image_path,
        output_path=output_path,
    )


def summarize_trellis2_golden_trace(trace: Trellis2ForwardTraceResult) -> dict[str, Any]:
    """Return a stable, JSON-compatible summary of tensor-bearing stage outputs."""

    tensor_outputs = {}
    for output in trace.outputs:
        if output.payload is None:
            continue
        summary = summarize_array(output.payload)
        if summary["shape"] != list(output.shape) or summary["dtype"] != output.dtype:
            raise AssertionError(
                f"trace metadata disagrees with payload for {output.name}: "
                f"declared shape={output.shape} dtype={output.dtype}; "
                f"actual shape={summary['shape']} dtype={summary['dtype']}"
            )
        tensor_outputs[output.name] = summary
    return {
        "completed_stages": list(trace.completed_stages),
        "tensor_outputs": tensor_outputs,
    }


def _slat_config(*, in_channels: int) -> SLatFlowConfig:
    return SLatFlowConfig(
        name="SLatFlowModel",
        resolution=32,
        in_channels=in_channels,
        out_channels=32,
        model_channels=64,
        cond_channels=64,
        num_blocks=1,
        num_heads=4,
        mlp_ratio=2.0,
        pe_mode="rope",
        share_mod=True,
        initialization="scaled",
        qk_rms_norm=True,
        qk_rms_norm_cross=True,
        dtype="float32",
    )


def _decoder_config(*, name: str, out_channels: int, pred_subdiv: bool) -> StructuredLatentDecoderConfig:
    return StructuredLatentDecoderConfig(
        name=name,
        latent_channels=32,
        model_channels=(64, 32),
        num_blocks=(1, 0),
        block_type=("SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d"),
        up_block_type=("SparseResBlockC2S3d",),
        use_fp16=False,
        out_channels=out_channels,
        resolution=256 if name == "FlexiDualGridVaeDecoder" else None,
        pred_subdiv=pred_subdiv,
    )


def _write_pipeline(root: Path) -> None:
    models = {
        "sparse_structure_decoder": "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
        "sparse_structure_flow_model": "ckpts/ss_flow_img_dit_1_3B_64_bf16",
        "shape_slat_decoder": "ckpts/shape_dec_next_dc_f16c32_fp16",
        "shape_slat_flow_model_512": "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16",
        "shape_slat_flow_model_1024": "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16",
        "tex_slat_decoder": "ckpts/tex_dec_next_dc_f16c32_fp16",
        "tex_slat_flow_model_512": "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16",
        "tex_slat_flow_model_1024": "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
    }
    sampler = {
        "name": "FlowEulerGuidanceIntervalSampler",
        "args": {"sigma_min": 1e-5},
        "params": {
            "steps": 1,
            "guidance_strength": 1.0,
            "guidance_rescale": 0.0,
            "guidance_interval": [0.0, 1.0],
            "rescale_t": 1.0,
        },
    }
    normalization = {"mean": [0.0] * 32, "std": [1.0] * 32}
    _write_json(
        root / "pipeline.json",
        {
            "name": "Trellis2ImageTo3DPipeline",
            "args": {
                "models": models,
                "sparse_structure_sampler": sampler,
                "shape_slat_sampler": sampler,
                "shape_slat_normalization": normalization,
                "tex_slat_sampler": sampler,
                "tex_slat_normalization": normalization,
                "image_cond_model": {
                    "name": "DinoV3FeatureExtractor",
                    "args": {"model_name": "miniature/dinov3"},
                },
                "rembg_model": {"name": "UnusedForRgbaFixture", "args": {}},
                "default_pipeline_type": "512",
            },
        },
    )
    _write_json(root / "texturing_pipeline.json", {"name": "MiniatureTexturingPipeline", "args": {}})


def _write_flow_checkpoint(
    path: Path,
    *,
    in_channels: int,
    out_channels: int,
    model_channels: int,
    cond_channels: int,
    num_heads: int,
    mlp_ratio: float,
) -> None:
    head_dim = model_channels // num_heads
    mlp_channels = int(model_channels * mlp_ratio)
    tensors = {
        "input_layer.weight": _pattern((model_channels, in_channels), scale=0.02),
        "input_layer.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "out_layer.weight": _pattern((out_channels, model_channels), scale=0.02),
        "out_layer.bias": mx.zeros((out_channels,), dtype=mx.float32),
        "t_embedder.mlp.0.weight": _pattern((model_channels, 256), scale=0.01),
        "t_embedder.mlp.0.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "t_embedder.mlp.2.weight": _pattern((model_channels, model_channels), scale=0.01),
        "t_embedder.mlp.2.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "adaLN_modulation.1.weight": _pattern((model_channels * 6, model_channels), scale=0.005),
        "adaLN_modulation.1.bias": mx.zeros((model_channels * 6,), dtype=mx.float32),
        "blocks.0.modulation": mx.zeros((model_channels * 6,), dtype=mx.float32),
        "blocks.0.norm2.weight": mx.ones((model_channels,), dtype=mx.float32),
        "blocks.0.norm2.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "blocks.0.self_attn.to_qkv.weight": _pattern((model_channels * 3, model_channels), scale=0.01),
        "blocks.0.self_attn.to_qkv.bias": mx.zeros((model_channels * 3,), dtype=mx.float32),
        "blocks.0.self_attn.q_rms_norm.gamma": mx.ones((num_heads, head_dim), dtype=mx.float32),
        "blocks.0.self_attn.k_rms_norm.gamma": mx.ones((num_heads, head_dim), dtype=mx.float32),
        "blocks.0.self_attn.to_out.weight": _pattern((model_channels, model_channels), scale=0.01),
        "blocks.0.self_attn.to_out.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "blocks.0.cross_attn.to_q.weight": _pattern((model_channels, model_channels), scale=0.01),
        "blocks.0.cross_attn.to_q.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "blocks.0.cross_attn.to_kv.weight": _pattern((model_channels * 2, cond_channels), scale=0.01),
        "blocks.0.cross_attn.to_kv.bias": mx.zeros((model_channels * 2,), dtype=mx.float32),
        "blocks.0.cross_attn.q_rms_norm.gamma": mx.ones((num_heads, head_dim), dtype=mx.float32),
        "blocks.0.cross_attn.k_rms_norm.gamma": mx.ones((num_heads, head_dim), dtype=mx.float32),
        "blocks.0.cross_attn.to_out.weight": _pattern((model_channels, model_channels), scale=0.01),
        "blocks.0.cross_attn.to_out.bias": mx.zeros((model_channels,), dtype=mx.float32),
        "blocks.0.mlp.mlp.0.weight": _pattern((mlp_channels, model_channels), scale=0.01),
        "blocks.0.mlp.mlp.0.bias": mx.zeros((mlp_channels,), dtype=mx.float32),
        "blocks.0.mlp.mlp.2.weight": _pattern((model_channels, mlp_channels), scale=0.01),
        "blocks.0.mlp.mlp.2.bias": mx.zeros((model_channels,), dtype=mx.float32),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors(path, tensors)


def _write_sparse_decoder_checkpoint(path: Path, config: SparseStructureDecoderConfig) -> None:
    tensors = {
        "input_layer.weight": _pattern((config.channels[0], config.latent_channels, 3, 3, 3), scale=0.02),
        "input_layer.bias": mx.zeros((config.channels[0],), dtype=mx.float32),
        "blocks.0.conv.weight": _pattern((config.channels[1] * 8, config.channels[0], 3, 3, 3), scale=0.01),
        "blocks.0.conv.bias": mx.zeros((config.channels[1] * 8,), dtype=mx.float32),
        "out_layer.0.weight": mx.ones((config.channels[-1],), dtype=mx.float32),
        "out_layer.0.bias": mx.zeros((config.channels[-1],), dtype=mx.float32),
        "out_layer.2.weight": _pattern((1, config.channels[-1], 3, 3, 3), scale=0.01),
        "out_layer.2.bias": mx.full((1,), 4.0, dtype=mx.float32),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors(path, tensors)


def _write_decoder_checkpoint(path: Path, config: StructuredLatentDecoderConfig) -> None:
    first_channels, second_channels = config.model_channels
    tensors = {
        "from_latent.weight": _pattern((first_channels, config.latent_channels), scale=0.02),
        "from_latent.bias": mx.zeros((first_channels,), dtype=mx.float32),
        "output_layer.weight": _pattern((config.out_channels, second_channels), scale=0.01),
        "output_layer.bias": mx.zeros((config.out_channels,), dtype=mx.float32),
        "blocks.0.0.conv.weight": _center_identity_conv(first_channels),
        "blocks.0.0.conv.bias": mx.zeros((first_channels,), dtype=mx.float32),
        "blocks.0.0.norm.weight": mx.ones((first_channels,), dtype=mx.float32),
        "blocks.0.0.norm.bias": mx.zeros((first_channels,), dtype=mx.float32),
        "blocks.0.0.mlp.0.weight": _pattern((first_channels * 4, first_channels), scale=0.01),
        "blocks.0.0.mlp.0.bias": mx.zeros((first_channels * 4,), dtype=mx.float32),
        "blocks.0.0.mlp.2.weight": _pattern((first_channels, first_channels * 4), scale=0.01),
        "blocks.0.0.mlp.2.bias": mx.zeros((first_channels,), dtype=mx.float32),
        "blocks.0.1.norm1.weight": mx.ones((first_channels,), dtype=mx.float32),
        "blocks.0.1.norm1.bias": mx.zeros((first_channels,), dtype=mx.float32),
        "blocks.0.1.conv1.weight": _pattern((second_channels * 8, 3, 3, 3, first_channels), scale=0.005),
        "blocks.0.1.conv1.bias": mx.zeros((second_channels * 8,), dtype=mx.float32),
        "blocks.0.1.conv2.weight": _center_identity_conv(second_channels),
        "blocks.0.1.conv2.bias": mx.zeros((second_channels,), dtype=mx.float32),
    }
    if config.pred_subdiv:
        tensors["blocks.0.1.to_subdiv.weight"] = _pattern((8, first_channels), scale=0.005)
        tensors["blocks.0.1.to_subdiv.bias"] = mx.full((8,), 8.0, dtype=mx.float32)
        tensors["output_layer.bias"] = mx.array([0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 0.0], dtype=mx.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors(path, tensors)


def _write_dinov3(root: Path, config: DinoV3ModelConfig) -> None:
    _write_json(
        root / "config.json",
        {
            "model_type": config.model_type,
            "image_size": config.image_size,
            "patch_size": config.patch_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "intermediate_size": config.intermediate_size,
            "layer_norm_eps": config.layer_norm_eps,
            "use_swiglu_ffn": config.use_swiglu_ffn,
            "num_register_tokens": config.num_register_tokens,
            "rope_theta": config.rope_theta,
            "pos_embed_rescale": config.pos_embed_rescale,
        },
    )
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    layer = "layer.0"
    tensors = {
        "embeddings.cls_token": mx.zeros((1, 1, hidden), dtype=mx.float32),
        "embeddings.patch_embeddings.bias": mx.zeros((hidden,), dtype=mx.float32),
        "embeddings.patch_embeddings.weight": _pattern(
            (hidden, 3, config.patch_size, config.patch_size),
            scale=0.002,
        ),
        "norm.bias": mx.zeros((hidden,), dtype=mx.float32),
        "norm.weight": mx.ones((hidden,), dtype=mx.float32),
        f"{layer}.attention.k_proj.weight": _pattern((hidden, hidden), scale=0.01),
        f"{layer}.attention.o_proj.bias": mx.zeros((hidden,), dtype=mx.float32),
        f"{layer}.attention.o_proj.weight": _pattern((hidden, hidden), scale=0.01),
        f"{layer}.attention.q_proj.bias": mx.zeros((hidden,), dtype=mx.float32),
        f"{layer}.attention.q_proj.weight": _pattern((hidden, hidden), scale=0.01),
        f"{layer}.attention.v_proj.bias": mx.zeros((hidden,), dtype=mx.float32),
        f"{layer}.attention.v_proj.weight": _pattern((hidden, hidden), scale=0.01),
        f"{layer}.layer_scale1.lambda1": mx.full((hidden,), 0.1, dtype=mx.float32),
        f"{layer}.layer_scale2.lambda1": mx.full((hidden,), 0.1, dtype=mx.float32),
        f"{layer}.mlp.down_proj.bias": mx.zeros((hidden,), dtype=mx.float32),
        f"{layer}.mlp.down_proj.weight": _pattern((hidden, intermediate), scale=0.01),
        f"{layer}.mlp.up_proj.bias": mx.zeros((intermediate,), dtype=mx.float32),
        f"{layer}.mlp.up_proj.weight": _pattern((intermediate, hidden), scale=0.01),
        f"{layer}.norm1.bias": mx.zeros((hidden,), dtype=mx.float32),
        f"{layer}.norm1.weight": mx.ones((hidden,), dtype=mx.float32),
        f"{layer}.norm2.bias": mx.zeros((hidden,), dtype=mx.float32),
        f"{layer}.norm2.weight": mx.ones((hidden,), dtype=mx.float32),
    }
    root.mkdir(parents=True, exist_ok=True)
    save_safetensors(root / "model.safetensors", tensors)


def _sparse_flow_config_payload(config: SparseStructureFlowConfig) -> dict[str, Any]:
    return {"name": config.name, "args": _flow_config_args(config)}


def _slat_config_payload(config: SLatFlowConfig, *, resolution: int) -> dict[str, Any]:
    args = _flow_config_args(config)
    args["resolution"] = resolution
    return {"name": config.name, "args": args}


def _flow_config_args(config: SparseStructureFlowConfig | SLatFlowConfig) -> dict[str, Any]:
    return {
        "resolution": config.resolution,
        "in_channels": config.in_channels,
        "out_channels": config.out_channels,
        "model_channels": config.model_channels,
        "cond_channels": config.cond_channels,
        "num_blocks": config.num_blocks,
        "num_heads": config.num_heads,
        "mlp_ratio": config.mlp_ratio,
        "pe_mode": config.pe_mode,
        "share_mod": config.share_mod,
        "initialization": config.initialization,
        "qk_rms_norm": config.qk_rms_norm,
        "qk_rms_norm_cross": config.qk_rms_norm_cross,
        "dtype": config.dtype,
    }


def _sparse_decoder_config_payload(config: SparseStructureDecoderConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "args": {
            "out_channels": config.out_channels,
            "latent_channels": config.latent_channels,
            "num_res_blocks": config.num_res_blocks,
            "channels": list(config.channels),
            "num_res_blocks_middle": config.num_res_blocks_middle,
            "norm_type": config.norm_type,
            "use_fp16": config.use_fp16,
        },
    }


def _decoder_config_payload(config: StructuredLatentDecoderConfig) -> dict[str, Any]:
    args: dict[str, Any] = {
        "model_channels": list(config.model_channels),
        "latent_channels": config.latent_channels,
        "num_blocks": list(config.num_blocks),
        "block_type": list(config.block_type),
        "up_block_type": list(config.up_block_type),
        "block_args": [{} for _ in config.model_channels],
        "use_fp16": config.use_fp16,
    }
    if config.name == "FlexiDualGridVaeDecoder":
        args["resolution"] = config.resolution
    else:
        args["out_channels"] = config.out_channels
        args["pred_subdiv"] = config.pred_subdiv
    return {"name": config.name, "args": args}


def _write_rgba_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    pixels = image.load()
    for y in range(3, 13):
        for x in range(4, 12):
            pixels[x, y] = (40 + x * 4, 30 + y * 5, 120, 255)
    image.save(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pattern(shape: tuple[int, ...], *, scale: float) -> mx.array:
    size = math.prod(shape)
    values = (np.arange(size, dtype=np.float32) % 23.0 - 11.0) / 11.0
    return mx.array((values * scale).reshape(shape), dtype=mx.float32)


def _center_identity_conv(channels: int) -> mx.array:
    values = np.zeros((channels, 3, 3, 3, channels), dtype=np.float32)
    indices = np.arange(channels)
    values[indices, 1, 1, 1, indices] = 1.0
    return mx.array(values)


__all__ = [
    "TRELLIS2_MINIATURE_EXPORT_GRID_SIZE",
    "TRELLIS2_MINIATURE_EXPORT_TARGET_FACES",
    "TRELLIS2_MINIATURE_EXPORT_TEXTURE_SIZE",
    "Trellis2MiniatureGoldenFixture",
    "Trellis2MiniatureSpatialKitExporter",
    "build_trellis2_miniature_golden_fixture",
    "summarize_trellis2_golden_trace",
]
