"""SLat sampling contracts for TRELLIS.2 forward tracing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from .checkpoint import CheckpointTensorInfo, inspect_checkpoint, load_checkpoint_tensors
from .trellis2_sparse_structure import (
    _attention,
    _gelu_tanh,
    _layer_norm,
    _layer_norm_no_affine,
    _linear,
    _multi_head_rms_norm,
    _silu,
    _split_modulation,
    _timestep_embedding,
    flow_euler_schedule,
)


SLAT_INPUT_TENSOR_NAMES = (
    "input_layer.weight",
    "input_layer.bias",
)

SLAT_BLOCK0_INSPECTION_NAMES = (
    "t_embedder.mlp.0.weight",
    "t_embedder.mlp.0.bias",
    "t_embedder.mlp.2.weight",
    "t_embedder.mlp.2.bias",
    "adaLN_modulation.1.weight",
    "adaLN_modulation.1.bias",
    "blocks.0.modulation",
    "blocks.0.norm2.weight",
    "blocks.0.norm2.bias",
    "blocks.0.self_attn.to_qkv.weight",
    "blocks.0.self_attn.to_qkv.bias",
    "blocks.0.cross_attn.to_q.weight",
    "blocks.0.cross_attn.to_q.bias",
    "blocks.0.cross_attn.to_kv.weight",
    "blocks.0.cross_attn.to_kv.bias",
    "out_layer.weight",
    "out_layer.bias",
)

SLAT_DENSE_SELF_ATTN_THRESHOLD = 4096
SLAT_WINDOWED_SELF_ATTN_THRESHOLD = SLAT_DENSE_SELF_ATTN_THRESHOLD
SLAT_FULL_SELF_ATTN_TOKEN_LIMIT = 49152
SLAT_FULL_SELF_ATTN_QUERY_CHUNK_SIZE = 512
SLAT_WINDOW_SIZE = 8


@dataclass(frozen=True)
class SLatFlowConfig:
    name: str
    resolution: int
    in_channels: int
    out_channels: int
    model_channels: int
    cond_channels: int
    num_blocks: int
    num_heads: int
    mlp_ratio: float
    pe_mode: str
    share_mod: bool
    initialization: str
    qk_rms_norm: bool
    qk_rms_norm_cross: bool
    dtype: str


@dataclass(frozen=True)
class ShapeSLatRoute:
    pipeline_type: str
    model_keys: tuple[str, ...]
    output_resolution: int
    cascade: bool


@dataclass(frozen=True)
class ShapeSLatForwardProbe:
    checkpoint_path: str
    coordinate_shape: tuple[int, int]
    feature_shape: tuple[int, int]
    input_projection_shape: tuple[int, int]
    block0_output_shape: tuple[int, int] | None
    completed_blocks: int
    stack_output_shape: tuple[int, int] | None
    output_projection_shape: tuple[int, int] | None
    sampled_feature_shape: tuple[int, int] | None
    loaded_tensor_names: tuple[str, ...]
    inspected_tensor_names: tuple[str, ...]
    blocker_operation: str
    blocker_detail: str
    sampled_features: mx.array | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class TextureSLatRoute:
    pipeline_type: str
    model_key: str
    output_resolution: int


@dataclass(frozen=True)
class TextureSLatForwardProbe:
    checkpoint_path: str
    coordinate_shape: tuple[int, int]
    shape_feature_shape: tuple[int, int]
    noise_feature_shape: tuple[int, int]
    concat_feature_shape: tuple[int, int]
    input_projection_shape: tuple[int, int]
    block0_output_shape: tuple[int, int] | None
    completed_blocks: int
    stack_output_shape: tuple[int, int] | None
    output_projection_shape: tuple[int, int] | None
    sampled_feature_shape: tuple[int, int] | None
    loaded_tensor_names: tuple[str, ...]
    inspected_tensor_names: tuple[str, ...]
    blocker_operation: str
    blocker_detail: str
    sampled_features: mx.array | None = field(default=None, compare=False, repr=False)


def read_slat_flow_config(root: str | Path, config_path: str) -> SLatFlowConfig:
    path = Path(root) / config_path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        args = payload["args"]
        return SLatFlowConfig(
            name=str(payload["name"]),
            resolution=int(args["resolution"]),
            in_channels=int(args["in_channels"]),
            out_channels=int(args["out_channels"]),
            model_channels=int(args["model_channels"]),
            cond_channels=int(args["cond_channels"]),
            num_blocks=int(args["num_blocks"]),
            num_heads=int(args["num_heads"]),
            mlp_ratio=float(args["mlp_ratio"]),
            pe_mode=str(args["pe_mode"]),
            share_mod=bool(args["share_mod"]),
            initialization=str(args["initialization"]),
            qk_rms_norm=bool(args["qk_rms_norm"]),
            qk_rms_norm_cross=bool(args["qk_rms_norm_cross"]),
            dtype=str(args["dtype"]),
        )
    except FileNotFoundError:
        raise
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"SLat flow config is invalid: {error}") from error


def select_shape_slat_route(pipeline_type: str) -> ShapeSLatRoute:
    if pipeline_type == "512":
        return ShapeSLatRoute(
            pipeline_type=pipeline_type,
            model_keys=("shape_slat_flow_model_512",),
            output_resolution=512,
            cascade=False,
        )
    if pipeline_type == "1024":
        return ShapeSLatRoute(
            pipeline_type=pipeline_type,
            model_keys=("shape_slat_flow_model_1024",),
            output_resolution=1024,
            cascade=False,
        )
    if pipeline_type == "1024_cascade":
        return ShapeSLatRoute(
            pipeline_type=pipeline_type,
            model_keys=("shape_slat_flow_model_512", "shape_slat_flow_model_1024"),
            output_resolution=1024,
            cascade=True,
        )
    if pipeline_type == "1536_cascade":
        return ShapeSLatRoute(
            pipeline_type=pipeline_type,
            model_keys=("shape_slat_flow_model_512", "shape_slat_flow_model_1024"),
            output_resolution=1536,
            cascade=True,
        )
    raise ValueError(f"unsupported shape SLat pipeline type: {pipeline_type}")


def select_texture_slat_route(pipeline_type: str) -> TextureSLatRoute:
    if pipeline_type == "512":
        return TextureSLatRoute(
            pipeline_type=pipeline_type,
            model_key="tex_slat_flow_model_512",
            output_resolution=512,
        )
    if pipeline_type in {"1024", "1024_cascade", "1536_cascade"}:
        return TextureSLatRoute(
            pipeline_type=pipeline_type,
            model_key="tex_slat_flow_model_1024",
            output_resolution=1024,
        )
    raise ValueError(f"unsupported texture SLat pipeline type: {pipeline_type}")


def probe_shape_slat_forward_boundary(
    checkpoint_path: str | Path,
    config: SLatFlowConfig,
    sparse_coordinates: mx.array,
    *,
    conditioning: mx.array | None = None,
    steps: int = 1,
    rescale_t: float = 1.0,
    guidance_strength: float = 1.0,
    guidance_rescale: float = 0.0,
    guidance_interval: tuple[float, float] = (0.0, 1.0),
    sigma_min: float = 1e-5,
) -> ShapeSLatForwardProbe:
    coordinates_shape = tuple(int(dim) for dim in sparse_coordinates.shape)
    _validate_sparse_coordinates(coordinates_shape)

    tensor_names = _slat_forward_tensor_names(config)
    inspection_names = _slat_stack_inspection_names(config)
    tensors = load_checkpoint_tensors(checkpoint_path, names=tensor_names)
    input_weight = tensors["input_layer.weight"]
    input_bias = tensors["input_layer.bias"]
    _validate_tensor_shape("input_layer.weight", input_weight.shape, (config.model_channels, config.in_channels))
    _validate_tensor_shape("input_layer.bias", input_bias.shape, (config.model_channels,))

    infos = inspect_checkpoint(checkpoint_path, names=inspection_names)
    _require_checkpoint_infos(infos, inspection_names)
    _validate_slat_stack_infos(infos, config)

    features = mx.random.normal((coordinates_shape[0], config.in_channels)).astype(input_weight.dtype)
    cond = conditioning
    if cond is None:
        cond = mx.zeros((1, 1, config.cond_channels), dtype=input_weight.dtype)
    _validate_slat_conditioning_shape(cond.shape, config)
    projected, block0, stack, output = _slat_model_forward(
        features,
        sparse_coordinates,
        1.0,
        cond,
        config,
        tensors,
    )
    mx.eval(projected)
    projection_shape = tuple(int(dim) for dim in projected.shape)
    expected_projection_shape = (coordinates_shape[0], config.model_channels)
    if projection_shape != expected_projection_shape:
        raise ValueError(
            f"shape SLat input projection shape mismatch: expected {expected_projection_shape}, got {projection_shape}"
        )
    block0_shape = tuple(int(dim) for dim in block0.shape)
    if block0_shape != expected_projection_shape:
        raise ValueError(f"shape SLat block-0 output shape mismatch: expected {expected_projection_shape}, got {block0_shape}")
    stack_shape = tuple(int(dim) for dim in stack.shape)
    output_shape = tuple(int(dim) for dim in output.shape)
    expected_output_shape = (coordinates_shape[0], config.out_channels)
    if output_shape != expected_output_shape:
        raise ValueError(f"shape SLat output projection shape mismatch: expected {expected_output_shape}, got {output_shape}")

    sampled = _flow_euler_sample_slat(
        features,
        sparse_coordinates,
        cond,
        config,
        tensors,
        steps=steps,
        rescale_t=rescale_t,
        guidance_strength=guidance_strength,
        guidance_rescale=guidance_rescale,
        guidance_interval=guidance_interval,
        sigma_min=sigma_min,
    )
    mx.eval(sampled)
    sampled_shape = tuple(int(dim) for dim in sampled.shape)

    return ShapeSLatForwardProbe(
        checkpoint_path=str(checkpoint_path),
        coordinate_shape=coordinates_shape,
        feature_shape=tuple(int(dim) for dim in features.shape),
        input_projection_shape=projection_shape,
        block0_output_shape=block0_shape,
        completed_blocks=config.num_blocks,
        stack_output_shape=stack_shape,
        output_projection_shape=output_shape,
        sampled_feature_shape=sampled_shape,
        sampled_features=sampled,
        loaded_tensor_names=tuple(name for name in tensor_names if name in tensors),
        inspected_tensor_names=tuple(info.name for info in infos),
        blocker_operation="shape SLat texture handoff",
        blocker_detail=(
            f"input projection executed with output shape {projection_shape}; "
            f"block-0 ModulatedSparseTransformerCrossBlock executed with output shape {block0_shape}; "
            f"all {config.num_blocks} shape SLat transformer blocks executed with stack output shape {stack_shape}; "
            f"output projection executed with output shape {output_shape}; "
            f"FlowEuler sampler executed {steps} steps with sampled feature shape {sampled_shape}; "
            "next boundary is texture SLat sampling from shape SLat coordinates/features"
        ),
    )


def probe_texture_slat_forward_boundary(
    checkpoint_path: str | Path,
    config: SLatFlowConfig,
    shape_slat_coordinates: mx.array,
    shape_slat_features: mx.array,
    *,
    conditioning: mx.array | None = None,
    steps: int = 1,
    rescale_t: float = 1.0,
    guidance_strength: float = 1.0,
    guidance_rescale: float = 0.0,
    guidance_interval: tuple[float, float] = (0.0, 1.0),
    sigma_min: float = 1e-5,
) -> TextureSLatForwardProbe:
    coordinates_shape = tuple(int(dim) for dim in shape_slat_coordinates.shape)
    _validate_sparse_coordinates(coordinates_shape)
    shape_feature_shape = tuple(int(dim) for dim in shape_slat_features.shape)
    _validate_shape_slat_features(shape_feature_shape, coordinates_shape, config)

    tensor_names = _slat_forward_tensor_names(config)
    inspection_names = _slat_stack_inspection_names(config)
    tensors = load_checkpoint_tensors(checkpoint_path, names=tensor_names)
    input_weight = tensors["input_layer.weight"]
    input_bias = tensors["input_layer.bias"]
    _validate_tensor_shape("input_layer.weight", input_weight.shape, (config.model_channels, config.in_channels))
    _validate_tensor_shape("input_layer.bias", input_bias.shape, (config.model_channels,))

    infos = inspect_checkpoint(checkpoint_path, names=inspection_names)
    _require_checkpoint_infos(infos, inspection_names)
    _validate_slat_stack_infos(infos, config)

    noise_channels = config.in_channels - shape_feature_shape[1]
    noise_features = mx.random.normal((coordinates_shape[0], noise_channels)).astype(input_weight.dtype)
    concat_features = mx.concatenate([noise_features, shape_slat_features.astype(input_weight.dtype)], axis=1)
    cond = conditioning
    if cond is None:
        cond = mx.zeros((1, 1, config.cond_channels), dtype=input_weight.dtype)
    _validate_slat_conditioning_shape(cond.shape, config)
    projected, block0, stack, output = _slat_model_forward(
        noise_features,
        shape_slat_coordinates,
        1.0,
        cond,
        config,
        tensors,
        concat_features=shape_slat_features.astype(input_weight.dtype),
    )
    mx.eval(projected)
    projection_shape = tuple(int(dim) for dim in projected.shape)
    expected_projection_shape = (coordinates_shape[0], config.model_channels)
    if projection_shape != expected_projection_shape:
        raise ValueError(
            f"texture SLat input projection shape mismatch: expected {expected_projection_shape}, got {projection_shape}"
        )
    block0_shape = tuple(int(dim) for dim in block0.shape)
    stack_shape = tuple(int(dim) for dim in stack.shape)
    output_shape = tuple(int(dim) for dim in output.shape)
    expected_output_shape = (coordinates_shape[0], config.out_channels)
    if output_shape != expected_output_shape:
        raise ValueError(f"texture SLat output projection shape mismatch: expected {expected_output_shape}, got {output_shape}")
    sampled = _flow_euler_sample_slat(
        noise_features,
        shape_slat_coordinates,
        cond,
        config,
        tensors,
        steps=steps,
        rescale_t=rescale_t,
        guidance_strength=guidance_strength,
        guidance_rescale=guidance_rescale,
        guidance_interval=guidance_interval,
        sigma_min=sigma_min,
        concat_features=shape_slat_features.astype(input_weight.dtype),
    )
    mx.eval(sampled)
    sampled_shape = tuple(int(dim) for dim in sampled.shape)

    return TextureSLatForwardProbe(
        checkpoint_path=str(checkpoint_path),
        coordinate_shape=coordinates_shape,
        shape_feature_shape=shape_feature_shape,
        noise_feature_shape=tuple(int(dim) for dim in noise_features.shape),
        concat_feature_shape=tuple(int(dim) for dim in concat_features.shape),
        input_projection_shape=projection_shape,
        block0_output_shape=block0_shape,
        completed_blocks=config.num_blocks,
        stack_output_shape=stack_shape,
        output_projection_shape=output_shape,
        sampled_feature_shape=sampled_shape,
        sampled_features=sampled,
        loaded_tensor_names=tuple(name for name in tensor_names if name in tensors),
        inspected_tensor_names=tuple(info.name for info in infos),
        blocker_operation="texture SLat decode handoff",
        blocker_detail=(
            f"concat feature projection executed with output shape {projection_shape}; "
            f"block-0 ModulatedSparseTransformerCrossBlock executed with output shape {block0_shape}; "
            f"all {config.num_blocks} texture SLat transformer blocks executed with stack output shape {stack_shape}; "
            f"output projection executed with output shape {output_shape}; "
            f"FlowEuler sampler executed {steps} steps with sampled feature shape {sampled_shape}; "
            "next boundary is latent decoding from shape and texture SLat features"
        ),
    )


def _slat_model_forward(
    features: mx.array,
    coordinates: mx.array,
    t: float,
    conditioning: mx.array,
    config: SLatFlowConfig,
    tensors: dict[str, mx.array],
    *,
    concat_features: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    _validate_single_batch_coordinates(coordinates)
    input_features = features if concat_features is None else mx.concatenate([features, concat_features.astype(features.dtype)], axis=1)
    if int(input_features.shape[-1]) != config.in_channels:
        raise ValueError(f"SLat input feature width mismatch: expected {config.in_channels}, got {int(input_features.shape[-1])}")
    projected = _linear(
        input_features,
        tensors["input_layer.weight"].astype(mx.float32),
        tensors["input_layer.bias"].astype(mx.float32),
    )
    mod = _shared_slat_modulation(config, tensors, t=t)
    hidden_states = projected
    block0 = None
    for block_index in range(config.num_blocks):
        hidden_states = _slat_block_forward(
            hidden_states,
            coordinates,
            conditioning,
            config,
            tensors,
            mod=mod,
            block_index=block_index,
        )
        if block_index == 0:
            block0 = hidden_states
    if block0 is None:
        raise ValueError("SLat flow config must contain at least one transformer block")
    stack = _layer_norm_no_affine(hidden_states, eps=1e-6)
    output = _linear(
        stack,
        tensors["out_layer.weight"].astype(mx.float32),
        tensors["out_layer.bias"].astype(mx.float32),
    )
    return projected, block0, stack, output


def _flow_euler_sample_slat(
    sample: mx.array,
    coordinates: mx.array,
    conditioning: mx.array,
    config: SLatFlowConfig,
    tensors: dict[str, mx.array],
    *,
    steps: int,
    rescale_t: float,
    guidance_strength: float,
    guidance_rescale: float,
    guidance_interval: tuple[float, float],
    sigma_min: float,
    concat_features: mx.array | None = None,
) -> mx.array:
    schedule = flow_euler_schedule(steps=steps, rescale_t=rescale_t, guidance_interval=guidance_interval)
    neg_conditioning = mx.zeros_like(conditioning)
    for t, t_prev in schedule.pairs:
        pred_v = _flow_euler_guided_slat_prediction(
            sample,
            coordinates,
            t,
            conditioning,
            neg_conditioning,
            config,
            tensors,
            guidance_strength=guidance_strength if guidance_interval[0] <= t <= guidance_interval[1] else 1.0,
            guidance_rescale=guidance_rescale,
            sigma_min=sigma_min,
            concat_features=concat_features,
        )
        sample = sample - (t - t_prev) * pred_v
        mx.eval(sample)
    return sample


def _flow_euler_guided_slat_prediction(
    sample: mx.array,
    coordinates: mx.array,
    t: float,
    conditioning: mx.array,
    neg_conditioning: mx.array,
    config: SLatFlowConfig,
    tensors: dict[str, mx.array],
    *,
    guidance_strength: float,
    guidance_rescale: float,
    sigma_min: float,
    concat_features: mx.array | None = None,
) -> mx.array:
    if guidance_strength == 1:
        return _flow_euler_slat_model_prediction(sample, coordinates, t, conditioning, config, tensors, concat_features=concat_features)
    if guidance_strength == 0:
        return _flow_euler_slat_model_prediction(sample, coordinates, t, neg_conditioning, config, tensors, concat_features=concat_features)

    pred_pos = _flow_euler_slat_model_prediction(sample, coordinates, t, conditioning, config, tensors, concat_features=concat_features)
    pred_neg = _flow_euler_slat_model_prediction(sample, coordinates, t, neg_conditioning, config, tensors, concat_features=concat_features)
    pred = guidance_strength * pred_pos + (1.0 - guidance_strength) * pred_neg
    if guidance_rescale <= 0:
        return pred

    x0_pos = _flow_euler_slat_pred_to_xstart(sample, t, pred_pos, sigma_min=sigma_min)
    x0_cfg = _flow_euler_slat_pred_to_xstart(sample, t, pred, sigma_min=sigma_min)
    std_pos = mx.std(x0_pos, axis=0, keepdims=True)
    std_cfg = mx.maximum(mx.std(x0_cfg, axis=0, keepdims=True), mx.array(1e-12, dtype=mx.float32))
    x0_rescaled = x0_cfg * (std_pos / std_cfg)
    x0 = guidance_rescale * x0_rescaled + (1.0 - guidance_rescale) * x0_cfg
    return _flow_euler_slat_xstart_to_pred(sample, t, x0, sigma_min=sigma_min)


def _flow_euler_slat_model_prediction(
    sample: mx.array,
    coordinates: mx.array,
    t: float,
    conditioning: mx.array,
    config: SLatFlowConfig,
    tensors: dict[str, mx.array],
    *,
    concat_features: mx.array | None = None,
) -> mx.array:
    _, _, _, output = _slat_model_forward(
        sample,
        coordinates,
        t,
        conditioning,
        config,
        tensors,
        concat_features=concat_features,
    )
    return output


def _flow_euler_slat_pred_to_xstart(sample: mx.array, t: float, pred: mx.array, *, sigma_min: float) -> mx.array:
    return (1.0 - sigma_min) * sample - (sigma_min + (1.0 - sigma_min) * t) * pred


def _flow_euler_slat_xstart_to_pred(sample: mx.array, t: float, x0: mx.array, *, sigma_min: float) -> mx.array:
    return ((1.0 - sigma_min) * sample - x0) / (sigma_min + (1.0 - sigma_min) * t)


def _slat_block_forward(
    hidden_states: mx.array,
    coordinates: mx.array,
    conditioning: mx.array,
    config: SLatFlowConfig,
    tensors: dict[str, mx.array],
    *,
    mod: mx.array,
    block_index: int,
) -> mx.array:
    hidden_states = hidden_states.astype(mx.float32)
    conditioning = conditioning.astype(mx.float32)
    prefix = f"blocks.{block_index}"
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _split_modulation(
        tensors[f"{prefix}.modulation"].astype(mx.float32)[None, :] + mod
    )

    normalized = _layer_norm_no_affine(hidden_states, eps=1e-6)
    normalized = normalized * (1.0 + scale_msa[0]) + shift_msa[0]
    self_attn = _slat_self_attention(normalized, coordinates, config, tensors, block_index=block_index)
    hidden_states = hidden_states + (self_attn * gate_msa[0])

    normalized = _layer_norm(
        hidden_states,
        tensors[f"{prefix}.norm2.weight"].astype(mx.float32),
        tensors[f"{prefix}.norm2.bias"].astype(mx.float32),
        eps=1e-6,
    )
    hidden_states = hidden_states + _slat_cross_attention(normalized, conditioning, config, tensors, block_index=block_index)

    normalized = _layer_norm_no_affine(hidden_states, eps=1e-6)
    normalized = normalized * (1.0 + scale_mlp[0]) + shift_mlp[0]
    mlp = _slat_mlp(normalized, tensors, block_index=block_index)
    return hidden_states + (mlp * gate_mlp[0])


def _shared_slat_modulation(config: SLatFlowConfig, tensors: dict[str, mx.array], *, t: float) -> mx.array:
    timestep = mx.array([1000.0 * t], dtype=mx.float32)
    embedding = _timestep_embedding(timestep, 256)
    hidden = _linear(
        embedding,
        tensors["t_embedder.mlp.0.weight"].astype(mx.float32),
        tensors["t_embedder.mlp.0.bias"].astype(mx.float32),
    )
    hidden = _silu(hidden)
    hidden = _linear(
        hidden,
        tensors["t_embedder.mlp.2.weight"].astype(mx.float32),
        tensors["t_embedder.mlp.2.bias"].astype(mx.float32),
    )
    if not config.share_mod:
        raise ValueError("non-shared SLat block modulation is not mapped yet")
    hidden = _silu(hidden)
    return _linear(
        hidden,
        tensors["adaLN_modulation.1.weight"].astype(mx.float32),
        tensors["adaLN_modulation.1.bias"].astype(mx.float32),
    )


def _slat_self_attention(
    hidden_states: mx.array,
    coordinates: mx.array,
    config: SLatFlowConfig,
    tensors: dict[str, mx.array],
    *,
    block_index: int,
) -> mx.array:
    token_count = int(hidden_states.shape[0])
    head_dim = _slat_head_dim(config)
    prefix = f"blocks.{block_index}.self_attn"
    qkv = _linear(
        hidden_states,
        tensors[f"{prefix}.to_qkv.weight"].astype(mx.float32),
        tensors[f"{prefix}.to_qkv.bias"].astype(mx.float32),
    )
    qkv = mx.reshape(qkv, (1, token_count, 3, config.num_heads, head_dim))
    query = qkv[:, :, 0, :, :]
    key = qkv[:, :, 1, :, :]
    value = qkv[:, :, 2, :, :]
    if config.qk_rms_norm:
        query = _multi_head_rms_norm(query, tensors[f"{prefix}.q_rms_norm.gamma"].astype(mx.float32))
        key = _multi_head_rms_norm(key, tensors[f"{prefix}.k_rms_norm.gamma"].astype(mx.float32))
    if config.pe_mode == "rope":
        cos, sin = _slat_rope_cos_sin(coordinates, config)
        query = _apply_slat_rope(query, cos, sin)
        key = _apply_slat_rope(key, cos, sin)

    attended = _slat_self_attention_kernel(query, key, value, coordinates, head_dim=head_dim)
    attended = mx.reshape(attended, (token_count, config.model_channels))
    return _linear(
        attended,
        tensors[f"{prefix}.to_out.weight"].astype(mx.float32),
        tensors[f"{prefix}.to_out.bias"].astype(mx.float32),
    )


def _slat_cross_attention(
    hidden_states: mx.array,
    conditioning: mx.array,
    config: SLatFlowConfig,
    tensors: dict[str, mx.array],
    *,
    block_index: int,
) -> mx.array:
    token_count = int(hidden_states.shape[0])
    cond_count = int(conditioning.shape[1])
    head_dim = _slat_head_dim(config)
    prefix = f"blocks.{block_index}.cross_attn"
    query = _linear(
        hidden_states,
        tensors[f"{prefix}.to_q.weight"].astype(mx.float32),
        tensors[f"{prefix}.to_q.bias"].astype(mx.float32),
    )
    key_value = _linear(
        conditioning,
        tensors[f"{prefix}.to_kv.weight"].astype(mx.float32),
        tensors[f"{prefix}.to_kv.bias"].astype(mx.float32),
    )
    query = mx.reshape(query, (1, token_count, config.num_heads, head_dim))
    key_value = mx.reshape(key_value, (1, cond_count, 2, config.num_heads, head_dim))
    key = key_value[:, :, 0, :, :]
    value = key_value[:, :, 1, :, :]
    if config.qk_rms_norm_cross:
        query = _multi_head_rms_norm(query, tensors[f"{prefix}.q_rms_norm.gamma"].astype(mx.float32))
        key = _multi_head_rms_norm(key, tensors[f"{prefix}.k_rms_norm.gamma"].astype(mx.float32))

    attended = _attention(query, key, value, head_dim=head_dim)
    attended = mx.reshape(attended, (token_count, config.model_channels))
    return _linear(
        attended,
        tensors[f"{prefix}.to_out.weight"].astype(mx.float32),
        tensors[f"{prefix}.to_out.bias"].astype(mx.float32),
    )


def _slat_self_attention_kernel(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    coordinates: mx.array,
    *,
    head_dim: int,
) -> mx.array:
    token_count = int(query.shape[1])
    if token_count > SLAT_FULL_SELF_ATTN_TOKEN_LIMIT:
        raise ValueError(
            "exact full SLat self-attention would exceed the configured token guard: "
            f"{token_count} > {SLAT_FULL_SELF_ATTN_TOKEN_LIMIT}"
        )
    if token_count <= SLAT_DENSE_SELF_ATTN_THRESHOLD:
        return _attention(query, key, value, head_dim=head_dim)
    return _slat_full_self_attention_chunked(query, key, value, head_dim=head_dim)


def _slat_full_self_attention_chunked(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    head_dim: int,
    query_chunk_size: int = SLAT_FULL_SELF_ATTN_QUERY_CHUNK_SIZE,
) -> mx.array:
    """Run exact full self-attention by query blocks without allocating all query rows at once."""

    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
    token_count = int(query.shape[1])
    chunks: list[mx.array] = []
    for start in range(0, token_count, query_chunk_size):
        end = min(start + query_chunk_size, token_count)
        attended = _attention(query[:, start:end, :, :], key, value, head_dim=head_dim)
        mx.eval(attended)
        chunks.append(attended)
    return mx.concatenate(chunks, axis=1)


def _slat_windowed_self_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    coordinates: mx.array,
    *,
    head_dim: int,
    window_size: int,
) -> mx.array:
    """Apply block-diagonal sparse self-attention inside 3D coordinate windows."""

    groups = _slat_window_groups(coordinates, window_size=window_size)
    if not groups:
        return mx.zeros_like(query)
    max_len = max(len(group) for group in groups)
    padded = np.full((len(groups), max_len), 0, dtype=np.int32)
    valid = np.zeros((len(groups), max_len), dtype=bool)
    ordered_indices: list[int] = []
    for row, group in enumerate(groups):
        padded[row, : len(group)] = np.array(group, dtype=np.int32)
        valid[row, : len(group)] = True
        ordered_indices.extend(group)

    padded_indices = mx.array(padded, dtype=mx.int32)
    query_windows = query[0][padded_indices]
    key_windows = key[0][padded_indices]
    value_windows = value[0][padded_indices]
    valid_mask = mx.array(valid, dtype=mx.bool_)
    attn_mask = valid_mask[:, None, :, None] & valid_mask[:, None, None, :]
    attended = _slat_attention(query_windows, key_windows, value_windows, head_dim=head_dim, mask=attn_mask)

    flat_attended = mx.reshape(attended, (len(groups) * max_len, int(query.shape[2]), int(query.shape[3])))
    flat_valid_positions = np.flatnonzero(valid.reshape(-1)).astype(np.int32)
    serialized = flat_attended[mx.array(flat_valid_positions, dtype=mx.int32)]
    inverse = np.empty((len(ordered_indices),), dtype=np.int32)
    inverse[np.array(ordered_indices, dtype=np.int32)] = np.arange(len(ordered_indices), dtype=np.int32)
    restored = serialized[mx.array(inverse, dtype=mx.int32)]
    return restored[None, :, :, :]


def _slat_window_groups(coordinates: mx.array, *, window_size: int) -> tuple[tuple[int, ...], ...]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    coords = np.array(coordinates[:, 1:], dtype=np.int32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"SLat window coordinates must have shape (num_tokens, 3), got {coords.shape}")
    window_coords = coords // window_size
    keys = window_coords[:, 0].astype(np.int64)
    keys = (keys << 42) + (window_coords[:, 1].astype(np.int64) << 21) + window_coords[:, 2].astype(np.int64)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    groups: list[tuple[int, ...]] = []
    start = 0
    for index in range(1, len(order) + 1):
        if index == len(order) or sorted_keys[index] != sorted_keys[start]:
            groups.append(tuple(int(item) for item in order[start:index]))
            start = index
    return tuple(groups)


def _slat_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    head_dim: int,
    mask: mx.array | None = None,
) -> mx.array:
    query = mx.transpose(query, (0, 2, 1, 3))
    key = mx.transpose(key, (0, 2, 1, 3))
    value = mx.transpose(value, (0, 2, 1, 3))
    attended = mx.fast.scaled_dot_product_attention(query, key, value, scale=head_dim**-0.5, mask=mask)
    return mx.transpose(attended, (0, 2, 1, 3))


def _slat_mlp(hidden_states: mx.array, tensors: dict[str, mx.array], *, block_index: int) -> mx.array:
    prefix = f"blocks.{block_index}.mlp.mlp"
    hidden = _linear(
        hidden_states,
        tensors[f"{prefix}.0.weight"].astype(mx.float32),
        tensors[f"{prefix}.0.bias"].astype(mx.float32),
    )
    hidden = _gelu_tanh(hidden)
    return _linear(
        hidden,
        tensors[f"{prefix}.2.weight"].astype(mx.float32),
        tensors[f"{prefix}.2.bias"].astype(mx.float32),
    )


def _slat_rope_cos_sin(coordinates: mx.array, config: SLatFlowConfig) -> tuple[mx.array, mx.array]:
    head_dim = _slat_head_dim(config)
    if head_dim % 2:
        raise ValueError(f"SLat 3D RoPE requires even head_dim, got {head_dim}")
    freq_dim = head_dim // 2 // 3
    if freq_dim <= 0:
        raise ValueError(f"SLat 3D RoPE requires head_dim large enough for 3 axes, got {head_dim}")
    positions = coordinates[:, 1:].astype(mx.float32)
    freq_index = mx.arange(freq_dim, dtype=mx.float32) / float(freq_dim)
    freqs = 1.0 / (10000.0 ** freq_index)
    angles = mx.reshape(positions[:, :, None] * freqs[None, None, :], (int(coordinates.shape[0]), -1))
    pair_count = head_dim // 2
    if int(angles.shape[-1]) < pair_count:
        padding = mx.zeros((int(angles.shape[0]), pair_count - int(angles.shape[-1])), dtype=angles.dtype)
        angles = mx.concatenate((angles, padding), axis=-1)
    return mx.cos(angles), mx.sin(angles)


def _apply_slat_rope(values: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    batch, token_count, heads, head_dim = tuple(int(dim) for dim in values.shape)
    pairs = mx.reshape(values.astype(mx.float32), (batch, token_count, heads, head_dim // 2, 2))
    first = pairs[..., 0]
    second = pairs[..., 1]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rotated = mx.stack((first * cos - second * sin, first * sin + second * cos), axis=-1)
    return mx.reshape(rotated, (batch, token_count, heads, head_dim))


def _slat_forward_tensor_names(config: SLatFlowConfig) -> tuple[str, ...]:
    return (
        *SLAT_INPUT_TENSOR_NAMES,
        "out_layer.weight",
        "out_layer.bias",
        *_slat_stack_inspection_names(config),
    )


def _slat_stack_inspection_names(config: SLatFlowConfig) -> tuple[str, ...]:
    names = [
        "t_embedder.mlp.0.weight",
        "t_embedder.mlp.0.bias",
        "t_embedder.mlp.2.weight",
        "t_embedder.mlp.2.bias",
        "adaLN_modulation.1.weight",
        "adaLN_modulation.1.bias",
        "out_layer.weight",
        "out_layer.bias",
    ]
    for block_index in range(config.num_blocks):
        names.extend(_slat_block_tensor_names(block_index))
    return tuple(names)


def _slat_block_tensor_names(block_index: int) -> tuple[str, ...]:
    prefix = f"blocks.{block_index}"
    return (
        f"{prefix}.modulation",
        f"{prefix}.norm2.weight",
        f"{prefix}.norm2.bias",
        f"{prefix}.self_attn.to_qkv.weight",
        f"{prefix}.self_attn.to_qkv.bias",
        f"{prefix}.self_attn.q_rms_norm.gamma",
        f"{prefix}.self_attn.k_rms_norm.gamma",
        f"{prefix}.self_attn.to_out.weight",
        f"{prefix}.self_attn.to_out.bias",
        f"{prefix}.cross_attn.to_q.weight",
        f"{prefix}.cross_attn.to_q.bias",
        f"{prefix}.cross_attn.to_kv.weight",
        f"{prefix}.cross_attn.to_kv.bias",
        f"{prefix}.cross_attn.q_rms_norm.gamma",
        f"{prefix}.cross_attn.k_rms_norm.gamma",
        f"{prefix}.cross_attn.to_out.weight",
        f"{prefix}.cross_attn.to_out.bias",
        f"{prefix}.mlp.mlp.0.weight",
        f"{prefix}.mlp.mlp.0.bias",
        f"{prefix}.mlp.mlp.2.weight",
        f"{prefix}.mlp.mlp.2.bias",
    )


def _validate_slat_stack_infos(infos: tuple[CheckpointTensorInfo, ...], config: SLatFlowConfig) -> None:
    if config.share_mod:
        _validate_checkpoint_info_shape(
            infos,
            "adaLN_modulation.1.weight",
            (config.model_channels * 6, config.model_channels),
        )
    _validate_checkpoint_info_shape(infos, "adaLN_modulation.1.bias", (config.model_channels * 6,))
    _validate_checkpoint_info_shape(infos, "out_layer.weight", (config.out_channels, config.model_channels))
    _validate_checkpoint_info_shape(infos, "out_layer.bias", (config.out_channels,))
    intermediate_channels = int(config.model_channels * config.mlp_ratio)
    head_dim = _slat_head_dim(config)
    for block_index in range(config.num_blocks):
        prefix = f"blocks.{block_index}"
        _validate_checkpoint_info_shape(infos, f"{prefix}.modulation", (config.model_channels * 6,))
        _validate_checkpoint_info_shape(infos, f"{prefix}.norm2.weight", (config.model_channels,))
        _validate_checkpoint_info_shape(infos, f"{prefix}.norm2.bias", (config.model_channels,))
        _validate_checkpoint_info_shape(
            infos,
            f"{prefix}.self_attn.to_qkv.weight",
            (config.model_channels * 3, config.model_channels),
        )
        _validate_checkpoint_info_shape(infos, f"{prefix}.self_attn.to_qkv.bias", (config.model_channels * 3,))
        _validate_checkpoint_info_shape(infos, f"{prefix}.self_attn.q_rms_norm.gamma", (config.num_heads, head_dim))
        _validate_checkpoint_info_shape(infos, f"{prefix}.self_attn.k_rms_norm.gamma", (config.num_heads, head_dim))
        _validate_checkpoint_info_shape(
            infos,
            f"{prefix}.self_attn.to_out.weight",
            (config.model_channels, config.model_channels),
        )
        _validate_checkpoint_info_shape(infos, f"{prefix}.self_attn.to_out.bias", (config.model_channels,))
        _validate_checkpoint_info_shape(
            infos,
            f"{prefix}.cross_attn.to_q.weight",
            (config.model_channels, config.model_channels),
        )
        _validate_checkpoint_info_shape(infos, f"{prefix}.cross_attn.to_q.bias", (config.model_channels,))
        _validate_checkpoint_info_shape(
            infos,
            f"{prefix}.cross_attn.to_kv.weight",
            (config.model_channels * 2, config.cond_channels),
        )
        _validate_checkpoint_info_shape(infos, f"{prefix}.cross_attn.to_kv.bias", (config.model_channels * 2,))
        _validate_checkpoint_info_shape(infos, f"{prefix}.cross_attn.q_rms_norm.gamma", (config.num_heads, head_dim))
        _validate_checkpoint_info_shape(infos, f"{prefix}.cross_attn.k_rms_norm.gamma", (config.num_heads, head_dim))
        _validate_checkpoint_info_shape(
            infos,
            f"{prefix}.cross_attn.to_out.weight",
            (config.model_channels, config.model_channels),
        )
        _validate_checkpoint_info_shape(infos, f"{prefix}.cross_attn.to_out.bias", (config.model_channels,))
        _validate_checkpoint_info_shape(infos, f"{prefix}.mlp.mlp.0.weight", (intermediate_channels, config.model_channels))
        _validate_checkpoint_info_shape(infos, f"{prefix}.mlp.mlp.0.bias", (intermediate_channels,))
        _validate_checkpoint_info_shape(infos, f"{prefix}.mlp.mlp.2.weight", (config.model_channels, intermediate_channels))
        _validate_checkpoint_info_shape(infos, f"{prefix}.mlp.mlp.2.bias", (config.model_channels,))


def _slat_head_dim(config: SLatFlowConfig) -> int:
    if config.model_channels % config.num_heads:
        raise ValueError(
            f"SLat attention head dimension mismatch: model_channels={config.model_channels}, num_heads={config.num_heads}"
        )
    return config.model_channels // config.num_heads


def _validate_single_batch_coordinates(coordinates: mx.array) -> None:
    batches = np.unique(np.array(coordinates[:, 0]))
    if batches.shape[0] != 1 or int(batches[0]) != 0:
        raise ValueError("SLat sparse MLX forward currently expects a single batch with coordinate batch index 0")


def _validate_slat_conditioning_shape(shape: tuple[int, ...], config: SLatFlowConfig) -> None:
    actual = tuple(int(dim) for dim in shape)
    if len(actual) != 3:
        raise ValueError(f"SLat conditioning must have shape (batch, tokens, channels), got {actual}")
    if actual[0] != 1:
        raise ValueError(f"SLat conditioning currently expects batch size 1, got {actual[0]}")
    if actual[1] <= 0:
        raise ValueError("SLat conditioning must contain at least one token")
    if actual[2] != config.cond_channels:
        raise ValueError(f"SLat conditioning width mismatch: expected {config.cond_channels}, got {actual[2]}")


def _validate_sparse_coordinates(shape: tuple[int, ...]) -> None:
    if len(shape) != 2 or shape[1] != 4:
        raise ValueError(f"sparse coordinates must have shape (num_tokens, 4), got {shape}")
    if shape[0] <= 0:
        raise ValueError("sparse coordinates must contain at least one token")


def _validate_shape_slat_features(
    feature_shape: tuple[int, ...],
    coordinate_shape: tuple[int, int],
    config: SLatFlowConfig,
) -> None:
    if len(feature_shape) != 2:
        raise ValueError(f"shape SLat features must have shape (num_tokens, channels), got {feature_shape}")
    if feature_shape[0] != coordinate_shape[0]:
        raise ValueError(
            f"shape SLat feature/token mismatch: expected {coordinate_shape[0]} tokens, got {feature_shape[0]}"
        )
    if feature_shape[1] <= 0 or feature_shape[1] >= config.in_channels:
        raise ValueError(
            f"shape SLat feature width must be between 1 and texture flow in_channels-1, got {feature_shape[1]}"
        )


def _require_checkpoint_infos(
    infos: tuple[CheckpointTensorInfo, ...],
    expected_names: tuple[str, ...],
) -> None:
    present = {info.name for info in infos}
    missing = sorted(set(expected_names).difference(present))
    if missing:
        raise ValueError(f"checkpoint is missing requested tensors: {missing}")


def _validate_checkpoint_info_shape(
    infos: tuple[CheckpointTensorInfo, ...],
    name: str,
    expected: tuple[int, ...],
) -> None:
    for info in infos:
        if info.name == name:
            _validate_tensor_shape(name, info.shape, expected)
            return
    raise ValueError(f"checkpoint is missing requested tensor: {name}")


def _validate_tensor_shape(name: str, actual: tuple[int, ...], expected: tuple[int, ...]) -> None:
    actual_shape = tuple(int(dim) for dim in actual)
    if actual_shape != expected:
        raise ValueError(f"{name} shape mismatch: expected {expected}, got {actual_shape}")
