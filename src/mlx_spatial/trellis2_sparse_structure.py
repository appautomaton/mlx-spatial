"""Sparse-structure sampling contracts for TRELLIS.2 forward tracing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from .checkpoint import CheckpointTensorInfo, inspect_checkpoint, load_checkpoint_tensors


SPARSE_STRUCTURE_INPUT_TENSOR_NAMES = (
    "input_layer.weight",
    "input_layer.bias",
)

SPARSE_STRUCTURE_BLOCK0_INSPECTION_NAMES = (
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
    "blocks.0.self_attn.q_rms_norm.gamma",
    "blocks.0.self_attn.k_rms_norm.gamma",
    "blocks.0.self_attn.to_out.weight",
    "blocks.0.self_attn.to_out.bias",
    "blocks.0.cross_attn.to_q.weight",
    "blocks.0.cross_attn.to_q.bias",
    "blocks.0.cross_attn.to_kv.weight",
    "blocks.0.cross_attn.to_kv.bias",
    "blocks.0.cross_attn.q_rms_norm.gamma",
    "blocks.0.cross_attn.k_rms_norm.gamma",
    "blocks.0.cross_attn.to_out.weight",
    "blocks.0.cross_attn.to_out.bias",
    "blocks.0.mlp.mlp.0.weight",
    "blocks.0.mlp.mlp.0.bias",
    "blocks.0.mlp.mlp.2.weight",
    "blocks.0.mlp.mlp.2.bias",
)

SPARSE_STRUCTURE_BLOCK0_FORWARD_TENSOR_NAMES = SPARSE_STRUCTURE_INPUT_TENSOR_NAMES + SPARSE_STRUCTURE_BLOCK0_INSPECTION_NAMES

SPARSE_STRUCTURE_DECODER_TENSOR_NAMES = (
    "input_layer.weight",
    "input_layer.bias",
    "out_layer.0.weight",
    "out_layer.0.bias",
    "out_layer.2.weight",
    "out_layer.2.bias",
)


@dataclass(frozen=True)
class SparseStructureFlowConfig:
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
    image_attn_mode: str = "cross"
    proj_in_channels: int | None = None


@dataclass(frozen=True)
class FlowEulerSchedule:
    steps: int
    rescale_t: float
    pairs: tuple[tuple[float, float], ...]
    guidance_active: tuple[bool, ...]


@dataclass(frozen=True)
class SparseStructureSamplingMetadata:
    noise_shape: tuple[int, int, int, int, int]
    sample_shape: tuple[int, int, int, int, int]
    dtype: str
    steps: int
    guidance_active_steps: int


@dataclass(frozen=True)
class SparseStructureForwardProbe:
    token_shape: tuple[int, int, int]
    input_projection_shape: tuple[int, int, int]
    block0_output_shape: tuple[int, int, int] | None
    completed_blocks: int
    stack_output_shape: tuple[int, int, int] | None
    output_projection_shape: tuple[int, int, int] | None
    sampled_latent_shape: tuple[int, int, int, int, int] | None
    loaded_tensor_names: tuple[str, ...]
    inspected_tensor_names: tuple[str, ...]
    blocker_operation: str
    blocker_detail: str
    sampled_latent: mx.array | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class SparseStructureDecoderConfig:
    name: str
    out_channels: int
    latent_channels: int
    num_res_blocks: int
    channels: tuple[int, ...]
    num_res_blocks_middle: int
    norm_type: str
    use_fp16: bool


@dataclass(frozen=True)
class SparseStructureDecoderProbe:
    checkpoint_path: str
    loaded_tensor_names: tuple[str, ...]
    inspected_tensor_names: tuple[str, ...]
    latent_shape: tuple[int, ...] | None
    decoded_shape: tuple[int, int, int, int, int] | None
    coordinates_shape: tuple[int, int] | None
    target_resolution: int | None
    blocker_operation: str
    blocker_detail: str
    decoded_logits: mx.array | None = field(default=None, compare=False, repr=False)
    coordinates: mx.array | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class SparseStructureCoordinates:
    decoded_shape: tuple[int, int, int, int, int]
    coordinate_shape: tuple[int, int]
    target_resolution: int
    coordinates: mx.array


def read_sparse_structure_flow_config(root: str | Path, config_path: str) -> SparseStructureFlowConfig:
    path = Path(root) / config_path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        args = payload["args"]
        return SparseStructureFlowConfig(
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
            image_attn_mode=str(args.get("image_attn_mode", "cross")),
            proj_in_channels=int(args["proj_in_channels"]) if args.get("proj_in_channels") is not None else None,
        )
    except FileNotFoundError:
        raise
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"sparse structure flow config is invalid: {error}") from error


def read_sparse_structure_decoder_config(root: str | Path, config_path: str) -> SparseStructureDecoderConfig:
    path = Path(root) / config_path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        args = payload["args"]
        channels = tuple(int(channel) for channel in args["channels"])
        if not channels:
            raise ValueError("channels must be non-empty")
        return SparseStructureDecoderConfig(
            name=str(payload["name"]),
            out_channels=int(args["out_channels"]),
            latent_channels=int(args["latent_channels"]),
            num_res_blocks=int(args["num_res_blocks"]),
            channels=channels,
            num_res_blocks_middle=int(args.get("num_res_blocks_middle", 2)),
            norm_type=str(args.get("norm_type", "layer")),
            use_fp16=bool(args.get("use_fp16", False)),
        )
    except FileNotFoundError:
        raise
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"sparse structure decoder config is invalid: {error}") from error


def flow_euler_schedule(
    *,
    steps: int,
    rescale_t: float,
    guidance_interval: tuple[float, float],
) -> FlowEulerSchedule:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if rescale_t <= 0:
        raise ValueError("rescale_t must be positive")
    if len(guidance_interval) != 2 or guidance_interval[0] > guidance_interval[1]:
        raise ValueError("guidance_interval must be ordered pair")
    t_seq = np.linspace(1.0, 0.0, steps + 1)
    t_seq = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)
    pairs = tuple((float(t_seq[index]), float(t_seq[index + 1])) for index in range(steps))
    active = tuple(guidance_interval[0] <= t <= guidance_interval[1] for t, _ in pairs)
    return FlowEulerSchedule(
        steps=steps,
        rescale_t=rescale_t,
        pairs=pairs,
        guidance_active=active,
    )


def expected_sparse_noise_shape(config: SparseStructureFlowConfig, *, batch_size: int = 1) -> tuple[int, int, int, int, int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return (
        batch_size,
        config.in_channels,
        config.resolution,
        config.resolution,
        config.resolution,
    )


def validate_sparse_noise_shape(shape: tuple[int, ...], config: SparseStructureFlowConfig, *, batch_size: int = 1) -> None:
    expected = expected_sparse_noise_shape(config, batch_size=batch_size)
    if shape != expected:
        raise ValueError(f"sparse noise shape mismatch: expected {expected}, got {shape}")


def fake_sparse_structure_sampling_metadata(
    config: SparseStructureFlowConfig,
    *,
    steps: int,
    rescale_t: float,
    guidance_interval: tuple[float, float],
    batch_size: int = 1,
) -> SparseStructureSamplingMetadata:
    noise_shape = expected_sparse_noise_shape(config, batch_size=batch_size)
    schedule = flow_euler_schedule(steps=steps, rescale_t=rescale_t, guidance_interval=guidance_interval)
    return SparseStructureSamplingMetadata(
        noise_shape=noise_shape,
        sample_shape=(
            batch_size,
            config.out_channels,
            config.resolution,
            config.resolution,
            config.resolution,
        ),
        dtype=config.dtype,
        steps=schedule.steps,
        guidance_active_steps=sum(schedule.guidance_active),
    )


def probe_sparse_structure_forward_boundary(
    checkpoint_path: str | Path,
    config: SparseStructureFlowConfig,
    *,
    batch_size: int = 1,
    conditioning: mx.array | dict[str, mx.array] | tuple[mx.array, mx.array] | None = None,
    steps: int = 1,
    rescale_t: float = 1.0,
    guidance_strength: float = 1.0,
    guidance_rescale: float = 0.0,
    guidance_interval: tuple[float, float] = (0.0, 1.0),
    sigma_min: float = 1e-5,
) -> SparseStructureForwardProbe:
    """Run the MLX sparse-flow stack and deterministic FlowEuler sampling probe."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    forward_tensor_names = _sparse_structure_forward_tensor_names(config)
    inspection_names = _sparse_structure_stack_inspection_names(config)
    tensors = load_checkpoint_tensors(checkpoint_path, names=forward_tensor_names)
    input_weight = tensors["input_layer.weight"]
    input_bias = tensors["input_layer.bias"]
    _validate_tensor_shape("input_layer.weight", input_weight.shape, (config.model_channels, config.in_channels))
    _validate_tensor_shape("input_layer.bias", input_bias.shape, (config.model_channels,))

    infos = inspect_checkpoint(checkpoint_path, names=inspection_names)
    _require_checkpoint_infos(infos, inspection_names)
    _validate_sparse_structure_stack_infos(infos, config)

    sample = mx.random.normal(expected_sparse_noise_shape(config, batch_size=batch_size)).astype(input_weight.dtype)
    cond = conditioning
    if cond is None:
        cond = mx.zeros((batch_size, 1, config.cond_channels), dtype=input_weight.dtype)
    _validate_conditioning(cond, config, batch_size=batch_size, token_count=config.resolution**3)
    projected, block0, stack, output = _sparse_structure_model_forward(sample, 1.0, cond, config, tensors)
    mx.eval(projected)
    projection_shape = tuple(int(dim) for dim in projected.shape)
    token_count = config.resolution**3
    expected_projection_shape = (batch_size, token_count, config.model_channels)
    if projection_shape != expected_projection_shape:
        raise ValueError(
            f"sparse input projection shape mismatch: expected {expected_projection_shape}, got {projection_shape}"
        )
    block0_shape = tuple(int(dim) for dim in block0.shape)
    if block0_shape != expected_projection_shape:
        raise ValueError(f"sparse block-0 output shape mismatch: expected {expected_projection_shape}, got {block0_shape}")

    completed_blocks = config.num_blocks
    stack_shape = tuple(int(dim) for dim in stack.shape)
    output_shape = tuple(int(dim) for dim in output.shape)
    expected_output_shape = (batch_size, token_count, config.out_channels)
    if output_shape != expected_output_shape:
        raise ValueError(f"sparse output projection shape mismatch: expected {expected_output_shape}, got {output_shape}")
    sampled_latent = _flow_euler_sample_sparse_structure(
        sample,
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
    mx.eval(sampled_latent)
    sampled_shape = tuple(int(dim) for dim in sampled_latent.shape)

    return SparseStructureForwardProbe(
        token_shape=(batch_size, token_count, config.in_channels),
        input_projection_shape=projection_shape,
        block0_output_shape=block0_shape,
        completed_blocks=completed_blocks,
        stack_output_shape=stack_shape,
        output_projection_shape=output_shape,
        sampled_latent_shape=sampled_shape,
        sampled_latent=sampled_latent,
        loaded_tensor_names=tuple(name for name in forward_tensor_names if name in tensors),
        inspected_tensor_names=tuple(info.name for info in infos),
        blocker_operation="sparse structure decoder handoff",
        blocker_detail=(
            f"input projection executed with output shape {projection_shape}; "
            f"block-0 ModulatedTransformerCrossBlock executed with output shape {block0_shape}; "
            f"all {completed_blocks} sparse flow transformer blocks executed with stack output shape {stack_shape}; "
            f"output projection executed with output shape {output_shape}; "
            f"FlowEuler sampler executed {steps} steps with sampled latent shape {sampled_shape}; "
            "next boundary is sparse structure decoder config/checkpoint execution"
        ),
    )


def probe_sparse_structure_decoder_boundary(
    checkpoint_path: str | Path,
    config: SparseStructureDecoderConfig,
    *,
    sparse_latent: mx.array | None = None,
    target_resolution: int | None = None,
) -> SparseStructureDecoderProbe:
    """Validate sparse decoder checkpoint tensors and report the next decoder boundary."""

    checkpoint = Path(checkpoint_path)
    tensor_names = _sparse_structure_decoder_tensor_names(config)
    tensors = load_checkpoint_tensors(checkpoint, names=tensor_names)
    infos = inspect_checkpoint(checkpoint, names=tensor_names)
    _require_checkpoint_infos(infos, tensor_names)
    _validate_sparse_structure_decoder_infos(infos, config)
    _validate_checkpoint_info_shape(
        infos,
        "input_layer.weight",
        (config.channels[0], config.latent_channels, 3, 3, 3),
    )
    _validate_checkpoint_info_shape(infos, "input_layer.bias", (config.channels[0],))
    _validate_checkpoint_info_shape(
        infos,
        "out_layer.2.weight",
        (config.out_channels, config.channels[-1], 3, 3, 3),
    )
    _validate_checkpoint_info_shape(infos, "out_layer.2.bias", (config.out_channels,))

    latent_shape = tuple(int(dim) for dim in sparse_latent.shape) if sparse_latent is not None else None
    if sparse_latent is None:
        return SparseStructureDecoderProbe(
            checkpoint_path=str(checkpoint),
            loaded_tensor_names=tuple(name for name in tensor_names if name in tensors),
            inspected_tensor_names=tuple(info.name for info in infos),
            latent_shape=None,
            decoded_shape=None,
            coordinates_shape=None,
            target_resolution=target_resolution,
            blocker_operation="sparse structure decoder upstream latent availability",
            blocker_detail="sparse structure decoder is mapped, but no sampled sparse latent is available from the sparse flow stage",
        )

    _validate_decoder_latent_shape(sparse_latent.shape, config)
    decoded = _decode_sparse_structure_latent(sparse_latent, config, tensors)
    mx.eval(decoded)
    decoded_shape = tuple(int(dim) for dim in decoded.shape)
    effective_target_resolution = target_resolution
    if effective_target_resolution is not None and effective_target_resolution > decoded_shape[2]:
        effective_target_resolution = decoded_shape[2]
    coords = extract_sparse_structure_coordinates(decoded, target_resolution=effective_target_resolution)
    coordinates_shape = tuple(int(dim) for dim in coords.coordinates.shape)
    return SparseStructureDecoderProbe(
        checkpoint_path=str(checkpoint),
        loaded_tensor_names=tuple(name for name in tensor_names if name in tensors),
        inspected_tensor_names=tuple(info.name for info in infos),
        latent_shape=latent_shape,
        decoded_shape=decoded_shape,
        coordinates_shape=coordinates_shape,
        target_resolution=coords.target_resolution,
        blocker_operation="sparse structure decoder coordinate extraction",
        blocker_detail=(
            f"sparse latent shape {latent_shape} decoded to logits shape {decoded_shape}; "
            f"target_resolution={coords.target_resolution} thresholding produced sparse coordinate shape {coordinates_shape}; "
            "next boundary is shape SLat sampling from sparse coordinates"
        ),
        decoded_logits=decoded,
        coordinates=coords.coordinates,
    )


def _sparse_structure_model_forward(
    sample: mx.array,
    t: float,
    conditioning: mx.array | dict[str, mx.array] | tuple[mx.array, mx.array],
    config: SparseStructureFlowConfig,
    tensors: dict[str, mx.array],
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    batch_size = int(sample.shape[0])
    tokens = mx.transpose(mx.reshape(sample, (batch_size, config.in_channels, -1)), (0, 2, 1))
    projected = _linear(
        tokens,
        tensors["input_layer.weight"].astype(mx.float32),
        tensors["input_layer.bias"].astype(mx.float32),
    )
    mod = _shared_sparse_structure_modulation(batch_size, config, tensors, t=t)
    hidden_states = projected
    block0 = None
    for block_index in range(config.num_blocks):
        hidden_states = _sparse_structure_block_forward(
            hidden_states,
            conditioning,
            config,
            tensors,
            mod=mod,
            block_index=block_index,
        )
        if block_index == 0:
            block0 = hidden_states
    if block0 is None:
        raise ValueError("sparse flow config must contain at least one transformer block")
    stack = _layer_norm_no_affine(hidden_states, eps=1e-6)
    output = _linear(
        stack,
        tensors["out_layer.weight"].astype(mx.float32),
        tensors["out_layer.bias"].astype(mx.float32),
    )
    return projected, block0, stack, output


def _flow_euler_sample_sparse_structure(
    sample: mx.array,
    conditioning: mx.array | dict[str, mx.array] | tuple[mx.array, mx.array],
    config: SparseStructureFlowConfig,
    tensors: dict[str, mx.array],
    *,
    steps: int,
    rescale_t: float,
    guidance_strength: float,
    guidance_rescale: float,
    guidance_interval: tuple[float, float],
    sigma_min: float,
) -> mx.array:
    schedule = flow_euler_schedule(steps=steps, rescale_t=rescale_t, guidance_interval=guidance_interval)
    neg_conditioning = _conditioning_zeros_like(conditioning)
    for t, t_prev in schedule.pairs:
        pred_v = _flow_euler_guided_prediction(
            sample,
            t,
            conditioning,
            neg_conditioning,
            config,
            tensors,
            guidance_strength=guidance_strength if guidance_interval[0] <= t <= guidance_interval[1] else 1.0,
            guidance_rescale=guidance_rescale,
            sigma_min=sigma_min,
        )
        sample = sample - (t - t_prev) * pred_v
        mx.eval(sample)
    return sample


def _flow_euler_guided_prediction(
    sample: mx.array,
    t: float,
    conditioning: mx.array | dict[str, mx.array] | tuple[mx.array, mx.array],
    neg_conditioning: mx.array | dict[str, mx.array] | tuple[mx.array, mx.array],
    config: SparseStructureFlowConfig,
    tensors: dict[str, mx.array],
    *,
    guidance_strength: float,
    guidance_rescale: float,
    sigma_min: float,
) -> mx.array:
    if guidance_strength == 1:
        return _flow_euler_model_prediction(sample, t, conditioning, config, tensors)
    if guidance_strength == 0:
        return _flow_euler_model_prediction(sample, t, neg_conditioning, config, tensors)

    pred_pos = _flow_euler_model_prediction(sample, t, conditioning, config, tensors)
    pred_neg = _flow_euler_model_prediction(sample, t, neg_conditioning, config, tensors)
    pred = guidance_strength * pred_pos + (1.0 - guidance_strength) * pred_neg
    if guidance_rescale <= 0:
        return pred

    x0_pos = _flow_euler_pred_to_xstart(sample, t, pred_pos, sigma_min=sigma_min)
    x0_cfg = _flow_euler_pred_to_xstart(sample, t, pred, sigma_min=sigma_min)
    axes = tuple(range(1, sample.ndim))
    std_pos = mx.std(x0_pos, axis=axes, keepdims=True)
    std_cfg = mx.maximum(mx.std(x0_cfg, axis=axes, keepdims=True), mx.array(1e-12, dtype=mx.float32))
    x0_rescaled = x0_cfg * (std_pos / std_cfg)
    x0 = guidance_rescale * x0_rescaled + (1.0 - guidance_rescale) * x0_cfg
    return _flow_euler_xstart_to_pred(sample, t, x0, sigma_min=sigma_min)


def _flow_euler_model_prediction(
    sample: mx.array,
    t: float,
    conditioning: mx.array | dict[str, mx.array] | tuple[mx.array, mx.array],
    config: SparseStructureFlowConfig,
    tensors: dict[str, mx.array],
) -> mx.array:
    _, _, _, output = _sparse_structure_model_forward(sample, t, conditioning, config, tensors)
    batch_size = int(sample.shape[0])
    return mx.reshape(mx.transpose(output, (0, 2, 1)), (batch_size, config.out_channels, config.resolution, config.resolution, config.resolution))


def _flow_euler_pred_to_xstart(sample: mx.array, t: float, pred: mx.array, *, sigma_min: float) -> mx.array:
    return (1.0 - sigma_min) * sample - (sigma_min + (1.0 - sigma_min) * t) * pred


def _flow_euler_xstart_to_pred(sample: mx.array, t: float, x0: mx.array, *, sigma_min: float) -> mx.array:
    return ((1.0 - sigma_min) * sample - x0) / (sigma_min + (1.0 - sigma_min) * t)


def _decode_sparse_structure_latent(
    sparse_latent: mx.array,
    config: SparseStructureDecoderConfig,
    tensors: dict[str, mx.array],
) -> mx.array:
    h = _conv3d_ncdhw(
        sparse_latent.astype(mx.float32),
        tensors["input_layer.weight"],
        tensors["input_layer.bias"],
    )
    torso_dtype = mx.float16 if config.use_fp16 else mx.float32
    h = h.astype(torso_dtype)

    for index in range(config.num_res_blocks_middle):
        h = _decoder_resblock3d(h, f"middle_block.{index}", tensors)

    block_index = 0
    for level, channels in enumerate(config.channels):
        for _ in range(config.num_res_blocks):
            h = _decoder_resblock3d(h, f"blocks.{block_index}", tensors)
            block_index += 1
        if level < len(config.channels) - 1:
            h = _decoder_upsample3d(h, f"blocks.{block_index}", tensors)
            block_index += 1

    h = h.astype(mx.float32)
    h = _channel_layer_norm_ncdhw(
        h,
        tensors["out_layer.0.weight"].astype(mx.float32),
        tensors["out_layer.0.bias"].astype(mx.float32),
        eps=1e-5,
    )
    h = _silu(h)
    return _conv3d_ncdhw(
        h,
        tensors["out_layer.2.weight"].astype(mx.float32),
        tensors["out_layer.2.bias"].astype(mx.float32),
    )


def _decoder_resblock3d(x: mx.array, prefix: str, tensors: dict[str, mx.array]) -> mx.array:
    h = _channel_layer_norm_ncdhw(
        x,
        tensors[f"{prefix}.norm1.weight"].astype(mx.float32),
        tensors[f"{prefix}.norm1.bias"].astype(mx.float32),
        eps=1e-5,
    )
    h = _silu(h)
    h = _conv3d_ncdhw(h, tensors[f"{prefix}.conv1.weight"], tensors[f"{prefix}.conv1.bias"])
    h = _channel_layer_norm_ncdhw(
        h,
        tensors[f"{prefix}.norm2.weight"].astype(mx.float32),
        tensors[f"{prefix}.norm2.bias"].astype(mx.float32),
        eps=1e-5,
    )
    h = _silu(h)
    h = _conv3d_ncdhw(h, tensors[f"{prefix}.conv2.weight"], tensors[f"{prefix}.conv2.bias"])
    return h.astype(x.dtype) + x


def _decoder_upsample3d(x: mx.array, prefix: str, tensors: dict[str, mx.array]) -> mx.array:
    h = _conv3d_ncdhw(x, tensors[f"{prefix}.conv.weight"], tensors[f"{prefix}.conv.bias"])
    return _pixel_shuffle_3d_ncdhw(h, upscale_factor=2)


def _conv3d_ncdhw(values: mx.array, weight: mx.array, bias: mx.array | None, *, padding: int = 1) -> mx.array:
    values_ndhwc = mx.transpose(values, (0, 2, 3, 4, 1))
    weight_ndhwc = mx.transpose(weight.astype(values.dtype), (0, 2, 3, 4, 1))
    output = mx.conv3d(values_ndhwc, weight_ndhwc, padding=padding)
    if bias is not None:
        output = output + bias.astype(output.dtype)[None, None, None, None, :]
    return mx.transpose(output, (0, 4, 1, 2, 3))


def _channel_layer_norm_ncdhw(values: mx.array, weight: mx.array, bias: mx.array, *, eps: float) -> mx.array:
    values_ndhwc = mx.transpose(values.astype(mx.float32), (0, 2, 3, 4, 1))
    mean = mx.mean(values_ndhwc, axis=-1, keepdims=True)
    variance = mx.var(values_ndhwc, axis=-1, keepdims=True)
    output = ((values_ndhwc - mean) * mx.rsqrt(variance + eps)) * weight + bias
    return mx.transpose(output, (0, 4, 1, 2, 3)).astype(values.dtype)


def _pixel_shuffle_3d_ncdhw(values: mx.array, *, upscale_factor: int) -> mx.array:
    batch, channels, depth, height, width = tuple(int(dim) for dim in values.shape)
    factor = upscale_factor
    if channels % (factor**3):
        raise ValueError(f"3D pixel shuffle channel count {channels} is not divisible by upscale_factor^3={factor**3}")
    out_channels = channels // (factor**3)
    values = mx.reshape(values, (batch, out_channels, factor, factor, factor, depth, height, width))
    values = mx.transpose(values, (0, 1, 5, 2, 6, 3, 7, 4))
    return mx.reshape(values, (batch, out_channels, depth * factor, height * factor, width * factor))


def _sparse_structure_block_forward(
    hidden_states: mx.array,
    conditioning: mx.array | dict[str, mx.array] | tuple[mx.array, mx.array],
    config: SparseStructureFlowConfig,
    tensors: dict[str, mx.array],
    *,
    mod: mx.array,
    block_index: int,
) -> mx.array:
    hidden_states = hidden_states.astype(mx.float32)
    conditioning = _cast_conditioning(conditioning, mx.float32)
    prefix = f"blocks.{block_index}"
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _split_modulation(
        tensors[f"{prefix}.modulation"].astype(mx.float32)[None, :] + mod
    )

    normalized = _layer_norm_no_affine(hidden_states, eps=1e-6)
    normalized = normalized * (1.0 + scale_msa[:, None, :]) + shift_msa[:, None, :]
    self_attn = _sparse_structure_self_attention(normalized, config, tensors, block_index=block_index)
    hidden_states = hidden_states + (self_attn * gate_msa[:, None, :])

    normalized = _layer_norm(
        hidden_states,
        tensors[f"{prefix}.norm2.weight"].astype(mx.float32),
        tensors[f"{prefix}.norm2.bias"].astype(mx.float32),
        eps=1e-6,
    )
    hidden_states = hidden_states + _sparse_structure_cross_attention(
        normalized,
        conditioning,
        config,
        tensors,
        block_index=block_index,
    )

    normalized = _layer_norm_no_affine(hidden_states, eps=1e-6)
    normalized = normalized * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
    mlp = _sparse_structure_mlp(normalized, tensors, block_index=block_index)
    return hidden_states + (mlp * gate_mlp[:, None, :])


def _shared_sparse_structure_modulation(
    batch_size: int,
    config: SparseStructureFlowConfig,
    tensors: dict[str, mx.array],
    *,
    t: float,
) -> mx.array:
    timestep = mx.full((batch_size,), 1000.0 * t, dtype=mx.float32)
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
        raise ValueError("non-shared sparse structure block modulation is not mapped yet")
    hidden = _silu(hidden)
    return _linear(
        hidden,
        tensors["adaLN_modulation.1.weight"].astype(mx.float32),
        tensors["adaLN_modulation.1.bias"].astype(mx.float32),
    )


def _sparse_structure_self_attention(
    hidden_states: mx.array,
    config: SparseStructureFlowConfig,
    tensors: dict[str, mx.array],
    *,
    block_index: int,
) -> mx.array:
    batch, token_count, _ = tuple(int(dim) for dim in hidden_states.shape)
    head_dim = _head_dim(config)
    prefix = f"blocks.{block_index}.self_attn"
    qkv = _linear(
        hidden_states,
        tensors[f"{prefix}.to_qkv.weight"].astype(mx.float32),
        tensors[f"{prefix}.to_qkv.bias"].astype(mx.float32),
    )
    qkv = mx.reshape(qkv, (batch, token_count, 3, config.num_heads, head_dim))
    query = qkv[:, :, 0, :, :]
    key = qkv[:, :, 1, :, :]
    value = qkv[:, :, 2, :, :]
    if config.qk_rms_norm:
        query = _multi_head_rms_norm(query, tensors[f"{prefix}.q_rms_norm.gamma"].astype(mx.float32))
        key = _multi_head_rms_norm(key, tensors[f"{prefix}.k_rms_norm.gamma"].astype(mx.float32))
    if config.pe_mode == "rope":
        cos, sin = _sparse_structure_rope_cos_sin(config)
        query = _apply_pairwise_rope(query, cos, sin)
        key = _apply_pairwise_rope(key, cos, sin)

    attended = _attention(query, key, value, head_dim=head_dim)
    attended = mx.reshape(attended, (batch, token_count, config.model_channels))
    return _linear(
        attended,
        tensors[f"{prefix}.to_out.weight"].astype(mx.float32),
        tensors[f"{prefix}.to_out.bias"].astype(mx.float32),
    )


def _sparse_structure_cross_attention(
    hidden_states: mx.array,
    conditioning: mx.array | dict[str, mx.array] | tuple[mx.array, mx.array],
    config: SparseStructureFlowConfig,
    tensors: dict[str, mx.array],
    *,
    block_index: int,
) -> mx.array:
    if config.image_attn_mode == "proj":
        return _sparse_structure_project_attention(hidden_states, conditioning, config, tensors, block_index=block_index)
    if config.image_attn_mode != "cross":
        raise ValueError(f"unsupported sparse structure image_attn_mode: {config.image_attn_mode!r}")

    batch, token_count, _ = tuple(int(dim) for dim in hidden_states.shape)
    cond_count = int(conditioning.shape[1])
    head_dim = _head_dim(config)
    prefix = f"blocks.{block_index}.cross_attn"
    return _sparse_structure_cross_attention_with_prefix(
        hidden_states,
        conditioning,
        config,
        tensors,
        prefix=prefix,
        cond_count=cond_count,
        head_dim=head_dim,
    )


def _sparse_structure_project_attention(
    hidden_states: mx.array,
    conditioning: mx.array | dict[str, mx.array] | tuple[mx.array, mx.array],
    config: SparseStructureFlowConfig,
    tensors: dict[str, mx.array],
    *,
    block_index: int,
) -> mx.array:
    global_context, proj_context = _split_projection_conditioning(conditioning)
    batch, token_count, _ = tuple(int(dim) for dim in hidden_states.shape)
    _validate_projection_context_shape(global_context.shape, proj_context.shape, config, batch_size=batch, token_count=token_count)
    head_dim = _head_dim(config)
    prefix = f"blocks.{block_index}.cross_attn"
    global_out = _sparse_structure_cross_attention_with_prefix(
        hidden_states,
        global_context,
        config,
        tensors,
        prefix=f"{prefix}.cross_attn_block",
        cond_count=int(global_context.shape[1]),
        head_dim=head_dim,
    )
    proj_out = _linear(
        proj_context,
        tensors[f"{prefix}.proj_linear.weight"].astype(mx.float32),
        tensors[f"{prefix}.proj_linear.bias"].astype(mx.float32),
    )
    return global_out + proj_out


def _sparse_structure_cross_attention_with_prefix(
    hidden_states: mx.array,
    conditioning: mx.array,
    config: SparseStructureFlowConfig,
    tensors: dict[str, mx.array],
    *,
    prefix: str,
    cond_count: int,
    head_dim: int,
) -> mx.array:
    batch, token_count, _ = tuple(int(dim) for dim in hidden_states.shape)
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
    query = mx.reshape(query, (batch, token_count, config.num_heads, head_dim))
    key_value = mx.reshape(key_value, (batch, cond_count, 2, config.num_heads, head_dim))
    key = key_value[:, :, 0, :, :]
    value = key_value[:, :, 1, :, :]
    if config.qk_rms_norm_cross:
        query = _multi_head_rms_norm(query, tensors[f"{prefix}.q_rms_norm.gamma"].astype(mx.float32))
        key = _multi_head_rms_norm(key, tensors[f"{prefix}.k_rms_norm.gamma"].astype(mx.float32))

    attended = _attention(query, key, value, head_dim=head_dim)
    attended = mx.reshape(attended, (batch, token_count, config.model_channels))
    return _linear(
        attended,
        tensors[f"{prefix}.to_out.weight"].astype(mx.float32),
        tensors[f"{prefix}.to_out.bias"].astype(mx.float32),
    )


def _sparse_structure_mlp(hidden_states: mx.array, tensors: dict[str, mx.array], *, block_index: int) -> mx.array:
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


def _attention(query: mx.array, key: mx.array, value: mx.array, *, head_dim: int) -> mx.array:
    query = mx.transpose(query, (0, 2, 1, 3))
    key = mx.transpose(key, (0, 2, 1, 3))
    value = mx.transpose(value, (0, 2, 1, 3))
    attended = mx.fast.scaled_dot_product_attention(query, key, value, scale=head_dim**-0.5)
    return mx.transpose(attended, (0, 2, 1, 3))


def _sparse_structure_rope_cos_sin(config: SparseStructureFlowConfig) -> tuple[mx.array, mx.array]:
    head_dim = _head_dim(config)
    if head_dim % 2:
        raise ValueError(f"sparse 3D RoPE requires even head_dim, got {head_dim}")
    freq_dim = head_dim // 2 // 3
    if freq_dim <= 0:
        raise ValueError(f"sparse 3D RoPE requires head_dim large enough for 3 axes, got {head_dim}")
    coords = mx.arange(config.resolution, dtype=mx.float32)
    grid_z, grid_y, grid_x = mx.meshgrid(coords, coords, coords, indexing="ij")
    positions = mx.reshape(mx.stack((grid_z, grid_y, grid_x), axis=-1), (-1, 3))
    freq_index = mx.arange(freq_dim, dtype=mx.float32) / float(freq_dim)
    freqs = 1.0 / (10000.0 ** freq_index)
    angles = mx.reshape(positions[:, :, None] * freqs[None, None, :], (config.resolution**3, -1))
    pair_count = head_dim // 2
    if int(angles.shape[-1]) < pair_count:
        padding = mx.zeros((int(angles.shape[0]), pair_count - int(angles.shape[-1])), dtype=angles.dtype)
        angles = mx.concatenate((angles, padding), axis=-1)
    return mx.cos(angles), mx.sin(angles)


def _apply_pairwise_rope(values: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    batch, token_count, heads, head_dim = tuple(int(dim) for dim in values.shape)
    pairs = mx.reshape(values.astype(mx.float32), (batch, token_count, heads, head_dim // 2, 2))
    first = pairs[..., 0]
    second = pairs[..., 1]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rotated = mx.stack((first * cos - second * sin, first * sin + second * cos), axis=-1)
    return mx.reshape(rotated, (batch, token_count, heads, head_dim))


def _multi_head_rms_norm(values: mx.array, gamma: mx.array) -> mx.array:
    norm = mx.sqrt(mx.sum(values.astype(mx.float32) * values.astype(mx.float32), axis=-1, keepdims=True))
    return (values.astype(mx.float32) / mx.maximum(norm, mx.array(1e-12, dtype=mx.float32))) * gamma[None, None, :, :] * (
        int(values.shape[-1]) ** 0.5
    )


def _split_modulation(values: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    chunks = mx.split(values, 6, axis=-1)
    return chunks[0], chunks[1], chunks[2], chunks[3], chunks[4], chunks[5]


def _timestep_embedding(timestep: mx.array, dim: int, *, max_period: int = 10000) -> mx.array:
    half = dim // 2
    freqs = mx.exp(-np.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half)
    args = timestep[:, None].astype(mx.float32) * freqs[None, :]
    embedding = mx.concatenate((mx.cos(args), mx.sin(args)), axis=-1)
    if dim % 2:
        embedding = mx.concatenate((embedding, mx.zeros((int(timestep.shape[0]), 1), dtype=embedding.dtype)), axis=-1)
    return embedding


def _layer_norm_no_affine(values: mx.array, *, eps: float) -> mx.array:
    mean = mx.mean(values.astype(mx.float32), axis=-1, keepdims=True)
    variance = mx.var(values.astype(mx.float32), axis=-1, keepdims=True)
    return (values.astype(mx.float32) - mean) * mx.rsqrt(variance + eps)


def _layer_norm(values: mx.array, weight: mx.array, bias: mx.array, *, eps: float) -> mx.array:
    return _layer_norm_no_affine(values, eps=eps) * weight + bias


def _linear(values: mx.array, weight: mx.array, bias: mx.array | None) -> mx.array:
    output = values @ mx.transpose(weight)
    if bias is not None:
        output = output + bias
    return output


def _silu(values: mx.array) -> mx.array:
    return values * mx.sigmoid(values)


def _gelu_tanh(values: mx.array) -> mx.array:
    return 0.5 * values * (1.0 + mx.tanh(0.7978845608028654 * (values + 0.044715 * values * values * values)))


def _head_dim(config: SparseStructureFlowConfig) -> int:
    if config.model_channels % config.num_heads:
        raise ValueError(
            f"sparse attention head dimension mismatch: model_channels={config.model_channels}, num_heads={config.num_heads}"
        )
    return config.model_channels // config.num_heads


def _validate_conditioning_shape(
    shape: tuple[int, ...],
    config: SparseStructureFlowConfig,
    *,
    batch_size: int,
) -> None:
    actual = tuple(int(dim) for dim in shape)
    if len(actual) != 3:
        raise ValueError(f"sparse flow conditioning must have shape (batch, tokens, channels), got {actual}")
    if actual[0] != batch_size:
        raise ValueError(f"sparse flow conditioning batch mismatch: expected {batch_size}, got {actual[0]}")
    if actual[1] <= 0:
        raise ValueError("sparse flow conditioning must contain at least one token")
    if actual[2] != config.cond_channels:
        raise ValueError(f"sparse flow conditioning width mismatch: expected {config.cond_channels}, got {actual[2]}")


def _validate_conditioning(
    conditioning: mx.array | dict[str, mx.array] | tuple[mx.array, mx.array],
    config: SparseStructureFlowConfig,
    *,
    batch_size: int,
    token_count: int,
) -> None:
    if config.image_attn_mode == "cross":
        _validate_conditioning_shape(conditioning.shape, config, batch_size=batch_size)
        return
    if config.image_attn_mode == "proj":
        global_context, proj_context = _split_projection_conditioning(conditioning)
        _validate_projection_context_shape(global_context.shape, proj_context.shape, config, batch_size=batch_size, token_count=token_count)
        return
    raise ValueError(f"unsupported sparse structure image_attn_mode: {config.image_attn_mode!r}")


def _split_projection_conditioning(
    conditioning: mx.array | dict[str, mx.array] | tuple[mx.array, mx.array],
) -> tuple[mx.array, mx.array]:
    if isinstance(conditioning, dict):
        return conditioning["global"], conditioning["proj"]
    if isinstance(conditioning, tuple) and len(conditioning) == 2:
        return conditioning
    raise ValueError("projection conditioning must be a dict with 'global'/'proj' or a (global, proj) tuple")


def _validate_projection_context_shape(
    global_shape: tuple[int, ...],
    proj_shape: tuple[int, ...],
    config: SparseStructureFlowConfig,
    *,
    batch_size: int,
    token_count: int,
) -> None:
    _validate_conditioning_shape(global_shape, config, batch_size=batch_size)
    expected_proj_channels = config.proj_in_channels or config.cond_channels
    actual = tuple(int(dim) for dim in proj_shape)
    expected = (batch_size, token_count, expected_proj_channels)
    if actual != expected:
        raise ValueError(f"sparse projection context shape mismatch: expected {expected}, got {actual}")


def _cast_conditioning(
    conditioning: mx.array | dict[str, mx.array] | tuple[mx.array, mx.array],
    dtype: mx.Dtype,
) -> mx.array | dict[str, mx.array] | tuple[mx.array, mx.array]:
    if isinstance(conditioning, dict):
        return {key: value.astype(dtype) for key, value in conditioning.items()}
    if isinstance(conditioning, tuple):
        return tuple(value.astype(dtype) for value in conditioning)
    return conditioning.astype(dtype)


def _conditioning_zeros_like(
    conditioning: mx.array | dict[str, mx.array] | tuple[mx.array, mx.array],
) -> mx.array | dict[str, mx.array] | tuple[mx.array, mx.array]:
    if isinstance(conditioning, dict):
        return {key: mx.zeros_like(value) for key, value in conditioning.items()}
    if isinstance(conditioning, tuple):
        return tuple(mx.zeros_like(value) for value in conditioning)
    return mx.zeros_like(conditioning)


def extract_sparse_structure_coordinates(
    decoded_logits: mx.array,
    *,
    target_resolution: int | None = None,
    threshold: float = 0.0,
) -> SparseStructureCoordinates:
    """Apply TRELLIS.2 sparse decoder thresholding and return `(batch, z, y, x)` coordinates."""

    decoded_shape = tuple(int(dim) for dim in decoded_logits.shape)
    if len(decoded_shape) != 5:
        raise ValueError(f"sparse decoder logits must have shape (batch, channels, depth, height, width), got {decoded_shape}")
    batch, channels, depth, height, width = decoded_shape
    if batch <= 0 or channels != 1 or depth <= 0 or height <= 0 or width <= 0:
        raise ValueError("sparse decoder logits must have positive batch/spatial dims and exactly one output channel")
    if depth != height or height != width:
        raise ValueError(f"sparse decoder logits must use cubic spatial dims, got {(depth, height, width)}")

    resolution = target_resolution or depth
    if resolution <= 0:
        raise ValueError("target_resolution must be positive")
    mask = np.array(decoded_logits > threshold)
    if resolution != depth:
        if depth % resolution != 0:
            raise ValueError(f"target_resolution {resolution} must divide decoded resolution {depth}")
        ratio = depth // resolution
        mask = mask.reshape(batch, channels, resolution, ratio, resolution, ratio, resolution, ratio)
        mask = mask.max(axis=(3, 5, 7))

    argwhere = np.argwhere(mask)
    coords = argwhere[:, [0, 2, 3, 4]].astype(np.int32) if argwhere.size else np.zeros((0, 4), dtype=np.int32)
    coordinates = mx.array(coords, dtype=mx.int32)
    return SparseStructureCoordinates(
        decoded_shape=decoded_shape,
        coordinate_shape=tuple(int(dim) for dim in coordinates.shape),
        target_resolution=resolution,
        coordinates=coordinates,
    )


def _require_checkpoint_infos(
    infos: tuple[CheckpointTensorInfo, ...],
    expected_names: tuple[str, ...],
) -> None:
    present = {info.name for info in infos}
    missing = sorted(set(expected_names).difference(present))
    if missing:
        raise ValueError(f"checkpoint is missing requested tensors: {missing}")


def _sparse_structure_forward_tensor_names(config: SparseStructureFlowConfig) -> tuple[str, ...]:
    return (
        *SPARSE_STRUCTURE_INPUT_TENSOR_NAMES,
        "out_layer.weight",
        "out_layer.bias",
        *_sparse_structure_stack_inspection_names(config),
    )


def _sparse_structure_stack_inspection_names(config: SparseStructureFlowConfig) -> tuple[str, ...]:
    names = [
        "t_embedder.mlp.0.weight",
        "t_embedder.mlp.0.bias",
        "t_embedder.mlp.2.weight",
        "t_embedder.mlp.2.bias",
        "adaLN_modulation.1.weight",
        "adaLN_modulation.1.bias",
    ]
    for block_index in range(config.num_blocks):
        names.extend(_sparse_structure_block_tensor_names(block_index, config))
    return tuple(names)


def _sparse_structure_block_tensor_names(block_index: int, config: SparseStructureFlowConfig) -> tuple[str, ...]:
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
        *_sparse_structure_cross_attention_tensor_names(block_index, config),
        f"{prefix}.mlp.mlp.0.weight",
        f"{prefix}.mlp.mlp.0.bias",
        f"{prefix}.mlp.mlp.2.weight",
        f"{prefix}.mlp.mlp.2.bias",
    )


def _sparse_structure_cross_attention_tensor_names(block_index: int, config: SparseStructureFlowConfig) -> tuple[str, ...]:
    prefix = f"blocks.{block_index}.cross_attn"
    if config.image_attn_mode == "cross":
        cross_prefix = prefix
        extra: tuple[str, ...] = ()
    elif config.image_attn_mode == "proj":
        cross_prefix = f"{prefix}.cross_attn_block"
        extra = (f"{prefix}.proj_linear.weight", f"{prefix}.proj_linear.bias")
    else:
        raise ValueError(f"unsupported sparse structure image_attn_mode: {config.image_attn_mode!r}")
    return (
        f"{cross_prefix}.to_q.weight",
        f"{cross_prefix}.to_q.bias",
        f"{cross_prefix}.to_kv.weight",
        f"{cross_prefix}.to_kv.bias",
        f"{cross_prefix}.q_rms_norm.gamma",
        f"{cross_prefix}.k_rms_norm.gamma",
        f"{cross_prefix}.to_out.weight",
        f"{cross_prefix}.to_out.bias",
        *extra,
    )


def _validate_sparse_structure_stack_infos(
    infos: tuple[CheckpointTensorInfo, ...],
    config: SparseStructureFlowConfig,
) -> None:
    if config.share_mod:
        _validate_checkpoint_info_shape(
            infos,
            "adaLN_modulation.1.weight",
            (config.model_channels * 6, config.model_channels),
        )
    _validate_checkpoint_info_shape(infos, "adaLN_modulation.1.bias", (config.model_channels * 6,))
    intermediate_channels = int(config.model_channels * config.mlp_ratio)
    head_dim = _head_dim(config)
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
            f"{_sparse_structure_cross_attention_prefix(block_index, config)}.to_q.weight",
            (config.model_channels, config.model_channels),
        )
        _validate_checkpoint_info_shape(
            infos,
            f"{_sparse_structure_cross_attention_prefix(block_index, config)}.to_q.bias",
            (config.model_channels,),
        )
        _validate_checkpoint_info_shape(
            infos,
            f"{_sparse_structure_cross_attention_prefix(block_index, config)}.to_kv.weight",
            (config.model_channels * 2, config.cond_channels),
        )
        _validate_checkpoint_info_shape(
            infos,
            f"{_sparse_structure_cross_attention_prefix(block_index, config)}.to_kv.bias",
            (config.model_channels * 2,),
        )
        _validate_checkpoint_info_shape(
            infos,
            f"{_sparse_structure_cross_attention_prefix(block_index, config)}.q_rms_norm.gamma",
            (config.num_heads, head_dim),
        )
        _validate_checkpoint_info_shape(
            infos,
            f"{_sparse_structure_cross_attention_prefix(block_index, config)}.k_rms_norm.gamma",
            (config.num_heads, head_dim),
        )
        _validate_checkpoint_info_shape(
            infos,
            f"{_sparse_structure_cross_attention_prefix(block_index, config)}.to_out.weight",
            (config.model_channels, config.model_channels),
        )
        _validate_checkpoint_info_shape(
            infos,
            f"{_sparse_structure_cross_attention_prefix(block_index, config)}.to_out.bias",
            (config.model_channels,),
        )
        if config.image_attn_mode == "proj":
            _validate_checkpoint_info_shape(
                infos,
                f"{prefix}.cross_attn.proj_linear.weight",
                (config.model_channels, config.proj_in_channels or config.cond_channels),
            )
            _validate_checkpoint_info_shape(infos, f"{prefix}.cross_attn.proj_linear.bias", (config.model_channels,))
        _validate_checkpoint_info_shape(
            infos,
            f"{prefix}.mlp.mlp.0.weight",
            (intermediate_channels, config.model_channels),
        )
        _validate_checkpoint_info_shape(infos, f"{prefix}.mlp.mlp.0.bias", (intermediate_channels,))
        _validate_checkpoint_info_shape(
            infos,
            f"{prefix}.mlp.mlp.2.weight",
            (config.model_channels, intermediate_channels),
        )
        _validate_checkpoint_info_shape(infos, f"{prefix}.mlp.mlp.2.bias", (config.model_channels,))


def _sparse_structure_cross_attention_prefix(block_index: int, config: SparseStructureFlowConfig) -> str:
    prefix = f"blocks.{block_index}.cross_attn"
    if config.image_attn_mode == "cross":
        return prefix
    if config.image_attn_mode == "proj":
        return f"{prefix}.cross_attn_block"
    raise ValueError(f"unsupported sparse structure image_attn_mode: {config.image_attn_mode!r}")


def _sparse_structure_decoder_tensor_names(config: SparseStructureDecoderConfig) -> tuple[str, ...]:
    names = ["input_layer.weight", "input_layer.bias"]
    for index in range(config.num_res_blocks_middle):
        names.extend(_decoder_resblock_tensor_names(f"middle_block.{index}"))
    block_index = 0
    for level, _channels in enumerate(config.channels):
        for _ in range(config.num_res_blocks):
            names.extend(_decoder_resblock_tensor_names(f"blocks.{block_index}"))
            block_index += 1
        if level < len(config.channels) - 1:
            names.extend((f"blocks.{block_index}.conv.weight", f"blocks.{block_index}.conv.bias"))
            block_index += 1
    names.extend(("out_layer.0.weight", "out_layer.0.bias", "out_layer.2.weight", "out_layer.2.bias"))
    return tuple(names)


def _decoder_resblock_tensor_names(prefix: str) -> tuple[str, ...]:
    return (
        f"{prefix}.norm1.weight",
        f"{prefix}.norm1.bias",
        f"{prefix}.conv1.weight",
        f"{prefix}.conv1.bias",
        f"{prefix}.norm2.weight",
        f"{prefix}.norm2.bias",
        f"{prefix}.conv2.weight",
        f"{prefix}.conv2.bias",
    )


def _validate_sparse_structure_decoder_infos(
    infos: tuple[CheckpointTensorInfo, ...],
    config: SparseStructureDecoderConfig,
) -> None:
    _validate_checkpoint_info_shape(infos, "input_layer.weight", (config.channels[0], config.latent_channels, 3, 3, 3))
    _validate_checkpoint_info_shape(infos, "input_layer.bias", (config.channels[0],))
    for index in range(config.num_res_blocks_middle):
        _validate_decoder_resblock_infos(infos, f"middle_block.{index}", config.channels[0])
    block_index = 0
    for level, channels in enumerate(config.channels):
        for _ in range(config.num_res_blocks):
            _validate_decoder_resblock_infos(infos, f"blocks.{block_index}", channels)
            block_index += 1
        if level < len(config.channels) - 1:
            next_channels = config.channels[level + 1]
            _validate_checkpoint_info_shape(infos, f"blocks.{block_index}.conv.weight", (next_channels * 8, channels, 3, 3, 3))
            _validate_checkpoint_info_shape(infos, f"blocks.{block_index}.conv.bias", (next_channels * 8,))
            block_index += 1
    _validate_checkpoint_info_shape(infos, "out_layer.0.weight", (config.channels[-1],))
    _validate_checkpoint_info_shape(infos, "out_layer.0.bias", (config.channels[-1],))
    _validate_checkpoint_info_shape(infos, "out_layer.2.weight", (config.out_channels, config.channels[-1], 3, 3, 3))
    _validate_checkpoint_info_shape(infos, "out_layer.2.bias", (config.out_channels,))


def _validate_decoder_resblock_infos(
    infos: tuple[CheckpointTensorInfo, ...],
    prefix: str,
    channels: int,
) -> None:
    _validate_checkpoint_info_shape(infos, f"{prefix}.norm1.weight", (channels,))
    _validate_checkpoint_info_shape(infos, f"{prefix}.norm1.bias", (channels,))
    _validate_checkpoint_info_shape(infos, f"{prefix}.conv1.weight", (channels, channels, 3, 3, 3))
    _validate_checkpoint_info_shape(infos, f"{prefix}.conv1.bias", (channels,))
    _validate_checkpoint_info_shape(infos, f"{prefix}.norm2.weight", (channels,))
    _validate_checkpoint_info_shape(infos, f"{prefix}.norm2.bias", (channels,))
    _validate_checkpoint_info_shape(infos, f"{prefix}.conv2.weight", (channels, channels, 3, 3, 3))
    _validate_checkpoint_info_shape(infos, f"{prefix}.conv2.bias", (channels,))


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


def _validate_decoder_latent_shape(shape: tuple[int, ...], config: SparseStructureDecoderConfig) -> None:
    actual = tuple(int(dim) for dim in shape)
    if len(actual) != 5:
        raise ValueError(f"sparse decoder latent must have shape (batch, channels, depth, height, width), got {actual}")
    if actual[1] != config.latent_channels:
        raise ValueError(f"sparse decoder latent channel mismatch: expected {config.latent_channels}, got {actual[1]}")
    if actual[2] <= 0 or actual[2] != actual[3] or actual[3] != actual[4]:
        raise ValueError(f"sparse decoder latent must use positive cubic spatial dims, got {actual[2:]}")
