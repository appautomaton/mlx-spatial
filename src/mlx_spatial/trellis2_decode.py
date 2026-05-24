"""Latent decode contracts for TRELLIS.2 forward tracing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from .checkpoint import CheckpointTensorInfo, inspect_checkpoint, load_checkpoint_tensors
from .sparse_conv import sparse_conv_map_vectorized, weighted_sparse_conv_chunked
from .trellis2_sparse_structure import _layer_norm, _linear, _silu


STRUCTURED_LATENT_DECODER_BASE_TENSOR_NAMES = (
    "from_latent.weight",
    "from_latent.bias",
    "output_layer.weight",
    "output_layer.bias",
)

STRUCTURED_LATENT_ENCODER_BASE_TENSOR_NAMES = (
    "input_layer.weight",
    "input_layer.bias",
    "to_latent.weight",
    "to_latent.bias",
)

STRUCTURED_LATENT_DECODER_TENSOR_NAMES = STRUCTURED_LATENT_DECODER_BASE_TENSOR_NAMES + (
    "blocks.0.0.conv.weight",
    "blocks.0.0.conv.bias",
    "blocks.0.0.norm.weight",
    "blocks.0.0.norm.bias",
    "blocks.0.0.mlp.0.weight",
    "blocks.0.0.mlp.0.bias",
    "blocks.0.0.mlp.2.weight",
    "blocks.0.0.mlp.2.bias",
)

STRUCTURED_LATENT_DECODER_REFERENCE_TOKEN_LIMIT = 4096


@dataclass(frozen=True)
class StructuredLatentDecoderConfig:
    name: str
    latent_channels: int
    model_channels: tuple[int, ...]
    num_blocks: tuple[int, ...]
    block_type: tuple[str, ...]
    up_block_type: tuple[str, ...]
    use_fp16: bool
    out_channels: int
    resolution: int | None = None
    pred_subdiv: bool | None = None


@dataclass(frozen=True)
class StructuredLatentEncoderConfig:
    name: str
    in_channels: int
    latent_channels: int
    model_channels: tuple[int, ...]
    num_blocks: tuple[int, ...]
    block_type: tuple[str, ...]
    down_block_type: tuple[str, ...]
    use_fp16: bool


@dataclass(frozen=True)
class StructuredLatentEncoderProbe:
    checkpoint_path: str
    encoder_name: str
    coordinate_shape: tuple[int, int]
    vertex_shape: tuple[int, int]
    intersected_shape: tuple[int, int]
    input_feature_shape: tuple[int, int]
    input_projection_shape: tuple[int, int]
    completed_levels: int
    first_downblock_coordinate_shape: tuple[int, int] | None
    first_downblock_output_shape: tuple[int, int] | None
    latent_feature_shape: tuple[int, int]
    mean_shape: tuple[int, int]
    logvar_shape: tuple[int, int]
    loaded_tensor_names: tuple[str, ...]
    inspected_tensor_names: tuple[str, ...]


@dataclass(frozen=True)
class FlexiDualGridEncoderResult:
    coordinates: mx.array
    features: mx.array
    mean: mx.array
    logvar: mx.array
    probe: StructuredLatentEncoderProbe


@dataclass(frozen=True)
class _StructuredLatentEncoderExecution:
    coordinates: mx.array
    features: mx.array
    completed_levels: int
    first_downblock_coordinate_shape: tuple[int, int] | None
    first_downblock_output_shape: tuple[int, int] | None


@dataclass(frozen=True)
class FlexiDualGridVaeEncoder:
    """Callable MLX contract for the TRELLIS.2 mesh-to-shape-SLat encoder."""

    checkpoint_path: str | Path
    config: StructuredLatentEncoderConfig

    def __call__(
        self,
        coordinates: mx.array,
        dual_vertices: mx.array,
        intersected: mx.array,
        *,
        sample_posterior: bool = False,
    ) -> FlexiDualGridEncoderResult:
        return run_flexi_dual_grid_vae_encoder(
            self.checkpoint_path,
            self.config,
            coordinates,
            dual_vertices,
            intersected,
            sample_posterior=sample_posterior,
        )


@dataclass(frozen=True)
class StructuredLatentDecoderProbe:
    checkpoint_path: str
    latent_name: str
    coordinate_shape: tuple[int, int]
    feature_shape: tuple[int, int]
    input_projection_shape: tuple[int, int]
    convnext0_output_shape: tuple[int, int] | None
    level0_completed_blocks: int
    level0_output_shape: tuple[int, int] | None
    first_upblock_coordinate_shape: tuple[int, int] | None
    first_upblock_output_shape: tuple[int, int] | None
    first_upblock_subdivision_shape: tuple[int, int] | None
    completed_levels: int
    subdivision_shapes: tuple[tuple[int, int], ...]
    decoder_output_coordinate_shape: tuple[int, int] | None
    decoder_output_shape: tuple[int, int] | None
    reference_stop: str | None
    reference_token_limit: int
    loaded_tensor_names: tuple[str, ...]
    inspected_tensor_names: tuple[str, ...]


@dataclass(frozen=True)
class DecodeLatentsProbe:
    shape_probe: StructuredLatentDecoderProbe
    texture_probe: StructuredLatentDecoderProbe
    resolution: int
    blocker_operation: str
    blocker_detail: str


@dataclass(frozen=True)
class ShapeDecoderResult:
    coordinates: mx.array
    fields: mx.array
    subdivisions: tuple[mx.array, ...]
    probe: StructuredLatentDecoderProbe


@dataclass(frozen=True)
class TextureDecoderResult:
    coordinates: mx.array
    attributes: mx.array
    probe: StructuredLatentDecoderProbe
    guide_subdivision_shapes: tuple[tuple[int, int], ...]
    spatial_shape: tuple[int, int, int]
    batch_size: int
    decode_resolution: int | None
    voxel_size: float | None
    shape_decoder_coordinate_shape: tuple[int, int] | None


@dataclass(frozen=True)
class ShapeDecoderUpsampleResult:
    coordinates: mx.array
    subdivisions: tuple[mx.array, ...]
    input_coordinate_shape: tuple[int, int]
    output_coordinate_shape: tuple[int, int]
    completed_upsamples: int


@dataclass(frozen=True)
class _StructuredLatentDecoderExecution:
    convnext0_output_shape: tuple[int, int] | None
    level0_completed_blocks: int
    level0_output_shape: tuple[int, int] | None
    first_upblock_coordinate_shape: tuple[int, int] | None
    first_upblock_output_shape: tuple[int, int] | None
    first_upblock_subdivision_shape: tuple[int, int] | None
    completed_levels: int
    subdivisions: tuple[mx.array, ...]
    decoder_output_coordinates: mx.array | None
    decoder_output_features: mx.array | None
    decoder_output_coordinate_shape: tuple[int, int] | None
    decoder_output_shape: tuple[int, int] | None
    reference_stop: str | None


def read_structured_latent_decoder_config(root: str | Path, config_path: str) -> StructuredLatentDecoderConfig:
    path = Path(root) / config_path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        args = payload["args"]
        model_channels = tuple(int(value) for value in args["model_channels"])
        num_blocks = tuple(int(value) for value in args["num_blocks"])
        block_type = tuple(str(value) for value in args["block_type"])
        up_block_type = tuple(str(value) for value in args["up_block_type"])
        if not model_channels or len(model_channels) != len(num_blocks) or len(model_channels) != len(block_type):
            raise ValueError("model_channels, num_blocks, and block_type must be non-empty with equal length")
        if len(up_block_type) != len(model_channels) - 1:
            raise ValueError("up_block_type length must be one less than model_channels")
        name = str(payload["name"])
        out_channels = int(args.get("out_channels", 7 if name == "FlexiDualGridVaeDecoder" else 0))
        pred_subdiv = bool(args["pred_subdiv"]) if "pred_subdiv" in args else None
        if pred_subdiv is None and name == "FlexiDualGridVaeDecoder":
            pred_subdiv = True
        return StructuredLatentDecoderConfig(
            name=name,
            latent_channels=int(args["latent_channels"]),
            model_channels=model_channels,
            num_blocks=num_blocks,
            block_type=block_type,
            up_block_type=up_block_type,
            use_fp16=bool(args.get("use_fp16", False)),
            out_channels=out_channels,
            resolution=int(args["resolution"]) if "resolution" in args else None,
            pred_subdiv=pred_subdiv,
        )
    except FileNotFoundError:
        raise
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"structured latent decoder config is invalid: {error}") from error


def read_structured_latent_encoder_config(root: str | Path, config_path: str) -> StructuredLatentEncoderConfig:
    path = Path(root) / config_path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        args = payload["args"]
        name = str(payload["name"])
        model_channels = tuple(int(value) for value in args["model_channels"])
        num_blocks = tuple(int(value) for value in args["num_blocks"])
        block_type = tuple(str(value) for value in args["block_type"])
        down_block_type = tuple(str(value) for value in args["down_block_type"])
        if not model_channels or len(model_channels) != len(num_blocks) or len(model_channels) != len(block_type):
            raise ValueError("model_channels, num_blocks, and block_type must be non-empty with equal length")
        if len(down_block_type) != len(model_channels) - 1:
            raise ValueError("down_block_type length must be one less than model_channels")
        in_channels = int(args.get("in_channels", 6 if name == "FlexiDualGridVaeEncoder" else 0))
        if in_channels <= 0:
            raise ValueError("encoder in_channels must be positive")
        return StructuredLatentEncoderConfig(
            name=name,
            in_channels=in_channels,
            latent_channels=int(args["latent_channels"]),
            model_channels=model_channels,
            num_blocks=num_blocks,
            block_type=block_type,
            down_block_type=down_block_type,
            use_fp16=bool(args.get("use_fp16", False)),
        )
    except FileNotFoundError:
        raise
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"structured latent encoder config is invalid: {error}") from error


def probe_structured_latent_decoder_boundary(
    checkpoint_path: str | Path,
    config: StructuredLatentDecoderConfig,
    coordinates: mx.array,
    features: mx.array,
    *,
    latent_name: str,
    guide_subdivisions: tuple[mx.array, ...] | None = None,
    reference_token_limit: int = STRUCTURED_LATENT_DECODER_REFERENCE_TOKEN_LIMIT,
) -> StructuredLatentDecoderProbe:
    probe, _ = _probe_structured_latent_decoder_boundary(
        checkpoint_path,
        config,
        coordinates,
        features,
        latent_name=latent_name,
        guide_subdivisions=guide_subdivisions,
        reference_token_limit=reference_token_limit,
    )
    return probe


def _probe_structured_latent_decoder_boundary(
    checkpoint_path: str | Path,
    config: StructuredLatentDecoderConfig,
    coordinates: mx.array,
    features: mx.array,
    *,
    latent_name: str,
    guide_subdivisions: tuple[mx.array, ...] | None = None,
    reference_token_limit: int = STRUCTURED_LATENT_DECODER_REFERENCE_TOKEN_LIMIT,
) -> tuple[StructuredLatentDecoderProbe, tuple[mx.array, ...]]:
    reference_token_limit = _reference_token_limit(reference_token_limit)
    coordinate_shape = tuple(int(dim) for dim in coordinates.shape)
    feature_shape = tuple(int(dim) for dim in features.shape)
    _validate_latent_layout(coordinate_shape, feature_shape, config, latent_name)

    run_reference = feature_shape[0] <= reference_token_limit
    tensor_names = _structured_latent_decoder_tensor_names(
        config,
        max_level_count=len(config.num_blocks) if run_reference else 1,
    )
    tensors = load_checkpoint_tensors(checkpoint_path, names=tensor_names)
    from_latent_weight = tensors["from_latent.weight"]
    from_latent_bias = tensors["from_latent.bias"]
    _validate_tensor_shape("from_latent.weight", from_latent_weight.shape, (config.model_channels[0], config.latent_channels))
    _validate_tensor_shape("from_latent.bias", from_latent_bias.shape, (config.model_channels[0],))

    infos = inspect_checkpoint(checkpoint_path, names=tensor_names)
    _require_checkpoint_infos(infos, tensor_names)
    _validate_checkpoint_info_shape(infos, "output_layer.weight", (config.out_channels, config.model_channels[-1]))
    _validate_checkpoint_info_shape(infos, "output_layer.bias", (config.out_channels,))
    _validate_checkpoint_infos(infos, config, max_level_count=len(config.num_blocks) if run_reference else 1)

    projected = _linear_chunked(features.astype(from_latent_weight.dtype), from_latent_weight, from_latent_bias)
    mx.eval(projected)
    projection_shape = tuple(int(dim) for dim in projected.shape)
    expected_projection_shape = (feature_shape[0], config.model_channels[0])
    if projection_shape != expected_projection_shape:
        raise ValueError(
            f"{latent_name} decoder from_latent projection shape mismatch: expected {expected_projection_shape}, got {projection_shape}"
        )
    convnext0_shape = None
    level0_completed_blocks = 0
    level0_shape = None
    first_upblock_coordinate_shape = None
    first_upblock_shape = None
    first_upblock_subdivision_shape = None
    completed_levels = 0
    subdivision_shapes: tuple[tuple[int, int], ...] = ()
    decoder_output_coordinate_shape = None
    decoder_output_shape = None
    reference_stop = None
    subdivisions: tuple[mx.array, ...] = ()
    if run_reference:
        execution = _run_sparse_unet_decoder_reference(
            config,
            coordinates,
            projected,
            tensors,
            guide_subdivisions=guide_subdivisions,
            reference_token_limit=reference_token_limit,
        )
        convnext0_shape = execution.convnext0_output_shape
        level0_completed_blocks = execution.level0_completed_blocks
        level0_shape = execution.level0_output_shape
        first_upblock_coordinate_shape = execution.first_upblock_coordinate_shape
        first_upblock_shape = execution.first_upblock_output_shape
        first_upblock_subdivision_shape = execution.first_upblock_subdivision_shape
        completed_levels = execution.completed_levels
        subdivisions = execution.subdivisions
        subdivision_shapes = tuple(tuple(int(dim) for dim in subdivision.shape) for subdivision in subdivisions)
        decoder_output_coordinate_shape = execution.decoder_output_coordinate_shape
        decoder_output_shape = execution.decoder_output_shape
        reference_stop = execution.reference_stop
        if convnext0_shape != expected_projection_shape:
            raise ValueError(
                f"{latent_name} decoder ConvNeXt block-0 shape mismatch: expected {expected_projection_shape}, got {convnext0_shape}"
            )
        if level0_shape != expected_projection_shape:
            raise ValueError(f"{latent_name} decoder level-0 shape mismatch: expected {expected_projection_shape}, got {level0_shape}")
        if _can_run_first_upblock(config) and first_upblock_shape is not None:
            if first_upblock_coordinate_shape[0] != first_upblock_shape[0]:
                raise ValueError(
                    f"{latent_name} decoder first C2S up-block coordinate/feature mismatch: "
                    f"{first_upblock_coordinate_shape} vs {first_upblock_shape}"
                )
            expected_up_channels = config.model_channels[1]
            if first_upblock_shape[1] != expected_up_channels:
                raise ValueError(
                    f"{latent_name} decoder first C2S up-block channel mismatch: expected {expected_up_channels}, "
                    f"got {first_upblock_shape[1]}"
                )

    return StructuredLatentDecoderProbe(
        checkpoint_path=str(checkpoint_path),
        latent_name=latent_name,
        coordinate_shape=coordinate_shape,
        feature_shape=feature_shape,
        input_projection_shape=projection_shape,
        convnext0_output_shape=convnext0_shape,
        level0_completed_blocks=level0_completed_blocks,
        level0_output_shape=level0_shape,
        first_upblock_coordinate_shape=first_upblock_coordinate_shape,
        first_upblock_output_shape=first_upblock_shape,
        first_upblock_subdivision_shape=first_upblock_subdivision_shape,
        completed_levels=completed_levels,
        subdivision_shapes=subdivision_shapes,
        decoder_output_coordinate_shape=decoder_output_coordinate_shape,
        decoder_output_shape=decoder_output_shape,
        reference_stop=reference_stop,
        reference_token_limit=reference_token_limit,
        loaded_tensor_names=tuple(name for name in tensor_names if name in tensors),
        inspected_tensor_names=tuple(info.name for info in infos),
    ), subdivisions


def probe_decode_latents_boundary(
    shape_checkpoint_path: str | Path,
    shape_config: StructuredLatentDecoderConfig,
    texture_checkpoint_path: str | Path,
    texture_config: StructuredLatentDecoderConfig,
    *,
    shape_slat_coordinates: mx.array,
    shape_slat_features: mx.array,
    texture_slat_coordinates: mx.array,
    texture_slat_features: mx.array,
    resolution: int,
    reference_token_limit: int = STRUCTURED_LATENT_DECODER_REFERENCE_TOKEN_LIMIT,
) -> DecodeLatentsProbe:
    if resolution <= 0:
        raise ValueError("decode resolution must be positive")
    reference_token_limit = _reference_token_limit(reference_token_limit)

    shape_probe, shape_subdivisions = _probe_structured_latent_decoder_boundary(
        shape_checkpoint_path,
        shape_config,
        shape_slat_coordinates,
        shape_slat_features,
        latent_name="shape_slat",
        reference_token_limit=reference_token_limit,
    )
    texture_guide_subdivisions = shape_subdivisions if shape_subdivisions else None
    texture_probe, _ = _probe_structured_latent_decoder_boundary(
        texture_checkpoint_path,
        texture_config,
        texture_slat_coordinates,
        texture_slat_features,
        latent_name="texture_slat",
        guide_subdivisions=texture_guide_subdivisions,
        reference_token_limit=reference_token_limit,
    )

    return DecodeLatentsProbe(
        shape_probe=shape_probe,
        texture_probe=texture_probe,
        resolution=resolution,
        blocker_operation="MLX shape latent decoder SparseConvNeXt/FlexiDualGrid forward",
        blocker_detail=(
            f"shape decoder from_latent projection executed with output shape {shape_probe.input_projection_shape}; "
            f"{_decoder_reference_detail('shape decoder', shape_probe)}; "
            f"texture decoder from_latent projection validates with output shape {texture_probe.input_projection_shape}; "
            f"{_decoder_reference_detail('texture decoder', texture_probe)}; "
            "next unported component is large-token sparse ConvNeXt/up-block decoder execution, followed by "
            "FlexiDualGrid mesh extraction and real-token texture voxel decoding"
        ),
    )


def run_flexi_dual_grid_vae_encoder(
    checkpoint_path: str | Path,
    config: StructuredLatentEncoderConfig,
    coordinates: mx.array,
    dual_vertices: mx.array,
    intersected: mx.array,
    *,
    sample_posterior: bool = False,
) -> FlexiDualGridEncoderResult:
    """Run the TRELLIS.2 FlexiDualGrid encoder on prepared dual-grid tensors."""

    if config.name != "FlexiDualGridVaeEncoder":
        raise ValueError(f"shape encoder must be FlexiDualGridVaeEncoder, got {config.name}")
    if config.in_channels != 6:
        raise ValueError(f"FlexiDualGridVaeEncoder expects 6 input channels, got {config.in_channels}")
    coordinate_shape, vertex_shape, intersected_shape = _validate_flexi_dual_grid_encoder_inputs(
        coordinates,
        dual_vertices,
        intersected,
    )
    input_features = mx.concatenate(
        (
            dual_vertices.astype(mx.float32) - 0.5,
            intersected.astype(mx.float32) - 0.5,
        ),
        axis=1,
    )
    tensor_names = _structured_latent_encoder_tensor_names(config)
    tensors = load_checkpoint_tensors(checkpoint_path, names=tensor_names)
    infos = inspect_checkpoint(checkpoint_path, names=tensor_names)
    _require_checkpoint_infos(infos, tensor_names)
    _validate_tensor_shape("input_layer.weight", tensors["input_layer.weight"].shape, (config.model_channels[0], config.in_channels))
    _validate_tensor_shape("input_layer.bias", tensors["input_layer.bias"].shape, (config.model_channels[0],))
    _validate_tensor_shape("to_latent.weight", tensors["to_latent.weight"].shape, (2 * config.latent_channels, config.model_channels[-1]))
    _validate_tensor_shape("to_latent.bias", tensors["to_latent.bias"].shape, (2 * config.latent_channels,))

    projected = _linear_chunked(
        input_features.astype(tensors["input_layer.weight"].dtype),
        tensors["input_layer.weight"],
        tensors["input_layer.bias"],
    )
    mx.eval(projected)
    execution = _run_sparse_unet_encoder_reference(config, coordinates, projected, tensors)
    latent_raw = _linear_chunked(
        _layer_norm_no_affine(execution.features.astype(mx.float32), eps=1e-5),
        tensors["to_latent.weight"].astype(mx.float32),
        tensors["to_latent.bias"].astype(mx.float32),
    )
    mean, logvar = mx.split(latent_raw, 2, axis=1)
    if sample_posterior:
        std = mx.exp(0.5 * logvar)
        features = mean + std * mx.random.normal(mean.shape, dtype=mean.dtype)
    else:
        features = mean
    mx.eval(features, mean, logvar)
    probe = StructuredLatentEncoderProbe(
        checkpoint_path=str(checkpoint_path),
        encoder_name=config.name,
        coordinate_shape=coordinate_shape,
        vertex_shape=vertex_shape,
        intersected_shape=intersected_shape,
        input_feature_shape=tuple(int(dim) for dim in input_features.shape),
        input_projection_shape=tuple(int(dim) for dim in projected.shape),
        completed_levels=execution.completed_levels,
        first_downblock_coordinate_shape=execution.first_downblock_coordinate_shape,
        first_downblock_output_shape=execution.first_downblock_output_shape,
        latent_feature_shape=tuple(int(dim) for dim in features.shape),
        mean_shape=tuple(int(dim) for dim in mean.shape),
        logvar_shape=tuple(int(dim) for dim in logvar.shape),
        loaded_tensor_names=tuple(name for name in tensor_names if name in tensors),
        inspected_tensor_names=tuple(info.name for info in infos),
    )
    return FlexiDualGridEncoderResult(
        coordinates=execution.coordinates,
        features=features,
        mean=mean,
        logvar=logvar,
        probe=probe,
    )


def run_shape_decoder_to_fields(
    checkpoint_path: str | Path,
    config: StructuredLatentDecoderConfig,
    coordinates: mx.array,
    features: mx.array,
    *,
    decoder_token_limit: int = 12_000_000,
) -> ShapeDecoderResult:
    """Run the full shape decoder and return 7-channel FlexiDualGrid fields."""

    if config.name != "FlexiDualGridVaeDecoder":
        raise ValueError(f"shape decoder must be FlexiDualGridVaeDecoder, got {config.name}")
    if config.out_channels != 7:
        raise ValueError(f"shape decoder must produce 7 output channels, got {config.out_channels}")

    decoder_token_limit = _reference_token_limit(decoder_token_limit)
    coordinate_shape = tuple(int(dim) for dim in coordinates.shape)
    feature_shape = tuple(int(dim) for dim in features.shape)
    _validate_latent_layout(coordinate_shape, feature_shape, config, "shape_slat")
    tensor_names = _structured_latent_decoder_tensor_names(config, max_level_count=len(config.num_blocks))
    tensors = load_checkpoint_tensors(checkpoint_path, names=tensor_names)
    infos = inspect_checkpoint(checkpoint_path, names=tensor_names)
    _require_checkpoint_infos(infos, tensor_names)
    _validate_tensor_shape("from_latent.weight", tensors["from_latent.weight"].shape, (config.model_channels[0], config.latent_channels))
    _validate_tensor_shape("from_latent.bias", tensors["from_latent.bias"].shape, (config.model_channels[0],))
    _validate_checkpoint_info_shape(infos, "output_layer.weight", (config.out_channels, config.model_channels[-1]))
    _validate_checkpoint_info_shape(infos, "output_layer.bias", (config.out_channels,))
    _validate_checkpoint_infos(infos, config, max_level_count=len(config.num_blocks))
    projected = _linear_chunked(
        features.astype(tensors["from_latent.weight"].dtype),
        tensors["from_latent.weight"],
        tensors["from_latent.bias"],
    )
    mx.eval(projected)
    execution = _run_sparse_unet_decoder_reference(
        config,
        coordinates,
        projected,
        tensors,
        reference_token_limit=decoder_token_limit,
    )
    if execution.decoder_output_coordinates is None or execution.decoder_output_features is None:
        raise ValueError(
            f"shape decoder stopped before 7-channel output: completed_levels={execution.completed_levels}, "
            f"input_tokens={feature_shape[0]}, output_coordinates={execution.decoder_output_coordinate_shape}, "
            f"output_features={execution.decoder_output_shape}, reason={execution.reference_stop}"
        )
    probe = StructuredLatentDecoderProbe(
        checkpoint_path=str(checkpoint_path),
        latent_name="shape_slat",
        coordinate_shape=coordinate_shape,
        feature_shape=feature_shape,
        input_projection_shape=tuple(int(dim) for dim in projected.shape),
        convnext0_output_shape=execution.convnext0_output_shape,
        level0_completed_blocks=execution.level0_completed_blocks,
        level0_output_shape=execution.level0_output_shape,
        first_upblock_coordinate_shape=execution.first_upblock_coordinate_shape,
        first_upblock_output_shape=execution.first_upblock_output_shape,
        first_upblock_subdivision_shape=execution.first_upblock_subdivision_shape,
        completed_levels=execution.completed_levels,
        subdivision_shapes=tuple(tuple(int(dim) for dim in subdivision.shape) for subdivision in execution.subdivisions),
        decoder_output_coordinate_shape=execution.decoder_output_coordinate_shape,
        decoder_output_shape=execution.decoder_output_shape,
        reference_stop=execution.reference_stop,
        reference_token_limit=decoder_token_limit,
        loaded_tensor_names=tuple(name for name in tensor_names if name in tensors),
        inspected_tensor_names=tuple(info.name for info in infos),
    )
    return ShapeDecoderResult(
        coordinates=execution.decoder_output_coordinates,
        fields=execution.decoder_output_features,
        subdivisions=execution.subdivisions,
        probe=probe,
    )


def run_texture_decoder_to_representation(
    checkpoint_path: str | Path,
    config: StructuredLatentDecoderConfig,
    coordinates: mx.array,
    features: mx.array,
    *,
    guide_subdivisions: tuple[mx.array, ...],
    decoder_token_limit: int = 12_000_000,
    decode_resolution: int | None = None,
    shape_decoder_coordinates: mx.array | None = None,
) -> TextureDecoderResult:
    if config.name != "SparseUnetVaeDecoder":
        raise ValueError(f"texture decoder must be SparseUnetVaeDecoder, got {config.name}")
    if config.out_channels != 6:
        raise ValueError(f"texture decoder must produce 6 output channels, got {config.out_channels}")
    if config.pred_subdiv is not False:
        raise ValueError(f"texture decoder requires pred_subdiv=False, got {config.pred_subdiv}")
    if not guide_subdivisions:
        raise ValueError("texture decoder requires shape decoder guide_subdivisions")

    decoder_token_limit = _reference_token_limit(decoder_token_limit)
    coordinate_shape = tuple(int(dim) for dim in coordinates.shape)
    feature_shape = tuple(int(dim) for dim in features.shape)
    _validate_latent_layout(coordinate_shape, feature_shape, config, "texture_slat")
    tensor_names = _structured_latent_decoder_tensor_names(config, max_level_count=len(config.num_blocks))
    tensors = load_checkpoint_tensors(checkpoint_path, names=tensor_names)
    infos = inspect_checkpoint(checkpoint_path, names=tensor_names)
    _require_checkpoint_infos(infos, tensor_names)
    _validate_tensor_shape("from_latent.weight", tensors["from_latent.weight"].shape, (config.model_channels[0], config.latent_channels))
    _validate_tensor_shape("from_latent.bias", tensors["from_latent.bias"].shape, (config.model_channels[0],))
    _validate_checkpoint_info_shape(infos, "output_layer.weight", (config.out_channels, config.model_channels[-1]))
    _validate_checkpoint_info_shape(infos, "output_layer.bias", (config.out_channels,))
    _validate_checkpoint_infos(infos, config, max_level_count=len(config.num_blocks))
    projected = _linear_chunked(
        features.astype(tensors["from_latent.weight"].dtype),
        tensors["from_latent.weight"],
        tensors["from_latent.bias"],
    )
    mx.eval(projected)
    execution = _run_sparse_unet_decoder_reference(
        config,
        coordinates,
        projected,
        tensors,
        guide_subdivisions=guide_subdivisions,
        reference_token_limit=decoder_token_limit,
    )
    if execution.decoder_output_coordinates is None or execution.decoder_output_features is None:
        raise ValueError(
            f"texture decoder stopped before 6-channel output: completed_levels={execution.completed_levels}, "
            f"input_tokens={feature_shape[0]}, output_coordinates={execution.decoder_output_coordinate_shape}, "
            f"output_features={execution.decoder_output_shape}, reason={execution.reference_stop}"
        )

    attributes = execution.decoder_output_features * 0.5 + 0.5
    mx.eval(attributes)
    output_coordinates = execution.decoder_output_coordinates
    spatial_shape = _spatial_shape_from_coordinates(output_coordinates[:, 1:])
    batch_size = int(mx.max(output_coordinates[:, 0].astype(mx.int32)).item()) + 1
    probe = StructuredLatentDecoderProbe(
        checkpoint_path=str(checkpoint_path),
        latent_name="texture_slat",
        coordinate_shape=coordinate_shape,
        feature_shape=feature_shape,
        input_projection_shape=tuple(int(dim) for dim in projected.shape),
        convnext0_output_shape=execution.convnext0_output_shape,
        level0_completed_blocks=execution.level0_completed_blocks,
        level0_output_shape=execution.level0_output_shape,
        first_upblock_coordinate_shape=execution.first_upblock_coordinate_shape,
        first_upblock_output_shape=execution.first_upblock_output_shape,
        first_upblock_subdivision_shape=execution.first_upblock_subdivision_shape,
        completed_levels=execution.completed_levels,
        subdivision_shapes=tuple(tuple(int(dim) for dim in subdivision.shape) for subdivision in execution.subdivisions),
        decoder_output_coordinate_shape=execution.decoder_output_coordinate_shape,
        decoder_output_shape=execution.decoder_output_shape,
        reference_stop=execution.reference_stop,
        reference_token_limit=decoder_token_limit,
        loaded_tensor_names=tuple(name for name in tensor_names if name in tensors),
        inspected_tensor_names=tuple(info.name for info in infos),
    )
    shape_coordinate_shape = None
    if shape_decoder_coordinates is not None:
        shape_coordinate_shape = tuple(int(dim) for dim in shape_decoder_coordinates.shape)
    return TextureDecoderResult(
        coordinates=output_coordinates,
        attributes=attributes,
        probe=probe,
        guide_subdivision_shapes=tuple(tuple(int(dim) for dim in guide.shape) for guide in guide_subdivisions),
        spatial_shape=spatial_shape,
        batch_size=batch_size,
        decode_resolution=decode_resolution,
        voxel_size=(1.0 / float(decode_resolution)) if decode_resolution is not None else None,
        shape_decoder_coordinate_shape=shape_coordinate_shape,
    )


def run_shape_decoder_upsample_coordinates(
    checkpoint_path: str | Path,
    config: StructuredLatentDecoderConfig,
    coordinates: mx.array,
    features: mx.array,
    *,
    upsample_times: int = 4,
    decoder_token_limit: int = 1_000_000,
) -> ShapeDecoderUpsampleResult:
    if config.name != "FlexiDualGridVaeDecoder":
        raise ValueError(f"shape upsample decoder must be FlexiDualGridVaeDecoder, got {config.name}")
    if not config.pred_subdiv:
        raise ValueError("shape upsample decoder requires pred_subdiv=True")
    if upsample_times < 0 or upsample_times >= len(config.num_blocks):
        raise ValueError(f"upsample_times must be in [0, {len(config.num_blocks) - 1}], got {upsample_times}")

    decoder_token_limit = _reference_token_limit(decoder_token_limit)
    coordinate_shape = tuple(int(dim) for dim in coordinates.shape)
    feature_shape = tuple(int(dim) for dim in features.shape)
    _validate_latent_layout(coordinate_shape, feature_shape, config, "shape_slat")
    tensor_names = _structured_latent_decoder_tensor_names(config, max_level_count=upsample_times)
    tensors = load_checkpoint_tensors(checkpoint_path, names=tensor_names)
    infos = inspect_checkpoint(checkpoint_path, names=tensor_names)
    _require_checkpoint_infos(infos, tensor_names)
    _validate_tensor_shape("from_latent.weight", tensors["from_latent.weight"].shape, (config.model_channels[0], config.latent_channels))
    _validate_tensor_shape("from_latent.bias", tensors["from_latent.bias"].shape, (config.model_channels[0],))
    _validate_checkpoint_infos(infos, config, max_level_count=upsample_times)

    current_coordinates = coordinates
    current_features = _linear_chunked(
        features.astype(tensors["from_latent.weight"].dtype),
        tensors["from_latent.weight"],
        tensors["from_latent.bias"],
    )
    mx.eval(current_features)
    subdivisions: list[mx.array] = []
    completed_upsamples = 0
    for level_index in range(len(config.num_blocks)):
        if level_index == upsample_times:
            return ShapeDecoderUpsampleResult(
                coordinates=current_coordinates,
                subdivisions=tuple(subdivisions),
                input_coordinate_shape=coordinate_shape,
                output_coordinate_shape=tuple(int(dim) for dim in current_coordinates.shape),
                completed_upsamples=completed_upsamples,
            )
        token_count = int(current_coordinates.shape[0])
        if token_count > decoder_token_limit:
            raise ValueError(
                f"shape decoder upsample stopped before level {level_index}: token_count={token_count} "
                f"exceeds decoder_token_limit={decoder_token_limit}"
            )
        for block_index in range(config.num_blocks[level_index]):
            current_features = _sparse_convnext_block_forward(
                current_coordinates,
                current_features,
                tensors,
                prefix=f"blocks.{level_index}.{block_index}",
            )
            mx.eval(current_features)
        if level_index >= len(config.num_blocks) - 1:
            break
        if config.up_block_type[level_index] != "SparseResBlockC2S3d":
            raise ValueError(f"shape decoder upsample stopped at unsupported up-block {config.up_block_type[level_index]}")
        current_coordinates, current_features, subdiv_logits = _sparse_c2s_upblock_forward(
            current_coordinates,
            current_features,
            tensors,
            prefix=f"blocks.{level_index}.{config.num_blocks[level_index]}",
            in_channels=config.model_channels[level_index],
            out_channels=config.model_channels[level_index + 1],
            pred_subdiv=True,
        )
        mx.eval(current_coordinates, current_features, subdiv_logits)
        subdivisions.append(subdiv_logits)
        completed_upsamples += 1

    raise ValueError(f"shape decoder ended before upsample_times={upsample_times}")


def _structured_latent_decoder_tensor_names(
    config: StructuredLatentDecoderConfig,
    *,
    max_level_count: int,
) -> tuple[str, ...]:
    names: list[str] = list(STRUCTURED_LATENT_DECODER_BASE_TENSOR_NAMES)
    bounded_level_count = max(0, min(int(max_level_count), len(config.num_blocks)))
    for level_index in range(bounded_level_count):
        for block_index in range(config.num_blocks[level_index]):
            names.extend(_convnext_block_tensor_names(f"blocks.{level_index}.{block_index}"))
        if level_index < len(config.model_channels) - 1 and config.up_block_type[level_index] == "SparseResBlockC2S3d":
            names.extend(
                _c2s_upblock_tensor_names(
                    f"blocks.{level_index}.{config.num_blocks[level_index]}",
                    pred_subdiv=bool(config.pred_subdiv),
                )
            )
    return tuple(dict.fromkeys(names))


def _convnext_block_tensor_names(prefix: str) -> tuple[str, ...]:
    return (
        f"{prefix}.conv.weight",
        f"{prefix}.conv.bias",
        f"{prefix}.norm.weight",
        f"{prefix}.norm.bias",
        f"{prefix}.mlp.0.weight",
        f"{prefix}.mlp.0.bias",
        f"{prefix}.mlp.2.weight",
        f"{prefix}.mlp.2.bias",
    )


def _c2s_upblock_tensor_names(prefix: str, *, pred_subdiv: bool) -> tuple[str, ...]:
    names = [
        f"{prefix}.norm1.weight",
        f"{prefix}.norm1.bias",
        f"{prefix}.conv1.weight",
        f"{prefix}.conv1.bias",
        f"{prefix}.conv2.weight",
        f"{prefix}.conv2.bias",
    ]
    if pred_subdiv:
        names.extend((f"{prefix}.to_subdiv.weight", f"{prefix}.to_subdiv.bias"))
    return tuple(names)


def _validate_latent_layout(
    coordinate_shape: tuple[int, ...],
    feature_shape: tuple[int, ...],
    config: StructuredLatentDecoderConfig,
    latent_name: str,
) -> None:
    if len(coordinate_shape) != 2 or coordinate_shape[1] != 4:
        raise ValueError(f"{latent_name} coordinates must have shape (num_tokens, 4), got {coordinate_shape}")
    if coordinate_shape[0] <= 0:
        raise ValueError(f"{latent_name} coordinates must contain at least one token")
    if len(feature_shape) != 2:
        raise ValueError(f"{latent_name} features must have shape (num_tokens, channels), got {feature_shape}")
    if feature_shape[0] != coordinate_shape[0]:
        raise ValueError(f"{latent_name} feature/token mismatch: expected {coordinate_shape[0]} tokens, got {feature_shape[0]}")
    if feature_shape[1] != config.latent_channels:
        raise ValueError(f"{latent_name} feature width mismatch: expected {config.latent_channels}, got {feature_shape[1]}")


def _reference_token_limit(limit: int) -> int:
    value = int(limit)
    if value <= 0:
        raise ValueError(f"decoder reference token limit must be positive, got {value}")
    return value


def _validate_checkpoint_infos(
    infos: tuple[CheckpointTensorInfo, ...],
    config: StructuredLatentDecoderConfig,
    *,
    max_level_count: int,
) -> None:
    bounded_level_count = max(0, min(int(max_level_count), len(config.num_blocks)))
    for level_index in range(bounded_level_count):
        channels = config.model_channels[level_index]
        for block_index in range(config.num_blocks[level_index]):
            prefix = f"blocks.{level_index}.{block_index}"
            _validate_checkpoint_info_shape(infos, f"{prefix}.conv.weight", (channels, 3, 3, 3, channels))
            _validate_checkpoint_info_shape(infos, f"{prefix}.conv.bias", (channels,))
            _validate_checkpoint_info_shape(infos, f"{prefix}.norm.weight", (channels,))
            _validate_checkpoint_info_shape(infos, f"{prefix}.norm.bias", (channels,))
            _validate_checkpoint_info_shape(infos, f"{prefix}.mlp.0.weight", (channels * 4, channels))
            _validate_checkpoint_info_shape(infos, f"{prefix}.mlp.0.bias", (channels * 4,))
            _validate_checkpoint_info_shape(infos, f"{prefix}.mlp.2.weight", (channels, channels * 4))
            _validate_checkpoint_info_shape(infos, f"{prefix}.mlp.2.bias", (channels,))
        if level_index < len(config.model_channels) - 1:
            _validate_upblock_checkpoint_infos(infos, config, level_index=level_index)


def _validate_upblock_checkpoint_infos(
    infos: tuple[CheckpointTensorInfo, ...],
    config: StructuredLatentDecoderConfig,
    *,
    level_index: int,
) -> None:
    if config.up_block_type[level_index] != "SparseResBlockC2S3d":
        return
    in_channels = config.model_channels[level_index]
    out_channels = config.model_channels[level_index + 1]
    prefix = f"blocks.{level_index}.{config.num_blocks[level_index]}"
    _validate_checkpoint_info_shape(infos, f"{prefix}.norm1.weight", (in_channels,))
    _validate_checkpoint_info_shape(infos, f"{prefix}.norm1.bias", (in_channels,))
    _validate_checkpoint_info_shape(infos, f"{prefix}.conv1.weight", (out_channels * 8, 3, 3, 3, in_channels))
    _validate_checkpoint_info_shape(infos, f"{prefix}.conv1.bias", (out_channels * 8,))
    _validate_checkpoint_info_shape(infos, f"{prefix}.conv2.weight", (out_channels, 3, 3, 3, out_channels))
    _validate_checkpoint_info_shape(infos, f"{prefix}.conv2.bias", (out_channels,))
    if config.pred_subdiv:
        _validate_checkpoint_info_shape(infos, f"{prefix}.to_subdiv.weight", (8, in_channels))
        _validate_checkpoint_info_shape(infos, f"{prefix}.to_subdiv.bias", (8,))


def _can_run_first_upblock(config: StructuredLatentDecoderConfig) -> bool:
    return (
        len(config.model_channels) > 1
        and config.up_block_type[0] == "SparseResBlockC2S3d"
        and bool(config.pred_subdiv)
    )


def _structured_latent_encoder_tensor_names(config: StructuredLatentEncoderConfig) -> tuple[str, ...]:
    names = list(STRUCTURED_LATENT_ENCODER_BASE_TENSOR_NAMES)
    for level_index, block_count in enumerate(config.num_blocks):
        if config.block_type[level_index] != "SparseConvNeXtBlock3d":
            raise ValueError(f"unsupported encoder block type: {config.block_type[level_index]}")
        for block_index in range(block_count):
            prefix = f"blocks.{level_index}.{block_index}"
            names.extend(
                (
                    f"{prefix}.conv.weight",
                    f"{prefix}.conv.bias",
                    f"{prefix}.norm.weight",
                    f"{prefix}.norm.bias",
                    f"{prefix}.mlp.0.weight",
                    f"{prefix}.mlp.0.bias",
                    f"{prefix}.mlp.2.weight",
                    f"{prefix}.mlp.2.bias",
                )
            )
        if level_index < len(config.num_blocks) - 1:
            if config.down_block_type[level_index] != "SparseResBlockS2C3d":
                raise ValueError(f"unsupported encoder down-block type: {config.down_block_type[level_index]}")
            prefix = f"blocks.{level_index}.{block_count}"
            names.extend(
                (
                    f"{prefix}.norm1.weight",
                    f"{prefix}.norm1.bias",
                    f"{prefix}.conv1.weight",
                    f"{prefix}.conv1.bias",
                    f"{prefix}.conv2.weight",
                    f"{prefix}.conv2.bias",
                )
            )
    return tuple(names)


def _validate_flexi_dual_grid_encoder_inputs(
    coordinates: mx.array,
    dual_vertices: mx.array,
    intersected: mx.array,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    coordinate_shape = tuple(int(dim) for dim in coordinates.shape)
    vertex_shape = tuple(int(dim) for dim in dual_vertices.shape)
    intersected_shape = tuple(int(dim) for dim in intersected.shape)
    if coordinate_shape[1:] != (4,) or len(coordinate_shape) != 2:
        raise ValueError(f"FlexiDualGrid encoder coordinates must have shape (num_tokens, 4), got {coordinate_shape}")
    expected_feature_shape = (coordinate_shape[0], 3)
    if vertex_shape != expected_feature_shape:
        raise ValueError(f"FlexiDualGrid dual_vertices must have shape {expected_feature_shape}, got {vertex_shape}")
    if intersected_shape != expected_feature_shape:
        raise ValueError(f"FlexiDualGrid intersected must have shape {expected_feature_shape}, got {intersected_shape}")
    if coordinate_shape[0] == 0:
        raise ValueError("FlexiDualGrid encoder requires at least one token")
    if bool(mx.any(coordinates[:, 1:].astype(mx.int32) < 0).item()):
        raise ValueError("FlexiDualGrid encoder spatial coordinates must be non-negative")
    return coordinate_shape, vertex_shape, intersected_shape


def _run_sparse_unet_encoder_reference(
    config: StructuredLatentEncoderConfig,
    coordinates: mx.array,
    projected: mx.array,
    tensors: dict[str, mx.array],
) -> _StructuredLatentEncoderExecution:
    current_coordinates = coordinates
    current_features = projected.astype(mx.float32)
    completed_levels = 0
    first_downblock_coordinate_shape = None
    first_downblock_output_shape = None
    for level_index, block_count in enumerate(config.num_blocks):
        for block_index in range(block_count):
            current_features = _sparse_convnext_block_forward(
                current_coordinates,
                current_features,
                tensors,
                prefix=f"blocks.{level_index}.{block_index}",
            )
            mx.eval(current_features)
        completed_levels = level_index + 1
        if level_index >= len(config.num_blocks) - 1:
            continue
        current_coordinates, current_features = _sparse_s2c_downblock_forward(
            current_coordinates,
            current_features,
            tensors,
            prefix=f"blocks.{level_index}.{block_count}",
            in_channels=config.model_channels[level_index],
            out_channels=config.model_channels[level_index + 1],
        )
        mx.eval(current_coordinates, current_features)
        if level_index == 0:
            first_downblock_coordinate_shape = tuple(int(dim) for dim in current_coordinates.shape)
            first_downblock_output_shape = tuple(int(dim) for dim in current_features.shape)
    return _StructuredLatentEncoderExecution(
        coordinates=current_coordinates,
        features=current_features,
        completed_levels=completed_levels,
        first_downblock_coordinate_shape=first_downblock_coordinate_shape,
        first_downblock_output_shape=first_downblock_output_shape,
    )


def _sparse_s2c_downblock_forward(
    coordinates: mx.array,
    features: mx.array,
    tensors: dict[str, mx.array],
    *,
    prefix: str,
    in_channels: int,
    out_channels: int,
) -> tuple[mx.array, mx.array]:
    if int(features.shape[1]) != in_channels:
        raise ValueError(f"S2C down-block expected {in_channels} input channels, got {int(features.shape[1])}")
    if out_channels % 8 != 0:
        raise ValueError(f"S2C down-block output channels must be divisible by 8, got {out_channels}")
    normalized = _layer_norm(
        features.astype(mx.float32),
        tensors[f"{prefix}.norm1.weight"].astype(mx.float32),
        tensors[f"{prefix}.norm1.bias"].astype(mx.float32),
        eps=1e-6,
    )
    hidden = _silu(normalized)
    hidden = _sparse_conv3d_forward(coordinates, hidden, tensors, prefix=f"{prefix}.conv1")
    down_coordinates, hidden = _spatial_to_channel(coordinates, hidden)
    skip_coordinates, skip_features = _spatial_to_channel(coordinates, features.astype(mx.float32))
    if tuple(down_coordinates.tolist()) != tuple(skip_coordinates.tolist()):
        raise ValueError("S2C down-block skip coordinates do not match hidden coordinates")

    hidden = _layer_norm_no_affine(hidden, eps=1e-6)
    hidden = _silu(hidden)
    hidden = _sparse_conv3d_forward(down_coordinates, hidden, tensors, prefix=f"{prefix}.conv2")

    skip_width = int(skip_features.shape[1])
    if skip_width % out_channels != 0:
        raise ValueError(f"S2C down-block cannot reduce skip width {skip_width} to {out_channels} channels")
    skip_features = mx.mean(mx.reshape(skip_features, (int(skip_features.shape[0]), out_channels, skip_width // out_channels)), axis=-1)
    return down_coordinates, hidden + skip_features


def _spatial_to_channel(coordinates: mx.array, features: mx.array) -> tuple[mx.array, mx.array]:
    coordinate_shape = tuple(int(dim) for dim in coordinates.shape)
    feature_shape = tuple(int(dim) for dim in features.shape)
    if coordinate_shape[1:] != (4,) or len(coordinate_shape) != 2:
        raise ValueError(f"S2C coordinates must have shape (num_tokens, 4), got {coordinate_shape}")
    if len(feature_shape) != 2 or feature_shape[0] != coordinate_shape[0]:
        raise ValueError(f"S2C features must have shape ({coordinate_shape[0]}, channels), got {feature_shape}")

    coords_np = np.array(coordinates.tolist(), dtype=np.int32)
    parent_np = coords_np.copy()
    parent_np[:, 1:] //= 2
    child_indices = (
        (coords_np[:, 1] % 2)
        + 2 * (coords_np[:, 2] % 2)
        + 4 * (coords_np[:, 3] % 2)
    ).astype(np.int64)
    parent_keys = sorted({tuple(int(value) for value in row) for row in parent_np.tolist()})
    parent_index = {key: index for index, key in enumerate(parent_keys)}
    output = np.zeros((len(parent_keys), feature_shape[1] * 8), dtype=np.float32)
    source_features = np.array(features.tolist(), dtype=np.float32)
    for row_index, parent in enumerate(parent_np.tolist()):
        key = tuple(int(value) for value in parent)
        child = int(child_indices[row_index])
        start = child * feature_shape[1]
        output[parent_index[key], start : start + feature_shape[1]] = source_features[row_index]
    return mx.array(np.asarray(parent_keys, dtype=np.int32), dtype=mx.int32), mx.array(output, dtype=features.dtype)


def _run_sparse_unet_decoder_reference(
    config: StructuredLatentDecoderConfig,
    coordinates: mx.array,
    projected: mx.array,
    tensors: dict[str, mx.array],
    *,
    guide_subdivisions: tuple[mx.array, ...] | None = None,
    reference_token_limit: int = STRUCTURED_LATENT_DECODER_REFERENCE_TOKEN_LIMIT,
) -> _StructuredLatentDecoderExecution:
    current_coordinates = coordinates
    current_features = projected
    convnext0_shape = None
    level0_completed_blocks = 0
    level0_shape = None
    first_upblock_coordinate_shape = None
    first_upblock_shape = None
    first_upblock_subdivision_shape = None
    completed_levels = 0
    subdivisions: list[mx.array] = []
    reference_stop = None

    for level_index in range(len(config.num_blocks)):
        token_count = int(current_coordinates.shape[0])
        final_zero_block_level = level_index == len(config.num_blocks) - 1 and config.num_blocks[level_index] == 0
        if token_count > reference_token_limit and not final_zero_block_level:
            reference_stop = (
                f"decoder reference stopped before level {level_index} for {token_count} tokens "
                f"above limit {reference_token_limit}"
            )
            break

        for block_index in range(config.num_blocks[level_index]):
            current_features = _sparse_convnext_block_forward(
                current_coordinates,
                current_features,
                tensors,
                prefix=f"blocks.{level_index}.{block_index}",
            )
            mx.eval(current_features)
            if level_index == 0:
                if block_index == 0:
                    convnext0_shape = tuple(int(dim) for dim in current_features.shape)
                level0_completed_blocks += 1

        if level_index == 0:
            level0_shape = tuple(int(dim) for dim in current_features.shape)
        completed_levels = level_index + 1

        if level_index >= len(config.num_blocks) - 1:
            continue
        if config.up_block_type[level_index] != "SparseResBlockC2S3d":
            reference_stop = f"decoder reference stopped at unsupported up-block {config.up_block_type[level_index]}"
            break

        guide_subdivision = None
        if not config.pred_subdiv:
            if guide_subdivisions is None or level_index >= len(guide_subdivisions):
                reference_stop = f"decoder reference stopped before level {level_index} C2S because guide subdivision is unavailable"
                break
            guide_subdivision = guide_subdivisions[level_index]
            guide_shape = tuple(int(dim) for dim in guide_subdivision.shape)
            expected_guide_shape = (token_count, 8)
            if guide_shape != expected_guide_shape:
                raise ValueError(
                    f"guide_subdivisions[{level_index}] must have shape {expected_guide_shape} for current decoder tokens, "
                    f"got {guide_shape}"
                )

        next_coordinates, next_features, subdiv_logits = _sparse_c2s_upblock_forward(
            current_coordinates,
            current_features,
            tensors,
            prefix=f"blocks.{level_index}.{config.num_blocks[level_index]}",
            in_channels=config.model_channels[level_index],
            out_channels=config.model_channels[level_index + 1],
            pred_subdiv=bool(config.pred_subdiv),
            guide_subdivision=guide_subdivision,
        )
        mx.eval(next_coordinates, next_features, subdiv_logits)
        if config.pred_subdiv:
            subdivisions.append(subdiv_logits)
        if level_index == 0:
            first_upblock_coordinate_shape = tuple(int(dim) for dim in next_coordinates.shape)
            first_upblock_shape = tuple(int(dim) for dim in next_features.shape)
            first_upblock_subdivision_shape = tuple(int(dim) for dim in subdiv_logits.shape)
        current_coordinates = next_coordinates
        current_features = next_features

    decoder_output_coordinates = None
    decoder_output_features = None
    decoder_output_coordinate_shape = None
    decoder_output_shape = None
    if reference_stop is None and completed_levels == len(config.num_blocks):
        current_features = _layer_norm_no_affine(current_features.astype(mx.float32), eps=1e-5)
        current_features = _linear_chunked(
            current_features,
            tensors["output_layer.weight"].astype(mx.float32),
            tensors["output_layer.bias"].astype(mx.float32),
        )
        mx.eval(current_features)
        decoder_output_coordinates = current_coordinates
        decoder_output_features = current_features
        decoder_output_coordinate_shape = tuple(int(dim) for dim in current_coordinates.shape)
        decoder_output_shape = tuple(int(dim) for dim in current_features.shape)

    return _StructuredLatentDecoderExecution(
        convnext0_output_shape=convnext0_shape,
        level0_completed_blocks=level0_completed_blocks,
        level0_output_shape=level0_shape,
        first_upblock_coordinate_shape=first_upblock_coordinate_shape,
        first_upblock_output_shape=first_upblock_shape,
        first_upblock_subdivision_shape=first_upblock_subdivision_shape,
        completed_levels=completed_levels,
        subdivisions=tuple(subdivisions),
        decoder_output_coordinates=decoder_output_coordinates,
        decoder_output_features=decoder_output_features,
        decoder_output_coordinate_shape=decoder_output_coordinate_shape,
        decoder_output_shape=decoder_output_shape,
        reference_stop=reference_stop,
    )


def _sparse_convnext_block_forward(
    coordinates: mx.array,
    features: mx.array,
    tensors: dict[str, mx.array],
    *,
    prefix: str,
) -> mx.array:
    spatial_coordinates = coordinates[:, 1:]
    spatial_shape = _spatial_shape_from_coordinates(spatial_coordinates)
    conv_weight = tensors[f"{prefix}.conv.weight"].astype(mx.float32)
    kernel_weights = mx.reshape(
        mx.contiguous(mx.transpose(conv_weight, (1, 2, 3, 4, 0))),
        (27, int(conv_weight.shape[-1]), int(conv_weight.shape[0])),
    )
    map_rows = sparse_conv_map_vectorized(spatial_coordinates.astype(mx.int32), spatial_shape, kernel_size=(3, 3, 3))
    conv = weighted_sparse_conv_chunked(features.astype(mx.float32), map_rows, kernel_weights, target_count=int(features.shape[0]))
    conv = conv + tensors[f"{prefix}.conv.bias"].astype(mx.float32)
    normalized = _layer_norm(
        conv,
        tensors[f"{prefix}.norm.weight"].astype(mx.float32),
        tensors[f"{prefix}.norm.bias"].astype(mx.float32),
        eps=1e-6,
    )
    hidden = _linear_chunked(
        normalized,
        tensors[f"{prefix}.mlp.0.weight"].astype(mx.float32),
        tensors[f"{prefix}.mlp.0.bias"].astype(mx.float32),
    )
    hidden = _silu(hidden)
    hidden = _linear_chunked(
        hidden,
        tensors[f"{prefix}.mlp.2.weight"].astype(mx.float32),
        tensors[f"{prefix}.mlp.2.bias"].astype(mx.float32),
    )
    return hidden + features.astype(mx.float32)


def _sparse_c2s_upblock_forward(
    coordinates: mx.array,
    features: mx.array,
    tensors: dict[str, mx.array],
    *,
    prefix: str,
    in_channels: int,
    out_channels: int,
    pred_subdiv: bool,
    guide_subdivision: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    if not pred_subdiv and guide_subdivision is None:
        raise ValueError("C2S up-block without pred_subdiv requires guide_subdivision")
    if int(features.shape[1]) != in_channels:
        raise ValueError(f"C2S up-block expected {in_channels} input channels, got {int(features.shape[1])}")
    if in_channels % 8 != 0:
        raise ValueError(f"C2S up-block input channels must be divisible by 8, got {in_channels}")

    if pred_subdiv:
        subdiv_logits = _linear_chunked(
            features.astype(mx.float32),
            tensors[f"{prefix}.to_subdiv.weight"].astype(mx.float32),
            tensors[f"{prefix}.to_subdiv.bias"].astype(mx.float32),
        )
        subdivision = subdiv_logits > 0
    else:
        subdiv_logits = guide_subdivision.astype(mx.float32)
        subdivision = guide_subdivision > 0

    normalized = _layer_norm(
        features.astype(mx.float32),
        tensors[f"{prefix}.norm1.weight"].astype(mx.float32),
        tensors[f"{prefix}.norm1.bias"].astype(mx.float32),
        eps=1e-6,
    )
    hidden = _silu(normalized)
    hidden = _sparse_conv3d_forward(coordinates, hidden, tensors, prefix=f"{prefix}.conv1")
    up_coordinates, hidden = _channel_to_spatial(coordinates, hidden, subdivision)
    skip_coordinates, skip_features = _channel_to_spatial(coordinates, features.astype(mx.float32), subdivision)
    if tuple(up_coordinates.tolist()) != tuple(skip_coordinates.tolist()):
        raise ValueError("C2S up-block skip coordinates do not match hidden coordinates")

    hidden = _layer_norm_no_affine(hidden, eps=1e-6)
    hidden = _silu(hidden)
    hidden = _sparse_conv3d_forward(up_coordinates, hidden, tensors, prefix=f"{prefix}.conv2")

    skip_width = int(skip_features.shape[1])
    if skip_width <= 0 or out_channels % skip_width != 0:
        raise ValueError(f"C2S up-block cannot repeat skip width {skip_width} to {out_channels} channels")
    skip_features = mx.repeat(skip_features, repeats=out_channels // skip_width, axis=1)
    return up_coordinates, hidden + skip_features, subdiv_logits


def _sparse_conv3d_forward(
    coordinates: mx.array,
    features: mx.array,
    tensors: dict[str, mx.array],
    *,
    prefix: str,
) -> mx.array:
    spatial_coordinates = coordinates[:, 1:]
    spatial_shape = _spatial_shape_from_coordinates(spatial_coordinates)
    conv_weight = tensors[f"{prefix}.weight"].astype(mx.float32)
    kernel_weights = mx.reshape(
        mx.contiguous(mx.transpose(conv_weight, (1, 2, 3, 4, 0))),
        (27, int(conv_weight.shape[-1]), int(conv_weight.shape[0])),
    )
    map_rows = sparse_conv_map_vectorized(spatial_coordinates.astype(mx.int32), spatial_shape, kernel_size=(3, 3, 3))
    conv = weighted_sparse_conv_chunked(features.astype(mx.float32), map_rows, kernel_weights, target_count=int(features.shape[0]))
    return conv + tensors[f"{prefix}.bias"].astype(mx.float32)


def _channel_to_spatial(
    coordinates: mx.array,
    features: mx.array,
    subdivision: mx.array,
) -> tuple[mx.array, mx.array]:
    coordinate_shape = tuple(int(dim) for dim in coordinates.shape)
    feature_shape = tuple(int(dim) for dim in features.shape)
    subdivision_shape = tuple(int(dim) for dim in subdivision.shape)
    if len(coordinate_shape) != 2 or coordinate_shape[1] != 4:
        raise ValueError(f"C2S coordinates must have shape (num_tokens, 4), got {coordinate_shape}")
    if len(feature_shape) != 2 or feature_shape[0] != coordinate_shape[0]:
        raise ValueError(f"C2S features must have shape ({coordinate_shape[0]}, channels), got {feature_shape}")
    if feature_shape[1] % 8 != 0:
        raise ValueError(f"C2S feature channels must be divisible by 8, got {feature_shape[1]}")
    if subdivision_shape != (coordinate_shape[0], 8):
        raise ValueError(f"C2S subdivision must have shape ({coordinate_shape[0]}, 8), got {subdivision_shape}")

    subdivision_np = np.array(subdivision.tolist(), dtype=np.bool_)
    leaf_counts = subdivision_np.sum(axis=1).astype(np.int64)
    total_leaves = int(leaf_counts.sum())
    if total_leaves <= 0:
        raise ValueError("C2S subdivision must select at least one child voxel")
    source_indices, child_indices = np.nonzero(subdivision_np)

    coordinates_np = np.array(coordinates.tolist(), dtype=np.int32)
    up_coordinates_np = np.repeat(coordinates_np, leaf_counts, axis=0)
    up_coordinates_np[:, 1:] *= 2
    for axis in range(3):
        up_coordinates_np[:, axis + 1] += (child_indices // (2**axis)) % 2

    reshaped_features = mx.reshape(features, (feature_shape[0] * 8, feature_shape[1] // 8))
    gather_indices = mx.array(source_indices * 8 + child_indices, dtype=mx.int32)
    return mx.array(up_coordinates_np, dtype=mx.int32), reshaped_features[gather_indices]


def _layer_norm_no_affine(features: mx.array, *, eps: float) -> mx.array:
    mean = mx.mean(features, axis=-1, keepdims=True)
    variance = mx.mean(mx.square(features - mean), axis=-1, keepdims=True)
    return (features - mean) / mx.sqrt(variance + eps)


def _linear_chunked(features: mx.array, weight: mx.array, bias: mx.array, *, token_chunk_size: int = 16384) -> mx.array:
    token_count = int(features.shape[0])
    chunk_size = int(token_chunk_size)
    if chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    if token_count <= chunk_size:
        return _linear(features, weight, bias)
    chunks = []
    for start in range(0, token_count, chunk_size):
        chunk = _linear(features[start : start + chunk_size], weight, bias)
        mx.eval(chunk)
        chunks.append(chunk)
    return mx.concatenate(chunks, axis=0)


def _decoder_reference_detail(label: str, probe: StructuredLatentDecoderProbe) -> str:
    if probe.convnext0_output_shape is None:
        return (
            f"{label} bounded decoder reference path skipped for {probe.feature_shape[0]} tokens "
            f"above limit {probe.reference_token_limit}"
        )
    detail = (
        f"{label} level-0 executed {probe.level0_completed_blocks} ConvNeXt block(s) "
        f"with output shape {probe.level0_output_shape}"
    )
    if probe.first_upblock_output_shape is not None:
        detail += (
            f"; first C2S up-block produced coordinates {probe.first_upblock_coordinate_shape}, "
            f"features {probe.first_upblock_output_shape}, subdivisions {probe.first_upblock_subdivision_shape}"
        )
    if probe.decoder_output_shape is not None:
        detail += (
            f"; decoder output layer produced coordinates {probe.decoder_output_coordinate_shape}, "
            f"features {probe.decoder_output_shape}"
        )
    if probe.reference_stop is not None:
        detail += f"; {probe.reference_stop}"
    return detail


def _spatial_shape_from_coordinates(coordinates: mx.array) -> tuple[int, int, int]:
    if len(coordinates.shape) != 2 or int(coordinates.shape[1]) != 3:
        raise ValueError(f"spatial coordinates must have shape (num_tokens, 3), got {tuple(coordinates.shape)}")
    if int(coordinates.shape[0]) <= 0:
        raise ValueError("spatial coordinates must contain at least one token")
    maxima = [int(value) for value in mx.max(coordinates.astype(mx.int32), axis=0).tolist()]
    return tuple(value + 1 for value in maxima)


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
