"""MapAnything MLX prediction heads for inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import mlx.core as mx
import mlx.nn as nn

from .checkpoint import load_checkpoint_tensors
from .mapanything_assets import (
    MAPANYTHING_DEFAULT_ROOT,
    MapAnythingModelConfig,
    read_mapanything_model_config,
)


MAPANYTHING_HEAD_BASE_REQUIRED_KEYS = (
    "fusion_norm_layer.weight",
    "fusion_norm_layer.bias",
)
MAPANYTHING_DPT_INPUT_PROCESS_REQUIRED_KEYS = (
    "dense_head.0.input_process.0.0.0.weight",
    "dense_head.0.input_process.0.0.0.bias",
    "dense_head.0.input_process.0.0.1.weight",
    "dense_head.0.input_process.0.0.1.bias",
    "dense_head.0.input_process.0.1.weight",
    "dense_head.0.input_process.1.0.0.weight",
    "dense_head.0.input_process.1.0.0.bias",
    "dense_head.0.input_process.1.0.1.weight",
    "dense_head.0.input_process.1.0.1.bias",
    "dense_head.0.input_process.1.1.weight",
    "dense_head.0.input_process.2.0.0.weight",
    "dense_head.0.input_process.2.0.0.bias",
    "dense_head.0.input_process.2.1.weight",
    "dense_head.0.input_process.3.0.0.weight",
    "dense_head.0.input_process.3.0.0.bias",
    "dense_head.0.input_process.3.0.1.weight",
    "dense_head.0.input_process.3.0.1.bias",
    "dense_head.0.input_process.3.1.weight",
)
MAPANYTHING_DPT_REFINENET_REQUIRED_KEYS = tuple(
    key
    for block in ("refinenet1", "refinenet2", "refinenet3")
    for key in (
        f"dense_head.0.scratch.{block}.resConfUnit1.conv1.weight",
        f"dense_head.0.scratch.{block}.resConfUnit1.conv1.bias",
        f"dense_head.0.scratch.{block}.resConfUnit1.conv2.weight",
        f"dense_head.0.scratch.{block}.resConfUnit1.conv2.bias",
        f"dense_head.0.scratch.{block}.resConfUnit2.conv1.weight",
        f"dense_head.0.scratch.{block}.resConfUnit2.conv1.bias",
        f"dense_head.0.scratch.{block}.resConfUnit2.conv2.weight",
        f"dense_head.0.scratch.{block}.resConfUnit2.conv2.bias",
        f"dense_head.0.scratch.{block}.out_conv.weight",
        f"dense_head.0.scratch.{block}.out_conv.bias",
    )
) + (
    "dense_head.0.scratch.refinenet4.resConfUnit2.conv1.weight",
    "dense_head.0.scratch.refinenet4.resConfUnit2.conv1.bias",
    "dense_head.0.scratch.refinenet4.resConfUnit2.conv2.weight",
    "dense_head.0.scratch.refinenet4.resConfUnit2.conv2.bias",
    "dense_head.0.scratch.refinenet4.out_conv.weight",
    "dense_head.0.scratch.refinenet4.out_conv.bias",
)
MAPANYTHING_DPT_REGRESSOR_REQUIRED_KEYS = (
    "dense_head.1.conv1.weight",
    "dense_head.1.conv1.bias",
    "dense_head.1.conv2.0.weight",
    "dense_head.1.conv2.0.bias",
    "dense_head.1.conv2.2.weight",
    "dense_head.1.conv2.2.bias",
)
MAPANYTHING_POSE_HEAD_REQUIRED_KEYS = (
    "pose_head.proj.weight",
    "pose_head.proj.bias",
    "pose_head.res_conv.0.res_conv1.weight",
    "pose_head.res_conv.0.res_conv1.bias",
    "pose_head.res_conv.0.res_conv2.weight",
    "pose_head.res_conv.0.res_conv2.bias",
    "pose_head.res_conv.0.res_conv3.weight",
    "pose_head.res_conv.0.res_conv3.bias",
    "pose_head.res_conv.1.res_conv1.weight",
    "pose_head.res_conv.1.res_conv1.bias",
    "pose_head.res_conv.1.res_conv2.weight",
    "pose_head.res_conv.1.res_conv2.bias",
    "pose_head.res_conv.1.res_conv3.weight",
    "pose_head.res_conv.1.res_conv3.bias",
    "pose_head.more_mlps.0.weight",
    "pose_head.more_mlps.0.bias",
    "pose_head.more_mlps.2.weight",
    "pose_head.more_mlps.2.bias",
    "pose_head.fc_t.weight",
    "pose_head.fc_t.bias",
    "pose_head.fc_rot.weight",
    "pose_head.fc_rot.bias",
)
MAPANYTHING_SCALE_HEAD_REQUIRED_KEYS = (
    "scale_head.proj.weight",
    "scale_head.proj.bias",
    "scale_head.mlp.0.0.weight",
    "scale_head.mlp.0.0.bias",
    "scale_head.mlp.1.0.weight",
    "scale_head.mlp.1.0.bias",
    "scale_head.output_proj.weight",
    "scale_head.output_proj.bias",
)


@dataclass(frozen=True)
class MapAnythingHeadsConfig:
    """Inference config for MapAnything's image-only prediction heads."""

    input_feature_dim: int = 1536
    patch_size: int = 14
    layer_dims: tuple[int, int, int, int] = (96, 192, 384, 768)
    feature_dim: int = 256
    dense_output_dim: int = 6
    pose_resconv_blocks: int = 2
    pose_rot_dim: int = 4
    scale_hidden_dim: int = 196
    scale_output_dim: int = 1
    use_encoder_features_for_dpt: bool = True
    layer_norm_eps: float = 1e-6

    @property
    def pose_hidden_dim(self) -> int:
        return 4 * (self.patch_size**2)


@dataclass(frozen=True)
class MapAnythingDenseHeadOutput:
    """Adapted dense output from MapAnything's DPT head."""

    value: mx.array
    confidence: mx.array
    mask: mx.array
    logits: mx.array
    decoded_channels: mx.array


@dataclass(frozen=True)
class MapAnythingHeadsOutput:
    """Dense, pose, and scale head outputs."""

    dense: MapAnythingDenseHeadOutput
    pose_value: mx.array
    scale_value: mx.array
    trace: dict[str, object]

    @property
    def parity_tensors(self) -> dict[str, mx.array]:
        return {
            "head.dense.value": self.dense.value,
            "head.dense.confidence": self.dense.confidence,
            "head.dense.mask": self.dense.mask,
            "head.dense.logits": self.dense.logits,
            "head.pose.value": self.pose_value,
            "head.scale.value": self.scale_value,
        }


class MapAnythingHeads:
    """MLX implementation of MapAnything's fusion, dense, pose, and scale heads."""

    def __init__(
        self,
        weights: Mapping[str, mx.array],
        config: MapAnythingHeadsConfig | None = None,
    ) -> None:
        self.config = config or MapAnythingHeadsConfig()
        self.weights = dict(weights)
        validate_mapanything_heads_weights(self.weights, self.config)

    def apply_fusion_norm(self, features: Sequence[mx.array]) -> tuple[mx.array, ...]:
        return apply_mapanything_fusion_norm(features, self.weights, config=self.config)

    def __call__(
        self,
        dense_features: Sequence[mx.array],
        scale_tokens: mx.array,
        *,
        image_shape: tuple[int, int],
    ) -> MapAnythingHeadsOutput:
        return run_mapanything_heads(dense_features, scale_tokens, self.weights, image_shape=image_shape, config=self.config)


def mapanything_heads_config_from_model_config(model_config: MapAnythingModelConfig) -> MapAnythingHeadsConfig:
    """Build head config from the parsed official MapAnything config subset."""

    if model_config.pred_head_type != "dpt+pose":
        raise ValueError(f"only dpt+pose MapAnything heads are supported, got {model_config.pred_head_type!r}")
    if model_config.pred_head_adaptor_type != "raydirs+depth+pose+confidence+mask":
        raise ValueError(
            "only raydirs+depth+pose+confidence+mask adaptor is supported, "
            f"got {model_config.pred_head_adaptor_type!r}"
        )
    return MapAnythingHeadsConfig(
        input_feature_dim=model_config.info_sharing_dim,
        patch_size=model_config.patch_size,
        use_encoder_features_for_dpt=len(model_config.info_sharing_indices) == 2,
    )


def mapanything_heads_required_keys(config: MapAnythingHeadsConfig | None = None) -> tuple[str, ...]:
    """Return checkpoint keys required by the configured prediction heads."""

    _ = config or MapAnythingHeadsConfig()
    return (
        *MAPANYTHING_HEAD_BASE_REQUIRED_KEYS,
        *MAPANYTHING_DPT_INPUT_PROCESS_REQUIRED_KEYS,
        *MAPANYTHING_DPT_REFINENET_REQUIRED_KEYS,
        *MAPANYTHING_DPT_REGRESSOR_REQUIRED_KEYS,
        *MAPANYTHING_POSE_HEAD_REQUIRED_KEYS,
        *MAPANYTHING_SCALE_HEAD_REQUIRED_KEYS,
    )


def load_mapanything_heads_weights(
    root: str | Path = MAPANYTHING_DEFAULT_ROOT,
    *,
    config: MapAnythingHeadsConfig | None = None,
    dtype: mx.Dtype = mx.float32,
) -> dict[str, mx.array]:
    """Load MapAnything fusion, dense, pose, and scale head tensors."""

    root_path = Path(root)
    if config is None:
        model_config = read_mapanything_model_config(root_path / "config.json")
        config = mapanything_heads_config_from_model_config(model_config)
    required = mapanything_heads_required_keys(config)
    tensors = load_checkpoint_tensors(root_path / "model.safetensors", names=required)
    loaded = {name: tensors[name].astype(dtype) for name in required}
    validate_mapanything_heads_weights(loaded, config)
    return loaded


def validate_mapanything_heads_weights(
    weights: Mapping[str, mx.array],
    config: MapAnythingHeadsConfig | None = None,
) -> None:
    """Validate key presence and shapes for MapAnything prediction heads."""

    cfg = config or MapAnythingHeadsConfig()
    missing = tuple(name for name in mapanything_heads_required_keys(cfg) if name not in weights)
    if missing:
        raise ValueError(f"missing MapAnything head tensors: {missing}")

    expected: dict[str, tuple[int, ...]] = {
        "fusion_norm_layer.weight": (cfg.input_feature_dim,),
        "fusion_norm_layer.bias": (cfg.input_feature_dim,),
    }
    for index, channels in enumerate(cfg.layer_dims):
        expected[f"dense_head.0.input_process.{index}.0.0.weight"] = (
            channels,
            cfg.input_feature_dim,
            1,
            1,
        )
        expected[f"dense_head.0.input_process.{index}.0.0.bias"] = (channels,)
        expected[f"dense_head.0.input_process.{index}.1.weight"] = (
            cfg.feature_dim,
            channels,
            3,
            3,
        )
    expected["dense_head.0.input_process.0.0.1.weight"] = (cfg.layer_dims[0], cfg.layer_dims[0], 4, 4)
    expected["dense_head.0.input_process.0.0.1.bias"] = (cfg.layer_dims[0],)
    expected["dense_head.0.input_process.1.0.1.weight"] = (cfg.layer_dims[1], cfg.layer_dims[1], 2, 2)
    expected["dense_head.0.input_process.1.0.1.bias"] = (cfg.layer_dims[1],)
    expected["dense_head.0.input_process.3.0.1.weight"] = (cfg.layer_dims[3], cfg.layer_dims[3], 3, 3)
    expected["dense_head.0.input_process.3.0.1.bias"] = (cfg.layer_dims[3],)

    for block in ("refinenet1", "refinenet2", "refinenet3"):
        for unit in ("resConfUnit1", "resConfUnit2"):
            for conv in ("conv1", "conv2"):
                expected[f"dense_head.0.scratch.{block}.{unit}.{conv}.weight"] = (
                    cfg.feature_dim,
                    cfg.feature_dim,
                    3,
                    3,
                )
                expected[f"dense_head.0.scratch.{block}.{unit}.{conv}.bias"] = (cfg.feature_dim,)
        expected[f"dense_head.0.scratch.{block}.out_conv.weight"] = (cfg.feature_dim, cfg.feature_dim, 1, 1)
        expected[f"dense_head.0.scratch.{block}.out_conv.bias"] = (cfg.feature_dim,)
    for conv in ("conv1", "conv2"):
        expected[f"dense_head.0.scratch.refinenet4.resConfUnit2.{conv}.weight"] = (
            cfg.feature_dim,
            cfg.feature_dim,
            3,
            3,
        )
        expected[f"dense_head.0.scratch.refinenet4.resConfUnit2.{conv}.bias"] = (cfg.feature_dim,)
    expected["dense_head.0.scratch.refinenet4.out_conv.weight"] = (cfg.feature_dim, cfg.feature_dim, 1, 1)
    expected["dense_head.0.scratch.refinenet4.out_conv.bias"] = (cfg.feature_dim,)

    expected["dense_head.1.conv1.weight"] = (cfg.feature_dim // 2, cfg.feature_dim, 3, 3)
    expected["dense_head.1.conv1.bias"] = (cfg.feature_dim // 2,)
    expected["dense_head.1.conv2.0.weight"] = (cfg.feature_dim // 2, cfg.feature_dim // 2, 3, 3)
    expected["dense_head.1.conv2.0.bias"] = (cfg.feature_dim // 2,)
    expected["dense_head.1.conv2.2.weight"] = (cfg.dense_output_dim, cfg.feature_dim // 2, 1, 1)
    expected["dense_head.1.conv2.2.bias"] = (cfg.dense_output_dim,)

    pose_hidden = cfg.pose_hidden_dim
    expected["pose_head.proj.weight"] = (pose_hidden, cfg.input_feature_dim, 1, 1)
    expected["pose_head.proj.bias"] = (pose_hidden,)
    for block_index in range(cfg.pose_resconv_blocks):
        for conv in ("res_conv1", "res_conv2", "res_conv3"):
            expected[f"pose_head.res_conv.{block_index}.{conv}.weight"] = (pose_hidden, pose_hidden, 1, 1)
            expected[f"pose_head.res_conv.{block_index}.{conv}.bias"] = (pose_hidden,)
    expected["pose_head.more_mlps.0.weight"] = (pose_hidden, pose_hidden)
    expected["pose_head.more_mlps.0.bias"] = (pose_hidden,)
    expected["pose_head.more_mlps.2.weight"] = (pose_hidden, pose_hidden)
    expected["pose_head.more_mlps.2.bias"] = (pose_hidden,)
    expected["pose_head.fc_t.weight"] = (3, pose_hidden)
    expected["pose_head.fc_t.bias"] = (3,)
    expected["pose_head.fc_rot.weight"] = (cfg.pose_rot_dim, pose_hidden)
    expected["pose_head.fc_rot.bias"] = (cfg.pose_rot_dim,)

    expected["scale_head.proj.weight"] = (cfg.scale_hidden_dim, cfg.input_feature_dim)
    expected["scale_head.proj.bias"] = (cfg.scale_hidden_dim,)
    expected["scale_head.mlp.0.0.weight"] = (cfg.scale_hidden_dim, cfg.scale_hidden_dim)
    expected["scale_head.mlp.0.0.bias"] = (cfg.scale_hidden_dim,)
    expected["scale_head.mlp.1.0.weight"] = (cfg.scale_hidden_dim, cfg.scale_hidden_dim)
    expected["scale_head.mlp.1.0.bias"] = (cfg.scale_hidden_dim,)
    expected["scale_head.output_proj.weight"] = (cfg.scale_output_dim, cfg.scale_hidden_dim)
    expected["scale_head.output_proj.bias"] = (cfg.scale_output_dim,)

    for name, shape in expected.items():
        actual = tuple(int(dim) for dim in weights[name].shape)
        if actual != shape:
            raise ValueError(f"{name} has shape {actual}, expected {shape}")


def apply_mapanything_fusion_norm(
    features: Sequence[mx.array],
    weights: Mapping[str, mx.array],
    *,
    config: MapAnythingHeadsConfig | None = None,
) -> tuple[mx.array, ...]:
    """Apply the image-only MapAnything post-fusion LayerNorm to per-view features."""

    cfg = config or MapAnythingHeadsConfig()
    output = []
    for feature in features:
        if feature.ndim != 4:
            raise ValueError(f"features must have shape [B, C, H, W], got {tuple(feature.shape)}")
        if int(feature.shape[1]) != cfg.input_feature_dim:
            raise ValueError(f"feature channels must be {cfg.input_feature_dim}, got {int(feature.shape[1])}")
        nhwc = mx.transpose(feature.astype(mx.float32), (0, 2, 3, 1))
        normalized = _layer_norm_last(
            nhwc,
            weights["fusion_norm_layer.weight"],
            weights["fusion_norm_layer.bias"],
            eps=cfg.layer_norm_eps,
        )
        output.append(mx.transpose(normalized, (0, 3, 1, 2)))
    return tuple(output)


def run_mapanything_heads(
    dense_features: Sequence[mx.array],
    scale_tokens: mx.array,
    weights: Mapping[str, mx.array],
    *,
    image_shape: tuple[int, int],
    config: MapAnythingHeadsConfig | None = None,
) -> MapAnythingHeadsOutput:
    """Run DPT dense, pose, and scale heads with MapAnything adaptors."""

    cfg = config or MapAnythingHeadsConfig()
    validate_mapanything_heads_weights(weights, cfg)
    if len(dense_features) != 4:
        raise ValueError(f"dense_features must contain 4 BCHW tensors, got {len(dense_features)}")
    dense_features_tuple = tuple(feature.astype(mx.float32) for feature in dense_features)
    batch = int(dense_features_tuple[0].shape[0])
    for index, feature in enumerate(dense_features_tuple):
        if feature.ndim != 4:
            raise ValueError(f"dense feature {index} must have shape [B, C, H, W], got {tuple(feature.shape)}")
        if tuple(int(dim) for dim in feature.shape[:2]) != (batch, cfg.input_feature_dim):
            raise ValueError(
                f"dense feature {index} must have batch/channels {(batch, cfg.input_feature_dim)}, "
                f"got {tuple(int(dim) for dim in feature.shape[:2])}"
            )
    if scale_tokens.ndim != 3 or int(scale_tokens.shape[1]) != cfg.input_feature_dim:
        raise ValueError(f"scale_tokens must have shape [B, {cfg.input_feature_dim}, T], got {tuple(scale_tokens.shape)}")

    decoded_channels = _run_dpt_dense_head(dense_features_tuple, weights, image_shape=image_shape, config=cfg)
    dense = _adapt_dense_channels(decoded_channels)
    pose_value = _adapt_pose_value(_run_pose_head(dense_features_tuple[-1], weights, config=cfg))
    scale_value = _adapt_scale_value(_run_scale_head(scale_tokens.astype(mx.float32), weights))
    return MapAnythingHeadsOutput(
        dense=dense,
        pose_value=pose_value,
        scale_value=scale_value,
        trace={
            "stage": "prediction-heads",
            "runtime_depends_on_torch": False,
            "batch": batch,
            "image_shape": tuple(int(dim) for dim in image_shape),
            "dense_input_count": len(dense_features_tuple),
            "dense_output_type": "raydirs+depth+confidence+mask",
            "pose_output_type": "cam_trans+quaternion",
            "scale_output_type": "exp",
        },
    )


def mapanything_heads_outputs_for_parity(
    output: MapAnythingHeadsOutput,
    *,
    names: Sequence[str] | None = None,
) -> dict[str, mx.array]:
    """Return named head tensors for comparison with a reference bundle."""

    tensors = output.parity_tensors
    if names is None:
        return tensors
    return {name: tensors[name] for name in names if name in tensors}


def _run_dpt_dense_head(
    features: tuple[mx.array, mx.array, mx.array, mx.array],
    weights: Mapping[str, mx.array],
    *,
    image_shape: tuple[int, int],
    config: MapAnythingHeadsConfig,
) -> mx.array:
    layers = [_run_dpt_input_process(index, feature, weights) for index, feature in enumerate(features)]
    path_4 = _run_dpt_refinenet4(layers[3], weights)
    path_4 = path_4[:, :, : int(layers[2].shape[2]), : int(layers[2].shape[3])]
    path_3 = _run_dpt_refinenet(path_4, layers[2], weights, block="refinenet3")
    path_2 = _run_dpt_refinenet(path_3, layers[1], weights, block="refinenet2")
    upsampled = _run_dpt_refinenet(path_2, layers[0], weights, block="refinenet1")

    x = _conv2d_nchw(
        upsampled,
        weights["dense_head.1.conv1.weight"],
        weights["dense_head.1.conv1.bias"],
        padding=1,
    )
    x = _resize_nchw(x, image_shape)
    x = _conv2d_nchw(
        x,
        weights["dense_head.1.conv2.0.weight"],
        weights["dense_head.1.conv2.0.bias"],
        padding=1,
    )
    x = mx.maximum(x, 0)
    return _conv2d_nchw(
        x,
        weights["dense_head.1.conv2.2.weight"],
        weights["dense_head.1.conv2.2.bias"],
        padding=0,
    )


def _run_dpt_input_process(index: int, feature: mx.array, weights: Mapping[str, mx.array]) -> mx.array:
    prefix = f"dense_head.0.input_process.{index}"
    x = _conv2d_nchw(feature, weights[f"{prefix}.0.0.weight"], weights[f"{prefix}.0.0.bias"], padding=0)
    if index == 0:
        x = _conv_transpose2d_nchw(x, weights[f"{prefix}.0.1.weight"], weights[f"{prefix}.0.1.bias"], stride=4, padding=0)
    elif index == 1:
        x = _conv_transpose2d_nchw(x, weights[f"{prefix}.0.1.weight"], weights[f"{prefix}.0.1.bias"], stride=2, padding=0)
    elif index == 3:
        x = _conv2d_nchw(x, weights[f"{prefix}.0.1.weight"], weights[f"{prefix}.0.1.bias"], stride=2, padding=1)
    return _conv2d_nchw(x, weights[f"{prefix}.1.weight"], None, padding=1)


def _run_dpt_refinenet4(x: mx.array, weights: Mapping[str, mx.array]) -> mx.array:
    prefix = "dense_head.0.scratch.refinenet4"
    x = _run_residual_conv_unit(x, weights, f"{prefix}.resConfUnit2")
    x = _upsample_nchw(x, scale=2, align_corners=True)
    return _conv2d_nchw(x, weights[f"{prefix}.out_conv.weight"], weights[f"{prefix}.out_conv.bias"], padding=0)


def _run_dpt_refinenet(
    x: mx.array,
    residual: mx.array,
    weights: Mapping[str, mx.array],
    *,
    block: str,
) -> mx.array:
    prefix = f"dense_head.0.scratch.{block}"
    x = x + _run_residual_conv_unit(residual, weights, f"{prefix}.resConfUnit1")
    x = _run_residual_conv_unit(x, weights, f"{prefix}.resConfUnit2")
    x = _upsample_nchw(x, scale=2, align_corners=True)
    return _conv2d_nchw(x, weights[f"{prefix}.out_conv.weight"], weights[f"{prefix}.out_conv.bias"], padding=0)


def _run_residual_conv_unit(x: mx.array, weights: Mapping[str, mx.array], prefix: str) -> mx.array:
    residual = x
    x = mx.maximum(x, 0)
    x = _conv2d_nchw(x, weights[f"{prefix}.conv1.weight"], weights[f"{prefix}.conv1.bias"], padding=1)
    x = mx.maximum(x, 0)
    x = _conv2d_nchw(x, weights[f"{prefix}.conv2.weight"], weights[f"{prefix}.conv2.bias"], padding=1)
    return x + residual


def _adapt_dense_channels(decoded_channels: mx.array) -> MapAnythingDenseHeadOutput:
    ray_dirs, depth_raw, confidence_raw, mask_logits = mx.split(decoded_channels, [3, 4, 5], axis=1)
    ray_norm = mx.sqrt(mx.sum(ray_dirs * ray_dirs, axis=1, keepdims=True))
    ray_dirs = ray_dirs / mx.maximum(ray_norm, 1e-8)
    depth = mx.exp(depth_raw)
    confidence = 1.0 + mx.exp(confidence_raw)
    mask = mx.sigmoid(mask_logits)
    value = mx.concatenate((ray_dirs, depth), axis=1)
    return MapAnythingDenseHeadOutput(
        value=value,
        confidence=confidence,
        mask=mask,
        logits=mask_logits,
        decoded_channels=decoded_channels,
    )


def _run_pose_head(feature: mx.array, weights: Mapping[str, mx.array], *, config: MapAnythingHeadsConfig) -> mx.array:
    x = _conv2d_nchw(feature, weights["pose_head.proj.weight"], weights["pose_head.proj.bias"], padding=0)
    for block_index in range(config.pose_resconv_blocks):
        prefix = f"pose_head.res_conv.{block_index}"
        residual = x
        x = mx.maximum(_conv2d_nchw(x, weights[f"{prefix}.res_conv1.weight"], weights[f"{prefix}.res_conv1.bias"], padding=0), 0)
        x = mx.maximum(_conv2d_nchw(x, weights[f"{prefix}.res_conv2.weight"], weights[f"{prefix}.res_conv2.bias"], padding=0), 0)
        x = mx.maximum(_conv2d_nchw(x, weights[f"{prefix}.res_conv3.weight"], weights[f"{prefix}.res_conv3.bias"], padding=0), 0)
        x = residual + x
    x = mx.mean(x, axis=(2, 3))
    x = mx.maximum(_linear(x, weights["pose_head.more_mlps.0.weight"], weights["pose_head.more_mlps.0.bias"]), 0)
    x = mx.maximum(_linear(x, weights["pose_head.more_mlps.2.weight"], weights["pose_head.more_mlps.2.bias"]), 0)
    trans = _linear(x, weights["pose_head.fc_t.weight"], weights["pose_head.fc_t.bias"])
    rot = _linear(x, weights["pose_head.fc_rot.weight"], weights["pose_head.fc_rot.bias"])
    return mx.concatenate((trans, rot), axis=1)


def _adapt_pose_value(decoded: mx.array) -> mx.array:
    trans, quat = mx.split(decoded, [3], axis=1)
    quat_norm = mx.sqrt(mx.sum(quat * quat, axis=1, keepdims=True))
    quat = quat / mx.maximum(quat_norm, 1e-8)
    return mx.concatenate((trans, quat), axis=1)


def _run_scale_head(scale_tokens: mx.array, weights: Mapping[str, mx.array]) -> mx.array:
    x = mx.transpose(scale_tokens, (0, 2, 1))
    x = _linear(x, weights["scale_head.proj.weight"], weights["scale_head.proj.bias"])
    x = mx.maximum(_linear(x, weights["scale_head.mlp.0.0.weight"], weights["scale_head.mlp.0.0.bias"]), 0)
    x = mx.maximum(_linear(x, weights["scale_head.mlp.1.0.weight"], weights["scale_head.mlp.1.0.bias"]), 0)
    x = _linear(x, weights["scale_head.output_proj.weight"], weights["scale_head.output_proj.bias"])
    return mx.transpose(x, (0, 2, 1))


def _adapt_scale_value(decoded: mx.array) -> mx.array:
    return mx.squeeze(mx.exp(decoded), axis=-1)


def _conv2d_nchw(
    x: mx.array,
    weight: mx.array,
    bias: mx.array | None,
    *,
    stride: int = 1,
    padding: int = 1,
) -> mx.array:
    x_nhwc = mx.transpose(x, (0, 2, 3, 1))
    weight_ohwi = mx.transpose(weight.astype(x.dtype), (0, 2, 3, 1))
    output = mx.conv2d(x_nhwc, weight_ohwi, stride=stride, padding=padding)
    if bias is not None:
        output = output + bias.astype(output.dtype)[None, None, None, :]
    return mx.transpose(output, (0, 3, 1, 2))


def _conv_transpose2d_nchw(
    x: mx.array,
    weight: mx.array,
    bias: mx.array | None,
    *,
    stride: int,
    padding: int,
) -> mx.array:
    x_nhwc = mx.transpose(x, (0, 2, 3, 1))
    weight_ohwi = mx.transpose(weight.astype(x.dtype), (1, 2, 3, 0))
    output = mx.conv_transpose2d(x_nhwc, weight_ohwi, stride=stride, padding=padding)
    if bias is not None:
        output = output + bias.astype(output.dtype)[None, None, None, :]
    return mx.transpose(output, (0, 3, 1, 2))


def _upsample_nchw(x: mx.array, *, scale: int, align_corners: bool) -> mx.array:
    x_nhwc = mx.transpose(x, (0, 2, 3, 1))
    output = nn.Upsample(scale, mode="linear", align_corners=align_corners)(x_nhwc)
    return mx.transpose(output, (0, 3, 1, 2))


def _resize_nchw(x: mx.array, size: tuple[int, int]) -> mx.array:
    height, width = (int(dim) for dim in size)
    if int(x.shape[2]) == height and int(x.shape[3]) == width:
        return x
    scale = (height / int(x.shape[2]), width / int(x.shape[3]))
    x_nhwc = mx.transpose(x, (0, 2, 3, 1))
    output = nn.Upsample(scale, mode="linear", align_corners=True)(x_nhwc)
    if int(output.shape[1]) != height or int(output.shape[2]) != width:
        output = output[:, :height, :width, :]
    return mx.transpose(output, (0, 3, 1, 2))


def _linear(values: mx.array, weight: mx.array, bias: mx.array | None) -> mx.array:
    output = values @ mx.transpose(weight.astype(values.dtype))
    if bias is not None:
        output = output + bias.astype(output.dtype)
    return output


def _layer_norm_last(values: mx.array, weight: mx.array, bias: mx.array, *, eps: float) -> mx.array:
    values = values.astype(mx.float32)
    mean = mx.mean(values, axis=-1, keepdims=True)
    centered = values - mean
    variance = mx.mean(centered * centered, axis=-1, keepdims=True)
    return centered * mx.rsqrt(variance + eps) * weight.astype(values.dtype) + bias.astype(values.dtype)
