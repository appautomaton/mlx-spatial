"""Fixture-backed MLX heads for HY-World-2.0 WorldMirror."""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from .hyworld2_worldmirror import HyWorld2BackboneOutput


@dataclass(frozen=True)
class HyWorld2HeadBlocker:
    stage: str
    operation: str
    reason: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CameraHeadConfig:
    dim_in: int | None = None
    output_dim: int = 9
    steps: int = 4
    refine_depth: int = 4
    num_heads: int = 16
    trans_activation: str = "linear"
    quat_activation: str = "linear"
    focal_activation: str = "relu"
    layer_norm_eps: float = 1e-5
    adaptive_layer_norm_eps: float = 1e-6


@dataclass(frozen=True)
class DPTHeadConfig:
    head_type: str = "points"
    dim_in: int | None = None
    patch_size: int = 14
    attr_channels: int = 3
    activation: str = "inv_log+expp1"
    down_ratio: int = 1
    required_feature_levels: int = 4
    enable_depth_mask: bool = False

    @property
    def output_dim(self) -> int:
        return self.attr_channels + 1 + int(self.enable_depth_mask)


@dataclass(frozen=True)
class HyWorld2CameraHeadOutput:
    camera_params: mx.array | None = None
    blocker: HyWorld2HeadBlocker | None = None

    @property
    def ready(self) -> bool:
        return self.camera_params is not None and self.blocker is None


@dataclass(frozen=True)
class HyWorld2DenseHeadOutput:
    head_type: str
    values: mx.array | None = None
    confidence: mx.array | None = None
    depth_mask_logits: mx.array | None = None
    blocker: HyWorld2HeadBlocker | None = None

    @property
    def ready(self) -> bool:
        return self.values is not None and self.confidence is not None and self.blocker is None


@dataclass(frozen=True)
class GaussianAttributeConfig:
    feature_channels: int = 128
    required_feature_levels: int = 4
    enable_depth_mask: bool = False


@dataclass(frozen=True)
class HyWorld2GaussianAttributeOutput:
    features: mx.array | None = None
    depth: mx.array | None = None
    confidence: mx.array | None = None
    depth_mask_logits: mx.array | None = None
    raw_params: mx.array | None = None
    blocker: HyWorld2HeadBlocker | None = None

    @property
    def ready(self) -> bool:
        return (
            self.features is not None
            and self.depth is not None
            and self.confidence is not None
            and self.raw_params is not None
            and self.blocker is None
        )


@dataclass(frozen=True)
class HyWorld2AttentionLikeOutput:
    hidden_states: mx.array | None = None
    blocker: HyWorld2HeadBlocker | None = None


def default_camera_head_tensors(config: CameraHeadConfig) -> dict[str, mx.array]:
    dim_in = _require_dim(config.dim_in, "CameraHeadConfig.dim_in")
    return {
        "camera.norm.weight": mx.ones((dim_in,), dtype=mx.float32),
        "camera.norm.bias": mx.zeros((dim_in,), dtype=mx.float32),
        "camera.output.weight": mx.zeros((config.output_dim, dim_in), dtype=mx.float32),
        "camera.output.bias": mx.zeros((config.output_dim,), dtype=mx.float32),
    }


def default_dpt_head_tensors(config: DPTHeadConfig) -> dict[str, mx.array]:
    return {
        "dense.level_scales": mx.ones((config.required_feature_levels,), dtype=mx.float32),
        "dense.output.weight": mx.ones((config.output_dim,), dtype=mx.float32),
        "dense.output.bias": mx.zeros((config.output_dim,), dtype=mx.float32),
    }


def apply_hyworld2_activation(
    values: mx.array,
    activation: str,
    *,
    eps: float = 1e-8,
) -> tuple[mx.array | None, HyWorld2HeadBlocker | None]:
    if activation == "linear":
        return values, None
    if activation == "exp":
        return mx.exp(values), None
    if activation == "expp1":
        return 1.0 + mx.exp(values), None
    if activation == "inv_log":
        return mx.sign(values) * mx.expm1(mx.abs(values)), None
    if activation == "norm":
        norm = mx.sqrt(mx.sum(values * values, axis=-1, keepdims=True))
        return values / norm, None
    if activation == "relu":
        return mx.maximum(values, mx.zeros_like(values)), None
    return None, _blocker(
        "model-head",
        "HY-World head activation",
        f"unknown HY-World head activation: {activation}",
        {"activation": activation},
    )


def run_camera_head(
    backbone: HyWorld2BackboneOutput,
    config: CameraHeadConfig,
    tensors: dict[str, mx.array] | None = None,
) -> HyWorld2CameraHeadOutput:
    blocker = _validate_backbone_for_camera(backbone)
    if blocker is not None:
        return HyWorld2CameraHeadOutput(blocker=blocker)
    camera_tokens, token_dim, token_blocker = _camera_tokens_from_backbone(backbone)
    if token_blocker is not None:
        return HyWorld2CameraHeadOutput(blocker=token_blocker)
    assert camera_tokens is not None
    batch, frames, _ = tuple(int(dim) for dim in camera_tokens.shape)

    effective_config = config if config.dim_in is not None else CameraHeadConfig(**{**config.__dict__, "dim_in": token_dim})
    if effective_config.output_dim != 9:
        return HyWorld2CameraHeadOutput(
            blocker=_blocker(
                "camera-head",
                "HY-World camera output dimension validation",
                f"camera output_dim must be 9, got {effective_config.output_dim}",
                {"output_dim": effective_config.output_dim},
            )
        )
    if tensors is None:
        tensors = default_camera_head_tensors(effective_config)
    if _has_official_camera_tensors(tensors):
        tensor_blocker = _validate_official_camera_tensors(tensors, token_dim, effective_config)
        if tensor_blocker is not None:
            return HyWorld2CameraHeadOutput(blocker=tensor_blocker)
        return _run_official_camera_head(camera_tokens, effective_config, tensors)

    tensor_blocker = _validate_camera_tensors(tensors, token_dim, effective_config)
    if tensor_blocker is not None:
        return HyWorld2CameraHeadOutput(blocker=tensor_blocker)

    normalized = _layer_norm(
        camera_tokens,
        tensors["camera.norm.weight"],
        tensors["camera.norm.bias"],
        eps=effective_config.layer_norm_eps,
    )
    delta = _linear(normalized, tensors["camera.output.weight"], tensors["camera.output.bias"])
    params = mx.zeros((batch, frames, effective_config.output_dim), dtype=mx.float32)
    for _ in range(effective_config.steps):
        params = params + delta
    activated, activation_blocker = _activate_camera_params(params, effective_config)
    if activation_blocker is not None:
        return HyWorld2CameraHeadOutput(blocker=activation_blocker)
    return HyWorld2CameraHeadOutput(camera_params=activated)


def run_dpt_head(
    backbone: HyWorld2BackboneOutput,
    images: mx.array,
    config: DPTHeadConfig,
    tensors: dict[str, mx.array] | None = None,
    frames_chunk_size: int | None = None,
) -> HyWorld2DenseHeadOutput:
    blocker = _validate_dpt_inputs(backbone, images, config, frames_chunk_size)
    if blocker is not None:
        return HyWorld2DenseHeadOutput(head_type=config.head_type, blocker=blocker)
    if tensors is None:
        tensors = default_dpt_head_tensors(config)
    tensor_blocker = (
        _validate_official_dpt_tensors(tensors, config)
        if _has_official_dpt_tensors(tensors)
        else _validate_dpt_tensors(tensors, config)
    )
    if tensor_blocker is not None:
        return HyWorld2DenseHeadOutput(head_type=config.head_type, blocker=tensor_blocker)

    frames = int(images.shape[1])
    chunk_size = frames if frames_chunk_size is None or frames_chunk_size >= frames else frames_chunk_size
    value_chunks: list[mx.array] = []
    conf_chunks: list[mx.array] = []
    mask_chunks: list[mx.array] = []
    for frame_start in range(0, frames, chunk_size):
        frame_end = min(frame_start + chunk_size, frames)
        chunk_backbone = HyWorld2BackboneOutput(
            tokens=backbone.tokens,
            intermediate_tokens=tuple(
                level[:, frame_start:frame_end, :, :]
                for level in backbone.intermediate_tokens[: config.required_feature_levels]
            ),
            patch_start_idx=backbone.patch_start_idx,
            patch_grid=backbone.patch_grid,
            frame_token_count=backbone.frame_token_count,
            attention_modes=backbone.attention_modes,
        )
        chunk = _run_dpt_head_unchunked(
            chunk_backbone,
            images[:, frame_start:frame_end, :, :, :],
            config,
            tensors,
        )
        if chunk.blocker is not None or chunk.values is None or chunk.confidence is None:
            return chunk
        eval_targets = [chunk.values, chunk.confidence]
        if chunk.depth_mask_logits is not None:
            eval_targets.append(chunk.depth_mask_logits)
        mx.eval(*eval_targets)
        value_chunks.append(chunk.values)
        conf_chunks.append(chunk.confidence)
        if chunk.depth_mask_logits is not None:
            mask_chunks.append(chunk.depth_mask_logits)

    return HyWorld2DenseHeadOutput(
        head_type=config.head_type,
        values=mx.concatenate(value_chunks, axis=1),
        confidence=mx.concatenate(conf_chunks, axis=1),
        depth_mask_logits=mx.concatenate(mask_chunks, axis=1) if mask_chunks else None,
    )


def run_gaussian_attribute_head(
    backbone: HyWorld2BackboneOutput,
    images: mx.array,
    config: GaussianAttributeConfig,
    tensors: dict[str, mx.array] | None = None,
    renderer_tensors: dict[str, mx.array] | None = None,
    frames_chunk_size: int | None = None,
) -> HyWorld2GaussianAttributeOutput:
    if config.feature_channels <= 0:
        return HyWorld2GaussianAttributeOutput(
            blocker=_blocker(
                "gaussian-head",
                "HY-World Gaussian feature channel validation",
                f"feature_channels must be positive, got {config.feature_channels}",
                {"feature_channels": config.feature_channels},
            )
        )
    if tensors is not None and _has_official_dpt_tensors(tensors):
        return _run_official_gaussian_attribute_head(
            backbone,
            images,
            config,
            tensors,
            renderer_tensors,
            frames_chunk_size=frames_chunk_size,
        )

    depth = run_dpt_head(
        backbone,
        images,
        DPTHeadConfig(
            head_type="gaussian",
            attr_channels=1,
            activation="exp+expp1+linear" if config.enable_depth_mask else "exp+expp1",
            required_feature_levels=config.required_feature_levels,
            enable_depth_mask=config.enable_depth_mask,
        ),
        frames_chunk_size=frames_chunk_size,
    )
    if depth.blocker is not None:
        return HyWorld2GaussianAttributeOutput(blocker=_as_gaussian_blocker(depth.blocker))
    if depth.values is None or depth.confidence is None:
        return HyWorld2GaussianAttributeOutput(
            blocker=_blocker(
                "gaussian-head",
                "HY-World Gaussian depth attribute execution",
                "Gaussian DPT head returned no depth attributes",
                {},
            )
        )

    channel_scale = mx.reshape(
        mx.arange(1, config.feature_channels + 1, dtype=mx.float32),
        (1, 1, 1, 1, config.feature_channels),
    )
    feature_map = depth.values * channel_scale
    features = mx.transpose(feature_map, (0, 1, 4, 2, 3))
    mx.eval(features, depth.values, depth.confidence)
    if depth.depth_mask_logits is not None:
        mx.eval(depth.depth_mask_logits)
    raw_params = _default_gaussian_raw_params(feature_map)
    return HyWorld2GaussianAttributeOutput(
        features=features,
        depth=depth.values,
        confidence=depth.confidence,
        depth_mask_logits=depth.depth_mask_logits,
        raw_params=raw_params,
    )


def _run_official_camera_head(
    camera_tokens: mx.array,
    config: CameraHeadConfig,
    tensors: dict[str, mx.array],
) -> HyWorld2CameraHeadOutput:
    cam_tokens = _layer_norm(
        camera_tokens,
        tensors["token_norm.weight"],
        tensors["token_norm.bias"],
        eps=config.layer_norm_eps,
    )
    batch, frames, _ = tuple(int(dim) for dim in cam_tokens.shape)
    current: mx.array | None = None
    for _ in range(config.steps):
        pred_input = (
            mx.broadcast_to(tensors["init_token"], (batch, frames, config.output_dim))
            if current is None
            else mx.stop_gradient(current)
        )
        net_input = _linear(pred_input, tensors["param_embed.weight"], tensors["param_embed.bias"]).astype(cam_tokens.dtype)
        adapt = _linear(_silu(net_input), tensors["adapt_norm_gen.1.weight"], tensors["adapt_norm_gen.1.bias"])
        split = int(adapt.shape[-1]) // 3
        shift = adapt[..., :split]
        scale = adapt[..., split : 2 * split]
        gate = adapt[..., 2 * split :]
        adaptive_norm = _layer_norm(
            cam_tokens,
            mx.ones_like(tensors["token_norm.weight"]),
            mx.zeros_like(tensors["token_norm.bias"]),
            eps=config.adaptive_layer_norm_eps,
        )
        refined = gate * (adaptive_norm * (1.0 + scale) + shift) + cam_tokens
        for block_index in range(config.refine_depth):
            block = _run_camera_refine_block(refined, config, tensors, block_index)
            if block.blocker is not None or block.hidden_states is None:
                return HyWorld2CameraHeadOutput(blocker=block.blocker)
            refined = block.hidden_states
        refined = _layer_norm(
            refined,
            tensors["out_norm.weight"],
            tensors["out_norm.bias"],
            eps=config.layer_norm_eps,
        )
        delta = _linear(
            nn.gelu(
                _linear(
                    refined.astype(mx.float32),
                    tensors["param_predictor.fc1.weight"],
                    tensors["param_predictor.fc1.bias"],
                )
            ),
            tensors["param_predictor.fc2.weight"],
            tensors["param_predictor.fc2.bias"],
        )
        current = delta if current is None else current + delta
    assert current is not None
    activated, activation_blocker = _activate_camera_params(current, config)
    if activation_blocker is not None:
        return HyWorld2CameraHeadOutput(blocker=activation_blocker)
    return HyWorld2CameraHeadOutput(camera_params=activated)


def _run_camera_refine_block(
    hidden_states: mx.array,
    config: CameraHeadConfig,
    tensors: dict[str, mx.array],
    block_index: int,
) -> HyWorld2AttentionLikeOutput:
    prefix = f"refine_net.{block_index}"
    required = (
        f"{prefix}.norm1.weight",
        f"{prefix}.norm1.bias",
        f"{prefix}.norm2.weight",
        f"{prefix}.norm2.bias",
        f"{prefix}.attn.qkv.weight",
        f"{prefix}.attn.qkv.bias",
        f"{prefix}.attn.proj.weight",
        f"{prefix}.attn.proj.bias",
        f"{prefix}.mlp.fc1.weight",
        f"{prefix}.mlp.fc1.bias",
        f"{prefix}.mlp.fc2.weight",
        f"{prefix}.mlp.fc2.bias",
    )
    missing = tuple(key for key in required if key not in tensors)
    if missing:
        return HyWorld2AttentionLikeOutput(
            blocker=_blocker(
                "camera-head",
                "HY-World official camera refine tensor lookup",
                f"missing tensor for official camera refine block: {missing[0]}",
                {"missing": missing, "block_index": block_index},
            )
        )

    residual = hidden_states
    normalized = _layer_norm(
        hidden_states,
        tensors[f"{prefix}.norm1.weight"],
        tensors[f"{prefix}.norm1.bias"],
        eps=config.layer_norm_eps,
    )
    attended = _camera_self_attention(normalized, config, tensors, prefix)
    if attended.blocker is not None or attended.hidden_states is None:
        return attended
    hidden_states = residual + _apply_layer_scale(attended.hidden_states, tensors.get(f"{prefix}.ls1.gamma"))

    residual = hidden_states
    normalized = _layer_norm(
        hidden_states,
        tensors[f"{prefix}.norm2.weight"],
        tensors[f"{prefix}.norm2.bias"],
        eps=config.layer_norm_eps,
    )
    mlp = _linear(
        nn.gelu(_linear(normalized, tensors[f"{prefix}.mlp.fc1.weight"], tensors[f"{prefix}.mlp.fc1.bias"])),
        tensors[f"{prefix}.mlp.fc2.weight"],
        tensors[f"{prefix}.mlp.fc2.bias"],
    )
    return HyWorld2AttentionLikeOutput(
        hidden_states=residual + _apply_layer_scale(mlp, tensors.get(f"{prefix}.ls2.gamma"))
    )


def _camera_self_attention(
    hidden_states: mx.array,
    config: CameraHeadConfig,
    tensors: dict[str, mx.array],
    prefix: str,
) -> HyWorld2AttentionLikeOutput:
    batch, token_count, dim = tuple(int(value) for value in hidden_states.shape)
    if dim % config.num_heads:
        return HyWorld2AttentionLikeOutput(
            blocker=_blocker(
                "camera-head",
                "HY-World official camera attention head dimension validation",
                f"dim={dim} is not divisible by num_heads={config.num_heads}",
                {"dim": dim, "num_heads": config.num_heads},
            )
        )
    head_dim = dim // config.num_heads
    qkv = _linear(hidden_states, tensors[f"{prefix}.attn.qkv.weight"], tensors[f"{prefix}.attn.qkv.bias"])
    query, key, value = (_camera_heads(part, config.num_heads) for part in mx.split(qkv, 3, axis=-1))
    scores = (query @ mx.transpose(key, (0, 1, 3, 2))) * (head_dim**-0.5)
    weights = mx.softmax(scores, axis=-1)
    attended = weights @ value
    merged = mx.reshape(mx.transpose(attended, (0, 2, 1, 3)), (batch, token_count, dim))
    return HyWorld2AttentionLikeOutput(
        hidden_states=_linear(merged, tensors[f"{prefix}.attn.proj.weight"], tensors[f"{prefix}.attn.proj.bias"])
    )


def _camera_heads(values: mx.array, num_heads: int) -> mx.array:
    batch, token_count, dim = tuple(int(value) for value in values.shape)
    return mx.transpose(mx.reshape(values, (batch, token_count, num_heads, dim // num_heads)), (0, 2, 1, 3))


def _run_official_gaussian_attribute_head(
    backbone: HyWorld2BackboneOutput,
    images: mx.array,
    config: GaussianAttributeConfig,
    tensors: dict[str, mx.array],
    renderer_tensors: dict[str, mx.array] | None,
    *,
    frames_chunk_size: int | None,
) -> HyWorld2GaussianAttributeOutput:
    dpt_config = DPTHeadConfig(
        head_type="gaussian",
        attr_channels=1,
        activation="exp+expp1+linear" if config.enable_depth_mask else "exp+expp1",
        required_feature_levels=config.required_feature_levels,
        enable_depth_mask=config.enable_depth_mask,
    )
    input_blocker = _validate_dpt_inputs(backbone, images, dpt_config, frames_chunk_size)
    if input_blocker is not None:
        return HyWorld2GaussianAttributeOutput(blocker=_as_gaussian_blocker(input_blocker))
    tensor_blocker = _validate_official_dpt_tensors(tensors, dpt_config) or _validate_official_gs_input_merger(tensors)
    if tensor_blocker is not None:
        return HyWorld2GaussianAttributeOutput(blocker=_as_gaussian_blocker(tensor_blocker))
    renderer_blocker = _validate_gaussian_renderer_tensors(renderer_tensors)
    if renderer_blocker is not None:
        return HyWorld2GaussianAttributeOutput(blocker=renderer_blocker)
    assert renderer_tensors is not None

    frames = int(images.shape[1])
    chunk_size = frames if frames_chunk_size is None or frames_chunk_size >= frames else frames_chunk_size
    feature_chunks: list[mx.array] = []
    depth_chunks: list[mx.array] = []
    conf_chunks: list[mx.array] = []
    mask_chunks: list[mx.array] = []
    raw_chunks: list[mx.array] = []
    for frame_start in range(0, frames, chunk_size):
        frame_end = min(frame_start + chunk_size, frames)
        chunk_backbone = HyWorld2BackboneOutput(
            tokens=backbone.tokens,
            intermediate_tokens=tuple(
                level[:, frame_start:frame_end, :, :]
                for level in backbone.intermediate_tokens[: config.required_feature_levels]
            ),
            patch_start_idx=backbone.patch_start_idx,
            patch_grid=backbone.patch_grid,
            frame_token_count=backbone.frame_token_count,
            attention_modes=backbone.attention_modes,
        )
        chunk = _run_official_gs_dpt_head_unchunked(
            chunk_backbone,
            images[:, frame_start:frame_end, :, :, :],
            dpt_config,
            tensors,
        )
        if chunk.blocker is not None or chunk.features is None or chunk.depth is None or chunk.confidence is None:
            return chunk
        raw = _run_gaussian_renderer_head(chunk.features, renderer_tensors)
        mx.eval(chunk.features, chunk.depth, chunk.confidence, raw)
        feature_chunks.append(chunk.features)
        depth_chunks.append(chunk.depth)
        conf_chunks.append(chunk.confidence)
        raw_chunks.append(raw)
        if chunk.depth_mask_logits is not None:
            mx.eval(chunk.depth_mask_logits)
            mask_chunks.append(chunk.depth_mask_logits)

    return HyWorld2GaussianAttributeOutput(
        features=mx.concatenate(feature_chunks, axis=1),
        depth=mx.concatenate(depth_chunks, axis=1),
        confidence=mx.concatenate(conf_chunks, axis=1),
        depth_mask_logits=mx.concatenate(mask_chunks, axis=1) if mask_chunks else None,
        raw_params=mx.concatenate(raw_chunks, axis=1),
    )


def _run_official_gs_dpt_head_unchunked(
    backbone: HyWorld2BackboneOutput,
    images: mx.array,
    config: DPTHeadConfig,
    tensors: dict[str, mx.array],
) -> HyWorld2GaussianAttributeOutput:
    assert backbone.patch_grid is not None
    batch, frames, _, height, width = tuple(int(dim) for dim in images.shape)
    patch_h, patch_w = backbone.patch_grid
    features: list[mx.array] = []
    for index, tokens in enumerate(backbone.intermediate_tokens[: config.required_feature_levels]):
        patch_tokens = mx.reshape(tokens, (batch * frames, patch_h * patch_w, int(tokens.shape[-1])))
        patch_tokens = _layer_norm(patch_tokens, tensors["norm.weight"], tensors["norm.bias"], eps=1e-5)
        feature = mx.reshape(
            mx.transpose(patch_tokens, (0, 2, 1)),
            (batch * frames, int(tokens.shape[-1]), patch_h, patch_w),
        )
        feature = _conv2d_nchw(
            feature,
            tensors[f"projects.{index}.weight"],
            tensors[f"projects.{index}.bias"],
            padding=0,
        )
        feature = _apply_dpt_pos_embed(feature, width=width, height=height)
        feature = _official_dpt_resize(feature, index, tensors)
        features.append(feature)

    fused = _official_dpt_scratch_forward(features, tensors)
    fused = _resize_nchw(
        fused,
        (
            int(patch_h * config.patch_size / config.down_ratio),
            int(patch_w * config.patch_size / config.down_ratio),
        ),
    )
    fused = _apply_dpt_pos_embed(fused, width=width, height=height)
    output = _conv2d_nchw(
        fused.astype(mx.float32),
        tensors["scratch.output_conv2.0.weight"],
        tensors["scratch.output_conv2.0.bias"],
    )
    output = mx.maximum(output, 0)
    output = _conv2d_nchw(
        output,
        tensors["scratch.output_conv2.2.weight"],
        tensors["scratch.output_conv2.2.bias"],
        padding=0,
    )
    raw = mx.reshape(
        mx.transpose(output, (0, 2, 3, 1)),
        (batch, frames, int(output.shape[2]), int(output.shape[3]), int(output.shape[1])),
    )
    activated = _activate_dense_output(raw, config)
    if activated.blocker is not None:
        return HyWorld2GaussianAttributeOutput(blocker=_as_gaussian_blocker(activated.blocker))
    if activated.values is None or activated.confidence is None:
        return HyWorld2GaussianAttributeOutput(
            blocker=_blocker(
                "gaussian-head",
                "HY-World official GS DPT activation",
                "official GS DPT head returned no depth attributes",
                {},
            )
        )

    flat_images = mx.reshape(images, (batch * frames, int(images.shape[2]), height, width))
    image_features = _conv2d_nchw(
        flat_images,
        tensors["input_merger.0.weight"],
        tensors["input_merger.0.bias"],
        padding=3,
    )
    fused = fused + mx.maximum(image_features, 0)
    return HyWorld2GaussianAttributeOutput(
        features=mx.reshape(fused, (batch, frames, int(fused.shape[1]), int(fused.shape[2]), int(fused.shape[3]))),
        depth=activated.values,
        confidence=activated.confidence,
        depth_mask_logits=activated.depth_mask_logits,
        raw_params=mx.zeros((batch, frames, 12, int(fused.shape[2]), int(fused.shape[3])), dtype=mx.float32),
    )


def _run_gaussian_renderer_head(
    features: mx.array,
    tensors: dict[str, mx.array],
) -> mx.array:
    batch, frames, channels, height, width = tuple(int(dim) for dim in features.shape)
    flat = mx.reshape(features, (batch * frames, channels, height, width))
    hidden = _conv2d_nchw(flat, tensors["gs_head.0.weight"], None, padding=1)
    hidden = mx.maximum(hidden, 0)
    raw = _conv2d_nchw(hidden, tensors["gs_head.2.weight"], tensors["gs_head.2.bias"], padding=0)
    return mx.reshape(raw, (batch, frames, int(raw.shape[1]), int(raw.shape[2]), int(raw.shape[3])))


def _default_gaussian_raw_params(feature_map: mx.array) -> mx.array:
    batch, frames, height, width, _ = tuple(int(dim) for dim in feature_map.shape)
    quats = mx.broadcast_to(mx.array([0.0, 0.0, 0.0, 1.0], dtype=mx.float32), (batch, frames, height, width, 4))
    scales = mx.full((batch, frames, height, width, 3), -7.0, dtype=mx.float32)
    opacities = mx.full((batch, frames, height, width, 1), -2.0, dtype=mx.float32)
    residual_sh = mx.zeros((batch, frames, height, width, 3), dtype=mx.float32)
    weights = mx.full((batch, frames, height, width, 1), -2.0, dtype=mx.float32)
    raw = mx.concatenate((quats, scales, opacities, residual_sh, weights), axis=-1)
    return mx.transpose(raw, (0, 1, 4, 2, 3))


def _run_dpt_head_unchunked(
    backbone: HyWorld2BackboneOutput,
    images: mx.array,
    config: DPTHeadConfig,
    tensors: dict[str, mx.array],
) -> HyWorld2DenseHeadOutput:
    if _has_official_dpt_tensors(tensors):
        return _run_official_dpt_head_unchunked(backbone, images, config, tensors)

    assert backbone.patch_grid is not None
    batch, frames, _, height, width = tuple(int(dim) for dim in images.shape)
    patch_h, patch_w = backbone.patch_grid
    fused: mx.array | None = None
    scales = tensors["dense.level_scales"]
    for index, tokens in enumerate(backbone.intermediate_tokens[: config.required_feature_levels]):
        patch_scalar = mx.mean(tokens, axis=-1, keepdims=True) * scales[index]
        level = mx.reshape(patch_scalar, (batch, frames, patch_h, patch_w, 1))
        fused = level if fused is None else fused + level
    assert fused is not None
    fused = fused / float(config.required_feature_levels)

    target_h = height // config.down_ratio
    target_w = width // config.down_ratio
    if target_h % patch_h or target_w % patch_w:
        return HyWorld2DenseHeadOutput(
            head_type=config.head_type,
            blocker=_blocker(
                f"{config.head_type}-head",
                "HY-World DPT output grid validation",
                "dense output size must be an integer multiple of the patch grid",
                {
                    "target_size": (target_h, target_w),
                    "patch_grid": (patch_h, patch_w),
                    "down_ratio": config.down_ratio,
                },
            ),
        )
    dense = mx.repeat(fused, repeats=target_h // patch_h, axis=2)
    dense = mx.repeat(dense, repeats=target_w // patch_w, axis=3)
    raw = dense * tensors["dense.output.weight"] + tensors["dense.output.bias"]

    activated = _activate_dense_output(raw, config)
    if activated.blocker is not None:
        return activated
    return activated


def _run_official_dpt_head_unchunked(
    backbone: HyWorld2BackboneOutput,
    images: mx.array,
    config: DPTHeadConfig,
    tensors: dict[str, mx.array],
) -> HyWorld2DenseHeadOutput:
    assert backbone.patch_grid is not None
    batch, frames, _, height, width = tuple(int(dim) for dim in images.shape)
    patch_h, patch_w = backbone.patch_grid
    features: list[mx.array] = []

    for index, tokens in enumerate(backbone.intermediate_tokens[: config.required_feature_levels]):
        patch_tokens = mx.reshape(tokens, (batch * frames, patch_h * patch_w, int(tokens.shape[-1])))
        patch_tokens = _layer_norm(
            patch_tokens,
            tensors["norm.weight"],
            tensors["norm.bias"],
            eps=1e-5,
        )
        feature = mx.reshape(
            mx.transpose(patch_tokens, (0, 2, 1)),
            (batch * frames, int(tokens.shape[-1]), patch_h, patch_w),
        )
        feature = _conv2d_nchw(
            feature,
            tensors[f"projects.{index}.weight"],
            tensors[f"projects.{index}.bias"],
            padding=0,
        )
        feature = _apply_dpt_pos_embed(feature, width=width, height=height)
        feature = _official_dpt_resize(feature, index, tensors)
        features.append(feature)

    fused = _official_dpt_scratch_forward(features, tensors)
    fused = _resize_nchw(
        fused,
        (
            int(patch_h * config.patch_size / config.down_ratio),
            int(patch_w * config.patch_size / config.down_ratio),
        ),
    )
    fused = _apply_dpt_pos_embed(fused, width=width, height=height)
    output = _conv2d_nchw(
        fused,
        tensors["scratch.output_conv2.0.weight"],
        tensors["scratch.output_conv2.0.bias"],
    )
    output = mx.maximum(output, 0)
    output = _conv2d_nchw(
        output,
        tensors["scratch.output_conv2.2.weight"],
        tensors["scratch.output_conv2.2.bias"],
        padding=0,
    )
    raw = mx.reshape(
        mx.transpose(output, (0, 2, 3, 1)),
        (batch, frames, int(output.shape[2]), int(output.shape[3]), int(output.shape[1])),
    )
    return _activate_dense_output(raw, config)


def _activate_dense_output(raw: mx.array, config: DPTHeadConfig) -> HyWorld2DenseHeadOutput:
    parts = config.activation.split("+")
    if config.enable_depth_mask:
        if len(parts) == 1:
            parts = [parts[0], "expp1", "linear"]
        if len(parts) != 3:
            return HyWorld2DenseHeadOutput(
                head_type=config.head_type,
                blocker=_blocker(
                    f"{config.head_type}-head",
                    "HY-World DPT activation parsing",
                    "depth-mask DPT activation must have attr+conf+mask parts",
                    {"activation": config.activation},
                ),
            )
        attr_act, conf_act, mask_act = parts
        attr_raw = raw[..., : config.attr_channels]
        conf_raw = raw[..., config.attr_channels]
        mask_raw = raw[..., config.attr_channels + 1]
    else:
        if len(parts) == 1:
            parts = [parts[0], "expp1"]
        if len(parts) != 2:
            return HyWorld2DenseHeadOutput(
                head_type=config.head_type,
                blocker=_blocker(
                    f"{config.head_type}-head",
                    "HY-World DPT activation parsing",
                    "DPT activation must have attr+conf parts",
                    {"activation": config.activation},
                ),
            )
        attr_act, conf_act = parts
        mask_act = None
        attr_raw = raw[..., : config.attr_channels]
        conf_raw = raw[..., config.attr_channels]
        mask_raw = None

    values, blocker = apply_hyworld2_activation(attr_raw, attr_act)
    if blocker is not None:
        return HyWorld2DenseHeadOutput(head_type=config.head_type, blocker=blocker)
    confidence, blocker = apply_hyworld2_activation(conf_raw, conf_act)
    if blocker is not None:
        return HyWorld2DenseHeadOutput(head_type=config.head_type, blocker=blocker)
    mask = None
    if mask_raw is not None and mask_act is not None:
        mask, blocker = apply_hyworld2_activation(mask_raw, mask_act)
        if blocker is not None:
            return HyWorld2DenseHeadOutput(head_type=config.head_type, blocker=blocker)
    return HyWorld2DenseHeadOutput(
        head_type=config.head_type,
        values=values,
        confidence=confidence,
        depth_mask_logits=mask,
    )


def _activate_camera_params(
    params: mx.array,
    config: CameraHeadConfig,
) -> tuple[mx.array | None, HyWorld2HeadBlocker | None]:
    parts: list[mx.array] = []
    for values, activation in (
        (params[..., :3], config.trans_activation),
        (params[..., 3:7], config.quat_activation),
        (params[..., 7:], config.focal_activation),
    ):
        activated, blocker = apply_hyworld2_activation(values, activation)
        if blocker is not None:
            return None, blocker
        assert activated is not None
        parts.append(activated)
    return mx.concatenate(parts, axis=-1), None


def _validate_backbone_for_camera(backbone: HyWorld2BackboneOutput) -> HyWorld2HeadBlocker | None:
    if backbone.blocker is not None:
        return _blocker(
            "camera-head",
            "HY-World camera backbone readiness",
            "backbone returned a blocker before camera execution",
            {"backbone_stage": backbone.blocker.stage, "backbone_reason": backbone.blocker.reason},
        )
    if backbone.frame_token_count is None:
        return _blocker(
            "camera-head",
            "HY-World camera token lookup",
            "camera head requires frame_token_count from the backbone",
            {},
        )
    if backbone.frame_token_count <= 0:
        return _blocker(
            "camera-head",
            "HY-World camera token framing",
            "frame_token_count must be positive",
            {"frame_token_count": backbone.frame_token_count},
        )
    if not backbone.intermediate_full_tokens and backbone.tokens is None:
        return _blocker(
            "camera-head",
            "HY-World camera token lookup",
            "camera head requires full intermediate tokens or final full backbone tokens",
            {},
        )
    return None


def _camera_tokens_from_backbone(
    backbone: HyWorld2BackboneOutput,
) -> tuple[mx.array | None, int | None, HyWorld2HeadBlocker | None]:
    assert backbone.frame_token_count is not None

    if backbone.intermediate_full_tokens:
        full_tokens = backbone.intermediate_full_tokens[-1]
        if full_tokens.ndim != 4:
            return None, None, _blocker(
                "camera-head",
                "HY-World camera full intermediate token shape validation",
                f"expected full intermediate tokens [B,S,N,C], got {tuple(full_tokens.shape)}",
                {"shape": tuple(full_tokens.shape)},
            )
        batch, frames, frame_token_count, token_dim = tuple(int(dim) for dim in full_tokens.shape)
        if frame_token_count != backbone.frame_token_count:
            return None, None, _blocker(
                "camera-head",
                "HY-World camera full intermediate token framing",
                "full intermediate frame token count must match backbone.frame_token_count",
                {
                    "expected_frame_token_count": backbone.frame_token_count,
                    "actual_frame_token_count": frame_token_count,
                    "shape": (batch, frames, frame_token_count, token_dim),
                },
            )
        return full_tokens[:, :, 0, :], token_dim, None

    assert backbone.tokens is not None
    batch, total_tokens, token_dim = tuple(int(dim) for dim in backbone.tokens.shape)
    if total_tokens % backbone.frame_token_count:
        return None, None, _blocker(
            "camera-head",
            "HY-World camera token framing",
            "final token count is not divisible by frame_token_count",
            {"tokens": total_tokens, "frame_token_count": backbone.frame_token_count},
        )
    frames = total_tokens // backbone.frame_token_count
    framed = mx.reshape(backbone.tokens, (batch, frames, backbone.frame_token_count, token_dim))
    return framed[:, :, 0, :], token_dim, None


def _validate_dpt_inputs(
    backbone: HyWorld2BackboneOutput,
    images: mx.array,
    config: DPTHeadConfig,
    frames_chunk_size: int | None,
) -> HyWorld2HeadBlocker | None:
    if backbone.blocker is not None:
        return _blocker(
            f"{config.head_type}-head",
            "HY-World DPT backbone readiness",
            "backbone returned a blocker before dense-head execution",
            {"backbone_stage": backbone.blocker.stage, "backbone_reason": backbone.blocker.reason},
        )
    if images.ndim != 5:
        return _blocker(
            f"{config.head_type}-head",
            "HY-World DPT image tensor validation",
            f"expected image tensor shape [B,S,3,H,W], got {tuple(images.shape)}",
            {"shape": tuple(images.shape)},
        )
    batch, frames, channels, height, width = tuple(int(dim) for dim in images.shape)
    if channels != 3:
        return _blocker(
            f"{config.head_type}-head",
            "HY-World DPT image tensor validation",
            f"expected RGB channel count 3, got {channels}",
            {"shape": tuple(images.shape)},
        )
    if frames <= 0:
        return _blocker(
            f"{config.head_type}-head",
            "HY-World DPT frame count validation",
            f"DPT head requires at least one frame, got {frames}",
            {"frames": frames, "shape": tuple(images.shape)},
        )
    if frames_chunk_size is not None and frames_chunk_size <= 0:
        return _blocker(
            f"{config.head_type}-head",
            "HY-World DPT frame chunk validation",
            f"frames_chunk_size must be a positive integer or None, got {frames_chunk_size}",
            {"frames_chunk_size": frames_chunk_size},
        )
    if config.required_feature_levels <= 0:
        return _blocker(
            f"{config.head_type}-head",
            "HY-World DPT feature level validation",
            f"required_feature_levels must be positive, got {config.required_feature_levels}",
            {"required_feature_levels": config.required_feature_levels},
        )
    if config.down_ratio <= 0:
        return _blocker(
            f"{config.head_type}-head",
            "HY-World DPT down_ratio validation",
            f"down_ratio must be positive, got {config.down_ratio}",
            {"down_ratio": config.down_ratio},
        )
    if height % config.down_ratio or width % config.down_ratio:
        return _blocker(
            f"{config.head_type}-head",
            "HY-World DPT output size validation",
            "image spatial size must be divisible by down_ratio",
            {"height": height, "width": width, "down_ratio": config.down_ratio},
        )
    if len(backbone.intermediate_tokens) < config.required_feature_levels:
        return _blocker(
            f"{config.head_type}-head",
            "HY-World DPT intermediate token lookup",
            (
                f"DPT head requires {config.required_feature_levels} intermediate feature levels, "
                f"got {len(backbone.intermediate_tokens)}"
            ),
            {
                "required_feature_levels": config.required_feature_levels,
                "actual_feature_levels": len(backbone.intermediate_tokens),
            },
        )
    if backbone.patch_grid is None:
        return _blocker(
            f"{config.head_type}-head",
            "HY-World DPT patch grid lookup",
            "DPT head requires patch_grid from the backbone",
            {},
        )
    patch_h, patch_w = backbone.patch_grid
    patch_count = patch_h * patch_w
    for index, tokens in enumerate(backbone.intermediate_tokens[: config.required_feature_levels]):
        if tokens.ndim != 4:
            return _blocker(
                f"{config.head_type}-head",
                "HY-World DPT intermediate token shape validation",
                f"expected patch-only intermediate tokens [B,S,P,C], got {tuple(tokens.shape)}",
                {"feature_level": index, "shape": tuple(tokens.shape)},
            )
        token_batch, token_frames, token_patches, _ = tuple(int(dim) for dim in tokens.shape)
        if (token_batch, token_frames, token_patches) != (batch, frames, patch_count):
            return _blocker(
                f"{config.head_type}-head",
                "HY-World DPT intermediate token shape validation",
                (
                    "patch-only intermediate tokens must match image batch, frame count, "
                    "and patch grid"
                ),
                {
                    "feature_level": index,
                    "expected": (batch, frames, patch_count),
                    "actual": (token_batch, token_frames, token_patches),
                },
            )
    return None


def _validate_camera_tensors(
    tensors: dict[str, mx.array],
    token_dim: int,
    config: CameraHeadConfig,
) -> HyWorld2HeadBlocker | None:
    expected = {
        "camera.norm.weight": (token_dim,),
        "camera.norm.bias": (token_dim,),
        "camera.output.weight": (config.output_dim, token_dim),
        "camera.output.bias": (config.output_dim,),
    }
    for key, shape in expected.items():
        if key not in tensors:
            return _blocker(
                "camera-head",
                "HY-World camera tensor lookup",
                f"missing tensor for camera head: {key}",
                {"missing": key},
            )
        if tuple(tensors[key].shape) != shape:
            return _blocker(
                "camera-head",
                "HY-World camera tensor shape validation",
                f"expected {key} shape {shape}, got {tuple(tensors[key].shape)}",
                {"key": key, "expected": shape, "actual": tuple(tensors[key].shape)},
            )
    return None


def _has_official_camera_tensors(tensors: dict[str, mx.array]) -> bool:
    return "token_norm.weight" in tensors and "param_predictor.fc2.weight" in tensors


def _validate_official_camera_tensors(
    tensors: dict[str, mx.array],
    token_dim: int,
    config: CameraHeadConfig,
) -> HyWorld2HeadBlocker | None:
    expected = {
        "token_norm.weight": (token_dim,),
        "token_norm.bias": (token_dim,),
        "out_norm.weight": (token_dim,),
        "out_norm.bias": (token_dim,),
        "init_token": (1, 1, config.output_dim),
        "param_embed.weight": (token_dim, config.output_dim),
        "param_embed.bias": (token_dim,),
        "adapt_norm_gen.1.weight": (3 * token_dim, token_dim),
        "adapt_norm_gen.1.bias": (3 * token_dim,),
        "param_predictor.fc1.weight": (token_dim // 2, token_dim),
        "param_predictor.fc1.bias": (token_dim // 2,),
        "param_predictor.fc2.weight": (config.output_dim, token_dim // 2),
        "param_predictor.fc2.bias": (config.output_dim,),
    }
    for index in range(config.refine_depth):
        prefix = f"refine_net.{index}"
        expected.update(
            {
                f"{prefix}.norm1.weight": (token_dim,),
                f"{prefix}.norm1.bias": (token_dim,),
                f"{prefix}.norm2.weight": (token_dim,),
                f"{prefix}.norm2.bias": (token_dim,),
                f"{prefix}.attn.qkv.weight": (3 * token_dim, token_dim),
                f"{prefix}.attn.qkv.bias": (3 * token_dim,),
                f"{prefix}.attn.proj.weight": (token_dim, token_dim),
                f"{prefix}.attn.proj.bias": (token_dim,),
                f"{prefix}.mlp.fc1.weight": (4 * token_dim, token_dim),
                f"{prefix}.mlp.fc1.bias": (4 * token_dim,),
                f"{prefix}.mlp.fc2.weight": (token_dim, 4 * token_dim),
                f"{prefix}.mlp.fc2.bias": (token_dim,),
            }
        )
    for key, shape in expected.items():
        if key not in tensors:
            return _blocker(
                "camera-head",
                "HY-World official camera tensor lookup",
                f"missing tensor for official camera head: {key}",
                {"missing": key},
            )
        if tuple(tensors[key].shape) != shape:
            return _blocker(
                "camera-head",
                "HY-World official camera tensor shape validation",
                f"expected {key} shape {shape}, got {tuple(tensors[key].shape)}",
                {"key": key, "expected": shape, "actual": tuple(tensors[key].shape)},
            )
    return None


def _validate_dpt_tensors(
    tensors: dict[str, mx.array],
    config: DPTHeadConfig,
) -> HyWorld2HeadBlocker | None:
    expected = {
        "dense.level_scales": (config.required_feature_levels,),
        "dense.output.weight": (config.output_dim,),
        "dense.output.bias": (config.output_dim,),
    }
    for key, shape in expected.items():
        if key not in tensors:
            return _blocker(
                f"{config.head_type}-head",
                "HY-World DPT tensor lookup",
                f"missing tensor for DPT head: {key}",
                {"missing": key},
            )
        if tuple(tensors[key].shape) != shape:
            return _blocker(
                f"{config.head_type}-head",
                "HY-World DPT tensor shape validation",
                f"expected {key} shape {shape}, got {tuple(tensors[key].shape)}",
                {"key": key, "expected": shape, "actual": tuple(tensors[key].shape)},
            )
    return None


def _has_official_dpt_tensors(tensors: dict[str, mx.array]) -> bool:
    return "projects.0.weight" in tensors and "scratch.output_conv2.2.weight" in tensors


def _validate_official_gs_input_merger(tensors: dict[str, mx.array]) -> HyWorld2HeadBlocker | None:
    expected = {
        "input_merger.0.weight": (128, 3, 7, 7),
        "input_merger.0.bias": (128,),
    }
    for key, shape in expected.items():
        if key not in tensors:
            return _blocker(
                "gaussian-head",
                "HY-World official GS input-merger tensor lookup",
                f"missing tensor for official GS DPT head: {key}",
                {"missing": key},
            )
        if tuple(tensors[key].shape) != shape:
            return _blocker(
                "gaussian-head",
                "HY-World official GS input-merger tensor shape validation",
                f"expected {key} shape {shape}, got {tuple(tensors[key].shape)}",
                {"key": key, "expected": shape, "actual": tuple(tensors[key].shape)},
            )
    return None


def _validate_gaussian_renderer_tensors(
    tensors: dict[str, mx.array] | None,
) -> HyWorld2HeadBlocker | None:
    if tensors is None:
        return _blocker(
            "gaussian-head",
            "HY-World Gaussian renderer tensor lookup",
            "official Gaussian export requires gs_renderer checkpoint tensors",
            {"missing": "gs_renderer"},
        )
    expected = {
        "gs_head.0.weight": (256, 128, 3, 3),
        "gs_head.2.weight": (12, 256, 1, 1),
        "gs_head.2.bias": (12,),
    }
    for key, shape in expected.items():
        if key not in tensors:
            return _blocker(
                "gaussian-head",
                "HY-World Gaussian renderer tensor lookup",
                f"missing tensor for Gaussian renderer: {key}",
                {"missing": key},
            )
        if tuple(tensors[key].shape) != shape:
            return _blocker(
                "gaussian-head",
                "HY-World Gaussian renderer tensor shape validation",
                f"expected {key} shape {shape}, got {tuple(tensors[key].shape)}",
                {"key": key, "expected": shape, "actual": tuple(tensors[key].shape)},
            )
    return None


def _validate_official_dpt_tensors(
    tensors: dict[str, mx.array],
    config: DPTHeadConfig,
) -> HyWorld2HeadBlocker | None:
    required = [
        "norm.weight",
        "norm.bias",
        "scratch.output_conv1.weight",
        "scratch.output_conv1.bias",
        "scratch.output_conv2.0.weight",
        "scratch.output_conv2.0.bias",
        "scratch.output_conv2.2.weight",
        "scratch.output_conv2.2.bias",
    ]
    for index in range(config.required_feature_levels):
        required.extend((f"projects.{index}.weight", f"projects.{index}.bias"))
    required.extend(
        (
            "resize_layers.0.weight",
            "resize_layers.0.bias",
            "resize_layers.1.weight",
            "resize_layers.1.bias",
            "resize_layers.3.weight",
            "resize_layers.3.bias",
            "scratch.layer1_rn.weight",
            "scratch.layer2_rn.weight",
            "scratch.layer3_rn.weight",
            "scratch.layer4_rn.weight",
        )
    )
    for level in range(1, 5):
        required.extend((f"scratch.refinenet{level}.out_conv.weight", f"scratch.refinenet{level}.out_conv.bias"))
        units = (("resConfUnit2",), ("resConfUnit1", "resConfUnit2"))[level < 4]
        for unit in units:
            for conv in ("conv1", "conv2"):
                required.extend(
                    (
                        f"scratch.refinenet{level}.{unit}.{conv}.weight",
                        f"scratch.refinenet{level}.{unit}.{conv}.bias",
                    )
                )
    missing = tuple(key for key in required if key not in tensors)
    if missing:
        return _blocker(
            f"{config.head_type}-head",
            "HY-World official DPT tensor lookup",
            f"missing tensor for official DPT head: {missing[0]}",
            {"missing": missing[:8]},
        )
    output_channels = int(tensors["scratch.output_conv2.2.bias"].shape[0])
    if output_channels != config.output_dim:
        return _blocker(
            f"{config.head_type}-head",
            "HY-World official DPT output shape validation",
            f"expected official DPT output_dim {config.output_dim}, got {output_channels}",
            {"expected": config.output_dim, "actual": output_channels},
        )
    return None


def _official_dpt_resize(x: mx.array, index: int, tensors: dict[str, mx.array]) -> mx.array:
    if index == 0:
        return _conv_transpose2d_nchw(
            x,
            tensors["resize_layers.0.weight"],
            tensors["resize_layers.0.bias"],
            stride=4,
            padding=0,
        )
    if index == 1:
        return _conv_transpose2d_nchw(
            x,
            tensors["resize_layers.1.weight"],
            tensors["resize_layers.1.bias"],
            stride=2,
            padding=0,
        )
    if index == 2:
        return x
    if index == 3:
        return _conv2d_nchw(
            x,
            tensors["resize_layers.3.weight"],
            tensors["resize_layers.3.bias"],
            stride=2,
            padding=1,
        )
    raise ValueError(f"unsupported DPT resize layer index: {index}")


def _official_dpt_scratch_forward(features: list[mx.array], tensors: dict[str, mx.array]) -> mx.array:
    layer1 = _conv2d_nchw(features[0], tensors["scratch.layer1_rn.weight"], None)
    layer2 = _conv2d_nchw(features[1], tensors["scratch.layer2_rn.weight"], None)
    layer3 = _conv2d_nchw(features[2], tensors["scratch.layer3_rn.weight"], None)
    layer4 = _conv2d_nchw(features[3], tensors["scratch.layer4_rn.weight"], None)

    out = _feature_fusion_block(layer4, None, prefix="scratch.refinenet4", tensors=tensors, size=tuple(int(dim) for dim in layer3.shape[2:]))
    out = _feature_fusion_block(out, layer3, prefix="scratch.refinenet3", tensors=tensors, size=tuple(int(dim) for dim in layer2.shape[2:]))
    out = _feature_fusion_block(out, layer2, prefix="scratch.refinenet2", tensors=tensors, size=tuple(int(dim) for dim in layer1.shape[2:]))
    out = _feature_fusion_block(out, layer1, prefix="scratch.refinenet1", tensors=tensors)
    return _conv2d_nchw(out, tensors["scratch.output_conv1.weight"], tensors["scratch.output_conv1.bias"])


def _feature_fusion_block(
    x: mx.array,
    residual: mx.array | None,
    *,
    prefix: str,
    tensors: dict[str, mx.array],
    size: tuple[int, int] | None = None,
) -> mx.array:
    output = x
    if residual is not None:
        output = output + _residual_conv_unit(residual, prefix=f"{prefix}.resConfUnit1", tensors=tensors)
    output = _residual_conv_unit(output, prefix=f"{prefix}.resConfUnit2", tensors=tensors)
    if size is None:
        size = (int(output.shape[2]) * 2, int(output.shape[3]) * 2)
    output = _resize_nchw(output, size)
    return _conv2d_nchw(output, tensors[f"{prefix}.out_conv.weight"], tensors[f"{prefix}.out_conv.bias"], padding=0)


def _residual_conv_unit(x: mx.array, *, prefix: str, tensors: dict[str, mx.array]) -> mx.array:
    output = mx.maximum(x, 0)
    output = _conv2d_nchw(output, tensors[f"{prefix}.conv1.weight"], tensors[f"{prefix}.conv1.bias"])
    output = mx.maximum(output, 0)
    output = _conv2d_nchw(output, tensors[f"{prefix}.conv2.weight"], tensors[f"{prefix}.conv2.bias"])
    return output + x


def _apply_dpt_pos_embed(x: mx.array, *, width: int, height: int, ratio: float = 0.1) -> mx.array:
    channels = int(x.shape[1])
    if channels % 4:
        return x
    pos = _position_grid_to_embed(
        width=int(x.shape[3]),
        height=int(x.shape[2]),
        embed_dim=channels,
        aspect_ratio=width / height,
    )
    pos = mx.transpose(pos * ratio, (2, 0, 1))[None, :, :, :]
    return x + pos.astype(x.dtype)


def _position_grid_to_embed(*, width: int, height: int, embed_dim: int, aspect_ratio: float, omega_0: float = 100.0) -> mx.array:
    diag = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag
    span_y = 1.0 / diag
    xs = mx.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width)
    ys = mx.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height)
    grid_x = mx.broadcast_to(xs[None, :], (height, width))
    grid_y = mx.broadcast_to(ys[:, None], (height, width))
    omega = mx.arange(embed_dim // 4, dtype=mx.float32)
    omega = mx.exp(-mx.log(mx.array(float(omega_0), dtype=mx.float32)) * (omega / (embed_dim / 4.0)))
    flat_x = mx.reshape(grid_x, (-1, 1))
    flat_y = mx.reshape(grid_y, (-1, 1))
    out_x = flat_x * omega[None, :]
    out_y = flat_y * omega[None, :]
    emb = mx.concatenate((mx.sin(out_x), mx.cos(out_x), mx.sin(out_y), mx.cos(out_y)), axis=1)
    return mx.reshape(emb, (height, width, embed_dim))


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


def _require_dim(dim: int | None, name: str) -> int:
    if dim is None:
        raise ValueError(f"{name} is required for default tensor creation")
    return dim


def _linear(values: mx.array, weight: mx.array, bias: mx.array | None) -> mx.array:
    output = values @ mx.transpose(weight)
    if bias is not None:
        output = output + bias
    return output


def _silu(values: mx.array) -> mx.array:
    return values * mx.sigmoid(values)


def _apply_layer_scale(values: mx.array, gamma: mx.array | None) -> mx.array:
    if gamma is None:
        return values
    return values * gamma.astype(values.dtype)


def _layer_norm(values: mx.array, weight: mx.array, bias: mx.array, *, eps: float) -> mx.array:
    mean = mx.mean(values, axis=-1, keepdims=True)
    centered = values - mean
    variance = mx.mean(centered * centered, axis=-1, keepdims=True)
    return centered * mx.rsqrt(variance + eps) * weight + bias


def _blocker(
    stage: str,
    operation: str,
    reason: str,
    metadata: dict[str, object],
) -> HyWorld2HeadBlocker:
    return HyWorld2HeadBlocker(stage=stage, operation=operation, reason=reason, metadata=metadata)


def _as_gaussian_blocker(blocker: HyWorld2HeadBlocker) -> HyWorld2HeadBlocker:
    if blocker.stage == "gaussian-head":
        return blocker
    return HyWorld2HeadBlocker(
        stage="gaussian-head",
        operation=blocker.operation,
        reason=blocker.reason,
        metadata=blocker.metadata,
    )
