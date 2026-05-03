"""MLX DINOv3 forward probes for TRELLIS.2 image conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from .checkpoint import CheckpointTensorInfo, inspect_checkpoint, load_checkpoint_tensors
from .trellis2_dinov3 import (
    DinoV3CheckpointInventory,
    DinoV3ConditioningResult,
    DinoV3ModelConfig,
    DinoV3PortBlocker,
)


@dataclass(frozen=True)
class DinoV3ForwardKeyMap:
    checkpoint_path: Path
    required_keys: tuple[str, ...]
    present_keys: tuple[str, ...]


@dataclass(frozen=True)
class DinoV3ForwardTensorLoad:
    key_map: DinoV3ForwardKeyMap | None = None
    tensors: dict[str, mx.array] | None = None
    blocker: DinoV3PortBlocker | None = None

    @property
    def ready(self) -> bool:
        return self.key_map is not None and self.tensors is not None and self.blocker is None


@dataclass(frozen=True)
class DinoV3TokenAssembly:
    tokens: mx.array | None = None
    patch_grid: tuple[int, int] | None = None
    blocker: DinoV3PortBlocker | None = None

    @property
    def ready(self) -> bool:
        return self.tokens is not None and self.patch_grid is not None and self.blocker is None


@dataclass(frozen=True)
class DinoV3RopeProbe:
    patch_grid: tuple[int, int] | None = None
    patch_token_count: int | None = None
    head_dim: int | None = None
    theta: float | None = None
    blocker: DinoV3PortBlocker | None = None

    @property
    def ready(self) -> bool:
        return (
            self.patch_grid is not None
            and self.patch_token_count is not None
            and self.head_dim is not None
            and self.theta is not None
            and self.blocker is None
        )


@dataclass(frozen=True)
class DinoV3BlockForward:
    hidden_states: mx.array | None = None
    blocker: DinoV3PortBlocker | None = None

    @property
    def ready(self) -> bool:
        return self.hidden_states is not None and self.blocker is None


@dataclass(frozen=True)
class DinoV3LayerStackForward:
    hidden_states: mx.array | None = None
    completed_layers: int = 0
    blocker: DinoV3PortBlocker | None = None

    @property
    def ready(self) -> bool:
        return self.hidden_states is not None and self.blocker is None


def dinov3_forward_required_keys(config: DinoV3ModelConfig, *, layer_index: int = 0) -> tuple[str, ...]:
    """Return the DINOv3 keys required for the current forward probe."""

    keys = [
        "embeddings.cls_token",
        "embeddings.patch_embeddings.bias",
        "embeddings.patch_embeddings.weight",
        *_dinov3_layer_required_keys(layer_index),
    ]
    if config.num_register_tokens:
        keys.insert(1, "embeddings.register_tokens")
    return tuple(keys)


def dinov3_full_forward_required_keys(config: DinoV3ModelConfig) -> tuple[str, ...]:
    """Return the DINOv3 keys required for complete MLX conditioning forward."""

    keys = [
        "embeddings.cls_token",
        "embeddings.patch_embeddings.bias",
        "embeddings.patch_embeddings.weight",
    ]
    if config.num_register_tokens:
        keys.insert(1, "embeddings.register_tokens")
    for layer_index in range(config.num_hidden_layers):
        keys.extend(_dinov3_layer_required_keys(layer_index))
    return tuple(keys)


def inspect_dinov3_forward_key_map(
    checkpoint_path: str | Path,
    config: DinoV3ModelConfig,
    *,
    layer_index: int = 0,
    all_layers: bool = False,
) -> DinoV3ForwardTensorLoad:
    """Inspect the selected DINOv3 forward keys without loading tensors."""

    path = Path(checkpoint_path)
    required = (
        dinov3_full_forward_required_keys(config)
        if all_layers
        else dinov3_forward_required_keys(config, layer_index=layer_index)
    )
    try:
        infos = inspect_checkpoint(path, names=required)
    except (FileNotFoundError, ValueError) as error:
        return DinoV3ForwardTensorLoad(
            blocker=_blocker(
                "DINOv3 forward checkpoint key validation",
                str(path),
                str(error),
                "map all checkpoint keys required by the current DINOv3 forward probe",
            )
        )

    info_by_name = {info.name: info for info in infos}
    missing = tuple(key for key in required if key not in info_by_name)
    if missing:
        return DinoV3ForwardTensorLoad(
            blocker=_blocker(
                "DINOv3 forward checkpoint key validation",
                str(path),
                f"checkpoint is missing required DINOv3 forward key: {missing[0]}",
                "map all checkpoint keys required by the current DINOv3 forward probe",
            )
        )

    layer_indexes = range(config.num_hidden_layers) if all_layers else (layer_index,)
    blocker = _validate_selected_shapes(info_by_name, config, layer_indexes=layer_indexes)
    if blocker is not None:
        return DinoV3ForwardTensorLoad(blocker=blocker)

    return DinoV3ForwardTensorLoad(
        key_map=DinoV3ForwardKeyMap(
            checkpoint_path=path,
            required_keys=required,
            present_keys=tuple(info.name for info in infos),
        ),
        tensors={},
    )


def load_dinov3_forward_tensors(
    checkpoint_path: str | Path,
    config: DinoV3ModelConfig,
    *,
    layer_index: int = 0,
    all_layers: bool = False,
) -> DinoV3ForwardTensorLoad:
    """Load only the tensors required by the current DINOv3 forward probe."""

    inspected = inspect_dinov3_forward_key_map(
        checkpoint_path,
        config,
        layer_index=layer_index,
        all_layers=all_layers,
    )
    if inspected.blocker is not None or inspected.key_map is None:
        return inspected

    try:
        tensors = load_checkpoint_tensors(
            checkpoint_path,
            names=inspected.key_map.required_keys,
        )
    except (FileNotFoundError, ValueError, TypeError) as error:
        return DinoV3ForwardTensorLoad(
            key_map=inspected.key_map,
            blocker=_blocker(
                "DINOv3 forward checkpoint tensor loading",
                str(checkpoint_path),
                str(error),
                "load the selected DINOv3 forward checkpoint tensors with MLX",
            ),
        )

    return DinoV3ForwardTensorLoad(key_map=inspected.key_map, tensors=tensors)


def assemble_dinov3_tokens(
    image_tensor: mx.array,
    config: DinoV3ModelConfig,
    tensors: dict[str, mx.array],
) -> DinoV3TokenAssembly:
    """Run patch embedding and assemble cls/register/patch tokens."""

    if image_tensor.ndim != 4:
        return DinoV3TokenAssembly(
            blocker=_blocker(
                "DINOv3 image tensor validation",
                "prepare_dinov3_image_tensor",
                f"expected BCHW image tensor with 4 dimensions, got shape {tuple(image_tensor.shape)}",
                "provide a normalized BCHW image tensor before DINOv3 patch embedding",
            )
        )

    batch, channels, height, width = tuple(image_tensor.shape)
    if channels != 3:
        return DinoV3TokenAssembly(
            blocker=_blocker(
                "DINOv3 image tensor validation",
                "prepare_dinov3_image_tensor",
                f"expected RGB channel count 3, got {channels}",
                "provide a normalized RGB image tensor before DINOv3 patch embedding",
            )
        )
    if height % config.patch_size or width % config.patch_size:
        return DinoV3TokenAssembly(
            blocker=_blocker(
                "DINOv3 patch embedding shape validation",
                "prepare_dinov3_image_tensor",
                f"image spatial size {(height, width)} is not divisible by patch_size={config.patch_size}",
                "resize the DINOv3 input image to a patch-aligned resolution",
            )
        )

    try:
        weight = tensors["embeddings.patch_embeddings.weight"]
        bias = tensors["embeddings.patch_embeddings.bias"]
        cls_token = tensors["embeddings.cls_token"]
    except KeyError as error:
        return DinoV3TokenAssembly(
            blocker=_blocker(
                "DINOv3 patch embedding tensor lookup",
                "model.safetensors",
                f"missing tensor for patch/token assembly: {error.args[0]}",
                "load all embedding tensors before patch embedding",
            )
        )

    expected_weight = (config.hidden_size, 3, config.patch_size, config.patch_size)
    if tuple(weight.shape) != expected_weight:
        return DinoV3TokenAssembly(
            blocker=_blocker(
                "DINOv3 patch embedding shape validation",
                "embeddings.patch_embeddings.weight",
                f"expected patch embedding weight {expected_weight}, got {tuple(weight.shape)}",
                "align patch embedding checkpoint weights with config.hidden_size and config.patch_size",
            )
        )

    nhwc = mx.transpose(image_tensor, (0, 2, 3, 1))
    hwio_weight = mx.transpose(weight, (0, 2, 3, 1))
    embedded = mx.conv2d(nhwc, hwio_weight, stride=config.patch_size)
    embedded = embedded + bias
    patch_grid = (int(embedded.shape[1]), int(embedded.shape[2]))
    patch_tokens = mx.reshape(embedded, (batch, patch_grid[0] * patch_grid[1], config.hidden_size))

    token_parts = [mx.broadcast_to(cls_token, (batch, int(cls_token.shape[1]), config.hidden_size))]
    if config.num_register_tokens:
        try:
            register_tokens = tensors["embeddings.register_tokens"]
        except KeyError:
            return DinoV3TokenAssembly(
                blocker=_blocker(
                    "DINOv3 register token lookup",
                    "embeddings.register_tokens",
                    "config.num_register_tokens is non-zero but embeddings.register_tokens is missing",
                    "load DINOv3 register tokens before token assembly",
                )
            )
        token_parts.append(mx.broadcast_to(register_tokens, (batch, config.num_register_tokens, config.hidden_size)))
    token_parts.append(patch_tokens)

    return DinoV3TokenAssembly(tokens=mx.concatenate(token_parts, axis=1), patch_grid=patch_grid)


def probe_dinov3_rope(config: DinoV3ModelConfig, *, runtime_image_size: int | None = None) -> DinoV3RopeProbe:
    """Validate DINOv3 RoPE parameters and token geometry for the next forward slice."""

    theta = config.rope_theta
    if theta is None:
        return DinoV3RopeProbe(
            blocker=_blocker(
                "DINOv3 RoPE parameter validation",
                "config.json:rope_theta",
                "DINOv3 config is missing rope_theta for RoPE position embedding construction",
                "map Hugging Face DINOv3 RoPE parameters before attention execution",
            )
        )
    if config.hidden_size % config.num_attention_heads != 0:
        return DinoV3RopeProbe(
            blocker=_blocker(
                "DINOv3 RoPE head dimension validation",
                "config.json:hidden_size,num_attention_heads",
                (
                    f"hidden_size={config.hidden_size} is not divisible by "
                    f"num_attention_heads={config.num_attention_heads}"
                ),
                "resolve DINOv3 attention head dimensions before RoPE construction",
            )
        )

    image_size = runtime_image_size or config.image_size
    if image_size % config.patch_size:
        return DinoV3RopeProbe(
            blocker=_blocker(
                "DINOv3 RoPE grid validation",
                "prepare_dinov3_image_tensor",
                f"runtime image size {image_size} is not divisible by patch_size={config.patch_size}",
                "use a patch-aligned DINOv3 image-conditioning resolution",
            )
        )

    grid = image_size // config.patch_size
    return DinoV3RopeProbe(
        patch_grid=(grid, grid),
        patch_token_count=grid * grid,
        head_dim=config.hidden_size // config.num_attention_heads,
        theta=theta,
    )


def run_dinov3_transformer_block(
    hidden_states: mx.array,
    config: DinoV3ModelConfig,
    tensors: dict[str, mx.array],
    *,
    patch_grid: tuple[int, int],
    layer_index: int = 0,
) -> DinoV3BlockForward:
    """Execute one DINOv3 ViT transformer block with MLX arrays."""

    shape_blocker = _validate_hidden_states(hidden_states, config, patch_grid=patch_grid)
    if shape_blocker is not None:
        return DinoV3BlockForward(blocker=shape_blocker)

    if config.use_gated_mlp:
        return DinoV3BlockForward(
            blocker=_blocker(
                "DINOv3 gated MLP forward",
                "config.json:use_gated_mlp",
                "DINOv3 gated MLP checkpoints require gate_proj tensors, which are not mapped by this slice",
                "map gated MLP checkpoint keys before transformer block execution",
            )
        )

    layer = f"layer.{layer_index}"
    required = (
        f"{layer}.attention.k_proj.weight",
        f"{layer}.attention.o_proj.bias",
        f"{layer}.attention.o_proj.weight",
        f"{layer}.attention.q_proj.bias",
        f"{layer}.attention.q_proj.weight",
        f"{layer}.attention.v_proj.bias",
        f"{layer}.attention.v_proj.weight",
        f"{layer}.layer_scale1.lambda1",
        f"{layer}.layer_scale2.lambda1",
        f"{layer}.mlp.down_proj.bias",
        f"{layer}.mlp.down_proj.weight",
        f"{layer}.mlp.up_proj.bias",
        f"{layer}.mlp.up_proj.weight",
        f"{layer}.norm1.bias",
        f"{layer}.norm1.weight",
        f"{layer}.norm2.bias",
        f"{layer}.norm2.weight",
    )
    missing = tuple(key for key in required if key not in tensors)
    if missing:
        return DinoV3BlockForward(
            blocker=_blocker(
                "DINOv3 transformer block tensor lookup",
                "model.safetensors",
                f"missing tensor for DINOv3 transformer block: {missing[0]}",
                "load all selected DINOv3 transformer block tensors before block execution",
            )
        )

    residual = hidden_states
    normalized = _layer_norm(
        hidden_states,
        tensors[f"{layer}.norm1.weight"],
        tensors[f"{layer}.norm1.bias"],
        eps=config.layer_norm_eps,
    )
    attended = _self_attention(normalized, config, tensors, patch_grid=patch_grid, layer_index=layer_index)
    if attended.blocker is not None or attended.hidden_states is None:
        return attended

    hidden_states = residual + attended.hidden_states * tensors[f"{layer}.layer_scale1.lambda1"]

    residual = hidden_states
    normalized = _layer_norm(
        hidden_states,
        tensors[f"{layer}.norm2.weight"],
        tensors[f"{layer}.norm2.bias"],
        eps=config.layer_norm_eps,
    )
    mlp_up = _linear(
        normalized,
        tensors[f"{layer}.mlp.up_proj.weight"],
        tensors[f"{layer}.mlp.up_proj.bias"],
    )
    mlp_output = _linear(
        _gelu(mlp_up),
        tensors[f"{layer}.mlp.down_proj.weight"],
        tensors[f"{layer}.mlp.down_proj.bias"],
    )
    hidden_states = residual + mlp_output * tensors[f"{layer}.layer_scale2.lambda1"]
    return DinoV3BlockForward(hidden_states=hidden_states)


def run_dinov3_layer_stack(
    hidden_states: mx.array,
    config: DinoV3ModelConfig,
    tensors: dict[str, mx.array],
    *,
    patch_grid: tuple[int, int],
) -> DinoV3LayerStackForward:
    """Execute all DINOv3 transformer layers and the final layer norm."""

    for layer_index in range(config.num_hidden_layers):
        block = run_dinov3_transformer_block(
            hidden_states,
            config,
            tensors,
            patch_grid=patch_grid,
            layer_index=layer_index,
        )
        if block.blocker is not None:
            return DinoV3LayerStackForward(
                completed_layers=layer_index,
                blocker=block.blocker,
            )
        if block.hidden_states is None:
            return DinoV3LayerStackForward(
                completed_layers=layer_index,
                blocker=_blocker(
                    "MLX DINOv3 layer stack forward",
                    f"layer.{layer_index}",
                    f"DINOv3 transformer layer {layer_index} returned no hidden states",
                    "return hidden states from each MLX DINOv3 transformer layer",
                ),
            )
        hidden_states = block.hidden_states

    hidden_states = _layer_norm_no_affine(hidden_states, eps=config.layer_norm_eps)
    return DinoV3LayerStackForward(
        hidden_states=hidden_states,
        completed_layers=config.num_hidden_layers,
    )


def assess_dinov3_forward_probe(
    root: Path,
    *,
    config: DinoV3ModelConfig,
    inventory: DinoV3CheckpointInventory,
    image_tensor: mx.array | None = None,
) -> DinoV3ConditioningResult:
    """Run the implemented DINOv3 forward probes and return the next exact blocker."""

    key_map = inspect_dinov3_forward_key_map(inventory.checkpoint_path, config)
    if key_map.blocker is not None:
        return DinoV3ConditioningResult(blocker=key_map.blocker)

    if image_tensor is not None:
        loaded = load_dinov3_forward_tensors(
            inventory.checkpoint_path,
            config,
            all_layers=True,
        )
        if loaded.blocker is not None:
            return DinoV3ConditioningResult(blocker=loaded.blocker)
        if loaded.tensors is None:
            return DinoV3ConditioningResult(
                blocker=_blocker(
                    "DINOv3 forward checkpoint tensor loading",
                    str(inventory.checkpoint_path),
                    "selected DINOv3 forward tensor load returned no tensors",
                    "load selected DINOv3 tensors before patch-token assembly",
                )
            )
        assembled = assemble_dinov3_tokens(image_tensor, config, loaded.tensors)
        if assembled.blocker is not None:
            return DinoV3ConditioningResult(blocker=assembled.blocker)
        if assembled.tokens is not None and assembled.patch_grid is not None:
            stack = run_dinov3_layer_stack(
                assembled.tokens,
                config,
                loaded.tensors,
                patch_grid=assembled.patch_grid,
            )
            if stack.blocker is not None:
                return DinoV3ConditioningResult(blocker=stack.blocker)
            if stack.hidden_states is None:
                return DinoV3ConditioningResult(
                    blocker=_blocker(
                        "MLX DINOv3 layer stack forward",
                        str(inventory.checkpoint_path),
                        "DINOv3 layer stack execution returned no hidden states",
                        "return hidden states from the MLX DINOv3 layer stack",
                    )
                )
            try:
                mx.eval(stack.hidden_states)
            except RuntimeError as error:
                return DinoV3ConditioningResult(
                    blocker=_blocker(
                        "MLX DINOv3 layer stack forward",
                        str(inventory.checkpoint_path),
                        f"DINOv3 layer stack execution failed during MLX evaluation: {error}",
                        "resolve the MLX layer stack runtime failure before sparse-structure dispatch",
                    )
                )
            return DinoV3ConditioningResult(
                shape=tuple(stack.hidden_states.shape),
                dtype=str(stack.hidden_states.dtype).removeprefix("mlx.core."),
                hidden_states=stack.hidden_states,
                detail=(
                    f"MLX DINOv3 conditioning from {root} after "
                    f"{stack.completed_layers} transformer layers; "
                    f"patch grid={assembled.patch_grid}"
                ),
            )

    runtime_size = int(image_tensor.shape[-1]) if image_tensor is not None and image_tensor.ndim == 4 else None
    rope = probe_dinov3_rope(config, runtime_image_size=runtime_size)
    if rope.blocker is not None:
        return DinoV3ConditioningResult(blocker=rope.blocker)

    return DinoV3ConditioningResult(
        blocker=_blocker(
            "MLX DINOv3 attention block forward",
            str(root / "model.safetensors"),
            (
                "DINOv3 forward key map and RoPE geometry are valid, but no image "
                "tensor was provided for executing the transformer layer stack"
            ),
            "provide an image tensor and execute the MLX DINOv3 transformer layer stack",
        )
    )


def _self_attention(
    hidden_states: mx.array,
    config: DinoV3ModelConfig,
    tensors: dict[str, mx.array],
    *,
    patch_grid: tuple[int, int],
    layer_index: int,
) -> DinoV3BlockForward:
    layer = f"layer.{layer_index}"
    batch, token_count, _ = tuple(hidden_states.shape)
    head_dim = config.hidden_size // config.num_attention_heads

    query = _linear(
        hidden_states,
        tensors[f"{layer}.attention.q_proj.weight"],
        tensors[f"{layer}.attention.q_proj.bias"],
    )
    key = _linear(hidden_states, tensors[f"{layer}.attention.k_proj.weight"], None)
    value = _linear(
        hidden_states,
        tensors[f"{layer}.attention.v_proj.weight"],
        tensors[f"{layer}.attention.v_proj.bias"],
    )

    query = mx.transpose(
        mx.reshape(query, (batch, token_count, config.num_attention_heads, head_dim)),
        (0, 2, 1, 3),
    )
    key = mx.transpose(
        mx.reshape(key, (batch, token_count, config.num_attention_heads, head_dim)),
        (0, 2, 1, 3),
    )
    value = mx.transpose(
        mx.reshape(value, (batch, token_count, config.num_attention_heads, head_dim)),
        (0, 2, 1, 3),
    )

    rope = _rope_position_embeddings(config, patch_grid=patch_grid)
    if rope.blocker is not None:
        return DinoV3BlockForward(blocker=rope.blocker)
    if rope.cos is None or rope.sin is None:
        return DinoV3BlockForward(
            blocker=_blocker(
                "DINOv3 RoPE position embedding construction",
                "config.json:rope_theta",
                "DINOv3 RoPE position embedding construction returned no cos/sin arrays",
                "construct DINOv3 RoPE cos/sin arrays before attention execution",
            )
        )

    query, key = _apply_rotary_pos_emb(query, key, rope.cos, rope.sin)
    output = mx.fast.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=head_dim**-0.5,
    )
    output = mx.reshape(mx.transpose(output, (0, 2, 1, 3)), (batch, token_count, config.hidden_size))
    output = _linear(
        output,
        tensors[f"{layer}.attention.o_proj.weight"],
        tensors[f"{layer}.attention.o_proj.bias"],
    )
    return DinoV3BlockForward(hidden_states=output)


@dataclass(frozen=True)
class _RopeArrays:
    cos: mx.array | None = None
    sin: mx.array | None = None
    blocker: DinoV3PortBlocker | None = None


def _rope_position_embeddings(
    config: DinoV3ModelConfig,
    *,
    patch_grid: tuple[int, int],
) -> _RopeArrays:
    if config.rope_theta is None:
        return _RopeArrays(
            blocker=_blocker(
                "DINOv3 RoPE parameter validation",
                "config.json:rope_theta",
                "DINOv3 config is missing rope_theta for RoPE position embedding construction",
                "map Hugging Face DINOv3 RoPE parameters before attention execution",
            )
        )

    head_dim = config.hidden_size // config.num_attention_heads
    if head_dim % 4 != 0:
        return _RopeArrays(
            blocker=_blocker(
                "DINOv3 RoPE head dimension validation",
                "config.json:hidden_size,num_attention_heads",
                f"DINOv3 2D RoPE requires head_dim divisible by 4, got head_dim={head_dim}",
                "use DINOv3 attention dimensions compatible with 2D RoPE",
            )
        )

    patches_h, patches_w = patch_grid
    coords_h = mx.arange(0.5, patches_h, dtype=mx.float32) / patches_h
    coords_w = mx.arange(0.5, patches_w, dtype=mx.float32) / patches_w
    grid_h, grid_w = mx.meshgrid(coords_h, coords_w, indexing="ij")
    coords = mx.reshape(mx.stack((grid_h, grid_w), axis=-1), (patches_h * patches_w, 2))
    coords = 2.0 * coords - 1.0
    inv_freq = 1 / (config.rope_theta ** mx.arange(0, 1, 4 / head_dim, dtype=mx.float32))
    angles = 2 * 3.141592653589793 * coords[:, :, None] * inv_freq[None, None, :]
    angles = mx.tile(mx.reshape(angles, (patches_h * patches_w, head_dim // 2)), (1, 2))
    return _RopeArrays(cos=mx.cos(angles), sin=mx.sin(angles))


def _apply_rotary_pos_emb(
    query: mx.array,
    key: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array]:
    token_count = int(query.shape[-2])
    patch_count = int(cos.shape[-2])
    prefix_count = token_count - patch_count
    query_prefix = query[:, :, :prefix_count, :]
    key_prefix = key[:, :, :prefix_count, :]
    query_patches = query[:, :, prefix_count:, :]
    key_patches = key[:, :, prefix_count:, :]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    query_patches = (query_patches * cos) + (_rotate_half(query_patches) * sin)
    key_patches = (key_patches * cos) + (_rotate_half(key_patches) * sin)
    return (
        mx.concatenate((query_prefix, query_patches), axis=-2),
        mx.concatenate((key_prefix, key_patches), axis=-2),
    )


def _rotate_half(values: mx.array) -> mx.array:
    half = int(values.shape[-1]) // 2
    return mx.concatenate((-values[..., half:], values[..., :half]), axis=-1)


def _layer_norm(values: mx.array, weight: mx.array, bias: mx.array, *, eps: float) -> mx.array:
    mean = mx.mean(values, axis=-1, keepdims=True)
    variance = mx.var(values, axis=-1, keepdims=True)
    return ((values - mean) * mx.rsqrt(variance + eps)) * weight + bias


def _layer_norm_no_affine(values: mx.array, *, eps: float) -> mx.array:
    mean = mx.mean(values, axis=-1, keepdims=True)
    variance = mx.var(values, axis=-1, keepdims=True)
    return (values - mean) * mx.rsqrt(variance + eps)


def _linear(values: mx.array, weight: mx.array, bias: mx.array | None) -> mx.array:
    output = values @ mx.transpose(weight)
    if bias is not None:
        output = output + bias
    return output


def _gelu(values: mx.array) -> mx.array:
    return 0.5 * values * (1.0 + mx.erf(values / mx.sqrt(mx.array(2.0, dtype=values.dtype))))


def _validate_hidden_states(
    hidden_states: mx.array,
    config: DinoV3ModelConfig,
    *,
    patch_grid: tuple[int, int],
) -> DinoV3PortBlocker | None:
    if hidden_states.ndim != 3:
        return _blocker(
            "DINOv3 transformer block input validation",
            "DINOv3 token assembly",
            f"expected hidden states with shape (B, N, D), got {tuple(hidden_states.shape)}",
            "provide assembled DINOv3 tokens before transformer block execution",
        )
    if int(hidden_states.shape[-1]) != config.hidden_size:
        return _blocker(
            "DINOv3 transformer block input validation",
            "DINOv3 token assembly",
            f"expected hidden width {config.hidden_size}, got {int(hidden_states.shape[-1])}",
            "align DINOv3 token hidden width with config.hidden_size",
        )
    patch_count = patch_grid[0] * patch_grid[1]
    if int(hidden_states.shape[1]) < patch_count + 1:
        return _blocker(
            "DINOv3 transformer block input validation",
            "DINOv3 token assembly",
            f"token count {int(hidden_states.shape[1])} cannot contain {patch_count} patch tokens plus prefix tokens",
            "assemble cls/register/patch tokens before transformer block execution",
        )
    if config.hidden_size % config.num_attention_heads:
        return _blocker(
            "DINOv3 attention head dimension validation",
            "config.json:hidden_size,num_attention_heads",
            (
                f"hidden_size={config.hidden_size} is not divisible by "
                f"num_attention_heads={config.num_attention_heads}"
            ),
            "resolve DINOv3 attention head dimensions before transformer block execution",
        )
    return None


def _validate_selected_shapes(
    infos: dict[str, CheckpointTensorInfo],
    config: DinoV3ModelConfig,
    *,
    layer_indexes: range | tuple[int, ...],
) -> DinoV3PortBlocker | None:
    expected_shapes = {
        "embeddings.cls_token": (1, 1, config.hidden_size),
        "embeddings.patch_embeddings.bias": (config.hidden_size,),
        "embeddings.patch_embeddings.weight": (config.hidden_size, 3, config.patch_size, config.patch_size),
    }
    if config.num_register_tokens:
        expected_shapes["embeddings.register_tokens"] = (1, config.num_register_tokens, config.hidden_size)
    for layer_index in layer_indexes:
        layer = f"layer.{layer_index}"
        expected_shapes.update(
            {
                f"{layer}.attention.k_proj.weight": (config.hidden_size, config.hidden_size),
                f"{layer}.attention.o_proj.bias": (config.hidden_size,),
                f"{layer}.attention.o_proj.weight": (config.hidden_size, config.hidden_size),
                f"{layer}.attention.q_proj.bias": (config.hidden_size,),
                f"{layer}.attention.q_proj.weight": (config.hidden_size, config.hidden_size),
                f"{layer}.attention.v_proj.bias": (config.hidden_size,),
                f"{layer}.attention.v_proj.weight": (config.hidden_size, config.hidden_size),
                f"{layer}.layer_scale1.lambda1": (config.hidden_size,),
                f"{layer}.layer_scale2.lambda1": (config.hidden_size,),
                f"{layer}.mlp.down_proj.bias": (config.hidden_size,),
                f"{layer}.mlp.down_proj.weight": (config.hidden_size, config.intermediate_size),
                f"{layer}.mlp.up_proj.bias": (config.intermediate_size,),
                f"{layer}.mlp.up_proj.weight": (config.intermediate_size, config.hidden_size),
                f"{layer}.norm1.bias": (config.hidden_size,),
                f"{layer}.norm1.weight": (config.hidden_size,),
                f"{layer}.norm2.bias": (config.hidden_size,),
                f"{layer}.norm2.weight": (config.hidden_size,),
            }
        )

    for key, expected in expected_shapes.items():
        actual = infos[key].shape
        if actual != expected:
            return _blocker(
                "DINOv3 forward checkpoint shape validation",
                infos[key].source,
                f"{key} has shape {actual}; expected {expected}",
                "align selected DINOv3 forward checkpoint tensor shapes with config fields",
            )
    return None


def _dinov3_layer_required_keys(layer_index: int) -> tuple[str, ...]:
    layer = f"layer.{layer_index}"
    return (
        f"{layer}.attention.k_proj.weight",
        f"{layer}.attention.o_proj.bias",
        f"{layer}.attention.o_proj.weight",
        f"{layer}.attention.q_proj.bias",
        f"{layer}.attention.q_proj.weight",
        f"{layer}.attention.v_proj.bias",
        f"{layer}.attention.v_proj.weight",
        f"{layer}.layer_scale1.lambda1",
        f"{layer}.layer_scale2.lambda1",
        f"{layer}.mlp.down_proj.bias",
        f"{layer}.mlp.down_proj.weight",
        f"{layer}.mlp.up_proj.bias",
        f"{layer}.mlp.up_proj.weight",
        f"{layer}.norm1.bias",
        f"{layer}.norm1.weight",
        f"{layer}.norm2.bias",
        f"{layer}.norm2.weight",
    )


def _blocker(
    operation: str,
    reference: str,
    reason: str,
    next_slice: str,
) -> DinoV3PortBlocker:
    return DinoV3PortBlocker(
        stage="image-conditioning",
        operation=operation,
        reference=reference,
        reason=reason,
        next_slice=next_slice,
    )
