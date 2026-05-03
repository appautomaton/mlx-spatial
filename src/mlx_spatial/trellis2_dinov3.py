"""Offline DINOv3 config and checkpoint inspection for TRELLIS.2 conditioning."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx

from .checkpoint import CheckpointTensorInfo, inspect_checkpoint
from .model_assets import DINOv3_VITL16_ASSETS, ModelAssetValidation, validate_model_assets

DINOv3_VITL16_REPO_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINOv3_ACCESS_NOTE = (
    "facebook/dinov3-vitl16-pretrain-lvd1689m may require Hugging Face authentication "
    "or explicit terms acceptance before download."
)

_PATCH_EMBEDDING_KEYS = (
    "embeddings.patch_embeddings.weight",
    "embeddings.patch_embeddings.projection.weight",
)


@dataclass(frozen=True)
class DinoV3PortBlocker:
    stage: str
    operation: str
    reference: str
    reason: str
    next_slice: str


@dataclass(frozen=True)
class DinoV3ModelConfig:
    model_type: str
    image_size: int
    patch_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    layer_norm_eps: float
    use_swiglu_ffn: bool
    num_register_tokens: int
    expected_feature_width: int
    fake_conditioning: bool = False
    rope_theta: float | None = None
    pos_embed_rescale: float | None = None
    query_bias: bool = True
    key_bias: bool = False
    value_bias: bool = True
    proj_bias: bool = True
    mlp_bias: bool = True
    use_gated_mlp: bool = False


@dataclass(frozen=True)
class DinoV3CheckpointInventory:
    checkpoint_path: Path
    tensor_count: int
    patch_embedding_key: str
    patch_embedding_shape: tuple[int, ...]
    layer_prefix: str
    observed_layer_count: int
    norm_keys: tuple[str, ...]
    sample_keys: tuple[str, ...]


@dataclass(frozen=True)
class DinoV3InspectionResult:
    root: Path
    config: DinoV3ModelConfig | None = None
    inventory: DinoV3CheckpointInventory | None = None
    blocker: DinoV3PortBlocker | None = None

    @property
    def ready(self) -> bool:
        return self.config is not None and self.inventory is not None and self.blocker is None


@dataclass(frozen=True)
class DinoV3ConditioningResult:
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    detail: str | None = None
    hidden_states: mx.array | None = field(default=None, compare=False, repr=False)
    blocker: DinoV3PortBlocker | None = None

    @property
    def ready(self) -> bool:
        return self.shape is not None and self.dtype is not None and self.blocker is None


def validate_dinov3_assets(root: str | Path = DINOv3_VITL16_ASSETS.root_hint) -> ModelAssetValidation:
    """Validate a local DINOv3 asset root without downloading or loading tensors."""

    return validate_model_assets(root, DINOv3_VITL16_ASSETS)


def dinov3_download_command(root: str | Path = DINOv3_VITL16_ASSETS.root_hint) -> tuple[str, ...]:
    """Return the explicit dev-environment HF command for local DINOv3 assets."""

    return (
        "uv",
        "run",
        "hf",
        "download",
        DINOv3_VITL16_REPO_ID,
        "config.json",
        "model.safetensors",
        "--local-dir",
        str(root),
    )


def inspect_dinov3_assets(root: str | Path) -> DinoV3InspectionResult:
    """Inspect local DINOv3 config and safetensors metadata without Transformers."""

    root_path = Path(root)
    validation = validate_dinov3_assets(root_path)
    if not validation.ready:
        return DinoV3InspectionResult(
            root=root_path,
            blocker=_blocker(
                "local DINOv3 asset validation",
                DINOv3_VITL16_ASSETS.root_hint,
                f"missing local DINOv3 files at {root_path}: {list(validation.missing)}",
                "place config.json and model.safetensors under the local DINOv3 asset root",
            ),
        )

    config_result = read_dinov3_config(root_path / "config.json")
    if config_result.blocker is not None or config_result.config is None:
        return DinoV3InspectionResult(root=root_path, blocker=config_result.blocker)

    inventory_result = inspect_dinov3_checkpoint(root_path / "model.safetensors", config_result.config)
    if inventory_result.blocker is not None:
        return DinoV3InspectionResult(root=root_path, config=config_result.config, blocker=inventory_result.blocker)

    return DinoV3InspectionResult(
        root=root_path,
        config=config_result.config,
        inventory=inventory_result.inventory,
    )


def read_dinov3_config(path: str | Path) -> DinoV3InspectionResult:
    """Read and validate the DINOv3 fields needed by TRELLIS.2 conditioning."""

    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return DinoV3InspectionResult(
            root=config_path.parent,
            blocker=_blocker(
                "DINOv3 config discovery",
                str(config_path),
                f"DINOv3 config file not found: {config_path}",
                "place config.json under the local DINOv3 asset root",
            ),
        )
    except json.JSONDecodeError as error:
        return DinoV3InspectionResult(
            root=config_path.parent,
            blocker=_blocker(
                "DINOv3 config discovery",
                str(config_path),
                f"DINOv3 config is not valid JSON: {error}",
                "replace config.json with the Hugging Face DINOv3 model config",
            ),
        )

    try:
        config = DinoV3ModelConfig(
            model_type=str(_required(raw, "model_type")),
            image_size=_int_field(raw, "image_size"),
            patch_size=_int_field(raw, "patch_size"),
            hidden_size=_int_field(raw, "hidden_size"),
            num_hidden_layers=_int_field(raw, "num_hidden_layers"),
            num_attention_heads=_int_field(raw, "num_attention_heads"),
            intermediate_size=_int_field(raw, "intermediate_size"),
            layer_norm_eps=float(raw.get("layer_norm_eps", 1e-6)),
            use_swiglu_ffn=bool(raw.get("use_swiglu_ffn", raw.get("use_swiglu", False))),
            num_register_tokens=int(raw.get("num_register_tokens", 0)),
            expected_feature_width=int(raw.get("hidden_size")),
            fake_conditioning=bool(raw.get("mlx_spatial_fake_conditioning", False))
            and str(raw.get("model_type", "")).startswith("mlx_spatial_fake"),
            rope_theta=float(raw["rope_theta"]) if raw.get("rope_theta") is not None else None,
            pos_embed_rescale=float(raw["pos_embed_rescale"]) if raw.get("pos_embed_rescale") is not None else None,
            query_bias=bool(raw.get("query_bias", True)),
            key_bias=bool(raw.get("key_bias", False)),
            value_bias=bool(raw.get("value_bias", True)),
            proj_bias=bool(raw.get("proj_bias", True)),
            mlp_bias=bool(raw.get("mlp_bias", True)),
            use_gated_mlp=bool(raw.get("use_gated_mlp", False)),
        )
    except (KeyError, TypeError, ValueError) as error:
        return DinoV3InspectionResult(
            root=config_path.parent,
            blocker=_blocker(
                "DINOv3 config field validation",
                str(config_path),
                f"DINOv3 config is missing or has an invalid field: {error}",
                "map the Hugging Face DINOv3 config fields before module construction",
            ),
        )

    if config.image_size % config.patch_size != 0:
        return DinoV3InspectionResult(
            root=config_path.parent,
            blocker=_blocker(
                "DINOv3 config field validation",
                str(config_path),
                f"image_size={config.image_size} is not divisible by patch_size={config.patch_size}",
                "resolve DINOv3 patch grid sizing before conditioning forward",
            ),
        )
    if config.hidden_size % config.num_attention_heads != 0:
        return DinoV3InspectionResult(
            root=config_path.parent,
            blocker=_blocker(
                "DINOv3 config field validation",
                str(config_path),
                (
                    f"hidden_size={config.hidden_size} is not divisible by "
                    f"num_attention_heads={config.num_attention_heads}"
                ),
                "resolve DINOv3 attention head dimensions before conditioning forward",
            ),
        )

    return DinoV3InspectionResult(root=config_path.parent, config=config)


def inspect_dinov3_checkpoint(
    path: str | Path,
    config: DinoV3ModelConfig,
) -> DinoV3InspectionResult:
    """Inspect DINOv3 checkpoint keys and shapes needed for an MLX forward probe."""

    checkpoint_path = Path(path)
    try:
        infos = inspect_checkpoint(checkpoint_path)
    except (FileNotFoundError, ValueError) as error:
        return DinoV3InspectionResult(
            root=checkpoint_path.parent,
            config=config,
            blocker=_blocker(
                "DINOv3 checkpoint inspection",
                str(checkpoint_path),
                str(error),
                "inspect the local DINOv3 checkpoint format before module construction",
            ),
        )

    by_name = {info.name: info for info in infos}
    patch_info = next((by_name[key] for key in _PATCH_EMBEDDING_KEYS if key in by_name), None)
    if patch_info is None:
        return DinoV3InspectionResult(
            root=checkpoint_path.parent,
            config=config,
            blocker=_blocker(
                "DINOv3 checkpoint key validation",
                str(checkpoint_path),
                f"checkpoint is missing a patch embedding tensor; expected one of {list(_PATCH_EMBEDDING_KEYS)}",
                "map the DINOv3 patch embedding checkpoint key before module construction",
            ),
        )

    patch_blocker = _validate_patch_embedding(patch_info, config)
    if patch_blocker is not None:
        return DinoV3InspectionResult(root=checkpoint_path.parent, config=config, blocker=patch_blocker)

    layer_prefix, observed_layers = _detect_layer_prefix(tuple(by_name))
    if observed_layers == 0:
        return DinoV3InspectionResult(
            root=checkpoint_path.parent,
            config=config,
            blocker=_blocker(
                "DINOv3 checkpoint key validation",
                str(checkpoint_path),
                "checkpoint has no transformer layer tensors under layer.* or encoder.layer.*",
                "map the DINOv3 transformer layer checkpoint keys before module construction",
            ),
        )

    norm_keys = tuple(
        name
        for name in sorted(by_name)
        if name.endswith("norm.weight")
        or name.endswith("norm1.weight")
        or name.endswith("norm2.weight")
        or name.endswith("layernorm.weight")
    )
    inventory = DinoV3CheckpointInventory(
        checkpoint_path=checkpoint_path,
        tensor_count=len(infos),
        patch_embedding_key=patch_info.name,
        patch_embedding_shape=patch_info.shape,
        layer_prefix=layer_prefix,
        observed_layer_count=observed_layers,
        norm_keys=norm_keys,
        sample_keys=tuple(info.name for info in infos[:5]),
    )
    return DinoV3InspectionResult(root=checkpoint_path.parent, config=config, inventory=inventory)


def assess_dinov3_mlx_conditioning(
    root: str | Path,
    *,
    expected_feature_width: int,
    image_tensor: mx.array | None = None,
) -> DinoV3ConditioningResult:
    """Attempt the local DINOv3 conditioning boundary or return its first blocker."""

    inspected = inspect_dinov3_assets(root)
    if inspected.blocker is not None:
        return DinoV3ConditioningResult(blocker=inspected.blocker)
    if inspected.config is None or inspected.inventory is None:
        return DinoV3ConditioningResult(
            blocker=_blocker(
                "DINOv3 asset inventory",
                str(root),
                "DINOv3 inspection returned no config or checkpoint inventory",
                "complete DINOv3 config and checkpoint inspection before module construction",
            )
        )

    config = inspected.config
    if config.expected_feature_width != expected_feature_width:
        return DinoV3ConditioningResult(
            blocker=_blocker(
                "DINOv3 feature width validation",
                str(Path(root) / "config.json"),
                f"DINOv3 hidden_size={config.expected_feature_width} does not match cond_channels={expected_feature_width}",
                "use the TRELLIS.2 DINOv3 model whose feature width matches sparse flow cond_channels",
            )
        )

    if config.fake_conditioning:
        batch = int(image_tensor.shape[0]) if image_tensor is not None and image_tensor.ndim == 4 else 1
        tokens_per_side = config.image_size // config.patch_size
        token_count = tokens_per_side * tokens_per_side + config.num_register_tokens
        return DinoV3ConditioningResult(
            shape=(batch, token_count, config.expected_feature_width),
            dtype="float32",
            hidden_states=mx.zeros((batch, token_count, config.expected_feature_width), dtype=mx.float32),
            detail=(
                f"fake DINOv3 conditioning from {Path(root)} using "
                f"{inspected.inventory.patch_embedding_key}"
            ),
        )

    from .trellis2_dinov3_forward import assess_dinov3_forward_probe

    return assess_dinov3_forward_probe(
        Path(root),
        config=config,
        inventory=inspected.inventory,
        image_tensor=image_tensor,
    )


def _validate_patch_embedding(
    info: CheckpointTensorInfo,
    config: DinoV3ModelConfig,
) -> DinoV3PortBlocker | None:
    if len(info.shape) != 4:
        return _blocker(
            "DINOv3 checkpoint shape validation",
            info.source,
            f"{info.name} has shape {info.shape}; expected 4D convolution weights",
            "map the DINOv3 patch embedding tensor shape before module construction",
        )
    if info.shape[0] != config.hidden_size:
        return _blocker(
            "DINOv3 checkpoint shape validation",
            info.source,
            f"{info.name} output channels={info.shape[0]} do not match hidden_size={config.hidden_size}",
            "align DINOv3 patch embedding weights with config.hidden_size",
        )
    if info.shape[1] != 3:
        return _blocker(
            "DINOv3 checkpoint shape validation",
            info.source,
            f"{info.name} input channels={info.shape[1]} do not match RGB image channels=3",
            "align DINOv3 patch embedding weights with RGB image input",
        )
    if info.shape[2:] != (config.patch_size, config.patch_size):
        return _blocker(
            "DINOv3 checkpoint shape validation",
            info.source,
            f"{info.name} kernel={info.shape[2:]} does not match patch_size={config.patch_size}",
            "align DINOv3 patch embedding weights with config.patch_size",
        )
    return None


def _detect_layer_prefix(names: tuple[str, ...]) -> tuple[str, int]:
    for prefix in ("layer.", "encoder.layer."):
        indexes = set()
        for name in names:
            if not name.startswith(prefix):
                continue
            remainder = name[len(prefix) :]
            index = remainder.split(".", 1)[0]
            if index.isdigit():
                indexes.add(int(index))
        if indexes:
            return prefix, len(indexes)
    return "layer.", 0


def _required(raw: dict[str, object], key: str) -> object:
    if key not in raw:
        raise KeyError(key)
    return raw[key]


def _int_field(raw: dict[str, object], key: str) -> int:
    value = _required(raw, key)
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{key} must be positive")
    return parsed


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
