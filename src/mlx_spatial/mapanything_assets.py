"""MapAnything asset validation and checkpoint inspection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from safetensors import SafetensorError

from .checkpoint import CheckpointTensorInfo, inspect_checkpoint


MAPANYTHING_REPO_ID = "facebook/map-anything"
MAPANYTHING_DEFAULT_ROOT = "weights/map-anything"
MAPANYTHING_REQUIRED_PATHS = ("config.json", "model.safetensors")
MAPANYTHING_COMPONENT_GROUPS = (
    "encoder",
    "info_sharing",
    "ray_dirs_encoder",
    "depth_encoder",
    "cam_rot_encoder",
    "cam_trans_encoder",
    "cam_trans_scale_encoder",
    "depth_scale_encoder",
    "dense_head",
    "pose_head",
    "scale_head",
    "fusion_norm_layer",
    "scale_token",
)
MAPANYTHING_DEFAULT_REQUIRED_GROUPS = ("encoder", "info_sharing")


@dataclass(frozen=True)
class MapAnythingAssetValidation:
    """Deterministic presence report for a local MapAnything checkpoint root."""

    root: Path
    config_path: Path
    checkpoint_path: Path
    present: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.missing


@dataclass(frozen=True)
class MapAnythingModelConfig:
    """MLX-facing subset of the official MapAnything model config."""

    data_norm_type: str
    encoder_name: str
    encoder_size: str
    encoder_keep_first_n_layers: int
    encoder_uses_torch_hub: bool
    encoder_with_registers: bool
    patch_size: int
    info_sharing_model_type: str
    info_sharing_return_type: str
    info_sharing_depth: int
    info_sharing_dim: int
    info_sharing_num_heads: int
    info_sharing_indices: tuple[int, ...]
    pred_head_type: str
    pred_head_adaptor_type: str
    pred_head_output_type: str
    use_register_tokens_from_encoder: bool

    @property
    def encoder(self) -> dict[str, object]:
        return {
            "name": self.encoder_name,
            "size": self.encoder_size,
            "keep_first_n_layers": self.encoder_keep_first_n_layers,
            "uses_torch_hub": self.encoder_uses_torch_hub,
            "with_registers": self.encoder_with_registers,
            "data_norm_type": self.data_norm_type,
            "patch_size": self.patch_size,
        }

    @property
    def info_sharing(self) -> dict[str, object]:
        return {
            "model_type": self.info_sharing_model_type,
            "return_type": self.info_sharing_return_type,
            "depth": self.info_sharing_depth,
            "dim": self.info_sharing_dim,
            "num_heads": self.info_sharing_num_heads,
            "indices": self.info_sharing_indices,
        }

    @property
    def prediction(self) -> dict[str, object]:
        return {
            "type": self.pred_head_type,
            "adaptor_type": self.pred_head_adaptor_type,
            "output_type": self.pred_head_output_type,
        }


@dataclass(frozen=True)
class MapAnythingComponentGroup:
    """Checkpoint tensors belonging to one MapAnything component prefix."""

    name: str
    keys: tuple[str, ...]

    @property
    def present(self) -> bool:
        return bool(self.keys)


@dataclass(frozen=True)
class MapAnythingAssetBlocker:
    """Structured setup blocker discovered before model execution."""

    stage: str
    operation: str
    reason: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class MapAnythingCheckpointInspection:
    """Config and checkpoint routing metadata without loading full tensors."""

    validation: MapAnythingAssetValidation
    config: MapAnythingModelConfig | None
    groups: tuple[MapAnythingComponentGroup, ...]
    blocker: MapAnythingAssetBlocker | None

    @property
    def ready(self) -> bool:
        return self.blocker is None

    @property
    def missing_groups(self) -> tuple[str, ...]:
        return tuple(group.name for group in self.groups if not group.present)


def validate_mapanything_assets(
    root: str | Path = MAPANYTHING_DEFAULT_ROOT,
) -> MapAnythingAssetValidation:
    """Validate a MapAnything root without downloading or loading checkpoint tensors."""

    root_path = Path(root)
    config_path = root_path / "config.json"
    checkpoint_path = root_path / "model.safetensors"
    present: list[str] = []
    missing: list[str] = []

    for relative_path, path in (
        ("config.json", config_path),
        ("model.safetensors", checkpoint_path),
    ):
        if path.is_file():
            present.append(relative_path)
        else:
            missing.append(relative_path)

    return MapAnythingAssetValidation(
        root=root_path,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        present=tuple(present),
        missing=tuple(missing),
    )


def read_mapanything_model_config(path: str | Path) -> MapAnythingModelConfig:
    """Read the local HF MapAnything config subset required by the MLX port."""

    config_path = Path(path)
    if not config_path.is_file():
        raise ValueError(f"MapAnything config file not found: {config_path}")

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"MapAnything JSON config is invalid: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("MapAnything config must be a mapping")

    try:
        encoder_config = _require_mapping(raw, "encoder_config")
        info_sharing_config = _require_mapping(raw, "info_sharing_config")
        info_args = _require_mapping(info_sharing_config, "module_args")
        pred_head_config = _require_mapping(raw, "pred_head_config")
        adaptor_config = _require_mapping(pred_head_config, "adaptor_config")
        dense_pred_init = _require_mapping(adaptor_config, "dense_pred_init_dict")
        patch_size = _resolve_patch_size(encoder_config, pred_head_config)
        return MapAnythingModelConfig(
            data_norm_type=str(encoder_config["data_norm_type"]),
            encoder_name=str(encoder_config["name"]),
            encoder_size=str(encoder_config["size"]),
            encoder_keep_first_n_layers=int(encoder_config["keep_first_n_layers"]),
            encoder_uses_torch_hub=bool(encoder_config["uses_torch_hub"]),
            encoder_with_registers=bool(encoder_config.get("with_registers", False)),
            patch_size=patch_size,
            info_sharing_model_type=str(info_sharing_config["model_type"]),
            info_sharing_return_type=str(info_sharing_config["model_return_type"]),
            info_sharing_depth=int(info_args["depth"]),
            info_sharing_dim=int(info_args["dim"]),
            info_sharing_num_heads=int(info_args["num_heads"]),
            info_sharing_indices=tuple(int(index) for index in info_args["indices"]),
            pred_head_type=str(pred_head_config["type"]),
            pred_head_adaptor_type=str(pred_head_config["adaptor_type"]),
            pred_head_output_type=str(dense_pred_init["name"]),
            use_register_tokens_from_encoder=bool(raw["use_register_tokens_from_encoder"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"MapAnything config has an invalid model field in {config_path}: {error}") from error


def inspect_mapanything_checkpoint(
    root: str | Path = MAPANYTHING_DEFAULT_ROOT,
    *,
    prefixes: Iterable[str] | None = None,
) -> tuple[CheckpointTensorInfo, ...]:
    """Inspect tensors in a local MapAnything model.safetensors checkpoint."""

    validation = validate_mapanything_assets(root)
    if not validation.checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {validation.checkpoint_path}")
    return inspect_checkpoint(validation.checkpoint_path, prefixes=prefixes)


def inspect_mapanything_model_assets(
    root: str | Path = MAPANYTHING_DEFAULT_ROOT,
    *,
    required_groups: Sequence[str] = MAPANYTHING_DEFAULT_REQUIRED_GROUPS,
) -> MapAnythingCheckpointInspection:
    """Inspect MapAnything config and checkpoint component groups before execution."""

    validation = validate_mapanything_assets(root)
    normalized_required = _normalize_required_groups(required_groups)
    if not validation.ready:
        return MapAnythingCheckpointInspection(
            validation=validation,
            config=None,
            groups=_empty_component_groups(),
            blocker=MapAnythingAssetBlocker(
                stage="asset-validation",
                operation="validate MapAnything checkpoint layout",
                reason="missing required MapAnything checkpoint assets",
                metadata={"missing": validation.missing, "required_groups": normalized_required},
            ),
        )

    try:
        config = read_mapanything_model_config(validation.config_path)
    except ValueError as error:
        return MapAnythingCheckpointInspection(
            validation=validation,
            config=None,
            groups=_empty_component_groups(),
            blocker=MapAnythingAssetBlocker(
                stage="checkpoint-inspection",
                operation="parse MapAnything model config",
                reason=str(error),
                metadata={"config": str(validation.config_path), "required_groups": normalized_required},
            ),
        )

    try:
        infos = inspect_mapanything_checkpoint(validation.root)
    except (SafetensorError, OSError, ValueError) as error:
        return MapAnythingCheckpointInspection(
            validation=validation,
            config=config,
            groups=_empty_component_groups(),
            blocker=MapAnythingAssetBlocker(
                stage="checkpoint-inspection",
                operation="inspect MapAnything safetensors checkpoint metadata",
                reason=f"checkpoint safetensors metadata could not be inspected: {error}",
                metadata={
                    "checkpoint": str(validation.checkpoint_path),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "required_groups": normalized_required,
                },
            ),
        )

    groups = group_mapanything_checkpoint_keys(info.name for info in infos)
    missing = tuple(
        group.name for group in groups if group.name in normalized_required and not group.present
    )
    if missing:
        return MapAnythingCheckpointInspection(
            validation=validation,
            config=config,
            groups=groups,
            blocker=MapAnythingAssetBlocker(
                stage="checkpoint-inspection",
                operation="route MapAnything safetensors keys into model components",
                reason=f"checkpoint is missing required MapAnything component groups: {', '.join(missing)}",
                metadata={"missing_groups": missing, "required_groups": normalized_required},
            ),
        )

    return MapAnythingCheckpointInspection(
        validation=validation,
        config=config,
        groups=groups,
        blocker=None,
    )


def group_mapanything_checkpoint_keys(keys: Iterable[str]) -> tuple[MapAnythingComponentGroup, ...]:
    """Group safetensors keys by MapAnything component prefix."""

    grouped = {name: [] for name in MAPANYTHING_COMPONENT_GROUPS}
    for key in sorted(keys):
        component = key.split(".", 1)[0]
        if component in grouped:
            grouped[component].append(key)
    return tuple(
        MapAnythingComponentGroup(name=name, keys=tuple(grouped[name]))
        for name in MAPANYTHING_COMPONENT_GROUPS
    )


def mapanything_download_command(root: str | Path = MAPANYTHING_DEFAULT_ROOT) -> tuple[str, ...]:
    """Return the dev-environment HF command for downloading MapAnything assets."""

    return (
        "uv",
        "run",
        "hf",
        "download",
        MAPANYTHING_REPO_ID,
        "--local-dir",
        str(root),
    )


def _require_mapping(raw: dict[str, object], key: str) -> dict[str, object]:
    value = raw[key]
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _resolve_patch_size(
    encoder_config: dict[str, object],
    pred_head_config: dict[str, object],
) -> int:
    if "patch_size" in encoder_config:
        return int(encoder_config["patch_size"])
    feature_head = _require_mapping(pred_head_config, "feature_head")
    return int(feature_head["patch_size"])


def _empty_component_groups() -> tuple[MapAnythingComponentGroup, ...]:
    return tuple(MapAnythingComponentGroup(name=name, keys=()) for name in MAPANYTHING_COMPONENT_GROUPS)


def _normalize_required_groups(required_groups: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for group in required_groups:
        value = str(group)
        if value not in MAPANYTHING_COMPONENT_GROUPS:
            raise ValueError(f"unsupported MapAnything checkpoint group: {value!r}")
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("at least one MapAnything checkpoint group must be required")
    return tuple(normalized)
