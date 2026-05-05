"""HY-World-2.0 WorldMirror asset validation and checkpoint inspection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from safetensors import SafetensorError

from .checkpoint import CheckpointTensorInfo, inspect_checkpoint


HYWORLD2_REPO_ID = "tencent/HY-World-2.0"
HYWORLD2_WORLDMIRROR_SUBFOLDER = "HY-WorldMirror-2.0"
HYWORLD2_DEFAULT_ROOT = "weights/hy-world-2"
HYWORLD2_COMPONENT_GROUPS = (
    "visual_geometry_transformer",
    "cam_head",
    "pts_head",
    "depth_head",
    "norm_head",
    "gs_head",
    "gs_renderer",
)
HYWORLD2_CHECKPOINT_HEAD_GROUPS = {
    "camera": "cam_head",
    "depth": "depth_head",
    "normal": "norm_head",
    "points": "pts_head",
    "gs": "gs_head",
}
HYWORLD2_CHECKPOINT_EXTRA_HEAD_GROUPS = {
    "gs": ("gs_renderer",),
}
HYWORLD2_DEFAULT_CHECKPOINT_HEADS = ("depth", "normal", "points")

_CONFIG_DEFAULTS = {
    "condition_strategy": ("token", "pow3r", "token"),
    "depth": 24,
    "disable_gs_depth": False,
    "dpt_gradient_checkpoint": False,
    "embed_dim": 1024,
    "enable_bf16": False,
    "enable_cam": True,
    "enable_cond": True,
    "enable_depth": True,
    "enable_depth_mask": True,
    "enable_gs": True,
    "enable_norm": True,
    "enable_pts": True,
    "fixed_patch_embed": True,
    "gs_dim": 256,
    "img_size": 518,
    "mlp_ratio": 4.0,
    "model_size": "large",
    "normalized_rope": True,
    "num_heads": 16,
    "num_register_tokens": 4,
    "patch_embed": "dinov2_vitl14_reg",
    "patch_size": 14,
    "rope_base": 100.0,
    "rope_jitter_coords": None,
    "rope_normalize_coords": "separate",
    "rope_rescale_coords": None,
    "rope_shift_coords": None,
    "sampling_strategy": "uniform",
    "set_sky_region_to_maxdepth": False,
    "sp_size": 1,
}


@dataclass(frozen=True)
class HyWorld2AssetValidation:
    """Deterministic presence report for a local WorldMirror checkpoint root."""

    root: Path
    model_dir: Path
    checkpoint_path: Path
    config_path: Path | None
    present: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.missing

    @property
    def config_kind(self) -> str | None:
        if self.config_path is None:
            return None
        return self.config_path.suffix.removeprefix(".")


@dataclass(frozen=True)
class HyWorld2AssetBlocker:
    """Structured setup blocker discovered before model execution."""

    stage: str
    operation: str
    reason: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class HyWorld2ModelConfig:
    """Resolved WorldMirror model kwargs needed for staged MLX construction."""

    condition_strategy: tuple[str, ...]
    depth: int
    disable_gs_depth: bool
    dpt_gradient_checkpoint: bool
    embed_dim: int
    enable_bf16: bool
    enable_cam: bool
    enable_cond: bool
    enable_depth: bool
    enable_depth_mask: bool
    enable_gs: bool
    enable_norm: bool
    enable_pts: bool
    fixed_patch_embed: bool
    gs_dim: int
    img_size: int
    mlp_ratio: float
    model_size: str
    normalized_rope: bool
    num_heads: int
    num_register_tokens: int
    patch_embed: str
    patch_size: int
    rope_base: float
    rope_jitter_coords: object
    rope_normalize_coords: str | None
    rope_rescale_coords: object
    rope_shift_coords: object
    sampling_strategy: str
    set_sky_region_to_maxdepth: bool
    sp_size: int

    @property
    def visual_geometry_transformer(self) -> dict[str, object]:
        return {
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "patch_embed": self.patch_embed,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "num_register_tokens": self.num_register_tokens,
            "rope_base": self.rope_base,
            "normalized_rope": self.normalized_rope,
            "rope_normalize_coords": self.rope_normalize_coords,
            "condition_strategy": self.condition_strategy,
        }

    @property
    def camera_head(self) -> dict[str, object]:
        return {
            "enabled": self.enable_cam,
            "embed_dim": self.embed_dim,
            "num_register_tokens": self.num_register_tokens,
        }

    @property
    def dpt_heads(self) -> dict[str, object]:
        return {
            "embed_dim": self.embed_dim,
            "enable_depth": self.enable_depth,
            "enable_depth_mask": self.enable_depth_mask,
            "enable_norm": self.enable_norm,
            "enable_pts": self.enable_pts,
            "enable_gs": self.enable_gs,
            "gs_dim": self.gs_dim,
            "gradient_checkpoint": self.dpt_gradient_checkpoint,
            "disable_gs_depth": self.disable_gs_depth,
            "set_sky_region_to_maxdepth": self.set_sky_region_to_maxdepth,
        }


@dataclass(frozen=True)
class HyWorld2ComponentGroup:
    """Checkpoint tensors belonging to one official WorldMirror component."""

    name: str
    keys: tuple[str, ...]

    @property
    def present(self) -> bool:
        return bool(self.keys)


@dataclass(frozen=True)
class HyWorld2CheckpointInspection:
    """Config and checkpoint routing metadata without loading full tensors."""

    validation: HyWorld2AssetValidation
    config: HyWorld2ModelConfig | None
    groups: tuple[HyWorld2ComponentGroup, ...]
    blocker: HyWorld2AssetBlocker | None

    @property
    def ready(self) -> bool:
        return self.blocker is None

    @property
    def missing_groups(self) -> tuple[str, ...]:
        return tuple(group.name for group in self.groups if not group.present)


def validate_hyworld2_assets(root: str | Path = HYWORLD2_DEFAULT_ROOT) -> HyWorld2AssetValidation:
    """Validate a HY-World-2.0 WorldMirror root without downloading files."""

    root_path = Path(root)
    model_dir = resolve_hyworld2_model_dir(root_path)
    checkpoint_path = model_dir / "model.safetensors"
    config_candidates = (model_dir / "config.yaml", model_dir / "config.json")
    config_path = next((path for path in config_candidates if path.is_file()), None)

    present: list[str] = []
    missing: list[str] = []

    checkpoint_rel = _relative_report_path(root_path, checkpoint_path)
    if checkpoint_path.is_file():
        present.append(checkpoint_rel)
    else:
        missing.append(checkpoint_rel)

    if config_path is None:
        missing.append(
            " or ".join(_relative_report_path(root_path, path) for path in config_candidates)
        )
    else:
        present.append(_relative_report_path(root_path, config_path))

    return HyWorld2AssetValidation(
        root=root_path,
        model_dir=model_dir,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        present=tuple(present),
        missing=tuple(missing),
    )


def resolve_hyworld2_model_dir(root: str | Path) -> Path:
    """Resolve a root that is either the repo root or the WorldMirror subfolder."""

    root_path = Path(root)
    direct_markers = (
        root_path / "model.safetensors",
        root_path / "config.yaml",
        root_path / "config.json",
    )
    if any(path.exists() for path in direct_markers):
        return root_path
    return root_path / HYWORLD2_WORLDMIRROR_SUBFOLDER


def inspect_hyworld2_checkpoint(
    root: str | Path = HYWORLD2_DEFAULT_ROOT,
    *,
    prefixes: Iterable[str] | None = None,
) -> tuple[CheckpointTensorInfo, ...]:
    """Inspect tensors in a local WorldMirror model.safetensors checkpoint."""

    validation = validate_hyworld2_assets(root)
    if not validation.checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {validation.checkpoint_path}")
    return inspect_checkpoint(validation.checkpoint_path, prefixes=prefixes)


def inspect_hyworld2_model_assets(
    root: str | Path = HYWORLD2_DEFAULT_ROOT,
    *,
    requested_heads: Sequence[str] | str | None = None,
) -> HyWorld2CheckpointInspection:
    """Inspect WorldMirror config and checkpoint component groups before execution."""

    validation = validate_hyworld2_assets(root)
    normalized_heads = _normalize_requested_heads(requested_heads)
    if not validation.ready:
        return HyWorld2CheckpointInspection(
            validation=validation,
            config=None,
            groups=_empty_component_groups(),
            blocker=HyWorld2AssetBlocker(
                stage="asset-validation",
                operation="validate HY-World-2.0 WorldMirror checkpoint layout",
                reason="missing required WorldMirror checkpoint assets",
                metadata={"missing": validation.missing, "requested_heads": normalized_heads},
            ),
        )

    try:
        config = read_hyworld2_model_config(validation.config_path)
    except ValueError as error:
        return HyWorld2CheckpointInspection(
            validation=validation,
            config=None,
            groups=_empty_component_groups(),
            blocker=HyWorld2AssetBlocker(
                stage="checkpoint-inspection",
                operation="parse HY-World WorldMirror model config",
                reason=str(error),
                metadata={"config": str(validation.config_path), "requested_heads": normalized_heads},
            ),
        )

    try:
        infos = inspect_hyworld2_checkpoint(validation.model_dir)
    except (SafetensorError, OSError, ValueError) as error:
        return HyWorld2CheckpointInspection(
            validation=validation,
            config=config,
            groups=_empty_component_groups(),
            blocker=HyWorld2AssetBlocker(
                stage="checkpoint-inspection",
                operation="inspect HY-World safetensors checkpoint metadata",
                reason=f"checkpoint safetensors metadata could not be inspected: {error}",
                metadata={
                    "checkpoint": str(validation.checkpoint_path),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "requested_heads": normalized_heads,
                },
            ),
        )
    groups = group_hyworld2_checkpoint_keys(info.name for info in infos)
    required_groups = _required_checkpoint_groups(normalized_heads)
    missing = tuple(group.name for group in groups if group.name in required_groups and not group.present)
    if missing:
        return HyWorld2CheckpointInspection(
            validation=validation,
            config=config,
            groups=groups,
            blocker=HyWorld2AssetBlocker(
                stage="checkpoint-inspection",
                operation="route HY-World safetensors keys into WorldMirror components",
                reason=f"checkpoint is missing required WorldMirror component groups: {', '.join(missing)}",
                metadata={
                    "missing_groups": missing,
                    "required_groups": required_groups,
                    "requested_heads": normalized_heads,
                },
            ),
        )

    return HyWorld2CheckpointInspection(
        validation=validation,
        config=config,
        groups=groups,
        blocker=None,
    )


def read_hyworld2_model_config(path: str | Path | None) -> HyWorld2ModelConfig:
    """Read JSON/YAML config into official WorldMirror model kwargs with defaults."""

    if path is None:
        raise ValueError("WorldMirror config file not found")
    config_path = Path(path)
    if not config_path.is_file():
        raise ValueError(f"WorldMirror config file not found: {config_path}")

    if config_path.suffix == ".json":
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"WorldMirror JSON config is invalid: {error}") from error
    elif config_path.suffix in {".yaml", ".yml"}:
        raw = _read_yaml_config(config_path)
    else:
        raise ValueError(f"unsupported WorldMirror config format: {config_path.suffix or '<none>'}")

    if not isinstance(raw, dict):
        raise ValueError("WorldMirror config must be a mapping")
    model = _extract_model_config(raw)
    merged = dict(_CONFIG_DEFAULTS)
    merged.update(model)
    return _coerce_model_config(merged, config_path)


def group_hyworld2_checkpoint_keys(keys: Iterable[str]) -> tuple[HyWorld2ComponentGroup, ...]:
    """Group safetensors keys by official WorldMirror component prefix."""

    grouped = {name: [] for name in HYWORLD2_COMPONENT_GROUPS}
    for key in sorted(keys):
        component = key.split(".", 1)[0]
        if component in grouped:
            grouped[component].append(key)
    return tuple(
        HyWorld2ComponentGroup(name=name, keys=tuple(grouped[name]))
        for name in HYWORLD2_COMPONENT_GROUPS
    )


def hyworld2_download_command(root: str | Path = HYWORLD2_DEFAULT_ROOT) -> tuple[str, ...]:
    """Return the dev-environment HF command for downloading WorldMirror assets."""

    return (
        "uv",
        "run",
        "hf",
        "download",
        HYWORLD2_REPO_ID,
        "--include",
        f"{HYWORLD2_WORLDMIRROR_SUBFOLDER}/*",
        "--local-dir",
        str(root),
    )


def _relative_report_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _empty_component_groups() -> tuple[HyWorld2ComponentGroup, ...]:
    return tuple(HyWorld2ComponentGroup(name=name, keys=()) for name in HYWORLD2_COMPONENT_GROUPS)


def _normalize_requested_heads(heads: Sequence[str] | str | None) -> tuple[str, ...]:
    raw_values = HYWORLD2_DEFAULT_CHECKPOINT_HEADS if heads is None else ([heads] if isinstance(heads, str) else heads)
    normalized: list[str] = []
    for value in raw_values:
        for item in str(value).split(","):
            head = item.strip().lower()
            if not head:
                continue
            if head not in HYWORLD2_CHECKPOINT_HEAD_GROUPS:
                raise ValueError(f"unsupported HY-World head: {head!r}")
            if head not in normalized:
                normalized.append(head)
    if not normalized:
        raise ValueError("at least one HY-World head must be requested")
    return tuple(normalized)


def _required_checkpoint_groups(requested_heads: Sequence[str]) -> tuple[str, ...]:
    required = ["visual_geometry_transformer"]
    for head in requested_heads:
        group = HYWORLD2_CHECKPOINT_HEAD_GROUPS[head]
        if group not in required:
            required.append(group)
        for extra_group in HYWORLD2_CHECKPOINT_EXTRA_HEAD_GROUPS.get(head, ()):
            if extra_group not in required:
                required.append(extra_group)
    return tuple(required)


def _extract_model_config(raw: dict[str, object]) -> dict[str, object]:
    if isinstance(raw.get("wrapper"), dict):
        wrapper = raw["wrapper"]
        if isinstance(wrapper.get("model"), dict):
            return dict(wrapper["model"])
    if isinstance(raw.get("model"), dict):
        return dict(raw["model"])
    return dict(raw)


def _coerce_model_config(raw: dict[str, object], config_path: Path) -> HyWorld2ModelConfig:
    try:
        raw["condition_strategy"] = tuple(str(item) for item in raw["condition_strategy"])
        raw["depth"] = int(raw["depth"])
        raw["disable_gs_depth"] = bool(raw["disable_gs_depth"])
        raw["dpt_gradient_checkpoint"] = bool(raw["dpt_gradient_checkpoint"])
        raw["embed_dim"] = int(raw["embed_dim"])
        raw["enable_bf16"] = bool(raw["enable_bf16"])
        raw["enable_cam"] = bool(raw["enable_cam"])
        raw["enable_cond"] = bool(raw["enable_cond"])
        raw["enable_depth"] = bool(raw["enable_depth"])
        raw["enable_depth_mask"] = bool(raw["enable_depth_mask"])
        raw["enable_gs"] = bool(raw["enable_gs"])
        raw["enable_norm"] = bool(raw["enable_norm"])
        raw["enable_pts"] = bool(raw["enable_pts"])
        raw["fixed_patch_embed"] = bool(raw["fixed_patch_embed"])
        raw["gs_dim"] = int(raw["gs_dim"])
        raw["img_size"] = int(raw["img_size"])
        raw["mlp_ratio"] = float(raw["mlp_ratio"])
        raw["model_size"] = str(raw["model_size"])
        raw["normalized_rope"] = bool(raw["normalized_rope"])
        raw["num_heads"] = int(raw["num_heads"])
        raw["num_register_tokens"] = int(raw["num_register_tokens"])
        raw["patch_embed"] = str(raw["patch_embed"])
        raw["patch_size"] = int(raw["patch_size"])
        raw["rope_base"] = float(raw["rope_base"])
        raw["rope_normalize_coords"] = (
            None if raw["rope_normalize_coords"] is None else str(raw["rope_normalize_coords"])
        )
        raw["sampling_strategy"] = str(raw["sampling_strategy"])
        raw["set_sky_region_to_maxdepth"] = bool(raw["set_sky_region_to_maxdepth"])
        raw["sp_size"] = int(raw["sp_size"])
        fields = HyWorld2ModelConfig.__dataclass_fields__
        return HyWorld2ModelConfig(**{name: raw[name] for name in fields})
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"WorldMirror config has an invalid model field in {config_path}: {error}") from error


def _read_yaml_config(path: Path) -> dict[str, object]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return _parse_simple_yaml_mapping(path.read_text(encoding="utf-8"))

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as error:  # pragma: no cover - exact PyYAML exception varies by version.
        raise ValueError(f"WorldMirror YAML config is invalid: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("WorldMirror YAML config must be a mapping")
    return raw


def _parse_simple_yaml_mapping(text: str) -> dict[str, object]:
    root: dict[str, object] = {}
    stack: list[tuple[int, dict[str, object]]] = [(-1, root)]
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].rstrip()
        if not stripped:
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        content = stripped.strip()
        if ":" not in content:
            raise ValueError("WorldMirror YAML config uses unsupported syntax")
        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError("WorldMirror YAML indentation is invalid")
        parent = stack[-1][1]
        if value:
            parent[key] = _parse_simple_yaml_scalar(value)
            continue
        child: dict[str, object] = {}
        parent[key] = child
        stack.append((indent, child))
    return root


def _parse_simple_yaml_scalar(value: str) -> object:
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_simple_yaml_scalar(item.strip()) for item in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
