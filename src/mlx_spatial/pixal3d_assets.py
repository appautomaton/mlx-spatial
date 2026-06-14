"""Pixal3D asset validation and checkpoint inspection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .checkpoint import CheckpointTensorInfo, inspect_checkpoint
from .model_assets import PIXAL3D_ASSETS, ModelAssetValidation, validate_model_assets


PIXAL3D_REPO_ID = "TencentARC/Pixal3D"
PIXAL3D_DEFAULT_ROOT = PIXAL3D_ASSETS.root_hint
PIXAL3D_LICENSE_NOTE = (
    "TencentARC/Pixal3D model metadata is MIT and the HF model card marks "
    "extra_gated_eu_disallowed=true; authenticate and review access terms before downloading."
)


@dataclass(frozen=True)
class Pixal3DProbeGroup:
    """Named tensor selection for one Pixal3D checkpoint file."""

    name: str
    checkpoint_path: str
    names: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()
    reference: str = ""


@dataclass(frozen=True)
class Pixal3DModelAsset:
    """One model entry discovered from Pixal3D pipeline.json."""

    key: str
    base_path: str
    config_path: str
    checkpoint_path: str


@dataclass(frozen=True)
class Pixal3DSamplerConfig:
    """Pixal3D flow sampler settings from pipeline.json."""

    name: str
    sigma_min: float
    steps: int
    guidance_strength: float
    guidance_rescale: float
    guidance_interval: tuple[float, float]
    rescale_t: float


@dataclass(frozen=True)
class Pixal3DNormalizationConfig:
    """Per-channel structured latent normalization values."""

    mean: tuple[float, ...]
    std: tuple[float, ...]


@dataclass(frozen=True)
class Pixal3DPipelineConfig:
    """MLX-facing Pixal3D pipeline config subset."""

    default_pipeline_type: str
    models: tuple[Pixal3DModelAsset, ...]
    sparse_structure_sampler: Pixal3DSamplerConfig
    shape_slat_sampler: Pixal3DSamplerConfig
    texture_slat_sampler: Pixal3DSamplerConfig
    shape_slat_normalization: Pixal3DNormalizationConfig
    texture_slat_normalization: Pixal3DNormalizationConfig


PIXAL3D_PROBE_GROUPS = (
    Pixal3DProbeGroup(
        name="sparse-structure-flow",
        checkpoint_path="ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
        names=("blocks.0.norm2.weight",),
        reference="Pixal3D sparse structure stage uses ss_flow_img_dit_1_3B_64_bf16 with projection attention.",
    ),
    Pixal3DProbeGroup(
        name="sparse-structure-decoder",
        checkpoint_path="ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
        prefixes=("layers.", "out_layer."),
        reference="Pixal3D sparse occupancy decoder maps sparse structure latent to occupied coordinates.",
    ),
    Pixal3DProbeGroup(
        name="shape-slat-flow-512",
        checkpoint_path="ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
        names=("blocks.0.norm2.weight", "blocks.0.cross_attn.proj_linear.weight"),
        reference="Pixal3D shape LR stage uses 512 SLat flow with projected image features.",
    ),
    Pixal3DProbeGroup(
        name="shape-slat-flow-1024",
        checkpoint_path="ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
        names=("blocks.0.norm2.weight", "blocks.0.cross_attn.proj_linear.weight"),
        reference="Pixal3D shape HR stage uses 1024 SLat flow for 1024/1536 cascade routes.",
    ),
    Pixal3DProbeGroup(
        name="texture-slat-flow-1024",
        checkpoint_path="ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
        names=("blocks.0.norm2.weight", "blocks.0.cross_attn.proj_linear.weight"),
        reference="Pixal3D texture stage uses 1024 SLat flow conditioned on shape SLat plus projected image features.",
    ),
    Pixal3DProbeGroup(
        name="shape-decoder",
        checkpoint_path="ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        names=("blocks.0.0.norm.weight",),
        reference="Pixal3D shape decoder follows the TRELLIS.2 FlexiDualGrid decoder family.",
    ),
    Pixal3DProbeGroup(
        name="texture-decoder",
        checkpoint_path="ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
        names=("blocks.0.0.norm.weight",),
        reference="Pixal3D texture decoder produces sparse PBR voxel attributes.",
    ),
)


def validate_pixal3d_assets(root: str | Path = PIXAL3D_DEFAULT_ROOT) -> ModelAssetValidation:
    """Validate a local Pixal3D asset root without loading large checkpoints."""

    return validate_model_assets(root, PIXAL3D_ASSETS)


def pixal3d_probe_group(name: str) -> Pixal3DProbeGroup:
    """Return a named Pixal3D probe group."""

    for group in PIXAL3D_PROBE_GROUPS:
        if group.name == name:
            return group
    raise ValueError(f"unknown Pixal3D probe group: {name!r}")


def read_pixal3d_pipeline_config(root: str | Path = PIXAL3D_DEFAULT_ROOT) -> Pixal3DPipelineConfig:
    """Read the Pixal3D pipeline config needed by MLX inference planning."""

    path = Path(root) / "pipeline.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        args = payload["args"]
        models = tuple(_model_asset(key, value) for key, value in sorted(args["models"].items()))
        return Pixal3DPipelineConfig(
            default_pipeline_type=str(args.get("default_pipeline_type", "1536_cascade")),
            models=models,
            sparse_structure_sampler=_sampler_config(args["sparse_structure_sampler"]),
            shape_slat_sampler=_sampler_config(args["shape_slat_sampler"]),
            texture_slat_sampler=_sampler_config(args["tex_slat_sampler"]),
            shape_slat_normalization=_normalization_config(args["shape_slat_normalization"]),
            texture_slat_normalization=_normalization_config(args["tex_slat_normalization"]),
        )
    except FileNotFoundError:
        raise
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Pixal3D pipeline config is invalid: {error}") from error


def inspect_pixal3d_checkpoints(
    root: str | Path = PIXAL3D_DEFAULT_ROOT,
    *,
    checkpoint_paths: Iterable[str] | None = None,
) -> dict[str, tuple[CheckpointTensorInfo, ...]]:
    """Inspect configured Pixal3D safetensors checkpoints under a local root."""

    root_path = Path(root)
    paths = tuple(checkpoint_paths) if checkpoint_paths is not None else tuple(group.checkpoint_path for group in PIXAL3D_PROBE_GROUPS)
    return {relative_path: inspect_checkpoint(root_path / relative_path) for relative_path in paths}


def inspect_pixal3d_probe(
    root: str | Path,
    group: str | Pixal3DProbeGroup,
) -> tuple[CheckpointTensorInfo, ...]:
    """Inspect tensors matched by a named Pixal3D probe group."""

    probe_group = pixal3d_probe_group(group) if isinstance(group, str) else group
    return inspect_checkpoint(
        Path(root) / probe_group.checkpoint_path,
        names=probe_group.names or None,
        prefixes=probe_group.prefixes or None,
    )


def pixal3d_download_command(root: str | Path = PIXAL3D_DEFAULT_ROOT) -> tuple[str, ...]:
    """Return the dev-environment HF command for downloading Pixal3D assets."""

    return (
        "uv",
        "run",
        "hf",
        "download",
        PIXAL3D_REPO_ID,
        "--local-dir",
        str(root),
    )


def _model_asset(key: str, base_path: str) -> Pixal3DModelAsset:
    return Pixal3DModelAsset(
        key=str(key),
        base_path=str(base_path),
        config_path=f"{base_path}.json",
        checkpoint_path=f"{base_path}.safetensors",
    )


def _sampler_config(raw: dict[str, object]) -> Pixal3DSamplerConfig:
    args = raw.get("args", {})
    params = raw["params"]
    if not isinstance(args, dict) or not isinstance(params, dict):
        raise ValueError("sampler args and params must be mappings")
    interval = params["guidance_interval"]
    if not isinstance(interval, list | tuple) or len(interval) != 2:
        raise ValueError("guidance_interval must contain two values")
    return Pixal3DSamplerConfig(
        name=str(raw["name"]),
        sigma_min=float(args.get("sigma_min", 1e-5)),
        steps=int(params["steps"]),
        guidance_strength=float(params["guidance_strength"]),
        guidance_rescale=float(params["guidance_rescale"]),
        guidance_interval=(float(interval[0]), float(interval[1])),
        rescale_t=float(params["rescale_t"]),
    )


def _normalization_config(raw: dict[str, object]) -> Pixal3DNormalizationConfig:
    mean = raw["mean"]
    std = raw["std"]
    if not isinstance(mean, list | tuple) or not isinstance(std, list | tuple):
        raise ValueError("normalization mean/std must be sequences")
    return Pixal3DNormalizationConfig(
        mean=tuple(float(value) for value in mean),
        std=tuple(float(value) for value in std),
    )
