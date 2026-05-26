"""Dependency-free model asset manifests and local validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class ModelAssetManifest:
    """Expected local files for an external model asset bundle."""

    name: str
    root_hint: str
    required_paths: tuple[str, ...]


@dataclass(frozen=True)
class ModelAssetValidation:
    """Deterministic presence report for a model asset manifest."""

    name: str
    root: Path
    present: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.missing


TRELLIS2_ASSETS = ModelAssetManifest(
    name="TRELLIS.2",
    root_hint="weights/trellis2",
    required_paths=(
        "pipeline.json",
        "texturing_pipeline.json",
        "ckpts/ss_flow_img_dit_1_3B_64_bf16.json",
        "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
        "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json",
        "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
        "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json",
        "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
        "ckpts/shape_enc_next_dc_f16c32_fp16.json",
        "ckpts/shape_enc_next_dc_f16c32_fp16.safetensors",
        "ckpts/shape_dec_next_dc_f16c32_fp16.json",
        "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        "ckpts/tex_dec_next_dc_f16c32_fp16.json",
        "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
    ),
)

PIXAL3D_ASSETS = ModelAssetManifest(
    name="Pixal3D",
    root_hint="weights/pixal3d",
    required_paths=(
        "pipeline.json",
        "ckpts/ss_flow_img_dit_1_3B_64_bf16.json",
        "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
        "ckpts/ss_dec_conv3d_16l8_fp16.json",
        "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
        "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json",
        "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
        "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json",
        "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
        "ckpts/shape_dec_next_dc_f16c32_fp16.json",
        "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        "ckpts/tex_dec_next_dc_f16c32_fp16.json",
        "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
    ),
)

DINOv3_VITL16_ASSETS = ModelAssetManifest(
    name="DINOv3 ViT-L/16",
    root_hint="weights/dinov3-vitl16-pretrain-lvd1689m",
    required_paths=(
        "config.json",
        "model.safetensors",
    ),
)

RMBG2_ASSETS = ModelAssetManifest(
    name="BRIA RMBG-2.0",
    root_hint="weights/rmbg2",
    required_paths=(
        "model.safetensors",
        "config.json",
        "BiRefNet_config.py",
        "birefnet.py",
    ),
)


def validate_model_assets(
    root: str | Path,
    manifest: ModelAssetManifest = TRELLIS2_ASSETS,
) -> ModelAssetValidation:
    """Check a local asset root without downloading or loading checkpoints."""

    root_path = Path(root)
    present: list[str] = []
    missing: list[str] = []

    for relative_path in manifest.required_paths:
        _validate_relative_asset_path(relative_path)
        if (root_path / relative_path).is_file():
            present.append(relative_path)
        else:
            missing.append(relative_path)

    return ModelAssetValidation(
        name=manifest.name,
        root=root_path,
        present=tuple(present),
        missing=tuple(missing),
    )


def _validate_relative_asset_path(relative_path: str) -> None:
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"asset path must be relative and confined: {relative_path!r}")
