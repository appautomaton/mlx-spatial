from pathlib import PurePosixPath

import mlx_spatial
from mlx_spatial.model_assets import DINOv3_VITL16_ASSETS, RMBG2_ASSETS, TRELLIS2_ASSETS, validate_model_assets


def test_trellis2_manifest_names_model_and_relative_asset_paths():
    assert TRELLIS2_ASSETS.name == "TRELLIS.2"
    assert TRELLIS2_ASSETS.root_hint == "weights/trellis2"
    assert "pipeline.json" in TRELLIS2_ASSETS.required_paths
    assert "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors" in TRELLIS2_ASSETS.required_paths

    for asset_path in TRELLIS2_ASSETS.required_paths:
        parsed = PurePosixPath(asset_path)
        assert not parsed.is_absolute()
        assert ".." not in parsed.parts


def test_validate_model_assets_reports_all_missing_for_empty_root(tmp_path):
    result = validate_model_assets(tmp_path)

    assert result.name == "TRELLIS.2"
    assert result.root == tmp_path
    assert result.present == ()
    assert result.missing == TRELLIS2_ASSETS.required_paths
    assert not result.ready


def test_validate_model_assets_reports_present_files_deterministically(tmp_path):
    present_paths = TRELLIS2_ASSETS.required_paths[:2] + TRELLIS2_ASSETS.required_paths[4:5]
    for asset_path in present_paths:
        fake_file = tmp_path / asset_path
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.write_bytes(b"fake")

    result = validate_model_assets(tmp_path)

    assert result.present == present_paths
    assert result.missing == (
        TRELLIS2_ASSETS.required_paths[2],
        TRELLIS2_ASSETS.required_paths[3],
        *TRELLIS2_ASSETS.required_paths[5:],
    )
    assert not result.ready


def test_validate_model_assets_reports_ready_when_all_files_exist(tmp_path):
    for asset_path in TRELLIS2_ASSETS.required_paths:
        fake_file = tmp_path / asset_path
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.write_bytes(b"fake")

    result = validate_model_assets(tmp_path)

    assert result.present == TRELLIS2_ASSETS.required_paths
    assert result.missing == ()
    assert result.ready


def test_rmbg2_manifest_names_gated_assets_and_relative_paths():
    assert RMBG2_ASSETS.name == "BRIA RMBG-2.0"
    assert RMBG2_ASSETS.root_hint == "weights/rmbg2"
    assert RMBG2_ASSETS.required_paths == (
        "model.safetensors",
        "config.json",
        "BiRefNet_config.py",
        "birefnet.py",
    )

    for asset_path in RMBG2_ASSETS.required_paths:
        parsed = PurePosixPath(asset_path)
        assert not parsed.is_absolute()
        assert ".." not in parsed.parts


def test_dinov3_manifest_names_local_assets_and_relative_paths():
    assert DINOv3_VITL16_ASSETS.name == "DINOv3 ViT-L/16"
    assert DINOv3_VITL16_ASSETS.root_hint == "weights/dinov3-vitl16-pretrain-lvd1689m"
    assert DINOv3_VITL16_ASSETS.required_paths == ("config.json", "model.safetensors")

    for asset_path in DINOv3_VITL16_ASSETS.required_paths:
        parsed = PurePosixPath(asset_path)
        assert not parsed.is_absolute()
        assert ".." not in parsed.parts


def test_model_asset_helpers_are_public_exports():
    assert mlx_spatial.TRELLIS2_ASSETS is TRELLIS2_ASSETS
    assert mlx_spatial.RMBG2_ASSETS is RMBG2_ASSETS
    assert mlx_spatial.DINOv3_VITL16_ASSETS is DINOv3_VITL16_ASSETS
    assert mlx_spatial.validate_model_assets is validate_model_assets
