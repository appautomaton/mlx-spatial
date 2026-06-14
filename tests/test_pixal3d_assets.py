import json

import mlx_spatial
from mlx_spatial.model_assets import PIXAL3D_ASSETS
from mlx_spatial.pixal3d_assets import (
    PIXAL3D_DEFAULT_ROOT,
    PIXAL3D_PROBE_GROUPS,
    PIXAL3D_REPO_ID,
    inspect_pixal3d_probe,
    pixal3d_download_command,
    read_pixal3d_pipeline_config,
    validate_pixal3d_assets,
)
from pixal3d_fixtures import write_fake_pixal3d_root


def test_validate_pixal3d_assets_reports_missing_and_ready(tmp_path):
    missing = validate_pixal3d_assets(tmp_path)

    assert missing.name == "Pixal3D"
    assert missing.root == tmp_path
    assert "pipeline.json" in missing.missing
    assert not missing.ready

    write_fake_pixal3d_root(tmp_path)
    ready = validate_pixal3d_assets(tmp_path)

    assert ready.ready
    assert ready.missing == ()
    assert len(ready.present) == len(PIXAL3D_ASSETS.required_paths)


def test_read_pixal3d_pipeline_config_extracts_runtime_fields(tmp_path):
    write_fake_pixal3d_root(tmp_path)

    config = read_pixal3d_pipeline_config(tmp_path)

    assert config.default_pipeline_type == "1536_cascade"
    assert [asset.key for asset in config.models] == [
        "shape_slat_decoder",
        "shape_slat_flow_model_1024",
        "shape_slat_flow_model_512",
        "sparse_structure_decoder",
        "sparse_structure_flow_model",
        "tex_slat_decoder",
        "tex_slat_flow_model_1024",
    ]
    assert config.sparse_structure_sampler.steps == 12
    assert config.sparse_structure_sampler.guidance_interval == (0.6, 1.0)
    assert config.shape_slat_sampler.guidance_rescale == 0.5
    assert config.texture_slat_sampler.guidance_interval == (0.6, 0.9)
    assert len(config.shape_slat_normalization.mean) == 32
    assert len(config.texture_slat_normalization.std) == 32


def test_read_pixal3d_pipeline_config_reports_invalid_config(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "pipeline.json").write_text(json.dumps({"args": {"models": {}}}), encoding="utf-8")

    try:
        read_pixal3d_pipeline_config(tmp_path)
    except ValueError as error:
        assert "invalid" in str(error)
    else:
        raise AssertionError("expected invalid Pixal3D pipeline config")


def test_inspect_pixal3d_probe_reads_named_checkpoint_group(tmp_path):
    write_fake_pixal3d_root(tmp_path)

    infos = inspect_pixal3d_probe(tmp_path, "shape-slat-flow-512")

    assert [info.name for info in infos] == [
        "blocks.0.cross_attn.proj_linear.weight",
        "blocks.0.norm2.weight",
    ]
    assert infos[0].shape == (1,)


def test_pixal3d_download_command_points_to_hf_model():
    assert pixal3d_download_command("weights/pixal3d") == (
        "uv",
        "run",
        "hf",
        "download",
        PIXAL3D_REPO_ID,
        "--local-dir",
        "weights/pixal3d",
    )


def test_pixal3d_asset_helpers_are_public_exports():
    assert mlx_spatial.PIXAL3D_ASSETS is PIXAL3D_ASSETS
    assert mlx_spatial.PIXAL3D_DEFAULT_ROOT == PIXAL3D_DEFAULT_ROOT
    assert mlx_spatial.PIXAL3D_PROBE_GROUPS == PIXAL3D_PROBE_GROUPS
    assert mlx_spatial.validate_pixal3d_assets is validate_pixal3d_assets
    assert mlx_spatial.read_pixal3d_pipeline_config is read_pixal3d_pipeline_config
