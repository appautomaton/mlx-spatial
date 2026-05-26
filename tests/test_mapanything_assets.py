import json
from pathlib import Path

import mlx.core as mx
import pytest
from safetensors.mlx import save_file

import mlx_spatial
from mlx_spatial.mapanything_assets import (
    MAPANYTHING_COMPONENT_GROUPS,
    MAPANYTHING_DEFAULT_ROOT,
    MapAnythingModelConfig,
    group_mapanything_checkpoint_keys,
    inspect_mapanything_model_assets,
    mapanything_download_command,
    read_mapanything_model_config,
    validate_mapanything_assets,
)


def _minimal_config(**overrides):
    config = {
        "encoder_config": {
            "data_norm_type": "dinov2",
            "name": "dinov2_giant_24_layers",
            "size": "giant",
            "keep_first_n_layers": 24,
            "uses_torch_hub": True,
        },
        "info_sharing_config": {
            "model_type": "alternating_attention",
            "model_return_type": "intermediate_features",
            "module_args": {
                "depth": 16,
                "dim": 1536,
                "num_heads": 24,
                "indices": [7, 11],
            },
        },
        "pred_head_config": {
            "type": "dpt+pose",
            "adaptor_type": "raydirs+depth+pose+confidence+mask",
            "feature_head": {"patch_size": 14},
            "adaptor_config": {
                "dense_pred_init_dict": {
                    "name": "raydirs+depth+pose+confidence+mask+scale",
                },
            },
        },
        "use_register_tokens_from_encoder": True,
    }
    config.update(overrides)
    return config


def _write_model_root(root, *, config=None, omit_groups=()):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config or _minimal_config()), encoding="utf-8")
    tensors = {
        f"{group}.weight": mx.array([index + 1.0], dtype=mx.float32)
        for index, group in enumerate(MAPANYTHING_COMPONENT_GROUPS)
        if group not in omit_groups
    }
    save_file(tensors, root / "model.safetensors")


def test_read_mapanything_config_extracts_mlx_facing_fields(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_minimal_config()), encoding="utf-8")

    config = read_mapanything_model_config(path)

    assert isinstance(config, MapAnythingModelConfig)
    assert config.data_norm_type == "dinov2"
    assert config.encoder_name == "dinov2_giant_24_layers"
    assert config.encoder_size == "giant"
    assert config.encoder_keep_first_n_layers == 24
    assert config.encoder_uses_torch_hub is True
    assert config.encoder_with_registers is False
    assert config.patch_size == 14
    assert config.info_sharing_model_type == "alternating_attention"
    assert config.info_sharing_depth == 16
    assert config.info_sharing_dim == 1536
    assert config.info_sharing_num_heads == 24
    assert config.info_sharing_indices == (7, 11)
    assert config.pred_head_type == "dpt+pose"
    assert config.pred_head_adaptor_type == "raydirs+depth+pose+confidence+mask"
    assert config.pred_head_output_type == "raydirs+depth+pose+confidence+mask+scale"
    assert config.encoder["patch_size"] == 14
    assert config.encoder["with_registers"] is False
    assert config.info_sharing["indices"] == (7, 11)
    assert config.prediction["output_type"] == "raydirs+depth+pose+confidence+mask+scale"


def test_read_mapanything_config_reports_invalid_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"encoder_config": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid model field"):
        read_mapanything_model_config(path)


def test_validate_mapanything_assets_reports_missing_and_ready(tmp_path):
    missing = validate_mapanything_assets(tmp_path)

    assert missing.root == tmp_path
    assert missing.present == ()
    assert missing.missing == ("config.json", "model.safetensors")
    assert not missing.ready

    _write_model_root(tmp_path)
    ready = validate_mapanything_assets(tmp_path)

    assert ready.present == ("config.json", "model.safetensors")
    assert ready.missing == ()
    assert ready.ready


def test_group_mapanything_checkpoint_keys_maps_known_components():
    groups = group_mapanything_checkpoint_keys(
        [
            "encoder.model.patch_embed.proj.weight",
            "info_sharing.self_attention_blocks.0.attn.qkv.weight",
            "ray_dirs_encoder.conv_in.weight",
            "depth_encoder.conv_in.weight",
            "dense_head.0.scratch.output_conv.weight",
            "pose_head.fc_t.weight",
            "scale_head.output_proj.weight",
            "unrelated.weight",
        ]
    )

    by_name = {group.name: group for group in groups}
    assert tuple(by_name) == MAPANYTHING_COMPONENT_GROUPS
    assert by_name["encoder"].keys == ("encoder.model.patch_embed.proj.weight",)
    assert by_name["info_sharing"].keys == (
        "info_sharing.self_attention_blocks.0.attn.qkv.weight",
    )
    assert by_name["ray_dirs_encoder"].keys == ("ray_dirs_encoder.conv_in.weight",)
    assert by_name["depth_encoder"].keys == ("depth_encoder.conv_in.weight",)
    assert by_name["dense_head"].keys == ("dense_head.0.scratch.output_conv.weight",)
    assert by_name["pose_head"].keys == ("pose_head.fc_t.weight",)
    assert by_name["scale_head"].keys == ("scale_head.output_proj.weight",)
    assert by_name["scale_token"].keys == ()


def test_inspect_mapanything_model_assets_accepts_complete_fake_fixture(tmp_path):
    _write_model_root(tmp_path)

    inspection = inspect_mapanything_model_assets(tmp_path)

    assert inspection.ready
    assert inspection.blocker is None
    assert inspection.config is not None
    assert inspection.config.encoder_size == "giant"
    assert inspection.missing_groups == ()
    assert [group.name for group in inspection.groups] == list(MAPANYTHING_COMPONENT_GROUPS)


def test_inspect_mapanything_model_assets_reports_missing_required_groups(tmp_path):
    _write_model_root(tmp_path, omit_groups=("encoder", "info_sharing"))

    inspection = inspect_mapanything_model_assets(tmp_path)

    assert not inspection.ready
    assert inspection.blocker is not None
    assert inspection.blocker.stage == "checkpoint-inspection"
    assert inspection.blocker.operation == "route MapAnything safetensors keys into model components"
    assert inspection.blocker.metadata["missing_groups"] == ("encoder", "info_sharing")
    assert "encoder, info_sharing" in inspection.blocker.reason


def test_inspect_mapanything_model_assets_returns_blocker_for_corrupt_safetensors(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps(_minimal_config()), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"not a safetensors checkpoint")

    inspection = inspect_mapanything_model_assets(tmp_path)

    assert not inspection.ready
    assert inspection.blocker is not None
    assert inspection.blocker.stage == "checkpoint-inspection"
    assert inspection.blocker.operation == "inspect MapAnything safetensors checkpoint metadata"
    assert inspection.blocker.metadata["checkpoint"] == str(tmp_path / "model.safetensors")


def test_local_mapanything_checkpoint_layout_is_recognized_when_present():
    root = Path(MAPANYTHING_DEFAULT_ROOT)
    if not root.is_dir():
        pytest.skip(f"local MapAnything weights not present: {root}")

    inspection = inspect_mapanything_model_assets(root)

    assert inspection.ready
    assert inspection.config is not None
    assert inspection.config.data_norm_type == "dinov2"
    assert inspection.config.encoder_size == "giant"
    assert inspection.config.info_sharing_dim == 1536
    groups = {group.name: group for group in inspection.groups}
    assert len(groups["encoder"].keys) == 340
    assert len(groups["info_sharing"].keys) == 227
    assert len(groups["dense_head"].keys) == 60
    assert len(groups["pose_head"].keys) == 22
    assert len(groups["scale_head"].keys) == 8


def test_mapanything_download_command_points_to_best_performance_model():
    assert mapanything_download_command("weights/map-anything") == (
        "uv",
        "run",
        "hf",
        "download",
        "facebook/map-anything",
        "--local-dir",
        "weights/map-anything",
    )


def test_mapanything_asset_helpers_are_public_exports():
    assert mlx_spatial.MAPANYTHING_DEFAULT_ROOT == MAPANYTHING_DEFAULT_ROOT
    assert mlx_spatial.MapAnythingModelConfig is MapAnythingModelConfig
    assert mlx_spatial.validate_mapanything_assets is validate_mapanything_assets
    assert mlx_spatial.inspect_mapanything_model_assets is inspect_mapanything_model_assets
