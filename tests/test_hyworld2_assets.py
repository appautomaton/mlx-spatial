import json

import mlx.core as mx
from safetensors.mlx import save_file

from mlx_spatial.hyworld2_assets import (
    HYWORLD2_COMPONENT_GROUPS,
    HYWORLD2_WORLDMIRROR_SUBFOLDER,
    group_hyworld2_checkpoint_keys,
    inspect_hyworld2_model_assets,
    read_hyworld2_model_config,
)


def _write_model_dir(root, *, config_text=None, config_name="config.json", omit_groups=()):
    model_dir = root / HYWORLD2_WORLDMIRROR_SUBFOLDER
    model_dir.mkdir(parents=True)
    if config_text is None:
        config_text = json.dumps({"model": {"model_size": "small", "embed_dim": 64}})
    (model_dir / config_name).write_text(config_text, encoding="utf-8")
    tensors = {
        f"{group}.weight": mx.array([index + 1.0], dtype=mx.float32)
        for index, group in enumerate(HYWORLD2_COMPONENT_GROUPS)
        if group not in omit_groups
    }
    save_file(tensors, model_dir / "model.safetensors")
    return model_dir


def test_read_hyworld2_json_model_config_applies_official_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model_size": "small", "embed_dim": 64}), encoding="utf-8")

    config = read_hyworld2_model_config(path)

    assert config.model_size == "small"
    assert config.embed_dim == 64
    assert config.img_size == 518
    assert config.patch_size == 14
    assert config.depth == 24
    assert config.num_heads == 16
    assert config.visual_geometry_transformer["patch_embed"] == "dinov2_vitl14_reg"
    assert config.camera_head == {
        "enabled": True,
        "embed_dim": 64,
        "num_register_tokens": 4,
    }
    assert config.dpt_heads["enable_depth"] is True
    assert config.dpt_heads["enable_norm"] is True
    assert config.dpt_heads["enable_pts"] is True
    assert config.dpt_heads["gs_dim"] == 256


def test_read_hyworld2_yaml_wrapper_model_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
wrapper:
  model:
    model_size: tiny-fixture
    img_size: 252
    patch_size: 14
    embed_dim: 32
    depth: 2
    num_heads: 4
    condition_strategy: [token, pow3r, token]
""".strip(),
        encoding="utf-8",
    )

    config = read_hyworld2_model_config(path)

    assert config.model_size == "tiny-fixture"
    assert config.img_size == 252
    assert config.embed_dim == 32
    assert config.condition_strategy == ("token", "pow3r", "token")


def test_group_hyworld2_checkpoint_keys_maps_official_components():
    groups = group_hyworld2_checkpoint_keys(
        [
            "depth_head.norm.weight",
            "visual_geometry_transformer.blocks.0.weight",
            "unrelated.weight",
            "cam_head.token_norm.weight",
            "gs_renderer.opacity.weight",
        ]
    )

    by_name = {group.name: group for group in groups}
    assert tuple(by_name) == HYWORLD2_COMPONENT_GROUPS
    assert by_name["visual_geometry_transformer"].keys == (
        "visual_geometry_transformer.blocks.0.weight",
    )
    assert by_name["cam_head"].keys == ("cam_head.token_norm.weight",)
    assert by_name["gs_renderer"].keys == ("gs_renderer.opacity.weight",)
    assert by_name["pts_head"].keys == ()


def test_inspect_hyworld2_model_assets_accepts_complete_fake_fixture(tmp_path):
    _write_model_dir(tmp_path)

    inspection = inspect_hyworld2_model_assets(tmp_path)

    assert inspection.ready
    assert inspection.blocker is None
    assert inspection.config is not None
    assert inspection.config.model_size == "small"
    assert inspection.config.embed_dim == 64
    assert inspection.missing_groups == ()
    assert [group.name for group in inspection.groups] == list(HYWORLD2_COMPONENT_GROUPS)


def test_inspect_hyworld2_model_assets_allows_first_milestone_without_gaussian_groups(tmp_path):
    _write_model_dir(tmp_path, omit_groups=("gs_head", "gs_renderer"))

    inspection = inspect_hyworld2_model_assets(
        tmp_path,
        requested_heads=("depth", "normal", "points"),
    )

    assert inspection.ready
    assert inspection.blocker is None
    assert inspection.missing_groups == ("gs_head", "gs_renderer")


def test_inspect_hyworld2_model_assets_reports_missing_requested_component_groups(tmp_path):
    _write_model_dir(tmp_path, omit_groups=("cam_head", "norm_head", "gs_head", "gs_renderer"))

    inspection = inspect_hyworld2_model_assets(tmp_path, requested_heads=("camera", "normal", "gs"))

    assert not inspection.ready
    assert inspection.blocker is not None
    assert inspection.blocker.stage == "checkpoint-inspection"
    assert inspection.blocker.operation == "route HY-World safetensors keys into WorldMirror components"
    assert inspection.blocker.metadata["missing_groups"] == ("cam_head", "norm_head", "gs_head", "gs_renderer")
    assert inspection.blocker.metadata["required_groups"] == (
        "visual_geometry_transformer",
        "cam_head",
        "norm_head",
        "gs_head",
        "gs_renderer",
    )
    assert "cam_head, norm_head, gs_head, gs_renderer" in inspection.blocker.reason


def test_inspect_hyworld2_model_assets_returns_blocker_for_corrupt_safetensors(tmp_path):
    model_dir = tmp_path / HYWORLD2_WORLDMIRROR_SUBFOLDER
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"model": {"model_size": "small"}}), encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"not a safetensors checkpoint")

    inspection = inspect_hyworld2_model_assets(tmp_path, requested_heads=("depth", "normal", "points"))

    assert not inspection.ready
    assert inspection.blocker is not None
    assert inspection.blocker.stage == "checkpoint-inspection"
    assert inspection.blocker.operation == "inspect HY-World safetensors checkpoint metadata"
    assert inspection.blocker.metadata["checkpoint"] == str(model_dir / "model.safetensors")
    assert inspection.blocker.metadata["error_type"]
