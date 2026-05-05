import tomllib

import mlx.core as mx
from safetensors.mlx import save_file

from mlx_spatial.hyworld2 import main
from mlx_spatial.hyworld2_assets import (
    HYWORLD2_REPO_ID,
    HYWORLD2_WORLDMIRROR_SUBFOLDER,
    hyworld2_download_command,
    inspect_hyworld2_checkpoint,
    validate_hyworld2_assets,
)


def _write_hyworld2_root(root, *, config_name="config.json"):
    model_dir = root / HYWORLD2_WORLDMIRROR_SUBFOLDER
    model_dir.mkdir(parents=True)
    (model_dir / config_name).write_text('{"model": {"model_size": "small"}}', encoding="utf-8")
    save_file(
        {
            "visual_geometry_transformer.patch_embed.proj.weight": mx.array([1.0], dtype=mx.float32),
            "cam_head.token_norm.weight": mx.array([2.0], dtype=mx.float32),
            "depth_head.norm.weight": mx.array([3.0], dtype=mx.float32),
            "pts_head.norm.weight": mx.array([4.0], dtype=mx.float32),
        },
        model_dir / "model.safetensors",
    )
    return model_dir


def test_hyworld2_runtime_dependencies_exclude_torch_cuda_and_hf_runtime_clients():
    config = tomllib.loads(open("pyproject.toml", "rb").read().decode())
    runtime_dependencies = "\n".join(config["project"]["dependencies"]).lower()
    dev_dependencies = "\n".join(config["dependency-groups"]["dev"]).lower()

    for forbidden in ("torch", "cuda", "gsplat", "flash-attn", "xfuser", "deepspeed", "huggingface"):
        assert forbidden not in runtime_dependencies
    assert "huggingface-hub" in dev_dependencies


def test_validate_hyworld2_assets_reports_missing_worldmirror_layout(tmp_path):
    validation = validate_hyworld2_assets(tmp_path)

    assert not validation.ready
    assert validation.model_dir == tmp_path / HYWORLD2_WORLDMIRROR_SUBFOLDER
    assert validation.present == ()
    assert validation.missing == (
        f"{HYWORLD2_WORLDMIRROR_SUBFOLDER}/model.safetensors",
        f"{HYWORLD2_WORLDMIRROR_SUBFOLDER}/config.yaml or {HYWORLD2_WORLDMIRROR_SUBFOLDER}/config.json",
    )


def test_validate_hyworld2_assets_accepts_json_or_yaml_config(tmp_path):
    _write_hyworld2_root(tmp_path, config_name="config.yaml")

    validation = validate_hyworld2_assets(tmp_path)

    assert validation.ready
    assert validation.config_kind == "yaml"
    assert validation.present == (
        f"{HYWORLD2_WORLDMIRROR_SUBFOLDER}/model.safetensors",
        f"{HYWORLD2_WORLDMIRROR_SUBFOLDER}/config.yaml",
    )


def test_validate_hyworld2_assets_accepts_direct_model_dir(tmp_path):
    model_dir = _write_hyworld2_root(tmp_path)

    validation = validate_hyworld2_assets(model_dir)

    assert validation.ready
    assert validation.model_dir == model_dir
    assert validation.present == ("model.safetensors", "config.json")


def test_inspect_hyworld2_checkpoint_filters_by_prefix(tmp_path):
    _write_hyworld2_root(tmp_path)

    infos = inspect_hyworld2_checkpoint(tmp_path, prefixes=("depth_head.",))

    assert [info.name for info in infos] == ["depth_head.norm.weight"]


def test_hyworld2_download_command_is_explicit_and_dev_cli_based():
    command = hyworld2_download_command("weights/hy-world-2")

    assert command[:4] == ("uv", "run", "hf", "download")
    assert HYWORLD2_REPO_ID in command
    assert "--include" in command
    assert f"{HYWORLD2_WORLDMIRROR_SUBFOLDER}/*" in command
    assert "weights/hy-world-2" in command


def test_hyworld2_cli_runs_against_fake_fixture(tmp_path, capsys):
    _write_hyworld2_root(tmp_path)

    assert main(["validate", str(tmp_path)]) == 0
    validate_output = capsys.readouterr().out
    assert "ready=True" in validate_output
    assert "config_kind=json" in validate_output

    assert main(["inspect", str(tmp_path), "--prefix", "cam_head.", "--limit", "5"]) == 0
    inspect_output = capsys.readouterr().out
    assert "cam_head.token_norm.weight" in inspect_output

    assert main(["download-command", "weights/hy-world-2"]) == 0
    download_output = capsys.readouterr().out
    assert "hf download" in download_output
    assert "tencent/HY-World-2.0" in download_output


def test_hyworld2_cli_inspect_returns_setup_blocker_for_missing_checkpoint(tmp_path, capsys):
    assert main(["inspect", str(tmp_path)]) == 2

    output = capsys.readouterr().out
    assert "ready=False" in output
    assert "missing HY-WorldMirror-2.0/model.safetensors" in output
