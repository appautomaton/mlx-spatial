import json
import tomllib
from types import SimpleNamespace

import mlx.core as mx
from safetensors.mlx import save_file

import mlx_spatial
import mlx_spatial.trellis2_inference as trellis2_inference
from mlx_spatial.model_assets import DINOv3_VITL16_ASSETS, TRELLIS2_ASSETS
from mlx_spatial.trellis2 import (
    DINOv3_ACCESS_NOTE,
    DINOv3_VITL16_REPO_ID,
    RMBG2_LICENSE_NOTE,
    RMBG2_REPO_ID,
    TRELLIS2_PROBE_GROUPS,
    TRELLIS2_REPO_ID,
    Trellis2ProbeGroup,
    dinov3_download_command,
    inspect_trellis2_checkpoints,
    inspect_trellis2_probe,
    load_trellis2_probe,
    main,
    rmbg2_download_command,
    trellis2_download_command,
    trellis2_probe_group,
    validate_dinov3_assets,
    validate_rmbg2_assets,
    validate_trellis2_assets,
)


def _write_trellis2_root(root):
    for asset_path in TRELLIS2_ASSETS.required_paths:
        path = root / asset_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if asset_path.endswith(".safetensors"):
            save_file(
                {
                    "blocks.0.0.norm.weight": mx.array([7.0, 8.0], dtype=mx.float32),
                    "blocks.0.norm2.weight": mx.array([9.0, 10.0], dtype=mx.float32),
                    "blocks.0.weight": mx.array([[1.0, 2.0]], dtype=mx.float32),
                    "decoder.layer.weight": mx.array([3.0, 4.0], dtype=mx.float32),
                    "time_embed.weight": mx.array([5.0], dtype=mx.float32),
                    "unmatched.weight": mx.array([6.0], dtype=mx.float32),
                },
                path,
            )
        else:
            path.write_text("{}")


def test_hugging_face_cli_is_dev_only_dependency():
    config = tomllib.loads(open("pyproject.toml", "rb").read().decode())
    runtime_dependencies = "\n".join(config["project"]["dependencies"]).lower()
    dev_dependencies = "\n".join(config["dependency-groups"]["dev"]).lower()

    assert "huggingface" not in runtime_dependencies
    assert "torch" not in runtime_dependencies
    assert "transformers" not in runtime_dependencies
    assert "huggingface-hub" in dev_dependencies


def test_probe_groups_are_named_and_reference_backed():
    assert [group.name for group in TRELLIS2_PROBE_GROUPS] == [
        "sparse-structure-flow",
        "shape-slat-flow",
        "texture-slat-flow",
        "shape-decoder",
        "texture-decoder",
    ]
    for group in TRELLIS2_PROBE_GROUPS:
        assert group.checkpoint_path in TRELLIS2_ASSETS.required_paths
        assert group.names or group.prefixes
        assert "trellis" in group.reference.lower()


def test_validate_trellis2_assets_reports_fake_root_deterministically(tmp_path):
    _write_trellis2_root(tmp_path)

    result = validate_trellis2_assets(tmp_path)

    assert result.ready
    assert result.present == TRELLIS2_ASSETS.required_paths
    assert result.missing == ()


def test_inspect_trellis2_checkpoints_returns_grouped_metadata(tmp_path):
    _write_trellis2_root(tmp_path)
    checkpoint_path = "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"

    result = inspect_trellis2_checkpoints(tmp_path, checkpoint_paths=[checkpoint_path])

    assert list(result) == [checkpoint_path]
    assert [info.name for info in result[checkpoint_path]] == [
        "blocks.0.0.norm.weight",
        "blocks.0.norm2.weight",
        "blocks.0.weight",
        "decoder.layer.weight",
        "time_embed.weight",
        "unmatched.weight",
    ]


def test_inspect_trellis2_probe_filters_by_named_group(tmp_path):
    _write_trellis2_root(tmp_path)

    infos = inspect_trellis2_probe(tmp_path, "sparse-structure-flow")

    assert [info.name for info in infos] == ["blocks.0.norm2.weight"]


def test_load_trellis2_probe_loads_mlx_arrays_deterministically(tmp_path):
    _write_trellis2_root(tmp_path)

    tensors = load_trellis2_probe(tmp_path, "shape-decoder")

    assert [tensor.name for tensor in tensors] == ["blocks.0.0.norm.weight"]
    assert tensors[0].group == "shape-decoder"
    assert tensors[0].checkpoint_path == "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors"
    assert tensors[0].shape == (2,)
    assert tensors[0].dtype == "float32"
    assert tensors[0].array.tolist() == [7.0, 8.0]


def test_trellis2_tools_reject_invalid_inputs(tmp_path):
    _write_trellis2_root(tmp_path)
    invalid_group = Trellis2ProbeGroup(
        name="invalid",
        checkpoint_path="ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
    )
    no_match_group = Trellis2ProbeGroup(
        name="missing",
        checkpoint_path="ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
        prefixes=("missing.",),
    )

    invalid_cases = (
        lambda: validate_trellis2_assets(tmp_path / "missing"),
        lambda: inspect_trellis2_checkpoints(tmp_path / "missing"),
        lambda: inspect_trellis2_checkpoints(tmp_path, checkpoint_paths=[]),
        lambda: inspect_trellis2_checkpoints(tmp_path, checkpoint_paths=["ckpts/model.pt"]),
        lambda: inspect_trellis2_checkpoints(tmp_path, checkpoint_paths=["ckpts/missing.safetensors"]),
        lambda: trellis2_probe_group("missing"),
        lambda: load_trellis2_probe(tmp_path, invalid_group),
        lambda: load_trellis2_probe(tmp_path, no_match_group),
    )
    expected_errors = (
        FileNotFoundError,
        FileNotFoundError,
        ValueError,
        ValueError,
        FileNotFoundError,
        ValueError,
        ValueError,
        ValueError,
    )

    for call, expected_error in zip(invalid_cases, expected_errors, strict=True):
        try:
            call()
        except expected_error:
            pass
        else:
            raise AssertionError("invalid TRELLIS.2 tool input should be rejected")


def test_download_command_uses_dev_hugging_face_cli_and_local_root():
    command = trellis2_download_command("weights/trellis2")

    assert command[:4] == ("uv", "run", "hf", "download")
    assert TRELLIS2_REPO_ID in command
    assert "weights/trellis2" in command


def test_rmbg2_asset_validation_and_download_command_are_explicit(tmp_path):
    validation = validate_rmbg2_assets(tmp_path)

    assert not validation.ready
    assert validation.name == "BRIA RMBG-2.0"
    assert validation.missing == (
        "model.safetensors",
        "config.json",
        "BiRefNet_config.py",
        "birefnet.py",
    )

    command = rmbg2_download_command("weights/rmbg2")

    assert command[:4] == ("uv", "run", "hf", "download")
    assert RMBG2_REPO_ID in command
    assert "weights/rmbg2" in command
    assert "gated" in RMBG2_LICENSE_NOTE
    assert "non-commercial" in RMBG2_LICENSE_NOTE


def test_dinov3_asset_validation_and_download_command_are_explicit(tmp_path):
    validation = validate_dinov3_assets(tmp_path)

    assert not validation.ready
    assert validation.name == "DINOv3 ViT-L/16"
    assert validation.missing == DINOv3_VITL16_ASSETS.required_paths

    command = dinov3_download_command("weights/dinov3-vitl16-pretrain-lvd1689m")

    assert command[:4] == ("uv", "run", "hf", "download")
    assert DINOv3_VITL16_REPO_ID in command
    assert "config.json" in command
    assert "model.safetensors" in command
    assert "weights/dinov3-vitl16-pretrain-lvd1689m" in command
    assert "authentication" in DINOv3_ACCESS_NOTE


def test_cli_runs_against_fake_fixtures(tmp_path, capsys):
    _write_trellis2_root(tmp_path)

    assert main(["--root", str(tmp_path), "validate"]) == 0
    assert "ready=True" in capsys.readouterr().out

    assert main(["validate", "--root", str(tmp_path)]) == 0
    assert "ready=True" in capsys.readouterr().out

    assert main(["--root", str(tmp_path), "probe", "sparse-structure-flow"]) == 0
    assert "blocks.0.norm2.weight" in capsys.readouterr().out

    assert main(["--root", str(tmp_path), "probe", "shape-decoder", "--load"]) == 0
    assert "group=shape-decoder" in capsys.readouterr().out

    assert main(["--root", str(tmp_path), "download-command"]) == 0
    assert "hf download" in capsys.readouterr().out

    assert main(["rmbg-validate", "--root", str(tmp_path / "missing-rmbg")]) == 0
    assert "model.safetensors" in capsys.readouterr().out

    assert main(["rmbg-download-command", "--root", "weights/rmbg2"]) == 0
    rmbg_help = capsys.readouterr().out
    assert "briaai/RMBG-2.0" in rmbg_help
    assert "non-commercial" in rmbg_help

    assert main(["dinov3-validate", str(tmp_path / "missing-dinov3")]) == 0
    assert "model.safetensors" in capsys.readouterr().out

    assert main(["dinov3-download-command", "weights/dinov3-vitl16-pretrain-lvd1689m"]) == 0
    dinov3_help = capsys.readouterr().out
    assert "facebook/dinov3-vitl16-pretrain-lvd1689m" in dinov3_help
    assert "authentication" in dinov3_help

    trace_output = tmp_path / "trace.json"
    assert main(["attempt-forward-trace", str(tmp_path), str(tmp_path / "missing.png"), "--output", str(trace_output)]) == 0
    trace_help = capsys.readouterr().out
    assert "blocker_stage=input-image" in trace_help
    trace_payload = json.loads(trace_output.read_text())
    assert trace_payload["blocker"]["stage"] == "input-image"

    assert main(["generate-shape", str(tmp_path), str(tmp_path / "missing.png"), "--output", str(tmp_path / "bad.obj")]) == 2
    shape_help = capsys.readouterr().out
    assert "blocker_stage=mesh-export" in shape_help
    assert "must stay under outputs" in shape_help

    assert main(["generate-textured", str(tmp_path), str(tmp_path / "missing.png"), "--output", str(tmp_path / "bad.obj")]) == 2
    textured_help = capsys.readouterr().out
    assert "blocker_stage=mesh-export" in textured_help
    assert "operation=textured GLB output format validation" in textured_help
    assert "reason=generate-textured only writes .glb outputs" in textured_help


def test_generate_textured_cli_accepts_shared_generation_flags(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _write_trellis2_root(tmp_path / "trellis")

    status = main(
        [
            "generate-textured",
            str(tmp_path / "trellis"),
            str(tmp_path / "missing.png"),
            "--output",
            "outputs/trellis2/demo.glb",
            "--pipeline-type",
            "512",
            "--seed",
            "7",
            "--dino-root",
            str(tmp_path / "dino"),
            "--slat-steps",
            "2",
            "--max-num-tokens",
            "32",
            "--decoder-token-limit",
            "64",
            "--texture-size",
            "128",
            "--glb-target-faces",
            "100000",
            "--xatlas-face-guard",
            "auto",
            "--xatlas-parallel-chunks",
            "4",
            "--texture-bake-backend",
            "trilinear",
        ]
    )

    output = capsys.readouterr().out
    assert status == 2
    assert "blocker_stage=" in output
    assert "operation=" in output


def test_generate_shape_cli_forwards_generation_flags(tmp_path, monkeypatch, capsys):
    calls = {}

    class FakePipeline:
        def __init__(self, root, *, rmbg_root=None):
            calls["root"] = root
            calls["rmbg_root"] = rmbg_root

        def generate_shape_obj(self, image, **kwargs):
            calls["image"] = image
            calls["kwargs"] = kwargs
            return SimpleNamespace(
                trace=SimpleNamespace(completed_stages=("mesh-export",), outputs=(), blocker=None),
                artifact=SimpleNamespace(path=tmp_path / "outputs/trellis2/demo.obj", bytes_written=10),
            )

    monkeypatch.setattr(trellis2_inference, "Trellis2InferencePipeline", FakePipeline)

    status = main(
        [
            "generate-shape",
            str(tmp_path / "trellis"),
            str(tmp_path / "image.png"),
            "--output",
            "outputs/trellis2/demo.obj",
            "--pipeline-type",
            "512",
            "--seed",
            "7",
            "--dino-root",
            str(tmp_path / "dino"),
            "--rmbg-root",
            str(tmp_path / "rmbg"),
            "--slat-steps",
            "2",
            "--max-num-tokens",
            "32",
            "--decoder-token-limit",
            "64",
        ]
    )

    output = capsys.readouterr().out
    assert status == 0
    assert "artifact=" in output
    assert calls["kwargs"]["pipeline_type"] == "512"
    assert calls["kwargs"]["seed"] == 7
    assert calls["kwargs"]["max_num_tokens"] == 32
    assert calls["kwargs"]["slat_steps"] == 2
    assert calls["kwargs"]["decoder_token_limit"] == 64
    assert calls["kwargs"]["dino_root"] == str(tmp_path / "dino")
    assert calls["rmbg_root"] == str(tmp_path / "rmbg")


def test_trellis2_helpers_are_public_exports():
    assert mlx_spatial.TRELLIS2_PROBE_GROUPS is TRELLIS2_PROBE_GROUPS
    assert mlx_spatial.TRELLIS2_REPO_ID == TRELLIS2_REPO_ID
    assert mlx_spatial.validate_trellis2_assets is validate_trellis2_assets
    assert mlx_spatial.inspect_trellis2_checkpoints is inspect_trellis2_checkpoints
    assert mlx_spatial.inspect_trellis2_probe is inspect_trellis2_probe
    assert mlx_spatial.load_trellis2_probe is load_trellis2_probe
    assert mlx_spatial.validate_dinov3_assets is validate_dinov3_assets
    assert mlx_spatial.validate_rmbg2_assets is validate_rmbg2_assets
    assert mlx_spatial.dinov3_download_command is dinov3_download_command
    assert mlx_spatial.rmbg2_download_command is rmbg2_download_command
    assert mlx_spatial.trellis2_download_command is trellis2_download_command
