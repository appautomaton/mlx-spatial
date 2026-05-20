import json
import shutil
from pathlib import Path

import mlx.core as mx
from safetensors.mlx import save_file

from mlx_spatial.hyworld2 import main
from mlx_spatial.hyworld2_assets import HYWORLD2_COMPONENT_GROUPS, HYWORLD2_WORLDMIRROR_SUBFOLDER
from mlx_spatial.hyworld2_inference import (
    HyWorld2InferencePipeline,
    _intermediate_layers_for_hyworld2,
    normalize_hyworld2_heads,
    validate_hyworld2_output_path,
)


def _write_hyworld2_root(root):
    _write_hyworld2_root_with_omissions(root)


def _write_hyworld2_root_with_omissions(root, *, omit_groups=()):
    model_dir = root / HYWORLD2_WORLDMIRROR_SUBFOLDER
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model": {"model_size": "small"}}', encoding="utf-8")
    save_file(
        {
            f"{group}.weight": mx.array([index + 1.0], dtype=mx.float32)
            for index, group in enumerate(HYWORLD2_COMPONENT_GROUPS)
            if group not in omit_groups
        },
        model_dir / "model.safetensors",
    )


def _write_tiny_fixture_root(root):
    model_dir = root / HYWORLD2_WORLDMIRROR_SUBFOLDER
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model": {
                    "model_size": "tiny-fixture",
                    "img_size": 28,
                    "patch_size": 14,
                    "embed_dim": 8,
                    "depth": 4,
                    "num_heads": 2,
                    "mlp_ratio": 2.0,
                    "num_register_tokens": 2,
                    "enable_depth_mask": True,
                }
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            f"{group}.weight": mx.array([index + 1.0], dtype=mx.float32)
            for index, group in enumerate(HYWORLD2_COMPONENT_GROUPS)
        },
        model_dir / "model.safetensors",
    )


def _write_fixture_images(path):
    from PIL import Image

    path.mkdir()
    Image.new("RGB", (28, 28), (10, 20, 30)).save(path / "a.png")
    Image.new("RGB", (28, 28), (40, 50, 60)).save(path / "b.png")


def _output_dir(name):
    path = Path("outputs") / "hyworld2" / name
    shutil.rmtree(path, ignore_errors=True)
    return path


def _assert_local_mlx_timing(metadata, completed_stages, *, blocker_stage, successful):
    timing = metadata["mlx_timing"]

    assert timing["runtime"] == "mlx"
    assert timing["timing_source"] == "local_mlx_time_perf_counter"
    assert timing["clock"] == "time.perf_counter"
    assert timing["source_reference_timings_included"] is False
    assert timing["total_elapsed_seconds"] >= 0.0
    assert timing["completed_stages"] == list(completed_stages)
    assert set(timing["stage_elapsed_seconds"]) == set(completed_stages)
    assert all(value >= 0.0 for value in timing["stage_elapsed_seconds"].values())
    assert timing["blocker_stage"] == blocker_stage
    assert timing["successful_inference"] is successful
    assert timing["speed_parity_claimed"] is False
    assert timing["numeric_parity_claimed"] is False


def test_normalize_hyworld2_heads_accepts_comma_lists_and_dedupes():
    assert normalize_hyworld2_heads("depth,normal,depth,points") == ("depth", "normal", "points")


def test_normalize_hyworld2_heads_rejects_invalid_or_empty():
    for value in ("", "depth,texture"):
        try:
            normalize_hyworld2_heads(value)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid HY-World heads should be rejected")


def test_validate_hyworld2_output_path_rejects_paths_outside_outputs(tmp_path):
    blocker = validate_hyworld2_output_path(tmp_path / "bad")

    assert blocker is not None
    assert blocker.stage == "output-path"
    assert "must stay under outputs" in blocker.reason


def test_reconstruct_blocks_on_missing_assets_before_missing_input(tmp_path):
    result = HyWorld2InferencePipeline(tmp_path / "missing-weights").reconstruct(
        tmp_path / "missing-input",
        output_path="outputs/hyworld2/demo",
    )

    assert result.trace.completed_stages == ()
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "asset-validation"
    assert result.trace.blocker.metadata["missing"] == (
        "HY-WorldMirror-2.0/model.safetensors",
        "HY-WorldMirror-2.0/config.yaml or HY-WorldMirror-2.0/config.json",
    )


def test_reconstruct_blocks_on_missing_input_after_asset_validation(tmp_path):
    _write_hyworld2_root(tmp_path)

    result = HyWorld2InferencePipeline(tmp_path).reconstruct(
        tmp_path / "missing-input",
        output_path="outputs/hyworld2/demo",
        heads=("depth", "points"),
        memory_profile="safe",
    )

    assert result.trace.completed_stages == ("asset-validation",)
    assert result.trace.requested_heads == ("depth", "points")
    assert result.trace.memory_profile == "safe"
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "input-discovery"


def test_reconstruct_inspects_checkpoint_before_model_construction_blocker(tmp_path):
    _write_hyworld2_root_with_omissions(tmp_path, omit_groups=("gs_head", "gs_renderer"))
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    from PIL import Image

    Image.new("RGB", (640, 480), (1, 2, 3)).save(image_dir / "a.png")

    result = HyWorld2InferencePipeline(tmp_path).reconstruct(
        image_dir,
        output_path="outputs/hyworld2/demo",
    )

    assert result.trace.completed_stages == (
        "asset-validation",
        "input-discovery",
        "image-preprocessing",
        "checkpoint-inspection",
    )
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "model-construction"
    assert result.trace.enabled_heads == ()
    assert result.trace.metadata["input"]["frame_count"] == 1
    assert result.trace.metadata["input"]["patch_grid"] == (34, 45)
    assert result.trace.metadata["checkpoint"]["ready"] is True
    assert result.trace.metadata["checkpoint"]["missing_groups"] == ("gs_head", "gs_renderer")
    assert result.trace.metadata["checkpoint"]["model_config"]["img_size"] == 518
    assert result.trace.metadata["checkpoint"]["component_groups"]["visual_geometry_transformer"]["present"] is True
    assert result.trace.metadata["checkpoint"]["component_groups"]["gs_head"]["present"] is False
    assert result.trace.metadata["checkpoint"]["component_groups"]["gs_renderer"]["present"] is False
    assert result.trace.metadata["heads"]["depth"]["requested"] is True
    assert result.trace.metadata["heads"]["depth"]["enabled"] is False
    assert result.trace.metadata["heads"]["gs"]["reason"] == "not requested"


def test_reconstruct_reports_missing_checkpoint_groups(tmp_path):
    model_dir = tmp_path / HYWORLD2_WORLDMIRROR_SUBFOLDER
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model": {"model_size": "small"}}', encoding="utf-8")
    save_file(
        {
            "visual_geometry_transformer.weight": mx.array([1.0], dtype=mx.float32),
            "cam_head.weight": mx.array([2.0], dtype=mx.float32),
        },
        model_dir / "model.safetensors",
    )
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    from PIL import Image

    Image.new("RGB", (640, 480), (1, 2, 3)).save(image_dir / "a.png")

    result = HyWorld2InferencePipeline(tmp_path).reconstruct(
        image_dir,
        output_path="outputs/hyworld2/demo",
    )

    assert result.trace.completed_stages == ("asset-validation", "input-discovery", "image-preprocessing")
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "checkpoint-inspection"
    assert result.trace.blocker.metadata["missing_groups"] == (
        "pts_head",
        "depth_head",
        "norm_head",
    )


def test_reconstruct_reports_corrupt_checkpoint_as_structured_blocker(tmp_path):
    model_dir = tmp_path / HYWORLD2_WORLDMIRROR_SUBFOLDER
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model": {"model_size": "small"}}', encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"not a safetensors checkpoint")
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    from PIL import Image

    Image.new("RGB", (640, 480), (1, 2, 3)).save(image_dir / "a.png")

    result = HyWorld2InferencePipeline(tmp_path).reconstruct(
        image_dir,
        output_path="outputs/hyworld2/demo",
    )

    assert result.trace.completed_stages == ("asset-validation", "input-discovery", "image-preprocessing")
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "checkpoint-inspection"
    assert result.trace.blocker.operation == "inspect HY-World safetensors checkpoint metadata"
    assert result.trace.blocker.metadata["checkpoint"] == str(model_dir / "model.safetensors")
    assert result.trace.blocker.metadata["error_type"]


def test_hyworld2_reconstruct_cli_writes_trace_and_reports_blocker(tmp_path, capsys):
    _write_hyworld2_root(tmp_path)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    from PIL import Image

    Image.new("RGB", (640, 480), (1, 2, 3)).save(image_dir / "a.png")
    trace_output = tmp_path / "trace.json"

    assert main(
        [
            "reconstruct",
            str(tmp_path),
            str(image_dir),
            "--output",
            "outputs/hyworld2/demo",
            "--heads",
            "depth,normal,points",
            "--trace-output",
            str(trace_output),
        ]
    ) == 2

    output = capsys.readouterr().out
    assert "blocker_stage=model-construction" in output
    payload = json.loads(trace_output.read_text())
    assert payload["completed_stages"] == [
        "asset-validation",
        "input-discovery",
        "image-preprocessing",
        "checkpoint-inspection",
    ]
    assert payload["requested_heads"] == ["depth", "normal", "points"]
    assert payload["memory_profile"] == "large"
    assert payload["metadata"]["official_defaults"]["matches_official_target_size"] is True
    assert payload["metadata"]["input"]["token_count"] == 34 * 45
    assert payload["metadata"]["checkpoint"]["ready"] is True
    _assert_local_mlx_timing(
        payload["metadata"],
        payload["completed_stages"],
        blocker_stage="model-construction",
        successful=False,
    )


def test_hyworld2_intermediate_layers_follow_official_model_size_map():
    assert _intermediate_layers_for_hyworld2("small", 12, False) == (2, 5, 8, 11)
    assert _intermediate_layers_for_hyworld2("base", 12, False) == (2, 5, 8, 11)
    assert _intermediate_layers_for_hyworld2("large", 24, False) == (4, 11, 17, 23)
    assert _intermediate_layers_for_hyworld2("giant", 40, False) == (9, 19, 29, 39)
    assert _intermediate_layers_for_hyworld2("fixture", 4, True) == (0, 1, 2, 3)


def test_hyworld2_reconstruct_cli_rejects_output_outside_outputs(tmp_path, capsys):
    _write_hyworld2_root(tmp_path)
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    assert main(["reconstruct", str(tmp_path), str(image_dir), "--output", str(tmp_path / "bad")]) == 2

    output = capsys.readouterr().out
    assert "blocker_stage=output-path" in output
    assert "must stay under outputs" in output


def test_fixture_reconstruct_writes_staged_outputs_under_outputs(tmp_path):
    _write_tiny_fixture_root(tmp_path)
    image_dir = tmp_path / "images"
    _write_fixture_images(image_dir)
    out = _output_dir("fixture-reconstruct")

    result = HyWorld2InferencePipeline(tmp_path).reconstruct(
        image_dir,
        output_path=out,
        heads=("camera", "depth", "normal", "points"),
        memory_profile="safe",
        fixture_tensors=True,
    )

    assert result.trace.blocker is None
    assert result.output_dir == out
    assert result.trace.completed_stages == (
        "asset-validation",
        "input-discovery",
        "image-preprocessing",
        "checkpoint-inspection",
        "model-construction",
        "visual-transformer",
        "head-execution",
        "export",
    )
    assert result.trace.enabled_heads == ("camera", "depth", "normal", "points")
    assert (out / "depth" / "depth.npy").stat().st_size > 0
    assert (out / "normal" / "normal.npy").stat().st_size > 0
    assert (out / "camera_params.json").stat().st_size > 0
    assert (out / "points" / "points.ply").stat().st_size > 0
    trace = json.loads((out / "trace.json").read_text())
    assert trace["metadata"]["fixture_tensors"] is True
    assert trace["metadata"]["heads"]["points"]["export"] is True
    assert trace["outputs"]
    _assert_local_mlx_timing(
        trace["metadata"],
        trace["completed_stages"],
        blocker_stage=None,
        successful=True,
    )


def test_fixture_reconstruct_writes_optional_mlx_parity_bundle(tmp_path):
    _write_tiny_fixture_root(tmp_path)
    image_dir = tmp_path / "images"
    _write_fixture_images(image_dir)
    out = _output_dir("fixture-parity-bundle")
    parity_output = Path("outputs") / "hyworld2" / "fixture-parity-bundle.npz"
    parity_output.unlink(missing_ok=True)

    result = HyWorld2InferencePipeline(tmp_path).reconstruct(
        image_dir,
        output_path=out,
        heads=("camera", "depth"),
        memory_profile="safe",
        fixture_tensors=True,
        parity_output_path=parity_output,
    )

    assert result.trace.blocker is None
    assert parity_output.is_file()
    assert result.trace.metadata["parity"]["runtime_depends_on_torch"] is False
    assert result.trace.metadata["parity"]["numeric_parity_verified"] is False
    assert result.trace.metadata["parity"]["mlx_bundle_path"] == str(parity_output)
    assert "parity-mlx-bundle" in [output.name for output in result.trace.outputs]


def test_fixture_reconstruct_heads_depth_exports_only_depth(tmp_path):
    _write_tiny_fixture_root(tmp_path)
    image_dir = tmp_path / "images"
    _write_fixture_images(image_dir)
    out = _output_dir("fixture-depth-only")

    result = HyWorld2InferencePipeline(tmp_path).reconstruct(
        image_dir,
        output_path=out,
        heads=("depth",),
        memory_profile="safe",
        fixture_tensors=True,
    )

    assert result.trace.blocker is None
    assert result.trace.enabled_heads == ("depth",)
    assert (out / "depth" / "depth.npy").is_file()
    assert not (out / "normal").exists()
    assert not (out / "points").exists()
    assert not (out / "camera").exists()
    heads = result.trace.metadata["heads"]
    assert heads["depth"] == {"requested": True, "enabled": True, "export": True, "reason": "exported"}
    assert heads["normal"] == {"requested": False, "enabled": False, "export": False, "reason": "not requested"}
    assert heads["points"] == {"requested": False, "enabled": False, "export": False, "reason": "not requested"}
    assert heads["camera"] == {"requested": False, "enabled": False, "export": False, "reason": "not requested"}


def test_fixture_reconstruct_cleans_stale_artifacts_when_heads_change(tmp_path):
    _write_tiny_fixture_root(tmp_path)
    image_dir = tmp_path / "images"
    _write_fixture_images(image_dir)
    out = _output_dir("fixture-rerun-same-output")
    pipeline = HyWorld2InferencePipeline(tmp_path)

    first = pipeline.reconstruct(
        image_dir,
        output_path=out,
        heads=("camera", "depth", "normal", "points"),
        memory_profile="safe",
        fixture_tensors=True,
    )
    assert first.trace.blocker is None
    assert (out / "camera_params.json").is_file()
    assert (out / "normal" / "normal.npy").is_file()
    assert (out / "points" / "points.ply").is_file()

    second = pipeline.reconstruct(
        image_dir,
        output_path=out,
        heads=("depth",),
        memory_profile="safe",
        fixture_tensors=True,
    )

    assert second.trace.blocker is None
    assert second.trace.enabled_heads == ("depth",)
    assert (out / "depth" / "depth.npy").is_file()
    assert not (out / "camera").exists()
    assert not (out / "normal").exists()
    assert not (out / "points").exists()
    trace = json.loads((out / "trace.json").read_text())
    assert trace["requested_heads"] == ["depth"]
    assert trace["enabled_heads"] == ["depth"]
    assert trace["metadata"]["heads"]["depth"] == {
        "requested": True,
        "enabled": True,
        "export": True,
        "reason": "exported",
    }
    assert trace["metadata"]["heads"]["camera"]["reason"] == "not requested"
    assert trace["metadata"]["heads"]["normal"]["reason"] == "not requested"
    assert trace["metadata"]["heads"]["points"]["reason"] == "not requested"


def test_fixture_reconstruct_requested_gs_exports_gaussians_ply(tmp_path):
    _write_tiny_fixture_root(tmp_path)
    image_dir = tmp_path / "images"
    _write_fixture_images(image_dir)
    out = _output_dir("fixture-gs-ply")

    result = HyWorld2InferencePipeline(tmp_path).reconstruct(
        image_dir,
        output_path=out,
        heads=("gs",),
        memory_profile="safe",
        fixture_tensors=True,
    )

    assert result.trace.blocker is None
    assert result.trace.requested_heads == ("gs",)
    assert result.trace.enabled_heads == ("camera", "gs")
    assert not (out / "points").exists()
    assert not (out / "camera_params.json").exists()
    assert (out / "gaussian" / "attributes.npz").is_file()
    assert (out / "gaussian" / "metadata.json").is_file()
    assert (out / "gaussians.ply").is_file()
    assert result.trace.metadata["gaussian"] == {
        "gaussians_ply": "exported",
        "point_cloud_ply": "points.ply is a point-cloud artifact, not 3DGS",
        "renderer": "MLX gs_renderer attribute conv; no CUDA rasterization",
        "means_source": "gsdepth+predcamera",
        "camera_dependency": "executed",
        "requires_cuda_gsplat": False,
    }
    assert result.trace.metadata["heads"]["camera"] == {
        "requested": False,
        "enabled": True,
        "export": False,
        "reason": "executed as required dependency for official GS export",
    }
    assert result.trace.metadata["heads"]["gs"] == {
        "requested": True,
        "enabled": True,
        "export": True,
        "reason": "exported",
    }
    assert (out / "trace.json").is_file()


def test_hyworld2_reconstruct_cli_fixture_tensors_depth_only(tmp_path, capsys):
    _write_tiny_fixture_root(tmp_path)
    image_dir = tmp_path / "images"
    _write_fixture_images(image_dir)
    out = _output_dir("cli-fixture-depth")

    assert main(
        [
            "reconstruct",
            str(tmp_path),
            str(image_dir),
            "--output",
            str(out),
            "--heads",
            "depth",
            "--memory-profile",
            "safe",
            "--fixture-tensors",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "outputs=('depth', 'depth-confidence', 'depth-preview-000', 'depth-preview-001', 'trace')" in output
    assert (out / "depth" / "depth.npy").is_file()
    assert not (out / "normal").exists()
