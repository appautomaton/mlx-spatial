import os
import subprocess
import sys
import tomllib
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image
from safetensors.mlx import save_file

import mlx_spatial
from mlx_spatial.mapanything_inference import (
    MAPANYTHING_PREFIX_PARITY_ATOL,
    MAPANYTHING_PREFIX_PARITY_RTOL,
    MapAnythingPrefixPipeline,
)
from mlx_spatial.mapanything_scene import (
    MAPANYTHING_SCENE_OUTPUT_KEYS,
    MapAnythingScenePipeline,
    MapAnythingScenePredictions,
    write_mapanything_scene_npz,
)


ROOT = Path(__file__).resolve().parents[1]


def test_mapanything_prefix_pipeline_runs_tiny_smoke_path(tmp_path):
    model_root = tmp_path / "weights"
    _write_tiny_model_root(model_root)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8)).save(image_dir / "desk.png")

    result = MapAnythingPrefixPipeline(model_root).run(
        image_dir,
        resize_mode="fixed_size",
        size=(4, 4),
    )

    assert result.ready
    assert result.preprocessed is not None
    assert result.prefix is not None
    assert result.trace.completed_stages == (
        "asset-config-validation",
        "image-preprocessing",
        "checkpoint-loading",
        "encoder-prefix",
    )
    assert result.trace.frame_count == 1
    assert result.trace.target_size == (4, 4)
    assert result.trace.patch_grid == (2, 2)
    assert result.trace.metadata["runtime_depends_on_torch"] is False
    assert result.trace.metadata["parity"]["documented_atol"] == MAPANYTHING_PREFIX_PARITY_ATOL
    assert result.trace.metadata["parity"]["documented_rtol"] == MAPANYTHING_PREFIX_PARITY_RTOL
    summaries = {summary.name: summary for summary in result.trace.tensor_summaries}
    assert summaries["encoder.patch_embed"].shape == (1, 4, 8)
    assert summaries["encoder.tokens"].shape == (1, 5, 8)
    assert summaries["encoder.block0"].shape == (1, 5, 8)


def test_mapanything_prefix_pipeline_reports_missing_assets(tmp_path):
    result = MapAnythingPrefixPipeline(tmp_path / "missing").run(tmp_path / "images")

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "asset-validation"
    assert "missing required MapAnything checkpoint assets" in result.trace.blocker.reason
    assert result.trace.metadata["runtime_depends_on_torch"] is False


def test_mapanything_scene_pipeline_reports_missing_scene_head_weights(tmp_path):
    model_root = tmp_path / "weights"
    _write_tiny_model_root(model_root)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8)).save(image_dir / "desk.png")

    result = MapAnythingScenePipeline(model_root).generate(
        image_dir,
        resize_mode="fixed_size",
        size=(4, 4),
    )

    assert not result.ready
    assert result.preprocessed is not None
    assert result.predictions is None
    assert result.trace.completed_stages == (
        "asset-config-validation",
        "image-preprocessing",
        "model-config",
        "checkpoint-loading:encoder",
        "full-encoder",
    )
    assert result.trace.frame_count == 1
    assert result.trace.target_size == (4, 4)
    assert result.trace.output_keys == MAPANYTHING_SCENE_OUTPUT_KEYS
    assert result.trace.metadata["runtime_depends_on_torch"] is False
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "checkpoint-loading"
    assert result.trace.blocker.operation == "load MapAnything prediction-head tensors"
    assert "checkpoint is missing requested tensors" in result.trace.blocker.reason


def test_mapanything_scene_pipeline_reports_missing_assets(tmp_path):
    result = MapAnythingScenePipeline(tmp_path / "missing").generate(tmp_path / "images")

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "asset-validation"
    assert "missing required MapAnything checkpoint assets" in result.trace.blocker.reason
    assert result.trace.metadata["runtime_depends_on_torch"] is False


def test_mapanything_scene_prediction_bundle_schema(tmp_path):
    predictions = MapAnythingScenePredictions(
        images=np.zeros((1, 2, 2, 3), dtype=np.float32),
        depth=np.ones((1, 2, 2), dtype=np.float32),
        confidence=np.ones((1, 2, 2), dtype=np.float32),
        masks=np.ones((1, 2, 2), dtype=bool),
        intrinsics=np.eye(3, dtype=np.float32)[None],
        camera_poses=np.eye(4, dtype=np.float32)[None],
        extrinsics=np.eye(4, dtype=np.float32)[None, :3],
        world_points=np.zeros((1, 2, 2, 3), dtype=np.float32),
        metadata={"case": "tiny"},
    )

    path = write_mapanything_scene_npz(tmp_path / "scene.npz", predictions)

    with np.load(path, allow_pickle=False) as data:
        assert set(MAPANYTHING_SCENE_OUTPUT_KEYS).issubset(data.files)
        assert data["world_points"].shape == (1, 2, 2, 3)
        assert "case" in str(data["__metadata_json__"])


def test_mapanything_prefix_pipeline_runs_local_desk_when_assets_present():
    model_root = ROOT / "weights/map-anything"
    image_root = ROOT / "inputs/map-anything/desk"
    if not (model_root / "model.safetensors").is_file() or not image_root.is_dir():
        pytest.skip("local MapAnything weights or Desk inputs are absent")

    result = MapAnythingPrefixPipeline(model_root).run(image_root)

    assert result.ready
    assert result.trace.frame_count == 2
    assert result.trace.target_size == (518, 392)
    assert result.trace.patch_grid == (28, 37)
    summaries = {summary.name: summary for summary in result.trace.tensor_summaries}
    assert summaries["encoder.patch_embed"].shape == (2, 1036, 1536)
    assert summaries["encoder.tokens"].shape == (2, 1037, 1536)
    assert summaries["encoder.block0"].shape == (2, 1037, 1536)


def test_mapanything_package_imports_without_vendor_pythonpath():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from mlx_spatial import MapAnythingPrefixPipeline, MapAnythingScenePipeline; "
            "print(MapAnythingPrefixPipeline.__name__, MapAnythingScenePipeline.__name__)",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "MapAnythingPrefixPipeline MapAnythingScenePipeline"


def test_mapanything_runtime_dependencies_exclude_torch_vendor_stack():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = "\n".join(pyproject["project"]["dependencies"]).lower()

    assert "torch" not in dependencies
    assert "torchvision" not in dependencies
    assert "uniception" not in dependencies
    assert "opencv-python" not in dependencies


def test_mapanything_cli_and_script_surfaces_are_registered():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["mlx-spatial-mapanything"] == "mlx_spatial.mapanything:main"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    module_help = subprocess.run(
        [sys.executable, "-m", "mlx_spatial.mapanything", "--help"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert module_help.returncode == 0, module_help.stderr
    assert "generate" in module_help.stdout

    script_help = subprocess.run(
        [sys.executable, "scripts/mapanything/generate_scene.py", "--help"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert script_help.returncode == 0, script_help.stderr
    assert "fixed_mapping" in script_help.stdout


def test_mapanything_download_command_cli_points_to_best_performance_model():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_spatial.mapanything",
            "download-command",
            "--root",
            "weights/map-anything",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "uv run hf download facebook/map-anything --local-dir weights/map-anything"


def test_mapanything_prefix_pipeline_public_exports():
    assert mlx_spatial.MapAnythingPrefixPipeline is MapAnythingPrefixPipeline
    assert mlx_spatial.MAPANYTHING_PREFIX_PARITY_ATOL == MAPANYTHING_PREFIX_PARITY_ATOL
    assert mlx_spatial.MapAnythingScenePipeline is MapAnythingScenePipeline
    assert mlx_spatial.MAPANYTHING_SCENE_OUTPUT_KEYS == MAPANYTHING_SCENE_OUTPUT_KEYS


def _write_tiny_model_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(_tiny_config_json(), encoding="utf-8")
    save_file(_tiny_encoder_prefix_weights(), root / "model.safetensors")


def _tiny_encoder_prefix_weights() -> dict[str, mx.array]:
    embed_dim = 8
    patch_size = 2
    hidden = 24
    zeros = lambda shape: mx.zeros(shape, dtype=mx.float32)
    ones = lambda shape: mx.ones(shape, dtype=mx.float32)
    return {
        "encoder.model.cls_token": zeros((1, 1, embed_dim)),
        "encoder.model.pos_embed": zeros((1, 5, embed_dim)),
        "encoder.model.patch_embed.proj.weight": zeros((embed_dim, 3, patch_size, patch_size)),
        "encoder.model.patch_embed.proj.bias": zeros((embed_dim,)),
        "encoder.model.blocks.0.norm1.weight": ones((embed_dim,)),
        "encoder.model.blocks.0.norm1.bias": zeros((embed_dim,)),
        "encoder.model.blocks.0.attn.qkv.weight": zeros((3 * embed_dim, embed_dim)),
        "encoder.model.blocks.0.attn.qkv.bias": zeros((3 * embed_dim,)),
        "encoder.model.blocks.0.attn.proj.weight": zeros((embed_dim, embed_dim)),
        "encoder.model.blocks.0.attn.proj.bias": zeros((embed_dim,)),
        "encoder.model.blocks.0.ls1.gamma": ones((embed_dim,)),
        "encoder.model.blocks.0.norm2.weight": ones((embed_dim,)),
        "encoder.model.blocks.0.norm2.bias": zeros((embed_dim,)),
        "encoder.model.blocks.0.mlp.w12.weight": zeros((2 * hidden, embed_dim)),
        "encoder.model.blocks.0.mlp.w12.bias": zeros((2 * hidden,)),
        "encoder.model.blocks.0.mlp.w3.weight": zeros((embed_dim, hidden)),
        "encoder.model.blocks.0.mlp.w3.bias": zeros((embed_dim,)),
        "encoder.model.blocks.0.ls2.gamma": ones((embed_dim,)),
        "info_sharing.dummy": zeros((1,)),
        "dense_head.dummy": zeros((1,)),
        "pose_head.dummy": zeros((1,)),
        "scale_head.dummy": zeros((1,)),
        "fusion_norm_layer.weight": ones((embed_dim,)),
        "scale_token": zeros((embed_dim,)),
    }


def _tiny_config_json() -> str:
    return """{
  "encoder_config": {
    "data_norm_type": "dinov2",
    "name": "tiny-test",
    "size": "giant",
    "keep_first_n_layers": 1,
    "uses_torch_hub": false,
    "with_registers": false
  },
  "info_sharing_config": {
    "model_type": "alternating_attention",
    "model_return_type": "intermediate_features",
    "module_args": {
      "depth": 1,
      "dim": 8,
      "num_heads": 2,
      "indices": [0]
    }
  },
  "pred_head_config": {
    "type": "dpt+pose",
    "adaptor_type": "raydirs+depth+pose+confidence+mask",
    "feature_head": {"patch_size": 2},
    "adaptor_config": {
      "dense_pred_init_dict": {"name": "raydirs+depth+pose+confidence+mask+scale"}
    }
  },
  "use_register_tokens_from_encoder": true
}"""
