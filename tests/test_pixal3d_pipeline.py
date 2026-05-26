import json

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_spatial.pixal3d_camera import pixal3d_stage_plan
from mlx_spatial.pixal3d_inference import Pixal3DInferencePipeline
from mlx_spatial.pixal3d_projection import PIXAL3D_DEFAULT_NUM_REGISTER_TOKENS
from pixal3d_fixtures import (
    write_fake_pixal3d_dinov3_root,
    write_fake_pixal3d_root,
    write_fake_pixal3d_shape_slat_root,
    write_fake_pixal3d_sparse_decoder_root,
    write_fake_pixal3d_sparse_flow_root,
)


def test_pixal3d_pipeline_manual_fov_records_camera_and_stage_plan(tmp_path):
    root = write_fake_pixal3d_root(tmp_path / "weights")
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")

    result = Pixal3DInferencePipeline(root).generate(image, manual_fov=0.2, pipeline_type="1536_cascade")

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "image-conditioning"
    assert result.trace.metadata["camera"].camera_angle_x == 0.2
    assert result.trace.metadata["stage_plan"].requested_hr_resolution == 1536
    assert "memory_before" in result.trace.metadata
    assert "memory_after" in result.trace.metadata


def test_pixal3d_pipeline_reports_missing_dinov3_assets_with_manual_fov(tmp_path):
    root = write_fake_pixal3d_root(tmp_path / "weights")
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")

    result = Pixal3DInferencePipeline(root).generate(
        image,
        manual_fov=0.2,
        dino_root=tmp_path / "missing-dinov3",
    )

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "image-conditioning"
    assert result.trace.blocker.operation == "local DINOv3 asset validation"
    assert result.trace.blocker.metadata["dino_root"] == str(tmp_path / "missing-dinov3")
    assert "hf download facebook/dinov3-vitl16-pretrain-lvd1689m" in result.trace.blocker.metadata["download_command"]
    assert result.trace.completed_stages == ("input-image", "asset-validation", "pipeline-config", "camera-setup")


def test_pixal3d_pipeline_reaches_sparse_projection_boundary_with_fake_dinov3_root(tmp_path):
    root = write_fake_pixal3d_root(tmp_path / "weights")
    dino_root = write_fake_pixal3d_dinov3_root(tmp_path / "dinov3")
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), (128, 64, 32)).save(image)

    result = Pixal3DInferencePipeline(root).generate(
        image,
        output_dir=tmp_path / "out",
        manual_fov=0.2,
        dino_root=dino_root,
    )

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "sparse-structure-flow"
    assert result.trace.completed_stages == (
        "input-image",
        "asset-validation",
        "pipeline-config",
        "camera-setup",
        "image-conditioning",
        "projection-conditioning:ss",
        "artifact:sparse_projection",
    )
    assert result.trace.metadata["dino_conditioning"]["root"] == str(dino_root)
    assert result.trace.metadata["dino_conditioning"]["shape"] == (1, 21, 1024)
    assert result.trace.metadata["ss_projection"]["projected_shape"] == (1, 16**3, 1024)
    assert len(result.artifacts) == 1
    assert result.artifacts[0].name == "sparse_projection.npz"
    assert result.artifacts[0].is_file()


def test_pixal3d_pipeline_reaches_sparse_projection_boundary_with_hidden_states(tmp_path):
    root = write_fake_pixal3d_root(tmp_path / "weights")
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")
    token_count = 1 + PIXAL3D_DEFAULT_NUM_REGISTER_TOKENS + 32 * 32
    hidden_states = mx.zeros((1, token_count, 3), dtype=mx.float32)

    result = Pixal3DInferencePipeline(root).generate(
        image,
        manual_fov=0.2,
        projection_hidden_states=hidden_states,
    )

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "sparse-structure-flow"
    assert result.trace.completed_stages[-2:] == ("projection-conditioning:ss", "artifact:sparse_projection")
    assert len(result.artifacts) == 1
    assert result.artifacts[0].name == "sparse_projection.npz"
    assert result.artifacts[0].is_file()
    assert result.trace.metadata["ss_projection"]["projected_shape"] == (1, 16**3, 3)


def test_pixal3d_pipeline_runs_sparse_flow_with_valid_fake_checkpoint(tmp_path):
    root = write_fake_pixal3d_sparse_flow_root(tmp_path / "weights", proj_in_channels=3, sparse_steps=1)
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")
    patch_grid = 32
    token_count = 1 + PIXAL3D_DEFAULT_NUM_REGISTER_TOKENS + patch_grid * patch_grid
    hidden_states = mx.zeros((1, token_count, 3), dtype=mx.float32)

    result = Pixal3DInferencePipeline(root).generate(
        image,
        output_dir=tmp_path / "out",
        manual_fov=0.2,
        projection_hidden_states=hidden_states,
    )

    assert not result.ready
    assert result.trace.completed_stages == (
        "input-image",
        "asset-validation",
        "pipeline-config",
        "camera-setup",
        "projection-conditioning:ss",
        "artifact:sparse_projection",
        "sparse-structure-flow",
    )
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "sparse-structure-decoding"
    assert result.trace.metadata["sparse_flow"]["sampled_latent_shape"] == (1, 2, 16, 16, 16)
    assert result.trace.metadata["sparse_flow"]["completed_blocks"] == 1
    assert result.trace.blocker.metadata["config_path"] == "ckpts/ss_dec_conv3d_16l8_fp16.json"


def test_pixal3d_pipeline_writes_sparse_structure_artifact_with_valid_fake_decoder(tmp_path):
    root = write_fake_pixal3d_sparse_decoder_root(tmp_path / "weights", proj_in_channels=3, sparse_steps=1)
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")
    patch_grid = 32
    token_count = 1 + PIXAL3D_DEFAULT_NUM_REGISTER_TOKENS + patch_grid * patch_grid
    hidden_states = mx.zeros((1, token_count, 3), dtype=mx.float32)

    result = Pixal3DInferencePipeline(root).generate(
        image,
        output_dir=tmp_path / "out",
        manual_fov=0.2,
        projection_hidden_states=hidden_states,
    )

    assert not result.ready
    assert result.trace.completed_stages == (
        "input-image",
        "asset-validation",
        "pipeline-config",
        "camera-setup",
        "projection-conditioning:ss",
        "artifact:sparse_projection",
        "sparse-structure-flow",
        "sparse-structure-decoding",
        "artifact:sparse_structure",
    )
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "shape-projection-conditioning"
    assert result.trace.blocker.operation == "build Pixal3D high-resolution projected features"
    assert len(result.artifacts) == 2
    assert [path.name for path in result.artifacts] == ["sparse_projection.npz", "sparse_structure.npz"]
    payload = np.load(result.artifacts[1])
    assert payload["coordinates"].shape == (4096, 4)
    assert payload["decoded_shape"].tolist() == [1, 1, 16, 16, 16]
    assert int(payload["target_resolution"]) == 16
    metadata = json.loads(payload["metadata_json"].item())
    assert metadata["coordinate_order"] == "batch,z,y,x"
    assert metadata["pipeline_type"] == "1024_cascade"
    assert metadata["blocker_next_target"] == "shape-projection-conditioning"


def test_pixal3d_pipeline_writes_shape_slat_lr_artifact_with_fake_naf_and_shape_flow(tmp_path):
    root = write_fake_pixal3d_shape_slat_root(tmp_path / "weights", proj_in_channels=3, sparse_steps=1, shape_steps=1)
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")
    patch_grid = 32
    token_count = 1 + PIXAL3D_DEFAULT_NUM_REGISTER_TOKENS + patch_grid * patch_grid
    hidden_states = mx.zeros((1, token_count, 3), dtype=mx.float32)
    naf = mx.zeros((1, patch_grid, patch_grid, 3), dtype=mx.float32)

    result = Pixal3DInferencePipeline(root).generate(
        image,
        output_dir=tmp_path / "out",
        manual_fov=0.2,
        projection_hidden_states=hidden_states,
        shape_lr_naf_feature_map=naf,
    )

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "shape-slat-cascade"
    assert result.trace.completed_stages == (
        "input-image",
        "asset-validation",
        "pipeline-config",
        "camera-setup",
        "projection-conditioning:ss",
        "artifact:sparse_projection",
        "sparse-structure-flow",
        "sparse-structure-decoding",
        "artifact:sparse_structure",
        "projection-conditioning:shape_512",
        "shape-slat-sampling:512",
        "artifact:shape_slat_lr",
    )
    assert [path.name for path in result.artifacts] == [
        "sparse_projection.npz",
        "sparse_structure.npz",
        "shape_slat_lr.npz",
    ]
    payload = np.load(result.artifacts[2])
    assert payload["coordinates"].shape == (4096, 4)
    assert payload["features"].shape == (4096, 32)
    metadata = json.loads(payload["metadata_json"].item())
    assert metadata["stage"] == "shape_slat_lr"
    assert metadata["blocker_next_target"] == "shape-slat-cascade"
    assert result.trace.metadata["shape_lr_projection"]["selected_projected_shape"] == (4096, 6)
    assert result.trace.metadata["shape_slat_lr"]["sampled_feature_shape"] == (4096, 32)


def test_pixal3d_stage_plan_uses_upstream_hr_token_guard():
    coords = mx.array([[0, index, index, index] for index in range(64)], dtype=mx.int32)

    plan = pixal3d_stage_plan("1536_cascade", max_num_tokens=4, hr_coordinates=coords)

    assert plan.actual_hr_resolution == 1024
    assert plan.actual_hr_grid_resolution == 64
    assert plan.hr_token_count is not None
