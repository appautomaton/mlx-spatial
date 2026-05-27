import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from PIL import Image

import mlx_spatial.pixal3d_inference as pixal3d_inference
from mlx_spatial.ovoxel import FlexibleDualGridMesh
from mlx_spatial.pixal3d_camera import pixal3d_stage_plan
from mlx_spatial.pixal3d_inference import (
    PIXAL3D_DEFAULT_SHAPE_DECODER_TOKEN_LIMIT,
    PIXAL3D_DEFAULT_SHAPE_UPSAMPLE_TOKEN_LIMIT,
    PIXAL3D_DEFAULT_TEXTURE_DECODER_TOKEN_LIMIT,
    Pixal3DInferencePipeline,
)
from mlx_spatial.pixal3d_projection import PIXAL3D_DEFAULT_NUM_REGISTER_TOKENS, Pixal3DProjectionStageConfig
from mlx_spatial.sam3d_assets import Sam3dAssetBlocker
from mlx_spatial.sam3d_moge import Sam3dMogePointmap, Sam3dMogeResult
from mlx_spatial.trellis2_export import Trellis2TextureBakeResult
from pixal3d_fixtures import (
    write_fake_pixal3d_dinov3_root,
    write_fake_pixal3d_decode_root,
    write_fake_naf_root,
    write_fake_pixal3d_root,
    write_fake_pixal3d_shape_hr_root,
    write_fake_pixal3d_shape_slat_root,
    write_fake_pixal3d_sparse_decoder_root,
    write_fake_pixal3d_sparse_flow_root,
    write_fake_pixal3d_texture_slat_root,
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


def test_pixal3d_pipeline_manual_fov_does_not_invoke_moge(tmp_path, monkeypatch):
    root = write_fake_pixal3d_root(tmp_path / "weights")
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")
    token_count = 1 + PIXAL3D_DEFAULT_NUM_REGISTER_TOKENS + 32 * 32
    hidden_states = mx.zeros((1, token_count, 3), dtype=mx.float32)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("manual FOV path must not invoke MoGe")

    monkeypatch.setattr(pixal3d_inference, "run_sam3d_moge_pointmap", fail_if_called)

    result = Pixal3DInferencePipeline(root).generate(
        image,
        manual_fov=0.2,
        projection_hidden_states=hidden_states,
    )

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "sparse-structure-flow"
    assert result.trace.metadata["camera_source"] == "manual_fov"


def test_pixal3d_pipeline_uses_moge_auto_camera_when_manual_fov_omitted(tmp_path, monkeypatch):
    root = write_fake_pixal3d_root(tmp_path / "weights")
    image = tmp_path / "image.png"
    Image.new("RGB", (16, 8), (128, 64, 32)).save(image)
    token_count = 1 + PIXAL3D_DEFAULT_NUM_REGISTER_TOKENS + 32 * 32
    hidden_states = mx.zeros((1, token_count, 3), dtype=mx.float32)
    calls = {}

    def fake_moge(image_rgb, *, root, memory_profile):
        calls["shape"] = tuple(int(dim) for dim in image_rgb.shape)
        calls["root"] = str(root)
        calls["memory_profile"] = memory_profile
        return Sam3dMogeResult(
            pointmap=Sam3dMogePointmap(
                pointmap=np.zeros((8, 16, 3), dtype=np.float32),
                intrinsics=np.array([[2.5, 0.0, 0.5], [0.0, 2.5, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32),
                mask=np.ones((8, 16), dtype=bool),
                depth=np.ones((8, 16), dtype=np.float32),
                metadata={"fixture": True},
            )
        )

    monkeypatch.setattr(pixal3d_inference, "run_sam3d_moge_pointmap", fake_moge)

    result = Pixal3DInferencePipeline(root).generate(
        image,
        output_dir=tmp_path / "out",
        projection_hidden_states=hidden_states,
        moge_root=tmp_path / "moge",
        moge_memory_profile="safe",
    )

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "sparse-structure-flow"
    assert result.trace.completed_stages == (
        "input-image",
        "asset-validation",
        "pipeline-config",
        "camera-setup",
        "projection-conditioning:ss",
        "artifact:sparse_projection",
    )
    assert calls == {"shape": (8, 16, 3), "root": str(tmp_path / "moge"), "memory_profile": "safe"}
    assert result.trace.metadata["camera_source"] == "moge"
    assert result.trace.metadata["camera"].camera_angle_x > 0
    assert result.trace.metadata["moge_camera"]["root"] == str(tmp_path / "moge")
    assert result.trace.metadata["moge_camera"]["memory_profile"] == "safe"
    assert result.trace.metadata["moge_camera"]["exact_upstream_v2_parity"] is False


def test_pixal3d_pipeline_reports_moge_camera_blocker_when_manual_fov_omitted(tmp_path, monkeypatch):
    root = write_fake_pixal3d_root(tmp_path / "weights")
    image = tmp_path / "image.png"
    Image.new("RGB", (16, 8), (128, 64, 32)).save(image)

    def fake_blocked_moge(image_rgb, *, root, memory_profile):
        return Sam3dMogeResult(
            blocker=Sam3dAssetBlocker(
                stage="moge-pointmap",
                operation="validate converted MoGe safetensors checkpoint",
                reason="fixture missing MoGe checkpoint",
                metadata={"expected": str(Path(root) / "model.safetensors")},
            )
        )

    monkeypatch.setattr(pixal3d_inference, "run_sam3d_moge_pointmap", fake_blocked_moge)

    result = Pixal3DInferencePipeline(root).generate(
        image,
        moge_root=tmp_path / "missing-moge",
        moge_memory_profile="balanced",
    )

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "camera-setup"
    assert result.trace.blocker.operation == "validate converted MoGe safetensors checkpoint"
    assert result.trace.blocker.metadata["moge_root"] == str(tmp_path / "missing-moge")
    assert result.trace.blocker.metadata["memory_profile"] == "balanced"
    assert result.trace.blocker.metadata["expected"] == str(tmp_path / "missing-moge" / "model.safetensors")
    assert result.trace.completed_stages == ("input-image", "asset-validation", "pipeline-config")


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


def test_pixal3d_pipeline_validates_shape_upsample_token_limit(tmp_path):
    root = write_fake_pixal3d_root(tmp_path / "weights")
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")

    result = Pixal3DInferencePipeline(root).generate(
        image,
        manual_fov=0.2,
        shape_upsample_token_limit=0,
    )

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "input-validation"
    assert result.trace.blocker.operation == "validate Pixal3D shape upsample token limit"
    assert result.trace.blocker.metadata["shape_upsample_token_limit"] == 0


def test_pixal3d_pipeline_validates_decoder_token_limits(tmp_path):
    root = write_fake_pixal3d_root(tmp_path / "weights")
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")

    shape_result = Pixal3DInferencePipeline(root).generate(
        image,
        manual_fov=0.2,
        shape_decoder_token_limit=0,
    )
    texture_result = Pixal3DInferencePipeline(root).generate(
        image,
        manual_fov=0.2,
        texture_decoder_token_limit=0,
    )

    assert shape_result.trace.blocker is not None
    assert shape_result.trace.blocker.stage == "input-validation"
    assert shape_result.trace.blocker.operation == "validate Pixal3D shape decoder token limit"
    assert shape_result.trace.blocker.metadata["shape_decoder_token_limit"] == 0
    assert texture_result.trace.blocker is not None
    assert texture_result.trace.blocker.stage == "input-validation"
    assert texture_result.trace.blocker.operation == "validate Pixal3D texture decoder token limit"
    assert texture_result.trace.blocker.metadata["texture_decoder_token_limit"] == 0


def test_pixal3d_pipeline_validates_glb_export_backend(tmp_path):
    root = write_fake_pixal3d_root(tmp_path / "weights")
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")

    result = Pixal3DInferencePipeline(root).generate(
        image,
        manual_fov=0.2,
        glb_export_backend="bad",
    )

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "input-validation"
    assert result.trace.blocker.operation == "validate Pixal3D export options"
    assert result.trace.blocker.metadata["glb_export_backend"] == "bad"
    assert result.trace.blocker.metadata["supported_glb_export_backends"] == ("internal", "spatialkit")


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
        naf_root=tmp_path / "missing-naf",
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
        naf_root=tmp_path / "missing-naf",
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
    assert result.trace.blocker.stage == "naf-assets"
    assert result.trace.blocker.operation == "load converted NAF safetensors"
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
        naf_root=tmp_path / "missing-naf",
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


def test_pixal3d_pipeline_uses_mlx_naf_when_shape_lr_map_is_not_supplied(tmp_path, monkeypatch):
    root = write_fake_pixal3d_shape_slat_root(tmp_path / "weights", proj_in_channels=4, sparse_steps=1, shape_steps=1)
    naf_root = write_fake_naf_root(tmp_path / "naf")
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), (128, 64, 32)).save(image)
    patch_grid = 2
    token_count = 1 + PIXAL3D_DEFAULT_NUM_REGISTER_TOKENS + patch_grid * patch_grid
    hidden_states = mx.zeros((1, token_count, 4), dtype=mx.float32)
    original_stage_config = pixal3d_inference.pixal3d_projection_stage_config

    def fake_stage_config(name):
        stage = original_stage_config(name)
        if name == "shape_512":
            return Pixal3DProjectionStageConfig(
                name=stage.name,
                image_size=8,
                grid_resolution=stage.grid_resolution,
                use_naf_upsample=stage.use_naf_upsample,
                naf_target_size=8,
            )
        return stage

    monkeypatch.setattr(pixal3d_inference, "pixal3d_projection_stage_config", fake_stage_config)

    result = Pixal3DInferencePipeline(root).generate(
        image,
        output_dir=tmp_path / "out",
        manual_fov=0.2,
        projection_hidden_states=hidden_states,
        naf_root=naf_root,
        naf_coordinate_chunk_size=1024,
    )

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "shape-slat-cascade"
    assert "projection-conditioning:shape_512" in result.trace.completed_stages
    assert "artifact:shape_slat_lr" in result.trace.completed_stages
    assert result.trace.metadata["shape_lr_projection"]["source"] == "mlx-naf"
    assert result.trace.metadata["shape_lr_projection"]["selected_projected_shape"] == (4096, 8)
    assert result.trace.metadata["shape_lr_projection"]["full_map_avoidance"] is True


def test_pixal3d_pipeline_writes_hr_coordinates_with_fake_shape_decoder(tmp_path):
    root = write_fake_pixal3d_shape_hr_root(tmp_path / "weights", proj_in_channels=3, sparse_steps=1, shape_steps=1)
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
        naf_root=tmp_path / "missing-naf",
    )

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "naf-assets"
    assert result.trace.blocker.operation == "load converted NAF safetensors"
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
        "shape-slat-cascade:upsample",
        "artifact:shape_slat_hr_coordinates",
    )
    assert [path.name for path in result.artifacts] == [
        "sparse_projection.npz",
        "sparse_structure.npz",
        "shape_slat_lr.npz",
        "shape_slat_hr_coordinates.npz",
    ]
    payload = np.load(result.artifacts[3])
    assert payload["coordinates"].shape == (4096, 4)
    assert int(payload["actual_hr_resolution"]) == 1024
    assert int(payload["actual_hr_grid_resolution"]) == 64
    metadata = json.loads(payload["metadata_json"].item())
    assert metadata["stage"] == "shape_slat_hr_coordinates"
    assert metadata["raw_upsampled_shape"] == [4096, 4]
    assert metadata["blocker_next_target"] == "shape-hr-projection-conditioning"
    assert result.trace.metadata["shape_hr_cascade"]["completed_upsamples"] == 4
    assert result.trace.metadata["shape_hr_cascade"]["token_count"] == 4096
    assert result.trace.metadata["shape_hr_cascade"]["shape_upsample_token_limit"] == PIXAL3D_DEFAULT_SHAPE_UPSAMPLE_TOKEN_LIMIT


def test_pixal3d_pipeline_writes_shape_slat_hr_artifact_with_fake_hr_naf_and_shape_flow(tmp_path):
    root = write_fake_pixal3d_shape_hr_root(tmp_path / "weights", proj_in_channels=3, sparse_steps=1, shape_steps=1)
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
        shape_hr_naf_feature_map=naf,
        naf_root=tmp_path / "missing-naf",
    )

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "naf-assets"
    assert result.trace.blocker.operation == "load converted NAF safetensors"
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
        "shape-slat-cascade:upsample",
        "artifact:shape_slat_hr_coordinates",
        "projection-conditioning:shape_1024",
        "shape-slat-sampling:1024",
        "artifact:shape_slat_hr",
    )
    assert [path.name for path in result.artifacts] == [
        "sparse_projection.npz",
        "sparse_structure.npz",
        "shape_slat_lr.npz",
        "shape_slat_hr_coordinates.npz",
        "shape_slat_hr.npz",
    ]
    payload = np.load(result.artifacts[4])
    assert payload["coordinates"].shape == (4096, 4)
    assert payload["features"].shape == (4096, 32)
    metadata = json.loads(payload["metadata_json"].item())
    assert metadata["stage"] == "shape_slat_hr"
    assert metadata["actual_hr_resolution"] == 1024
    assert metadata["actual_hr_grid_resolution"] == 64
    assert metadata["blocker_next_target"] == "texture-projection-conditioning"
    assert result.trace.metadata["shape_hr_projection"]["selected_projected_shape"] == (4096, 6)
    assert result.trace.metadata["shape_slat_hr"]["sampled_feature_shape"] == (4096, 32)


def test_pixal3d_pipeline_writes_texture_slat_and_shape_decoder_then_blocks_without_texture_decoder(tmp_path):
    root = write_fake_pixal3d_texture_slat_root(
        tmp_path / "weights",
        proj_in_channels=3,
        sparse_steps=1,
        shape_steps=1,
        texture_steps=1,
    )
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
        shape_hr_naf_feature_map=naf,
        texture_naf_feature_map=naf,
    )

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "texture-decoder"
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
        "shape-slat-cascade:upsample",
        "artifact:shape_slat_hr_coordinates",
        "projection-conditioning:shape_1024",
        "shape-slat-sampling:1024",
        "artifact:shape_slat_hr",
        "projection-conditioning:tex_1024",
        "texture-slat-sampling:1024",
        "artifact:texture_slat",
        "shape-decoder",
        "artifact:shape_decoder_fields",
    )
    assert [path.name for path in result.artifacts] == [
        "sparse_projection.npz",
        "sparse_structure.npz",
        "shape_slat_lr.npz",
        "shape_slat_hr_coordinates.npz",
        "shape_slat_hr.npz",
        "texture_slat.npz",
        "shape_decoder_fields.npz",
    ]
    payload = np.load(result.artifacts[5])
    assert payload["coordinates"].shape == (4096, 4)
    assert payload["features"].shape == (4096, 32)
    metadata = json.loads(payload["metadata_json"].item())
    assert metadata["stage"] == "texture_slat"
    assert metadata["actual_hr_resolution"] == 1024
    assert metadata["actual_hr_grid_resolution"] == 64
    assert metadata["blocker_next_target"] == "latent-decoding"
    shape_decoder_payload = np.load(result.artifacts[6])
    assert shape_decoder_payload["coordinates"].shape == (4096, 4)
    assert shape_decoder_payload["fields"].shape == (4096, 7)
    shape_decoder_metadata = json.loads(shape_decoder_payload["metadata_json"].item())
    assert shape_decoder_metadata["stage"] == "shape_decoder_fields"
    assert shape_decoder_metadata["blocker_next_target"] == "texture-decoder"
    assert result.trace.metadata["texture_projection"]["selected_projected_shape"] == (4096, 6)
    assert result.trace.metadata["texture_slat"]["normalized_shape_feature_shape"] == (4096, 32)
    assert result.trace.metadata["texture_slat"]["sampled_feature_shape"] == (4096, 32)
    assert result.trace.metadata["shape_decoder"]["decoder_output_shape"] == (4096, 7)
    assert result.trace.metadata["shape_decoder"]["decoder_token_limit"] == PIXAL3D_DEFAULT_SHAPE_DECODER_TOKEN_LIMIT


def test_pixal3d_pipeline_writes_texture_decoder_pbr_artifact_with_fake_decode_assets(tmp_path):
    root = write_fake_pixal3d_decode_root(
        tmp_path / "weights",
        proj_in_channels=3,
        sparse_steps=1,
        shape_steps=1,
        texture_steps=1,
    )
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
        shape_hr_naf_feature_map=naf,
        texture_naf_feature_map=naf,
    )

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "mesh-export"
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
        "shape-slat-cascade:upsample",
        "artifact:shape_slat_hr_coordinates",
        "projection-conditioning:shape_1024",
        "shape-slat-sampling:1024",
        "artifact:shape_slat_hr",
        "projection-conditioning:tex_1024",
        "texture-slat-sampling:1024",
        "artifact:texture_slat",
        "shape-decoder",
        "artifact:shape_decoder_fields",
        "texture-decoder",
        "artifact:texture_decoder_pbr",
    )
    assert [path.name for path in result.artifacts] == [
        "sparse_projection.npz",
        "sparse_structure.npz",
        "shape_slat_lr.npz",
        "shape_slat_hr_coordinates.npz",
        "shape_slat_hr.npz",
        "texture_slat.npz",
        "shape_decoder_fields.npz",
        "texture_decoder_pbr.npz",
    ]
    shape_payload = np.load(result.artifacts[6])
    assert shape_payload["coordinates"].shape == (4096, 4)
    assert shape_payload["fields"].shape == (4096, 7)
    texture_payload = np.load(result.artifacts[7])
    assert texture_payload["coordinates"].shape == (4096, 4)
    assert texture_payload["attributes"].shape == (4096, 6)
    assert texture_payload["spatial_shape"].tolist() == [481, 481, 481]
    assert int(texture_payload["decode_resolution"]) == 1024
    metadata = json.loads(texture_payload["metadata_json"].item())
    assert metadata["stage"] == "texture_decoder_pbr"
    assert metadata["blocker_next_target"] == "mesh-extraction"
    assert result.trace.metadata["texture_decoder"]["decoder_output_shape"] == (4096, 6)
    assert result.trace.metadata["texture_decoder"]["decoder_token_limit"] == PIXAL3D_DEFAULT_TEXTURE_DECODER_TOKEN_LIMIT
    assert result.trace.blocker.metadata["shape_decoder_fields_shape"] == (4096, 7)
    assert result.trace.blocker.metadata["texture_decoder_attributes_shape"] == (4096, 6)


def test_pixal3d_pipeline_writes_textured_glb_with_fake_export_route(tmp_path, monkeypatch):
    root = write_fake_pixal3d_decode_root(tmp_path / "weights", proj_in_channels=3, sparse_steps=1, shape_steps=1, texture_steps=1)
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")
    patch_grid = 32
    token_count = 1 + PIXAL3D_DEFAULT_NUM_REGISTER_TOKENS + patch_grid * patch_grid
    hidden_states = mx.zeros((1, token_count, 3), dtype=mx.float32)
    naf = mx.zeros((1, patch_grid, patch_grid, 3), dtype=mx.float32)
    calls = _patch_pixal3d_export_fixtures(monkeypatch)

    result = Pixal3DInferencePipeline(root).generate(
        image,
        output=tmp_path / "out" / "pixal3d.glb",
        manual_fov=0.2,
        projection_hidden_states=hidden_states,
        shape_lr_naf_feature_map=naf,
        shape_hr_naf_feature_map=naf,
        texture_naf_feature_map=naf,
        texture_size=16,
        glb_target_faces=123,
        xatlas_face_guard=456,
        xatlas_parallel_chunks=1,
        texture_bake_backend="kdtree",
    )

    assert result.ready
    assert result.trace.blocker is None
    assert result.trace.completed_stages[-4:] == (
        "texture-decoder",
        "artifact:texture_decoder_pbr",
        "mesh-export",
        "artifact:textured_glb",
    )
    assert result.trace.output_path == tmp_path / "out" / "pixal3d.glb"
    assert result.artifacts[-1] == tmp_path / "out" / "pixal3d.glb"
    assert result.artifacts[-1].read_bytes() == b"glb"
    assert result.trace.metadata["mesh_export"]["source_mesh_vertices"] == 4
    assert result.trace.metadata["mesh_export"]["source_mesh_faces"] == 2
    assert result.trace.metadata["mesh_export"]["texture_size"] == 16
    assert result.trace.metadata["mesh_export"]["bake_backend"] == "xatlas-kdtree"
    assert result.trace.metadata["textured_glb_artifact"].bytes_written == 3
    assert calls["mesh_grid_size"] == 1024
    assert calls["postprocess_target_faces"] == 123
    assert calls["bake_texture_size"] == 16
    assert calls["bake_xatlas_face_guard"] == 456
    assert calls["bake_xatlas_parallel_chunks"] == 1
    assert calls["bake_texture_bake_backend"] == "kdtree"
    checkpoints = result.trace.metadata["memory_checkpoints"]
    assert tuple(checkpoints) == (
        "after_sparse_structure",
        "after_shape_slat_lr",
        "after_shape_hr_coordinates",
        "after_shape_slat_hr",
        "after_texture_slat",
        "after_shape_decoder",
        "after_texture_decoder",
        "after_glb_export",
    )
    for checkpoint in checkpoints.values():
        assert set(checkpoint) == {"active_bytes", "peak_bytes"}


def test_pixal3d_pipeline_uses_optional_spatialkit_export_backend(tmp_path, monkeypatch):
    root = write_fake_pixal3d_decode_root(tmp_path / "weights", proj_in_channels=3, sparse_steps=1, shape_steps=1, texture_steps=1)
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")
    patch_grid = 32
    token_count = 1 + PIXAL3D_DEFAULT_NUM_REGISTER_TOKENS + patch_grid * patch_grid
    hidden_states = mx.zeros((1, token_count, 3), dtype=mx.float32)
    naf = mx.zeros((1, patch_grid, patch_grid, 3), dtype=mx.float32)
    calls = {}

    def fake_spatialkit_exporter(
        decoded_dir,
        output_path,
        *,
        texture_size,
        target_faces,
        grid_size,
        diagnostics_path,
    ):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"glb")
        diagnostics = Path(diagnostics_path)
        diagnostics.write_text("{}", encoding="utf-8")
        calls["decoded_dir"] = Path(decoded_dir)
        calls["output_path"] = output
        calls["texture_size"] = texture_size
        calls["target_faces"] = target_faces
        calls["grid_size"] = grid_size
        calls["diagnostics_path"] = diagnostics
        return SimpleNamespace(
            glb=SimpleNamespace(
                path=output,
                format="glb",
                bytes_written=3,
                metadata={"stage": "textured_glb", "mesh_name": "Pixal3D_TexturedMesh"},
            ),
            diagnostics_path=diagnostics,
            diagnostics={
                "stages": {
                    "extract_mesh": {"source_vertices": 4, "source_faces": 2},
                    "simplify_mesh": {"stats": {"final_faces": 2}},
                    "uv": {
                        "vertices_shape": (4, 3),
                        "faces_shape": (2, 3),
                        "uvs_shape": (4, 2),
                        "stats": {"backend": "face-atlas"},
                    },
                    "texture_bake": {
                        "stats": {
                            "backend": "metal-face-atlas-nearest",
                            "texture_size": 16,
                            "voxel_count": 4096,
                            "coverage_ratio": 0.5,
                            "raw_coverage_ratio": 0.5,
                            "sampled_texel_count": 128,
                            "missing_texel_count": 0,
                            "out_of_grid_texel_count": 0,
                        }
                    },
                }
            },
        )

    monkeypatch.setattr(pixal3d_inference, "_load_spatialkit_exporter", lambda: (fake_spatialkit_exporter, None))

    result = Pixal3DInferencePipeline(root).generate(
        image,
        output=tmp_path / "out" / "pixal3d.glb",
        manual_fov=0.2,
        projection_hidden_states=hidden_states,
        shape_lr_naf_feature_map=naf,
        shape_hr_naf_feature_map=naf,
        texture_naf_feature_map=naf,
        texture_size=16,
        glb_target_faces=123,
        glb_export_backend="spatialkit",
        glb_diagnostics_path=tmp_path / "out" / "spatialkit-diagnostics.json",
    )

    assert result.ready
    assert result.trace.blocker is None
    assert result.trace.completed_stages[-2:] == ("mesh-export", "artifact:textured_glb")
    assert result.artifacts[-2:] == (tmp_path / "out" / "pixal3d.glb", tmp_path / "out" / "spatialkit-diagnostics.json")
    assert calls["decoded_dir"] == tmp_path / "out"
    assert calls["output_path"] == tmp_path / "out" / "pixal3d.glb"
    assert calls["diagnostics_path"] == tmp_path / "out" / "spatialkit-diagnostics.json"
    assert calls["texture_size"] == 16
    assert calls["target_faces"] == 123
    assert calls["grid_size"] == 1024
    assert result.trace.metadata["export_options"]["glb_export_backend_requested"] == "spatialkit"
    assert result.trace.metadata["export_options"]["glb_export_backend_used"] == "spatialkit"
    assert result.trace.metadata["mesh_export"]["export_backend"] == "spatialkit"
    assert result.trace.metadata["mesh_export"]["spatialkit_diagnostics_path"] == str(calls["diagnostics_path"])
    assert result.trace.metadata["mesh_export"]["source_mesh_faces"] == 2
    assert result.trace.metadata["mesh_export"]["bake_backend"] == "metal-face-atlas-nearest"
    assert result.trace.metadata["mesh_export"]["unwrap_chunks"] == 0
    assert result.trace.metadata["mesh_export"]["unwrap_chart_count"] == 0
    assert result.trace.metadata["mesh_export"]["unwrap_utilization"] == 0.0
    assert result.trace.metadata["mesh_export"]["xatlas_face_guard"] is None
    assert result.trace.metadata["mesh_export"]["xatlas_face_guard_mode"] == "not_used"
    assert result.trace.metadata["mesh_export"]["source_projection_used"] is False
    assert result.trace.metadata["artifact_paths"][-1] == tmp_path / "out" / "spatialkit-diagnostics.json"


def test_pixal3d_pipeline_falls_back_when_optional_spatialkit_is_missing(tmp_path, monkeypatch):
    root = write_fake_pixal3d_decode_root(tmp_path / "weights", proj_in_channels=3, sparse_steps=1, shape_steps=1, texture_steps=1)
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")
    patch_grid = 32
    token_count = 1 + PIXAL3D_DEFAULT_NUM_REGISTER_TOKENS + patch_grid * patch_grid
    hidden_states = mx.zeros((1, token_count, 3), dtype=mx.float32)
    naf = mx.zeros((1, patch_grid, patch_grid, 3), dtype=mx.float32)
    calls = _patch_pixal3d_export_fixtures(monkeypatch)
    monkeypatch.setattr(
        pixal3d_inference,
        "_load_spatialkit_exporter",
        lambda: (None, "mlx_spatialkit is not importable; falling back to internal Pixal3D GLB export"),
    )

    result = Pixal3DInferencePipeline(root).generate(
        image,
        output=tmp_path / "out" / "pixal3d.glb",
        manual_fov=0.2,
        projection_hidden_states=hidden_states,
        shape_lr_naf_feature_map=naf,
        shape_hr_naf_feature_map=naf,
        texture_naf_feature_map=naf,
        glb_export_backend="spatialkit",
    )

    assert result.ready
    assert result.trace.metadata["mesh_export"]["export_backend"] == "internal"
    assert result.trace.metadata["export_options"]["glb_export_backend_requested"] == "spatialkit"
    assert result.trace.metadata["export_options"]["glb_export_backend_used"] == "internal"
    assert "falling back to internal" in result.trace.metadata["export_options"]["glb_export_backend_fallback_reason"]
    assert calls["mesh_grid_size"] == 1024
    assert result.artifacts[-1].read_bytes() == b"glb"


def test_pixal3d_pipeline_glb_writer_failure_preserves_decoded_artifacts(tmp_path, monkeypatch):
    root = write_fake_pixal3d_decode_root(tmp_path / "weights", proj_in_channels=3, sparse_steps=1, shape_steps=1, texture_steps=1)
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")
    patch_grid = 32
    token_count = 1 + PIXAL3D_DEFAULT_NUM_REGISTER_TOKENS + patch_grid * patch_grid
    hidden_states = mx.zeros((1, token_count, 3), dtype=mx.float32)
    naf = mx.zeros((1, patch_grid, patch_grid, 3), dtype=mx.float32)
    _patch_pixal3d_export_fixtures(monkeypatch, writer_failure=OSError("fixture writer failure"))

    result = Pixal3DInferencePipeline(root).generate(
        image,
        output=tmp_path / "out" / "pixal3d.glb",
        manual_fov=0.2,
        projection_hidden_states=hidden_states,
        shape_lr_naf_feature_map=naf,
        shape_hr_naf_feature_map=naf,
        texture_naf_feature_map=naf,
    )

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "glb-export"
    assert "fixture writer failure" in result.trace.blocker.reason
    assert [path.name for path in result.artifacts][-2:] == ["shape_decoder_fields.npz", "texture_decoder_pbr.npz"]
    assert result.trace.completed_stages[-3:] == ("texture-decoder", "artifact:texture_decoder_pbr", "mesh-export")


def test_pixal3d_stage_plan_uses_upstream_hr_token_guard():
    coords = mx.array([[0, index, index, index] for index in range(64)], dtype=mx.int32)

    plan = pixal3d_stage_plan("1536_cascade", max_num_tokens=4, hr_coordinates=coords)

    assert plan.actual_hr_resolution == 1024
    assert plan.actual_hr_grid_resolution == 64
    assert plan.hr_token_count is not None


def _patch_pixal3d_export_fixtures(monkeypatch, *, writer_failure: Exception | None = None):
    calls = {}
    mesh = FlexibleDualGridMesh(
        vertices=np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0]],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64),
    )

    def fake_mesh_from_fields(coordinates, fields, *, grid_size):
        calls["mesh_coordinates_shape"] = tuple(int(dim) for dim in coordinates.shape)
        calls["mesh_fields_shape"] = tuple(int(dim) for dim in fields.shape)
        calls["mesh_grid_size"] = grid_size
        return mesh

    def fake_postprocess(source_mesh, *, target_faces):
        calls["postprocess_mesh"] = source_mesh
        calls["postprocess_target_faces"] = target_faces
        return SimpleNamespace(
            mesh=source_mesh,
            source_mesh=source_mesh,
            stats=SimpleNamespace(
                original_vertices=4,
                original_faces=2,
                final_vertices=4,
                final_faces=2,
            ),
        )

    def fake_bake(
        bake_mesh,
        texture_coordinates,
        texture_attributes,
        *,
        decode_resolution,
        texture_size,
        xatlas_face_guard,
        xatlas_parallel_chunks,
        texture_bake_backend,
        projection_source_mesh,
    ):
        calls["bake_mesh"] = bake_mesh
        calls["bake_texture_coordinates_shape"] = tuple(int(dim) for dim in texture_coordinates.shape)
        calls["bake_texture_attributes_shape"] = tuple(int(dim) for dim in texture_attributes.shape)
        calls["bake_decode_resolution"] = decode_resolution
        calls["bake_texture_size"] = texture_size
        calls["bake_xatlas_face_guard"] = xatlas_face_guard
        calls["bake_xatlas_parallel_chunks"] = xatlas_parallel_chunks
        calls["bake_texture_bake_backend"] = texture_bake_backend
        calls["bake_projection_source_mesh"] = projection_source_mesh
        return Trellis2TextureBakeResult(
            vertices=mesh.vertices,
            faces=mesh.faces,
            uvs=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
            base_color_rgba=np.zeros((texture_size, texture_size, 4), dtype=np.uint8),
            metallic_roughness=np.zeros((texture_size, texture_size, 3), dtype=np.uint8),
            coverage_mask=np.ones((texture_size, texture_size), dtype=bool),
            texture_size=texture_size,
            voxel_count=int(texture_coordinates.shape[0]),
            k_neighbors=4,
            origin=(-0.5, -0.5, -0.5),
            voxel_size=1.0 / float(decode_resolution),
            backend=f"xatlas-{texture_bake_backend}",
            raw_coverage_ratio=0.75,
            unwrap_backend="xatlas-global",
            unwrap_seconds=0.01,
            unwrap_chunks=xatlas_parallel_chunks,
            unwrap_chart_count=2,
            unwrap_utilization=0.5,
            xatlas_face_guard=int(xatlas_face_guard) if isinstance(xatlas_face_guard, int) else 999,
            xatlas_face_guard_mode="manual",
            sampled_texel_count=texture_size * texture_size,
            missing_texel_count=0,
            out_of_grid_texel_count=0,
            source_projection_used=False,
            source_projection_detail="source mesh matches export mesh; projection not needed",
        )

    def fake_write_glb(baked_texture, output_path, *, metadata=None):
        if writer_failure is not None:
            raise writer_failure
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"glb")
        return SimpleNamespace(
            path=path,
            format="glb",
            bytes_written=3,
            metadata={"stage": "textured_glb", **(metadata or {})},
        )

    monkeypatch.setattr(pixal3d_inference, "flexi_dual_grid_fields_to_mesh", fake_mesh_from_fields)
    monkeypatch.setattr(pixal3d_inference, "postprocess_trellis2_mesh_for_glb", fake_postprocess)
    monkeypatch.setattr(pixal3d_inference, "bake_trellis2_texture_fields_mac_native", fake_bake)
    monkeypatch.setattr(pixal3d_inference, "write_pixal3d_textured_glb", fake_write_glb)
    return calls
