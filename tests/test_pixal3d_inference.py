import mlx.core as mx
import pytest

from mlx_spatial.pixal3d_inference import (
    PIXAL3D_RECOMMENDED_PIPELINE_TYPE,
    Pixal3DInferencePipeline,
)
from mlx_spatial.pixal3d_projection import PIXAL3D_DEFAULT_NUM_REGISTER_TOKENS
from pixal3d_fixtures import write_fake_pixal3d_root


def test_pixal3d_pipeline_reports_missing_input(tmp_path):
    result = Pixal3DInferencePipeline(tmp_path / "weights").generate(tmp_path / "missing.png")

    assert not result.ready
    assert result.trace.completed_stages == ()
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "input-validation"


def test_pixal3d_pipeline_reports_missing_assets_after_input_validation(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")

    result = Pixal3DInferencePipeline(tmp_path / "weights").generate(image)

    assert not result.ready
    assert result.trace.completed_stages == ("input-image",)
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "asset-validation"


@pytest.mark.xfail(
    reason="Pixal3D stage-4 (auto-camera / Telea inpaint) in progress; pipeline stage boundary not final",
    strict=False,
)
def test_pixal3d_pipeline_requires_manual_fov_until_auto_camera_slice(tmp_path):
    root = write_fake_pixal3d_root(tmp_path / "weights")
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")

    result = Pixal3DInferencePipeline(root).generate(image)

    assert not result.ready
    assert result.trace.completed_stages == ("input-image", "asset-validation", "pipeline-config")
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "camera-setup"
    assert result.trace.metadata["default_pipeline_type"] == "1536_cascade"


@pytest.mark.xfail(
    reason="Pixal3D stage-4 (auto-camera / Telea inpaint) in progress; pipeline stage boundary not final",
    strict=False,
)
def test_pixal3d_pipeline_reports_image_conditioning_boundary_with_manual_fov(tmp_path):
    root = write_fake_pixal3d_root(tmp_path / "weights")
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")

    result = Pixal3DInferencePipeline(root).generate(
        image,
        output_dir=tmp_path / "out",
        manual_fov=0.2,
    )

    assert not result.ready
    assert result.trace.pipeline_type == PIXAL3D_RECOMMENDED_PIPELINE_TYPE
    assert result.trace.manual_fov == 0.2
    assert result.trace.output_path == tmp_path / "out" / "model.glb"
    assert result.trace.completed_stages == (
        "input-image",
        "asset-validation",
        "pipeline-config",
        "camera-setup",
    )
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "image-conditioning"
    assert "camera" in result.trace.metadata
    assert result.trace.metadata["stage_plan"].actual_hr_resolution == 1024


@pytest.mark.xfail(
    reason="Pixal3D stage-4 (auto-camera / Telea inpaint) in progress; pipeline stage boundary not final",
    strict=False,
)
def test_pixal3d_pipeline_runs_sparse_projection_boundary_with_hidden_states(tmp_path):
    root = write_fake_pixal3d_root(tmp_path / "weights")
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
    )
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "sparse-structure-flow"
    assert len(result.artifacts) == 1
    assert result.artifacts[0].is_file()
    assert result.trace.metadata["ss_projection"]["ready"] is True
    assert result.trace.metadata["ss_projection"]["global_shape"] == (1, 5, 3)
    assert result.trace.metadata["ss_projection"]["projected_shape"] == (1, 16**3, 3)


def test_pixal3d_pipeline_rejects_unknown_pipeline_type(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"placeholder")

    result = Pixal3DInferencePipeline(tmp_path / "weights").generate(image, pipeline_type="512")

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "input-validation"
    assert "unsupported pipeline_type" in result.trace.blocker.reason
