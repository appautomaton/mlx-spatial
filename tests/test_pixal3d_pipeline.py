import mlx.core as mx

from mlx_spatial.pixal3d_camera import pixal3d_stage_plan
from mlx_spatial.pixal3d_inference import Pixal3DInferencePipeline
from mlx_spatial.pixal3d_projection import PIXAL3D_DEFAULT_NUM_REGISTER_TOKENS
from pixal3d_fixtures import write_fake_pixal3d_root


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


def test_pixal3d_stage_plan_uses_upstream_hr_token_guard():
    coords = mx.array([[0, index, index, index] for index in range(64)], dtype=mx.int32)

    plan = pixal3d_stage_plan("1536_cascade", max_num_tokens=4, hr_coordinates=coords)

    assert plan.actual_hr_resolution == 1024
    assert plan.actual_hr_grid_resolution == 64
    assert plan.hr_token_count is not None
