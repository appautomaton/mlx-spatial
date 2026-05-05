import mlx.core as mx
import numpy as np

from mlx_spatial.hyworld2_heads import (
    CameraHeadConfig,
    DPTHeadConfig,
    apply_hyworld2_activation,
    default_camera_head_tensors,
    default_dpt_head_tensors,
    run_camera_head,
    run_dpt_head,
)
from mlx_spatial.hyworld2_worldmirror import HyWorld2BackboneOutput


def _backbone(batch=1, frames=2, patches=4, dim=8, levels=4):
    frame_token_count = patches + 1
    tokens = mx.reshape(
        mx.arange(batch * frames * frame_token_count * dim, dtype=mx.float32) / 100.0,
        (batch, frames * frame_token_count, dim),
    )
    intermediate_tokens = tuple(
        mx.reshape(
            mx.arange(batch * frames * patches * dim, dtype=mx.float32) / (10.0 + level),
            (batch, frames, patches, dim),
        )
        + float(level)
        for level in range(levels)
    )
    return HyWorld2BackboneOutput(
        tokens=tokens,
        intermediate_tokens=intermediate_tokens,
        patch_start_idx=1,
        patch_grid=(2, 2),
        frame_token_count=frame_token_count,
    )


def test_camera_fixture_returns_official_shape_and_relu_focal_activation():
    config = CameraHeadConfig(dim_in=8, steps=1)
    tensors = default_camera_head_tensors(config)
    tensors["camera.output.bias"] = mx.array(
        [1.0, 2.0, 3.0, -1.0, -2.0, -3.0, -4.0, -5.0, 6.0],
        dtype=mx.float32,
    )

    result = run_camera_head(_backbone(dim=8), config, tensors)

    assert result.ready
    assert result.camera_params is not None
    assert tuple(result.camera_params.shape) == (1, 2, 9)
    np.testing.assert_allclose(
        np.array(result.camera_params)[0, 0],
        np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0, -4.0, 0.0, 6.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_camera_head_uses_full_intermediate_tokens_without_final_tokens():
    full_tokens = mx.zeros((1, 2, 5, 8), dtype=mx.float32)
    full_tokens = full_tokens + mx.array(
        np.array(
            [
                [
                    [[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]] + [[0.0] * 8] * 4,
                    [[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]] + [[0.0] * 8] * 4,
                ]
            ],
            dtype=np.float32,
        )
    )
    backbone = HyWorld2BackboneOutput(
        tokens=None,
        intermediate_full_tokens=(full_tokens,),
        patch_start_idx=1,
        patch_grid=(2, 2),
        frame_token_count=5,
    )
    config = CameraHeadConfig(dim_in=8, steps=1, focal_activation="linear")
    tensors = default_camera_head_tensors(config)
    output_weight = np.zeros((9, 8), dtype=np.float32)
    output_weight[0, 0] = 1.0
    tensors["camera.output.weight"] = mx.array(output_weight)

    result = run_camera_head(backbone, config, tensors)

    assert result.ready
    assert result.camera_params is not None
    assert tuple(result.camera_params.shape) == (1, 2, 9)
    np.testing.assert_allclose(
        np.array(result.camera_params)[0, :, 0],
        np.array([-1.0, 1.0], dtype=np.float32),
        rtol=1e-3,
        atol=1e-3,
    )


def test_camera_head_prefers_full_intermediate_tokens_over_final_tokens():
    final_tokens = mx.zeros((1, 10, 4), dtype=mx.float32)
    full_tokens = mx.ones((1, 2, 5, 8), dtype=mx.float32)
    backbone = HyWorld2BackboneOutput(
        tokens=final_tokens,
        intermediate_full_tokens=(full_tokens,),
        patch_start_idx=1,
        patch_grid=(2, 2),
        frame_token_count=5,
    )
    config = CameraHeadConfig(dim_in=8, steps=1, focal_activation="linear")
    tensors = default_camera_head_tensors(config)
    tensors["camera.output.bias"] = mx.array([1.0] + [0.0] * 8, dtype=mx.float32)

    result = run_camera_head(backbone, config, tensors)

    assert result.ready
    assert result.camera_params is not None
    np.testing.assert_allclose(np.array(result.camera_params)[0, :, 0], np.ones((2,)))


def test_official_camera_head_runs_iterative_adaptive_refinement_fixture():
    dim = 8
    config = CameraHeadConfig(dim_in=dim, steps=2, refine_depth=1, num_heads=2, focal_activation="linear")
    full_tokens = mx.ones((1, 2, 5, dim), dtype=mx.float32)
    backbone = HyWorld2BackboneOutput(
        tokens=None,
        intermediate_full_tokens=(full_tokens,),
        patch_start_idx=1,
        patch_grid=(2, 2),
        frame_token_count=5,
    )
    tensors = {
        "token_norm.weight": mx.ones((dim,), dtype=mx.float32),
        "token_norm.bias": mx.zeros((dim,), dtype=mx.float32),
        "out_norm.weight": mx.ones((dim,), dtype=mx.float32),
        "out_norm.bias": mx.zeros((dim,), dtype=mx.float32),
        "init_token": mx.zeros((1, 1, 9), dtype=mx.float32),
        "param_embed.weight": mx.zeros((dim, 9), dtype=mx.float32),
        "param_embed.bias": mx.zeros((dim,), dtype=mx.float32),
        "adapt_norm_gen.1.weight": mx.zeros((3 * dim, dim), dtype=mx.float32),
        "adapt_norm_gen.1.bias": mx.zeros((3 * dim,), dtype=mx.float32),
        "param_predictor.fc1.weight": mx.zeros((dim // 2, dim), dtype=mx.float32),
        "param_predictor.fc1.bias": mx.zeros((dim // 2,), dtype=mx.float32),
        "param_predictor.fc2.weight": mx.zeros((9, dim // 2), dtype=mx.float32),
        "param_predictor.fc2.bias": mx.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.25], dtype=mx.float32),
    }
    prefix = "refine_net.0"
    tensors.update(
        {
            f"{prefix}.norm1.weight": mx.ones((dim,), dtype=mx.float32),
            f"{prefix}.norm1.bias": mx.zeros((dim,), dtype=mx.float32),
            f"{prefix}.norm2.weight": mx.ones((dim,), dtype=mx.float32),
            f"{prefix}.norm2.bias": mx.zeros((dim,), dtype=mx.float32),
            f"{prefix}.attn.qkv.weight": mx.zeros((3 * dim, dim), dtype=mx.float32),
            f"{prefix}.attn.qkv.bias": mx.zeros((3 * dim,), dtype=mx.float32),
            f"{prefix}.attn.proj.weight": mx.zeros((dim, dim), dtype=mx.float32),
            f"{prefix}.attn.proj.bias": mx.zeros((dim,), dtype=mx.float32),
            f"{prefix}.mlp.fc1.weight": mx.zeros((4 * dim, dim), dtype=mx.float32),
            f"{prefix}.mlp.fc1.bias": mx.zeros((4 * dim,), dtype=mx.float32),
            f"{prefix}.mlp.fc2.weight": mx.zeros((dim, 4 * dim), dtype=mx.float32),
            f"{prefix}.mlp.fc2.bias": mx.zeros((dim,), dtype=mx.float32),
        }
    )

    result = run_camera_head(backbone, config, tensors)

    assert result.ready
    assert result.camera_params is not None
    np.testing.assert_allclose(
        np.array(result.camera_params)[0, 0],
        np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.5], dtype=np.float32),
        atol=1e-6,
    )


def test_camera_head_blocks_for_bad_full_intermediate_shape():
    backbone = HyWorld2BackboneOutput(
        tokens=mx.zeros((1, 10, 4), dtype=mx.float32),
        intermediate_full_tokens=(mx.zeros((1, 2, 5), dtype=mx.float32),),
        patch_start_idx=1,
        patch_grid=(2, 2),
        frame_token_count=5,
    )

    result = run_camera_head(backbone, CameraHeadConfig(dim_in=4))

    assert result.blocker is not None
    assert result.blocker.operation == "HY-World camera full intermediate token shape validation"
    assert result.blocker.metadata["shape"] == (1, 2, 5)


def test_activation_helpers_match_official_references():
    values = mx.array([[-1.0, 0.0, 2.0]], dtype=mx.float32)

    exp_out, blocker = apply_hyworld2_activation(values, "exp")
    assert blocker is None
    np.testing.assert_allclose(np.array(exp_out), np.exp(np.array(values)), rtol=1e-6)

    expp1_out, blocker = apply_hyworld2_activation(values, "expp1")
    assert blocker is None
    np.testing.assert_allclose(np.array(expp1_out), 1.0 + np.exp(np.array(values)), rtol=1e-6)

    inv_log_out, blocker = apply_hyworld2_activation(values, "inv_log")
    assert blocker is None
    np.testing.assert_allclose(
        np.array(inv_log_out),
        np.sign(np.array(values)) * np.expm1(np.abs(np.array(values))),
        rtol=1e-5,
    )

    norm_out, blocker = apply_hyworld2_activation(mx.array([[3.0, 4.0, 0.0]], dtype=mx.float32), "norm")
    assert blocker is None
    np.testing.assert_allclose(np.array(norm_out), np.array([[0.6, 0.8, 0.0]], dtype=np.float32))

    linear_out, blocker = apply_hyworld2_activation(values, "linear")
    assert blocker is None
    np.testing.assert_allclose(np.array(linear_out), np.array(values), rtol=1e-6)


def test_dense_points_depth_and_normals_return_official_channel_last_shapes():
    images = mx.ones((1, 2, 3, 2, 2), dtype=mx.float32)

    points = run_dpt_head(
        _backbone(),
        images,
        DPTHeadConfig(head_type="points", patch_size=1, attr_channels=3, activation="inv_log+expp1"),
    )
    assert points.ready
    assert points.values is not None
    assert points.confidence is not None
    assert tuple(points.values.shape) == (1, 2, 2, 2, 3)
    assert tuple(points.confidence.shape) == (1, 2, 2, 2)

    depth = run_dpt_head(
        _backbone(),
        images,
        DPTHeadConfig(head_type="depth", patch_size=1, attr_channels=1, activation="exp+expp1"),
    )
    assert depth.ready
    assert depth.values is not None
    assert depth.confidence is not None
    assert tuple(depth.values.shape) == (1, 2, 2, 2, 1)
    assert tuple(depth.confidence.shape) == (1, 2, 2, 2)

    normals = run_dpt_head(
        _backbone(),
        images,
        DPTHeadConfig(head_type="normal", patch_size=1, attr_channels=3, activation="norm+expp1"),
    )
    assert normals.ready
    assert normals.values is not None
    assert normals.confidence is not None
    assert tuple(normals.values.shape) == (1, 2, 2, 2, 3)
    assert tuple(normals.confidence.shape) == (1, 2, 2, 2)
    np.testing.assert_allclose(
        np.linalg.norm(np.array(normals.values), axis=-1),
        np.ones((1, 2, 2, 2)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_depth_head_can_return_optional_depth_mask_logits():
    images = mx.ones((1, 2, 3, 2, 2), dtype=mx.float32)
    config = DPTHeadConfig(
        head_type="depth",
        patch_size=1,
        attr_channels=1,
        activation="exp+expp1+linear",
        enable_depth_mask=True,
    )

    result = run_dpt_head(_backbone(), images, config)

    assert result.ready
    assert result.values is not None
    assert result.confidence is not None
    assert result.depth_mask_logits is not None
    assert tuple(result.values.shape) == (1, 2, 2, 2, 1)
    assert tuple(result.confidence.shape) == (1, 2, 2, 2)
    assert tuple(result.depth_mask_logits.shape) == (1, 2, 2, 2)


def test_frame_chunked_dense_head_matches_unchunked_fixture():
    images = mx.ones((1, 3, 3, 2, 2), dtype=mx.float32)
    backbone = _backbone(frames=3)
    config = DPTHeadConfig(head_type="points", patch_size=1, attr_channels=3, activation="linear+expp1")
    tensors = default_dpt_head_tensors(config)
    tensors["dense.output.weight"] = mx.array([0.5, 1.0, 1.5, 2.0], dtype=mx.float32)
    tensors["dense.output.bias"] = mx.array([0.0, 0.1, 0.2, 0.3], dtype=mx.float32)

    unchunked = run_dpt_head(backbone, images, config, tensors, frames_chunk_size=None)
    chunked = run_dpt_head(backbone, images, config, tensors, frames_chunk_size=1)

    assert unchunked.ready
    assert chunked.ready
    assert unchunked.values is not None
    assert chunked.values is not None
    assert unchunked.confidence is not None
    assert chunked.confidence is not None
    np.testing.assert_allclose(np.array(chunked.values), np.array(unchunked.values), rtol=1e-6)
    np.testing.assert_allclose(
        np.array(chunked.confidence),
        np.array(unchunked.confidence),
        rtol=1e-6,
    )


def test_frame_chunked_dense_head_evaluates_each_chunk(monkeypatch):
    eval_calls = []

    def fake_eval(*arrays):
        eval_calls.append(tuple(tuple(array.shape) for array in arrays))

    monkeypatch.setattr("mlx_spatial.hyworld2_heads.mx.eval", fake_eval)
    images = mx.ones((1, 3, 3, 2, 2), dtype=mx.float32)
    config = DPTHeadConfig(
        head_type="depth",
        patch_size=1,
        attr_channels=1,
        activation="linear+expp1+linear",
        enable_depth_mask=True,
    )

    result = run_dpt_head(_backbone(frames=3), images, config, frames_chunk_size=1)

    assert result.ready
    assert eval_calls == [
        ((1, 1, 2, 2, 1), (1, 1, 2, 2), (1, 1, 2, 2)),
        ((1, 1, 2, 2, 1), (1, 1, 2, 2), (1, 1, 2, 2)),
        ((1, 1, 2, 2, 1), (1, 1, 2, 2), (1, 1, 2, 2)),
    ]


def test_dpt_blocks_for_insufficient_intermediate_tokens():
    result = run_dpt_head(
        _backbone(levels=3),
        mx.ones((1, 2, 3, 2, 2), dtype=mx.float32),
        DPTHeadConfig(patch_size=1),
    )

    assert result.blocker is not None
    assert result.blocker.operation == "HY-World DPT intermediate token lookup"
    assert result.blocker.metadata["required_feature_levels"] == 4
    assert result.blocker.metadata["actual_feature_levels"] == 3


def test_dpt_blocks_for_invalid_frame_chunk_size():
    result = run_dpt_head(
        _backbone(),
        mx.ones((1, 2, 3, 2, 2), dtype=mx.float32),
        DPTHeadConfig(patch_size=1),
        frames_chunk_size=0,
    )

    assert result.blocker is not None
    assert result.blocker.operation == "HY-World DPT frame chunk validation"


def test_dpt_blocks_for_zero_frame_dense_inputs():
    result = run_dpt_head(
        _backbone(frames=0),
        mx.ones((1, 0, 3, 2, 2), dtype=mx.float32),
        DPTHeadConfig(patch_size=1),
    )

    assert result.blocker is not None
    assert result.blocker.operation == "HY-World DPT frame count validation"
    assert result.blocker.metadata["frames"] == 0
    assert result.blocker.metadata["shape"] == (1, 0, 3, 2, 2)


def test_dpt_blocks_for_invalid_required_feature_levels():
    result = run_dpt_head(
        _backbone(),
        mx.ones((1, 2, 3, 2, 2), dtype=mx.float32),
        DPTHeadConfig(patch_size=1, required_feature_levels=0),
    )

    assert result.blocker is not None
    assert result.blocker.operation == "HY-World DPT feature level validation"
    assert result.blocker.metadata["required_feature_levels"] == 0


def test_unknown_activation_returns_structured_blocker():
    direct, blocker = apply_hyworld2_activation(mx.ones((1,), dtype=mx.float32), "mystery")

    assert direct is None
    assert blocker is not None
    assert blocker.operation == "HY-World head activation"

    result = run_dpt_head(
        _backbone(),
        mx.ones((1, 2, 3, 2, 2), dtype=mx.float32),
        DPTHeadConfig(patch_size=1, activation="mystery+expp1"),
    )
    assert result.blocker is not None
    assert result.blocker.operation == "HY-World head activation"
