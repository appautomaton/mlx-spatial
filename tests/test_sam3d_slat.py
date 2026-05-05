import numpy as np
import mlx.core as mx

from mlx_spatial.sam3d_slat import (
    Sam3dSparseTensor,
    _layout_for_coords,
    _sparse_conv3d,
    _sparse_downsample,
    _sparse_upsample,
    infer_sam3d_slat_flow_config,
    run_sam3d_slat_flow,
)


def test_sam3d_sparse_downsample_and_upsample_cache_round_trip():
    coords = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 2, 0, 0],
        ],
        dtype=np.int32,
    )
    feats = mx.array([[1.0], [3.0], [9.0]], dtype=mx.float32)
    tensor = Sam3dSparseTensor(coords=coords, feats=feats, layout=_layout_for_coords(coords), spatial_cache={})

    down = _sparse_downsample(tensor, factor=2)
    up = _sparse_upsample(down, factor=2)

    assert down.coords.tolist() == [[0, 0, 0, 0], [0, 1, 0, 0]]
    assert np.allclose(np.array(down.feats), np.array([[2.0], [9.0]], dtype=np.float32))
    assert up.coords.tolist() == coords.tolist()
    assert np.allclose(np.array(up.feats), np.array([[2.0], [2.0], [9.0]], dtype=np.float32))


def test_sam3d_sparse_conv3d_uses_center_kernel_weight():
    coords = np.array([[0, 0, 0, 0]], dtype=np.int32)
    feats = mx.array([[2.0, 4.0]], dtype=mx.float32)
    weight = np.zeros((1, 3, 3, 3, 2), dtype=np.float32)
    weight[0, 1, 1, 1, :] = np.array([10.0, 1.0], dtype=np.float32)
    tensors = {
        "conv.weight": mx.array(weight),
        "conv.bias": mx.array([0.5], dtype=mx.float32),
    }

    out = _sparse_conv3d(coords, feats, tensors, prefix="conv.")

    assert np.allclose(np.array(out), np.array([[24.5]], dtype=np.float32))


def test_sam3d_slat_zero_block_flow_returns_deterministic_denormalized_features():
    tensors = _tiny_slat_flow_tensors()
    config = infer_sam3d_slat_flow_config(
        tensors,
        cfg_strength=1.0,
        rescale_t=1.0,
        attention_chunk_size=4,
    )
    coords = np.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int32)
    cond = mx.zeros((1, 2, 4), dtype=mx.float32)

    first = run_sam3d_slat_flow(
        coords,
        cond,
        tensors,
        seed=5,
        steps=2,
        config=config,
        slat_mean=(1.0, 1.0),
        slat_std=(2.0, 2.0),
    )
    second = run_sam3d_slat_flow(
        coords,
        cond,
        tensors,
        seed=5,
        steps=2,
        config=config,
        slat_mean=(1.0, 1.0),
        slat_std=(2.0, 2.0),
    )

    assert first.metadata["schedule"] == (0.0, 0.5, 1.0)
    assert first.metadata["feature_shape"] == (2, 2)
    assert np.allclose(np.array(first.feats), np.array(second.feats))


def _tiny_slat_flow_tensors():
    prefix = "reverse_fn.backbone."
    return {
        f"{prefix}input_layer.weight": mx.zeros((4, 2), dtype=mx.float32),
        f"{prefix}input_layer.bias": mx.zeros((4,), dtype=mx.float32),
        f"{prefix}out_layer.weight": mx.zeros((2, 4), dtype=mx.float32),
        f"{prefix}out_layer.bias": mx.zeros((2,), dtype=mx.float32),
        f"{prefix}t_embedder.mlp.0.weight": mx.zeros((4, 256), dtype=mx.float32),
        f"{prefix}t_embedder.mlp.0.bias": mx.zeros((4,), dtype=mx.float32),
        f"{prefix}t_embedder.mlp.2.weight": mx.zeros((4, 4), dtype=mx.float32),
        f"{prefix}t_embedder.mlp.2.bias": mx.zeros((4,), dtype=mx.float32),
    }
