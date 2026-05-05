import mlx.core as mx
import numpy as np

from mlx_spatial.sam3d_transformer import (
    run_sam3d_multihead_attention,
    run_sam3d_timestep_embedder,
    sam3d_multihead_rms_norm,
    sam3d_scaled_dot_product_attention,
    sam3d_timestep_embedding,
)


def test_sam3d_timestep_embedding_matches_cos_sin_contract():
    emb = sam3d_timestep_embedding(mx.array([0.0], dtype=mx.float32), 5)

    assert tuple(emb.shape) == (1, 5)
    assert np.allclose(np.array(emb), np.array([[1.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32))


def test_run_sam3d_timestep_embedder_uses_two_linear_layers():
    tensors = {
        "time.mlp.0.weight": mx.ones((3, 4), dtype=mx.float32),
        "time.mlp.0.bias": mx.zeros((3,), dtype=mx.float32),
        "time.mlp.2.weight": mx.ones((2, 3), dtype=mx.float32),
        "time.mlp.2.bias": mx.zeros((2,), dtype=mx.float32),
    }

    out = run_sam3d_timestep_embedder(mx.array([0.0], dtype=mx.float32), tensors, prefix="time.")

    assert tuple(out.shape) == (1, 2)


def test_sam3d_scaled_dot_product_attention_chunked_matches_dense():
    query = mx.array(np.arange(24, dtype=np.float32).reshape(1, 3, 2, 4) / 10.0)
    key = mx.array(np.arange(32, dtype=np.float32).reshape(1, 4, 2, 4) / 20.0)
    value = mx.array(np.arange(40, dtype=np.float32).reshape(1, 4, 2, 5) / 30.0)

    dense = sam3d_scaled_dot_product_attention(query, key, value)
    chunked = sam3d_scaled_dot_product_attention(query, key, value, chunk_size=1)

    assert np.allclose(np.array(chunked), np.array(dense), atol=2e-3)


def test_sam3d_multihead_rms_norm_normalizes_last_dimension():
    values = mx.ones((1, 2, 3, 4), dtype=mx.float32)
    gamma = mx.ones((3, 4), dtype=mx.float32)

    out = sam3d_multihead_rms_norm(values, gamma)

    assert np.allclose(np.array(out), np.ones((1, 2, 3, 4), dtype=np.float32))


def test_run_sam3d_multihead_attention_self_shape():
    x = mx.ones((1, 3, 4), dtype=mx.float32)
    tensors = {
        "attn.to_qkv.weight": mx.zeros((12, 4), dtype=mx.float32),
        "attn.to_qkv.bias": mx.zeros((12,), dtype=mx.float32),
        "attn.to_out.weight": mx.ones((4, 4), dtype=mx.float32),
        "attn.to_out.bias": mx.zeros((4,), dtype=mx.float32),
    }

    out = run_sam3d_multihead_attention(x, tensors, prefix="attn.", num_heads=2, chunk_size=2)

    assert tuple(out.shape) == (1, 3, 4)
    assert np.allclose(np.array(out), np.zeros((1, 3, 4), dtype=np.float32))


def test_run_sam3d_multihead_attention_cross_shape():
    x = mx.ones((1, 2, 4), dtype=mx.float32)
    context = mx.ones((1, 3, 5), dtype=mx.float32)
    tensors = {
        "cross.to_q.weight": mx.zeros((4, 4), dtype=mx.float32),
        "cross.to_q.bias": mx.zeros((4,), dtype=mx.float32),
        "cross.to_kv.weight": mx.zeros((8, 5), dtype=mx.float32),
        "cross.to_kv.bias": mx.zeros((8,), dtype=mx.float32),
        "cross.to_out.weight": mx.ones((4, 4), dtype=mx.float32),
        "cross.to_out.bias": mx.zeros((4,), dtype=mx.float32),
    }

    out = run_sam3d_multihead_attention(x, tensors, prefix="cross.", num_heads=2, context=context, chunk_size=1)

    assert tuple(out.shape) == (1, 2, 4)
