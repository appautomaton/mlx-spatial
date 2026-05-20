import numpy as np
import mlx.core as mx

from mlx_spatial.sam3d_ss_flow import (
    SAM3D_SS_LATENT_ORDER,
    SAM3D_SS_SHARED_POSE_NAME,
    Sam3dSSFlowConfig,
    _run_ss_mot_self_attention,
    infer_sam3d_ss_flow_config,
    project_sam3d_ss_latents_to_transformer,
    project_sam3d_ss_transformer_to_latents,
    run_sam3d_ss_shortcut_flow,
)


def test_sam3d_ss_latent_mapping_merges_and_splits_pose_modalities():
    tensors = _tiny_latent_mapping_tensors(model_channels=3)
    latents = {
        "shape": mx.ones((1, 2, 2), dtype=mx.float32),
        "6drotation_normalized": mx.ones((1, 1, 1), dtype=mx.float32) * 2,
        "translation": mx.ones((1, 1, 1), dtype=mx.float32) * 3,
        "scale": mx.ones((1, 1, 1), dtype=mx.float32) * 4,
        "translation_scale": mx.ones((1, 1, 1), dtype=mx.float32) * 5,
    }

    projected = project_sam3d_ss_latents_to_transformer(latents, tensors)
    restored = project_sam3d_ss_transformer_to_latents(projected, tensors)

    assert tuple(projected["shape"].shape) == (1, 2, 3)
    assert tuple(projected["6drotation_normalized"].shape) == (1, 4, 3)
    assert set(restored) == set(SAM3D_SS_LATENT_ORDER)
    assert tuple(restored["shape"].shape) == (1, 2, 2)
    assert tuple(restored["translation_scale"].shape) == (1, 1, 1)


def test_sam3d_ss_mot_self_attention_keeps_shape_protected_from_pose_tokens():
    tensors = _tiny_attention_tensors()
    config = Sam3dSSFlowConfig(model_channels=2, num_heads=1, num_blocks=1, attention_chunk_size=None)
    first = {
        "shape": mx.array([[[1.0, 0.0]]], dtype=mx.float32),
        "6drotation_normalized": mx.array([[[0.0, 1.0]]], dtype=mx.float32),
    }
    second = {
        "shape": first["shape"],
        "6drotation_normalized": mx.array([[[10.0, -7.0]]], dtype=mx.float32),
    }

    out_first = _run_ss_mot_self_attention(first, tensors, prefix="block.self_attn.", config=config)
    out_second = _run_ss_mot_self_attention(second, tensors, prefix="block.self_attn.", config=config)

    assert np.allclose(np.array(out_first["shape"]), np.array(out_second["shape"]))
    assert not np.allclose(
        np.array(out_first["6drotation_normalized"]),
        np.array(out_second["6drotation_normalized"]),
    )


def test_sam3d_ss_mot_self_attention_matches_reference_shape_and_pose_outputs():
    tensors = _tiny_attention_tensors()
    config = Sam3dSSFlowConfig(model_channels=2, num_heads=1, num_blocks=1, attention_chunk_size=None)
    hidden = {
        "shape": mx.array([[[1.0, 0.0], [0.25, 0.75]]], dtype=mx.float32),
        SAM3D_SS_SHARED_POSE_NAME: mx.array([[[0.0, 1.0], [1.0, 1.0]]], dtype=mx.float32),
    }

    actual = _run_ss_mot_self_attention(hidden, tensors, prefix="block.self_attn.", config=config)

    shape = np.array(hidden["shape"], dtype=np.float32)[:, :, None, :]
    pose = np.array(hidden[SAM3D_SS_SHARED_POSE_NAME], dtype=np.float32)[:, :, None, :]
    q_shape = _reference_rms_norm(shape)
    q_pose = _reference_rms_norm(pose)
    expected_shape = _reference_attention(q_shape, q_shape, shape)
    expected_pose = _reference_attention(
        q_pose,
        np.concatenate((q_pose, q_shape), axis=1),
        np.concatenate((pose, shape), axis=1),
    )

    assert np.allclose(np.array(actual["shape"]), expected_shape[:, :, 0, :], atol=1e-3)
    assert np.allclose(
        np.array(actual[SAM3D_SS_SHARED_POSE_NAME]),
        expected_pose[:, :, 0, :],
        atol=1e-3,
    )


def test_sam3d_ss_shortcut_flow_zero_block_fixture_is_seeded_and_reproducible():
    tensors = _tiny_flow_tensors(model_channels=3)
    config = infer_sam3d_ss_flow_config(
        tensors,
        cfg_strength=7.0,
        rescale_t=3.0,
        attention_chunk_size=4,
    )
    cond = mx.zeros((1, 2, 3), dtype=mx.float32)

    first = run_sam3d_ss_shortcut_flow(cond, tensors, seed=11, steps=2, config=config)
    second = run_sam3d_ss_shortcut_flow(cond, tensors, seed=11, steps=2, config=config)

    assert first.metadata["schedule"] == (0.0, 0.25, 1.0)
    assert first.metadata["cfg_strength"] == 7.0
    assert first.metadata["num_blocks"] == 0
    for name in SAM3D_SS_LATENT_ORDER:
        assert np.allclose(np.array(first.latents[name]), np.array(second.latents[name]))


def _tiny_latent_mapping_tensors(model_channels: int) -> dict[str, mx.array]:
    tensors: dict[str, mx.array] = {}
    channels = {
        "shape": 2,
        "6drotation_normalized": 1,
        "translation": 1,
        "scale": 1,
        "translation_scale": 1,
    }
    token_counts = {
        "shape": 2,
        "6drotation_normalized": 1,
        "translation": 1,
        "scale": 1,
        "translation_scale": 1,
    }
    prefix = "reverse_fn.backbone.latent_mapping."
    for name in SAM3D_SS_LATENT_ORDER:
        in_channels = channels[name]
        weight = np.zeros((model_channels, in_channels), dtype=np.float32)
        weight[: min(model_channels, in_channels), : min(model_channels, in_channels)] = np.eye(
            min(model_channels, in_channels),
            dtype=np.float32,
        )
        tensors[f"{prefix}{name}.input_layer.weight"] = mx.array(weight)
        tensors[f"{prefix}{name}.input_layer.bias"] = mx.zeros((model_channels,), dtype=mx.float32)
        tensors[f"{prefix}{name}.pos_emb"] = mx.zeros((token_counts[name], model_channels), dtype=mx.float32)
        out_weight = np.zeros((in_channels, model_channels), dtype=np.float32)
        out_weight[: min(model_channels, in_channels), : min(model_channels, in_channels)] = np.eye(
            min(model_channels, in_channels),
            dtype=np.float32,
        )
        tensors[f"{prefix}{name}.out_layer.weight"] = mx.array(out_weight)
        tensors[f"{prefix}{name}.out_layer.bias"] = mx.zeros((in_channels,), dtype=mx.float32)
    return tensors


def _tiny_attention_tensors() -> dict[str, mx.array]:
    tensors: dict[str, mx.array] = {}
    qkv = np.concatenate([np.eye(2, dtype=np.float32)] * 3, axis=0)
    for name in ("shape", "6drotation_normalized"):
        tensors[f"block.self_attn.to_qkv.{name}.weight"] = mx.array(qkv)
        tensors[f"block.self_attn.to_qkv.{name}.bias"] = mx.zeros((6,), dtype=mx.float32)
        tensors[f"block.self_attn.q_rms_norm.{name}.gamma"] = mx.ones((1, 2), dtype=mx.float32)
        tensors[f"block.self_attn.k_rms_norm.{name}.gamma"] = mx.ones((1, 2), dtype=mx.float32)
        tensors[f"block.self_attn.to_out.{name}.weight"] = mx.eye(2, dtype=mx.float32)
        tensors[f"block.self_attn.to_out.{name}.bias"] = mx.zeros((2,), dtype=mx.float32)
    return tensors


def _tiny_flow_tensors(model_channels: int) -> dict[str, mx.array]:
    tensors = _tiny_latent_mapping_tensors(model_channels)
    prefix = "reverse_fn.backbone."
    tensors[f"{prefix}t_embedder.mlp.0.weight"] = mx.zeros((model_channels, 256), dtype=mx.float32)
    tensors[f"{prefix}t_embedder.mlp.0.bias"] = mx.zeros((model_channels,), dtype=mx.float32)
    tensors[f"{prefix}t_embedder.mlp.2.weight"] = mx.zeros((model_channels, model_channels), dtype=mx.float32)
    tensors[f"{prefix}t_embedder.mlp.2.bias"] = mx.zeros((model_channels,), dtype=mx.float32)
    tensors[f"{prefix}d_embedder.mlp.0.weight"] = mx.zeros((model_channels, 256), dtype=mx.float32)
    tensors[f"{prefix}d_embedder.mlp.0.bias"] = mx.zeros((model_channels,), dtype=mx.float32)
    tensors[f"{prefix}d_embedder.mlp.2.weight"] = mx.zeros((model_channels, model_channels), dtype=mx.float32)
    tensors[f"{prefix}d_embedder.mlp.2.bias"] = mx.zeros((model_channels,), dtype=mx.float32)
    return tensors


def _reference_rms_norm(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values.astype(np.float32), axis=-1, keepdims=True)
    return values / np.maximum(norm, 1e-12) * np.sqrt(values.shape[-1])


def _reference_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    scores = np.einsum("blhd,bmhd->bhlm", query, key) * (query.shape[-1] ** -0.5)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return np.einsum("bhlm,bmhd->blhd", weights, value)
