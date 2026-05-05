"""SAM 3D Objects Stage-1 sparse-structure ShortCut flow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from .checkpoint import load_checkpoint_tensors
from .sam3d_flow import sam3d_classifier_free_guidance, sam3d_seeded_normal, sam3d_shortcut_schedule
from .sam3d_transformer import (
    run_sam3d_timestep_embedder,
    sam3d_layer_norm,
    sam3d_multihead_rms_norm,
    sam3d_scaled_dot_product_attention,
)


SAM3D_SS_GENERATOR_PREFIX = "_base_models.generator."
SAM3D_SS_LATENT_ORDER = (
    "shape",
    "6drotation_normalized",
    "translation",
    "scale",
    "translation_scale",
)
SAM3D_SS_SHARED_POSE_NAME = "6drotation_normalized"
SAM3D_SS_SHARED_POSE_ORDER = (
    "6drotation_normalized",
    "translation",
    "scale",
    "translation_scale",
)


@dataclass(frozen=True)
class Sam3dSSFlowConfig:
    """Active SAM3D sparse-structure flow settings."""

    model_channels: int = 1024
    num_heads: int = 16
    num_blocks: int = 24
    cfg_strength: float = 7.0
    cfg_interval: tuple[float, float] = (0.0, 500.0)
    rescale_t: float = 3.0
    time_scale: float = 1000.0
    no_shortcut: bool = True
    attention_chunk_size: int = 128


@dataclass(frozen=True)
class Sam3dSSFlowOutput:
    """Generated Stage-1 latents before SS decoding."""

    latents: dict[str, mx.array]
    metadata: dict[str, object]


def load_sam3d_ss_generator_tensors(path: str | Path) -> dict[str, mx.array]:
    """Load active SAM3D SS generator tensors with the checkpoint prefix removed."""

    tensors = load_checkpoint_tensors(path, prefixes=(SAM3D_SS_GENERATOR_PREFIX,))
    return {
        key[len(SAM3D_SS_GENERATOR_PREFIX) :]: value
        for key, value in tensors.items()
        if key.startswith(SAM3D_SS_GENERATOR_PREFIX)
    }


def infer_sam3d_ss_flow_config(
    tensors: dict[str, mx.array],
    *,
    cfg_strength: float = 7.0,
    cfg_interval: tuple[float, float] = (0.0, 500.0),
    rescale_t: float = 3.0,
    attention_chunk_size: int = 128,
) -> Sam3dSSFlowConfig:
    """Infer active SS flow dimensions from checkpoint tensors."""

    prefix = "reverse_fn.backbone."
    model_channels = int(tensors[f"{prefix}t_embedder.mlp.2.bias"].shape[0])
    block_indices = {
        int(key.split(".")[3])
        for key in tensors
        if key.startswith(f"{prefix}blocks.") and key.split(".")[3].isdigit()
    }
    num_blocks = max(block_indices) + 1 if block_indices else 0
    gamma = tensors.get(f"{prefix}blocks.0.self_attn.q_rms_norm.shape.gamma")
    num_heads = int(gamma.shape[0]) if gamma is not None else 16
    return Sam3dSSFlowConfig(
        model_channels=model_channels,
        num_heads=num_heads,
        num_blocks=num_blocks,
        cfg_strength=float(cfg_strength),
        cfg_interval=cfg_interval,
        rescale_t=float(rescale_t),
        attention_chunk_size=int(attention_chunk_size),
    )


def make_sam3d_ss_initial_latents(
    tensors: dict[str, mx.array],
    *,
    seed: int = 42,
) -> dict[str, mx.array]:
    """Create official Stage-1 noise latents from latent mapping tensor shapes."""

    prefix = "reverse_fn.backbone.latent_mapping."
    latents: dict[str, mx.array] = {}
    for index, latent_name in enumerate(SAM3D_SS_LATENT_ORDER):
        weight = tensors[f"{prefix}{latent_name}.input_layer.weight"]
        pos_emb = tensors[f"{prefix}{latent_name}.pos_emb"]
        token_count = int(pos_emb.shape[0])
        channels = int(weight.shape[1])
        latents[latent_name] = sam3d_seeded_normal((1, token_count, channels), seed=seed + index)
    return latents


def run_sam3d_ss_shortcut_flow(
    condition_tokens: mx.array,
    tensors: dict[str, mx.array],
    *,
    seed: int = 42,
    steps: int = 2,
    config: Sam3dSSFlowConfig | None = None,
) -> Sam3dSSFlowOutput:
    """Run the exact active SAM3D Stage-1 ShortCut Euler sampler."""

    if condition_tokens.ndim != 3:
        raise ValueError(f"SAM3D SS condition tokens must have shape [B,T,C], got {tuple(condition_tokens.shape)}")
    cfg = config or infer_sam3d_ss_flow_config(tensors)
    latents = make_sam3d_ss_initial_latents(tensors, seed=seed)
    schedule = sam3d_shortcut_schedule(
        steps,
        rescale_t=cfg.rescale_t,
        no_shortcut=cfg.no_shortcut,
        time_scale=cfg.time_scale,
    )
    for index in range(len(schedule.t_seq) - 1):
        t0 = float(schedule.t_seq[index])
        t1 = float(schedule.t_seq[index + 1])
        t_scaled = t0 * cfg.time_scale
        d_scaled = float(schedule.shortcut_d or 0.0) * cfg.time_scale
        conditional = _run_ss_velocity(
            latents,
            condition_tokens,
            tensors,
            t_scaled=t_scaled,
            d_scaled=d_scaled,
            config=cfg,
        )
        unconditional = _run_ss_velocity(
            latents,
            mx.zeros_like(condition_tokens),
            tensors,
            t_scaled=t_scaled,
            d_scaled=d_scaled,
            config=cfg,
        )
        velocity = {
            name: sam3d_classifier_free_guidance(
                conditional[name],
                unconditional[name],
                strength=cfg.cfg_strength,
                interval=cfg.cfg_interval,
                t_scaled=t_scaled,
            )
            for name in latents
        }
        dt = t1 - t0
        latents = {name: latents[name] + dt * velocity[name] for name in latents}
        mx.eval(*latents.values())
    return Sam3dSSFlowOutput(
        latents=latents,
        metadata={
            "latent_shapes": {name: tuple(int(value) for value in latent.shape) for name, latent in latents.items()},
            "steps": int(steps),
            "schedule": tuple(float(value) for value in schedule.t_seq.tolist()),
            "cfg_strength": float(cfg.cfg_strength),
            "cfg_interval": tuple(float(value) for value in cfg.cfg_interval),
            "rescale_t": float(cfg.rescale_t),
            "time_scale": float(cfg.time_scale),
            "attention_chunk_size": int(cfg.attention_chunk_size),
            "num_blocks": int(cfg.num_blocks),
            "num_heads": int(cfg.num_heads),
        },
    )


def project_sam3d_ss_latents_to_transformer(
    latents: dict[str, mx.array],
    tensors: dict[str, mx.array],
) -> dict[str, mx.array]:
    """Project and merge SS latents into the active MOT transformer modalities."""

    prefix = "reverse_fn.backbone.latent_mapping."
    projected: dict[str, mx.array] = {}
    for latent_name in SAM3D_SS_LATENT_ORDER:
        projected[latent_name] = _linear(
            latents[latent_name],
            tensors[f"{prefix}{latent_name}.input_layer.weight"],
            tensors[f"{prefix}{latent_name}.input_layer.bias"],
        ) + tensors[f"{prefix}{latent_name}.pos_emb"][None].astype(latents[latent_name].dtype)
    pose = mx.concatenate([projected[name] for name in SAM3D_SS_SHARED_POSE_ORDER], axis=1)
    return {"shape": projected["shape"], SAM3D_SS_SHARED_POSE_NAME: pose}


def project_sam3d_ss_transformer_to_latents(
    hidden: dict[str, mx.array],
    tensors: dict[str, mx.array],
) -> dict[str, mx.array]:
    """Split and project active MOT transformer modalities back to Stage-1 latents."""

    prefix = "reverse_fn.backbone.latent_mapping."
    split: dict[str, mx.array] = {"shape": hidden["shape"]}
    pose_hidden = hidden[SAM3D_SS_SHARED_POSE_NAME]
    start = 0
    for latent_name in SAM3D_SS_SHARED_POSE_ORDER:
        token_count = int(tensors[f"{prefix}{latent_name}.pos_emb"].shape[0])
        split[latent_name] = pose_hidden[:, start : start + token_count, :]
        start += token_count
    return {
        latent_name: _linear(
            sam3d_layer_norm(split[latent_name]),
            tensors[f"{prefix}{latent_name}.out_layer.weight"],
            tensors[f"{prefix}{latent_name}.out_layer.bias"],
        )
        for latent_name in SAM3D_SS_LATENT_ORDER
    }


def _run_ss_velocity(
    latents: dict[str, mx.array],
    condition_tokens: mx.array,
    tensors: dict[str, mx.array],
    *,
    t_scaled: float,
    d_scaled: float,
    config: Sam3dSSFlowConfig,
) -> dict[str, mx.array]:
    prefix = "reverse_fn.backbone."
    hidden = project_sam3d_ss_latents_to_transformer(latents, tensors)
    t_emb = run_sam3d_timestep_embedder(
        mx.array([t_scaled], dtype=mx.float32),
        tensors,
        prefix=f"{prefix}t_embedder.",
    )
    d_emb = run_sam3d_timestep_embedder(
        mx.array([d_scaled], dtype=mx.float32),
        tensors,
        prefix=f"{prefix}d_embedder.",
    )
    modulation = t_emb + d_emb
    for block_index in range(config.num_blocks):
        hidden = _run_ss_mot_block(
            hidden,
            condition_tokens,
            modulation,
            tensors,
            prefix=f"{prefix}blocks.{block_index}.",
            config=config,
        )
        mx.eval(hidden["shape"], hidden[SAM3D_SS_SHARED_POSE_NAME])
    return project_sam3d_ss_transformer_to_latents(hidden, tensors)


def _run_ss_mot_block(
    hidden: dict[str, mx.array],
    condition_tokens: mx.array,
    modulation: mx.array,
    tensors: dict[str, mx.array],
    *,
    prefix: str,
    config: Sam3dSSFlowConfig,
) -> dict[str, mx.array]:
    t_emb = _linear(
        _silu(modulation.astype(hidden["shape"].dtype)),
        tensors[f"{prefix}adaLN_modulation.1.weight"],
        tensors[f"{prefix}adaLN_modulation.1.bias"],
    )
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(t_emb, 6, axis=1)

    normed = {
        name: _apply_adaln(sam3d_layer_norm(values), shift_msa, scale_msa)
        for name, values in hidden.items()
    }
    attended = _run_ss_mot_self_attention(normed, tensors, prefix=f"{prefix}self_attn.", config=config)
    hidden = {name: hidden[name] + attended[name] * gate_msa[:, None, :].astype(attended[name].dtype) for name in hidden}

    crossed: dict[str, mx.array] = {}
    for name, values in hidden.items():
        h = sam3d_layer_norm(
            values,
            tensors[f"{prefix}norm2.{name}.weight"],
            tensors[f"{prefix}norm2.{name}.bias"],
        )
        crossed[name] = _run_ss_cross_attention(
            h,
            condition_tokens,
            tensors,
            prefix=f"{prefix}cross_attn.{name}.",
            config=config,
        )
    hidden = {name: hidden[name] + crossed[name] for name in hidden}

    mlp_out: dict[str, mx.array] = {}
    for name, values in hidden.items():
        h = _apply_adaln(sam3d_layer_norm(values), shift_mlp, scale_mlp)
        mlp_out[name] = _feed_forward(h, tensors, prefix=f"{prefix}mlp.{name}.")
    return {name: hidden[name] + mlp_out[name] * gate_mlp[:, None, :].astype(mlp_out[name].dtype) for name in hidden}


def _run_ss_mot_self_attention(
    hidden: dict[str, mx.array],
    tensors: dict[str, mx.array],
    *,
    prefix: str,
    config: Sam3dSSFlowConfig,
) -> dict[str, mx.array]:
    qkv = {
        name: _qkv_for_latent(values, tensors, prefix=f"{prefix}to_qkv.{name}.", config=config)
        for name, values in hidden.items()
    }
    query = {
        name: sam3d_multihead_rms_norm(qkv[name][0], tensors[f"{prefix}q_rms_norm.{name}.gamma"])
        for name in hidden
    }
    key = {
        name: sam3d_multihead_rms_norm(qkv[name][1], tensors[f"{prefix}k_rms_norm.{name}.gamma"])
        for name in hidden
    }
    value = {name: qkv[name][2] for name in hidden}

    out: dict[str, mx.array] = {}
    out["shape"] = _attention_output_for_latent(
        query["shape"],
        key["shape"],
        value["shape"],
        hidden["shape"],
        tensors,
        prefix=f"{prefix}to_out.shape.",
        config=config,
    )
    pose_key = mx.concatenate((key[SAM3D_SS_SHARED_POSE_NAME], key["shape"]), axis=1)
    pose_value = mx.concatenate((value[SAM3D_SS_SHARED_POSE_NAME], value["shape"]), axis=1)
    out[SAM3D_SS_SHARED_POSE_NAME] = _attention_output_for_latent(
        query[SAM3D_SS_SHARED_POSE_NAME],
        pose_key,
        pose_value,
        hidden[SAM3D_SS_SHARED_POSE_NAME],
        tensors,
        prefix=f"{prefix}to_out.{SAM3D_SS_SHARED_POSE_NAME}.",
        config=config,
    )
    return out


def _qkv_for_latent(
    values: mx.array,
    tensors: dict[str, mx.array],
    *,
    prefix: str,
    config: Sam3dSSFlowConfig,
) -> tuple[mx.array, mx.array, mx.array]:
    batch, tokens, channels = tuple(int(value) for value in values.shape)
    head_dim = channels // config.num_heads
    qkv = _linear(values, tensors[f"{prefix}weight"], tensors[f"{prefix}bias"])
    qkv = mx.reshape(qkv, (batch, tokens, 3, config.num_heads, head_dim))
    return qkv[:, :, 0, :, :], qkv[:, :, 1, :, :], qkv[:, :, 2, :, :]


def _attention_output_for_latent(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    original: mx.array,
    tensors: dict[str, mx.array],
    *,
    prefix: str,
    config: Sam3dSSFlowConfig,
) -> mx.array:
    attended = sam3d_scaled_dot_product_attention(query, key, value, chunk_size=config.attention_chunk_size)
    merged = mx.reshape(attended, tuple(int(v) for v in original.shape))
    return _linear(merged, tensors[f"{prefix}weight"], tensors[f"{prefix}bias"])


def _run_ss_cross_attention(
    values: mx.array,
    condition_tokens: mx.array,
    tensors: dict[str, mx.array],
    *,
    prefix: str,
    config: Sam3dSSFlowConfig,
) -> mx.array:
    batch, tokens, channels = tuple(int(value) for value in values.shape)
    head_dim = channels // config.num_heads
    context = condition_tokens.astype(values.dtype)
    query = _linear(values, tensors[f"{prefix}to_q.weight"], tensors[f"{prefix}to_q.bias"])
    kv = _linear(context, tensors[f"{prefix}to_kv.weight"], tensors[f"{prefix}to_kv.bias"])
    query = mx.reshape(query, (batch, tokens, config.num_heads, head_dim))
    kv = mx.reshape(kv, (batch, int(context.shape[1]), 2, config.num_heads, head_dim))
    attended = sam3d_scaled_dot_product_attention(
        query,
        kv[:, :, 0, :, :],
        kv[:, :, 1, :, :],
        chunk_size=config.attention_chunk_size,
    )
    return _linear(mx.reshape(attended, (batch, tokens, channels)), tensors[f"{prefix}to_out.weight"], tensors[f"{prefix}to_out.bias"])


def _apply_adaln(values: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    return values * (1.0 + scale[:, None, :].astype(values.dtype)) + shift[:, None, :].astype(values.dtype)


def _feed_forward(values: mx.array, tensors: dict[str, mx.array], *, prefix: str) -> mx.array:
    hidden = _linear(values, tensors[f"{prefix}mlp.0.weight"], tensors[f"{prefix}mlp.0.bias"])
    hidden = _gelu_tanh(hidden)
    return _linear(hidden, tensors[f"{prefix}mlp.2.weight"], tensors[f"{prefix}mlp.2.bias"])


def _silu(values: mx.array) -> mx.array:
    return values * mx.sigmoid(values)


def _gelu_tanh(values: mx.array) -> mx.array:
    return 0.5 * values * (1.0 + mx.tanh(0.7978845608028654 * (values + 0.044715 * values * values * values)))


def _linear(values: mx.array, weight: mx.array, bias: mx.array | None) -> mx.array:
    out = values @ mx.transpose(weight.astype(values.dtype))
    if bias is not None:
        out = out + bias.astype(out.dtype)
    return out
