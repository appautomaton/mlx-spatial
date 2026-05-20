#!/usr/bin/env python3
"""Compare official Torch and MLX HY-World VGT block intermediates.

This is a dev-only diagnostic for Slice 4 parity work. It intentionally imports
the vendored official HY-World code and the local MLX port in one process.
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root")
    parser.add_argument("input")
    parser.add_argument("--vendor-root", default="vendors/HY-World-2.0")
    parser.add_argument("--subfolder", default="HY-WorldMirror-2.0")
    parser.add_argument("--target-size", type=int, default=518)
    args = parser.parse_args(argv)

    if os.environ.get("HYWORLD2_TORCH_REF") != "1":
        raise SystemExit("set HYWORLD2_TORCH_REF=1 to run dev-only Torch diagnostics")

    torch_caps = _run_torch(args)
    mlx_caps = _run_mlx(args)

    keys = sorted(set(torch_caps) & set(mlx_caps))
    for key in keys:
        expected = torch_caps[key].astype(np.float32)
        actual = mlx_caps[key].astype(np.float32)
        if expected.shape != actual.shape and expected.size == actual.size:
            actual = actual.reshape(expected.shape)
        diff = np.abs(expected - actual)
        print(
            f"{key}: shape={expected.shape} max={float(diff.max()):.9g} "
            f"mean={float(diff.mean()):.9g}"
        )
        if key == "frame.0.input":
            _print_initial_token_sections(expected, actual, patch_start_idx=7)
    missing_torch = sorted(set(mlx_caps) - set(torch_caps))
    missing_mlx = sorted(set(torch_caps) - set(mlx_caps))
    if missing_torch:
        print(f"missing_torch={missing_torch}")
    if missing_mlx:
        print(f"missing_mlx={missing_mlx}")
    return 0


def _print_initial_token_sections(expected: np.ndarray, actual: np.ndarray, *, patch_start_idx: int) -> None:
    if expected.shape != actual.shape and expected.size == actual.size:
        actual = actual.reshape(expected.shape)
    names = ["camera", "reg0", "reg1", "reg2", "reg3", "pose", "ray"]
    for index, name in enumerate(names):
        diff = np.abs(expected[:, index : index + 1, :] - actual[:, index : index + 1, :])
        print(f"  {name}: max={float(diff.max()):.9g} mean={float(diff.mean()):.9g}")
    patch_diff = np.abs(expected[:, patch_start_idx:, :] - actual[:, patch_start_idx:, :])
    print(f"  patches: max={float(patch_diff.max()):.9g} mean={float(patch_diff.mean()):.9g}")


def _run_torch(args: argparse.Namespace) -> dict[str, np.ndarray]:
    _install_macos_reference_shims()
    sys.path.insert(0, str(Path(args.vendor_root).resolve()))
    from hyworld2.worldrecon import pipeline as pipeline_module

    pipeline = pipeline_module.WorldMirrorPipeline.from_pretrained(
        pretrained_model_name_or_path=args.model_root,
        subfolder=args.subfolder,
        enable_bf16=False,
        disable_heads=["camera", "depth", "normal", "points", "gs"],
    )
    inner = pipeline.model.module if hasattr(pipeline.model, "module") else pipeline.model
    vgt = inner.visual_geometry_transformer
    caps: dict[str, np.ndarray] = {}
    _hook_torch_dino(vgt.patch_embed, caps)
    original = vgt._process_attention_blocks

    def wrapped(tokens, b, seq_len, patch_count, embed_dim, block_idx, blocks, block_type, pos=None):
        target = (
            (b * seq_len, patch_count, embed_dim)
            if block_type == "frame"
            else (b, seq_len * patch_count, embed_dim)
        )
        block_input = tokens.reshape(*target) if tuple(tokens.shape) != target else tokens
        caps[f"{block_type}.{block_idx}.input"] = _to_numpy(block_input)
        if pos is not None:
            pos_target = (
                (b * seq_len, patch_count, 2)
                if block_type == "frame"
                else (b, seq_len * patch_count, 2)
            )
            block_pos = pos.reshape(*pos_target) if tuple(pos.shape) != pos_target else pos
            caps.setdefault(f"{block_type}.{block_idx}.pos", _to_numpy(block_pos))
        result = original(tokens, b, seq_len, patch_count, embed_dim, block_idx, blocks, block_type, pos=pos)
        caps[f"{block_type}.{block_idx}.output"] = _to_numpy(result)
        return result

    vgt._process_attention_blocks = wrapped
    img_paths, _ = pipeline_module.prepare_input(
        args.input,
        target_size=args.target_size,
        fps=1,
        min_frames=1,
        max_frames=1,
    )
    effective_size = pipeline_module.compute_adaptive_target_size(img_paths, args.target_size)
    pipeline._run_inference(img_paths, effective_size, None, None)
    return caps


def _hook_torch_dino(patch_embed: Any, caps: dict[str, np.ndarray]) -> None:
    blocks = getattr(patch_embed, "blocks", None)
    if blocks is None:
        return
    if len(blocks) > 0:
        block0 = blocks[0]
        def capture_once(name: str):
            def hook(_module, _inputs, output):
                if name not in caps:
                    caps[name] = _to_numpy(output)
                return None

            return hook

        block0.norm1.register_forward_hook(capture_once("dino.0.norm1"))
        block0.attn.register_forward_hook(capture_once("dino.0.attn"))
        block0.norm2.register_forward_hook(capture_once("dino.0.norm2"))
        block0.mlp.register_forward_hook(capture_once("dino.0.mlp"))
    for index, block in enumerate(blocks):
        def hook(_module, inputs, output, *, block_index=index):
            caps[f"dino.{block_index}.input"] = _to_numpy(inputs[0])
            caps[f"dino.{block_index}.output"] = _to_numpy(output)

        block.register_forward_hook(hook)


def _run_mlx(args: argparse.Namespace) -> dict[str, np.ndarray]:
    import mlx.core as mx

    if os.environ.get("HYWORLD2_MLX_CPU") == "1":
        mx.set_default_device(mx.cpu)

    from mlx_spatial.hyworld2_assets import inspect_hyworld2_model_assets
    from mlx_spatial.hyworld2_inference import (
        _execution_heads_for_hyworld2,
        _fixture_max_tokens,
        _intermediate_layers_for_hyworld2,
        _load_real_hyworld2_tensors,
    )
    from mlx_spatial.hyworld2_preprocess import preprocess_hyworld2_images
    import mlx_spatial.hyworld2_vit as hyworld2_vit
    import mlx_spatial.hyworld2_worldmirror as worldmirror
    from mlx_spatial.hyworld2_layers import apply_layer_scale, layer_norm
    from mlx_spatial.hyworld2_transformer import _block_mlp_with_dict, _self_attention
    from mlx_spatial.hyworld2_worldmirror import (
        VisualGeometryTransformerConfig,
        run_visual_geometry_transformer,
    )

    caps: dict[str, np.ndarray] = {}
    inspection = inspect_hyworld2_model_assets(args.model_root, requested_heads=("camera",))
    if inspection.blocker is not None or inspection.config is None:
        raise RuntimeError(f"HY-World inspection failed: {inspection.blocker}")
    preprocessed = preprocess_hyworld2_images(args.input, memory_profile="balanced")
    execution_heads = _execution_heads_for_hyworld2(("camera",))
    real_tensors = _load_real_hyworld2_tensors(Path(args.model_root) / args.subfolder / "model.safetensors", execution_heads)
    config = VisualGeometryTransformerConfig.from_model_config(
        inspection.config,
        max_tokens=_fixture_max_tokens(preprocessed),
        max_attention_bytes=4_000_000_000,
        intermediate_layers=_intermediate_layers_for_hyworld2(inspection.config.model_size, inspection.config.depth, False),
    )

    original = worldmirror.run_vgt_block
    original_dino = hyworld2_vit.run_dino_block

    def wrapped_dino(hidden_states, config, tensors, *, block_index):
        caps[f"dino.{block_index}.input"] = np.array(hidden_states, dtype=np.float32)
        if block_index == 0:
            layer = "patch_embed.blocks.0"
            norm1 = layer_norm(
                hidden_states,
                tensors[f"{layer}.norm1.weight"],
                tensors[f"{layer}.norm1.bias"],
                eps=1e-6,
            )
            caps["dino.0.norm1"] = np.array(norm1, dtype=np.float32)
            attn, blocker = _self_attention(
                norm1,
                config,
                tensors,
                layer_index=0,
                mode="patch_embed.blocks",
                rope_positions=None,
            )
            if blocker is None:
                caps["dino.0.attn"] = np.array(attn, dtype=np.float32)
                after_attn = hidden_states + apply_layer_scale(attn, tensors.get(f"{layer}.ls1.gamma"))
                norm2 = layer_norm(
                    after_attn,
                    tensors[f"{layer}.norm2.weight"],
                    tensors[f"{layer}.norm2.bias"],
                    eps=1e-6,
                )
                caps["dino.0.norm2"] = np.array(norm2, dtype=np.float32)
                mlp = _block_mlp_with_dict(norm2, layer, tensors)
                caps["dino.0.mlp"] = np.array(mlp, dtype=np.float32)
        result, blocker = original_dino(
            hidden_states,
            config,
            tensors,
            block_index=block_index,
        )
        if blocker is None:
            caps[f"dino.{block_index}.output"] = np.array(result, dtype=np.float32)
        return result, blocker

    def wrapped(hidden_states, config, tensors, *, layer_index, mode, rope_positions=None):
        caps[f"{mode}.{layer_index}.input"] = np.array(hidden_states, dtype=np.float32)
        if rope_positions is not None:
            caps.setdefault(f"{mode}.{layer_index}.pos", np.array(rope_positions, dtype=np.float32))
        result, blocker = original(
            hidden_states,
            config,
            tensors,
            layer_index=layer_index,
            mode=mode,
            rope_positions=rope_positions,
        )
        if blocker is None:
            caps[f"{mode}.{layer_index}.output"] = np.array(result, dtype=np.float32)
        return result, blocker

    worldmirror.run_vgt_block = wrapped
    hyworld2_vit.run_dino_block = wrapped_dino
    try:
        output = run_visual_geometry_transformer(
            preprocessed.tensor,
            config,
            real_tensors["visual_transformer"],
        )
        if output.blocker is not None:
            raise RuntimeError(f"MLX VGT failed: {output.blocker}")
        mx.eval(output.tokens)
    finally:
        worldmirror.run_vgt_block = original
        hyworld2_vit.run_dino_block = original_dino
    return caps


def _install_macos_reference_shims() -> None:
    import torch

    def flash_attn_func(q, k, v, dropout_p: float = 0.0, **_: Any):
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t, dropout_p=dropout_p)
        return out.transpose(1, 2)

    flash_v3 = types.ModuleType("flash_attn_interface")
    flash_v3.flash_attn_func = flash_attn_func
    sys.modules.setdefault("flash_attn_interface", flash_v3)

    flash_pkg = types.ModuleType("flash_attn")
    flash_iface = types.ModuleType("flash_attn.flash_attn_interface")
    flash_iface.flash_attn_func = flash_attn_func
    flash_pkg.flash_attn_interface = flash_iface
    sys.modules.setdefault("flash_attn", flash_pkg)
    sys.modules.setdefault("flash_attn.flash_attn_interface", flash_iface)

    gsplat_pkg = types.ModuleType("gsplat")
    gsplat_rendering = types.ModuleType("gsplat.rendering")
    gsplat_strategy = types.ModuleType("gsplat.strategy")

    def rasterization(*_: Any, **__: Any):
        raise RuntimeError("gsplat rasterization is unavailable in local VGT diagnostics")

    class DefaultStrategy:
        pass

    gsplat_rendering.rasterization = rasterization
    gsplat_strategy.DefaultStrategy = DefaultStrategy
    gsplat_pkg.rendering = gsplat_rendering
    gsplat_pkg.strategy = gsplat_strategy
    sys.modules.setdefault("gsplat", gsplat_pkg)
    sys.modules.setdefault("gsplat.rendering", gsplat_rendering)
    sys.modules.setdefault("gsplat.strategy", gsplat_strategy)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
