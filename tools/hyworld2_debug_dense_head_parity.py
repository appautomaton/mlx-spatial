#!/usr/bin/env python3
"""Compare official Torch and MLX HY-World dense-head intermediates.

This is a dev-only diagnostic for Slice 4 parity work. It feeds the Torch
reference VGT tokens into the MLX DPT head so the comparison isolates the dense
head implementation rather than VGT drift.
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


HEADS = ("depth", "normal", "points")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root")
    parser.add_argument("input")
    parser.add_argument("--vendor-root", default="vendors/HY-World-2.0")
    parser.add_argument("--subfolder", default="HY-WorldMirror-2.0")
    parser.add_argument("--head", choices=HEADS, default="depth")
    parser.add_argument("--target-size", type=int, default=518)
    args = parser.parse_args(argv)

    if os.environ.get("HYWORLD2_TORCH_REF") != "1":
        raise SystemExit("set HYWORLD2_TORCH_REF=1 to run dev-only Torch diagnostics")

    torch_caps, torch_tokens, patch_start_idx, torch_imgs = _run_torch(args)
    mlx_caps = _run_mlx_from_torch_tokens(args, torch_tokens, patch_start_idx, torch_imgs)
    _print_comparison(torch_caps, mlx_caps)
    return 0


def _print_comparison(expected_caps: dict[str, np.ndarray], actual_caps: dict[str, np.ndarray]) -> None:
    keys = sorted(set(expected_caps) & set(actual_caps))
    for key in keys:
        expected = expected_caps[key].astype(np.float32)
        actual = actual_caps[key].astype(np.float32)
        if expected.shape != actual.shape and expected.size == actual.size:
            actual = actual.reshape(expected.shape)
        if expected.shape != actual.shape:
            print(f"{key}: shape mismatch torch={expected.shape} mlx={actual.shape}")
            continue
        diff = np.abs(expected - actual)
        print(
            f"{key}: shape={expected.shape} max={float(diff.max()):.9g} "
            f"mean={float(diff.mean()):.9g} "
            f"ref_mean={float(np.mean(np.abs(expected))):.9g} "
            f"mlx_mean={float(np.mean(np.abs(actual))):.9g}"
        )
    missing_torch = sorted(set(actual_caps) - set(expected_caps))
    missing_mlx = sorted(set(expected_caps) - set(actual_caps))
    if missing_torch:
        print(f"missing_torch={missing_torch}")
    if missing_mlx:
        print(f"missing_mlx={missing_mlx}")


def _run_torch(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, ...], int, np.ndarray]:
    _install_macos_reference_shims()
    sys.path.insert(0, str(Path(args.vendor_root).resolve()))
    from hyworld2.worldrecon import pipeline as pipeline_module

    pipeline = pipeline_module.WorldMirrorPipeline.from_pretrained(
        pretrained_model_name_or_path=args.model_root,
        subfolder=args.subfolder,
        enable_bf16=False,
        disable_heads=[head for head in ("camera", "depth", "normal", "points", "gs") if head != args.head],
    )
    inner = pipeline.model.module if hasattr(pipeline.model, "module") else pipeline.model
    caps: dict[str, np.ndarray] = {}
    token_caps: dict[str, object] = {}
    _wrap_torch_vgt(inner, token_caps)
    _hook_torch_dpt_head(getattr(inner, _torch_head_attr(args.head)), caps, args.head)

    img_paths, _ = pipeline_module.prepare_input(
        args.input,
        target_size=args.target_size,
        fps=1,
        min_frames=1,
        max_frames=1,
    )
    effective_size = pipeline_module.compute_adaptive_target_size(img_paths, args.target_size)
    predictions, imgs, _infer_time = pipeline._run_inference(img_paths, effective_size, None, None)
    caps["input.imgs"] = _to_numpy(imgs)
    _collect_torch_prediction(args.head, predictions, caps)
    return (
        caps,
        tuple(_to_numpy(tensor) for tensor in token_caps["token_list"]),
        int(token_caps["patch_start_idx"]),
        _to_numpy(imgs),
    )


def _wrap_torch_vgt(model: Any, captures: dict[str, object]) -> None:
    module = model.visual_geometry_transformer
    original_forward = module.forward

    def wrapped_forward(*args: Any, **kwargs: Any):
        result = original_forward(*args, **kwargs)
        token_list, patch_start_idx = result
        captures["token_list"] = tuple(token.detach().float().cpu() for token in token_list)
        captures["patch_start_idx"] = int(patch_start_idx)
        return result

    module.forward = wrapped_forward


def _hook_torch_dpt_head(head: Any, caps: dict[str, np.ndarray], head_name: str) -> None:
    norm_counter = {"value": 0}

    def norm_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        index = norm_counter["value"]
        norm_counter["value"] += 1
        caps[f"norm.{index}"] = _to_numpy(output)

    head.norm.register_forward_hook(norm_hook)

    for index, module in enumerate(head.projects):
        module.register_forward_hook(_capture_hook(caps, f"projects.{index}"))
    for index, module in enumerate(head.resize_layers):
        module.register_forward_hook(_capture_hook(caps, f"resize_layers.{index}"))

    scratch = head.scratch
    for name in ("layer1_rn", "layer2_rn", "layer3_rn", "layer4_rn"):
        getattr(scratch, name).register_forward_hook(_capture_hook(caps, f"scratch.{name}", capture_input=True))
    for index in range(1, 5):
        getattr(scratch, f"refinenet{index}").register_forward_hook(_capture_hook(caps, f"scratch.refinenet{index}"))
    scratch.output_conv1.register_forward_hook(_capture_hook(caps, "scratch.output_conv1"))
    scratch.output_conv2[0].register_forward_hook(_capture_hook(caps, "scratch.output_conv2.0"))
    scratch.output_conv2[1].register_forward_hook(_capture_hook(caps, "scratch.output_conv2.1"))
    scratch.output_conv2[2].register_forward_hook(_capture_hook(caps, "scratch.output_conv2.2"))
    scratch.output_conv2.register_forward_hook(_capture_hook(caps, "scratch.output_conv2"))


def _capture_hook(caps: dict[str, np.ndarray], key: str, *, capture_input: bool = False):
    def hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        if capture_input and inputs:
            caps[f"{key}.input"] = _to_numpy(inputs[0])
        caps[key] = _to_numpy(output)

    return hook


def _collect_torch_prediction(head: str, predictions: dict[str, Any], caps: dict[str, np.ndarray]) -> None:
    if head == "depth":
        caps["pred.values"] = _to_numpy(predictions["depth"])
        caps["pred.conf"] = _to_numpy(predictions["depth_conf"])
        if "depth_mask_logits" in predictions:
            caps["pred.depth_mask_logits"] = _to_numpy(predictions["depth_mask_logits"])
        return
    if head == "normal":
        caps["pred.values"] = _to_numpy(predictions["normals"])
        caps["pred.conf"] = _to_numpy(predictions["normals_conf"])
        return
    caps["pred.values"] = _to_numpy(predictions["pts3d"])
    caps["pred.conf"] = _to_numpy(predictions["pts3d_conf"])


def _run_mlx_from_torch_tokens(
    args: argparse.Namespace,
    token_list: tuple[np.ndarray, ...],
    patch_start_idx: int,
    imgs: np.ndarray,
) -> dict[str, np.ndarray]:
    import mlx.core as mx

    mx.set_default_device(mx.cpu)

    from mlx_spatial.checkpoint import load_checkpoint_tensors
    from mlx_spatial.hyworld2_assets import inspect_hyworld2_model_assets
    from mlx_spatial.hyworld2_heads import (
        DPTHeadConfig,
    )
    from mlx_spatial.hyworld2_inference import HYWORLD2_REAL_HEAD_PREFIXES, _strip_tensor_prefix
    from mlx_spatial.hyworld2_worldmirror import HyWorld2BackboneOutput

    inspection = inspect_hyworld2_model_assets(args.model_root, requested_heads=(args.head,))
    if inspection.blocker is not None or inspection.config is None:
        raise RuntimeError(f"HY-World inspection failed: {inspection.blocker}")

    checkpoint_path = Path(args.model_root) / args.subfolder / "model.safetensors"
    head_prefix = HYWORLD2_REAL_HEAD_PREFIXES[args.head]
    raw_tensors = load_checkpoint_tensors(checkpoint_path, prefixes=[head_prefix])
    tensors = _strip_tensor_prefix(raw_tensors, head_prefix)

    images = mx.array(imgs.astype(np.float32))
    batch, frames, _channels, height, width = tuple(int(dim) for dim in images.shape)
    patch_h = height // 14
    patch_w = width // 14
    backbone = HyWorld2BackboneOutput(
        tokens=None,
        intermediate_tokens=tuple(mx.array(level[:, :, patch_start_idx:, :].astype(np.float32)) for level in token_list),
        patch_start_idx=patch_start_idx,
        patch_grid=(patch_h, patch_w),
        frame_token_count=None,
        attention_modes=(),
    )
    config = _dpt_config(args.head, enable_depth_mask=bool(inspection.config.enable_depth_mask))
    caps = _run_mlx_dpt_debug(backbone, images, config, tensors)
    return caps


def _run_mlx_dpt_debug(backbone: Any, images: Any, config: Any, tensors: dict[str, Any]) -> dict[str, np.ndarray]:
    import mlx.core as mx

    from mlx_spatial.hyworld2_heads import (
        _activate_dense_output,
        _apply_dpt_pos_embed,
        _conv2d_nchw,
        _feature_fusion_block,
        _layer_norm,
        _official_dpt_resize,
        _resize_nchw,
    )

    caps: dict[str, np.ndarray] = {}
    batch, frames, _channels, height, width = tuple(int(dim) for dim in images.shape)
    patch_h, patch_w = backbone.patch_grid
    features = []
    for index, tokens in enumerate(backbone.intermediate_tokens[: config.required_feature_levels]):
        patch_tokens = mx.reshape(tokens, (batch * frames, patch_h * patch_w, int(tokens.shape[-1])))
        patch_tokens = _layer_norm(patch_tokens, tensors["norm.weight"], tensors["norm.bias"], eps=1e-5)
        _capture_mlx(caps, f"norm.{index}", patch_tokens)
        feature = mx.reshape(
            mx.transpose(patch_tokens, (0, 2, 1)),
            (batch * frames, int(tokens.shape[-1]), patch_h, patch_w),
        )
        feature = _conv2d_nchw(feature, tensors[f"projects.{index}.weight"], tensors[f"projects.{index}.bias"], padding=0)
        _capture_mlx(caps, f"projects.{index}", feature)
        feature = _apply_dpt_pos_embed(feature, width=width, height=height)
        _capture_mlx(caps, f"pos.{index}", feature)
        feature = _official_dpt_resize(feature, index, tensors)
        _capture_mlx(caps, f"resize_layers.{index}", feature)
        features.append(feature)

    layer1 = _conv2d_nchw(features[0], tensors["scratch.layer1_rn.weight"], None)
    layer2 = _conv2d_nchw(features[1], tensors["scratch.layer2_rn.weight"], None)
    layer3 = _conv2d_nchw(features[2], tensors["scratch.layer3_rn.weight"], None)
    layer4 = _conv2d_nchw(features[3], tensors["scratch.layer4_rn.weight"], None)
    for key, value in (
        ("scratch.layer1_rn.input", features[0]),
        ("scratch.layer2_rn.input", features[1]),
        ("scratch.layer3_rn.input", features[2]),
        ("scratch.layer4_rn.input", features[3]),
        ("scratch.layer1_rn", layer1),
        ("scratch.layer2_rn", layer2),
        ("scratch.layer3_rn", layer3),
        ("scratch.layer4_rn", layer4),
    ):
        _capture_mlx(caps, key, value)

    out = _feature_fusion_block(
        layer4,
        None,
        prefix="scratch.refinenet4",
        tensors=tensors,
        size=tuple(int(dim) for dim in layer3.shape[2:]),
    )
    _capture_mlx(caps, "scratch.refinenet4", out)
    out = _feature_fusion_block(
        out,
        layer3,
        prefix="scratch.refinenet3",
        tensors=tensors,
        size=tuple(int(dim) for dim in layer2.shape[2:]),
    )
    _capture_mlx(caps, "scratch.refinenet3", out)
    out = _feature_fusion_block(
        out,
        layer2,
        prefix="scratch.refinenet2",
        tensors=tensors,
        size=tuple(int(dim) for dim in layer1.shape[2:]),
    )
    _capture_mlx(caps, "scratch.refinenet2", out)
    out = _feature_fusion_block(out, layer1, prefix="scratch.refinenet1", tensors=tensors)
    _capture_mlx(caps, "scratch.refinenet1", out)
    fused = _conv2d_nchw(out, tensors["scratch.output_conv1.weight"], tensors["scratch.output_conv1.bias"])
    _capture_mlx(caps, "scratch.output_conv1", fused)
    fused = _resize_nchw(
        fused,
        (
            int(patch_h * config.patch_size / config.down_ratio),
            int(patch_w * config.patch_size / config.down_ratio),
        ),
    )
    _capture_mlx(caps, "fused.resized", fused)
    fused = _apply_dpt_pos_embed(fused, width=width, height=height)
    _capture_mlx(caps, "fused.pos", fused)
    out0 = _conv2d_nchw(fused.astype(mx.float32), tensors["scratch.output_conv2.0.weight"], tensors["scratch.output_conv2.0.bias"])
    _capture_mlx(caps, "scratch.output_conv2.0", out0)
    out1 = mx.maximum(out0, 0)
    _capture_mlx(caps, "scratch.output_conv2.1", out1)
    out2 = _conv2d_nchw(out1, tensors["scratch.output_conv2.2.weight"], tensors["scratch.output_conv2.2.bias"], padding=0)
    _capture_mlx(caps, "scratch.output_conv2.2", out2)
    _capture_mlx(caps, "scratch.output_conv2", out2)
    raw = mx.reshape(
        mx.transpose(out2, (0, 2, 3, 1)),
        (batch, frames, int(out2.shape[2]), int(out2.shape[3]), int(out2.shape[1])),
    )
    activated = _activate_dense_output(raw, config)
    if activated.blocker is not None or activated.values is None or activated.confidence is None:
        raise RuntimeError(f"MLX DPT activation failed: {activated.blocker}")
    _capture_mlx(caps, "pred.values", activated.values)
    _capture_mlx(caps, "pred.conf", activated.confidence)
    if activated.depth_mask_logits is not None:
        _capture_mlx(caps, "pred.depth_mask_logits", activated.depth_mask_logits)
    return caps


def _capture_mlx(caps: dict[str, np.ndarray], key: str, value: Any) -> None:
    import mlx.core as mx

    mx.eval(value)
    caps[key] = np.array(value, dtype=np.float32)


def _dpt_config(head: str, *, enable_depth_mask: bool):
    from mlx_spatial.hyworld2_heads import DPTHeadConfig

    if head == "depth":
        return DPTHeadConfig(
            head_type="depth",
            attr_channels=1,
            activation="exp+expp1+linear",
            enable_depth_mask=enable_depth_mask,
        )
    if head == "normal":
        return DPTHeadConfig(head_type="normal", attr_channels=3, activation="norm+expp1")
    return DPTHeadConfig(head_type="points", attr_channels=3, activation="inv_log+expp1")


def _torch_head_attr(head: str) -> str:
    if head == "normal":
        return "norm_head"
    if head == "points":
        return "pts_head"
    return "depth_head"


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
        raise RuntimeError("gsplat rasterization is unavailable in local dense-head diagnostics")

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
        return value.detach().float().cpu().numpy().copy()
    return np.asarray(value, dtype=np.float32).copy()


if __name__ == "__main__":
    raise SystemExit(main())
