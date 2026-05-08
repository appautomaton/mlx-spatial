#!/usr/bin/env python3
"""Dump official HY-World PyTorch tensors for dev-only MLX parity checks.

This script is intentionally outside the shipped CLI. It may import PyTorch and
the vendored official repo only when HYWORLD2_TORCH_REF=1 is set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

from mlx_spatial.hyworld2_parity import (
    HYWORLD2_TORCH_PARITY_ENV,
    require_hyworld2_torch_parity_enabled,
    write_hyworld2_parity_bundle,
)


DEFAULT_CAPTURE_KEYS = (
    "input.imgs",
    "vgt.patch_start_idx",
    "predictions.camera_params",
    "predictions.camera_poses",
    "predictions.camera_intrs",
    "predictions.depth",
    "predictions.depth_conf",
    "predictions.pts3d",
    "predictions.pts3d_conf",
    "predictions.normals",
    "predictions.normals_conf",
    "predictions.gs_depth",
    "predictions.gs_depth_conf",
    "predictions.gs_raw_params",
    "predictions.splats.means",
    "predictions.splats.scales",
    "predictions.splats.quats",
    "predictions.splats.opacities",
    "predictions.splats.sh",
    "predictions.splats.weights",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", help="root containing HY-WorldMirror-2.0")
    parser.add_argument("input", help="official input image/video directory")
    parser.add_argument("--vendor-root", default="vendors/HY-World-2.0")
    parser.add_argument("--subfolder", default="HY-WorldMirror-2.0")
    parser.add_argument("--output", required=True, help="output .npz reference bundle")
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--heads", default="camera,depth,normal,points,gs")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--video-min-frames", type=int, default=1)
    parser.add_argument("--video-max-frames", type=int, default=32)
    parser.add_argument("--no-shims", action="store_true", help="do not install macOS import shims")
    parser.add_argument("--capture-key", action="append", dest="capture_keys")
    args = parser.parse_args(argv)

    try:
        require_hyworld2_torch_parity_enabled()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2

    if not args.no_shims:
        _install_macos_reference_shims()

    vendor_root = Path(args.vendor_root).resolve()
    if not vendor_root.is_dir():
        print(f"vendor root not found: {vendor_root}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(vendor_root))

    torch = _import_torch()
    pipeline_module = _import_official_pipeline()
    pipeline = pipeline_module.WorldMirrorPipeline.from_pretrained(
        pretrained_model_name_or_path=args.model_root,
        subfolder=args.subfolder,
        enable_bf16=False,
        disable_heads=_disabled_heads(args.heads),
    )
    inner = pipeline.model.module if hasattr(pipeline.model, "module") else pipeline.model

    captures: dict[str, np.ndarray] = {}
    _wrap_visual_geometry_transformer(inner, captures)
    _hook_gaussian_renderer(inner, captures)

    img_paths, _ = pipeline_module.prepare_input(
        args.input,
        target_size=args.target_size,
        fps=args.fps,
        min_frames=args.video_min_frames,
        max_frames=args.video_max_frames,
    )
    effective_size = pipeline_module.compute_adaptive_target_size(img_paths, args.target_size)
    predictions, imgs, infer_time = pipeline._run_inference(img_paths, effective_size, None, None)
    captures["input.imgs"] = _to_numpy(imgs)
    _collect_predictions(predictions, captures)

    capture_keys = tuple(args.capture_keys or DEFAULT_CAPTURE_KEYS)
    tensors = {key: captures[key] for key in capture_keys if key in captures}
    missing = [key for key in capture_keys if key not in captures]
    metadata = {
        "model_root": str(Path(args.model_root).resolve()),
        "vendor_root": str(vendor_root),
        "subfolder": args.subfolder,
        "input": str(Path(args.input).resolve()),
        "target_size": args.target_size,
        "effective_target_size": effective_size,
        "heads": _enabled_heads(args.heads),
        "missing_capture_keys": missing,
        "inference_seconds": infer_time,
        "torch_version": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "capture_keys": sorted(tensors),
    }
    output_path = write_hyworld2_parity_bundle(args.output, tensors, metadata=metadata)
    print(json.dumps({"output": str(output_path), **metadata}, indent=2, sort_keys=True))
    return 0 if not missing else 1


def _import_torch():
    try:
        import torch
    except ImportError as error:
        raise SystemExit(f"PyTorch is required only for this dev reference script: {error}") from error
    return torch


def _import_official_pipeline():
    try:
        from hyworld2.worldrecon import pipeline as pipeline_module
    except ImportError as error:
        raise SystemExit(f"could not import vendored HY-World pipeline: {error}") from error
    return pipeline_module


def _install_macos_reference_shims() -> None:
    """Install minimal import shims for CUDA-only optional official deps."""

    torch = _import_torch()

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
        raise RuntimeError(
            "gsplat rasterization is CUDA-only and intentionally unavailable in local parity dumps"
        )

    class DefaultStrategy:
        pass

    gsplat_rendering.rasterization = rasterization
    gsplat_strategy.DefaultStrategy = DefaultStrategy
    gsplat_pkg.rendering = gsplat_rendering
    gsplat_pkg.strategy = gsplat_strategy
    sys.modules.setdefault("gsplat", gsplat_pkg)
    sys.modules.setdefault("gsplat.rendering", gsplat_rendering)
    sys.modules.setdefault("gsplat.strategy", gsplat_strategy)


def _wrap_visual_geometry_transformer(model: Any, captures: dict[str, np.ndarray]) -> None:
    if not hasattr(model, "visual_geometry_transformer"):
        return
    module = model.visual_geometry_transformer
    original_forward = module.forward

    def wrapped_forward(*args: Any, **kwargs: Any):
        result = original_forward(*args, **kwargs)
        token_list, patch_start_idx = result
        captures["vgt.patch_start_idx"] = np.asarray([int(patch_start_idx)], dtype=np.int64)
        for index, tokens in enumerate(token_list):
            captures[f"vgt.token_list.{index}"] = _to_numpy(tokens)
        return result

    module.forward = wrapped_forward


def _hook_gaussian_renderer(model: Any, captures: dict[str, np.ndarray]) -> None:
    renderer = getattr(model, "gs_renderer", None)
    gs_head = getattr(renderer, "gs_head", None)
    if gs_head is None:
        return

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any):
        captures["predictions.gs_raw_params"] = _to_numpy(output)

    gs_head.register_forward_hook(hook)


def _collect_predictions(predictions: dict[str, Any], captures: dict[str, np.ndarray]) -> None:
    for key, value in predictions.items():
        if key == "splats" and isinstance(value, dict):
            for splat_key, splat_value in value.items():
                if _is_tensor_like(splat_value):
                    captures[f"predictions.splats.{splat_key}"] = _to_numpy(splat_value)
            continue
        if _is_tensor_like(value):
            captures[f"predictions.{key}"] = _to_numpy(value)


def _disabled_heads(heads: str) -> list[str]:
    enabled = set(_enabled_heads(heads))
    return [head for head in ("camera", "depth", "normal", "points", "gs") if head not in enabled]


def _enabled_heads(heads: str) -> list[str]:
    return [head.strip().lower() for head in heads.split(",") if head.strip()]


def _is_tensor_like(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "cpu")


def _to_numpy(value: Any) -> np.ndarray:
    if not _is_tensor_like(value):
        return np.asarray(value)
    return value.detach().float().cpu().numpy()


if __name__ == "__main__":
    raise SystemExit(main())
