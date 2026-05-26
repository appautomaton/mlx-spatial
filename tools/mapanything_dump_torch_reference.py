#!/usr/bin/env python3
"""Dump vendored MapAnything PyTorch tensors for dev-only MLX parity checks.

This script is intentionally outside the shipped CLI. It may import PyTorch and
the vendored MapAnything repo only when MAPANYTHING_TORCH_REF=1 is set.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

MAPANYTHING_TORCH_PARITY_ENV = "MAPANYTHING_TORCH_REF"


DEFAULT_CAPTURE_KEYS = (
    "input.img.0",
    "input.img.1",
    "encoder.patch_embed",
    "encoder.tokens",
    "encoder.block0",
    "encoder.features.0",
    "encoder.features.1",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", help="root containing config.json and model.safetensors")
    parser.add_argument("input", help="image folder or image path")
    parser.add_argument("--vendor-root", default="vendors/map-anything")
    parser.add_argument("--output", required=True, help="output .npz reference bundle")
    parser.add_argument("--device", default=None, help="torch device override, e.g. cpu or mps")
    parser.add_argument("--capture-key", action="append", dest="capture_keys")
    parser.add_argument(
        "--prefix-only",
        action="store_true",
        help="capture only patch embedding and encoder block 0 without running the full encoder",
    )
    args = parser.parse_args(argv)

    if os.environ.get(MAPANYTHING_TORCH_PARITY_ENV) != "1":
        print(
            f"MapAnything PyTorch reference dumping is dev-only. Set {MAPANYTHING_TORCH_PARITY_ENV}=1 "
            "to run it. The shipped MLX runtime remains Torch/CUDA-free.",
            file=sys.stderr,
        )
        return 2

    parity_module = _load_parity_module()
    write_mapanything_parity_bundle = parity_module.write_mapanything_parity_bundle

    vendor_root = Path(args.vendor_root).resolve()
    model_root = Path(args.model_root).resolve()
    if not vendor_root.is_dir():
        print(f"vendor root not found: {vendor_root}", file=sys.stderr)
        return 2
    if not model_root.is_dir():
        print(f"model root not found: {model_root}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(vendor_root))

    torch = _import_torch()
    from mapanything.utils.image import load_images
    from safetensors.torch import load_file as load_safetensors

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    config_path = model_root / "config.json"
    checkpoint_path = model_root / "model.safetensors"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["encoder_config"] = dict(config["encoder_config"])
    config["encoder_config"]["uses_torch_hub"] = False
    config["encoder_config"]["torch_hub_pretrained"] = False
    if args.prefix_only:
        from mapanything.models.external.dinov2.hub.backbones import dinov2_vitg14

        model = dinov2_vitg14(pretrained=False)
        state_dict = _load_prefix_state_dict(checkpoint_path, strip_encoder_model_prefix=True)
        model.load_state_dict(state_dict, strict=False)
    else:
        from mapanything.models.mapanything import MapAnything

        model = MapAnything(**config)
        state_dict = load_safetensors(str(checkpoint_path))
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    captures: dict[str, np.ndarray] = {}
    views = load_images(str(args.input))
    for index, view in enumerate(views):
        captures[f"input.img.{index}"] = _to_numpy(view["img"])
        for key, value in list(view.items()):
            if hasattr(value, "to"):
                view[key] = value.to(device)

    with torch.no_grad():
        if args.prefix_only:
            _capture_encoder_prefix_only(torch, model, views, captures)
        else:
            _hook_encoder_prefix(model, captures)
            features, _registers = model._encode_n_views(views)
            for index, feature in enumerate(features):
                captures[f"encoder.features.{index}"] = _to_numpy(feature)

    default_capture_keys = (
        tuple(key for key in DEFAULT_CAPTURE_KEYS if not key.startswith("encoder.features."))
        if args.prefix_only
        else DEFAULT_CAPTURE_KEYS
    )
    capture_keys = tuple(args.capture_keys or default_capture_keys)
    tensors = {key: captures[key] for key in capture_keys if key in captures}
    missing = [key for key in capture_keys if key not in captures]
    metadata = {
        "model_root": str(model_root),
        "vendor_root": str(vendor_root),
        "input": str(Path(args.input).resolve()),
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_hub_disabled": bool(config["encoder_config"]["uses_torch_hub"] is False),
        "torch_hub_pretrained": bool(config["encoder_config"]["torch_hub_pretrained"]),
        "prefix_only": bool(args.prefix_only),
        "reference_model": "vendored_dinov2_vitg14" if args.prefix_only else "vendored_mapanything",
        "capture_keys": sorted(tensors),
        "missing_capture_keys": missing,
    }
    output_path = write_mapanything_parity_bundle(args.output, tensors, metadata=metadata)
    print(json.dumps({"output": str(output_path), **metadata}, indent=2, sort_keys=True))
    return 0 if not missing else 1


def _import_torch():
    try:
        import torch
    except ImportError as error:
        raise SystemExit(f"PyTorch is required only for this dev reference script: {error}") from error
    return torch


def _load_parity_module():
    parity_module_path = (
        Path(__file__).resolve().parents[1] / "src" / "mlx_spatial" / "mapanything_parity.py"
    )
    parity_spec = importlib.util.spec_from_file_location(
        "_mapanything_parity_dev",
        parity_module_path,
    )
    if parity_spec is None or parity_spec.loader is None:
        raise RuntimeError(f"could not load parity helper from {parity_module_path}")
    parity_module = importlib.util.module_from_spec(parity_spec)
    sys.modules[parity_spec.name] = parity_module
    parity_spec.loader.exec_module(parity_module)
    return parity_module


def _load_prefix_state_dict(
    checkpoint_path: Path,
    *,
    strip_encoder_model_prefix: bool = False,
) -> dict[str, object]:
    from safetensors import safe_open

    keys = (
        "encoder.model.cls_token",
        "encoder.model.pos_embed",
        "encoder.model.patch_embed.proj.bias",
        "encoder.model.patch_embed.proj.weight",
        "encoder.model.blocks.0.attn.proj.bias",
        "encoder.model.blocks.0.attn.proj.weight",
        "encoder.model.blocks.0.attn.qkv.bias",
        "encoder.model.blocks.0.attn.qkv.weight",
        "encoder.model.blocks.0.ls1.gamma",
        "encoder.model.blocks.0.ls2.gamma",
        "encoder.model.blocks.0.mlp.w12.bias",
        "encoder.model.blocks.0.mlp.w12.weight",
        "encoder.model.blocks.0.mlp.w3.bias",
        "encoder.model.blocks.0.mlp.w3.weight",
        "encoder.model.blocks.0.norm1.bias",
        "encoder.model.blocks.0.norm1.weight",
        "encoder.model.blocks.0.norm2.bias",
        "encoder.model.blocks.0.norm2.weight",
    )
    with safe_open(checkpoint_path, framework="pt", device="cpu") as tensors:
        loaded = {key: tensors.get_tensor(key) for key in keys}
    if not strip_encoder_model_prefix:
        return loaded
    return {key.removeprefix("encoder.model."): value for key, value in loaded.items()}


def _capture_encoder_prefix_only(
    torch: Any,
    encoder_model: Any,
    views: list[dict[str, object]],
    captures: dict[str, np.ndarray],
) -> None:
    images = torch.cat([view["img"] for view in views], dim=0)
    captures["encoder.patch_embed"] = _to_numpy(encoder_model.patch_embed(images))
    tokens = encoder_model.prepare_tokens_with_masks(images)
    captures["encoder.tokens"] = _to_numpy(tokens)
    captures["encoder.block0"] = _to_numpy(encoder_model.blocks[0](tokens))


def _hook_encoder_prefix(model: Any, captures: dict[str, np.ndarray]) -> None:
    import numpy as np

    encoder_model = getattr(getattr(model, "encoder", None), "model", None)
    patch_embed = getattr(encoder_model, "patch_embed", None)
    blocks = getattr(encoder_model, "blocks", None)
    if patch_embed is None or blocks is None or len(blocks) == 0:
        raise RuntimeError("could not locate MapAnything encoder patch_embed and block 0")

    patch_embed.register_forward_hook(
        lambda _module, _inputs, output: captures.__setitem__(
            "encoder.patch_embed",
            _to_numpy(output),
        )
    )
    blocks[0].register_forward_hook(
        lambda _module, _inputs, output: captures.__setitem__(
            "encoder.block0",
            _to_numpy(output),
        )
    )


def _to_numpy(value: object) -> np.ndarray:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


if __name__ == "__main__":
    raise SystemExit(main())
