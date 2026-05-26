#!/usr/bin/env python3
"""Dump vendored MapAnything scene tensors for dev-only MLX parity checks.

This script is intentionally outside the shipped CLI. It imports PyTorch and
the vendored MapAnything repo only when MAPANYTHING_TORCH_REF=1 is set.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

MAPANYTHING_TORCH_PARITY_ENV = "MAPANYTHING_TORCH_REF"

FINAL_OUTPUT_KEYS = (
    "pts3d",
    "pts3d_cam",
    "ray_directions",
    "depth_along_ray",
    "depth_z",
    "cam_trans",
    "cam_quats",
    "camera_poses",
    "metric_scaling_factor",
    "intrinsics",
    "conf",
    "non_ambiguous_mask",
    "non_ambiguous_mask_logits",
    "mask",
    "img_no_norm",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", help="root containing config.json and model.safetensors")
    parser.add_argument("input", help="image folder or image path")
    parser.add_argument("--vendor-root", default="vendors/map-anything")
    parser.add_argument("--output", required=True, help="output .npz reference bundle")
    parser.add_argument("--device", default=None, help="torch device override, e.g. cpu or mps")
    parser.add_argument("--minibatch-size", type=int, default=1)
    parser.add_argument(
        "--memory-efficient-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the dense head in memory-efficient mini-batches",
    )
    parser.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use vendored autocast inference",
    )
    parser.add_argument("--amp-dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument(
        "--apply-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="match MapAnything.infer default mask application",
    )
    parser.add_argument(
        "--mask-edges",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="match MapAnything.infer default edge masking",
    )
    args = parser.parse_args(argv)

    if os.environ.get(MAPANYTHING_TORCH_PARITY_ENV) != "1":
        print(
            f"MapAnything PyTorch scene reference dumping is dev-only. Set "
            f"{MAPANYTHING_TORCH_PARITY_ENV}=1 to run it. The shipped MLX runtime remains "
            "Torch/CUDA-free.",
            file=sys.stderr,
        )
        return 2

    parity_module = _load_parity_module()
    write_mapanything_parity_bundle = parity_module.write_mapanything_parity_bundle

    vendor_root = Path(args.vendor_root).resolve()
    model_root = Path(args.model_root).resolve()
    input_path = Path(args.input).resolve()
    if not vendor_root.is_dir():
        print(f"vendor root not found: {vendor_root}", file=sys.stderr)
        return 2
    if not model_root.is_dir():
        print(f"model root not found: {model_root}", file=sys.stderr)
        return 2
    if not input_path.exists():
        print(f"input path not found: {input_path}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(vendor_root))

    torch = _import_torch()
    _patch_dinov2_torch_hub_to_local_vendor(torch)
    from mapanything.models.mapanything import MapAnything
    from mapanything.utils.geometry import depthmap_to_world_frame
    from mapanything.utils.image import load_images
    from safetensors.torch import load_file as load_safetensors

    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    torch.manual_seed(0)

    config_path = model_root / "config.json"
    checkpoint_path = model_root / "model.safetensors"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["encoder_config"] = dict(config["encoder_config"])
    config["encoder_config"]["uses_torch_hub"] = False
    config["encoder_config"]["torch_hub_pretrained"] = False

    model = MapAnything(**config)
    state_dict = load_safetensors(str(checkpoint_path))
    load_result = model.load_state_dict(state_dict, strict=False)
    missing_alias_keys, missing_non_alias_keys = _classify_missing_keys(model, load_result.missing_keys)
    if missing_non_alias_keys:
        raise RuntimeError(
            "local MapAnything checkpoint did not load required model weights: "
            f"{missing_non_alias_keys[:8]}"
        )
    model = model.to(device).eval()

    captures: dict[str, Any] = {}
    _install_capture_hooks(model, captures)

    views = load_images(str(input_path))
    for index, view in enumerate(views):
        captures[f"input.img.{index}"] = _to_numpy(view["img"])
        captures[f"input.true_shape.{index}"] = _to_numpy(view.get("true_shape", []))

    with torch.no_grad():
        outputs = model.infer(
            views,
            memory_efficient_inference=args.memory_efficient_inference,
            minibatch_size=args.minibatch_size,
            use_amp=args.use_amp,
            amp_dtype=args.amp_dtype,
            apply_mask=args.apply_mask,
            mask_edges=args.mask_edges,
        )

    _capture_final_outputs(outputs, captures)
    _capture_scene_bundle(outputs, captures, depthmap_to_world_frame)

    metadata = {
        "model_root": str(model_root),
        "vendor_root": str(vendor_root),
        "input": str(input_path),
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "torch_hub_disabled": bool(config["encoder_config"]["uses_torch_hub"] is False),
        "torch_hub_patched_to_local_vendor": True,
        "torch_hub_pretrained": bool(config["encoder_config"]["torch_hub_pretrained"]),
        "reference_model": "vendored_mapanything_scene",
        "runtime_depends_on_torch": False,
        "memory_efficient_inference": bool(args.memory_efficient_inference),
        "minibatch_size": args.minibatch_size,
        "use_amp": bool(args.use_amp),
        "amp_dtype": args.amp_dtype,
        "apply_mask": bool(args.apply_mask),
        "mask_edges": bool(args.mask_edges),
        "view_count": len(views),
        "capture_keys": sorted(captures),
        "load_missing_keys": sorted(missing_non_alias_keys),
        "load_missing_alias_keys": sorted(missing_alias_keys),
        "load_unexpected_keys": sorted(load_result.unexpected_keys),
    }
    output_path = write_mapanything_parity_bundle(args.output, captures, metadata=metadata)
    print(json.dumps({"output": str(output_path), **metadata}, indent=2, sort_keys=True))
    return 0


def _install_capture_hooks(model: Any, captures: dict[str, Any]) -> None:
    _hook_encoder_prefix(model, captures)
    _wrap_encode_n_views(model, captures)
    _wrap_fusion_boundary(model, captures)
    _hook_info_sharing(model, captures)
    _wrap_downstream_head(model, captures)


def _classify_missing_keys(model: Any, missing_keys: list[str]) -> tuple[list[str], list[str]]:
    alias_prefixes = ()
    dense_head = getattr(model, "dense_head", None)
    if dense_head is not None and hasattr(dense_head, "__getitem__"):
        try:
            if dense_head[0] is getattr(model, "dpt_feature_head", None):
                alias_prefixes = (
                    *alias_prefixes,
                    "dpt_feature_head.",
                    "dense_head.0.scratch.layer1_rn.",
                    "dense_head.0.scratch.layer2_rn.",
                    "dense_head.0.scratch.layer3_rn.",
                    "dense_head.0.scratch.layer4_rn.",
                    "dense_head.0.scratch.layer_rn.",
                )
            if dense_head[1] is getattr(model, "dpt_regressor_head", None):
                alias_prefixes = (*alias_prefixes, "dpt_regressor_head.")
        except (IndexError, TypeError):
            pass
    alias_keys = [key for key in missing_keys if key.startswith(alias_prefixes)]
    non_alias_keys = [key for key in missing_keys if not key.startswith(alias_prefixes)]
    return alias_keys, non_alias_keys


def _patch_dinov2_torch_hub_to_local_vendor(torch: Any) -> None:
    original_load = torch.hub.load

    def local_dinov2_load(repo_or_dir: str, model: str, *args: Any, **kwargs: Any) -> Any:
        if repo_or_dir == "facebookresearch/dinov2":
            from mapanything.models.external.dinov2.hub import backbones

            if not hasattr(backbones, model):
                raise RuntimeError(f"vendored DINOv2 backbone not found: {model}")
            kwargs.pop("force_reload", None)
            factory = getattr(backbones, model)
            return factory(*args, **kwargs)
        return original_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = local_dinov2_load


def _hook_encoder_prefix(model: Any, captures: dict[str, Any]) -> None:
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

    prepare_tokens = getattr(encoder_model, "prepare_tokens_with_masks", None)
    if prepare_tokens is not None:

        def wrapped_prepare_tokens(*args: Any, **kwargs: Any) -> Any:
            output = prepare_tokens(*args, **kwargs)
            captures["encoder.tokens"] = _to_numpy(output)
            return output

        encoder_model.prepare_tokens_with_masks = wrapped_prepare_tokens


def _wrap_encode_n_views(model: Any, captures: dict[str, Any]) -> None:
    original = model._encode_n_views

    def wrapped_encode_n_views(views: list[dict[str, Any]]) -> Any:
        features, registers = original(views)
        _capture_tensor_sequence("encoder.features", features, captures)
        if registers is not None:
            _capture_tensor_sequence("encoder.registers", registers, captures)
        return features, registers

    model._encode_n_views = wrapped_encode_n_views


def _wrap_fusion_boundary(model: Any, captures: dict[str, Any]) -> None:
    original = model._encode_and_fuse_optional_geometric_inputs

    def wrapped_fusion(views: list[dict[str, Any]], features: Any) -> Any:
        output = original(views, features)
        _capture_tensor_sequence("fusion.features", output, captures)
        return output

    model._encode_and_fuse_optional_geometric_inputs = wrapped_fusion


def _hook_info_sharing(model: Any, captures: dict[str, Any]) -> None:
    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        final_output = output
        intermediate_outputs = None
        if isinstance(output, tuple) and len(output) == 2:
            final_output, intermediate_outputs = output
        _capture_multi_view_output("info.final", final_output, captures)
        if intermediate_outputs is not None:
            for index, intermediate in enumerate(intermediate_outputs):
                _capture_multi_view_output(f"info.intermediate.{index}", intermediate, captures)

    model.info_sharing.register_forward_hook(hook)


def _wrap_downstream_head(model: Any, captures: dict[str, Any]) -> None:
    original = model.downstream_head

    def wrapped_downstream_head(*args: Any, **kwargs: Any) -> Any:
        dense, pose, scale = original(*args, **kwargs)
        _capture_prediction_output("head.dense", dense, captures)
        if pose is not None:
            _capture_prediction_output("head.pose", pose, captures)
        captures["head.scale.value"] = _to_numpy(scale)
        return dense, pose, scale

    model.downstream_head = wrapped_downstream_head


def _capture_multi_view_output(prefix: str, output: Any, captures: dict[str, Any]) -> None:
    _capture_tensor_sequence(f"{prefix}.features", getattr(output, "features", None), captures)
    additional = getattr(output, "additional_token_features", None)
    if additional is not None:
        captures[f"{prefix}.additional_token_features"] = _to_numpy(additional)
    per_view = getattr(output, "additional_token_features_per_view", None)
    if per_view is not None:
        _capture_tensor_sequence(f"{prefix}.additional_token_features_per_view", per_view, captures)


def _capture_prediction_output(prefix: str, output: Any, captures: dict[str, Any]) -> None:
    for name in ("value", "confidence", "mask", "logits", "decoded_channels"):
        if hasattr(output, name):
            value = getattr(output, name)
            if value is not None:
                captures[f"{prefix}.{name}"] = _to_numpy(value)


def _capture_final_outputs(outputs: list[Mapping[str, Any]], captures: dict[str, Any]) -> None:
    for view_index, pred in enumerate(outputs):
        for key in FINAL_OUTPUT_KEYS:
            if key in pred:
                captures[f"final.{key}.{view_index}"] = _to_numpy(pred[key])


def _capture_scene_bundle(
    outputs: list[Mapping[str, Any]],
    captures: dict[str, Any],
    depthmap_to_world_frame: Any,
) -> None:
    import numpy as np

    world_points = []
    depths = []
    intrinsics = []
    camera_poses = []
    confidences = []
    images = []
    final_masks = []
    for pred in outputs:
        depthmap = pred["depth_z"][0].squeeze(-1)
        intrinsic = pred["intrinsics"][0]
        camera_pose = pred["camera_poses"][0]
        points, valid_mask = depthmap_to_world_frame(depthmap, intrinsic, camera_pose)
        if "mask" in pred:
            mask = pred["mask"][0].squeeze(-1).detach().cpu().numpy().astype(bool)
        else:
            mask = np.ones_like(depthmap.detach().cpu().numpy(), dtype=bool)
        mask = mask & valid_mask.detach().cpu().numpy().astype(bool)
        world_points.append(_to_numpy(points))
        depths.append(_to_numpy(depthmap))
        intrinsics.append(_to_numpy(intrinsic))
        camera_poses.append(_to_numpy(camera_pose))
        confidences.append(_to_numpy(pred["conf"][0]))
        images.append(_to_numpy(pred["img_no_norm"][0]))
        final_masks.append(mask)

    captures["scene.world_points"] = np.stack(world_points)
    captures["scene.depth"] = np.stack(depths)
    captures["scene.intrinsics"] = np.stack(intrinsics)
    captures["scene.camera_poses"] = np.stack(camera_poses)
    captures["scene.conf"] = np.stack(confidences)
    captures["scene.images"] = np.stack(images)
    captures["scene.final_masks"] = np.stack(final_masks).astype(np.float32)


def _capture_tensor_sequence(prefix: str, values: Any, captures: dict[str, Any]) -> None:
    if values is None:
        return
    for index, value in enumerate(tuple(values)):
        captures[f"{prefix}.{index}"] = _to_numpy(value)


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise SystemExit(f"PyTorch is required only for this dev reference script: {error}") from error
    return torch


def _load_parity_module() -> Any:
    parity_module_path = (
        Path(__file__).resolve().parents[1] / "src" / "mlx_spatial" / "mapanything_parity.py"
    )
    parity_spec = importlib.util.spec_from_file_location(
        "_mapanything_parity_scene_dev",
        parity_module_path,
    )
    if parity_spec is None or parity_spec.loader is None:
        raise RuntimeError(f"could not load parity helper from {parity_module_path}")
    parity_module = importlib.util.module_from_spec(parity_spec)
    sys.modules[parity_spec.name] = parity_module
    parity_spec.loader.exec_module(parity_module)
    return parity_module


def _to_numpy(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value
    if value is None:
        return np.array([], dtype=np.float32)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


if __name__ == "__main__":
    raise SystemExit(main())
