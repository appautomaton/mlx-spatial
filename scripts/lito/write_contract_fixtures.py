"""Write deterministic LiTo source-contract fixtures.

The fixtures generated here are not vendor numerical captures. They are small,
local contract fixtures that encode tensor names, shapes, dtypes, seeds, and
simple deterministic transforms for MLX bring-up tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from safetensors.numpy import save_file


SEED_BASE = 20260523
REQUIRED_FILES = (
    "manifest.json",
    "tokenizer_input_0.safetensors",
    "tokenizer_output_0.safetensors",
    "tokenizer_input_1.safetensors",
    "tokenizer_output_1.safetensors",
    "tokenizer_input_2.safetensors",
    "tokenizer_output_2.safetensors",
    "cond_input_0.safetensors",
    "cond_output_0.safetensors",
    "cond_input_1.safetensors",
    "cond_output_1.safetensors",
    "cond_input_2.safetensors",
    "cond_output_2.safetensors",
    "dit_input_0.safetensors",
    "dit_step_0_0.safetensors",
    "dit_step_mid_0.safetensors",
    "dit_step_final_0.safetensors",
    "render_input_0.safetensors",
    "render_output_0.safetensors",
    "render_output_0.png",
)


def _rng(seed_offset: int) -> np.random.Generator:
    return np.random.default_rng(SEED_BASE + seed_offset)


def _save(path: Path, tensors: dict[str, np.ndarray]) -> None:
    save_file({name: np.ascontiguousarray(value) for name, value in tensors.items()}, path)


def _tokenizer_fixture(root: Path, index: int) -> dict[str, Any]:
    rng = _rng(index)
    points = 2048
    target_tokens = 8192
    latent_dim = 32

    theta = np.linspace(0.0, np.pi * 2.0, points, dtype=np.float32)
    radius = np.float32(0.5 + 0.1 * index)
    xyz = np.stack(
        [
            radius * np.cos(theta),
            radius * np.sin(theta),
            np.linspace(-0.75, 0.75, points, dtype=np.float32),
        ],
        axis=-1,
    )[None, :, :]
    xyz += rng.normal(0.0, 0.002, size=xyz.shape).astype(np.float32)

    rgb = np.stack(
        [
            np.linspace(0.0, 1.0, points, dtype=np.float32),
            np.mod(theta / (np.pi * 2.0) + 0.17 * index, 1.0),
            np.full(points, 0.25 + 0.15 * index, dtype=np.float32),
        ],
        axis=-1,
    )[None, :, :]

    rays = np.concatenate(
        [
            np.broadcast_to(np.array([0.0, 0.0, -2.0], dtype=np.float32), (1, points, 3)),
            xyz / np.maximum(np.linalg.norm(xyz, axis=-1, keepdims=True), 1e-6),
        ],
        axis=-1,
    ).astype(np.float32)

    token_axis = np.linspace(-1.0, 1.0, target_tokens, dtype=np.float32)[:, None]
    channel_axis = np.linspace(0.1, 1.0, latent_dim, dtype=np.float32)[None, :]
    stats = np.concatenate([xyz.mean(axis=1), rgb.mean(axis=1)], axis=-1).astype(np.float32)
    stat_scale = np.float32(stats.mean() + 0.05 * index)
    latent = np.sin(token_axis * channel_axis * np.pi + stat_scale).astype(np.float32)
    latent = latent[None, :, :].astype(np.float16)

    input_name = f"tokenizer_input_{index}.safetensors"
    output_name = f"tokenizer_output_{index}.safetensors"
    _save(root / input_name, {"xyz_w": xyz, "rgb": rgb, "ray_origin_direction_w": rays})
    _save(root / output_name, {"latent_tokens": latent})
    return {
        "input": input_name,
        "output": output_name,
        "seed": SEED_BASE + index,
        "shape": {
            "xyz_w": list(xyz.shape),
            "rgb": list(rgb.shape),
            "ray_origin_direction_w": list(rays.shape),
            "latent_tokens": list(latent.shape),
        },
    }


def _condition_fixture(root: Path, index: int) -> dict[str, Any]:
    height = width = 32
    token_count = 17
    hidden = 64
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float32),
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    straight_rgb = np.stack(
        [
            xx,
            yy,
            np.mod(xx * 0.5 + yy * 0.5 + 0.11 * index, 1.0),
        ],
        axis=-1,
    )[None, None, :, :, :]
    alpha = np.clip(1.0 - ((xx - 0.5) ** 2 + (yy - 0.5) ** 2) * (2.0 + index), 0.0, 1.0)
    alpha = alpha[None, None, :, :, None].astype(np.float32)

    base = np.linspace(-1.0, 1.0, token_count * hidden, dtype=np.float32).reshape(1, token_count, hidden)
    rgb_mean = straight_rgb.mean(dtype=np.float32)
    alpha_mean = alpha.mean(dtype=np.float32)
    cond = np.tanh(base + rgb_mean + alpha_mean * 0.25 + 0.03 * index).astype(np.float16)

    input_name = f"cond_input_{index}.safetensors"
    output_name = f"cond_output_{index}.safetensors"
    _save(root / input_name, {"straight_rgb": straight_rgb.astype(np.float32), "alpha": alpha})
    _save(root / output_name, {"cond_tokens": cond})
    return {
        "input": input_name,
        "output": output_name,
        "seed": SEED_BASE + 100 + index,
        "shape": {
            "straight_rgb": list(straight_rgb.shape),
            "alpha": list(alpha.shape),
            "cond_tokens": list(cond.shape),
        },
    }


def _dit_fixture(root: Path) -> dict[str, Any]:
    rng = _rng(200)
    latent = rng.normal(0.0, 0.2, size=(1, 8192, 32)).astype(np.float16)
    cond = rng.normal(0.0, 0.1, size=(1, 17, 64)).astype(np.float16)
    cond_scale = np.float16(cond.mean(dtype=np.float32))

    step0 = (latent + np.float16(0.01) * cond_scale).astype(np.float16)
    step_mid = (step0 * np.float16(0.75) + np.float16(0.05)).astype(np.float16)
    step_final = (step_mid * np.float16(0.5) - np.float16(0.02)).astype(np.float16)

    _save(
        root / "dit_input_0.safetensors",
        {
            "latent": latent,
            "cond_tokens": cond,
            "num_steps": np.array([20], dtype=np.int64),
            "seed": np.array([42], dtype=np.int64),
        },
    )
    _save(root / "dit_step_0_0.safetensors", {"latent": step0, "t": np.array([0.0], dtype=np.float32)})
    _save(root / "dit_step_mid_0.safetensors", {"latent": step_mid, "t": np.array([0.5], dtype=np.float32)})
    _save(root / "dit_step_final_0.safetensors", {"latent": step_final, "t": np.array([1.0], dtype=np.float32)})
    return {
        "input": "dit_input_0.safetensors",
        "steps": ["dit_step_0_0.safetensors", "dit_step_mid_0.safetensors", "dit_step_final_0.safetensors"],
        "seed": 42,
        "shape": {"latent": list(latent.shape), "cond_tokens": list(cond.shape)},
    }


def _render_fixture(root: Path) -> dict[str, Any]:
    rng = _rng(300)
    count = 128
    xyz = rng.uniform(-0.8, 0.8, size=(count, 3)).astype(np.float32)
    scaling = rng.uniform(0.01, 0.08, size=(count, 3)).astype(np.float32)
    quat = rng.normal(0.0, 1.0, size=(count, 4)).astype(np.float32)
    quat /= np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-6)
    opacity = rng.uniform(0.2, 0.9, size=(count, 1)).astype(np.float32)
    rgb_sh = rng.uniform(0.0, 1.0, size=(count, 1, 3)).astype(np.float32)
    lf = rng.uniform(-1.0, 1.0, size=(count, 4)).astype(np.float32)
    intrinsic = np.array([[[64.0, 0.0, 32.0], [0.0, 64.0, 32.0], [0.0, 0.0, 1.0]]], dtype=np.float32)
    h_c2w = np.eye(4, dtype=np.float32)[None, :, :]

    height = width = 64
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    image = np.stack(
        [
            xx / np.float32(width - 1),
            yy / np.float32(height - 1),
            np.full((height, width), opacity.mean(dtype=np.float32), dtype=np.float32),
        ],
        axis=-1,
    )
    alpha = np.clip((image[..., :1] * 0.35 + image[..., 1:2] * 0.35 + 0.3), 0.0, 1.0).astype(np.float32)
    rgba = np.concatenate([image, alpha], axis=-1)
    png = (rgba * 255.0).round().clip(0, 255).astype(np.uint8)

    _save(
        root / "render_input_0.safetensors",
        {
            "xyz_w": xyz,
            "scaling": scaling,
            "quaternion": quat,
            "opacity": opacity,
            "rgb_sh": rgb_sh,
            "lf": lf,
            "intrinsic": intrinsic,
            "H_c2w": h_c2w,
            "height_px": np.array([height], dtype=np.int64),
            "width_px": np.array([width], dtype=np.int64),
        },
    )
    _save(root / "render_output_0.safetensors", {"image": image[None, None, :, :, :], "alpha": alpha[None, None, :, :, :]})
    Image.fromarray(png, mode="RGBA").save(root / "render_output_0.png")
    return {
        "input": "render_input_0.safetensors",
        "output": ["render_output_0.safetensors", "render_output_0.png"],
        "seed": SEED_BASE + 300,
        "shape": {"xyz_w": list(xyz.shape), "image": [1, 1, height, width, 3], "alpha": [1, 1, height, width, 1]},
    }


def _manifest(tokenizer: list[dict[str, Any]], condition: list[dict[str, Any]], dit: dict[str, Any], render: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fixture_kind": "source_contract_local",
        "generated_by": "scripts/lito/write_contract_fixtures.py",
        "no_cuda_contract": {
            "cuda_allowed": False,
            "vendor_runtime_imports": False,
            "torch_parity": "optional_cpu_mps_only",
        },
        "tokenizer": {
            "backend": "source_contract_local",
            "upstream_entry": "lito.trainers.lito_trainer.LightTokenizationTrainer.get_latents",
            "upstream_sources": [
                "vendors/ml-lito/src/lito/trainers/lito_trainer.py",
                "vendors/ml-lito/src/lito/models/spoint_encoder.py",
            ],
            "fixture_role": "shape_dtype_range_contract",
            "files": [name for item in tokenizer for name in (item["input"], item["output"])],
            "dtype": "float16 output, float32 input",
            "shape": {"latent_tokens": [1, 8192, 32]},
            "cases": tokenizer,
            "license": "synthetic local fixture; no Apple sample redistribution",
        },
        "condition": {
            "backend": "source_contract_local",
            "upstream_entry": "lito.trainers.lito_dit_trainer.LiToDiTTrainer.get_image_conditioning",
            "upstream_sources": [
                "vendors/ml-lito/src/lito/trainers/lito_dit_trainer.py",
                "vendors/ml-lito/src/lito/models/dino.py",
            ],
            "fixture_role": "shape_dtype_token_order_contract",
            "files": [name for item in condition for name in (item["input"], item["output"])],
            "dtype": "float16 output, float32 input",
            "shape": {"cond_tokens": [1, 17, 64]},
            "cases": condition,
            "license": "synthetic local fixture; no Apple sample redistribution",
        },
        "dit": {
            "backend": "source_contract_local",
            "upstream_entry": "lito.odelibs.ode_solvers.odeint",
            "upstream_sources": [
                "vendors/ml-lito/src/lito/mlx/models/dit.py",
                "vendors/ml-lito/src/lito/odelibs/ode_solvers.py",
                "vendors/ml-lito/src/lito/trainers/lito_dit_trainer.py",
            ],
            "fixture_role": "microtrajectory_contract",
            "files": [dit["input"], *dit["steps"]],
            "dtype": "float16",
            "shape": dit["shape"],
            "seed": dit["seed"],
            "license": "synthetic local fixture; no Apple sample redistribution",
        },
        "render": {
            "backend": "source_contract_local",
            "upstream_entry": "lito.trainers.lito_trainer.LightTokenizationTrainer.render_gaussians",
            "upstream_sources": [
                "vendors/ml-lito/src/lito/trainers/lito_trainer.py",
                "vendors/ml-lito/libraries/plibs/src/plibs/gs_utils.py",
            ],
            "fixture_role": "gaussian_camera_image_contract",
            "files": [render["input"], *render["output"]],
            "dtype": "float32",
            "shape": render["shape"],
            "seed": render["seed"],
            "license": "synthetic local fixture; no Apple sample redistribution",
        },
    }


def write_contract_fixtures(root: Path, overwrite: bool = False) -> None:
    root = root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    existing = [path for path in (root / name for name in REQUIRED_FILES) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing[:5])
        more = "" if len(existing) <= 5 else f", ... ({len(existing)} total)"
        raise FileExistsError(f"fixtures already exist; pass --overwrite to replace: {names}{more}")
    if overwrite:
        for name in REQUIRED_FILES:
            path = root / name
            if path.exists():
                path.unlink()

    tokenizer = [_tokenizer_fixture(root, index) for index in range(3)]
    condition = [_condition_fixture(root, index) for index in range(3)]
    dit = _dit_fixture(root)
    render = _render_fixture(root)
    manifest = _manifest(tokenizer, condition, dit, render)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic LiTo source-contract fixtures.")
    parser.add_argument("root", nargs="?", default="tests/fixtures/lito", help="fixture root to write")
    parser.add_argument("--overwrite", action="store_true", help="replace existing generated fixture files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    write_contract_fixtures(Path(args.root), overwrite=args.overwrite)
    print(f"OK: wrote LiTo source-contract fixtures to {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
