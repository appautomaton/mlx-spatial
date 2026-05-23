from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.numpy import load_file

from mlx_spatial.lito_render import LitoRenderer


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "lito"


def _render_fixture():
    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    inputs = load_file(FIXTURE_ROOT / "render_input_0.safetensors")
    expected = load_file(FIXTURE_ROOT / "render_output_0.safetensors")
    return manifest, inputs, expected


def test_render_matches_source_contract_safetensors_and_png():
    manifest, inputs, expected = _render_fixture()
    assert manifest["render"]["backend"] == "source_contract_local"
    assert manifest["no_cuda_contract"]["cuda_allowed"] is False

    renderer = LitoRenderer(source_contract_backend=manifest["render"]["backend"])
    result = renderer.render(inputs)

    assert result.image.shape == tuple(expected["image"].shape)
    assert result.alpha.shape == tuple(expected["alpha"].shape)
    np.testing.assert_allclose(np.asarray(result.image), expected["image"], rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(result.alpha), expected["alpha"], rtol=0.0, atol=1e-5)

    expected_png = np.asarray(Image.open(FIXTURE_ROOT / "render_output_0.png"))
    actual_png = np.rint(np.asarray(result.rgba)[0, 0] * 255.0).clip(0, 255).astype(np.uint8)
    np.testing.assert_array_equal(actual_png, expected_png)
    assert result.metadata["source_contract_backend"] == "source_contract_local"


def test_render_contract_input_exercises_adapter_rasterizer_path():
    _, inputs, expected = _render_fixture()

    result = LitoRenderer(use_metal=False, allow_cpu_fallback=True).render(inputs)
    image = np.asarray(result.image)
    alpha = np.asarray(result.alpha)

    assert image.shape == tuple(expected["image"].shape)
    assert alpha.shape == tuple(expected["alpha"].shape)
    assert np.isfinite(image).all()
    assert np.isfinite(alpha).all()
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0
    assert float(alpha.min()) >= 0.0
    assert float(alpha.max()) <= 1.0
    assert float(alpha.max()) > 0.0
    assert float(alpha.mean()) > 1e-3
    assert result.metadata["source_contract_backend"] is None
    assert result.metadata["backend"] in {"metal", "cpu", "cpu-no-metal"}
    assert result.metadata["adapter"] == "lito_render"
    assert result.metadata["visible_gaussian_count"] > 0
    assert result.metrics["wall_time_s"] > 0.0
    assert result.metrics["peak_active_memory_gb"] >= 0.0
    assert np.mean(np.abs(image - expected["image"])) > 1e-3


def test_render_shape_and_alpha_range_with_cpu_fallback_when_metal_unavailable():
    inputs = {
        "xyz_w": np.array([[0.0, 0.0, 2.0]], dtype=np.float16),
        "scaling": np.array([[0.18, 0.18, 0.18]], dtype=np.float16),
        "quaternion": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float16),
        "opacity": np.array([[0.8]], dtype=np.float16),
        "rgb_sh": np.array([[[1.0, 0.25, 0.0]]], dtype=np.float16),
        "lf": np.array([[0.5, 0.0, -0.5, 0.25]], dtype=np.float16),
        "intrinsic": np.array(
            [[[8.0, 0.0, 4.5], [0.0, 8.0, 4.5], [0.0, 0.0, 1.0]]],
            dtype=np.float32,
        ),
        "H_c2w": np.eye(4, dtype=np.float32)[None, :, :],
        "height_px": np.array([9], dtype=np.int64),
        "width_px": np.array([9], dtype=np.int64),
    }

    result = LitoRenderer(use_metal=True, allow_cpu_fallback=True).render(inputs)
    image = np.asarray(result.image)
    alpha = np.asarray(result.alpha)

    assert image.shape == (1, 1, 9, 9, 3)
    assert alpha.shape == (1, 1, 9, 9, 1)
    assert float(alpha.min()) >= 0.0
    assert float(alpha.max()) <= 1.0
    assert float(alpha.max()) > 0.0
    assert result.metadata["backend"] in {"metal", "cpu", "cpu-no-metal"}


def test_lf_conditioning_modulates_rasterizer_inputs_and_accepts_float16():
    base = {
        "xyz_w": np.array([[0.0, 0.0, 2.0]], dtype=np.float16),
        "scaling": np.array([[0.18, 0.18, 0.18]], dtype=np.float16),
        "quaternion": np.array([[0.0, 0.0, 0.0, 2.0]], dtype=np.float16),
        "opacity": np.array([[0.5]], dtype=np.float16),
        "rgb_sh": np.array([[[0.5, 0.5, 0.5]]], dtype=np.float16),
    }
    renderer = LitoRenderer()
    unconditioned = renderer.prepare_gaussians(base)
    conditioned = renderer.prepare_gaussians(
        base,
        lf_condition=np.array([[1.0, -1.0, 0.0, 1.0]], dtype=np.float16),
    )

    assert unconditioned.xyz_w.dtype == np.float32
    assert conditioned.rgb_sh.dtype == np.float32
    assert conditioned.opacity.dtype == np.float32
    assert np.allclose(conditioned.quaternion, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
    assert not np.allclose(conditioned.rgb_sh, unconditioned.rgb_sh)
    assert float(conditioned.opacity[0, 0]) > float(unconditioned.opacity[0, 0])


def test_lito_render_does_not_modify_gs_rasterize():
    unstaged = subprocess.run(
        ["git", "diff", "--quiet", "--", "src/mlx_spatial/gs_rasterize.py"],
        cwd=ROOT,
        check=False,
    )
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", "src/mlx_spatial/gs_rasterize.py"],
        cwd=ROOT,
        check=False,
    )
    assert unstaged.returncode == 0
    assert staged.returncode == 0
