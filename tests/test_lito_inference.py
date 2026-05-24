from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from safetensors.numpy import save_file
from safetensors.numpy import load_file

from mlx_spatial.lito import LitoInferencePipeline
from mlx_spatial.lito_inference import (
    LITO_RECOMMENDED_CFG_SCALE,
    LITO_RECOMMENDED_NUM_STEPS,
    LITO_STAGE_NAMES,
    LitoBackendUnavailable,
    LitoRealGenerationNotImplemented,
    _crop_and_pad_object,
    _preprocess_image,
)


ROOT = Path(__file__).resolve().parents[1]


def test_full_pipeline_runs_on_sample_input(tmp_path):
    image = _write_synthetic_image(tmp_path / "input.png")
    output = tmp_path / "result.ply"

    result = LitoInferencePipeline(
        tmp_path / "weights",
        memory_profile="safe",
        source_contract_smoke=True,
    ).generate(
        image,
        output_path=output,
        num_steps=LITO_RECOMMENDED_NUM_STEPS,
        cfg_scale=LITO_RECOMMENDED_CFG_SCALE,
        seed=7,
        resolution=32,
        render_size=16,
    )

    assert output.is_file()
    assert output.with_suffix(".safetensors").is_file()
    assert result.output_path == output
    assert result.gaussians["xyz_w"].shape == (64, 3)
    assert result.rendered_image is not None
    assert result.metadata["pipeline"] == "lito-source-contract-smoke"


def test_metrics_dict_has_all_stages(tmp_path):
    image = _write_synthetic_image(tmp_path / "input.png")
    result = LitoInferencePipeline(
        tmp_path / "weights",
        memory_profile="safe",
        source_contract_smoke=True,
    ).generate(
        image,
        output_path=tmp_path / "result.ply",
        resolution=24,
        render_size=12,
    )

    assert tuple(result.metrics) == LITO_STAGE_NAMES
    for stage in LITO_STAGE_NAMES:
        assert result.metrics[stage]["wall_time_s"] >= 0.0
        assert result.metrics[stage]["peak_active_memory_gb"] >= 0.0
        assert result.metrics[stage]["peak_cache_memory_gb"] >= 0.0


def test_tokenizer_execution_is_measured_in_tokenize_stage(tmp_path, monkeypatch):
    image = _write_synthetic_image(tmp_path / "input.png")
    pipeline = LitoInferencePipeline(
        tmp_path / "weights",
        memory_profile="safe",
        source_contract_smoke=True,
    )
    calls: list[str] = []
    original_stage = pipeline._stage
    original_tokenizer = pipeline.tokenizer

    def observing_stage(name, metrics, call):
        calls.append(f"stage:{name}")
        return original_stage(name, metrics, call)

    def observing_tokenizer(*args, **kwargs):
        calls.append("tokenizer")
        return original_tokenizer(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_stage", observing_stage)
    monkeypatch.setattr(pipeline, "tokenizer", observing_tokenizer)

    pipeline.generate(
        image,
        output_path=tmp_path / "result.ply",
        resolution=24,
        render_size=12,
    )

    assert calls.index("stage:tokenize") < calls.index("tokenizer") < calls.index("stage:dit")


def test_lito_inference_imports_without_vendor_runtime_dependencies():
    for relative in ("src/mlx_spatial/lito.py", "src/mlx_spatial/lito_inference.py"):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        assert imported_roots.isdisjoint({"vendors", "torch", "cuda", "xformers", "flash_attn", "gsplat"})


def test_crop_and_pad_object_preserves_optical_axis_for_useful_alpha():
    rgba = np.zeros((20, 20, 4), dtype=np.uint8)
    rgba[8:12, 17:20] = (220, 40, 10, 255)
    image = Image.fromarray(rgba, mode="RGBA")

    cropped = _crop_and_pad_object(image)

    alpha = np.asarray(cropped.getchannel("A"))
    assert cropped.size == (22, 22)
    assert cropped.getchannel("A").getbbox() == (18, 9, 21, 13)
    assert np.all(alpha[0, :] == 0)
    assert np.all(alpha[-1, :] == 0)
    assert np.all(alpha[:, 0] == 0)
    assert np.all(alpha[:, -1] == 0)


def test_preprocess_image_outputs_expected_shape_dtype_and_range(tmp_path):
    image = _write_off_axis_alpha_image(tmp_path / "off-axis.png")

    preprocessed = _preprocess_image(image, resolution=(12, 10))

    assert preprocessed.resolution == (12, 10)
    assert preprocessed.rgba.shape == (10, 12, 4)
    assert preprocessed.straight_rgb.shape == (1, 1, 10, 12, 3)
    assert preprocessed.alpha.shape == (1, 1, 10, 12, 1)
    assert preprocessed.rgba.dtype == np.float32
    assert preprocessed.straight_rgb.dtype == np.float32
    assert preprocessed.alpha.dtype == np.float32
    assert float(preprocessed.rgba.min()) >= 0.0
    assert float(preprocessed.rgba.max()) <= 1.0
    assert np.allclose(preprocessed.straight_rgb[0, 0], preprocessed.rgba[:, :, :3])
    assert np.allclose(preprocessed.alpha[0, 0], preprocessed.rgba[:, :, 3:4])


def test_recommended_constants_have_upstream_source_comments():
    source = (ROOT / "src/mlx_spatial/lito_inference.py").read_text(encoding="utf-8").splitlines()
    constants = [
        "LITO_RECOMMENDED_NUM_STEPS",
        "LITO_RECOMMENDED_SEED_POLICY",
        "LITO_RECOMMENDED_CFG_SCALE",
        "LITO_RECOMMENDED_RESOLUTION",
        "LITO_RECOMMENDED_IMAGE_MEAN",
        "LITO_RECOMMENDED_IMAGE_STD",
        "LITO_RECOMMENDED_SAMPLER",
        "LITO_RECOMMENDED_DECODE_STEPS_FOR_SAMPLE_XYZ",
        "LITO_RECOMMENDED_MLX_COMPUTE_DTYPE",
    ]
    for constant in constants:
        line_index = next(index for index, line in enumerate(source) if line.startswith(f"{constant} ="))
        assert "Source:" in source[line_index - 1], constant
        assert "upstream Apple LiTo" in source[line_index - 1], constant


def test_safetensors_export_contains_gaussian_fields(tmp_path):
    image = _write_synthetic_image(tmp_path / "input.png")
    output = tmp_path / "result.safetensors"

    LitoInferencePipeline(
        tmp_path / "weights",
        memory_profile="safe",
        source_contract_smoke=True,
    ).generate(
        image,
        output_path=output,
        output_format="safetensors",
        resolution=24,
        render_size=12,
    )

    tensors = load_file(output)
    assert set(tensors) >= {"xyz_w", "scaling", "quaternion", "opacity", "rgb_sh", "lf"}
    assert tensors["xyz_w"].shape == (64, 3)


def test_generate_requires_converted_weights_by_default(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint-backed LiTo generation requires converted assets"):
        LitoInferencePipeline(tmp_path / "missing", memory_profile="safe")


def test_generate_rejects_placeholder_weight_files_by_default(tmp_path):
    root = tmp_path / "weights"
    (root / "tokenizer").mkdir(parents=True)
    (root / "image_to_3d").mkdir(parents=True)
    save_file({"tokenizer.weight": np.ones((1,), dtype=np.float32)}, root / "tokenizer" / "lito_new.safetensors")
    save_file(
        {"dit.weight": np.ones((1,), dtype=np.float32)},
        root / "image_to_3d" / "lito_dit_rgba.safetensors",
    )

    with pytest.raises(ValueError, match="missing required real tensor keys"):
        LitoInferencePipeline(root, memory_profile="safe")


@pytest.mark.heavy
@pytest.mark.skipif(not (ROOT / "weights/lito-research-mlx/tokenizer/lito_new.safetensors").is_file(), reason="LiTo weights absent")
def test_generate_with_real_weight_headers_does_not_fall_back_to_smoke_on_backend_failure(tmp_path):
    image = _write_synthetic_image(tmp_path / "input.png")
    output = tmp_path / "result.ply"
    pipeline = LitoInferencePipeline(ROOT / "weights/lito-research-mlx", memory_profile="safe")

    with pytest.raises((LitoBackendUnavailable, LitoRealGenerationNotImplemented)):
        pipeline.generate(
            image,
            output_path=output,
            num_steps=1,
            resolution=24,
            render_size=12,
        )
    assert not output.exists()


def _write_synthetic_image(path: Path) -> Path:
    rgba = np.zeros((40, 40, 4), dtype=np.uint8)
    rgba[..., 0] = np.arange(40, dtype=np.uint8)[None, :] * 4
    rgba[..., 1] = np.arange(40, dtype=np.uint8)[:, None] * 4
    rgba[..., 2] = 180
    rgba[8:32, 8:32, 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(path)
    return path


def _write_off_axis_alpha_image(path: Path) -> Path:
    rgba = np.zeros((20, 20, 4), dtype=np.uint8)
    rgba[..., 2] = 30
    rgba[8:12, 17:20] = (220, 40, 10, 255)
    Image.fromarray(rgba, mode="RGBA").save(path)
    return path
