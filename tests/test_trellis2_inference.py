import json
from pathlib import Path
from types import SimpleNamespace

from safetensors.mlx import save_file
import mlx.core as mx
import numpy as np
from PIL import Image

import mlx_spatial
import mlx_spatial.trellis2_inference as trellis2_inference
from mlx_spatial.trellis2_forward import Trellis2StageOutput
from mlx_spatial.model_assets import TRELLIS2_ASSETS
from mlx_spatial.trellis2_inference import (
    TRELLIS2_INFERENCE_STAGES,
    Trellis2InferenceBlocker,
    Trellis2InferencePipeline,
    Trellis2TexturedGenerationResult,
    _conditioning_resolutions,
    _sample_texture_slat_model,
    _quantize_cascade_coordinates,
    _undo_slat_normalization,
)


def _write_trellis2_root(root):
    for asset_path in TRELLIS2_ASSETS.required_paths:
        path = root / asset_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if asset_path.endswith(".safetensors"):
            save_file(
                {
                    "blocks.0.0.norm.weight": mx.array([7.0, 8.0], dtype=mx.float32),
                    "blocks.0.norm2.weight": mx.array([9.0, 10.0], dtype=mx.float32),
                },
                path,
            )
        else:
            path.write_text("{}")


def _write_rmbg_root(root, *, deform_conv=False):
    root.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "bb.weight": mx.array([1.0], dtype=mx.float32),
            "decoder.weight": mx.array([2.0], dtype=mx.float32),
            "squeeze_module.weight": mx.array([3.0], dtype=mx.float32),
        },
        root / "model.safetensors",
    )
    (root / "config.json").write_text("{}")
    (root / "BiRefNet_config.py").write_text("config = {}\n")
    architecture = "from torchvision.ops import deform_conv2d\n" if deform_conv else "class BiRefNet: pass\n"
    (root / "birefnet.py").write_text(architecture)


def _write_textured_trellis2_root(root):
    _write_trellis2_root(root)
    root.mkdir(parents=True, exist_ok=True)
    pipeline = {
        "name": "Trellis2ImageTo3DPipeline",
        "args": {
            "models": {
                "sparse_structure_decoder": "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
                "sparse_structure_flow_model": "ckpts/ss_flow_img_dit_1_3B_64_bf16",
                "shape_slat_decoder": "ckpts/shape_dec_next_dc_f16c32_fp16",
                "shape_slat_flow_model_512": "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16",
                "shape_slat_flow_model_1024": "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16",
                "tex_slat_decoder": "ckpts/tex_dec_next_dc_f16c32_fp16",
                "tex_slat_flow_model_512": "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16",
                "tex_slat_flow_model_1024": "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
            },
            "sparse_structure_sampler": _sampler_config(guidance_rescale=0.7, rescale_t=5.0, interval=(0.6, 1.0)),
            "shape_slat_sampler": _sampler_config(guidance_rescale=0.5, rescale_t=3.0, interval=(0.6, 1.0)),
            "tex_slat_sampler": _sampler_config(
                guidance_strength=1.0,
                guidance_rescale=0.0,
                rescale_t=3.0,
                interval=(0.6, 0.9),
            ),
            "shape_slat_normalization": _normalization_config(),
            "tex_slat_normalization": _normalization_config(offset=1.0),
            "image_cond_model": {
                "name": "DinoV3FeatureExtractor",
                "args": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m", "image_size": 512},
            },
            "default_pipeline_type": "1024_cascade",
        },
    }
    (root / "pipeline.json").write_text(json.dumps(pipeline))
    (root / "ckpts/ss_flow_img_dit_1_3B_64_bf16.json").write_text(
        json.dumps(
            {
                "name": "SparseStructureFlowModel",
                "args": {
                    "resolution": 2,
                    "in_channels": 2,
                    "out_channels": 2,
                    "model_channels": 6,
                    "cond_channels": 1024,
                    "num_blocks": 1,
                    "num_heads": 1,
                    "mlp_ratio": 2.0,
                    "pe_mode": "rope",
                    "share_mod": True,
                    "initialization": "scaled",
                    "qk_rms_norm": True,
                    "qk_rms_norm_cross": True,
                    "dtype": "bfloat16",
                },
            }
        )
    )
    _write_slat_flow_config(root / "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json", resolution=32)
    _write_slat_flow_config(root / "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json", resolution=64)
    save_file({"blocks.0.norm2.weight": mx.array([1.0], dtype=mx.float32)}, root / "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors")
    _write_structured_decoder_config(root / "ckpts/tex_dec_next_dc_f16c32_fp16.json")


def _sampler_config(*, guidance_strength=7.5, guidance_rescale, rescale_t, interval):
    return {
        "name": "FlowEulerGuidanceIntervalSampler",
        "args": {"sigma_min": 1e-5},
        "params": {
            "steps": 12,
            "guidance_strength": guidance_strength,
            "guidance_rescale": guidance_rescale,
            "guidance_interval": list(interval),
            "rescale_t": rescale_t,
        },
    }


def _normalization_config(*, offset=0.0):
    return {
        "mean": [offset + index for index in range(32)],
        "std": [offset + index + 0.5 for index in range(32)],
    }


def _write_slat_flow_config(path, *, resolution):
    path.write_text(
        json.dumps(
            {
                "name": "SLatFlowModel",
                "args": {
                    "resolution": resolution,
                    "in_channels": 64,
                    "out_channels": 32,
                    "model_channels": 6,
                    "cond_channels": 1024,
                    "num_blocks": 1,
                    "num_heads": 1,
                    "mlp_ratio": 2.0,
                    "pe_mode": "rope",
                    "share_mod": True,
                    "initialization": "scaled",
                    "qk_rms_norm": True,
                    "qk_rms_norm_cross": True,
                    "dtype": "bfloat16",
                },
            }
        )
    )


def _write_structured_decoder_config(path):
    path.write_text(
        json.dumps(
            {
                "name": "SparseUnetVaeDecoder",
                "args": {
                    "model_channels": [16, 8],
                    "latent_channels": 32,
                    "num_blocks": [1, 0],
                    "block_type": ["SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d"],
                    "up_block_type": ["SparseResBlockC2S3d"],
                    "block_args": [{}, {}],
                    "use_fp16": False,
                    "out_channels": 6,
                    "pred_subdiv": False,
                },
            }
        )
    )


def _write_rgba_foreground(path):
    image = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    pixels = image.load()
    for y in range(2, 6):
        for x in range(2, 6):
            pixels[x, y] = (10, 20, 30, 255)
    image.save(path)


def test_pipeline_exposes_deterministic_stage_order():
    assert TRELLIS2_INFERENCE_STAGES == (
        "input-image",
        "asset-config-validation",
        "checkpoint-probe-readiness",
        "image-preprocessing-background",
        "image-conditioning",
        "sparse-structure-sampling",
        "shape-slat-sampling",
        "texture-slat-sampling",
        "shape-decoder",
        "texture-decoder",
        "decoded-artifact-write",
        "mesh-export",
    )
    assert Trellis2InferencePipeline().stages == TRELLIS2_INFERENCE_STAGES


def test_blocker_structure_has_required_fields():
    blocker = Trellis2InferenceBlocker(
        stage="image-conditioning",
        operation="feature extraction",
        reference="reference.py:1",
        reason="not implemented",
        next_slice="implement image conditioning",
    )

    assert blocker.stage == "image-conditioning"
    assert blocker.operation == "feature extraction"
    assert blocker.reference == "reference.py:1"
    assert blocker.reason == "not implemented"
    assert blocker.next_slice == "implement image conditioning"


def test_dry_run_validates_fake_assets_and_reports_available_stages(tmp_path):
    _write_trellis2_root(tmp_path)

    report = Trellis2InferencePipeline(tmp_path).dry_run()

    assert report.ready
    assert [stage.stage for stage in report.stages[:3]] == [
        "asset-config-validation",
        "checkpoint-probe-readiness",
        "image-preprocessing-background",
    ]
    assert report.stages[0].status == "ready"
    assert report.stages[1].status == "ready"
    assert report.stages[2].status == "available"
    assert report.stages[2].blocker is None
    assert report.blocker is None


def test_dry_run_can_load_fake_weight_probes(tmp_path):
    _write_trellis2_root(tmp_path)

    report = Trellis2InferencePipeline(tmp_path).dry_run(load_probes=True)

    assert report.stages[1].detail == "groups=5 tensors=5 loaded=True"


def test_dry_run_reports_missing_assets_deterministically(tmp_path):
    tmp_path.mkdir(exist_ok=True)

    report = Trellis2InferencePipeline(tmp_path).dry_run()

    assert not report.ready
    assert report.stages == (
        report.stages[0],
    )
    assert report.stages[0].status == "blocked"
    assert report.blocker is not None
    assert report.blocker.stage == "asset-config-validation"
    assert "pipeline.json" in report.blocker.reason


def test_attempt_reports_missing_image_before_compute(tmp_path):
    _write_trellis2_root(tmp_path)

    report = Trellis2InferencePipeline(tmp_path).attempt(tmp_path / "missing.png")

    assert not report.completed
    assert report.completed_stages == ()
    assert report.blocker is not None
    assert report.blocker.stage == "input-image"


def test_attempt_validates_image_and_stops_at_first_compute_blocker(tmp_path):
    _write_trellis2_root(tmp_path)
    image = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image)

    report = Trellis2InferencePipeline(tmp_path).attempt(image)

    assert not report.completed
    assert report.completed_stages == (
        "input-image",
        "asset-config-validation",
        "checkpoint-probe-readiness",
    )
    assert report.blocker is not None
    assert report.blocker.stage == "image-preprocessing-background"
    assert report.blocker.operation == "MLX RMBG background removal"
    assert report.blocker.next_slice == "validate local RMBG assets and wire MLX BiRefNet for RGB preprocessing"


def test_attempt_propagates_configured_rmbg_port_blocker(tmp_path):
    _write_trellis2_root(tmp_path / "trellis")
    _write_rmbg_root(tmp_path / "rmbg", deform_conv=True)
    image = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image)

    report = Trellis2InferencePipeline(tmp_path / "trellis", rmbg_root=tmp_path / "rmbg").attempt(image)

    assert not report.completed
    assert report.completed_stages == (
        "input-image",
        "asset-config-validation",
        "checkpoint-probe-readiness",
    )
    assert report.blocker is not None
    assert report.blocker.stage == "image-preprocessing-background"
    assert report.blocker.operation == "MLX BiRefNet checkpoint key mapping"
    assert "required forward keys" in report.blocker.reason


def test_attempt_runs_alpha_preprocessing_and_stops_at_image_conditioning(tmp_path):
    _write_trellis2_root(tmp_path)
    image = tmp_path / "alpha.png"
    rgba = Image.new("RGBA", (8, 8), (10, 20, 30, 0))
    pixels = rgba.load()
    for y in range(2, 6):
        for x in range(2, 6):
            pixels[x, y] = (200, 100, 50, 255)
    rgba.save(image)

    report = Trellis2InferencePipeline(tmp_path).attempt(image)

    assert not report.completed
    assert report.completed_stages == (
        "input-image",
        "asset-config-validation",
        "checkpoint-probe-readiness",
        "image-preprocessing-background",
    )
    assert report.blocker is not None
    assert report.blocker.stage == "image-conditioning"


def test_attempt_reports_invalid_image_at_preprocessing_boundary(tmp_path):
    _write_trellis2_root(tmp_path)
    image = tmp_path / "sample.png"
    image.write_bytes(b"not-real-image-but-local")

    report = Trellis2InferencePipeline(tmp_path).attempt(image)

    assert not report.completed
    assert report.completed_stages == (
        "input-image",
        "asset-config-validation",
        "checkpoint-probe-readiness",
    )
    assert report.blocker is not None
    assert report.blocker.stage == "image-preprocessing-background"
    assert report.blocker.operation == "image decode"


def test_shape_pipeline_conditioning_resolutions_match_upstream_routes():
    assert _conditioning_resolutions("512") == (512,)
    assert _conditioning_resolutions("1024") == (512, 1024)
    assert _conditioning_resolutions("1024_cascade") == (512, 1024)
    assert _conditioning_resolutions("1536_cascade") == (512, 1024)


def test_quantize_cascade_coordinates_reduces_resolution_until_token_cap():
    coords = mx.array(
        [
            [0, 0, 0, 0],
            [0, 8, 0, 0],
            [0, 16, 0, 0],
            [0, 24, 0, 0],
        ],
        dtype=mx.int32,
    )

    quantized, resolution = _quantize_cascade_coordinates(
        coords,
        lr_resolution=512,
        target_resolution=1536,
        max_num_tokens=3,
    )

    assert resolution == 1024
    assert quantized.shape[1] == 4


def test_undo_slat_normalization_reverses_denormalized_shape_features():
    normalization = SimpleNamespace(mean=(1.0, 2.0), std=(2.0, 4.0))
    normalized = mx.array([[0.5, -1.0]], dtype=mx.float32)
    denormalized = normalized * mx.array(normalization.std, dtype=mx.float32)[None, :] + mx.array(
        normalization.mean,
        dtype=mx.float32,
    )[None, :]

    restored = _undo_slat_normalization(denormalized, normalization, name="shape_slat")

    assert mx.allclose(restored, normalized)


def test_sample_texture_slat_model_passes_normalized_shape_features_to_probe(monkeypatch, tmp_path):
    shape_coordinates = mx.array([[0, 0, 0, 0]], dtype=mx.int32)
    shape_features = mx.array([[3.0, 10.0]], dtype=mx.float32)
    conditioning = mx.ones((1, 2, 1024), dtype=mx.float32)
    captured = {}

    def fake_read_slat_flow_config(root, config_path):
        captured["config_path"] = config_path
        return SimpleNamespace(in_channels=4)

    def fake_probe(checkpoint_path, slat_config, coordinates, features, **kwargs):
        captured["checkpoint_path"] = checkpoint_path
        captured["coordinates"] = coordinates
        captured["features"] = features
        captured["conditioning"] = kwargs["conditioning"]
        return SimpleNamespace(
            coordinate_shape=(1, 4),
            shape_feature_shape=(1, 2),
            noise_feature_shape=(1, 2),
            concat_feature_shape=(1, 4),
            sampled_feature_shape=(1, 2),
            sampled_features=mx.array([[2.0, 4.0]], dtype=mx.float32),
            blocker_detail="texture probe ok",
        )

    monkeypatch.setattr(trellis2_inference, "read_slat_flow_config", fake_read_slat_flow_config)
    monkeypatch.setattr(trellis2_inference, "probe_texture_slat_forward_boundary", fake_probe)
    config = SimpleNamespace(
        texture_slat_512_config_path="tex512.json",
        texture_slat_512_checkpoint_path="tex512.safetensors",
        shape_slat_normalization=SimpleNamespace(mean=(1.0, 2.0), std=(2.0, 4.0)),
        texture_slat_normalization=SimpleNamespace(mean=(10.0, 20.0), std=(3.0, 5.0)),
        texture_slat_sampler=SimpleNamespace(
            steps=2,
            rescale_t=3.0,
            guidance_strength=1.0,
            guidance_rescale=0.0,
            guidance_interval=(0.6, 0.9),
            sigma_min=1e-5,
        ),
    )

    coordinates, texture_features, detail = _sample_texture_slat_model(
        tmp_path,
        config,
        "tex_slat_flow_model_512",
        shape_coordinates,
        shape_features,
        conditioning,
    )

    assert coordinates is shape_coordinates
    assert captured["config_path"] == "tex512.json"
    assert str(captured["checkpoint_path"]).endswith("tex512.safetensors")
    assert mx.allclose(captured["features"], mx.array([[1.0, 2.0]], dtype=mx.float32))
    assert captured["conditioning"] is conditioning
    assert mx.allclose(texture_features, mx.array([[16.0, 40.0]], dtype=mx.float32))
    assert "concat feature shape (1, 4)" in detail


def test_generate_shape_rejects_glb_and_points_to_textured_command(tmp_path):
    _write_trellis2_root(tmp_path)
    image = tmp_path / "missing.png"

    result = Trellis2InferencePipeline(tmp_path).generate_shape_obj(
        image,
        output_path=tmp_path / "outputs/trellis2/demo.glb",
    )

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.operation == "TRELLIS.2 textured GLB export"
    assert "use generate-textured" in result.trace.blocker.reason


def test_generate_textured_glb_rejects_obj_format(tmp_path):
    _write_textured_trellis2_root(tmp_path / "trellis")

    result = Trellis2InferencePipeline(tmp_path / "trellis").generate_textured_glb(
        tmp_path / "missing.png",
        output_path=tmp_path / "outputs/trellis2/demo.obj",
    )

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.operation == "textured GLB output format validation"
    assert "only writes .glb" in result.trace.blocker.reason


def test_generate_textured_glb_allows_output_outside_repository_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_textured_trellis2_root(tmp_path / "trellis")

    result = Trellis2InferencePipeline(tmp_path / "trellis").generate_textured_glb(
        tmp_path / "missing.png",
        output_path=tmp_path / "outside/demo.glb",
    )

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.operation != "TRELLIS.2 textured GLB export path validation"
    assert result.trace.blocker.stage == "input-image"


def test_generate_textured_glb_reports_missing_image_after_texture_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_textured_trellis2_root(tmp_path / "trellis")

    result = Trellis2InferencePipeline(tmp_path / "trellis").generate_textured_glb(
        tmp_path / "missing.png",
        output_path="outputs/trellis2/demo.glb",
        dino_root=tmp_path / "dino",
        slat_steps=2,
        seed=7,
        max_num_tokens=32,
        decoder_token_limit=64,
        texture_size=128,
    )

    assert result.trace.completed_stages == ("asset-config-validation", "checkpoint-probe-readiness")
    assert result.trace.outputs[0].name == "texture_route"
    assert "seed=7" in result.trace.outputs[0].detail
    assert "dino_root=" in result.trace.outputs[0].detail
    assert "slat_steps=2" in result.trace.outputs[0].detail
    assert "max_num_tokens=32" in result.trace.outputs[0].detail
    assert "decoder_token_limit=64" in result.trace.outputs[0].detail
    assert "texture_size=128" in result.trace.outputs[0].detail
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "input-image"


def test_generate_textured_glb_validates_shared_scalar_guards(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_textured_trellis2_root(tmp_path / "trellis")

    cases = (
        ({"slat_steps": 0}, "texture-slat-sampling", "texture SLat sampler step validation", "slat_steps"),
        ({"max_num_tokens": 0}, "texture-slat-sampling", "texture cascade token cap validation", "max-num-tokens"),
        ({"decoder_token_limit": 0}, "texture-decoder", "texture decoder token limit validation", "decoder-token-limit"),
        ({"texture_size": 0}, "mesh-export", "texture size validation", "texture-size"),
        ({"glb_target_faces": 0}, "mesh-export", "GLB simplification target validation", "glb-target-faces"),
    )
    for kwargs, stage, operation, reason in cases:
        result = Trellis2InferencePipeline(tmp_path / "trellis").generate_textured_glb(
            tmp_path / "missing.png",
            output_path="outputs/trellis2/demo.glb",
            **kwargs,
        )

        assert result.trace.completed_stages == ()
        assert result.trace.blocker is not None
        assert result.trace.blocker.stage == stage
        assert result.trace.blocker.operation == operation
        assert reason in result.trace.blocker.reason


def test_generate_textured_glb_blocks_when_spatialkit_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_textured_trellis2_root(tmp_path / "trellis")
    monkeypatch.setattr(
        trellis2_inference,
        "load_spatialkit_exporter",
        lambda: (None, "SpatialKit native extension is unavailable"),
    )

    result = Trellis2InferencePipeline(tmp_path / "trellis").generate_textured_glb(
        tmp_path / "missing.png",
        output_path="outputs/trellis2/demo.glb",
    )

    assert result.trace.completed_stages == ()
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "mesh-export"
    assert result.trace.blocker.operation == "load integrated mlx_spatial.spatialkit exporter"
    assert "native extension is unavailable" in result.trace.blocker.reason


def test_generate_textured_glb_reports_missing_texture_route_before_image_compute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_textured_trellis2_root(tmp_path / "trellis")
    (tmp_path / "trellis/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors").unlink()
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image)

    result = Trellis2InferencePipeline(tmp_path / "trellis").generate_textured_glb(
        image,
        output_path="outputs/trellis2/demo.glb",
    )

    assert result.trace.completed_stages == ("asset-config-validation",)
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "texture-slat-sampling"
    assert result.trace.blocker.operation == "texture SLat checkpoint validation"
    assert "1024" in result.trace.blocker.reason


def test_generate_textured_glb_reports_missing_texture_decoder_before_image_compute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_textured_trellis2_root(tmp_path / "trellis")
    (tmp_path / "trellis/ckpts/tex_dec_next_dc_f16c32_fp16.json").unlink()
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image)

    result = Trellis2InferencePipeline(tmp_path / "trellis").generate_textured_glb(
        image,
        output_path="outputs/trellis2/demo.glb",
    )

    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "texture-decoder"
    assert result.trace.blocker.operation == "texture decoder config validation"


def _patch_textured_glb_exact_slat_fixtures(
    monkeypatch,
    *,
    expected_pipeline_type,
    expected_texture_model,
):
    sparse_coordinates = mx.array([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=mx.int32)
    shape_coordinates = mx.array([[0, 2, 2, 2], [0, 3, 3, 3]], dtype=mx.int32)
    shape_features = mx.array([[3.0, 4.0], [5.0, 6.0]], dtype=mx.float32)
    texture_features = mx.array([[7.0, 8.0], [9.0, 10.0]], dtype=mx.float32)
    shape_decoder_coordinates = mx.array([[0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2]], dtype=mx.int32)
    shape_fields = mx.ones((3, 7), dtype=mx.float32)
    shape_subdivisions = (mx.ones((2, 8), dtype=mx.bool_),)
    texture_coordinates = mx.array([[0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2]], dtype=mx.int32)
    texture_attributes = mx.full((3, 6), 0.75, dtype=mx.float32)
    expected_final_resolution = 1536 if expected_pipeline_type == "1536_cascade" else int(
        expected_pipeline_type.split("_", maxsplit=1)[0]
    )
    calls = {}

    def fake_conditioning(root, config, *, dino_root=None, image_tensor=None, conditioning=None):
        resolution = int(config.conditioning_resolution)
        payload = mx.full((1, 2, 1024), float(resolution), dtype=mx.float32)
        return (
            Trellis2StageOutput(
                stage="image-conditioning",
                name="cond",
                shape=tuple(payload.shape),
                dtype="float32",
                detail="fake DINOv3 conditioning",
                payload=payload,
            ),
            None,
        )

    def fake_sparse_forward(*args, **kwargs):
        return SimpleNamespace(
            sampled_latent=mx.ones((1, 2, 2, 2, 2), dtype=mx.float32),
            sampled_latent_shape=(1, 2, 2, 2, 2),
            blocker_operation="sparse ok",
            blocker_detail="sparse sampled",
            loaded_tensor_names=("input_layer.weight",),
            checkpoint_path="sparse.safetensors",
        )

    def fake_sparse_decoder(*args, **kwargs):
        calls["target_resolution"] = kwargs["target_resolution"]
        return SimpleNamespace(
            coordinates=sparse_coordinates,
            coordinates_shape=tuple(sparse_coordinates.shape),
            checkpoint_path="sparse-decoder.safetensors",
            blocker_operation="sparse decoder ok",
            blocker_detail="coordinates decoded",
        )

    def fake_shape(root, config, coordinates, cond_512, cond_1024, *, max_num_tokens, decoder_token_limit):
        calls["shape_coordinates_input"] = coordinates
        calls["shape_cond_512"] = cond_512
        calls["shape_cond_1024"] = cond_1024
        assert config.default_pipeline_type == expected_pipeline_type
        assert coordinates is sparse_coordinates
        return shape_coordinates, shape_features, "shape detail from current run", expected_final_resolution

    def fake_texture(root, config, model_key, coordinates, features, conditioning):
        calls["texture_model"] = model_key
        calls["texture_coordinates_input"] = coordinates
        calls["texture_features_input"] = features
        calls["texture_conditioning"] = conditioning
        assert model_key == expected_texture_model
        assert coordinates is shape_coordinates
        assert features is shape_features
        return coordinates, texture_features, "texture detail from current shape SLat"

    def fake_read_decoder_config(root, config_path):
        calls.setdefault("decoder_config_paths", []).append(config_path)
        return SimpleNamespace(name="fake decoder config")

    def fake_shape_decoder(checkpoint_path, decoder_config, coordinates, features, *, decoder_token_limit):
        calls["shape_decoder_checkpoint_path"] = checkpoint_path
        calls["shape_decoder_coordinates_input"] = coordinates
        calls["shape_decoder_features_input"] = features
        calls["shape_decoder_token_limit"] = decoder_token_limit
        assert coordinates is shape_coordinates
        assert features is shape_features
        return SimpleNamespace(
            coordinates=shape_decoder_coordinates,
            fields=shape_fields,
            subdivisions=shape_subdivisions,
            probe=SimpleNamespace(
                completed_levels=2,
                decoder_output_coordinate_shape=tuple(shape_decoder_coordinates.shape),
                subdivision_shapes=((2, 8),),
            ),
        )

    def fake_texture_decoder(
        checkpoint_path,
        decoder_config,
        coordinates,
        features,
        *,
        guide_subdivisions,
        decoder_token_limit,
        decode_resolution,
        shape_decoder_coordinates,
    ):
        calls["texture_decoder_checkpoint_path"] = checkpoint_path
        calls["texture_decoder_coordinates_input"] = coordinates
        calls["texture_decoder_features_input"] = features
        calls["texture_decoder_guides"] = guide_subdivisions
        calls["texture_decoder_token_limit"] = decoder_token_limit
        calls["texture_decoder_decode_resolution"] = decode_resolution
        calls["texture_decoder_shape_coordinates"] = shape_decoder_coordinates
        assert coordinates is shape_coordinates
        assert features is texture_features
        assert guide_subdivisions is shape_subdivisions
        return SimpleNamespace(
            coordinates=texture_coordinates,
            attributes=texture_attributes,
            probe=SimpleNamespace(
                completed_levels=2,
                decoder_output_coordinate_shape=tuple(texture_coordinates.shape),
                decoder_output_shape=tuple(texture_attributes.shape),
                subdivision_shapes=(),
            ),
            guide_subdivision_shapes=((2, 8),),
            spatial_shape=(3, 3, 3),
            batch_size=1,
            decode_resolution=expected_final_resolution,
            voxel_size=1 / expected_final_resolution,
            shape_decoder_coordinate_shape=tuple(shape_decoder_coordinates.shape),
        )

    def fake_spatialkit_exporter(decoded_dir, output_path, **kwargs):
        output = Path(output_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"glTF fixture")
        diagnostics_path = (
            Path(kwargs["diagnostics_path"])
            if kwargs.get("diagnostics_path") is not None
            else output.with_name("diagnostics.json")
        )
        diagnostics_path.write_text("{}", encoding="utf-8")
        calls["decoded_dir"] = Path(decoded_dir)
        calls["glb_output_path"] = Path(output_path)
        calls.update(kwargs)
        return SimpleNamespace(
            glb=SimpleNamespace(
                path=output,
                format="glb",
                bytes_written=output.stat().st_size,
                metadata={"stage": "textured_glb", "mesh_name": "TRELLIS2_TexturedMesh"},
            ),
            diagnostics_path=diagnostics_path,
            diagnostics={"stages": {}},
        )

    monkeypatch.setattr(trellis2_inference, "assess_dinov3_conditioning", fake_conditioning)
    monkeypatch.setattr(trellis2_inference, "probe_sparse_structure_forward_boundary", fake_sparse_forward)
    monkeypatch.setattr(trellis2_inference, "read_sparse_structure_decoder_config", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(trellis2_inference, "probe_sparse_structure_decoder_boundary", fake_sparse_decoder)
    monkeypatch.setattr(trellis2_inference, "_sample_shape_slat_for_pipeline", fake_shape)
    monkeypatch.setattr(trellis2_inference, "_sample_texture_slat_model", fake_texture)
    monkeypatch.setattr(trellis2_inference, "read_structured_latent_decoder_config", fake_read_decoder_config)
    monkeypatch.setattr(trellis2_inference, "run_shape_decoder_to_fields", fake_shape_decoder)
    monkeypatch.setattr(trellis2_inference, "run_texture_decoder_to_representation", fake_texture_decoder)
    monkeypatch.setattr(trellis2_inference, "load_spatialkit_exporter", lambda: (fake_spatialkit_exporter, None))
    return calls, shape_coordinates, shape_features, texture_features, shape_subdivisions, texture_attributes


def test_generate_textured_glb_valid_metadata_writes_textured_glb(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_textured_trellis2_root(tmp_path / "trellis")
    image = tmp_path / "image.png"
    _write_rgba_foreground(image)
    calls, shape_coordinates, shape_features, texture_features, shape_subdivisions, texture_attributes = (
        _patch_textured_glb_exact_slat_fixtures(
            monkeypatch,
            expected_pipeline_type="1024_cascade",
            expected_texture_model="tex_slat_flow_model_1024",
        )
    )

    result = Trellis2InferencePipeline(tmp_path / "trellis").generate_textured_glb(
        image,
        output_path="outputs/trellis2/demo.glb",
        slat_steps=2,
        retain_trace_payloads=True,
    )

    assert result.ready
    assert result.artifact is not None
    assert result.artifact.bytes_written == len(b"glTF fixture")
    assert result.trace.completed_stages == (
        "asset-config-validation",
        "checkpoint-probe-readiness",
        "input-image",
        "image-preprocessing-background",
        "image-conditioning",
        "sparse-structure-sampling",
        "shape-slat-sampling",
        "texture-slat-sampling",
        "shape-decoder",
        "texture-decoder",
        "decoded-artifact-write",
        "mesh-export",
    )
    assert result.trace.outputs[0].detail.startswith("pipeline_type=1024_cascade; texture_model=tex_slat_flow_model_1024")
    assert "conditioning_resolution=1024" in result.trace.outputs[0].detail
    shape_output = next(output for output in result.trace.outputs if output.name == "shape_slat")
    texture_output = next(output for output in result.trace.outputs if output.name == "texture_slat")
    shape_decoder_output = next(output for output in result.trace.outputs if output.name == "shape_flexidualgrid_fields")
    texture_attrs_output = next(output for output in result.trace.outputs if output.name == "texture_voxel_attrs")
    decoded_output = next(output for output in result.trace.outputs if output.name == "decoded_ovoxel_artifacts")
    glb_output = next(output for output in result.trace.outputs if output.name == "textured_glb")
    assert shape_output.payload is shape_features
    assert texture_output.payload is texture_features
    assert texture_attrs_output.payload is texture_attributes
    assert "shape_decoder_fields.npz" in decoded_output.detail
    assert "texture_decoder_pbr.npz" in decoded_output.detail
    assert "wrote textured GLB artifact" in glb_output.detail
    assert "texture_tokens=2" in texture_output.detail
    assert "shape_tokens=2" in texture_output.detail
    assert "final_decode_resolution=1024" in texture_output.detail
    assert "subdivisions ((2, 8),)" in shape_decoder_output.detail
    assert "owned_subdivisions=()" in texture_attrs_output.detail
    assert calls["texture_coordinates_input"] is shape_coordinates
    assert calls["texture_features_input"] is shape_features
    assert calls["shape_decoder_coordinates_input"] is shape_coordinates
    assert calls["shape_decoder_features_input"] is shape_features
    assert calls["texture_decoder_guides"] is shape_subdivisions
    assert calls["texture_decoder_decode_resolution"] == 1024
    assert calls["decoded_dir"] == Path("outputs/trellis2/decoded")
    assert calls["grid_size"] == 1024
    assert calls["target_faces"] == 200000
    assert calls["texture_size"] == 1024
    assert calls["quality_preset"] == "reference-target"
    assert calls["glb_output_path"] == Path("outputs/trellis2/demo.glb")
    assert set(result.trace.timings_sec) >= {
        "shape-decoder",
        "texture-decoder",
        "decoded-artifact-write",
        "mesh-export",
        "total",
    }
    assert result.trace.blocker is None


def test_generate_textured_glb_spatialkit_failure_preserves_decoded_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_textured_trellis2_root(tmp_path / "trellis")
    image = tmp_path / "image.png"
    _write_rgba_foreground(image)
    _patch_textured_glb_exact_slat_fixtures(
        monkeypatch,
        expected_pipeline_type="1024_cascade",
        expected_texture_model="tex_slat_flow_model_1024",
    )

    def fake_export(*args, **kwargs):
        raise OSError("fixture SpatialKit export failure")

    monkeypatch.setattr(trellis2_inference, "export_ovoxel_glb", fake_export)

    result = Trellis2InferencePipeline(tmp_path / "trellis").generate_textured_glb(
        image,
        output_path="outputs/trellis2/demo.glb",
    )

    assert not result.ready
    assert result.trace.completed_stages == (
        "asset-config-validation",
        "checkpoint-probe-readiness",
        "input-image",
        "image-preprocessing-background",
        "image-conditioning",
        "sparse-structure-sampling",
        "shape-slat-sampling",
        "texture-slat-sampling",
        "shape-decoder",
        "texture-decoder",
        "decoded-artifact-write",
    )
    assert any(output.name == "decoded_ovoxel_artifacts" for output in result.trace.outputs)
    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "mesh-export"
    assert result.trace.blocker.operation == "export decoded TRELLIS.2 O-Voxel artifacts through SpatialKit"
    assert "fixture SpatialKit export failure" in result.trace.blocker.reason


def test_generate_textured_glb_spatialkit_oserror_is_structured_blocker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_textured_trellis2_root(tmp_path / "trellis")
    image = tmp_path / "image.png"
    _write_rgba_foreground(image)
    _patch_textured_glb_exact_slat_fixtures(
        monkeypatch,
        expected_pipeline_type="1024_cascade",
        expected_texture_model="tex_slat_flow_model_1024",
    )

    def fake_export(*args, **kwargs):
        raise OSError("filesystem unavailable")

    monkeypatch.setattr(trellis2_inference, "export_ovoxel_glb", fake_export)

    result = Trellis2InferencePipeline(tmp_path / "trellis").generate_textured_glb(
        image,
        output_path="outputs/trellis2/demo.glb",
    )

    assert not result.ready
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "mesh-export"
    assert result.trace.blocker.operation == "export decoded TRELLIS.2 O-Voxel artifacts through SpatialKit"
    assert "filesystem unavailable" in result.trace.blocker.reason


def test_generate_textured_glb_fixture_writes_nonempty_glb(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_textured_trellis2_root(tmp_path / "trellis")
    image = tmp_path / "image.png"
    _write_rgba_foreground(image)
    _patch_textured_glb_exact_slat_fixtures(
        monkeypatch,
        expected_pipeline_type="1024_cascade",
        expected_texture_model="tex_slat_flow_model_1024",
    )

    result = Trellis2InferencePipeline(tmp_path / "trellis").generate_textured_glb(
        image,
        output_path="outputs/trellis2/demo.glb",
    )

    output = tmp_path / "outputs/trellis2/demo.glb"
    assert result.ready
    assert result.artifact is not None
    assert result.artifact.path == output.resolve()
    assert result.artifact.bytes_written == output.stat().st_size
    assert result.trace.completed_stages[-1] == "mesh-export"
    assert result.trace.blocker is None
    assert output.read_bytes().startswith(b"glTF")


def test_generate_textured_glb_texture_decoder_failure_preserves_completed_trace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_textured_trellis2_root(tmp_path / "trellis")
    image = tmp_path / "image.png"
    _write_rgba_foreground(image)
    _patch_textured_glb_exact_slat_fixtures(
        monkeypatch,
        expected_pipeline_type="1024_cascade",
        expected_texture_model="tex_slat_flow_model_1024",
    )

    def fake_texture_decoder(*args, **kwargs):
        raise ValueError("guided texture decoder fixture failure")

    monkeypatch.setattr(trellis2_inference, "run_texture_decoder_to_representation", fake_texture_decoder)

    result = Trellis2InferencePipeline(tmp_path / "trellis").generate_textured_glb(
        image,
        output_path="outputs/trellis2/demo.glb",
    )

    assert result.trace.completed_stages == (
        "asset-config-validation",
        "checkpoint-probe-readiness",
        "input-image",
        "image-preprocessing-background",
        "image-conditioning",
        "sparse-structure-sampling",
        "shape-slat-sampling",
        "texture-slat-sampling",
        "shape-decoder",
    )
    assert any(output.name == "texture_slat" for output in result.trace.outputs)
    assert any(output.name == "shape_flexidualgrid_fields" for output in result.trace.outputs)
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "texture-decoder"
    assert result.trace.blocker.operation == "MLX texture decoder guided SparseUnetVaeDecoder execution"
    assert "guided texture decoder fixture failure" in result.trace.blocker.reason


def test_generate_textured_glb_shape_decoder_failure_after_texture_slat_is_shape_blocker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_textured_trellis2_root(tmp_path / "trellis")
    image = tmp_path / "image.png"
    _write_rgba_foreground(image)
    _patch_textured_glb_exact_slat_fixtures(
        monkeypatch,
        expected_pipeline_type="1024_cascade",
        expected_texture_model="tex_slat_flow_model_1024",
    )

    def fake_shape_decoder(*args, **kwargs):
        raise ValueError("shape decoder fixture failure after texture slat")

    monkeypatch.setattr(trellis2_inference, "run_shape_decoder_to_fields", fake_shape_decoder)

    result = Trellis2InferencePipeline(tmp_path / "trellis").generate_textured_glb(
        image,
        output_path="outputs/trellis2/demo.glb",
    )

    assert result.trace.completed_stages == (
        "asset-config-validation",
        "checkpoint-probe-readiness",
        "input-image",
        "image-preprocessing-background",
        "image-conditioning",
        "sparse-structure-sampling",
        "shape-slat-sampling",
        "texture-slat-sampling",
    )
    assert any(output.name == "texture_slat" for output in result.trace.outputs)
    assert not any(output.name == "shape_flexidualgrid_fields" for output in result.trace.outputs)
    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "shape-decoder"
    assert result.trace.blocker.operation == "MLX shape decoder FlexiDualGrid field execution"
    assert result.trace.blocker.reference == "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors"
    assert "shape decoder fixture failure after texture slat" in result.trace.blocker.reason


def test_generate_textured_glb_route_metadata_for_512(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_textured_trellis2_root(tmp_path / "trellis")
    image = tmp_path / "image.png"
    _write_rgba_foreground(image)
    calls, _, _, _, _, _ = _patch_textured_glb_exact_slat_fixtures(
        monkeypatch,
        expected_pipeline_type="512",
        expected_texture_model="tex_slat_flow_model_512",
    )

    result = Trellis2InferencePipeline(tmp_path / "trellis").generate_textured_glb(
        image,
        output_path="outputs/trellis2/demo.glb",
        pipeline_type="512",
    )

    assert result.ready
    assert result.artifact is not None
    assert result.trace.blocker is None
    assert result.trace.completed_stages[-1] == "mesh-export"
    route_output = result.trace.outputs[0]
    texture_output = next(output for output in result.trace.outputs if output.name == "texture_slat")
    assert "pipeline_type=512; texture_model=tex_slat_flow_model_512" in route_output.detail
    assert "conditioning_resolution=512" in route_output.detail
    assert "final_decode_resolution=512" in texture_output.detail
    assert calls["shape_cond_1024"] is None


def test_generate_textured_glb_route_metadata_for_1536_cascade(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_textured_trellis2_root(tmp_path / "trellis")
    image = tmp_path / "image.png"
    _write_rgba_foreground(image)
    calls, _, _, _, _, _ = _patch_textured_glb_exact_slat_fixtures(
        monkeypatch,
        expected_pipeline_type="1536_cascade",
        expected_texture_model="tex_slat_flow_model_1024",
    )

    result = Trellis2InferencePipeline(tmp_path / "trellis").generate_textured_glb(
        image,
        output_path="outputs/trellis2/demo.glb",
        pipeline_type="1536_cascade",
    )

    assert result.ready
    assert result.artifact is not None
    assert result.trace.blocker is None
    assert result.trace.completed_stages[-1] == "mesh-export"
    route_output = result.trace.outputs[0]
    texture_output = next(output for output in result.trace.outputs if output.name == "texture_slat")
    assert "pipeline_type=1536_cascade; texture_model=tex_slat_flow_model_1024" in route_output.detail
    assert "conditioning_resolution=1024" in route_output.detail
    assert "final_decode_resolution=1536" in texture_output.detail
    assert calls["texture_model"] == "tex_slat_flow_model_1024"


def test_inference_helpers_are_public_exports():
    assert mlx_spatial.TRELLIS2_INFERENCE_STAGES is TRELLIS2_INFERENCE_STAGES
    assert mlx_spatial.Trellis2InferencePipeline is Trellis2InferencePipeline
    assert mlx_spatial.Trellis2InferenceBlocker is Trellis2InferenceBlocker
    assert mlx_spatial.Trellis2TexturedGenerationResult is Trellis2TexturedGenerationResult
