from safetensors.mlx import save_file
import mlx.core as mx
from PIL import Image

import mlx_spatial
from mlx_spatial.model_assets import TRELLIS2_ASSETS
from mlx_spatial.trellis2_inference import (
    TRELLIS2_INFERENCE_STAGES,
    Trellis2InferenceBlocker,
    Trellis2InferencePipeline,
    _conditioning_resolutions,
    _quantize_cascade_coordinates,
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


def test_dry_run_validates_fake_assets_and_reports_unimplemented_stage(tmp_path):
    _write_trellis2_root(tmp_path)

    report = Trellis2InferencePipeline(tmp_path).dry_run()

    assert not report.ready
    assert [stage.stage for stage in report.stages[:3]] == [
        "asset-config-validation",
        "checkpoint-probe-readiness",
        "image-preprocessing-background",
    ]
    assert report.stages[0].status == "ready"
    assert report.stages[1].status == "ready"
    assert report.stages[2].status == "unimplemented"
    assert report.blocker is not None
    assert report.blocker.stage == "image-preprocessing-background"
    assert "trellis2_image_to_3d.py:127-162" in report.blocker.reference


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
    assert report.blocker.operation == "MLX BiRefNet deformable convolution"
    assert "DeformConv2d" in report.blocker.reason


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


def test_generate_shape_rejects_glb_as_texture_export_blocker(tmp_path):
    _write_trellis2_root(tmp_path)
    image = tmp_path / "missing.png"

    result = Trellis2InferencePipeline(tmp_path).generate_shape_obj(
        image,
        output_path=tmp_path / "outputs/trellis2/demo.glb",
    )

    assert result.trace.blocker is not None
    assert result.trace.blocker.stage == "mesh-export"
    assert result.trace.blocker.operation == "TRELLIS.2 textured GLB export"
    assert "not implemented yet" in result.trace.blocker.reason


def test_inference_helpers_are_public_exports():
    assert mlx_spatial.TRELLIS2_INFERENCE_STAGES is TRELLIS2_INFERENCE_STAGES
    assert mlx_spatial.Trellis2InferencePipeline is Trellis2InferencePipeline
    assert mlx_spatial.Trellis2InferenceBlocker is Trellis2InferenceBlocker
