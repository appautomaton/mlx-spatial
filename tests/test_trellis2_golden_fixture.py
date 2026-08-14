"""End-to-end regression coverage for the miniature TRELLIS.2 golden fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mlx_spatial.trellis2_inference as trellis2_inference
from mlx_spatial.trellis2_inference import Trellis2InferencePipeline
from mlx_spatial.trellis2_quantization import read_trellis2_quantization_spec
from tests.golden_assertions import assert_golden_close, summarize_glb
from tests.trellis2_golden_fixture import (
    TRELLIS2_MINIATURE_EXPORT_GRID_SIZE,
    TRELLIS2_MINIATURE_EXPORT_TARGET_FACES,
    TRELLIS2_MINIATURE_EXPORT_TEXTURE_SIZE,
    Trellis2MiniatureSpatialKitExporter,
    build_trellis2_miniature_golden_fixture,
    summarize_trellis2_golden_trace,
)

GOLDEN_MANIFEST = Path(__file__).parent / "data/trellis2_miniature_golden.json"


@pytest.mark.heavy
def test_miniature_int8_pipeline_emits_golden_trace_and_glb(tmp_path, monkeypatch):
    fixture = build_trellis2_miniature_golden_fixture(tmp_path)
    exporter = Trellis2MiniatureSpatialKitExporter()
    monkeypatch.setattr(
        trellis2_inference,
        "load_spatialkit_exporter",
        lambda: (exporter, None),
    )
    quantization = read_trellis2_quantization_spec(
        fixture.quantized_root / "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"
    )
    assert quantization is not None
    assert quantization.bits == 8
    assert quantization.group_size == 64
    assert quantization.tensors

    dino_quantization = read_trellis2_quantization_spec(
        fixture.dino_root / "model.safetensors"
    )
    assert dino_quantization is not None
    assert dino_quantization.bits == 8
    assert dino_quantization.group_size == 64
    assert dino_quantization.tensors

    result = Trellis2InferencePipeline(fixture.quantized_root).generate_textured_glb(
        fixture.image_path,
        output_path=fixture.output_path,
        dino_root=fixture.dino_root,
        pipeline_type="512",
        seed=7,
        max_num_tokens=4_096,
        decoder_token_limit=10_000,
        texture_size=32,
        glb_target_faces=256,
        retain_trace_payloads=True,
    )

    assert result.ready, result.trace.blocker
    summary = {
        "schema_version": 2,
        "fixture": {
            "checkpoint_source": "generated synthetic tensors",
            "pipeline_type": "512",
            "seed": 7,
            "sampler_steps": 1,
            "quantization_bits": 8,
            "quantization_group_size": 64,
            "texture_size": 32,
            "glb_target_faces": 256,
        },
        "trace": summarize_trellis2_golden_trace(result.trace),
        "glb": summarize_glb(fixture.output_path),
        "export": {
            "requested": exporter.requested_options,
            "effective": {
                "grid_size": TRELLIS2_MINIATURE_EXPORT_GRID_SIZE,
                "target_faces": TRELLIS2_MINIATURE_EXPORT_TARGET_FACES,
                "texture_size": TRELLIS2_MINIATURE_EXPORT_TEXTURE_SIZE,
            },
        },
    }
    expected = json.loads(GOLDEN_MANIFEST.read_text(encoding="utf-8"))
    assert_golden_close(summary, expected)
