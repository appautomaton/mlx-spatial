import os
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_spatial.mapanything_assets import read_mapanything_model_config
from mlx_spatial.mapanything_heads import (
    apply_mapanything_fusion_norm,
    load_mapanything_heads_weights,
    mapanything_heads_config_from_model_config,
    mapanything_heads_outputs_for_parity,
    run_mapanything_heads,
)
from mlx_spatial.mapanything_parity import (
    MAPANYTHING_TORCH_PARITY_ENV,
    compare_mapanything_parity_tensors,
    load_mapanything_parity_bundle,
    mapanything_parity_report_to_dict,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = Path("/tmp/mapanything-desk-scene-reference.npz")

pytestmark = pytest.mark.skipif(
    os.environ.get(MAPANYTHING_TORCH_PARITY_ENV) != "1",
    reason="opt-in MapAnything Torch reference parity",
)


def test_mapanything_fusion_and_heads_match_desk_scene_reference():
    reference_path = Path(os.environ.get("MAPANYTHING_SCENE_REFERENCE", str(DEFAULT_REFERENCE)))
    if not reference_path.is_file():
        pytest.fail(
            f"missing scene reference bundle: {reference_path}; run "
            "tools/mapanything_dump_torch_scene_reference.py first"
        )

    model_root = ROOT / "weights/map-anything"
    if not (model_root / "model.safetensors").is_file():
        pytest.skip("local MapAnything weights are absent")

    reference = load_mapanything_parity_bundle(reference_path)
    config = mapanything_heads_config_from_model_config(read_mapanything_model_config(model_root / "config.json"))
    weights = load_mapanything_heads_weights(model_root, config=config)

    fused = apply_mapanything_fusion_norm(
        (
            mx.array(reference.tensors["encoder.features.0"]),
            mx.array(reference.tensors["encoder.features.1"]),
        ),
        weights,
        config=config,
    )
    mx.eval(*fused)
    fusion_report = compare_mapanything_parity_tensors(
        {
            "fusion.features.0": fused[0],
            "fusion.features.1": fused[1],
        },
        reference,
        names=("fusion.features.0", "fusion.features.1"),
        atol=1e-4,
        rtol=1e-4,
    )
    assert fusion_report.passed, mapanything_parity_report_to_dict(fusion_report)

    dense_features = (
        mx.concatenate(
            (
                mx.array(reference.tensors["fusion.features.0"]),
                mx.array(reference.tensors["fusion.features.1"]),
            ),
            axis=0,
        ),
        mx.concatenate(
            (
                mx.array(reference.tensors["info.intermediate.0.features.0"]),
                mx.array(reference.tensors["info.intermediate.0.features.1"]),
            ),
            axis=0,
        ),
        mx.concatenate(
            (
                mx.array(reference.tensors["info.intermediate.1.features.0"]),
                mx.array(reference.tensors["info.intermediate.1.features.1"]),
            ),
            axis=0,
        ),
        mx.concatenate(
            (
                mx.array(reference.tensors["info.final.features.0"]),
                mx.array(reference.tensors["info.final.features.1"]),
            ),
            axis=0,
        ),
    )
    output = run_mapanything_heads(
        dense_features,
        mx.array(reference.tensors["info.final.additional_token_features"]),
        weights,
        image_shape=(392, 518),
        config=config,
    )
    mx.eval(output.dense.value, output.dense.confidence, output.dense.mask, output.pose_value, output.scale_value)
    report = compare_mapanything_parity_tensors(
        mapanything_heads_outputs_for_parity(output),
        reference,
        names=(
            "head.dense.value",
            "head.dense.confidence",
            "head.dense.mask",
            "head.dense.logits",
            "head.pose.value",
            "head.scale.value",
        ),
        atol=1e-1,
        rtol=5e-2,
    )

    assert reference.metadata["torch_hub_disabled"] is True
    assert reference.metadata["runtime_depends_on_torch"] is False
    assert output.trace["dense_output_type"] == "raydirs+depth+confidence+mask"
    assert report.passed, mapanything_parity_report_to_dict(report)
