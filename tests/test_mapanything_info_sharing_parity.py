import os
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_spatial.mapanything_assets import read_mapanything_model_config
from mlx_spatial.mapanything_model import (
    load_mapanything_info_sharing_weights,
    mapanything_info_sharing_config_from_model_config,
    mapanything_info_sharing_outputs_for_parity,
    run_mapanything_info_sharing,
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


def test_mapanything_info_sharing_matches_desk_scene_reference():
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
    config = mapanything_info_sharing_config_from_model_config(
        read_mapanything_model_config(model_root / "config.json")
    )
    weights = load_mapanything_info_sharing_weights(model_root, config=config)
    features = (
        mx.array(reference.tensors["fusion.features.0"]),
        mx.array(reference.tensors["fusion.features.1"]),
    )
    registers = (
        mx.array(reference.tensors["encoder.registers.0"]),
        mx.array(reference.tensors["encoder.registers.1"]),
    )

    output = run_mapanything_info_sharing(
        features,
        weights,
        additional_tokens_per_view=registers,
        config=config,
    )
    mx.eval(
        output.final.additional_token_features,
        *output.final.features,
        *[feature for intermediate in output.intermediates for feature in intermediate.features],
    )
    report = compare_mapanything_parity_tensors(
        mapanything_info_sharing_outputs_for_parity(output),
        reference,
        names=(
            "info.intermediate.0.features.0",
            "info.intermediate.0.features.1",
            "info.intermediate.0.additional_token_features",
            "info.intermediate.0.additional_token_features_per_view.0",
            "info.intermediate.0.additional_token_features_per_view.1",
            "info.intermediate.1.features.0",
            "info.intermediate.1.features.1",
            "info.intermediate.1.additional_token_features",
            "info.intermediate.1.additional_token_features_per_view.0",
            "info.intermediate.1.additional_token_features_per_view.1",
            "info.final.features.0",
            "info.final.features.1",
            "info.final.additional_token_features",
            "info.final.additional_token_features_per_view.0",
            "info.final.additional_token_features_per_view.1",
        ),
        atol=5e-2,
        rtol=2e-2,
    )

    assert reference.metadata["torch_hub_disabled"] is True
    assert reference.metadata["runtime_depends_on_torch"] is False
    assert output.trace["attention_schedule"] == "even-global/odd-frame"
    assert report.passed, mapanything_parity_report_to_dict(report)
