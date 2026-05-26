import os
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_spatial.mapanything_assets import read_mapanything_model_config
from mlx_spatial.mapanything_model import (
    load_mapanything_full_encoder_weights,
    mapanything_encoder_prefix_config_from_model_config,
    mapanything_full_encoder_outputs_for_parity,
    run_mapanything_full_encoder,
)
from mlx_spatial.mapanything_parity import (
    MAPANYTHING_TORCH_PARITY_ENV,
    compare_mapanything_parity_tensors,
    load_mapanything_parity_bundle,
    mapanything_parity_report_to_dict,
)
from mlx_spatial.mapanything_preprocess import preprocess_mapanything_images


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = Path("/tmp/mapanything-desk-scene-reference.npz")

pytestmark = pytest.mark.skipif(
    os.environ.get(MAPANYTHING_TORCH_PARITY_ENV) != "1",
    reason="opt-in MapAnything Torch reference parity",
)


def test_mapanything_full_encoder_matches_desk_scene_reference():
    reference_path = Path(os.environ.get("MAPANYTHING_SCENE_REFERENCE", str(DEFAULT_REFERENCE)))
    if not reference_path.is_file():
        pytest.fail(
            f"missing scene reference bundle: {reference_path}; run "
            "tools/mapanything_dump_torch_scene_reference.py first"
        )

    model_root = ROOT / "weights/map-anything"
    image_root = ROOT / "inputs/map-anything/desk"
    if not (model_root / "model.safetensors").is_file() or not image_root.is_dir():
        pytest.skip("local MapAnything weights or Desk inputs are absent")

    reference = load_mapanything_parity_bundle(reference_path)
    config = mapanything_encoder_prefix_config_from_model_config(
        read_mapanything_model_config(model_root / "config.json")
    )
    weights = load_mapanything_full_encoder_weights(model_root, config=config)
    preprocessed = preprocess_mapanything_images(image_root, patch_size=config.patch_size)
    images = mx.concatenate([view.img for view in preprocessed.views], axis=0)

    output = run_mapanything_full_encoder(images, weights, config=config)
    mx.eval(output.features, output.registers, output.block0)
    report = compare_mapanything_parity_tensors(
        mapanything_full_encoder_outputs_for_parity(output),
        reference,
        names=(
            "encoder.patch_embed",
            "encoder.tokens",
            "encoder.block0",
            "encoder.features.0",
            "encoder.features.1",
            "encoder.registers.0",
            "encoder.registers.1",
        ),
        atol=2e-1,
        rtol=5e-2,
    )

    assert reference.metadata["torch_hub_disabled"] is True
    assert reference.metadata["runtime_depends_on_torch"] is False
    assert report.passed, mapanything_parity_report_to_dict(report)
