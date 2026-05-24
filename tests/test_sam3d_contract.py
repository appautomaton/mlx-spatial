from pathlib import Path

import mlx.core as mx
import pytest
from safetensors.mlx import save_file

from mlx_spatial.sam3d_contract import (
    SAM3D_REQUIRED_CONTRACT_COMPONENTS,
    audit_sam3d_source_weight_contract,
)


SAM3D_MLX_ROOT = Path("weights/sam-3d-objects-mlx")
SAM3D_MLX_PIPELINE = SAM3D_MLX_ROOT / "checkpoints" / "pipeline.yaml"


def _write_contract_fixture(root: Path, *, unknown_target: bool = False, omit_prefix: str | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pipeline = """
_target_: sam3d_objects.pipeline.inference_pipeline_pointmap.InferencePipelinePointMap
ss_generator_config_path: ss_generator.yaml
ss_generator_ckpt_path: ss_generator.safetensors
slat_generator_config_path: slat_generator.yaml
slat_generator_ckpt_path: slat_generator.safetensors
ss_decoder_config_path: ss_decoder.yaml
ss_decoder_ckpt_path: ss_decoder.safetensors
slat_decoder_gs_config_path: slat_decoder_gs.yaml
slat_decoder_gs_ckpt_path: slat_decoder_gs.safetensors
slat_decoder_mesh_config_path: slat_decoder_mesh.yaml
slat_decoder_mesh_ckpt_path: slat_decoder_mesh.safetensors
ss_preprocessor:
  _target_: sam3d_objects.data.dataset.tdfy.preprocessor.PreProcessor
depth_model:
  _target_: sam3d_objects.pipeline.depth_models.moge.MoGe
"""
    (root / "pipeline.yaml").write_text(pipeline.strip(), encoding="utf-8")
    ss_condition_target = (
        "sam3d_objects.model.backbone.unknown.UnknownCondition"
        if unknown_target
        else "sam3d_objects.model.backbone.dit.embedder.embedder_fuser.EmbedderFuser"
    )
    (root / "ss_generator.yaml").write_text(
        f"""
module:
  condition_embedder:
    backbone:
      _target_: {ss_condition_target}
  generator:
    backbone:
      _target_: sam3d_objects.model.backbone.generator.shortcut.model.ShortCut
      reverse_fn:
        _target_: sam3d_objects.model.backbone.generator.classifier_free_guidance.ClassifierFreeGuidanceWithExternalUnconditionalProbability
        backbone:
          _target_: sam3d_objects.model.backbone.tdfy_dit.models.mot_sparse_structure_flow.SparseStructureFlowTdfyWrapper
""".strip(),
        encoding="utf-8",
    )
    (root / "slat_generator.yaml").write_text(
        """
module:
  condition_embedder:
    backbone:
      _target_: sam3d_objects.model.backbone.dit.embedder.embedder_fuser.EmbedderFuser
  generator:
    backbone:
      _target_: sam3d_objects.model.backbone.generator.flow_matching.model.FlowMatching
      reverse_fn:
        _target_: sam3d_objects.model.backbone.generator.classifier_free_guidance.ClassifierFreeGuidance
        backbone:
          _target_: sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_flow.SLatFlowModelTdfyWrapper
""".strip(),
        encoding="utf-8",
    )
    (root / "ss_decoder.yaml").write_text(
        "_target_: sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae.SparseStructureDecoderTdfyWrapper\n",
        encoding="utf-8",
    )
    (root / "slat_decoder_gs.yaml").write_text(
        "_target_: sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.decoder_gs.SLatGaussianDecoderTdfyWrapper\n",
        encoding="utf-8",
    )
    (root / "slat_decoder_mesh.yaml").write_text(
        "_target_: sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.decoder_mesh.SLatMeshDecoderTdfyWrapper\n",
        encoding="utf-8",
    )
    _write_checkpoint(
        root / "ss_generator.safetensors",
        ("_base_models.condition_embedder.", "_base_models.generator."),
        omit_prefix=omit_prefix,
    )
    _write_checkpoint(
        root / "slat_generator.safetensors",
        ("_base_models.condition_embedder.", "_base_models.generator."),
        omit_prefix=omit_prefix,
    )
    _write_checkpoint(
        root / "ss_decoder.safetensors",
        ("input_layer.", "blocks.", "middle_block.", "out_layer."),
        omit_prefix=omit_prefix,
    )
    _write_checkpoint(
        root / "slat_decoder_gs.safetensors",
        ("input_layer.", "blocks.", "out_layer.", "offset_perturbation"),
        omit_prefix=omit_prefix,
    )
    _write_checkpoint(
        root / "slat_decoder_mesh.safetensors",
        ("input_layer.", "blocks.", "out_layer.", "upsample."),
        omit_prefix=omit_prefix,
    )


def _write_checkpoint(path: Path, prefixes: tuple[str, ...], *, omit_prefix: str | None) -> None:
    tensors = {
        f"{prefix}weight": mx.array([1.0], dtype=mx.float32)
        for prefix in prefixes
        if prefix != omit_prefix
    }
    save_file(tensors, path)


@pytest.mark.heavy
@pytest.mark.skipif(not SAM3D_MLX_PIPELINE.is_file(), reason="SAM3D MLX weights absent")
def test_real_sam3d_mlx_contract_maps_source_targets_and_weight_prefixes():
    audit = audit_sam3d_source_weight_contract(SAM3D_MLX_ROOT)

    assert audit.ready, [issue.reason for issue in audit.issues]
    components = {item.component: item for item in audit.component_mappings}
    assert set(SAM3D_REQUIRED_CONTRACT_COMPONENTS).issubset(components)
    assert components["ss_condition"].required_prefixes == ("_base_models.condition_embedder.",)
    assert components["ss_generator"].required_prefixes == ("_base_models.generator.",)
    assert components["ss_decoder"].required_prefixes == ("input_layer.", "blocks.", "middle_block.", "out_layer.")
    assert components["slat_condition"].required_prefixes == ("_base_models.condition_embedder.",)
    assert components["slat_generator"].required_prefixes == ("_base_models.generator.",)
    assert components["gs_decoder"].required_prefixes == ("input_layer.", "blocks.", "out_layer.", "offset_perturbation")
    assert components["mesh_decoder"].required_prefixes == ("input_layer.", "blocks.", "out_layer.", "upsample.")
    assert all(mapping.tensor_count > 0 for mapping in audit.prefix_mappings)


def test_sam3d_contract_fails_on_unmapped_active_target(tmp_path):
    _write_contract_fixture(tmp_path, unknown_target=True)

    audit = audit_sam3d_source_weight_contract(tmp_path)

    assert not audit.ready
    assert any(issue.code == "unmapped-active-target" for issue in audit.issues)
    assert any(issue.metadata.get("target") == "sam3d_objects.model.backbone.unknown.UnknownCondition" for issue in audit.issues)


def test_sam3d_contract_fails_on_missing_required_prefix(tmp_path):
    _write_contract_fixture(tmp_path, omit_prefix="_base_models.generator.")

    audit = audit_sam3d_source_weight_contract(tmp_path)

    assert not audit.ready
    missing = [issue for issue in audit.issues if issue.code == "missing-required-prefix"]
    assert missing
    assert any(issue.metadata["component"] in {"ss_generator", "slat_generator"} for issue in missing)
