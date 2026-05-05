import tomllib
import types

import mlx.core as mx
from safetensors.mlx import save_file

from mlx_spatial.sam3d_assets import (
    SAM3D_OBJECTS_REPO_ID,
    convert_sam3d_assets_to_safetensors,
    inspect_sam3d_model_assets,
    read_sam3d_pipeline_config,
    resolve_sam3d_pipeline_path,
    sam3d_download_command,
    validate_sam3d_assets,
)
from mlx_spatial.sam3d_moge import SAM3D_MOGE_REQUIRED_KEYS, inspect_sam3d_moge_assets


def _write_sam3d_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir()
    (root / "ckpts").mkdir()
    pipeline = """
_target_: sam3d_objects.pipeline.inference_pipeline_pointmap.InferencePipelinePointMap
dtype: bfloat16
rendering_engine: pytorch3d
decode_formats: [gaussian, mesh]
ss_generator_config_path: configs/ss_generator.yaml
ss_generator_ckpt_path: ckpts/ss_generator.safetensors
slat_generator_config_path: configs/slat_generator.yaml
slat_generator_ckpt_path: ckpts/slat_generator.safetensors
ss_decoder_config_path: configs/ss_decoder.yaml
ss_decoder_ckpt_path: ckpts/ss_decoder.safetensors
slat_decoder_gs_config_path: configs/slat_decoder_gs.yaml
slat_decoder_gs_ckpt_path: ckpts/slat_decoder_gs.safetensors
slat_decoder_mesh_config_path: configs/slat_decoder_mesh.yaml
slat_decoder_mesh_ckpt_path: ckpts/slat_decoder_mesh.safetensors
depth_model:
  _target_: sam3d_objects.pipeline.depth_models.moge.MoGe
"""
    (root / "pipeline.yaml").write_text(pipeline.strip(), encoding="utf-8")
    for name in (
        "ss_generator",
        "slat_generator",
        "ss_decoder",
        "slat_decoder_gs",
        "slat_decoder_mesh",
    ):
        (root / "configs" / f"{name}.yaml").write_text("_target_: fixture\n", encoding="utf-8")
        save_file({f"{name}.weight": mx.array([1.0], dtype=mx.float32)}, root / "ckpts" / f"{name}.safetensors")


def test_sam3d_runtime_dependencies_exclude_torch_cuda_and_hf_runtime_clients():
    config = tomllib.loads(open("pyproject.toml", "rb").read().decode())
    runtime_dependencies = "\n".join(config["project"]["dependencies"]).lower()
    dev_dependencies = "\n".join(config["dependency-groups"]["dev"]).lower()

    for forbidden in ("torch", "cuda", "gsplat", "flash-attn", "xformers", "spconv", "kaolin", "huggingface"):
        assert forbidden not in runtime_dependencies
    assert "pyyaml" in runtime_dependencies
    assert "huggingface-hub" in dev_dependencies
    assert "pt-safe-loader" in dev_dependencies
    assert "pt-safe-loader" not in runtime_dependencies


def test_validate_sam3d_assets_reports_missing_pipeline(tmp_path):
    validation = validate_sam3d_assets(tmp_path)

    assert not validation.ready
    assert validation.pipeline_path is None
    assert validation.missing == ("pipeline.yaml or checkpoints/pipeline.yaml or checkpoints/hf/pipeline.yaml",)


def test_validate_sam3d_assets_accepts_direct_pipeline(tmp_path):
    _write_sam3d_fixture(tmp_path)

    validation = validate_sam3d_assets(tmp_path)

    assert validation.ready
    assert validation.model_dir == tmp_path
    assert validation.pipeline_path == tmp_path / "pipeline.yaml"
    assert validation.present == ("pipeline.yaml",)


def test_resolve_sam3d_pipeline_path_accepts_nested_download_layout(tmp_path):
    nested = tmp_path / "checkpoints" / "hf"
    nested.mkdir(parents=True)
    (nested / "pipeline.yaml").write_text("{}\n", encoding="utf-8")

    assert resolve_sam3d_pipeline_path(tmp_path) == nested / "pipeline.yaml"


def test_read_sam3d_pipeline_config_reads_official_fields(tmp_path):
    _write_sam3d_fixture(tmp_path)

    config = read_sam3d_pipeline_config(tmp_path / "pipeline.yaml")

    assert config.target == "sam3d_objects.pipeline.inference_pipeline_pointmap.InferencePipelinePointMap"
    assert config.dtype == "bfloat16"
    assert config.rendering_engine == "pytorch3d"
    assert config.decode_formats == ("gaussian", "mesh")


def test_inspect_sam3d_model_assets_accepts_complete_fake_fixture(tmp_path):
    _write_sam3d_fixture(tmp_path)

    inspection = inspect_sam3d_model_assets(tmp_path)

    assert inspection.ready
    assert inspection.blocker is None
    assert inspection.config is not None
    assert inspection.missing_paths == ()
    assert len(inspection.checkpoints) == 5
    assert inspection.checkpoints[0].tensor_count == 1
    assert inspection.checkpoints[0].format == "safetensors"


def test_convert_sam3d_assets_rewrites_pipeline_to_safetensors_without_metadata_collision(
    tmp_path,
    monkeypatch,
):
    _write_sam3d_fixture(tmp_path)
    source_checkpoint = tmp_path / "ckpts" / "ss_decoder.ckpt"
    source_checkpoint.write_bytes(b"fake torch zip")
    pipeline = (tmp_path / "pipeline.yaml").read_text(encoding="utf-8")
    (tmp_path / "pipeline.yaml").write_text(
        pipeline.replace("ckpts/ss_decoder.safetensors", "ckpts/ss_decoder.ckpt"),
        encoding="utf-8",
    )

    class FakePtCheckpoint:
        @classmethod
        def load(cls, path, **kwargs):
            assert path.endswith("ss_decoder.ckpt")
            assert kwargs["max_archive_bytes"] == 16 * 1024**3
            return cls()

        def export(self, *, format, dir):
            assert format == "safetensors"
            output_dir = tmp_path / "fake-export"
            output_dir.mkdir(exist_ok=True)
            weights_path = output_dir / "ss_decoder.safetensors"
            metadata_path = output_dir / "ss_decoder.yaml"
            save_file({"converted.weight": mx.array([2.0], dtype=mx.float32)}, weights_path)
            metadata_path.write_text("source_sha256: fake-sha\n", encoding="utf-8")
            return {"weights_path": str(weights_path), "metadata_path": str(metadata_path)}

    monkeypatch.setitem(__import__("sys").modules, "pt_loader", types.SimpleNamespace(PtCheckpoint=FakePtCheckpoint))

    result = convert_sam3d_assets_to_safetensors(tmp_path, output_root=tmp_path / "converted")

    assert result.ready
    converted_pipeline = (tmp_path / "converted" / "pipeline.yaml").read_text(encoding="utf-8")
    assert "ckpts/ss_decoder.safetensors" in converted_pipeline
    assert "ckpts/ss_decoder.ckpt" not in converted_pipeline
    assert (tmp_path / "converted" / "ckpts" / "ss_decoder.safetensors").is_file()
    assert (tmp_path / "converted" / "ckpts" / "conversion_metadata" / "ss_decoder.yaml").is_file()
    assert (tmp_path / "converted" / "configs" / "ss_decoder.yaml").read_text(encoding="utf-8") == "_target_: fixture\n"


def test_inspect_sam3d_model_assets_blocks_missing_referenced_checkpoint(tmp_path):
    _write_sam3d_fixture(tmp_path)
    (tmp_path / "ckpts" / "slat_decoder_gs.safetensors").unlink()

    inspection = inspect_sam3d_model_assets(tmp_path)

    assert not inspection.ready
    assert inspection.blocker is not None
    assert inspection.blocker.stage == "pipeline-config"
    assert "ckpts/slat_decoder_gs.safetensors" in inspection.blocker.metadata["missing"]


def test_inspect_sam3d_model_assets_ignores_unrequired_unsupported_mesh_checkpoint(tmp_path):
    _write_sam3d_fixture(tmp_path)
    pipeline = (tmp_path / "pipeline.yaml").read_text(encoding="utf-8")
    (tmp_path / "pipeline.yaml").write_text(
        pipeline.replace("ckpts/slat_decoder_mesh.safetensors", "ckpts/slat_decoder_mesh.ckpt"),
        encoding="utf-8",
    )
    (tmp_path / "ckpts" / "slat_decoder_mesh.safetensors").unlink()
    (tmp_path / "ckpts" / "slat_decoder_mesh.ckpt").write_bytes(b"not safetensors")

    gaussian_only = inspect_sam3d_model_assets(
        tmp_path,
        required_roles=("ss_generator", "slat_generator", "ss_decoder", "slat_decoder_gs"),
    )
    mesh_required = inspect_sam3d_model_assets(
        tmp_path,
        required_roles=("ss_generator", "slat_generator", "ss_decoder", "slat_decoder_gs", "slat_decoder_mesh"),
    )

    assert gaussian_only.ready
    assert all(item.role != "slat_decoder_mesh" for item in gaussian_only.checkpoints)
    assert mesh_required.blocker is not None
    assert mesh_required.blocker.stage == "checkpoint-inspection"
    assert "ckpts/slat_decoder_mesh.ckpt" in mesh_required.blocker.metadata["unsupported"]


def test_sam3d_download_command_is_explicit_and_gated_hf_based():
    command = sam3d_download_command("weights/sam-3d-objects")

    assert command[:4] == ("uv", "run", "hf", "download")
    assert "--repo-type" in command
    assert "model" in command
    assert "--max-workers" in command
    assert "1" in command
    assert SAM3D_OBJECTS_REPO_ID in command
    assert "weights/sam-3d-objects" in command


def test_inspect_sam3d_moge_assets_validates_converted_safetensors(tmp_path):
    save_file(
        {name: mx.array([1.0], dtype=mx.float32) for name in SAM3D_MOGE_REQUIRED_KEYS},
        tmp_path / "model.safetensors",
    )

    inspection = inspect_sam3d_moge_assets(tmp_path)

    assert inspection.ready
    assert inspection.blocker is None
    assert inspection.tensor_count == len(SAM3D_MOGE_REQUIRED_KEYS)


def test_inspect_sam3d_moge_assets_blocks_missing_checkpoint(tmp_path):
    inspection = inspect_sam3d_moge_assets(tmp_path)

    assert not inspection.ready
    assert inspection.blocker is not None
    assert inspection.blocker.stage == "moge-pointmap"
    assert "MoGe checkpoint not found" in inspection.blocker.reason
