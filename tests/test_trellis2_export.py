from pathlib import Path

import pytest

import mlx_spatial
from mlx_spatial.trellis2_export import (
    SUPPORTED_TRELLIS2_EXPORT_SUFFIXES,
    Trellis2ExportArtifact,
    Trellis2ExportResult,
    assess_trellis2_export_boundary,
    sparse_coordinates_to_obj_payload,
    validate_trellis2_export_path,
    write_sparse_coordinate_preview_obj,
    write_trellis2_export_artifact,
)
from mlx_spatial.trellis2_forward import (
    Trellis2ForwardBlocker,
    Trellis2ForwardTraceResult,
)


def _blocked_trace(tmp_path: Path):
    return Trellis2ForwardTraceResult(
        root=tmp_path / "weights/trellis2",
        image_path=tmp_path / "inputs/demo.webp",
        completed_stages=("input-image", "image-conditioning"),
        blocker=Trellis2ForwardBlocker(
            stage="sparse-structure-sampling",
            operation="MLX sparse structure FlowEuler sampler update loop",
            reference="weights/trellis2/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
            reason="block-0 executed, remaining sparse transformer stack not implemented",
            next_slice="implement FlowEuler denoising updates with classifier-free guidance for sparse structure sampling",
        ),
    )


def test_validate_export_path_requires_outputs_tree(tmp_path):
    outputs = tmp_path / "outputs"
    path = validate_trellis2_export_path(outputs / "trellis2/demo.glb", outputs_root=outputs)

    assert path == (outputs / "trellis2/demo.glb").resolve()

    with pytest.raises(ValueError, match="must stay under"):
        validate_trellis2_export_path(tmp_path / "outside/demo.glb", outputs_root=outputs)


def test_validate_export_path_rejects_unsupported_suffix(tmp_path):
    outputs = tmp_path / "outputs"

    with pytest.raises(ValueError, match="unsupported TRELLIS.2 export format"):
        validate_trellis2_export_path(outputs / "trellis2/demo.txt", outputs_root=outputs)


def test_write_export_artifact_reports_metadata_and_writes_under_outputs(tmp_path):
    output = tmp_path / "outputs/trellis2/demo.glb"

    artifact = write_trellis2_export_artifact(b"glTF", output, outputs_root=tmp_path / "outputs")

    assert artifact == Trellis2ExportArtifact(
        path=output.resolve(),
        format="glb",
        bytes_written=4,
        detail="wrote TRELLIS.2 mesh export artifact under ignored outputs tree",
    )
    assert output.read_bytes() == b"glTF"


def test_sparse_coordinate_preview_obj_writes_exposed_voxel_faces(tmp_path):
    import mlx.core as mx

    output = tmp_path / "outputs/trellis2/preview.obj"
    coords = mx.array([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=mx.int32)

    artifact = write_sparse_coordinate_preview_obj(coords, output, outputs_root=tmp_path / "outputs", grid_size=2)
    payload = output.read_text()

    assert artifact.format == "obj"
    assert artifact.detail == "wrote coarse TRELLIS.2 sparse-structure occupancy OBJ preview"
    assert payload.startswith("# mlx-spatial TRELLIS.2 sparse-structure occupancy preview")
    assert payload.count("\nv ") == 40
    assert payload.count("\nf ") == 10


def test_sparse_coordinate_preview_obj_rejects_empty_coordinates():
    import mlx.core as mx

    with pytest.raises(ValueError, match="at least one token"):
        sparse_coordinates_to_obj_payload(mx.array([], dtype=mx.int32).reshape((0, 4)))


def test_assess_export_boundary_reports_upstream_blocker(tmp_path):
    result = assess_trellis2_export_boundary(
        _blocked_trace(tmp_path),
        output_path=tmp_path / "outputs/trellis2/demo.glb",
        outputs_root=tmp_path / "outputs",
    )

    assert not result.ready
    assert result.artifact is None
    assert result.blocker is not None
    assert result.blocker.stage == "mesh-export"
    assert result.blocker.operation == "upstream inference completion before export"
    assert "sparse-structure-sampling / MLX sparse structure FlowEuler sampler update loop" in result.blocker.reason


def test_assess_export_boundary_reports_bad_output_path_before_upstream(tmp_path):
    result = assess_trellis2_export_boundary(
        _blocked_trace(tmp_path),
        output_path=tmp_path / "outside/demo.glb",
        outputs_root=tmp_path / "outputs",
    )

    assert result.blocker is not None
    assert result.blocker.operation == "TRELLIS.2 export path validation"


def test_export_helpers_are_public():
    assert mlx_spatial.SUPPORTED_TRELLIS2_EXPORT_SUFFIXES == SUPPORTED_TRELLIS2_EXPORT_SUFFIXES
    assert mlx_spatial.Trellis2ExportArtifact is Trellis2ExportArtifact
    assert mlx_spatial.Trellis2ExportResult is Trellis2ExportResult
    assert mlx_spatial.assess_trellis2_export_boundary is assess_trellis2_export_boundary
    assert mlx_spatial.sparse_coordinates_to_obj_payload is sparse_coordinates_to_obj_payload
    assert mlx_spatial.validate_trellis2_export_path is validate_trellis2_export_path
    assert mlx_spatial.write_sparse_coordinate_preview_obj is write_sparse_coordinate_preview_obj
    assert mlx_spatial.write_trellis2_export_artifact is write_trellis2_export_artifact
