import json
import shutil
import struct
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import mlx_spatial
from mlx_spatial.ovoxel import FlexibleDualGridMesh
from mlx_spatial.trellis2_export import (
    SUPPORTED_TRELLIS2_EXPORT_SUFFIXES,
    TRELLIS2_GLB_DEFAULT_FACE_TARGET,
    TRELLIS2_TEXTURE_BAKE_BACKENDS,
    TRELLIS2_XATLAS_FACE_GUARD,
    Trellis2ExportArtifact,
    Trellis2ExportResult,
    Trellis2PostprocessParityItem,
    Trellis2SparseTrilinearSampleResult,
    Trellis2TextureBakeResult,
    Trellis2XAtlasUnwrapResult,
    Trellis2XAtlasUnwrapStats,
    assess_trellis2_export_boundary,
    bake_trellis2_texture_fields_mac_native,
    bake_trellis2_texture_fields,
    ensure_trellis2_mac_export_dependencies,
    make_trellis2_face_atlas_uvs,
    missing_trellis2_mac_export_dependencies,
    postprocess_trellis2_mesh_for_glb,
    resolve_trellis2_xatlas_face_guard,
    sample_trellis2_sparse_trilinear_attributes,
    sparse_coordinates_to_obj_payload,
    trellis2_postprocess_parity_audit,
    trellis2_textured_glb_payload,
    trellis2_texture_png_payload,
    unwrap_trellis2_mesh_xatlas_with_stats,
    validate_trellis2_export_path,
    write_sparse_coordinate_preview_obj,
    write_trellis2_textured_glb,
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


def _fixture_mesh() -> FlexibleDualGridMesh:
    return FlexibleDualGridMesh(
        vertices=np.array(
            [
                [-0.25, -0.25, 0.0],
                [0.25, -0.25, 0.0],
                [-0.25, 0.25, 0.0],
                [0.25, 0.25, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64),
    )


def _fixture_texture_field():
    coords = mx.array(
        [
            [0, 1, 1, 2],
            [0, 2, 1, 2],
            [0, 1, 2, 2],
            [0, 2, 2, 2],
        ],
        dtype=mx.int32,
    )
    attrs = mx.array(
        [
            [1.0, 0.0, 0.0, 0.1, 0.2, 1.0],
            [0.0, 1.0, 0.0, 0.2, 0.4, 1.0],
            [0.0, 0.0, 1.0, 0.3, 0.6, 1.0],
            [1.0, 1.0, 0.0, 0.4, 0.8, 1.0],
        ],
        dtype=mx.float32,
    )
    return coords, attrs


def _fixture_bake() -> Trellis2TextureBakeResult:
    coords, attrs = _fixture_texture_field()
    return bake_trellis2_texture_fields(_fixture_mesh(), coords, attrs, decode_resolution=4, texture_size=16)


def _glb_json(payload: bytes) -> dict:
    magic, version, total_length = struct.unpack_from("<III", payload, 0)
    assert magic == 0x46546C67
    assert version == 2
    assert total_length == len(payload)
    json_length, json_type = struct.unpack_from("<I4s", payload, 12)
    assert json_type == b"JSON"
    document = payload[20 : 20 + json_length].rstrip(b" ")
    return json.loads(document.decode("utf-8"))


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


def test_validate_export_path_can_require_glb_for_textured_command(tmp_path):
    outputs = tmp_path / "outputs"

    path = validate_trellis2_export_path(outputs / "trellis2/demo.glb", outputs_root=outputs, suffixes=(".glb",))

    assert path == (outputs / "trellis2/demo.glb").resolve()
    with pytest.raises(ValueError, match="supported suffixes are \\('.glb',\\)"):
        validate_trellis2_export_path(outputs / "trellis2/demo.obj", outputs_root=outputs, suffixes=(".glb",))


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
    with pytest.raises(ValueError, match="at least one token"):
        sparse_coordinates_to_obj_payload(mx.array([], dtype=mx.int32).reshape((0, 4)))


def test_face_atlas_uvs_duplicates_vertices_and_stays_in_unit_square():
    mesh = _fixture_mesh()

    vertices, faces, uvs = make_trellis2_face_atlas_uvs(mesh)

    assert vertices.shape == (6, 3)
    assert faces.shape == (2, 3)
    assert uvs.shape == (6, 2)
    assert np.all(uvs >= 0.0)
    assert np.all(uvs <= 1.0)
    np.testing.assert_array_equal(faces, np.array([[0, 1, 2], [3, 4, 5]]))


def test_mac_export_dependency_gate_reports_missing_modules(monkeypatch):
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name):
        if name == "xatlas":
            return None
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    assert missing_trellis2_mac_export_dependencies() == ("xatlas",)
    with pytest.raises(ValueError, match="missing xatlas"):
        ensure_trellis2_mac_export_dependencies()


def test_postprocess_parity_audit_records_known_source_gaps():
    audit = trellis2_postprocess_parity_audit()

    assert all(isinstance(item, Trellis2PostprocessParityItem) for item in audit)
    assert any(item.stage == "texture sampling" and "trilinear" in item.mlx_spatial for item in audit)
    assert any(item.stage == "remeshing" and item.parity == "missing" for item in audit)


def test_resolve_xatlas_face_guard_adapts_to_postprocessed_face_count():
    assert resolve_trellis2_xatlas_face_guard(80_209, "auto") == 125_000
    assert resolve_trellis2_xatlas_face_guard(160_000, "auto") == 240_000
    assert resolve_trellis2_xatlas_face_guard(250_000, "auto") == 300_000
    assert resolve_trellis2_xatlas_face_guard(80_209, 200_000) == 200_000
    with pytest.raises(ValueError, match="positive integer"):
        resolve_trellis2_xatlas_face_guard(80_209, "bad")


def test_postprocess_mesh_for_glb_removes_bad_faces_components_and_fills_small_holes():
    mesh = FlexibleDualGridMesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.005, 0.0, 0.0],
                [0.0, 0.005, 0.0],
                [0.0, 0.0, 0.005],
                [0.5, 0.5, 0.5],
                [0.51, 0.5, 0.5],
                [0.5, 0.51, 0.5],
                [0.8, 0.8, 0.8],
            ],
            dtype=np.float32,
        ),
        faces=np.array(
            [
                [0, 1, 3],
                [1, 2, 3],
                [2, 0, 3],
                [2, 0, 3],
                [0, 0, 1],
                [4, 5, 6],
            ],
            dtype=np.int64,
        ),
    )

    result = postprocess_trellis2_mesh_for_glb(mesh, target_faces=100, min_component_faces=2)

    assert result.stats.duplicate_faces_removed == 1
    assert result.stats.degenerate_faces_removed == 1
    assert result.stats.components_removed == 1
    assert result.stats.component_faces_removed == 1
    assert result.stats.hole_fill.filled_loops == 1
    assert result.stats.hole_fill.faces_added == 3
    assert result.stats.unreferenced_vertices_removed > 0
    assert result.mesh.faces.shape[0] == 6
    assert result.stats.final_faces == 6
    assert result.stats.boundary_edges == 0
    assert result.source_mesh is not None
    assert result.source_mesh.faces.shape[0] == result.stats.cleaned_faces


def test_texture_bake_rejects_bad_texture_field_shapes():
    mesh = _fixture_mesh()
    coords, attrs = _fixture_texture_field()

    with pytest.raises(ValueError, match="texture coordinates must have shape"):
        bake_trellis2_texture_fields(mesh, coords[:, :3], attrs, texture_size=16)
    with pytest.raises(ValueError, match="currently supports only batch index 0"):
        bake_trellis2_texture_fields(mesh, mx.array([[1, 1, 1, 1]], dtype=mx.int32), attrs[:1], texture_size=16)
    with pytest.raises(ValueError, match="texture attributes must have shape"):
        bake_trellis2_texture_fields(mesh, coords, attrs[:, :5], texture_size=16)


def test_texture_bake_rejects_non_finite_attributes():
    coords, attrs = _fixture_texture_field()
    bad_attrs = np.array(attrs, dtype=np.float32)
    bad_attrs[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        bake_trellis2_texture_fields(_fixture_mesh(), coords, bad_attrs, texture_size=16)


def test_texture_bake_rejects_coordinate_range_violations():
    mesh = _fixture_mesh()
    coords, attrs = _fixture_texture_field()

    bad_negative = mx.array([[0, -1, 0, 0]], dtype=mx.int32)
    with pytest.raises(ValueError, match="non-negative"):
        bake_trellis2_texture_fields(mesh, bad_negative, attrs[:1], decode_resolution=4, texture_size=16)

    bad_high = mx.array([[0, 4, 0, 0]], dtype=mx.int32)
    with pytest.raises(ValueError, match="decode_resolution"):
        bake_trellis2_texture_fields(mesh, bad_high, attrs[:1], decode_resolution=4, texture_size=16)

    with pytest.raises(ValueError, match="unique"):
        bake_trellis2_texture_fields(
            mesh,
            mx.concatenate([coords[:2], coords[:2]], axis=0),
            mx.concatenate([attrs[:2], attrs[:2]], axis=0),
            texture_size=16,
        )


def test_texture_bake_rejects_invalid_mesh_and_size():
    coords, attrs = _fixture_texture_field()
    empty_mesh = FlexibleDualGridMesh(
        vertices=np.zeros((0, 3), dtype=np.float32),
        faces=np.zeros((0, 3), dtype=np.int64),
    )

    with pytest.raises(ValueError, match="mesh must contain vertices and faces"):
        bake_trellis2_texture_fields(empty_mesh, coords, attrs, texture_size=16)
    with pytest.raises(ValueError, match="texture_size must be positive"):
        bake_trellis2_texture_fields(_fixture_mesh(), coords, attrs, texture_size=0)


def test_texture_bake_rejects_large_public_allocations_before_rasterizing():
    coords, attrs = _fixture_texture_field()

    with pytest.raises(ValueError, match="above guard"):
        bake_trellis2_texture_fields(
            _fixture_mesh(),
            coords,
            attrs,
            texture_size=32,
            max_texture_pixels=128,
        )


def test_texture_bake_spatial_hash_matches_dense_sampler_when_pair_guard_exceeded():
    coords, attrs = _fixture_texture_field()
    dense = bake_trellis2_texture_fields(
        _fixture_mesh(),
        coords,
        attrs,
        decode_resolution=4,
        texture_size=16,
        k_neighbors=3,
        max_query_voxel_pairs=1_000_000,
    )
    hashed = bake_trellis2_texture_fields(
        _fixture_mesh(),
        coords,
        attrs,
        decode_resolution=4,
        texture_size=16,
        k_neighbors=3,
        max_query_voxel_pairs=1,
    )

    np.testing.assert_array_equal(hashed.base_color_rgba, dense.base_color_rgba)
    np.testing.assert_array_equal(hashed.metallic_roughness, dense.metallic_roughness)


def test_texture_bake_produces_deterministic_uvs_and_nonconstant_images():
    mesh = _fixture_mesh()
    coords, attrs = _fixture_texture_field()

    first = bake_trellis2_texture_fields(mesh, coords, attrs, decode_resolution=4, texture_size=16, k_neighbors=2)
    second = bake_trellis2_texture_fields(mesh, coords, attrs, decode_resolution=4, texture_size=16, k_neighbors=2)

    assert isinstance(first, Trellis2TextureBakeResult)
    assert first.vertices.shape == (6, 3)
    assert first.faces.shape == (2, 3)
    assert first.uvs.shape == (6, 2)
    assert first.base_color_rgba.shape == (16, 16, 4)
    assert first.metallic_roughness.shape == (16, 16, 3)
    assert first.coverage_ratio > 0.05
    assert first.voxel_count == 4
    assert first.voxel_size == 0.25
    assert first.origin == (-0.5, -0.5, -0.5)
    np.testing.assert_array_equal(first.uvs, second.uvs)
    np.testing.assert_array_equal(first.base_color_rgba, second.base_color_rgba)
    covered = first.base_color_rgba[first.coverage_mask]
    assert np.unique(covered[:, :3], axis=0).shape[0] > 1
    assert np.all(first.base_color_rgba[:, :, 3] == 255)
    assert np.count_nonzero(first.base_color_rgba[~first.coverage_mask, :3].sum(axis=-1)) > 0
    assert np.count_nonzero(first.metallic_roughness[:, :, 1]) > 0
    assert np.count_nonzero(first.metallic_roughness[:, :, 2]) > 0


def _small_dense_texture_grid():
    rows = []
    attrs = []
    for z in range(2):
        for y in range(2):
            for x in range(2):
                rows.append([0, z, y, x])
                attrs.append([float(z + y * 2 + x * 4)])
    return np.array(rows, dtype=np.int32), np.array(attrs, dtype=np.float32)


def test_sparse_trilinear_sampling_matches_dense_reference():
    coords, attrs = _small_dense_texture_grid()
    queries = np.array([[-0.5, -0.5, -0.5], [-0.25, -0.25, -0.25]], dtype=np.float32)

    result = sample_trellis2_sparse_trilinear_attributes(queries, coords, attrs, decode_resolution=2)

    assert isinstance(result, Trellis2SparseTrilinearSampleResult)
    np.testing.assert_allclose(result.attributes[:, 0], np.array([0.0, 3.5], dtype=np.float32))
    np.testing.assert_array_equal(result.valid_mask, np.array([True, True]))
    assert result.sampled_texel_count == 2
    assert result.missing_texel_count == 0
    assert result.out_of_grid_texel_count == 0


def test_sparse_trilinear_sampling_reports_missing_required_corners():
    coords, attrs = _small_dense_texture_grid()
    coords = coords[:-1]
    attrs = attrs[:-1]
    queries = np.array([[-0.25, -0.25, -0.25]], dtype=np.float32)

    result = sample_trellis2_sparse_trilinear_attributes(queries, coords, attrs, decode_resolution=2)

    assert result.valid_mask.tolist() == [True]
    assert result.missing_texel_count == 1
    with pytest.raises(ValueError, match="requires all non-zero interpolation corners"):
        sample_trellis2_sparse_trilinear_attributes(
            queries,
            coords,
            attrs,
            decode_resolution=2,
            require_all_corners=True,
        )


def test_mac_native_texture_bake_defaults_to_xatlas_trilinear_and_fills_texture():
    mesh = _fixture_mesh()
    coords, attrs = _fixture_texture_field()

    baked = bake_trellis2_texture_fields_mac_native(mesh, coords, attrs, decode_resolution=4, texture_size=16)

    assert baked.backend == "xatlas-trilinear"
    assert baked.vertices.shape[0] == baked.uvs.shape[0]
    assert baked.faces.shape == (2, 3)
    assert baked.base_color_rgba.shape == (16, 16, 4)
    assert baked.metallic_roughness.shape == (16, 16, 3)
    assert baked.raw_coverage_ratio is not None
    assert 0.0 < baked.raw_coverage_ratio < 1.0
    assert baked.coverage_ratio == 1.0
    assert baked.sampled_texel_count > 0
    assert baked.missing_texel_count >= 0
    assert baked.out_of_grid_texel_count >= 0
    assert baked.source_projection_used is False
    assert np.count_nonzero(baked.base_color_rgba[:, :, 3]) == baked.base_color_rgba[:, :, 3].size
    assert np.count_nonzero(baked.base_color_rgba[:, :, :3]) > 0


def test_mac_native_texture_bake_keeps_kdtree_as_explicit_debug_backend():
    mesh = _fixture_mesh()
    coords, attrs = _fixture_texture_field()

    baked = bake_trellis2_texture_fields_mac_native(
        mesh,
        coords,
        attrs,
        decode_resolution=4,
        texture_size=16,
        texture_bake_backend="kdtree",
    )

    assert baked.backend == "xatlas-kdtree"
    assert baked.sampled_texel_count > 0
    assert baked.coverage_ratio == 1.0


def test_xatlas_chunked_unwrap_reports_parallel_stats():
    result = unwrap_trellis2_mesh_xatlas_with_stats(_fixture_mesh(), parallel_chunks=2)

    assert isinstance(result, Trellis2XAtlasUnwrapResult)
    assert isinstance(result.stats, Trellis2XAtlasUnwrapStats)
    assert result.stats.backend == "xatlas-parallel-spatial"
    assert result.stats.chunks == 2
    assert result.stats.input_faces == 2
    assert result.stats.output_faces == 2
    assert result.vertices.shape[0] == result.uvs.shape[0]
    assert result.faces.shape == (2, 3)
    assert result.stats.chart_count is not None
    assert result.stats.elapsed_seconds >= 0.0
    assert np.all(result.uvs >= 0.0)
    assert np.all(result.uvs <= 1.0)


def test_mac_native_texture_bake_can_request_chunked_xatlas():
    mesh = _fixture_mesh()
    coords, attrs = _fixture_texture_field()

    baked = bake_trellis2_texture_fields_mac_native(
        mesh,
        coords,
        attrs,
        decode_resolution=4,
        texture_size=16,
        xatlas_parallel_chunks=2,
    )

    assert baked.backend == "xatlas-trilinear"
    assert baked.unwrap_backend == "xatlas-parallel-spatial"
    assert baked.unwrap_chunks == 2
    assert baked.unwrap_seconds is not None
    assert baked.unwrap_chart_count is not None
    assert baked.coverage_ratio == 1.0


def test_texture_bake_is_stable_under_voxel_order_permutation():
    mesh = _fixture_mesh()
    coords, attrs = _fixture_texture_field()
    permutation = mx.array([2, 0, 3, 1], dtype=mx.int32)

    original = bake_trellis2_texture_fields(mesh, coords, attrs, decode_resolution=4, texture_size=16, k_neighbors=3)
    shuffled = bake_trellis2_texture_fields(
        mesh,
        coords[permutation],
        attrs[permutation],
        decode_resolution=4,
        texture_size=16,
        k_neighbors=3,
    )

    np.testing.assert_array_equal(original.base_color_rgba, shuffled.base_color_rgba)
    np.testing.assert_array_equal(original.metallic_roughness, shuffled.metallic_roughness)


def test_texture_png_payload_encodes_baked_image():
    coords, attrs = _fixture_texture_field()
    baked = bake_trellis2_texture_fields(_fixture_mesh(), coords, attrs, decode_resolution=4, texture_size=8)

    payload = trellis2_texture_png_payload(baked.base_color_rgba)

    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(payload) > 64


def test_textured_glb_payload_contains_mesh_material_and_embedded_images():
    payload = trellis2_textured_glb_payload(_fixture_bake())
    document = _glb_json(payload)

    assert payload.startswith(b"glTF")
    assert document["asset"]["version"] == "2.0"
    assert document["buffers"][0]["byteLength"] > 0
    assert len(document["bufferViews"]) == 5
    assert len(document["accessors"]) == 3
    primitive = document["meshes"][0]["primitives"][0]
    assert primitive["attributes"] == {"POSITION": 0, "TEXCOORD_0": 1}
    assert primitive["indices"] == 2
    material = document["materials"][0]["pbrMetallicRoughness"]
    assert material["baseColorTexture"] == {"index": 0}
    assert material["metallicRoughnessTexture"] == {"index": 1}
    assert document["images"][0]["mimeType"] == "image/png"
    assert document["images"][1]["mimeType"] == "image/png"


def test_textured_glb_payload_rejects_non_finite_geometry():
    baked = _fixture_bake()
    bad_vertices = baked.vertices.copy()
    bad_vertices[0, 0] = np.inf
    bad_bake = Trellis2TextureBakeResult(
        vertices=bad_vertices,
        faces=baked.faces,
        uvs=baked.uvs,
        base_color_rgba=baked.base_color_rgba,
        metallic_roughness=baked.metallic_roughness,
        coverage_mask=baked.coverage_mask,
        texture_size=baked.texture_size,
        voxel_count=baked.voxel_count,
        k_neighbors=baked.k_neighbors,
        origin=baked.origin,
        voxel_size=baked.voxel_size,
    )

    with pytest.raises(ValueError, match="finite"):
        trellis2_textured_glb_payload(bad_bake)


def test_textured_glb_payload_rejects_bad_texture_dtype():
    baked = _fixture_bake()
    bad_bake = Trellis2TextureBakeResult(
        vertices=baked.vertices,
        faces=baked.faces,
        uvs=baked.uvs,
        base_color_rgba=baked.base_color_rgba.astype(np.float32),
        metallic_roughness=baked.metallic_roughness,
        coverage_mask=baked.coverage_mask,
        texture_size=baked.texture_size,
        voxel_count=baked.voxel_count,
        k_neighbors=baked.k_neighbors,
        origin=baked.origin,
        voxel_size=baked.voxel_size,
    )

    with pytest.raises(ValueError, match="uint8"):
        trellis2_textured_glb_payload(bad_bake)


def test_write_textured_glb_writes_under_outputs(tmp_path):
    output = tmp_path / "outputs/trellis2/fixture-textured.glb"

    artifact = write_trellis2_textured_glb(_fixture_bake(), output, outputs_root=tmp_path / "outputs")

    assert artifact.path == output.resolve()
    assert artifact.format == "glb"
    assert artifact.bytes_written == output.stat().st_size
    assert artifact.detail == "wrote TRELLIS.2 textured GLB"
    assert output.read_bytes().startswith(b"glTF")


def test_write_textured_glb_validates_output_path_before_payload(tmp_path):
    baked = _fixture_bake()
    bad_vertices = baked.vertices.copy()
    bad_vertices[0, 0] = np.inf
    bad_bake = Trellis2TextureBakeResult(
        vertices=bad_vertices,
        faces=baked.faces,
        uvs=baked.uvs,
        base_color_rgba=baked.base_color_rgba,
        metallic_roughness=baked.metallic_roughness,
        coverage_mask=baked.coverage_mask,
        texture_size=baked.texture_size,
        voxel_count=baked.voxel_count,
        k_neighbors=baked.k_neighbors,
        origin=baked.origin,
        voxel_size=baked.voxel_size,
    )

    with pytest.raises(ValueError, match="must stay under"):
        write_trellis2_textured_glb(bad_bake, tmp_path / "outside/fixture.glb", outputs_root=tmp_path / "outputs")


@pytest.mark.skipif(shutil.which("blender") is None, reason="Blender is not installed")
def test_textured_glb_fixture_imports_in_blender(tmp_path):
    output = tmp_path / "outputs/trellis2/fixture-textured.glb"
    write_trellis2_textured_glb(_fixture_bake(), output, outputs_root=tmp_path / "outputs")
    blender = shutil.which("blender")
    expression = (
        "import bpy; "
        f"p={str(output)!r}; "
        "bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(); "
        "[bpy.data.materials.remove(m) for m in list(bpy.data.materials)]; "
        "[bpy.data.images.remove(i) for i in list(bpy.data.images)]; "
        "bpy.ops.import_scene.gltf(filepath=p); "
        "print('GLB_OK', "
        "len([o for o in bpy.context.scene.objects if o.type=='MESH']), "
        "len(bpy.data.materials), "
        "len(bpy.data.images))"
    )

    result = subprocess.run(
        [blender, "--background", "--python-expr", expression],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    glb_line = next(line for line in result.stdout.splitlines() if line.startswith("GLB_OK "))
    _, mesh_count, material_count, image_count = glb_line.split()
    assert int(mesh_count) >= 1
    assert int(material_count) >= 1
    assert int(image_count) >= 1


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
    assert mlx_spatial.TRELLIS2_GLB_DEFAULT_FACE_TARGET == TRELLIS2_GLB_DEFAULT_FACE_TARGET
    assert mlx_spatial.TRELLIS2_TEXTURE_BAKE_BACKENDS == TRELLIS2_TEXTURE_BAKE_BACKENDS
    assert mlx_spatial.TRELLIS2_XATLAS_FACE_GUARD == TRELLIS2_XATLAS_FACE_GUARD
    assert mlx_spatial.Trellis2ExportArtifact is Trellis2ExportArtifact
    assert mlx_spatial.Trellis2ExportResult is Trellis2ExportResult
    assert mlx_spatial.Trellis2PostprocessParityItem is Trellis2PostprocessParityItem
    assert mlx_spatial.Trellis2SparseTrilinearSampleResult is Trellis2SparseTrilinearSampleResult
    assert mlx_spatial.Trellis2TextureBakeResult is Trellis2TextureBakeResult
    assert mlx_spatial.Trellis2XAtlasUnwrapResult is Trellis2XAtlasUnwrapResult
    assert mlx_spatial.Trellis2XAtlasUnwrapStats is Trellis2XAtlasUnwrapStats
    assert mlx_spatial.assess_trellis2_export_boundary is assess_trellis2_export_boundary
    assert mlx_spatial.bake_trellis2_texture_fields_mac_native is bake_trellis2_texture_fields_mac_native
    assert mlx_spatial.bake_trellis2_texture_fields is bake_trellis2_texture_fields
    assert mlx_spatial.ensure_trellis2_mac_export_dependencies is ensure_trellis2_mac_export_dependencies
    assert mlx_spatial.make_trellis2_face_atlas_uvs is make_trellis2_face_atlas_uvs
    assert mlx_spatial.missing_trellis2_mac_export_dependencies is missing_trellis2_mac_export_dependencies
    assert mlx_spatial.postprocess_trellis2_mesh_for_glb is postprocess_trellis2_mesh_for_glb
    assert mlx_spatial.resolve_trellis2_xatlas_face_guard is resolve_trellis2_xatlas_face_guard
    assert mlx_spatial.sample_trellis2_sparse_trilinear_attributes is sample_trellis2_sparse_trilinear_attributes
    assert mlx_spatial.sparse_coordinates_to_obj_payload is sparse_coordinates_to_obj_payload
    assert mlx_spatial.trellis2_postprocess_parity_audit is trellis2_postprocess_parity_audit
    assert mlx_spatial.trellis2_textured_glb_payload is trellis2_textured_glb_payload
    assert mlx_spatial.trellis2_texture_png_payload is trellis2_texture_png_payload
    assert mlx_spatial.unwrap_trellis2_mesh_xatlas_with_stats is unwrap_trellis2_mesh_xatlas_with_stats
    assert mlx_spatial.validate_trellis2_export_path is validate_trellis2_export_path
    assert mlx_spatial.write_sparse_coordinate_preview_obj is write_sparse_coordinate_preview_obj
    assert mlx_spatial.write_trellis2_textured_glb is write_trellis2_textured_glb
    assert mlx_spatial.write_trellis2_export_artifact is write_trellis2_export_artifact
