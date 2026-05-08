import numpy as np

from mlx_spatial.sam3d_export import (
    SAM3D_GAUSSIAN_PLY_FIELDS,
    SAM3D_XATLAS_FACE_GUARD,
    SAM3D_SH_C0,
    Sam3dGaussianTextureBakeResult,
    Sam3dGaussianTextureBakeStats,
    bake_sam3d_gaussian_texture_for_glb,
    compute_sam3d_vertex_normals,
    pack_sam3d_gaussian_rows,
    postprocess_sam3d_mesh_for_glb,
    read_sam3d_gaussian_ply_vertex_count,
    sam3d_basic_glb_payload,
    sam3d_binary_row_size,
    sam3d_gaussian_dc_to_rgb,
    sam3d_textured_glb_payload,
    write_sam3d_basic_glb,
    write_sam3d_textured_glb,
    write_sam3d_gaussians_ply,
)
from mlx_spatial.sam3d_mesh import assemble_sam3d_mesh_fields, extract_sam3d_mesh_from_features


def _fixture_gaussians():
    return {
        "xyz": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        "features_dc": np.array([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]], dtype=np.float32),
        "opacity": np.array([[0.7], [0.8]], dtype=np.float32),
        "scale": np.array([[0.01, 0.02, 0.03], [0.03, 0.02, 0.01]], dtype=np.float32),
        "rotation": np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
    }


def _interior_mesh_features():
    coords = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
        ],
        dtype=np.int32,
    )
    feats = np.zeros((8, 101), dtype=np.float32)
    feats[:, :8] = 2.0
    for row, coord in enumerate(coords[:, 1:]):
        local_corner = tuple(1 - coord)
        corner_index = int(local_corner[0] + local_corner[1] * 2 + local_corner[2] * 4)
        feats[row, corner_index] = -2.0
    feats[:, 53:101] = 0.25
    return coords, feats


def test_pack_sam3d_gaussian_rows_uses_official_column_order():
    rows = pack_sam3d_gaussian_rows(**_fixture_gaussians())

    assert rows.shape == (2, len(SAM3D_GAUSSIAN_PLY_FIELDS))
    assert rows[0, :3].tolist() == [0.10000000149011612, 0.20000000298023224, 0.30000001192092896]
    assert rows[0, 3:6].tolist() == [0.0, 0.0, 0.0]
    assert rows[0, 6:9].tolist() == [1.0, 0.0, 0.0]
    assert rows[0, 9] == np.float32(0.7)


def test_write_sam3d_gaussians_ply_writes_binary_official_fields(tmp_path):
    path = tmp_path / "gaussians.ply"

    stats = write_sam3d_gaussians_ply(path, **_fixture_gaussians())
    payload = path.read_bytes()

    assert stats.vertex_count == 2
    assert stats.format == "binary_little_endian"
    assert stats.fields == SAM3D_GAUSSIAN_PLY_FIELDS
    assert b"format binary_little_endian 1.0\n" in payload
    assert b"property float f_dc_0\n" in payload
    assert b"property float opacity\n" in payload
    assert b"property float scale_2\n" in payload
    assert b"property float rot_3\n" in payload
    assert read_sam3d_gaussian_ply_vertex_count(path) == 2
    assert stats.bytes_written == len(payload)
    assert len(payload.split(b"end_header\n", 1)[1]) == 2 * sam3d_binary_row_size()


def test_write_sam3d_gaussians_ply_can_write_ascii_debug_output(tmp_path):
    path = tmp_path / "gaussians_ascii.ply"

    stats = write_sam3d_gaussians_ply(path, binary=False, **_fixture_gaussians())

    text = path.read_text(encoding="utf-8")
    assert stats.format == "ascii"
    assert "format ascii 1.0" in text
    assert "element vertex 2" in text


def test_write_sam3d_gaussians_ply_uses_atomic_replace_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "gaussians.ply"

    import mlx_spatial.sam3d_export as sam3d_export

    def fail_replace(_src, _dst):
        raise OSError("replace failed")

    monkeypatch.setattr(sam3d_export.os, "replace", fail_replace)

    try:
        write_sam3d_gaussians_ply(path, **_fixture_gaussians())
    except OSError as error:
        assert "replace failed" in str(error)
    else:
        raise AssertionError("expected atomic replace failure to propagate")

    assert not path.exists()
    assert not (tmp_path / ".gaussians.ply.tmp").exists()


def test_write_sam3d_basic_glb_writes_blender_readable_container(tmp_path):
    path = tmp_path / "mesh.glb"
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    stats = write_sam3d_basic_glb(path, vertices=vertices, faces=faces, colors=colors)
    payload = path.read_bytes()

    assert payload[:4] == b"glTF"
    assert b"JSON" in payload[:32]
    assert b"BIN\x00" in payload
    assert b"POSITION" in payload
    assert b"NORMAL" in payload
    assert b"COLOR_0" in payload
    assert stats.vertex_count == 3
    assert stats.face_count == 1
    assert stats.has_vertex_color is True
    assert stats.has_normals is True
    assert stats.has_texture is False
    assert stats.bytes_written == len(payload)


def test_write_sam3d_basic_glb_uses_atomic_replace_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "mesh.glb"
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    import mlx_spatial.sam3d_export as sam3d_export

    def fail_replace(_src, _dst):
        raise OSError("replace failed")

    monkeypatch.setattr(sam3d_export.os, "replace", fail_replace)

    try:
        write_sam3d_basic_glb(path, vertices=vertices, faces=faces)
    except OSError as error:
        assert "replace failed" in str(error)
    else:
        raise AssertionError("expected atomic replace failure to propagate")

    assert not path.exists()
    assert not (tmp_path / ".mesh.glb.tmp").exists()


def test_sam3d_basic_glb_payload_rejects_invalid_faces():
    vertices = np.zeros((3, 3), dtype=np.float32)
    faces = np.array([[0, 1, 3]], dtype=np.int64)

    try:
        sam3d_basic_glb_payload(vertices=vertices, faces=faces)
    except ValueError as error:
        assert "outside the vertex array" in str(error)
    else:
        raise AssertionError("expected invalid face indices to fail")


def test_sam3d_mesh_field_guard_returns_blocker_without_writing_artifact(tmp_path):
    path = tmp_path / "mesh.glb"
    coords = np.array([[0, 0, 0, 0]], dtype=np.int32)
    feats = np.zeros((1, 53), dtype=np.float32)

    result = assemble_sam3d_mesh_fields(
        coords,
        feats,
        extraction_resolution=16,
        max_dense_bytes=1,
    )

    assert result.ready is False
    assert result.fields is None
    assert result.blocker is not None
    assert result.blocker.stage == "mesh-decoder"
    assert result.blocker.metadata["estimated_dense_bytes"] > result.blocker.metadata["max_dense_bytes"]
    assert not path.exists()


def test_write_sam3d_basic_glb_accepts_extracted_mesh_output(tmp_path):
    path = tmp_path / "mesh.glb"
    coords, feats = _interior_mesh_features()
    mesh = extract_sam3d_mesh_from_features(coords, feats, extraction_resolution=2, use_color=True)

    assert mesh.ready is True
    assert mesh.vertices is not None
    assert mesh.faces is not None
    stats = write_sam3d_basic_glb(path, vertices=mesh.vertices, faces=mesh.faces, colors=mesh.colors)

    assert stats.vertex_count == mesh.vertices.shape[0]
    assert stats.face_count == mesh.faces.shape[0]
    assert stats.has_vertex_color is True
    assert path.read_bytes()[:4] == b"glTF"


def test_postprocess_sam3d_mesh_for_glb_removes_bad_faces_components_and_compacts_colors():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 10.0, 10.0],
            [11.0, 10.0, 10.0],
            [10.0, 11.0, 10.0],
            [99.0, 99.0, 99.0],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [2, 1, 0],
            [0, 0, 1],
            [0, 1, 99],
            [4, 5, 6],
        ],
        dtype=np.int64,
    )
    colors = np.linspace(0.0, 1.0, vertices.size, dtype=np.float32).reshape(vertices.shape)

    result = postprocess_sam3d_mesh_for_glb(
        vertices,
        faces,
        colors=colors,
        target_faces=0,
        simplify=False,
        min_component_faces=2,
    )

    assert result.vertices.shape == (4, 3)
    assert result.faces.shape == (2, 3)
    assert result.colors is not None
    assert result.colors.shape == result.vertices.shape
    assert result.normals.shape == result.vertices.shape
    assert result.stats.invalid_faces_removed == 1
    assert result.stats.duplicate_faces_removed == 1
    assert result.stats.degenerate_faces_removed == 1
    assert result.stats.min_component_faces == 2
    assert result.stats.components_removed == 1
    assert result.stats.component_faces_removed == 1
    assert result.stats.components_after == 1
    assert result.stats.unreferenced_vertices_removed >= 4
    assert result.stats.boundary_edges == 4
    assert result.stats.nonmanifold_edges == 0
    assert result.stats.nonmanifold_edges_including_boundary == 4


def test_postprocess_sam3d_mesh_for_glb_fills_small_clean_hole():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [0.0, 0.001, 0.0],
            [0.0, 0.0, 0.001],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)

    result = postprocess_sam3d_mesh_for_glb(vertices, faces, target_faces=0, simplify=False, min_component_faces=1)

    assert result.stats.hole_fill.filled_loops == 1
    assert result.stats.hole_fill.vertices_added == 1
    assert result.stats.hole_fill.faces_added == 3
    assert result.vertices.shape[0] == 5
    assert result.faces.shape[0] == 6


def test_postprocess_sam3d_mesh_for_glb_can_simplify_large_mesh():
    size = 8
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, size), np.linspace(0.0, 1.0, size), indexing="ij")
    vertices = np.stack([xx.reshape(-1), yy.reshape(-1), np.zeros(size * size, dtype=np.float64)], axis=1).astype(np.float32)
    faces = []
    for y in range(size - 1):
        for x in range(size - 1):
            a = y * size + x
            b = a + 1
            c = a + size
            d = c + 1
            faces.append((a, b, d))
            faces.append((a, d, c))
    faces = np.asarray(faces, dtype=np.int64)

    result = postprocess_sam3d_mesh_for_glb(vertices, faces, target_faces=24, smooth_iterations=1, min_component_faces=1)

    assert result.stats.smoothed is True
    assert result.stats.simplified is True
    assert result.stats.components_after == 1
    assert result.faces.shape[0] < faces.shape[0]
    assert result.faces.shape[0] <= 24
    normals = compute_sam3d_vertex_normals(result.vertices, result.faces)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)


def test_sam3d_gaussian_dc_to_rgb_converts_sh_dc_preview_colors():
    features_dc = np.array([[[0.0, 0.0, 0.0]], [[1.0 / SAM3D_SH_C0, -1.0 / SAM3D_SH_C0, 0.5 / SAM3D_SH_C0]]], dtype=np.float32)

    colors = sam3d_gaussian_dc_to_rgb(features_dc)

    np.testing.assert_allclose(colors[0], [0.5, 0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(colors[1], [1.0, 0.0, 1.0], atol=1e-6)


def test_chunked_sam3d_gaussian_sampling_matches_unchunked():
    import mlx_spatial.sam3d_export as sam3d_export

    query = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=np.float32)
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    weights = np.array([1.0, 0.5], dtype=np.float32)
    radius = np.array([0.2, 0.2], dtype=np.float32)

    chunked = sam3d_export._sample_gaussian_colors_chunked(
        query,
        points,
        colors,
        weights,
        radius,
        k_neighbors=2,
        texel_chunk_size=1,
    )
    unchunked = sam3d_export._sample_gaussian_colors_chunked(
        query,
        points,
        colors,
        weights,
        radius,
        k_neighbors=2,
        texel_chunk_size=128,
    )

    np.testing.assert_allclose(chunked, unchunked, atol=1e-7)


def test_bake_sam3d_gaussian_texture_for_glb_produces_covered_texture():
    mesh = postprocess_sam3d_mesh_for_glb(
        np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.0, 0.7, 0.0]], dtype=np.float32),
        np.array([[0, 1, 2]], dtype=np.int64),
        target_faces=0,
        simplify=False,
        min_component_faces=1,
    )
    gaussian_xyz = np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.0, 0.7, 0.0]], dtype=np.float32)
    rgb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    features_dc = ((rgb - 0.5) / SAM3D_SH_C0).reshape(3, 1, 3).astype(np.float32)
    opacity = np.full((3, 1), 10.0, dtype=np.float32)
    scale = np.full((3, 3), np.log(0.15), dtype=np.float32)

    baked = bake_sam3d_gaussian_texture_for_glb(
        mesh,
        gaussian_xyz=gaussian_xyz,
        gaussian_features_dc=features_dc,
        gaussian_opacity=opacity,
        gaussian_scale=scale,
        texture_size=16,
        k_neighbors=3,
        texel_chunk_size=5,
    )

    assert baked.stats.backend == "gaussian-kdtree"
    assert baked.stats.sampled_texel_count > 0
    assert baked.stats.raw_coverage_ratio > 0.0
    assert baked.stats.final_coverage_ratio == 1.0
    assert baked.stats.xatlas_face_guard == SAM3D_XATLAS_FACE_GUARD
    assert baked.base_color_rgba.shape == (16, 16, 4)
    assert baked.base_color_rgba[..., :3].max() > baked.base_color_rgba[..., :3].min()
    assert baked.uvs.shape == (baked.vertices.shape[0], 2)


def test_bake_sam3d_gaussian_texture_for_glb_respects_xatlas_face_guard():
    mesh = postprocess_sam3d_mesh_for_glb(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        target_faces=0,
        simplify=False,
        min_component_faces=1,
    )
    gaussian_xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    features_dc = np.zeros((1, 1, 3), dtype=np.float32)
    opacity = np.ones((1, 1), dtype=np.float32)
    scale = np.zeros((1, 3), dtype=np.float32)

    try:
        bake_sam3d_gaussian_texture_for_glb(
            mesh,
            gaussian_xyz=gaussian_xyz,
            gaussian_features_dc=features_dc,
            gaussian_opacity=opacity,
            gaussian_scale=scale,
            texture_size=4,
            xatlas_face_guard=1,
        )
    except ValueError as error:
        assert "exceeds guard" in str(error)
    else:
        raise AssertionError("expected SAM3D texture bake to respect xatlas face guard")


def test_sam3d_textured_glb_payload_embeds_base_color_texture(tmp_path):
    baked = Sam3dGaussianTextureBakeResult(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        normals=np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32),
        uvs=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        base_color_rgba=np.full((4, 4, 4), 255, dtype=np.uint8),
        coverage_mask=np.ones((4, 4), dtype=bool),
        stats=Sam3dGaussianTextureBakeStats(
            backend="gaussian-kdtree",
            texture_size=4,
            gaussian_count=3,
            k_neighbors=1,
            texel_chunk_size=4,
            sampled_texel_count=4,
            raster_texel_count=4,
            raw_coverage_ratio=1.0,
            final_coverage_ratio=1.0,
            unwrap_backend="fixture",
            xatlas_face_guard=SAM3D_XATLAS_FACE_GUARD,
            unwrap_seconds=0.0,
            unwrap_chunks=1,
            unwrap_chart_count=1,
            unwrap_utilization=1.0,
            elapsed_seconds=0.0,
        ),
    )

    payload = sam3d_textured_glb_payload(baked)
    path = tmp_path / "mesh.glb"
    stats = write_sam3d_textured_glb(path, baked)

    assert payload[:4] == b"glTF"
    assert b"TEXCOORD_0" in payload
    assert b"baseColorTexture" in payload
    assert b"image/png" in payload
    assert stats.has_texture is True
    assert stats.has_vertex_color is False
    assert path.read_bytes()[:4] == b"glTF"
