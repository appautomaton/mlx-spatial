import numpy as np
import mlx.core as mx

from mlx_spatial.sam3d_mesh import (
    Sam3dFlexiCubesDualVertexCandidates,
    Sam3dFlexiCubesSurfaceEdges,
    activate_sam3d_flexicubes_weights,
    assemble_sam3d_mesh_fields,
    compute_sam3d_flexicubes_case_ids,
    estimate_sam3d_flexicubes_bytes,
    extract_sam3d_mesh,
    extract_sam3d_mesh_from_features,
    extract_sam3d_flexicubes_surface_core,
    fill_sam3d_mesh_holes,
    get_deformed_sam3d_grid_vertices,
    identify_sam3d_flexicubes_surface_cubes,
    identify_sam3d_flexicubes_surface_edges,
    run_sparse_subdivide_block3d,
    sam3d_mesh_feature_layout,
    sam3d_dense_cube_vertex_indices,
    sparse_group_norm,
    sparse_cube2verts,
    sparse_subdivide3d,
    triangulate_sam3d_flexicubes_surface,
)
from mlx_spatial.sam3d_slat import Sam3dSparseTensor


def _sparse_tensor(coords: np.ndarray, feats: mx.array) -> Sam3dSparseTensor:
    return Sam3dSparseTensor(coords=coords, feats=feats, layout=(slice(0, coords.shape[0]),), spatial_cache={})


def _block_tensors(in_channels: int, out_channels: int, *, skip_projection: bool) -> dict[str, mx.array]:
    tensors = {
        "block.act_layers.0.weight": mx.ones((in_channels,), dtype=mx.float32),
        "block.act_layers.0.bias": mx.zeros((in_channels,), dtype=mx.float32),
        "block.out_layers.0.conv.weight": mx.zeros((out_channels, 3, 3, 3, in_channels), dtype=mx.float32),
        "block.out_layers.0.conv.bias": mx.zeros((out_channels,), dtype=mx.float32),
        "block.out_layers.1.weight": mx.ones((out_channels,), dtype=mx.float32),
        "block.out_layers.1.bias": mx.zeros((out_channels,), dtype=mx.float32),
        "block.out_layers.3.conv.weight": mx.zeros((out_channels, 3, 3, 3, out_channels), dtype=mx.float32),
        "block.out_layers.3.conv.bias": mx.zeros((out_channels,), dtype=mx.float32),
    }
    if skip_projection:
        skip = np.zeros((out_channels, 1, 1, 1, in_channels), dtype=np.float32)
        skip[0, 0, 0, 0, 0] = 1.0
        skip[1, 0, 0, 0, 1] = 1.0
        skip[2, 0, 0, 0, :] = 1.0
        tensors["block.skip_connection.conv.weight"] = mx.array(skip)
        tensors["block.skip_connection.conv.bias"] = mx.array([0.1, 0.2, 0.3], dtype=mx.float32)
    return tensors


def _interior_sign_change_fields(*, use_color: bool = False):
    res = 2
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
    feats = np.zeros((8, 101 if use_color else 53), dtype=np.float32)
    feats[:, :8] = 2.0
    for row, coord in enumerate(coords[:, 1:]):
        local_corner = tuple(1 - coord)
        corner_index = int(local_corner[0] + local_corner[1] * 2 + local_corner[2] * 4)
        feats[row, corner_index] = -2.0
    if use_color:
        color_start = 53
        raw_color = np.array([0.0, 2.0, -2.0, 1.0, -1.0, 0.5], dtype=np.float32)
        for corner in range(8):
            feats[:, color_start + corner * 6 : color_start + (corner + 1) * 6] = raw_color
    result = assemble_sam3d_mesh_fields(coords, feats, extraction_resolution=res, use_color=use_color)
    assert result.ready is True
    assert result.fields is not None
    return result.fields


def test_sparse_subdivide_expands_children_in_official_corner_order():
    tensor = _sparse_tensor(
        np.array([[0, 2, 3, 4], [0, 3, 3, 4]], dtype=np.int32),
        mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32),
    )

    out = sparse_subdivide3d(tensor)

    expected_first = np.array(
        [
            [0, 4, 6, 8],
            [0, 4, 6, 9],
            [0, 4, 7, 8],
            [0, 4, 7, 9],
            [0, 5, 6, 8],
            [0, 5, 6, 9],
            [0, 5, 7, 8],
            [0, 5, 7, 9],
        ],
        dtype=np.int32,
    )
    assert out.coords.shape == (16, 4)
    np.testing.assert_array_equal(out.coords[:8], expected_first)
    np.testing.assert_allclose(np.array(out.feats[:8]), np.repeat([[1.0, 2.0]], 8, axis=0))
    np.testing.assert_allclose(np.array(out.feats[8:]), np.repeat([[3.0, 4.0]], 8, axis=0))


def test_sparse_group_norm_handles_channels_below_32_with_affine():
    tensor = _sparse_tensor(
        np.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int32),
        mx.array([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]], dtype=mx.float32),
    )
    weight = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
    bias = mx.array([0.5, 1.0, 1.5, 2.0], dtype=mx.float32)

    out = sparse_group_norm(tensor, weight, bias, num_groups=32, eps=1e-5)

    scale = 1.0 / np.sqrt(1.0 + 1e-5)
    expected = np.array(
        [
            [0.5 - scale, 1.0 - 2.0 * scale, 1.5 - 3.0 * scale, 2.0 - 4.0 * scale],
            [0.5 + scale, 1.0 + 2.0 * scale, 1.5 + 3.0 * scale, 2.0 + 4.0 * scale],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.array(out.feats), expected, rtol=1e-5, atol=1e-5)


def test_sparse_subdivide_block_projects_skip_when_channels_change():
    tensor = _sparse_tensor(
        np.array([[0, 0, 0, 0]], dtype=np.int32),
        mx.array([[2.0, 3.0]], dtype=mx.float32),
    )

    out = run_sparse_subdivide_block3d(
        tensor,
        _block_tensors(2, 3, skip_projection=True),
        prefix="block.",
    )

    assert out.coords.shape == (8, 4)
    assert tuple(out.feats.shape) == (8, 3)
    np.testing.assert_allclose(np.array(out.feats), np.repeat([[2.1, 3.2, 5.3]], 8, axis=0), rtol=1e-6)


def test_sam3d_mesh_feature_layout_matches_official_channel_sizes():
    layout = sam3d_mesh_feature_layout(use_color=False)

    assert layout.ranges == {"sdf": (0, 8), "deform": (8, 32), "weights": (32, 53)}
    assert layout.shapes == {"sdf": (8, 1), "deform": (8, 3), "weights": (21,)}
    assert layout.total_channels == 53

    color_layout = sam3d_mesh_feature_layout(use_color=True)
    assert color_layout.ranges["color"] == (53, 101)
    assert color_layout.shapes["color"] == (8, 6)
    assert color_layout.total_channels == 101


def test_sparse_cube2verts_averages_shared_cube_corner_attributes():
    coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    attrs = np.zeros((2, 8, 1), dtype=np.float32)
    attrs[0, :, 0] = 1.0
    attrs[0, [1, 3, 5, 7], 0] = 10.0
    attrs[1, :, 0] = 2.0
    attrs[1, [0, 2, 4, 6], 0] = 20.0

    out = sparse_cube2verts(coords, attrs)

    by_coord = {tuple(coord): float(value[0]) for coord, value in zip(out.coords, out.attrs, strict=True)}
    assert by_coord[(0, 0, 0)] == 1.0
    assert by_coord[(2, 0, 0)] == 2.0
    assert by_coord[(1, 0, 0)] == 15.0
    assert by_coord[(1, 1, 1)] == 15.0
    assert out.cubes.shape == (2, 8)


def test_assemble_sam3d_mesh_fields_initializes_dense_fields_and_deforms_grid_vertices():
    res = 2
    coords = np.array([[0, 0, 0, 0]], dtype=np.int32)
    feats = np.zeros((1, 53), dtype=np.float32)
    feats[:, 32:53] = np.arange(21, dtype=np.float32)
    deform_start = 8
    corner_100_x = deform_start + 1 * 3
    feats[0, corner_100_x : corner_100_x + 3] = [1.0, 0.0, 0.0]

    result = assemble_sam3d_mesh_fields(coords, feats, extraction_resolution=res)

    assert result.ready is True
    assert result.fields is not None
    fields = result.fields
    assert fields.sdf.shape == ((res + 1) ** 3,)
    assert fields.weights.shape == (res**3, 21)
    np.testing.assert_allclose(fields.sdf[0], -0.5)
    np.testing.assert_allclose(fields.sdf[-1], 1.0)
    np.testing.assert_allclose(fields.weights[0], np.arange(21, dtype=np.float32))
    np.testing.assert_allclose(fields.weights[-1], np.zeros((21,), dtype=np.float32))

    vertex_index_100 = 1 * (res + 1) ** 2
    scale = (1.0 - 1e-8) / (res * 2.0)
    expected = np.array([0.0, -0.5, -0.5], dtype=np.float32) + np.array([scale * np.tanh(1.0), 0.0, 0.0])
    np.testing.assert_allclose(fields.deformed_vertices[vertex_index_100], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(fields.deform[vertex_index_100], [1.0, 0.0, 0.0])
    assert result.metadata["sdf_bias"] == -0.5


def test_assemble_sam3d_mesh_fields_averages_optional_color_attrs():
    res = 3
    coords = np.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int32)
    feats = np.zeros((2, 101), dtype=np.float32)
    color_start = 53
    feats[0, color_start + 1 * 6 : color_start + 2 * 6] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    feats[1, color_start + 0 * 6 : color_start + 1 * 6] = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0]

    result = assemble_sam3d_mesh_fields(coords, feats, extraction_resolution=res, use_color=True)

    assert result.ready is True
    assert result.fields is not None
    assert result.fields.colors is not None
    assert result.fields.colors.shape == ((res + 1) ** 3, 6)
    vertex_index_100 = 1 * (res + 1) ** 2
    np.testing.assert_allclose(
        result.fields.colors[vertex_index_100],
        [15.0, 25.0, 35.0, 45.0, 55.0, 65.0],
    )
    assert "dense_vertex_colors" not in result.fields.memory_estimate


def test_get_deformed_sam3d_grid_vertices_matches_upstream_formula():
    grid = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32)
    deform = np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.5]], dtype=np.float32)

    out = get_deformed_sam3d_grid_vertices(grid, deform, 2)

    expected = grid / 2 - 0.5 + (1.0 - 1e-8) / 4.0 * np.tanh(deform)
    np.testing.assert_allclose(out, expected.astype(np.float32), rtol=1e-6, atol=1e-6)


def test_flexicubes_surface_cube_detection_matches_sign_change_semantics():
    cube_idx = sam3d_dense_cube_vertex_indices(2)
    sdf = np.ones((3**3,), dtype=np.float32)
    sdf[0] = -1.0

    surface = identify_sam3d_flexicubes_surface_cubes(sdf, cube_idx)

    np.testing.assert_array_equal(surface.indices, np.array([0], dtype=np.int64))
    assert surface.mask.tolist() == [True, False, False, False, False, False, False, False]
    assert surface.occupancy[0].tolist() == [True, False, False, False, False, False, False, False]


def test_flexicubes_surface_edges_deduplicate_shared_grid_edges_and_mark_cube_edges():
    cube_idx = sam3d_dense_cube_vertex_indices(2)
    sdf = np.ones((3**3,), dtype=np.float32)
    sdf[9] = -1.0
    surface = identify_sam3d_flexicubes_surface_cubes(sdf, cube_idx)

    edges = identify_sam3d_flexicubes_surface_edges(sdf, cube_idx, surface.mask)

    shared_edge_matches = np.where(np.all(edges.edges == np.array([9, 10], dtype=np.int64), axis=1))[0]
    assert shared_edge_matches.shape == (1,)
    shared_edge = int(shared_edge_matches[0])
    assert np.count_nonzero(edges.cube_edge_indices == shared_edge) == 2
    assert edges.cube_edge_counts[edges.cube_edge_indices == shared_edge].tolist() == [2, 2]
    assert np.all(edges.cube_edge_indices[edges.cube_surface_edge_mask] >= 0)
    assert np.all(edges.cube_edge_indices[~edges.cube_surface_edge_mask] == -1)


def test_flexicubes_weight_activations_match_official_inference_rules():
    raw_weights = np.zeros((2, 21), dtype=np.float32)
    raw_weights[0, 0] = 1.0
    raw_weights[0, 12] = -1.0
    raw_weights[0, 20] = 2.0
    raw_weights[1, :] = 100.0
    surface_mask = np.array([True, False])

    weights = activate_sam3d_flexicubes_weights(raw_weights, surface_mask)

    np.testing.assert_allclose(weights.beta[0, 0], np.tanh(1.0) * 0.99 + 1.0, rtol=1e-6)
    np.testing.assert_allclose(weights.alpha[0, 0], np.tanh(-1.0) * 0.99 + 1.0, rtol=1e-6)
    np.testing.assert_allclose(weights.gamma[0], 1.0 / (1.0 + np.exp(-2.0)) * 0.99 + 0.005, rtol=1e-6)
    assert weights.beta.shape == (1, 12)
    assert weights.alpha.shape == (1, 8)


def test_flexicubes_case_ids_use_official_check_table_for_non_ambiguous_fixture():
    cube_idx = sam3d_dense_cube_vertex_indices(2)
    sdf = np.ones((3**3,), dtype=np.float32)
    sdf[0] = -1.0
    surface = identify_sam3d_flexicubes_surface_cubes(sdf, cube_idx)

    case_ids = compute_sam3d_flexicubes_case_ids(surface.occupancy, surface.mask, 2)

    np.testing.assert_array_equal(case_ids, np.array([1], dtype=np.int64))


def test_flexicubes_surface_core_produces_dual_vertex_candidates_without_triangulation():
    feats = np.zeros((1, 53), dtype=np.float32)
    feats[0, :8] = 2.0
    feats[0, 0] = 0.5
    result = assemble_sam3d_mesh_fields(
        np.array([[0, 0, 0, 0]], dtype=np.int32),
        feats,
        extraction_resolution=1,
    )
    assert result.fields is not None

    core = extract_sam3d_flexicubes_surface_core(result.fields, extraction_resolution=1)

    assert core.metadata["surface_cube_count"] == 1
    assert core.metadata["surface_edge_count"] == 3
    assert core.metadata["dual_vertex_candidate_count"] == 1
    assert core.case_ids.tolist() == [1]
    assert core.dual_vertices.vertices.shape == (1, 3)
    assert np.isfinite(core.dual_vertices.vertices).all()
    np.testing.assert_allclose(
        core.dual_vertices.vertices[0],
        [-0.3888889, -0.3888889, -0.3888889],
        rtol=1e-6,
        atol=1e-6,
    )


def test_extract_sam3d_mesh_triangulates_tiny_deterministic_sdf_field():
    fields = _interior_sign_change_fields()

    result = extract_sam3d_mesh(fields, extraction_resolution=2)

    assert result.ready is True
    assert result.blocker is None
    assert result.vertices is not None
    assert result.faces is not None
    assert result.vertices.shape[0] > 0
    assert result.faces.shape[0] > 0
    assert result.vertices.dtype == np.float32
    assert result.faces.dtype == np.int64
    assert np.isfinite(result.vertices).all()


def test_extract_sam3d_mesh_faces_reference_valid_vertices():
    fields = _interior_sign_change_fields()

    result = extract_sam3d_mesh(fields, extraction_resolution=2)

    assert result.vertices is not None
    assert result.faces is not None
    assert int(result.faces.min()) >= 0
    assert int(result.faces.max()) < result.vertices.shape[0]


def test_flexicubes_triangulation_rejects_incomplete_grouped_edges():
    surface_edges = Sam3dFlexiCubesSurfaceEdges(
        edges=np.array([[0, 1], [2, 3]], dtype=np.int64),
        cube_edge_indices=np.array([[0, 0, 0, 1], [1, 1, 1, 1]], dtype=np.int64),
        cube_edge_counts=np.full((2, 4), 4, dtype=np.int64),
        cube_surface_edge_mask=np.ones((2, 4), dtype=bool),
    )
    dual_vertices = Sam3dFlexiCubesDualVertexCandidates(
        vertices=np.zeros((8, 3), dtype=np.float32),
        gamma=np.ones((8,), dtype=np.float32),
        cube_edge_vertex_indices=np.arange(8, dtype=np.int64).reshape(2, 4),
        colors=None,
    )

    try:
        triangulate_sam3d_flexicubes_surface(
            np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32),
            surface_edges,
            dual_vertices,
        )
    except ValueError as error:
        assert "exactly four entries" in str(error)
    else:
        raise AssertionError("expected malformed FlexiCubes edge grouping to fail")


def test_extract_sam3d_mesh_sigmoid_activates_optional_colors_and_aligns_vertices():
    fields = _interior_sign_change_fields(use_color=True)

    result = extract_sam3d_mesh(fields, extraction_resolution=2)

    assert result.ready is True
    assert result.vertices is not None
    assert result.vertex_attrs is not None
    assert result.colors is not None
    assert result.vertex_attrs.shape == (result.vertices.shape[0], 6)
    assert result.colors.shape == result.vertices.shape
    expected = 1.0 / (1.0 + np.exp(-np.array([0.0, 2.0, -2.0], dtype=np.float32)))
    np.testing.assert_allclose(result.colors[0], expected, rtol=1e-6, atol=1e-6)


def test_extract_sam3d_mesh_no_surface_returns_structured_blocker():
    res = 2
    fields = _interior_sign_change_fields()
    no_surface = type(fields)(
        sdf=np.ones_like(fields.sdf),
        deform=fields.deform,
        weights=fields.weights,
        grid_vertices=fields.grid_vertices,
        deformed_vertices=fields.deformed_vertices,
        colors=fields.colors,
        memory_estimate=fields.memory_estimate,
    )

    result = extract_sam3d_mesh(no_surface, extraction_resolution=res)

    assert result.ready is False
    assert result.success is False
    assert result.vertices is None
    assert result.faces is None
    assert result.blocker is not None
    assert result.blocker.stage == "mesh-decoder"
    assert result.blocker.metadata["surface_cube_count"] == 0


def test_extract_sam3d_mesh_no_face_surface_returns_structured_blocker():
    feats = np.zeros((1, 53), dtype=np.float32)
    feats[0, :8] = 2.0
    feats[0, 0] = 0.5
    assembled = assemble_sam3d_mesh_fields(
        np.array([[0, 0, 0, 0]], dtype=np.int32),
        feats,
        extraction_resolution=1,
    )
    assert assembled.fields is not None

    result = extract_sam3d_mesh(assembled.fields, extraction_resolution=1)

    assert result.ready is False
    assert result.blocker is not None
    assert result.blocker.stage == "mesh-decoder"
    assert result.blocker.metadata["face_count"] == 0


def test_extract_sam3d_mesh_flexicubes_guard_returns_structured_blocker():
    fields = _interior_sign_change_fields()
    estimate = estimate_sam3d_flexicubes_bytes(2)

    result = extract_sam3d_mesh(fields, extraction_resolution=2, max_flexicubes_bytes=1)

    assert estimate["estimated_flexicubes_bytes"] > 1
    assert result.ready is False
    assert result.blocker is not None
    assert result.blocker.stage == "mesh-decoder"
    assert result.blocker.operation == "estimate SAM3D FlexiCubes intermediate arrays"
    assert result.blocker.metadata["estimated_flexicubes_bytes"] > result.blocker.metadata["max_flexicubes_bytes"]


def test_extract_sam3d_mesh_from_features_propagates_empty_surface_blocker():
    coords = np.array([[0, 0, 0, 0]], dtype=np.int32)
    feats = np.full((1, 53), 2.0, dtype=np.float32)

    result = extract_sam3d_mesh_from_features(coords, feats, extraction_resolution=1)

    assert result.ready is False
    assert result.blocker is not None
    assert result.blocker.stage == "mesh-decoder"


def test_fill_sam3d_mesh_holes_fills_simple_clean_boundary_loop():
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

    new_verts, new_faces, stats = fill_sam3d_mesh_holes(vertices, faces)

    assert stats["boundary_edges_before"] == 3
    assert stats["filled_loops"] == 1
    assert stats["faces_added"] == 3
    assert stats["vertices_added"] == 1
    assert new_verts.shape[0] == vertices.shape[0] + 1
    assert new_faces.shape[0] == faces.shape[0] + 3


def test_fill_sam3d_mesh_holes_skips_large_loop():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)

    new_verts, new_faces, stats = fill_sam3d_mesh_holes(vertices, faces, max_hole_area=1e-8)

    assert stats["filled_loops"] == 0
    assert stats["skipped_large"] == 1
    np.testing.assert_array_equal(new_faces, faces)


def test_fill_sam3d_mesh_holes_uses_camera_visibility_for_orientation():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.002, 0.0, 0.0],
            [0.0, 0.002, 0.0],
            [0.0, 0.0, 0.002],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
    camera_centers = np.array([[0.0, 0.0, 10.0]], dtype=np.float32)

    _, new_faces, stats = fill_sam3d_mesh_holes(vertices, faces, camera_centers=camera_centers)

    assert stats["filled_loops"] == 1
    assert new_faces.shape[0] == faces.shape[0] + 3


def test_fill_sam3d_mesh_holes_rejects_invalid_input():
    bad_vertices = np.array([[0.0, 0.0]], dtype=np.float32)
    bad_faces = np.array([[0, 1, 2]], dtype=np.int64)

    try:
        fill_sam3d_mesh_holes(bad_vertices, bad_faces)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid vertex shape")


def test_fill_sam3d_mesh_holes_no_boundary_returns_unchanged():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3], [0, 2, 1]], dtype=np.int64)

    new_verts, new_faces, stats = fill_sam3d_mesh_holes(vertices, faces)

    assert stats["boundary_edges_before"] == 0
    assert stats["filled_loops"] == 0
    np.testing.assert_array_equal(new_verts, vertices)
    np.testing.assert_array_equal(new_faces, faces)
