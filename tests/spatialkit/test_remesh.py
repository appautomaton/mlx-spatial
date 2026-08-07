"""Slice 2: native narrow-band dual-contour remesh.

Validates the remesh contract on synthetic inputs: closed and small-hole meshes
become watertight manifolds, surface fidelity is bounded, output is
deterministic, and the parameter/error contract holds. The heavy two-fixture
proof on real Pixal3D meshes lives in test_real_pixal3d_export.py (Slice 3).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mlx_spatial.spatialkit.mesh import mesh_metrics, remesh_narrow_band


def _octahedron(a: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    verts = np.array(
        [[a, 0, 0], [-a, 0, 0], [0, a, 0], [0, -a, 0], [0, 0, a], [0, 0, -a]],
        dtype=np.float32,
    )
    faces = np.array(
        [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]],
        dtype=np.int64,
    )
    return verts, faces


def _uv_sphere(r: float = 0.4, nlat: int = 16, nlon: int = 24) -> tuple[np.ndarray, np.ndarray]:
    verts = [(0.0, 0.0, r)]
    for i in range(1, nlat):
        theta = math.pi * i / nlat
        for j in range(nlon):
            phi = 2.0 * math.pi * j / nlon
            verts.append(
                (r * math.sin(theta) * math.cos(phi), r * math.sin(theta) * math.sin(phi), r * math.cos(theta))
            )
    verts.append((0.0, 0.0, -r))
    south = len(verts) - 1
    faces: list[tuple[int, int, int]] = []
    for j in range(nlon):
        faces.append((0, 1 + j, 1 + (j + 1) % nlon))
    for i in range(nlat - 2):
        base = 1 + i * nlon
        for j in range(nlon):
            a = base + j
            b = base + (j + 1) % nlon
            c = b + nlon
            d = a + nlon
            faces.append((a, b, c))
            faces.append((a, c, d))
    base = 1 + (nlat - 2) * nlon
    for j in range(nlon):
        faces.append((south, base + (j + 1) % nlon, base + j))
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int64)


def _assert_watertight(vertices: np.ndarray, faces: np.ndarray) -> dict:
    metrics = mesh_metrics(vertices, faces)
    assert metrics["boundary_loop_count"] == 0, metrics
    assert metrics["boundary_open_chain_count"] == 0, metrics
    assert metrics["boundary_branched_open_chain_count"] == 0, metrics
    assert metrics["nonmanifold_edges"] == 0, metrics
    return metrics


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    contributions = np.einsum(
        "ij,ij->i",
        triangles[:, 0],
        np.cross(triangles[:, 1], triangles[:, 2]),
    )
    return float(contributions.sum() / 6.0)


def test_remesh_closed_octahedron_is_watertight():
    verts, faces = _octahedron()
    mesh, stats = remesh_narrow_band(verts, faces, resolution=32, band=1.0, project_back=0.0)
    assert mesh.faces.shape[0] > faces.shape[0]
    assert stats["backend"] == "cpu-narrow-band-udf-double-cover-dc"
    assert stats["framework"] == "native-c++"
    assert stats["execution_device"] == "cpu"
    assert stats["surface_representation"] == "udf-offset-double-cover"
    assert stats["single_surface_ready"] is False
    assert stats["quad_split_strategy"] == "max-adjacent-face-normal-alignment"
    assert stats["quad_split_1_quads"] + stats["quad_split_2_quads"] == mesh.faces.shape[0] // 2
    metrics = _assert_watertight(mesh.vertices, mesh.faces)
    assert metrics["connected_components"] == 2
    assert _signed_volume(mesh.vertices, mesh.faces) > 0.0


def test_remesh_closed_octahedron_records_udf_double_cover_area() -> None:
    verts, faces = _octahedron()
    source = mesh_metrics(verts, faces)
    mesh, _ = remesh_narrow_band(verts, faces, resolution=48, band=1.0, project_back=0.0)
    remeshed = mesh_metrics(mesh.vertices, mesh.faces)

    assert remeshed["connected_components"] == 2
    assert remeshed["surface_area"] > 1.5 * source["surface_area"]


def test_remesh_closed_sphere_is_watertight():
    verts, faces = _uv_sphere()
    mesh, _ = remesh_narrow_band(verts, faces, resolution=48, band=1.0, project_back=0.0)
    _assert_watertight(mesh.vertices, mesh.faces)


def test_remesh_small_hole_sphere_is_closed_and_watertight():
    verts, faces = _uv_sphere()
    holed = faces[6:]  # drop the north cap -> one boundary loop
    assert mesh_metrics(verts, holed)["boundary_loop_count"] == 1
    mesh, _ = remesh_narrow_band(verts, holed, resolution=48, band=1.0, project_back=0.0)
    _assert_watertight(mesh.vertices, mesh.faces)


def test_remesh_surface_fidelity_bounded():
    r = 0.4
    res = 48
    verts, faces = _uv_sphere(r=r)
    mesh, stats = remesh_narrow_band(verts, faces, resolution=res, band=1.0, project_back=0.0)
    voxel_width = float(stats["scale"]) / res
    radii = np.linalg.norm(mesh.vertices, axis=1)
    # The eps shell offsets vertices ~1 voxel off the true surface; dual-contour
    # adds <1 voxel. Bound generously at 4 voxel widths to prove no gross drift.
    assert np.max(np.abs(radii - r)) <= 4.0 * voxel_width


def test_remesh_project_back_pulls_onto_surface():
    r = 0.4
    verts, faces = _uv_sphere(r=r)
    shell, _ = remesh_narrow_band(verts, faces, resolution=48, band=1.0, project_back=0.0)
    snapped, _ = remesh_narrow_band(verts, faces, resolution=48, band=1.0, project_back=1.0)
    shell_dev = float(np.max(np.abs(np.linalg.norm(shell.vertices, axis=1) - r)))
    snapped_dev = float(np.max(np.abs(np.linalg.norm(snapped.vertices, axis=1) - r)))
    assert snapped_dev < shell_dev


def test_remesh_is_deterministic():
    verts, faces = _uv_sphere()
    a, _ = remesh_narrow_band(verts, faces, resolution=48, band=1.0, project_back=0.0)
    b, _ = remesh_narrow_band(verts, faces, resolution=48, band=1.0, project_back=0.0)
    assert np.array_equal(a.vertices, b.vertices)
    assert np.array_equal(a.faces, b.faces)


def test_remesh_stats_fields():
    verts, faces = _octahedron()
    _, stats = remesh_narrow_band(verts, faces, resolution=32, band=1.0, project_back=0.0)
    for key in (
        "backend",
        "resolution",
        "band",
        "project_back",
        "eps",
        "scale",
        "active_voxels",
        "grid_vertices_sampled",
        "bvh_nodes",
        "input_vertices",
        "input_faces",
        "output_vertices",
        "output_faces",
    ):
        assert key in stats, key
    assert stats["resolution"] == 32
    assert stats["active_voxels"] > 0
    assert stats["output_faces"] > 0


def test_remesh_open_boundary_is_closed():
    # A heavily open sharp rim (25% of an octahedron removed): remesh must CLOSE
    # the boundary. simple dual contour can leave a tiny number of non-manifold
    # edges at such sharp open rims (a known DC limitation handled by the
    # reference's downstream cleanup); representative closed/small-hole inputs
    # above stay fully manifold. We assert boundary closure here.
    verts, faces = _octahedron()
    open_faces = faces[:6]
    assert mesh_metrics(verts, open_faces)["boundary_loop_count"] >= 1
    mesh, _ = remesh_narrow_band(verts, open_faces, resolution=48, band=1.0, project_back=0.0)
    metrics = mesh_metrics(mesh.vertices, mesh.faces)
    assert metrics["boundary_loop_count"] == 0, metrics
    assert metrics["boundary_open_chain_count"] == 0, metrics
    assert metrics["boundary_branched_open_chain_count"] == 0, metrics


def test_remesh_open_planar_sheet_is_watertight() -> None:
    vertices = np.array(
        [
            [-0.4, -0.3, 0.0],
            [0.4, -0.3, 0.0],
            [0.4, 0.3, 0.0],
            [-0.4, 0.3, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    mesh, _ = remesh_narrow_band(vertices, faces, resolution=48, band=1.0, project_back=0.0)

    metrics = _assert_watertight(mesh.vertices, mesh.faces)
    assert metrics["connected_components"] == 1
    assert _signed_volume(mesh.vertices, mesh.faces) > 0.0


def test_remesh_disconnected_open_surfaces_are_individually_watertight() -> None:
    vertices = np.array(
        [
            [-0.45, -0.2, 0.0],
            [-0.15, -0.2, 0.0],
            [-0.30, 0.2, 0.0],
            [0.15, -0.2, 0.0],
            [0.45, -0.2, 0.0],
            [0.30, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)

    mesh, _ = remesh_narrow_band(vertices, faces, resolution=64, band=1.0, project_back=0.0)

    metrics = _assert_watertight(mesh.vertices, mesh.faces)
    assert metrics["connected_components"] == 2
    assert _signed_volume(mesh.vertices, mesh.faces) > 0.0


def test_remesh_nonmanifold_input_rebuilds_closed_boundary() -> None:
    vertices = np.array(
        [
            [-0.4, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.0, 0.35, 0.0],
            [0.0, -0.35, 0.0],
            [0.0, 0.0, 0.35],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64)
    assert mesh_metrics(vertices, faces)["nonmanifold_edges"] == 1

    mesh, _ = remesh_narrow_band(vertices, faces, resolution=64, band=1.0, project_back=0.0)

    metrics = mesh_metrics(mesh.vertices, mesh.faces)
    assert metrics["boundary_loop_count"] == 0, metrics
    assert metrics["boundary_open_chain_count"] == 0, metrics
    assert metrics["boundary_branched_open_chain_count"] == 0, metrics
    assert metrics["nonmanifold_edges"] == 0, metrics


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resolution": 0},
        {"resolution": -8},
        {"band": 0.0},
        {"band": -1.0},
        {"project_back": -0.1},
        {"project_back": 1.5},
    ],
)
def test_remesh_rejects_invalid_args(kwargs):
    verts, faces = _octahedron()
    call = {"resolution": 32, "band": 1.0, "project_back": 0.0, **kwargs}
    with pytest.raises((ValueError, RuntimeError)):
        remesh_narrow_band(verts, faces, **call)


def test_remesh_manifold_cell_splitting_makes_repair_a_noop() -> None:
    # This sharp, heavily open rim used to expose non-manifold edges under the
    # one-vertex-per-cell reference DC. Component-split cell vertices must make
    # the remesh manifold before any destructive vertex-fan repair is attempted.
    verts, faces = _octahedron()
    open_faces = faces[:6]  # 25% removed -> sharp open rim

    pure, pure_stats = remesh_narrow_band(verts, open_faces, resolution=64, band=1.0, project_back=0.0)
    pure_metrics = _assert_watertight(pure.vertices, pure.faces)
    assert pure_metrics["nonmanifold_vertices"] == 0
    assert pure_stats["dual_contour_topology"] == "shared-face-contour-graph-component-split"
    assert pure_stats["split_dual_cells"] > 0
    assert pure_stats["manifold_repair_vertices_added"] == 0

    repaired, repaired_stats = remesh_narrow_band(
        verts, open_faces, resolution=64, band=1.0, project_back=0.0, repair_nonmanifold=True
    )
    _assert_watertight(repaired.vertices, repaired.faces)
    assert repaired_stats["manifold_repair_vertices_added"] == 0
    assert np.array_equal(pure.vertices, repaired.vertices)
    assert np.array_equal(pure.faces, repaired.faces)

    again, _ = remesh_narrow_band(
        verts, open_faces, resolution=64, band=1.0, project_back=0.0, repair_nonmanifold=True
    )
    assert np.array_equal(repaired.vertices, again.vertices)
    assert np.array_equal(repaired.faces, again.faces)
