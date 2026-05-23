"""SAM 3D Objects mesh decoder sparse upsample and field assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .sam3d_assets import Sam3dAssetBlocker
from .sam3d_decoder import Sam3dMeshDecoderConfig, run_sam3d_slat_decoder_torso
from .sam3d_flexicubes_tables import check_table, dmc_table, num_vd_table
from .sam3d_slat import Sam3dSparseTensor, _layout_for_coords, _sparse_conv3d, _sparse_linear
from .sam3d_ss_flow import _silu


_SUBDIVIDE_CORNERS_3D = np.array(
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


_MESH_CUBE_CORNERS_3D = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.int32,
)
_SAM3D_MESH_FIELD_SIZES = {
    "sdf": 8,
    "deform": 24,
    "weights": 21,
    "color": 48,
}
_FLEXICUBES_CUBE_EDGES = np.array(
    [
        0,
        1,
        1,
        5,
        4,
        5,
        0,
        4,
        2,
        3,
        3,
        7,
        6,
        7,
        2,
        6,
        2,
        0,
        3,
        1,
        7,
        5,
        6,
        4,
    ],
    dtype=np.int64,
)
_FLEXICUBES_CUBE_CORNER_IDS = (2 ** np.arange(8, dtype=np.int64)).astype(np.int64)
_FLEXICUBES_DMC_TABLE = np.asarray(dmc_table, dtype=np.int64)
_FLEXICUBES_NUM_VD_TABLE = np.asarray(num_vd_table, dtype=np.int64)
_FLEXICUBES_CHECK_TABLE = np.asarray(check_table, dtype=np.int64)


@dataclass(frozen=True)
class Sam3dMeshDecoderFeatureResult:
    coords: np.ndarray
    feats: mx.array
    metadata: dict[str, object]


@dataclass(frozen=True)
class Sam3dMeshFeatureLayout:
    ranges: dict[str, tuple[int, int]]
    shapes: dict[str, tuple[int, ...]]
    total_channels: int
    use_color: bool


@dataclass(frozen=True)
class Sam3dMeshParsedFeatures:
    sdf: np.ndarray
    deform: np.ndarray
    weights: np.ndarray
    color: np.ndarray | None
    layout: Sam3dMeshFeatureLayout


@dataclass(frozen=True)
class Sam3dSparseCubeVertexAttrs:
    coords: np.ndarray
    attrs: np.ndarray
    cubes: np.ndarray


@dataclass(frozen=True)
class Sam3dMeshDenseFields:
    sdf: np.ndarray
    deform: np.ndarray
    weights: np.ndarray
    grid_vertices: np.ndarray
    deformed_vertices: np.ndarray
    colors: np.ndarray | None
    memory_estimate: dict[str, int]


@dataclass(frozen=True)
class Sam3dMeshFieldAssemblyResult:
    fields: Sam3dMeshDenseFields | None
    blocker: Sam3dAssetBlocker | None
    metadata: dict[str, object]

    @property
    def ready(self) -> bool:
        return self.blocker is None and self.fields is not None


@dataclass(frozen=True)
class Sam3dFlexiCubesSurfaceCubes:
    mask: np.ndarray
    indices: np.ndarray
    occupancy: np.ndarray
    cube_vertex_indices: np.ndarray


@dataclass(frozen=True)
class Sam3dFlexiCubesSurfaceEdges:
    edges: np.ndarray
    cube_edge_indices: np.ndarray
    cube_edge_counts: np.ndarray
    cube_surface_edge_mask: np.ndarray


@dataclass(frozen=True)
class Sam3dFlexiCubesActivatedWeights:
    beta: np.ndarray
    alpha: np.ndarray
    gamma: np.ndarray


@dataclass(frozen=True)
class Sam3dFlexiCubesDualVertexCandidates:
    vertices: np.ndarray
    gamma: np.ndarray
    cube_edge_vertex_indices: np.ndarray
    colors: np.ndarray | None


@dataclass(frozen=True)
class Sam3dFlexiCubesSurfaceCore:
    surface_cubes: Sam3dFlexiCubesSurfaceCubes
    surface_edges: Sam3dFlexiCubesSurfaceEdges
    case_ids: np.ndarray
    weights: Sam3dFlexiCubesActivatedWeights
    dual_vertices: Sam3dFlexiCubesDualVertexCandidates
    metadata: dict[str, object]


@dataclass(frozen=True)
class Sam3dMeshExtractResult:
    vertices: np.ndarray | None
    faces: np.ndarray | None
    vertex_attrs: np.ndarray | None
    colors: np.ndarray | None
    metadata: dict[str, object]
    blocker: Sam3dAssetBlocker | None = None

    @property
    def success(self) -> bool:
        return (
            self.blocker is None
            and self.vertices is not None
            and self.faces is not None
            and self.vertices.shape[0] > 0
            and self.faces.shape[0] > 0
        )

    @property
    def ready(self) -> bool:
        return self.success


def estimate_sam3d_flexicubes_bytes(
    extraction_resolution: int,
    *,
    surface_cube_count: int | None = None,
    surface_edge_count: int | None = None,
    use_color: bool = False,
) -> dict[str, int]:
    res = _validate_extraction_resolution(extraction_resolution)
    cube_count = int(res**3)
    surface_count = cube_count if surface_cube_count is None else max(int(surface_cube_count), 0)
    edge_count = surface_count * 12 if surface_edge_count is None else max(int(surface_edge_count), 0)
    max_dual_per_cube = int(_FLEXICUBES_NUM_VD_TABLE.max()) if _FLEXICUBES_NUM_VD_TABLE.size else 0
    max_dual_vertices = surface_count * max_dual_per_cube
    edge_group_entries = max_dual_vertices * 7
    int64_bytes = np.dtype(np.int64).itemsize
    float32_bytes = np.dtype(np.float32).itemsize
    bool_bytes = np.dtype(bool).itemsize
    color_width = 6 if use_color else 0
    estimate = {
        "dense_cube_vertex_indices": cube_count * 8 * int64_bytes,
        "surface_all_edges": surface_count * 12 * 2 * int64_bytes,
        "surface_edge_maps": surface_count * 12 * (2 * int64_bytes + bool_bytes),
        "unique_surface_edges": edge_count * 2 * int64_bytes,
        "dual_group_arrays": edge_group_entries * 3 * int64_bytes,
        "dual_vertex_arrays": max_dual_vertices * ((3 + 1 + color_width) * float32_bytes + 12 * int64_bytes),
        "triangulation_arrays": surface_count * 12 * (2 * int64_bytes + bool_bytes) + edge_count * 6 * int64_bytes,
    }
    estimate["estimated_flexicubes_bytes"] = int(sum(estimate.values()))
    return {key: int(value) for key, value in estimate.items()}


def sparse_subdivide3d(tensor: Sam3dSparseTensor) -> Sam3dSparseTensor:
    coords = np.asarray(tensor.coords, dtype=np.int32)
    new_coords = coords[:, None, :].copy()
    new_coords[:, :, 1:] *= 2
    new_coords = (new_coords + _SUBDIVIDE_CORNERS_3D[None, :, :]).reshape(-1, 4)
    new_feats = mx.repeat(tensor.feats, repeats=8, axis=0)
    return Sam3dSparseTensor(
        coords=new_coords.astype(np.int32, copy=False),
        feats=new_feats,
        layout=_layout_for_coords(new_coords),
        spatial_cache=tensor.spatial_cache,
    )


def sam3d_mesh_feature_layout(*, use_color: bool = False) -> Sam3dMeshFeatureLayout:
    shapes: dict[str, tuple[int, ...]] = {
        "sdf": (8, 1),
        "deform": (8, 3),
        "weights": (21,),
    }
    if use_color:
        shapes["color"] = (8, 6)
    ranges: dict[str, tuple[int, int]] = {}
    offset = 0
    for name in ("sdf", "deform", "weights", "color"):
        if name not in shapes:
            continue
        size = _SAM3D_MESH_FIELD_SIZES[name]
        ranges[name] = (offset, offset + size)
        offset += size
    return Sam3dMeshFeatureLayout(ranges=ranges, shapes=shapes, total_channels=offset, use_color=use_color)


def parse_sam3d_mesh_features(feats: np.ndarray | mx.array, *, use_color: bool = False) -> Sam3dMeshParsedFeatures:
    layout = sam3d_mesh_feature_layout(use_color=use_color)
    feats_np = np.asarray(feats, dtype=np.float32)
    if feats_np.ndim != 2:
        raise ValueError(f"SAM3D mesh features must have shape (N,C), got {feats_np.shape}")
    if feats_np.shape[1] != layout.total_channels:
        raise ValueError(
            f"SAM3D mesh features require {layout.total_channels} channels for "
            f"use_color={use_color}, got {feats_np.shape[1]}"
        )

    def get(name: str) -> np.ndarray | None:
        if name not in layout.ranges:
            return None
        start, stop = layout.ranges[name]
        return feats_np[:, start:stop].reshape((-1, *layout.shapes[name]))

    sdf = get("sdf")
    deform = get("deform")
    weights = get("weights")
    if sdf is None or deform is None or weights is None:
        raise ValueError("SAM3D mesh feature layout is missing required fields")
    return Sam3dMeshParsedFeatures(
        sdf=sdf,
        deform=deform,
        weights=weights,
        color=get("color"),
        layout=layout,
    )


def sparse_cube2verts(coords: np.ndarray, attrs: np.ndarray) -> Sam3dSparseCubeVertexAttrs:
    coords_np = _cube_coords3d(coords)
    attrs_np = np.asarray(attrs, dtype=np.float32)
    if attrs_np.ndim != 3 or attrs_np.shape[0] != coords_np.shape[0] or attrs_np.shape[1] != 8:
        raise ValueError(f"SAM3D cube attrs must have shape (N,8,C), got {attrs_np.shape} for coords {coords_np.shape}")

    verts = (coords_np[:, None, :] + _MESH_CUBE_CORNERS_3D[None, :, :]).reshape(-1, 3)
    vertex_coords, inverse = np.unique(verts, axis=0, return_inverse=True)
    flat_attrs = attrs_np.reshape(-1, attrs_np.shape[2])
    vertex_attrs = np.zeros((vertex_coords.shape[0], flat_attrs.shape[1]), dtype=np.float32)
    counts = np.zeros((vertex_coords.shape[0], 1), dtype=np.float32)
    np.add.at(vertex_attrs, inverse, flat_attrs)
    np.add.at(counts, inverse, 1.0)
    vertex_attrs /= counts
    return Sam3dSparseCubeVertexAttrs(
        coords=vertex_coords.astype(np.int32, copy=False),
        attrs=vertex_attrs,
        cubes=inverse.reshape(-1, 8).astype(np.int64, copy=False),
    )


def estimate_sam3d_mesh_field_bytes(
    extraction_resolution: int,
    *,
    use_color: bool = False,
    dtype: np.dtype | type = np.float32,
) -> dict[str, int]:
    res = _validate_extraction_resolution(extraction_resolution)
    itemsize = np.dtype(dtype).itemsize
    vertex_count = (res + 1) ** 3
    cube_count = res**3
    vertex_attr_channels = 4 + (6 if use_color else 0)
    estimates = {
        "dense_vertex_attrs": int(vertex_count * vertex_attr_channels * itemsize),
        "dense_cube_weights": int(cube_count * _SAM3D_MESH_FIELD_SIZES["weights"] * itemsize),
        "grid_vertices": int(vertex_count * 3 * itemsize),
        "deformed_vertices": int(vertex_count * 3 * itemsize),
    }
    estimates["total"] = int(sum(estimates.values()))
    return estimates


def assemble_sam3d_mesh_fields(
    coords: np.ndarray,
    feats: np.ndarray | mx.array,
    *,
    extraction_resolution: int,
    use_color: bool = False,
    max_dense_bytes: int | None = None,
) -> Sam3dMeshFieldAssemblyResult:
    res = _validate_extraction_resolution(extraction_resolution)
    memory_estimate = estimate_sam3d_mesh_field_bytes(res, use_color=use_color)
    metadata: dict[str, object] = {
        "extraction_resolution": int(res),
        "use_color": bool(use_color),
        "memory_estimate": memory_estimate,
    }
    if max_dense_bytes is not None and memory_estimate["total"] > int(max_dense_bytes):
        return Sam3dMeshFieldAssemblyResult(
            fields=None,
            blocker=Sam3dAssetBlocker(
                stage="mesh-decoder",
                operation="assemble SAM3D SparseFeatures2Mesh dense fields",
                reason=(
                    f"estimated dense mesh field allocation {memory_estimate['total']} bytes "
                    f"exceeds guard {int(max_dense_bytes)} bytes"
                ),
                metadata={
                    "estimated_dense_bytes": int(memory_estimate["total"]),
                    "max_dense_bytes": int(max_dense_bytes),
                    "extraction_resolution": int(res),
                    "use_color": bool(use_color),
                },
            ),
            metadata=metadata,
        )

    parsed = parse_sam3d_mesh_features(feats, use_color=use_color)
    coords3d = _cube_coords3d(coords)
    bounds_blocker = _dense_bounds_blocker(coords3d, res)
    if bounds_blocker is not None:
        return Sam3dMeshFieldAssemblyResult(fields=None, blocker=bounds_blocker, metadata=metadata)

    sdf = parsed.sdf + np.float32(-1.0 / res)
    vertex_attrs = np.concatenate(
        (
            sdf,
            parsed.deform,
            parsed.color if parsed.color is not None else np.empty((coords3d.shape[0], 8, 0), dtype=np.float32),
        ),
        axis=2,
    )
    sparse_vertices = sparse_cube2verts(coords3d, vertex_attrs)
    dense_vertex_attrs = _dense_attrs(
        sparse_vertices.coords,
        sparse_vertices.attrs,
        res=res + 1,
        sdf_init=True,
    )
    dense_weights = _dense_attrs(coords3d, parsed.weights, res=res, sdf_init=False)
    grid_vertices = _dense_grid_vertices(res)
    deform = dense_vertex_attrs[:, 1:4]
    fields = Sam3dMeshDenseFields(
        sdf=dense_vertex_attrs[:, 0],
        deform=deform,
        weights=dense_weights,
        grid_vertices=grid_vertices,
        deformed_vertices=get_deformed_sam3d_grid_vertices(grid_vertices, deform, res),
        colors=dense_vertex_attrs[:, 4:] if use_color else None,
        memory_estimate=memory_estimate,
    )
    metadata.update(
        {
            "layout": {
                name: {
                    "range": value,
                    "shape": parsed.layout.shapes[name],
                    "size": value[1] - value[0],
                }
                for name, value in parsed.layout.ranges.items()
            },
            "sparse_cube_count": int(coords3d.shape[0]),
            "sparse_vertex_count": int(sparse_vertices.coords.shape[0]),
            "dense_vertex_count": int(grid_vertices.shape[0]),
            "dense_cube_count": int(res**3),
            "sdf_bias": float(-1.0 / res),
        }
    )
    return Sam3dMeshFieldAssemblyResult(fields=fields, blocker=None, metadata=metadata)


def get_deformed_sam3d_grid_vertices(v_pos: np.ndarray, deform: np.ndarray, extraction_resolution: int) -> np.ndarray:
    res = _validate_extraction_resolution(extraction_resolution)
    v_pos_np = np.asarray(v_pos, dtype=np.float32)
    deform_np = np.asarray(deform, dtype=np.float32)
    if v_pos_np.ndim != 2 or v_pos_np.shape[1] != 3:
        raise ValueError(f"SAM3D grid vertices must have shape (N,3), got {v_pos_np.shape}")
    if deform_np.shape != v_pos_np.shape:
        raise ValueError(f"SAM3D deform must have shape {v_pos_np.shape}, got {deform_np.shape}")
    return (v_pos_np / res - 0.5 + (1.0 - 1e-8) / (res * 2.0) * np.tanh(deform_np)).astype(np.float32)


def sam3d_dense_cube_vertex_indices(extraction_resolution: int) -> np.ndarray:
    res = _validate_extraction_resolution(extraction_resolution)
    cube_coords = _dense_cube_coords(res)
    corner_coords = cube_coords[:, None, :] + _MESH_CUBE_CORNERS_3D[None, :, :]
    return _vertex_indices_from_coords(corner_coords.reshape(-1, 3), res + 1).reshape(-1, 8)


def identify_sam3d_flexicubes_surface_cubes(
    sdf: np.ndarray,
    cube_vertex_indices: np.ndarray,
) -> Sam3dFlexiCubesSurfaceCubes:
    sdf_np = np.asarray(sdf, dtype=np.float32).reshape(-1)
    cube_idx = _validate_cube_vertex_indices(cube_vertex_indices)
    if cube_idx.size and (cube_idx.min() < 0 or cube_idx.max() >= sdf_np.shape[0]):
        raise ValueError("SAM3D FlexiCubes cube vertex indices are outside the SDF field")
    occupancy = (sdf_np[cube_idx.reshape(-1)].reshape(-1, 8) < 0) if cube_idx.shape[0] else np.zeros((0, 8), dtype=bool)
    occ_sum = occupancy.sum(axis=1)
    mask = (occ_sum > 0) & (occ_sum < 8)
    return Sam3dFlexiCubesSurfaceCubes(
        mask=mask.astype(bool, copy=False),
        indices=np.nonzero(mask)[0].astype(np.int64, copy=False),
        occupancy=occupancy.astype(bool, copy=False),
        cube_vertex_indices=cube_idx,
    )


def identify_sam3d_flexicubes_surface_edges(
    sdf: np.ndarray,
    cube_vertex_indices: np.ndarray,
    surface_cube_mask: np.ndarray,
) -> Sam3dFlexiCubesSurfaceEdges:
    sdf_np = np.asarray(sdf, dtype=np.float32).reshape(-1)
    cube_idx = _validate_cube_vertex_indices(cube_vertex_indices)
    surf_mask = np.asarray(surface_cube_mask, dtype=bool).reshape(-1)
    if surf_mask.shape[0] != cube_idx.shape[0]:
        raise ValueError(f"SAM3D surface cube mask must have shape ({cube_idx.shape[0]},), got {surf_mask.shape}")
    surf_cube_count = int(surf_mask.sum())
    if surf_cube_count == 0:
        empty_cube_edges = np.empty((0, 12), dtype=np.int64)
        return Sam3dFlexiCubesSurfaceEdges(
            edges=np.empty((0, 2), dtype=np.int64),
            cube_edge_indices=empty_cube_edges,
            cube_edge_counts=empty_cube_edges.copy(),
            cube_surface_edge_mask=np.empty((0, 12), dtype=bool),
        )

    all_edges = cube_idx[surf_mask][:, _FLEXICUBES_CUBE_EDGES].reshape(-1, 2)
    unique_edges, inverse, counts = np.unique(all_edges, axis=0, return_inverse=True, return_counts=True)
    edge_signs = (sdf_np[unique_edges.reshape(-1)].reshape(-1, 2) < 0)
    edge_surface_mask = edge_signs.sum(axis=1) == 1
    cube_surface_edge_mask = edge_surface_mask[inverse].reshape(surf_cube_count, 12)
    cube_edge_counts = counts[inverse].reshape(surf_cube_count, 12).astype(np.int64, copy=False)
    mapping = np.full((unique_edges.shape[0],), -1, dtype=np.int64)
    mapping[edge_surface_mask] = np.arange(int(edge_surface_mask.sum()), dtype=np.int64)
    cube_edge_indices = mapping[inverse].reshape(surf_cube_count, 12)
    return Sam3dFlexiCubesSurfaceEdges(
        edges=unique_edges[edge_surface_mask].astype(np.int64, copy=False),
        cube_edge_indices=cube_edge_indices,
        cube_edge_counts=cube_edge_counts,
        cube_surface_edge_mask=cube_surface_edge_mask,
    )


def activate_sam3d_flexicubes_weights(
    weights: np.ndarray | None,
    surface_cube_mask: np.ndarray,
    *,
    weight_scale: float = 0.99,
) -> Sam3dFlexiCubesActivatedWeights:
    surf_mask = np.asarray(surface_cube_mask, dtype=bool).reshape(-1)
    surface_count = int(surf_mask.sum())
    if weights is None:
        return Sam3dFlexiCubesActivatedWeights(
            beta=np.ones((surface_count, 12), dtype=np.float32),
            alpha=np.ones((surface_count, 8), dtype=np.float32),
            gamma=np.ones((surface_count,), dtype=np.float32),
        )
    weights_np = np.asarray(weights, dtype=np.float32)
    if weights_np.ndim != 2 or weights_np.shape != (surf_mask.shape[0], 21):
        raise ValueError(f"SAM3D FlexiCubes weights must have shape ({surf_mask.shape[0]},21), got {weights_np.shape}")
    beta = np.tanh(weights_np[:, :12]) * np.float32(weight_scale) + np.float32(1.0)
    alpha = np.tanh(weights_np[:, 12:20]) * np.float32(weight_scale) + np.float32(1.0)
    gamma = _sigmoid_np(weights_np[:, 20]) * np.float32(weight_scale) + np.float32((1.0 - weight_scale) / 2.0)
    return Sam3dFlexiCubesActivatedWeights(
        beta=beta[surf_mask].astype(np.float32, copy=False),
        alpha=alpha[surf_mask].astype(np.float32, copy=False),
        gamma=gamma[surf_mask].astype(np.float32, copy=False),
    )


def compute_sam3d_flexicubes_case_ids(
    occupancy: np.ndarray,
    surface_cube_mask: np.ndarray,
    extraction_resolution: int,
) -> np.ndarray:
    res = _validate_extraction_resolution(extraction_resolution)
    occupancy_np = np.asarray(occupancy, dtype=bool)
    surf_mask = np.asarray(surface_cube_mask, dtype=bool).reshape(-1)
    if occupancy_np.ndim != 2 or occupancy_np.shape[1] != 8:
        raise ValueError(f"SAM3D FlexiCubes occupancy must have shape (N,8), got {occupancy_np.shape}")
    if occupancy_np.shape[0] != surf_mask.shape[0]:
        raise ValueError(f"SAM3D surface cube mask must have shape ({occupancy_np.shape[0]},), got {surf_mask.shape}")

    case_ids = (occupancy_np[surf_mask].astype(np.int64) * _FLEXICUBES_CUBE_CORNER_IDS[None, :]).sum(axis=1)
    problem_config = _FLEXICUBES_CHECK_TABLE[case_ids]
    to_check = problem_config[:, 0] == 1
    if not np.any(to_check):
        return case_ids.astype(np.int64, copy=False)

    problem_full = np.zeros((res, res, res, 5), dtype=np.int64)
    all_cube_coords = _dense_cube_coords(res)
    problem_coords = all_cube_coords[surf_mask][to_check]
    checked_config = problem_config[to_check]
    problem_full[problem_coords[:, 0], problem_coords[:, 1], problem_coords[:, 2]] = checked_config
    adj_coords = problem_coords + checked_config[:, 1:4]
    within_range = np.all((adj_coords >= 0) & (adj_coords < res), axis=1)
    if not np.any(within_range):
        return case_ids.astype(np.int64, copy=False)
    adj_coords = adj_coords[within_range]
    checked_config = checked_config[within_range]
    adjacent_config = problem_full[adj_coords[:, 0], adj_coords[:, 1], adj_coords[:, 2]]
    to_invert = adjacent_config[:, 0] == 1
    if np.any(to_invert):
        surface_positions = np.nonzero(to_check)[0][within_range][to_invert]
        case_ids[surface_positions] = checked_config[to_invert, -1]
    return case_ids.astype(np.int64, copy=False)


def compute_sam3d_flexicubes_dual_vertex_candidates(
    vertices: np.ndarray,
    sdf: np.ndarray,
    surface_cube_vertex_indices: np.ndarray,
    surface_edges: np.ndarray,
    case_ids: np.ndarray,
    weights: Sam3dFlexiCubesActivatedWeights,
    cube_edge_indices: np.ndarray,
    *,
    colors: np.ndarray | None = None,
) -> Sam3dFlexiCubesDualVertexCandidates:
    vertices_np = np.asarray(vertices, dtype=np.float32)
    sdf_np = np.asarray(sdf, dtype=np.float32).reshape(-1)
    surf_cube_idx = _validate_cube_vertex_indices(surface_cube_vertex_indices)
    surface_edges_np = np.asarray(surface_edges, dtype=np.int64)
    case_ids_np = np.asarray(case_ids, dtype=np.int64).reshape(-1)
    idx_map = np.asarray(cube_edge_indices, dtype=np.int64)
    if vertices_np.ndim != 2 or vertices_np.shape[1] != 3:
        raise ValueError(f"SAM3D FlexiCubes vertices must have shape (N,3), got {vertices_np.shape}")
    if surface_edges_np.ndim != 2 or surface_edges_np.shape[1] != 2:
        raise ValueError(f"SAM3D FlexiCubes surface edges must have shape (N,2), got {surface_edges_np.shape}")
    if surf_cube_idx.shape[0] != case_ids_np.shape[0] or idx_map.shape != (case_ids_np.shape[0], 12):
        raise ValueError("SAM3D FlexiCubes surface cube, case id, and edge map counts must match")
    if case_ids_np.size == 0:
        return Sam3dFlexiCubesDualVertexCandidates(
            vertices=np.empty((0, 3), dtype=np.float32),
            gamma=np.empty((0,), dtype=np.float32),
            cube_edge_vertex_indices=np.empty((0, 12), dtype=np.int64),
            colors=None if colors is None else np.empty((0, np.asarray(colors).shape[-1]), dtype=np.float32),
        )
    if weights.beta.shape != (case_ids_np.shape[0], 12) or weights.alpha.shape != (case_ids_np.shape[0], 8):
        raise ValueError("SAM3D FlexiCubes activated beta/alpha shapes must match surface cubes")
    if weights.gamma.shape != (case_ids_np.shape[0],):
        raise ValueError("SAM3D FlexiCubes activated gamma shape must match surface cubes")

    edge_group, edge_group_to_vd, edge_group_to_cube, vd_gamma = _flexicubes_dual_edge_groups(case_ids_np, weights.gamma)
    total_vertices = int(edge_group_to_vd.max() + 1) if edge_group_to_vd.size else 0
    if total_vertices == 0:
        return Sam3dFlexiCubesDualVertexCandidates(
            vertices=np.empty((0, 3), dtype=np.float32),
            gamma=np.empty((0,), dtype=np.float32),
            cube_edge_vertex_indices=np.full((case_ids_np.shape[0], 12), -1, dtype=np.int64),
            colors=None,
        )

    alpha_edges = weights.alpha[:, _FLEXICUBES_CUBE_EDGES].reshape(-1, 12, 2)
    surf_edge_vertices = vertices_np[surface_edges_np.reshape(-1)].reshape(-1, 2, 3)
    surf_edge_sdf = sdf_np[surface_edges_np.reshape(-1)].reshape(-1, 2, 1)
    idx_group = idx_map[edge_group_to_cube, edge_group]
    if np.any(idx_group < 0):
        raise ValueError("SAM3D FlexiCubes dual groups referenced a non-surface edge")
    x_group = surf_edge_vertices[idx_group]
    s_group = surf_edge_sdf[idx_group]
    alpha_group = alpha_edges[edge_group_to_cube, edge_group].reshape(-1, 2, 1)
    edge_points = _flexicubes_linear_interp(s_group * alpha_group, x_group)
    beta_group = weights.beta[edge_group_to_cube, edge_group].reshape(-1, 1)
    dual_vertices = np.zeros((total_vertices, 3), dtype=np.float32)
    beta_sum = np.zeros((total_vertices, 1), dtype=np.float32)
    np.add.at(dual_vertices, edge_group_to_vd, edge_points * beta_group)
    np.add.at(beta_sum, edge_group_to_vd, beta_group)
    dual_vertices = dual_vertices / np.maximum(beta_sum, np.float32(1e-12))

    vd_idx_map = np.full((case_ids_np.shape[0], 12), -1, dtype=np.int64)
    vd_idx_map[edge_group_to_cube, edge_group] = edge_group_to_vd

    dual_colors = None
    if colors is not None:
        colors_np = np.asarray(colors, dtype=np.float32)
        if colors_np.ndim != 2 or colors_np.shape[0] != vertices_np.shape[0]:
            raise ValueError(f"SAM3D FlexiCubes colors must have shape ({vertices_np.shape[0]},C), got {colors_np.shape}")
        surf_edge_colors = colors_np[surface_edges_np.reshape(-1)].reshape(surface_edges_np.shape[0], 2, colors_np.shape[1])
        color_group = surf_edge_colors[idx_group]
        edge_colors = _flexicubes_linear_interp(s_group * alpha_group, color_group)
        dual_colors = np.zeros((total_vertices, colors_np.shape[1]), dtype=np.float32)
        np.add.at(dual_colors, edge_group_to_vd, edge_colors * beta_group)
        dual_colors = dual_colors / np.maximum(beta_sum, np.float32(1e-12))

    return Sam3dFlexiCubesDualVertexCandidates(
        vertices=dual_vertices.astype(np.float32, copy=False),
        gamma=vd_gamma.astype(np.float32, copy=False),
        cube_edge_vertex_indices=vd_idx_map,
        colors=dual_colors,
    )


def extract_sam3d_flexicubes_surface_core(
    fields: Sam3dMeshDenseFields,
    *,
    extraction_resolution: int,
) -> Sam3dFlexiCubesSurfaceCore:
    core, blocker, _ = _extract_sam3d_flexicubes_surface_core_or_blocker(
        fields,
        extraction_resolution=extraction_resolution,
        max_flexicubes_bytes=None,
    )
    if blocker is not None or core is None:
        raise RuntimeError("unexpected SAM3D FlexiCubes blocker with no memory guard")
    return core


def _extract_sam3d_flexicubes_surface_core_or_blocker(
    fields: Sam3dMeshDenseFields,
    *,
    extraction_resolution: int,
    max_flexicubes_bytes: int | None,
) -> tuple[Sam3dFlexiCubesSurfaceCore | None, Sam3dAssetBlocker | None, dict[str, object]]:
    res = _validate_extraction_resolution(extraction_resolution)
    use_color = fields.colors is not None
    pre_estimate = estimate_sam3d_flexicubes_bytes(res, use_color=use_color)
    metadata: dict[str, object] = {
        "extraction_resolution": int(res),
        "flexicubes_memory_estimate": pre_estimate,
        "estimated_flexicubes_bytes": pre_estimate["estimated_flexicubes_bytes"],
        "max_flexicubes_bytes": None if max_flexicubes_bytes is None else int(max_flexicubes_bytes),
    }
    blocker = _flexicubes_guard_blocker(pre_estimate, max_flexicubes_bytes, metadata)
    if blocker is not None:
        return None, blocker, metadata

    cube_idx = sam3d_dense_cube_vertex_indices(res)
    surface_cubes = identify_sam3d_flexicubes_surface_cubes(fields.sdf, cube_idx)
    surface_estimate = estimate_sam3d_flexicubes_bytes(
        res,
        surface_cube_count=int(surface_cubes.indices.shape[0]),
        use_color=use_color,
    )
    metadata.update(
        {
            "surface_cube_count": int(surface_cubes.indices.shape[0]),
            "flexicubes_memory_estimate": surface_estimate,
            "estimated_flexicubes_bytes": surface_estimate["estimated_flexicubes_bytes"],
        }
    )
    blocker = _flexicubes_guard_blocker(surface_estimate, max_flexicubes_bytes, metadata)
    if blocker is not None:
        return None, blocker, metadata

    weights = activate_sam3d_flexicubes_weights(fields.weights, surface_cubes.mask)
    case_ids = compute_sam3d_flexicubes_case_ids(surface_cubes.occupancy, surface_cubes.mask, res)
    surface_edges = identify_sam3d_flexicubes_surface_edges(fields.sdf, cube_idx, surface_cubes.mask)
    edge_estimate = estimate_sam3d_flexicubes_bytes(
        res,
        surface_cube_count=int(surface_cubes.indices.shape[0]),
        surface_edge_count=int(surface_edges.edges.shape[0]),
        use_color=use_color,
    )
    metadata.update(
        {
            "surface_edge_count": int(surface_edges.edges.shape[0]),
            "flexicubes_memory_estimate": edge_estimate,
            "estimated_flexicubes_bytes": edge_estimate["estimated_flexicubes_bytes"],
        }
    )
    blocker = _flexicubes_guard_blocker(edge_estimate, max_flexicubes_bytes, metadata)
    if blocker is not None:
        return None, blocker, metadata

    dual_vertices = compute_sam3d_flexicubes_dual_vertex_candidates(
        fields.deformed_vertices,
        fields.sdf,
        cube_idx[surface_cubes.mask],
        surface_edges.edges,
        case_ids,
        weights,
        surface_edges.cube_edge_indices,
        colors=_sigmoid_np(fields.colors) if fields.colors is not None else None,
    )
    metadata.update(
        {
            "dual_vertex_candidate_count": int(dual_vertices.vertices.shape[0]),
        }
    )
    return Sam3dFlexiCubesSurfaceCore(
        surface_cubes=surface_cubes,
        surface_edges=surface_edges,
        case_ids=case_ids,
        weights=weights,
        dual_vertices=dual_vertices,
        metadata=metadata,
    ), None, metadata


def triangulate_sam3d_flexicubes_surface(
    sdf: np.ndarray,
    surface_edges: Sam3dFlexiCubesSurfaceEdges,
    dual_vertices: Sam3dFlexiCubesDualVertexCandidates,
) -> np.ndarray:
    sdf_np = np.asarray(sdf, dtype=np.float32).reshape(-1)
    surf_edges_np = np.asarray(surface_edges.edges, dtype=np.int64)
    edge_counts = np.asarray(surface_edges.cube_edge_counts, dtype=np.int64)
    idx_map = np.asarray(surface_edges.cube_edge_indices, dtype=np.int64)
    surf_edges_mask = np.asarray(surface_edges.cube_surface_edge_mask, dtype=bool)
    vd_idx_map = np.asarray(dual_vertices.cube_edge_vertex_indices, dtype=np.int64)
    vd_gamma = np.asarray(dual_vertices.gamma, dtype=np.float32).reshape(-1)
    if surf_edges_np.ndim != 2 or surf_edges_np.shape[1] != 2:
        raise ValueError(f"SAM3D FlexiCubes surface edges must have shape (N,2), got {surf_edges_np.shape}")
    if edge_counts.shape != idx_map.shape or surf_edges_mask.shape != idx_map.shape or vd_idx_map.shape != idx_map.shape:
        raise ValueError("SAM3D FlexiCubes triangulation maps must have matching cube-edge shapes")
    if surf_edges_np.size and (surf_edges_np.min() < 0 or surf_edges_np.max() >= sdf_np.shape[0]):
        raise ValueError("SAM3D FlexiCubes triangulation surface edges are outside the SDF field")

    group_mask = (edge_counts == 4) & surf_edges_mask
    if not np.any(group_mask):
        return np.empty((0, 3), dtype=np.int64)
    group = idx_map[group_mask]
    vd_idx = vd_idx_map[group_mask]
    if np.any(group < 0) or np.any(vd_idx < 0):
        raise ValueError("SAM3D FlexiCubes triangulation found an unmapped grouped surface edge")
    order = np.argsort(group, kind="stable")
    sorted_group = group[order]
    sorted_vd_idx = vd_idx[order]
    if sorted_vd_idx.shape[0] % 4 != 0:
        raise ValueError("SAM3D FlexiCubes grouped surface edges did not form complete quads")
    quad_group = sorted_group.reshape(-1, 4)
    if not np.all(quad_group == quad_group[:, :1]):
        raise ValueError("SAM3D FlexiCubes grouped surface edges must contain exactly four entries per quad")
    quad_vd_idx = sorted_vd_idx.reshape(-1, 4)

    edge_indices = quad_group[:, 0]
    s_edges = sdf_np[surf_edges_np[edge_indices].reshape(-1)].reshape(-1, 2)
    flip_mask = s_edges[:, 0] > 0
    quad_vd_idx = np.concatenate(
        (
            quad_vd_idx[flip_mask][:, [0, 1, 3, 2]],
            quad_vd_idx[~flip_mask][:, [2, 3, 1, 0]],
        ),
        axis=0,
    )
    if quad_vd_idx.shape[0] == 0:
        return np.empty((0, 3), dtype=np.int64)
    if np.any(quad_vd_idx < 0) or np.any(quad_vd_idx >= vd_gamma.shape[0]):
        raise ValueError("SAM3D FlexiCubes triangulation referenced invalid dual vertices")

    quad_gamma = vd_gamma[quad_vd_idx.reshape(-1)].reshape(-1, 4)
    gamma_02 = quad_gamma[:, 0] * quad_gamma[:, 2]
    gamma_13 = quad_gamma[:, 1] * quad_gamma[:, 3]
    split_1 = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)
    split_2 = np.array([0, 1, 3, 3, 1, 2], dtype=np.int64)
    faces = np.empty((quad_vd_idx.shape[0], 6), dtype=np.int64)
    split_mask = gamma_02 > gamma_13
    faces[split_mask] = quad_vd_idx[split_mask][:, split_1]
    faces[~split_mask] = quad_vd_idx[~split_mask][:, split_2]
    return faces.reshape(-1, 3).astype(np.int64, copy=False)


def extract_sam3d_mesh(
    fields: Sam3dMeshDenseFields,
    *,
    extraction_resolution: int,
    max_flexicubes_bytes: int | None = None,
) -> Sam3dMeshExtractResult:
    core, blocker, core_metadata = _extract_sam3d_flexicubes_surface_core_or_blocker(
        fields,
        extraction_resolution=extraction_resolution,
        max_flexicubes_bytes=max_flexicubes_bytes,
    )
    metadata: dict[str, object] = dict(core_metadata)
    if blocker is not None or core is None:
        return Sam3dMeshExtractResult(
            vertices=None,
            faces=None,
            vertex_attrs=None,
            colors=None,
            metadata=metadata,
            blocker=blocker,
        )
    if core.metadata["surface_cube_count"] == 0 or core.metadata["surface_edge_count"] == 0:
        return Sam3dMeshExtractResult(
            vertices=None,
            faces=None,
            vertex_attrs=None,
            colors=None,
            metadata=metadata,
            blocker=_mesh_extract_blocker(
                "extract SAM3D FlexiCubes inference mesh",
                "dense SDF field contains no surface sign changes",
                metadata,
            ),
        )

    faces = triangulate_sam3d_flexicubes_surface(fields.sdf, core.surface_edges, core.dual_vertices)
    vertices = core.dual_vertices.vertices.astype(np.float32, copy=False)
    vertex_attrs = None if core.dual_vertices.colors is None else core.dual_vertices.colors.astype(np.float32, copy=False)
    colors = _mesh_glb_colors(vertex_attrs)
    metadata.update(
        {
            "vertex_count": int(vertices.shape[0]),
            "face_count": int(faces.shape[0]),
            "has_vertex_attrs": vertex_attrs is not None,
            "has_vertex_color": colors is not None,
        }
    )
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return Sam3dMeshExtractResult(
            vertices=None,
            faces=None,
            vertex_attrs=None,
            colors=None,
            metadata=metadata,
            blocker=_mesh_extract_blocker(
                "extract SAM3D FlexiCubes inference mesh",
                "FlexiCubes surface core produced no triangulatable faces",
                metadata,
            ),
        )
    if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
        raise ValueError("SAM3D FlexiCubes triangulation produced faces outside the vertex array")
    return Sam3dMeshExtractResult(
        vertices=vertices,
        faces=faces,
        vertex_attrs=vertex_attrs,
        colors=colors,
        metadata=metadata,
        blocker=None,
    )


def extract_sam3d_mesh_from_features(
    coords: np.ndarray,
    feats: np.ndarray | mx.array,
    *,
    extraction_resolution: int,
    use_color: bool = False,
    max_dense_bytes: int | None = None,
    max_flexicubes_bytes: int | None = None,
) -> Sam3dMeshExtractResult:
    assembled = assemble_sam3d_mesh_fields(
        coords,
        feats,
        extraction_resolution=extraction_resolution,
        use_color=use_color,
        max_dense_bytes=max_dense_bytes,
    )
    if not assembled.ready or assembled.fields is None:
        return Sam3dMeshExtractResult(
            vertices=None,
            faces=None,
            vertex_attrs=None,
            colors=None,
            metadata=assembled.metadata,
            blocker=assembled.blocker,
        )
    result = extract_sam3d_mesh(
        assembled.fields,
        extraction_resolution=extraction_resolution,
        max_flexicubes_bytes=max_flexicubes_bytes,
    )
    return Sam3dMeshExtractResult(
        vertices=result.vertices,
        faces=result.faces,
        vertex_attrs=result.vertex_attrs,
        colors=result.colors,
        metadata={**assembled.metadata, **result.metadata},
        blocker=result.blocker,
    )


def sparse_group_norm(
    tensor: Sam3dSparseTensor,
    weight: mx.array,
    bias: mx.array,
    *,
    num_groups: int = 32,
    eps: float = 1e-5,
) -> Sam3dSparseTensor:
    feats = tensor.feats
    channels = int(feats.shape[1])
    groups = _valid_group_count(num_groups, channels)
    normalized = []
    for layout in tensor.layout:
        batch_feats = feats[layout].astype(mx.float32)
        token_count = int(batch_feats.shape[0])
        if token_count == 0:
            continue
        grouped = mx.reshape(mx.transpose(batch_feats), (groups, channels // groups, token_count))
        mean = mx.mean(grouped, axis=(1, 2), keepdims=True)
        variance = mx.mean(mx.square(grouped - mean), axis=(1, 2), keepdims=True)
        batch_norm = mx.transpose(mx.reshape((grouped - mean) * mx.rsqrt(variance + eps), (channels, token_count)))
        batch_norm = batch_norm * weight.astype(mx.float32)[None, :] + bias.astype(mx.float32)[None, :]
        normalized.append(batch_norm.astype(feats.dtype))
    if not normalized:
        return tensor
    return tensor.replace(mx.concatenate(normalized, axis=0))


def run_sparse_subdivide_block3d(
    tensor: Sam3dSparseTensor,
    tensors: dict[str, mx.array],
    *,
    prefix: str,
    num_groups: int = 32,
) -> Sam3dSparseTensor:
    hidden = sparse_group_norm(
        tensor,
        tensors[f"{prefix}act_layers.0.weight"],
        tensors[f"{prefix}act_layers.0.bias"],
        num_groups=num_groups,
    )
    hidden = hidden.replace(_silu(hidden.feats))
    hidden = sparse_subdivide3d(hidden)

    residual = sparse_subdivide3d(tensor)
    hidden = hidden.replace(_sparse_conv3d(hidden.coords, hidden.feats, tensors, prefix=f"{prefix}out_layers.0.conv."))
    hidden = sparse_group_norm(
        hidden,
        tensors[f"{prefix}out_layers.1.weight"],
        tensors[f"{prefix}out_layers.1.bias"],
        num_groups=num_groups,
    )
    hidden = hidden.replace(_silu(hidden.feats))
    hidden = hidden.replace(_sparse_conv3d(hidden.coords, hidden.feats, tensors, prefix=f"{prefix}out_layers.3.conv."))

    if f"{prefix}skip_connection.conv.weight" in tensors:
        skip = _sparse_conv3d(residual.coords, residual.feats, tensors, prefix=f"{prefix}skip_connection.conv.")
    else:
        skip = residual.feats
    return residual.replace(hidden.feats + skip)


def run_sam3d_mesh_decoder_features(
    coords: np.ndarray,
    feats: mx.array,
    tensors: dict[str, mx.array],
    config: Sam3dMeshDecoderConfig = Sam3dMeshDecoderConfig(),
) -> Sam3dMeshDecoderFeatureResult:
    hidden = run_sam3d_slat_decoder_torso(coords, feats, tensors, config)
    coords_np = np.asarray(coords, dtype=np.int32)
    tensor = Sam3dSparseTensor(coords=coords_np, feats=hidden, layout=_layout_for_coords(coords_np), spatial_cache={})
    subdivisions = []
    for index in range(_count_upsample_blocks(tensors)):
        input_tokens = tensor.token_count
        tensor = run_sparse_subdivide_block3d(tensor, tensors, prefix=f"upsample.{index}.")
        subdivisions.append(
            {
                "block": index,
                "input_tokens": input_tokens,
                "output_tokens": tensor.token_count,
            }
        )
        mx.eval(tensor.feats)
    out_feats = _sparse_linear(tensor.feats.astype(feats.dtype), tensors, prefix="out_layer.")
    return Sam3dMeshDecoderFeatureResult(
        coords=tensor.coords,
        feats=out_feats,
        metadata={
            "input_tokens": int(coords_np.shape[0]),
            "torso_tokens": int(hidden.shape[0]),
            "subdivisions": subdivisions,
            "feature_shape": tuple(int(value) for value in out_feats.shape),
            "resolution": int(config.resolution),
        },
    )


def _valid_group_count(num_groups: int, channels: int) -> int:
    groups = min(int(num_groups), int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(groups, 1)


def _count_upsample_blocks(tensors: dict[str, mx.array]) -> int:
    indices = {
        int(key.split(".")[1])
        for key in tensors
        if key.startswith("upsample.") and key.split(".")[1].isdigit()
    }
    return max(indices) + 1 if indices else 0


def _cube_coords3d(coords: np.ndarray) -> np.ndarray:
    coords_np = np.asarray(coords, dtype=np.int32)
    if coords_np.ndim != 2 or coords_np.shape[1] not in {3, 4}:
        raise ValueError(f"SAM3D cube coords must have shape (N,3) or (N,4), got {coords_np.shape}")
    return coords_np[:, 1:] if coords_np.shape[1] == 4 else coords_np


def _validate_extraction_resolution(extraction_resolution: int) -> int:
    res = int(extraction_resolution)
    if res <= 0:
        raise ValueError(f"SAM3D extraction resolution must be positive, got {extraction_resolution}")
    return res


def _dense_bounds_blocker(coords: np.ndarray, res: int) -> Sam3dAssetBlocker | None:
    if coords.size == 0:
        return None
    min_coord = coords.min(axis=0)
    max_coord = coords.max(axis=0)
    if np.any(min_coord < 0) or np.any(max_coord >= res):
        return Sam3dAssetBlocker(
            stage="mesh-decoder",
            operation="assemble SAM3D SparseFeatures2Mesh dense fields",
            reason=f"sparse cube coordinates must stay in [0, {res}), got min={tuple(min_coord)} max={tuple(max_coord)}",
            metadata={
                "extraction_resolution": int(res),
                "coord_min": tuple(int(value) for value in min_coord),
                "coord_max": tuple(int(value) for value in max_coord),
            },
        )
    return None


def _dense_attrs(coords: np.ndarray, feats: np.ndarray, *, res: int, sdf_init: bool) -> np.ndarray:
    coords_np = np.asarray(coords, dtype=np.int32)
    feats_np = np.asarray(feats, dtype=np.float32)
    if coords_np.ndim != 2 or coords_np.shape[1] != 3:
        raise ValueError(f"SAM3D dense attrs coords must have shape (N,3), got {coords_np.shape}")
    if feats_np.ndim != 2 or feats_np.shape[0] != coords_np.shape[0]:
        raise ValueError(f"SAM3D dense attrs feats must have shape (N,F), got {feats_np.shape} for coords {coords_np.shape}")
    dense = np.zeros((res, res, res, feats_np.shape[1]), dtype=np.float32)
    if sdf_init:
        dense[..., 0] = 1.0
    if coords_np.shape[0]:
        dense[coords_np[:, 0], coords_np[:, 1], coords_np[:, 2], :] = feats_np
    return dense.reshape(-1, feats_np.shape[1])


def _dense_grid_vertices(res: int) -> np.ndarray:
    res_v = res + 1
    vertex_ids = np.arange(res_v**3, dtype=np.int64)
    return np.stack(
        (
            vertex_ids // (res_v**2),
            (vertex_ids // res_v) % res_v,
            vertex_ids % res_v,
        ),
        axis=1,
    ).astype(np.float32)


def _dense_cube_coords(res: int) -> np.ndarray:
    cube_ids = np.arange(res**3, dtype=np.int64)
    return np.stack(
        (
            cube_ids // (res**2),
            (cube_ids // res) % res,
            cube_ids % res,
        ),
        axis=1,
    ).astype(np.int32)


def _vertex_indices_from_coords(coords: np.ndarray, res_v: int) -> np.ndarray:
    coords_np = np.asarray(coords, dtype=np.int64)
    return (coords_np[:, 0] * (res_v**2) + coords_np[:, 1] * res_v + coords_np[:, 2]).astype(np.int64, copy=False)


def _validate_cube_vertex_indices(cube_vertex_indices: np.ndarray) -> np.ndarray:
    cube_idx = np.asarray(cube_vertex_indices, dtype=np.int64)
    if cube_idx.ndim != 2 or cube_idx.shape[1] != 8:
        raise ValueError(f"SAM3D FlexiCubes cube indices must have shape (N,8), got {cube_idx.shape}")
    return cube_idx


def _sigmoid_np(values: np.ndarray) -> np.ndarray:
    values_np = np.asarray(values, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-values_np))).astype(np.float32)


def _mesh_glb_colors(vertex_attrs: np.ndarray | None) -> np.ndarray | None:
    if vertex_attrs is None:
        return None
    attrs_np = np.asarray(vertex_attrs, dtype=np.float32)
    if attrs_np.ndim != 2 or attrs_np.shape[1] < 3:
        raise ValueError(f"SAM3D mesh vertex attrs must have at least 3 color channels, got {attrs_np.shape}")
    return attrs_np[:, :3].astype(np.float32, copy=False)


def _mesh_extract_blocker(operation: str, reason: str, metadata: dict[str, object]) -> Sam3dAssetBlocker:
    return Sam3dAssetBlocker(
        stage="mesh-decoder",
        operation=operation,
        reason=reason,
        metadata=dict(metadata),
    )


def _flexicubes_guard_blocker(
    estimate: dict[str, int],
    max_flexicubes_bytes: int | None,
    metadata: dict[str, object],
) -> Sam3dAssetBlocker | None:
    if max_flexicubes_bytes is None:
        return None
    estimated = int(estimate["estimated_flexicubes_bytes"])
    max_bytes = int(max_flexicubes_bytes)
    if estimated <= max_bytes:
        return None
    blocked_metadata = dict(metadata)
    blocked_metadata.update(
        {
            "estimated_flexicubes_bytes": estimated,
            "max_flexicubes_bytes": max_bytes,
            "flexicubes_memory_estimate": dict(estimate),
        }
    )
    return _mesh_extract_blocker(
        "estimate SAM3D FlexiCubes intermediate arrays",
        f"estimated FlexiCubes intermediate memory {estimated} bytes exceeds guard {max_bytes} bytes",
        blocked_metadata,
    )


def _flexicubes_linear_interp(edge_weights: np.ndarray, edge_values: np.ndarray) -> np.ndarray:
    weights_np = np.asarray(edge_weights, dtype=np.float32)
    values_np = np.asarray(edge_values, dtype=np.float32)
    if weights_np.shape[1] != 2 or values_np.shape[1] != 2:
        raise ValueError("SAM3D FlexiCubes interpolation expects two endpoints per edge")
    interp_weights = np.concatenate((weights_np[:, 1:2], -weights_np[:, 0:1]), axis=1)
    denominator = interp_weights.sum(axis=1)
    if denominator.ndim == 1:
        denominator = denominator[:, None]
    signed_eps = np.where(denominator < 0, np.float32(-1e-12), np.float32(1e-12))
    denominator = np.where(np.abs(denominator) < np.float32(1e-12), signed_eps, denominator)
    return (values_np * interp_weights).sum(axis=1) / denominator


def fill_sam3d_mesh_holes(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_hole_edges: int = 256,
    max_hole_area: float | None = None,
    camera_centers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fill clean boundary loops in a SAM3D mesh.

    Detects boundary edges, identifies simple closed loops, and fills
    each with a center vertex via fan triangulation.  When camera
    centers are provided, uses multi-view visibility to determine
    which side of the boundary faces outward so the fill normal is
    consistent with the exterior surface.

    Args:
        vertices: (N, 3) float32 vertex positions.
        faces: (M, 3) int64 face indices.
        max_hole_edges: Maximum perimeter edge count for a loop to fill.
        max_hole_area: Optional maximum area for a hole to fill. If None,
            all loops under max_hole_edges are considered.
        camera_centers: Optional (C, 3) float32 camera positions for
            multi-view visibility-guided normal orientation.

    Returns:
        Tuple of (filled_vertices, filled_faces, stats_dict) where
        stats_dict contains keys: boundary_edges_before, clean_loops,
        filled_loops, skipped_large, skipped_complex, vertices_added,
        faces_added.
    """
    verts = np.asarray(vertices, dtype=np.float32)
    tris = np.asarray(faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {verts.shape}")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {tris.shape}")
    if verts.shape[0] == 0 or tris.shape[0] == 0:
        raise ValueError("mesh must contain vertices and faces")
    if np.any(tris < 0) or np.any(tris >= verts.shape[0]):
        raise ValueError("faces contain vertex indices outside the vertex array")
    if max_hole_edges < 3:
        raise ValueError(f"max_hole_edges must be at least 3, got {max_hole_edges}")

    boundary_edges, boundary_adjacency, skipped_complex = _find_mesh_boundary_edges(tris)
    loops = _extract_clean_boundary_loops(boundary_edges, boundary_adjacency)
    directed_boundary = {(int(s), int(e)) for s, e in boundary_edges}

    cameras = None if camera_centers is None else np.asarray(camera_centers, dtype=np.float32)
    if cameras is not None and (cameras.ndim != 2 or cameras.shape[1] != 3):
        raise ValueError(f"camera_centers must have shape (C, 3), got {cameras.shape}")

    new_vertices: list[np.ndarray] = []
    new_faces: list[tuple[int, int, int]] = []
    skipped_large = 0

    for loop in loops:
        if len(loop) > max_hole_edges:
            skipped_large += 1
            continue

        loop_verts = verts[np.array(loop, dtype=np.int64)]
        if max_hole_area is not None:
            area = _polygon_area(loop_verts)
            if area > max_hole_area:
                skipped_large += 1
                continue

        center_index = verts.shape[0] + len(new_vertices)
        new_vertices.append(loop_verts.mean(axis=0).astype(np.float32, copy=False))

        reverse_boundary = (loop[0], loop[1]) not in directed_boundary
        if cameras is not None:
            reverse_boundary = _fill_orientation_from_cameras(verts, loop, cameras)

        for i, start in enumerate(loop):
            end = loop[(i + 1) % len(loop)]
            if reverse_boundary:
                new_faces.append((center_index, int(end), int(start)))
            else:
                new_faces.append((center_index, int(start), int(end)))

    filled_verts = verts
    filled_faces = tris
    if new_vertices:
        filled_verts = np.concatenate([verts, np.array(new_vertices, dtype=np.float32)], axis=0)
        filled_faces = np.concatenate([tris, np.array(new_faces, dtype=np.int64)], axis=0)

    stats: dict[str, object] = {
        "boundary_edges_before": int(boundary_edges.shape[0]),
        "clean_loops": len(loops),
        "filled_loops": len(new_vertices),
        "skipped_large": skipped_large,
        "skipped_complex": skipped_complex,
        "vertices_added": len(new_vertices),
        "faces_added": len(new_faces),
    }
    return filled_verts, filled_faces, stats


def _find_mesh_boundary_edges(faces: np.ndarray) -> tuple[np.ndarray, dict[int, list[int]], int]:
    directed = np.empty((faces.shape[0] * 3, 2), dtype=np.int64)
    directed[0::3] = faces[:, [0, 1]]
    directed[1::3] = faces[:, [1, 2]]
    directed[2::3] = faces[:, [2, 0]]

    keys = np.sort(directed, axis=1)
    key_view = np.ascontiguousarray(keys).view([("a", keys.dtype), ("b", keys.dtype)]).reshape(-1)
    _, first_indices, counts = np.unique(key_view, return_index=True, return_counts=True)
    boundary = directed[first_indices[counts == 1]]

    adjacency: dict[int, list[int]] = {}
    for s, e in boundary:
        adjacency.setdefault(int(s), []).append(int(e))
        adjacency.setdefault(int(e), []).append(int(s))

    if boundary.shape[0] == 0:
        return boundary, adjacency, 0

    visited_edges: set[tuple[int, int]] = set()
    complex_components = 0
    for s, e in boundary:
        edge = (int(min(s, e)), int(max(s, e)))
        if edge in visited_edges:
            continue
        stack = [int(s)]
        comp_verts: set[int] = set()
        comp_edges: set[tuple[int, int]] = set()
        while stack:
            v = stack.pop()
            if v in comp_verts:
                continue
            comp_verts.add(v)
            for nbr in adjacency.get(v, []):
                ce = (min(v, nbr), max(v, nbr))
                comp_edges.add(ce)
                if ce not in visited_edges:
                    visited_edges.add(ce)
                    stack.append(nbr)
        if any(len(adjacency.get(v, [])) != 2 for v in comp_verts) or len(comp_edges) != len(comp_verts):
            complex_components += 1

    return boundary, adjacency, complex_components


def _extract_clean_boundary_loops(
    boundary_edges: np.ndarray, adjacency: dict[int, list[int]]
) -> list[list[int]]:
    loops: list[list[int]] = []
    visited_edges: set[tuple[int, int]] = set()

    for raw_start, raw_end in boundary_edges:
        first_edge = (int(min(raw_start, raw_end)), int(max(raw_start, raw_end)))
        if first_edge in visited_edges:
            continue

        start = int(raw_start)
        previous = -1
        current = start
        loop: list[int] = []
        comp_edges: set[tuple[int, int]] = set()
        clean = True
        while True:
            if len(adjacency.get(current, [])) != 2 or current in loop:
                clean = current == start and len(loop) >= 3
                break
            loop.append(current)
            candidates = [n for n in adjacency[current] if n != previous]
            if not candidates:
                clean = False
                break
            nxt = candidates[0]
            ce = (min(current, nxt), max(current, nxt))
            comp_edges.add(ce)
            previous, current = current, nxt
            if current == start:
                clean = len(loop) >= 3
                break

        for ce in comp_edges:
            visited_edges.add(ce)
        if clean and all(len(adjacency.get(v, [])) == 2 for v in loop):
            loops.append(loop)

    return loops


def _polygon_area(polygon: np.ndarray) -> float:
    if polygon.shape[0] < 3:
        return 0.0
    center = polygon.mean(axis=0)
    total_area = 0.0
    for i in range(polygon.shape[0]):
        j = (i + 1) % polygon.shape[0]
        total_area += float(np.linalg.norm(np.cross(polygon[i] - center, polygon[j] - center)) * 0.5)
    return total_area


def _fill_orientation_from_cameras(
    vertices: np.ndarray, loop: list[int], cameras: np.ndarray
) -> bool:
    loop_verts = vertices[np.array(loop, dtype=np.int64)]
    center = loop_verts.mean(axis=0)
    normal = np.zeros(3, dtype=np.float64)
    for i in range(len(loop)):
        j = (i + 1) % len(loop)
        normal += np.cross(
            loop_verts[i].astype(np.float64) - center.astype(np.float64),
            loop_verts[j].astype(np.float64) - center.astype(np.float64),
        )
    normal = normal / np.maximum(np.linalg.norm(normal), 1e-12)
    votes = 0
    for cam in cameras:
        view_dir = (cam.astype(np.float64) - center.astype(np.float64))
        view_dir = view_dir / np.maximum(np.linalg.norm(view_dir), 1e-12)
        if np.dot(normal, view_dir) > 0:
            votes += 1
        else:
            votes -= 1
    return votes < 0


def _flexicubes_dual_edge_groups(case_ids: np.ndarray, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edge_group_parts = []
    edge_group_to_vd_parts = []
    edge_group_to_cube_parts = []
    vd_gamma_parts = []
    total_vertices = 0
    num_vd = _FLEXICUBES_NUM_VD_TABLE[case_ids]
    for count in np.unique(num_vd):
        cube_mask = num_vd == count
        cube_indices = np.nonzero(cube_mask)[0]
        if int(count) == 0 or cube_indices.size == 0:
            continue
        current_groups = _FLEXICUBES_DMC_TABLE[case_ids[cube_mask], : int(count)].reshape(-1, int(count) * 7)
        current_vertex_count = cube_indices.size * int(count)
        current_to_vd = np.arange(current_vertex_count, dtype=np.int64)[:, None].repeat(7, axis=1) + total_vertices
        total_vertices += int(current_vertex_count)
        current_to_cube = cube_indices[:, None].repeat(int(count) * 7, axis=1)
        valid = current_groups != -1
        edge_group_parts.append(current_groups[valid])
        edge_group_to_vd_parts.append(current_to_vd.reshape(current_groups.shape)[valid])
        edge_group_to_cube_parts.append(current_to_cube.reshape(current_groups.shape)[valid])
        vd_gamma_parts.append(gamma[cube_indices][:, None].repeat(int(count), axis=1).reshape(-1))

    if not edge_group_parts:
        empty_i64 = np.empty((0,), dtype=np.int64)
        return empty_i64, empty_i64, empty_i64, np.empty((0,), dtype=np.float32)
    return (
        np.concatenate(edge_group_parts).astype(np.int64, copy=False),
        np.concatenate(edge_group_to_vd_parts).astype(np.int64, copy=False),
        np.concatenate(edge_group_to_cube_parts).astype(np.int64, copy=False),
        np.concatenate(vd_gamma_parts).astype(np.float32, copy=False),
    )
