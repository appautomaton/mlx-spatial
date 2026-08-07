"""Thin Python entry points for native export functionality."""

from __future__ import annotations

import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..coordinate_systems import (
    GLTF_Y_UP,
    SUPPORTED_SOURCE_COORDINATE_SYSTEMS,
    vertices_to_gltf_y_up,
)
from ..ovoxel_export import (
    OVOXEL_DEFAULT_TARGET_FACES,
    OVOXEL_PREVIEW_TARGET_FACES,
)
from ._native import backend_info
from .contracts import (
    DecodedOVoxelInputs,
    load_decoded_ovoxel_npz,
    resolve_model_identity,
    resolve_source_coordinate_system,
    validate_decoded_ovoxel,
)
from .glb import NativeGlbArtifact, textured_glb_payload, write_textured_glb
from .glb_compare import compare_textured_glbs, inspect_glb
from .mesh import (
    NativeMesh,
    clean_mesh,
    extract_flexi_dual_grid,
    fill_holes,
    mesh_metrics,
    remesh_narrow_band,
    repair_nonmanifold_mesh,
    simplify_mesh,
    simplify_mesh_mlx_parallel_qem,
    unify_face_orientations,
)
from .monitoring import (
    PIXAL3D_MEMORY_POLL_INTERVAL_SEC,
    _ProcessMemoryMonitor,
    _observed_substage,
    _timed_stage,
)
from .pixal3d_quality import (
    _export_quality_summary as _export_quality_summary,
    _native_chart_uv_candidate_status,
    _normalize_quality_preset as _normalize_ovoxel_quality_preset,
    _pixal3d_reference_stage_contract as _pixal3d_reference_stage_contract,
    _resolve_chart_angle_degrees,
    _resolve_pixal3d_uv_backend as _resolve_pixal3d_uv_backend,
    _resolve_pixal3d_uv_backend as _resolve_ovoxel_uv_backend,
    _resolve_simplify_backend,
    _resolve_tile_padding,
    _simplifier_backend_for_quality_preset,
    _topology_blocker_map as _topology_blocker_map,
    _xatlas_chart_parity_summary,
)
from .pixal3d_reporting import (
    _build_ovoxel_run_manifest,
    _build_pixal3d_run_manifest as _build_pixal3d_run_manifest,
    _fixture_manifest_summary,
    _glb_viewer_compatibility_summary as _glb_viewer_compatibility_summary,
    _load_pixal3d_fixture_manifest as _load_pixal3d_fixture_manifest,
    _load_pixal3d_reference_trace,
    _production_equivalence_summary as _production_equivalence_summary,
    _reference_comparison,
    _reference_glb_path,
    _upstream_export_settings_summary as _upstream_export_settings_summary,
    _visual_comparison_summary,
    decoded_metadata_value,
)
from .quality import summarize_ovoxel_export_quality, unavailable_production_equivalence
from .uv import (
    NativeUvMesh,
    make_face_atlas_uvs,
    make_native_chart_uvs,
    make_reference_uvs,
    make_xatlas_uvs,
)

PIXAL3D_REFERENCE_TARGET_FACES = 212_542
OVOXEL_SMALL_BOUNDARY_LOOP_FILL_MAX_EDGES = 8
OVOXEL_SMALL_BOUNDARY_LOOP_FILL_MAX_PERIMETER = 0.03


@dataclass(frozen=True)
class OVoxelGlbExportResult:
    """End-to-end native O-Voxel GLB export result."""

    glb: NativeGlbArtifact
    diagnostics_path: Path
    diagnostics: dict[str, Any]
    # Populated only when export_decoded_ovoxel_glb(..., expose_raw_postprocess_inputs=True):
    # the raw pre-postprocess bake channels + coverage status (Slice-2 oracle contract).
    raw_texture_inputs: dict[str, np.ndarray] | None = None

def export_decoded_ovoxel_glb(
    decoded_dir: str | Path,
    output: str | Path,
    *,
    texture_size: int = 1024,
    target_faces: int | None = None,
    quality_preset: str = "preview",
    grid_size: int | None = None,
    min_component_faces: int = 32,
    uv_backend: str = "face-atlas",
    xatlas_parallel_chunks: int | None = None,
    chart_angle_degrees: float = 45.0,
    tile_padding: float | None = None,
    small_boundary_loop_fill_max_edges: int = OVOXEL_SMALL_BOUNDARY_LOOP_FILL_MAX_EDGES,
    small_boundary_loop_fill_max_perimeter: float = OVOXEL_SMALL_BOUNDARY_LOOP_FILL_MAX_PERIMETER,
    max_texture_pixels: int | None = None,
    source_projection: bool = True,
    source_projection_fallback_mode: str = "knn",
    source_projection_fallback_neighbors: int = 8,
    source_projection_fallback_max_distance_voxels: float = 12.0,
    render_padding: bool = True,
    remesh: bool = False,
    remesh_band: float = 1.0,
    remesh_resolution: int | None = None,
    remesh_project_back: float = 0.0,
    remesh_repair_nonmanifold: bool = False,
    simplify_backend: str | None = None,
    diagnostics_path: str | Path | None = None,
    texture_postprocess: str = "legacy-dilation",
    expose_raw_postprocess_inputs: bool = False,
    source_coordinate_system: str = "auto",
) -> OVoxelGlbExportResult:
    """Convert decoded O-Voxel NPZ artifacts into a textured GLB through native hot paths."""

    from .texture import bake_pbr_texture

    source_dir = Path(decoded_dir)
    if not source_dir.is_dir():
        raise ValueError(f"decoded O-Voxel directory does not exist: {source_dir}")
    if texture_size <= 0:
        raise ValueError("texture_size must be positive")
    if grid_size is not None and grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if min_component_faces <= 0:
        raise ValueError("min_component_faces must be positive")
    resolved_small_boundary_loop_fill_max_edges = int(small_boundary_loop_fill_max_edges)
    if resolved_small_boundary_loop_fill_max_edges < 0:
        raise ValueError("small_boundary_loop_fill_max_edges must be non-negative")
    resolved_small_boundary_loop_fill_max_perimeter = float(small_boundary_loop_fill_max_perimeter)
    if (
        not math.isfinite(resolved_small_boundary_loop_fill_max_perimeter)
        or resolved_small_boundary_loop_fill_max_perimeter <= 0
    ):
        raise ValueError("small_boundary_loop_fill_max_perimeter must be positive")
    if max_texture_pixels is not None and max_texture_pixels <= 0:
        raise ValueError("max_texture_pixels must be positive")
    if source_projection_fallback_mode not in {"knn", "disabled"}:
        raise ValueError("source_projection_fallback_mode must be 'knn' or 'disabled'")
    if source_projection_fallback_neighbors <= 0:
        raise ValueError("source_projection_fallback_neighbors must be positive")
    if source_projection_fallback_max_distance_voxels <= 0:
        raise ValueError("source_projection_fallback_max_distance_voxels must be positive")
    if source_coordinate_system != "auto" and source_coordinate_system not in SUPPORTED_SOURCE_COORDINATE_SYSTEMS:
        raise ValueError(
            "source_coordinate_system must be 'auto' or one of "
            f"{SUPPORTED_SOURCE_COORDINATE_SYSTEMS}, got {source_coordinate_system!r}"
        )
    if remesh:
        if not math.isfinite(remesh_band) or remesh_band <= 0:
            raise ValueError("remesh_band must be positive and finite")
        if remesh_resolution is not None and remesh_resolution <= 0:
            raise ValueError("remesh_resolution must be positive")
        if not math.isfinite(remesh_project_back) or not (0.0 <= remesh_project_back <= 1.0):
            raise ValueError("remesh_project_back must be in [0, 1]")
    resolved_simplify_backend = _resolve_simplify_backend(simplify_backend)
    if resolved_simplify_backend == "qem" and not (remesh and remesh_repair_nonmanifold):
        raise ValueError(
            "simplify_backend='qem' requires remesh=True and remesh_repair_nonmanifold=True "
            "to guarantee QEM receives a watertight manifold input"
        )
    if resolved_simplify_backend == "mlx-qem" and not remesh:
        raise ValueError("simplify_backend='mlx-qem' requires remesh=True")
    if resolved_simplify_backend in {"single-layer-qem", "single-layer-mlx-qem"} and remesh:
        raise ValueError(
            f"simplify_backend={resolved_simplify_backend!r} requires remesh=False because it preserves "
            "the original FlexiDualGrid single-layer surface"
        )
    resolved_uv_backend = _resolve_ovoxel_uv_backend(uv_backend)
    resolved_xatlas_parallel_chunks: int | None = None
    if resolved_uv_backend == "xatlas-parallel-spatial":
        if xatlas_parallel_chunks is None or xatlas_parallel_chunks <= 1:
            raise ValueError(
                "uv_backend='xatlas-parallel-spatial' requires xatlas_parallel_chunks > 1"
            )
        resolved_xatlas_parallel_chunks = int(xatlas_parallel_chunks)
    elif xatlas_parallel_chunks is not None:
        raise ValueError(
            "xatlas_parallel_chunks only applies to uv_backend='xatlas-parallel-spatial'"
        )
    resolved_chart_angle_degrees = _resolve_chart_angle_degrees(chart_angle_degrees)
    resolved_tile_padding, tile_padding_source = _resolve_tile_padding(tile_padding, resolved_uv_backend)
    glb_path, resolved_diagnostics_path = _resolve_ovoxel_export_paths(output, diagnostics_path)
    shape_path = source_dir / "shape_decoder_fields.npz"
    texture_path = source_dir / "texture_decoder_pbr.npz"
    if not shape_path.exists():
        raise ValueError(f"missing decoded shape artifact: {shape_path}")
    if not texture_path.exists():
        raise ValueError(f"missing decoded texture artifact: {texture_path}")

    resolved_quality_preset = _normalize_ovoxel_quality_preset(quality_preset)
    if target_faces is not None and int(target_faces) <= 0:
        raise ValueError("target_faces must be positive")
    requested_simplifier_backend = _simplifier_backend_for_quality_preset(resolved_quality_preset)
    if resolved_simplify_backend is not None:
        requested_simplifier_backend = resolved_simplify_backend
    diagnostics: dict[str, Any] = {
        "stage": "ovoxel_glb_export",
        "source_dir": str(source_dir),
        "output_path": str(glb_path),
        "diagnostics_path": str(resolved_diagnostics_path),
        "settings": {
            "quality_preset": resolved_quality_preset,
            "texture_size": int(texture_size),
            "requested_simplifier_backend": requested_simplifier_backend,
            "requested_target_faces": int(target_faces) if target_faces is not None else None,
            "grid_size": int(grid_size) if grid_size is not None else None,
            "min_component_faces": int(min_component_faces),
            "small_boundary_loop_fill_max_edges": resolved_small_boundary_loop_fill_max_edges,
            "small_boundary_loop_fill_max_perimeter": resolved_small_boundary_loop_fill_max_perimeter,
            "requested_uv_backend": str(uv_backend),
            "uv_backend": resolved_uv_backend,
            "xatlas_parallel_chunks": resolved_xatlas_parallel_chunks,
            "chart_angle_degrees": resolved_chart_angle_degrees,
            "tile_padding": resolved_tile_padding,
            "tile_padding_source": tile_padding_source,
            "max_texture_pixels": int(max_texture_pixels) if max_texture_pixels is not None else None,
            "source_projection": bool(source_projection),
            "source_projection_fallback_mode": source_projection_fallback_mode,
            "source_projection_fallback_neighbors": int(source_projection_fallback_neighbors),
            "source_projection_fallback_max_distance_voxels": float(source_projection_fallback_max_distance_voxels),
            "render_padding": bool(render_padding),
            "remesh": bool(remesh),
            "remesh_band": float(remesh_band),
            "remesh_resolution": int(remesh_resolution) if remesh_resolution is not None else None,
            "remesh_project_back": float(remesh_project_back),
            "remesh_repair_nonmanifold": bool(remesh_repair_nonmanifold),
            "simplify_backend": resolved_simplify_backend,
            "source_coordinate_system_requested": source_coordinate_system,
        },
        "stages": {},
        "timings_sec": {},
        "memory_samples": {},
    }

    memory_monitor = _ProcessMemoryMonitor()

    def sample(label: str) -> None:
        diagnostics["memory_samples"][label] = memory_monitor.sample(label)

    memory_monitor.start()
    sample("start")
    decoded = _timed_stage(
        diagnostics,
        "load_npz",
        lambda: load_decoded_ovoxel_npz(shape_path, texture_path),
        memory_monitor=memory_monitor,
    )
    model_identity = resolve_model_identity(decoded.shape_metadata, decoded.texture_metadata)
    fixture_manifest = None
    model_identity_source = "decoded-artifact-metadata"
    if model_identity["family"] in {"pixal3d", "ovoxel"}:
        fixture_manifest = _load_pixal3d_fixture_manifest(source_dir)
        if model_identity["family"] == "ovoxel" and fixture_manifest is not None:
            model_identity = {
                "family": "pixal3d",
                "label": "Pixal3D",
                "asset_prefix": "Pixal3D",
            }
            model_identity_source = "pixal3d-fixture-manifest"
    diagnostics["model"] = model_identity
    diagnostics["model_identity_source"] = model_identity_source
    export_settings = _resolve_ovoxel_export_settings(
        source_dir,
        model_identity["family"],
        resolved_quality_preset,
        target_faces,
        fixture_manifest=fixture_manifest,
    )
    reference = export_settings["reference"]
    resolved_target_faces = int(export_settings["target_faces"])
    diagnostics["settings"].update(
        {
            "target_faces": resolved_target_faces,
            "target_faces_source": export_settings["target_faces_source"],
            "reference_available": reference is not None,
            "reference_profile": export_settings["reference_profile"],
            "reference_trace_path": str(reference["trace_path"]) if reference is not None else None,
            "reference_target_faces": reference.get("final_faces") if reference is not None else None,
            "reference_texture_size": reference.get("texture_size") if reference is not None else None,
            "reference_xatlas_face_guard": reference.get("xatlas_face_guard") if reference is not None else None,
        }
    )
    if fixture_manifest is not None:
        diagnostics["fixture_manifest"] = _fixture_manifest_summary(fixture_manifest)
    diagnostics["contracts"] = decoded.contracts
    diagnostics["source"] = {
        "shape_decoder": {
            "path": str(shape_path),
            "metadata": decoded.shape_metadata,
        },
        "texture_decoder": {
            "path": str(texture_path),
            "metadata": decoded.texture_metadata,
            "spatial_shape": decoded.texture_spatial_shape,
            "batch_size": decoded.texture_batch_size,
            "decode_resolution": decoded.texture_decode_resolution,
            "voxel_size": decoded.texture_voxel_size,
        },
    }
    resolved_source_coordinate_system = resolve_source_coordinate_system(
        source_coordinate_system,
        decoded.shape_metadata,
        decoded.texture_metadata,
    )
    diagnostics["settings"]["source_coordinate_system"] = resolved_source_coordinate_system
    diagnostics["settings"]["gltf_coordinate_transform"] = (
        "identity"
        if resolved_source_coordinate_system == GLTF_Y_UP
        else "(x,y,z)->(x,z,-y)"
    )
    sample("after_load_npz")

    resolved_grid_size = _resolve_positive_int(
        grid_size,
        decoded.texture_decode_resolution,
        decoded.shape_metadata.get("actual_hr_resolution"),
        decoded.texture_metadata.get("decode_resolution"),
        default=1024,
        name="grid_size",
    )
    diagnostics["settings"]["grid_size"] = resolved_grid_size
    resolved_remesh_resolution = int(remesh_resolution) if remesh_resolution is not None else resolved_grid_size
    if remesh:
        diagnostics["settings"]["remesh_resolution"] = resolved_remesh_resolution
    resolved_max_texture_pixels = max_texture_pixels if max_texture_pixels is not None else int(texture_size) * int(texture_size)
    diagnostics["settings"]["max_texture_pixels"] = resolved_max_texture_pixels

    shape_coordinates = decoded.shape_coordinates
    shape_fields = decoded.shape_fields
    texture_coordinates = decoded.texture_coordinates
    texture_attributes = decoded.texture_attributes
    texture_decode_resolution = decoded.texture_decode_resolution or resolved_grid_size
    texture_voxel_size = decoded.texture_voxel_size
    del decoded

    mesh = _timed_stage(
        diagnostics,
        "extract_mesh",
        lambda: extract_flexi_dual_grid(shape_coordinates, shape_fields, grid_size=resolved_grid_size),
        memory_monitor=memory_monitor,
    )
    diagnostics["stages"]["extract_mesh"].update(_mesh_shape(mesh, "source"))
    del shape_coordinates, shape_fields
    gc.collect()
    sample("after_extract_mesh")

    pre_metrics = _timed_stage(
        diagnostics,
        "source_metrics",
        lambda: mesh_metrics(mesh.vertices, mesh.faces),
        memory_monitor=memory_monitor,
    )
    diagnostics["stages"]["source_metrics"]["metrics"] = pre_metrics

    cleaned, clean_stats = _timed_stage(
        diagnostics,
        "clean_mesh",
        lambda: clean_mesh(mesh.vertices, mesh.faces, min_component_faces=min_component_faces),
        memory_monitor=memory_monitor,
    )
    diagnostics["stages"]["clean_mesh"].update(_mesh_shape(cleaned, "cleaned"))
    diagnostics["stages"]["clean_mesh"]["stats"] = clean_stats
    del mesh
    gc.collect()
    sample("after_clean_mesh")
    source_projection_vertices = cleaned.vertices
    source_projection_faces = cleaned.faces

    simplify_source = cleaned
    if remesh:
        remeshed, remesh_stats = _timed_stage(
            diagnostics,
            "remesh",
            lambda: remesh_narrow_band(
                cleaned.vertices,
                cleaned.faces,
                resolution=resolved_remesh_resolution,
                band=remesh_band,
                project_back=remesh_project_back,
                repair_nonmanifold=remesh_repair_nonmanifold,
            ),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["remesh"].update(_mesh_shape(remeshed, "remeshed"))
        diagnostics["stages"]["remesh"]["stats"] = remesh_stats
        remesh_metrics = _timed_stage(
            diagnostics,
            "remesh_metrics",
            lambda: mesh_metrics(remeshed.vertices, remeshed.faces),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["remesh_metrics"]["metrics"] = remesh_metrics
        remesh_cleaned, remesh_clean_stats = _timed_stage(
            diagnostics,
            "clean_remeshed_mesh",
            lambda: clean_mesh(
                remeshed.vertices,
                remeshed.faces,
                min_component_faces=min_component_faces,
            ),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["clean_remeshed_mesh"].update(
            _mesh_shape(remesh_cleaned, "cleaned_remeshed")
        )
        diagnostics["stages"]["clean_remeshed_mesh"]["stats"] = remesh_clean_stats
        clean_remeshed_metrics = _timed_stage(
            diagnostics,
            "clean_remeshed_metrics",
            lambda: mesh_metrics(remesh_cleaned.vertices, remesh_cleaned.faces),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["clean_remeshed_metrics"]["metrics"] = clean_remeshed_metrics
        simplify_source = remesh_cleaned
        del remeshed
        sample("after_remesh")

    if requested_simplifier_backend in {"single-layer-qem", "single-layer-mlx-qem"}:
        coarse_target_faces = max(resolved_target_faces * 3, resolved_target_faces)
        if requested_simplifier_backend == "single-layer-mlx-qem":
            coarse_mlx_input, coarse_mlx_fill_stats = _timed_stage(
                diagnostics,
                "fill_pre_coarse_mlx_qem_holes",
                lambda: fill_holes(
                    simplify_source.vertices,
                    simplify_source.faces,
                    max_hole_perimeter=resolved_small_boundary_loop_fill_max_perimeter,
                ),
                memory_monitor=memory_monitor,
            )
            diagnostics["stages"]["fill_pre_coarse_mlx_qem_holes"].update(
                _mesh_shape(coarse_mlx_input, "filled_pre_coarse_mlx_qem")
            )
            diagnostics["stages"]["fill_pre_coarse_mlx_qem_holes"]["stats"] = coarse_mlx_fill_stats
            coarse_simplified, coarse_simplify_stats = _timed_stage(
                diagnostics,
                "coarse_simplify_mesh",
                lambda: simplify_mesh_mlx_parallel_qem(
                    coarse_mlx_input.vertices,
                    coarse_mlx_input.faces,
                    target_faces=coarse_target_faces,
                    max_rounds=128,
                    topology_policy="cumesh-reference",
                ),
                memory_monitor=memory_monitor,
            )
            coarse_simplify_stats["pipeline"] = "coarse-mlx-parallel-qem"
            coarse_simplify_stats["pre_simplify_hole_fill_backend"] = str(
                coarse_mlx_fill_stats.get("backend", "unknown")
            )
            coarse_simplify_stats["pre_simplify_hole_fill_algorithm"] = str(
                coarse_mlx_fill_stats.get("algorithm", "unknown")
            )
            coarse_simplify_stats["pre_simplify_hole_fill_filled_loops"] = int(
                coarse_mlx_fill_stats.get("filled_loops", 0)
            )
            coarse_simplify_stats["pre_simplify_hole_fill_residual_clean_boundary_loops"] = int(
                coarse_mlx_fill_stats.get("residual_clean_boundary_loops", 0)
            )
            del coarse_mlx_input
        else:
            coarse_simplified, coarse_simplify_stats = _timed_stage(
                diagnostics,
                "coarse_simplify_mesh",
                lambda: simplify_mesh(
                    simplify_source.vertices,
                    simplify_source.faces,
                    target_faces=coarse_target_faces,
                    min_component_faces=min_component_faces,
                    backend="topology-aware",
                    small_boundary_loop_fill_max_edges=resolved_small_boundary_loop_fill_max_edges,
                    small_boundary_loop_fill_max_perimeter=resolved_small_boundary_loop_fill_max_perimeter,
                ),
                memory_monitor=memory_monitor,
            )
        diagnostics["stages"]["coarse_simplify_mesh"].update(
            _mesh_shape(coarse_simplified, "coarse_simplified")
        )
        diagnostics["stages"]["coarse_simplify_mesh"]["stats"] = coarse_simplify_stats

        repaired, repair_stats = _timed_stage(
            diagnostics,
            "repair_nonmanifold_mesh",
            lambda: repair_nonmanifold_mesh(coarse_simplified.vertices, coarse_simplified.faces),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["repair_nonmanifold_mesh"].update(_mesh_shape(repaired, "repaired"))
        diagnostics["stages"]["repair_nonmanifold_mesh"]["stats"] = repair_stats
        repair_metrics = _timed_stage(
            diagnostics,
            "repair_nonmanifold_metrics",
            lambda: mesh_metrics(repaired.vertices, repaired.faces),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["repair_nonmanifold_metrics"]["metrics"] = repair_metrics

        repaired_cleaned, repaired_clean_stats = _timed_stage(
            diagnostics,
            "clean_repaired_mesh",
            lambda: clean_mesh(
                repaired.vertices,
                repaired.faces,
                min_component_faces=min_component_faces,
            ),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["clean_repaired_mesh"].update(
            _mesh_shape(repaired_cleaned, "cleaned_repaired")
        )
        diagnostics["stages"]["clean_repaired_mesh"]["stats"] = repaired_clean_stats
        repaired_clean_metrics = _timed_stage(
            diagnostics,
            "clean_repaired_metrics",
            lambda: mesh_metrics(repaired_cleaned.vertices, repaired_cleaned.faces),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["clean_repaired_metrics"]["metrics"] = repaired_clean_metrics

        if requested_simplifier_backend == "single-layer-mlx-qem":
            mlx_qem_input, mlx_qem_pre_fill_stats = _timed_stage(
                diagnostics,
                "fill_pre_mlx_qem_holes",
                lambda: fill_holes(
                    repaired_cleaned.vertices,
                    repaired_cleaned.faces,
                    max_hole_perimeter=resolved_small_boundary_loop_fill_max_perimeter,
                ),
                memory_monitor=memory_monitor,
            )
            diagnostics["stages"]["fill_pre_mlx_qem_holes"].update(
                _mesh_shape(mlx_qem_input, "filled_pre_mlx_qem")
            )
            diagnostics["stages"]["fill_pre_mlx_qem_holes"]["stats"] = mlx_qem_pre_fill_stats
            simplified, simplify_stats = _timed_stage(
                diagnostics,
                "simplify_mesh",
                lambda: simplify_mesh_mlx_parallel_qem(
                    mlx_qem_input.vertices,
                    mlx_qem_input.faces,
                    target_faces=resolved_target_faces,
                    max_rounds=128,
                    topology_policy="cumesh-reference",
                ),
                memory_monitor=memory_monitor,
            )
            simplify_stats["pipeline"] = "single-layer-coarse-repair-mlx-parallel-qem"
            mlx_qem_raw = simplified
            mlx_qem_raw_metrics = _timed_stage(
                diagnostics,
                "mlx_qem_raw_metrics",
                lambda: mesh_metrics(mlx_qem_raw.vertices, mlx_qem_raw.faces),
                memory_monitor=memory_monitor,
            )
            diagnostics["stages"]["mlx_qem_raw_metrics"]["metrics"] = mlx_qem_raw_metrics
            mlx_qem_cleaned, mlx_qem_clean_stats = _timed_stage(
                diagnostics,
                "clean_mlx_qem_mesh",
                lambda: clean_mesh(
                    mlx_qem_raw.vertices,
                    mlx_qem_raw.faces,
                    min_component_faces=min_component_faces,
                ),
                memory_monitor=memory_monitor,
            )
            diagnostics["stages"]["clean_mlx_qem_mesh"].update(
                _mesh_shape(mlx_qem_cleaned, "cleaned_mlx_qem")
            )
            diagnostics["stages"]["clean_mlx_qem_mesh"]["stats"] = mlx_qem_clean_stats
            mlx_qem_repaired, mlx_qem_repair_stats = _timed_stage(
                diagnostics,
                "repair_mlx_qem_mesh",
                lambda: repair_nonmanifold_mesh(mlx_qem_cleaned.vertices, mlx_qem_cleaned.faces),
                memory_monitor=memory_monitor,
            )
            diagnostics["stages"]["repair_mlx_qem_mesh"].update(
                _mesh_shape(mlx_qem_repaired, "repaired_mlx_qem")
            )
            diagnostics["stages"]["repair_mlx_qem_mesh"]["stats"] = mlx_qem_repair_stats
            simplified, mlx_qem_final_clean_stats = _timed_stage(
                diagnostics,
                "clean_repaired_mlx_qem_mesh",
                lambda: clean_mesh(
                    mlx_qem_repaired.vertices,
                    mlx_qem_repaired.faces,
                    min_component_faces=min_component_faces,
                ),
                memory_monitor=memory_monitor,
            )
            diagnostics["stages"]["clean_repaired_mlx_qem_mesh"].update(
                _mesh_shape(simplified, "cleaned_repaired_mlx_qem")
            )
            diagnostics["stages"]["clean_repaired_mlx_qem_mesh"]["stats"] = mlx_qem_final_clean_stats
            simplified, mlx_qem_hole_fill_stats = _timed_stage(
                diagnostics,
                "fill_repaired_mlx_qem_holes",
                lambda: fill_holes(
                    simplified.vertices,
                    simplified.faces,
                    max_hole_perimeter=resolved_small_boundary_loop_fill_max_perimeter,
                ),
                memory_monitor=memory_monitor,
            )
            diagnostics["stages"]["fill_repaired_mlx_qem_holes"].update(
                _mesh_shape(simplified, "filled_repaired_mlx_qem")
            )
            diagnostics["stages"]["fill_repaired_mlx_qem_holes"]["stats"] = mlx_qem_hole_fill_stats
            mlx_qem_repaired_metrics = _timed_stage(
                diagnostics,
                "mlx_qem_repaired_metrics",
                lambda: mesh_metrics(simplified.vertices, simplified.faces),
                memory_monitor=memory_monitor,
            )
            diagnostics["stages"]["mlx_qem_repaired_metrics"]["metrics"] = mlx_qem_repaired_metrics
            simplify_stats["qem_raw_final_vertices"] = int(mlx_qem_raw.vertices.shape[0])
            simplify_stats["qem_raw_final_faces"] = int(mlx_qem_raw.faces.shape[0])
            simplify_stats["pre_qem_hole_fill_backend"] = str(
                mlx_qem_pre_fill_stats.get("backend", "unknown")
            )
            simplify_stats["pre_qem_hole_fill_algorithm"] = str(
                mlx_qem_pre_fill_stats.get("algorithm", "unknown")
            )
            simplify_stats["pre_qem_hole_fill_filled_loops"] = int(
                mlx_qem_pre_fill_stats.get("filled_loops", 0)
            )
            simplify_stats["pre_qem_hole_fill_residual_clean_boundary_loops"] = int(
                mlx_qem_pre_fill_stats.get("residual_clean_boundary_loops", 0)
            )
            simplify_stats["pre_qem_hole_fill_max_perimeter"] = float(
                mlx_qem_pre_fill_stats.get("max_hole_perimeter", 0.0)
            )
            simplify_stats["post_qem_repair_backend"] = str(mlx_qem_repair_stats.get("backend", "unknown"))
            simplify_stats["post_qem_hole_fill_backend"] = str(
                mlx_qem_hole_fill_stats.get("backend", "unknown")
            )
            simplify_stats["post_qem_hole_fill_algorithm"] = str(
                mlx_qem_hole_fill_stats.get("algorithm", "unknown")
            )
            simplify_stats["post_qem_hole_fill_filled_loops"] = int(
                mlx_qem_hole_fill_stats.get("filled_loops", 0)
            )
            simplify_stats["post_qem_hole_fill_faces_added"] = int(
                mlx_qem_hole_fill_stats.get("faces_added", 0)
            )
            simplify_stats["post_qem_hole_fill_residual_clean_boundary_loops"] = int(
                mlx_qem_hole_fill_stats.get("residual_clean_boundary_loops", 0)
            )
            simplify_stats["post_qem_hole_fill_skipped_large_loops"] = int(
                mlx_qem_hole_fill_stats.get("skipped_large_loops", 0)
            )
            simplify_stats["post_qem_hole_fill_skipped_complex_components"] = int(
                mlx_qem_hole_fill_stats.get("skipped_complex_components", 0)
            )
            simplify_stats["post_qem_hole_fill_max_perimeter"] = float(
                mlx_qem_hole_fill_stats.get("max_hole_perimeter", 0.0)
            )
            simplify_stats["post_qem_final_vertices"] = int(simplified.vertices.shape[0])
            simplify_stats["post_qem_final_faces"] = int(simplified.faces.shape[0])
            simplify_stats["final_vertices"] = int(simplified.vertices.shape[0])
            simplify_stats["final_faces"] = int(simplified.faces.shape[0])
            simplify_stats["target_reached"] = int(mlx_qem_raw.faces.shape[0]) <= resolved_target_faces
            simplify_stats["final_cleanup_exceeds_target"] = (
                int(simplified.faces.shape[0]) > resolved_target_faces
            )
            simplify_stats["post_qem_topology_ready"] = not bool(
                mlx_qem_repaired_metrics.get("export_blocking_reasons")
            )
            del mlx_qem_raw
            del mlx_qem_input
            del mlx_qem_cleaned
            del mlx_qem_repaired
        else:
            simplified, simplify_stats = _timed_stage(
                diagnostics,
                "simplify_mesh",
                lambda: simplify_mesh(
                    repaired_cleaned.vertices,
                    repaired_cleaned.faces,
                    target_faces=resolved_target_faces,
                    min_component_faces=min_component_faces,
                    backend="qem",
                    small_boundary_loop_fill_max_edges=resolved_small_boundary_loop_fill_max_edges,
                    small_boundary_loop_fill_max_perimeter=resolved_small_boundary_loop_fill_max_perimeter,
                ),
                memory_monitor=memory_monitor,
            )
            simplify_stats["pipeline"] = "single-layer-coarse-repair-qem"
        simplify_stats["requested_backend"] = requested_simplifier_backend
        simplify_stats["coarse_backend"] = "topology-aware"
        simplify_stats["coarse_target_faces"] = coarse_target_faces
        simplify_stats["coarse_final_faces"] = int(coarse_simplified.faces.shape[0])
        simplify_stats["repair_backend"] = str(repair_stats.get("backend", "unknown"))
        simplify_stats["remesh_backend"] = "single-layer-flexi-dual-grid"
        simplify_stats["remesh_equivalence_status"] = "single-layer-alternative-measurement-pending"
        simplify_stats["remesh_surface_representation"] = "original-flexi-dual-grid-surface"
        simplify_stats["remesh_single_surface_ready"] = False
        del coarse_simplified
        del repaired
        del repaired_cleaned
    elif requested_simplifier_backend == "mlx-qem":
        mlx_qem_raw, simplify_stats = _timed_stage(
            diagnostics,
            "simplify_mesh",
            lambda: simplify_mesh_mlx_parallel_qem(
                simplify_source.vertices,
                simplify_source.faces,
                target_faces=resolved_target_faces,
                max_rounds=128,
                topology_policy="manifold-preserving",
            ),
            memory_monitor=memory_monitor,
        )
        simplify_stats["pipeline"] = "narrow-band-remesh-mlx-parallel-qem"
        mlx_qem_raw_metrics = _timed_stage(
            diagnostics,
            "mlx_qem_raw_metrics",
            lambda: mesh_metrics(mlx_qem_raw.vertices, mlx_qem_raw.faces),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["mlx_qem_raw_metrics"]["metrics"] = mlx_qem_raw_metrics
        mlx_qem_cleaned, mlx_qem_clean_stats = _timed_stage(
            diagnostics,
            "clean_mlx_qem_mesh",
            lambda: clean_mesh(
                mlx_qem_raw.vertices,
                mlx_qem_raw.faces,
                min_component_faces=min_component_faces,
            ),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["clean_mlx_qem_mesh"].update(
            _mesh_shape(mlx_qem_cleaned, "cleaned_mlx_qem")
        )
        diagnostics["stages"]["clean_mlx_qem_mesh"]["stats"] = mlx_qem_clean_stats
        mlx_qem_repaired, mlx_qem_repair_stats = _timed_stage(
            diagnostics,
            "repair_mlx_qem_mesh",
            lambda: repair_nonmanifold_mesh(mlx_qem_cleaned.vertices, mlx_qem_cleaned.faces),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["repair_mlx_qem_mesh"].update(
            _mesh_shape(mlx_qem_repaired, "repaired_mlx_qem")
        )
        diagnostics["stages"]["repair_mlx_qem_mesh"]["stats"] = mlx_qem_repair_stats
        simplified, mlx_qem_final_clean_stats = _timed_stage(
            diagnostics,
            "clean_repaired_mlx_qem_mesh",
            lambda: clean_mesh(
                mlx_qem_repaired.vertices,
                mlx_qem_repaired.faces,
                min_component_faces=min_component_faces,
            ),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["clean_repaired_mlx_qem_mesh"].update(
            _mesh_shape(simplified, "cleaned_repaired_mlx_qem")
        )
        diagnostics["stages"]["clean_repaired_mlx_qem_mesh"]["stats"] = mlx_qem_final_clean_stats
        simplified, mlx_qem_hole_fill_stats = _timed_stage(
            diagnostics,
            "fill_repaired_mlx_qem_holes",
            lambda: fill_holes(
                simplified.vertices,
                simplified.faces,
                max_hole_perimeter=resolved_small_boundary_loop_fill_max_perimeter,
            ),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["fill_repaired_mlx_qem_holes"].update(
            _mesh_shape(simplified, "filled_repaired_mlx_qem")
        )
        diagnostics["stages"]["fill_repaired_mlx_qem_holes"]["stats"] = mlx_qem_hole_fill_stats
        simplify_stats["qem_raw_final_vertices"] = int(mlx_qem_raw.vertices.shape[0])
        simplify_stats["qem_raw_final_faces"] = int(mlx_qem_raw.faces.shape[0])
        simplify_stats["post_qem_repair_backend"] = str(mlx_qem_repair_stats.get("backend", "unknown"))
        simplify_stats["post_qem_hole_fill_backend"] = str(
            mlx_qem_hole_fill_stats.get("backend", "unknown")
        )
        simplify_stats["post_qem_hole_fill_filled_loops"] = int(
            mlx_qem_hole_fill_stats.get("filled_loops", 0)
        )
        simplify_stats["post_qem_hole_fill_residual_clean_boundary_loops"] = int(
            mlx_qem_hole_fill_stats.get("residual_clean_boundary_loops", 0)
        )
        simplify_stats["post_qem_final_vertices"] = int(simplified.vertices.shape[0])
        simplify_stats["post_qem_final_faces"] = int(simplified.faces.shape[0])
        simplify_stats["final_vertices"] = int(simplified.vertices.shape[0])
        simplify_stats["final_faces"] = int(simplified.faces.shape[0])
        simplify_stats["target_reached"] = int(mlx_qem_raw.faces.shape[0]) <= resolved_target_faces
        simplify_stats["final_cleanup_exceeds_target"] = int(simplified.faces.shape[0]) > resolved_target_faces
        del mlx_qem_raw
        del mlx_qem_cleaned
        del mlx_qem_repaired
    else:
        simplified, simplify_stats = _timed_stage(
            diagnostics,
            "simplify_mesh",
            lambda: simplify_mesh(
                simplify_source.vertices,
                simplify_source.faces,
                target_faces=resolved_target_faces,
                min_component_faces=min_component_faces,
                backend=requested_simplifier_backend,
                small_boundary_loop_fill_max_edges=resolved_small_boundary_loop_fill_max_edges,
                small_boundary_loop_fill_max_perimeter=resolved_small_boundary_loop_fill_max_perimeter,
            ),
            memory_monitor=memory_monitor,
        )
    if requested_simplifier_backend in {"mlx-qem", "single-layer-qem", "single-layer-mlx-qem"}:
        simplified, orientation_stats = _timed_stage(
            diagnostics,
            "unify_face_orientations",
            lambda: unify_face_orientations(simplified.vertices, simplified.faces),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["unify_face_orientations"].update(
            _mesh_shape(simplified, "orientation_unified")
        )
        diagnostics["stages"]["unify_face_orientations"]["stats"] = orientation_stats
        simplify_stats["orientation_backend"] = str(orientation_stats.get("backend", "unknown"))
        simplify_stats["orientation_algorithm"] = str(orientation_stats.get("algorithm", "unknown"))
        simplify_stats["orientation_faces_flipped"] = int(orientation_stats.get("faces_flipped", 0))
        simplify_stats["orientation_consistent"] = bool(
            orientation_stats.get("orientation_consistent", False)
        )
    diagnostics["stages"]["simplify_mesh"].update(_mesh_shape(simplified, "simplified"))
    diagnostics["stages"]["simplify_mesh"]["stats"] = simplify_stats
    if remesh:
        simplify_stats["remesh_backend"] = "native-udf-double-cover-dc"
        simplify_stats["remesh_equivalence_status"] = "cumesh-reference-mechanism-matched"
        simplify_stats["remesh_surface_representation"] = "udf-offset-double-cover"
        simplify_stats["remesh_single_surface_ready"] = False
        simplify_stats["remesh_reference_mechanism_ready"] = True
    del simplify_source
    if remesh:
        del remesh_cleaned
    del cleaned
    gc.collect()
    sample("after_simplify_mesh")

    post_metrics = _timed_stage(
        diagnostics,
        "export_metrics",
        lambda: mesh_metrics(simplified.vertices, simplified.faces),
        memory_monitor=memory_monitor,
    )
    diagnostics["stages"]["export_metrics"]["metrics"] = post_metrics

    def build_uv_mesh() -> NativeUvMesh:
        if resolved_uv_backend == "xatlas-global":
            return make_xatlas_uvs(simplified.vertices, simplified.faces, clustered=False)
        if resolved_uv_backend == "xatlas-clustered":
            return make_xatlas_uvs(simplified.vertices, simplified.faces, clustered=True)
        if resolved_uv_backend == "xatlas-parallel-spatial":
            return make_xatlas_uvs(
                simplified.vertices,
                simplified.faces,
                parallel_chunks=resolved_xatlas_parallel_chunks or 1,
                spatial_tile_padding=resolved_tile_padding,
            )
        if resolved_uv_backend == "xatlas-equivalent-native":
            return make_reference_uvs(
                simplified.vertices,
                simplified.faces,
                texture_resolution=texture_size,
            )
        if resolved_uv_backend == "native-chart":
            return make_native_chart_uvs(
                simplified.vertices,
                simplified.faces,
                chart_angle_degrees=resolved_chart_angle_degrees,
                tile_padding=resolved_tile_padding,
            )
        return make_face_atlas_uvs(simplified.vertices, simplified.faces, tile_padding=resolved_tile_padding)

    uv_mesh = _timed_stage(diagnostics, "uv", build_uv_mesh, memory_monitor=memory_monitor)
    diagnostics["stages"]["uv"].update(_uv_shape(uv_mesh))
    del simplified
    gc.collect()
    sample("after_uv")

    baked = _timed_stage(
        diagnostics,
        "texture_bake",
        lambda: bake_pbr_texture(
            uv_mesh,
            texture_coordinates,
            texture_attributes,
            texture_size=texture_size,
            decode_resolution=texture_decode_resolution,
            voxel_size=texture_voxel_size,
            max_texture_pixels=resolved_max_texture_pixels,
            source_vertices=source_projection_vertices,
            source_faces=source_projection_faces,
            source_projection=source_projection,
            source_projection_fallback_mode=source_projection_fallback_mode,
            source_projection_fallback_neighbors=source_projection_fallback_neighbors,
            source_projection_fallback_max_distance_voxels=source_projection_fallback_max_distance_voxels,
            render_padding=render_padding,
            postprocess=texture_postprocess,
            expose_raw_postprocess_inputs=expose_raw_postprocess_inputs,
        ),
        memory_monitor=memory_monitor,
    )
    diagnostics["stages"]["texture_bake"].update(_texture_shape(baked))
    raw_texture_inputs: dict[str, np.ndarray] | None = None
    if expose_raw_postprocess_inputs and baked.raw_base_color_rgba is not None:
        raw_texture_inputs = {
            "raw_base_color_rgba": baked.raw_base_color_rgba,
            "raw_metallic_roughness": baked.raw_metallic_roughness,
            "raw_coverage_status": baked.raw_coverage_status,
        }
    del texture_coordinates, texture_attributes, source_projection_vertices, source_projection_faces
    gc.collect()
    sample("after_texture_bake")

    if reference is not None:
        diagnostics["reference"] = reference
        diagnostics["reference_comparison"] = _reference_comparison(diagnostics, reference)

    quality = summarize_ovoxel_export_quality(
        model_identity["family"],
        simplify_stats,
        post_metrics,
        baked.stats,
        reference,
        quality_preset=resolved_quality_preset,
        uv_stats=uv_mesh.stats,
    )
    if model_identity["family"] == "pixal3d":
        chart_uv_candidate = _native_chart_uv_candidate_status(
            uv_mesh.stats,
            baked.stats,
            resolved_uv_backend,
        )
        quality["native_chart_uv_candidate"] = chart_uv_candidate
        quality["xatlas_chart_parity"] = _xatlas_chart_parity_summary(
            reference,
            uv_mesh.stats,
            baked.stats,
            resolved_uv_backend,
        )
        if chart_uv_candidate.get("status") == "quality_blocked":
            quality["warnings"] = tuple(
                [*quality["warnings"], "native_chart_uv_candidate_quality_blocked"]
            )
        quality["upstream_export_settings"] = _upstream_export_settings_summary(
            resolved_target_faces,
            texture_size,
            simplify_stats,
            baked.stats,
            quality,
        )
        quality["production_equivalence"] = _production_equivalence_summary(quality, None)
    else:
        quality["production_equivalence"] = unavailable_production_equivalence(
            model_identity["family"],
            artifact_ready=bool(quality["artifact_ready"]),
        )
    diagnostics["quality"] = quality

    gltf_uv_mesh = NativeUvMesh(
        vertices=vertices_to_gltf_y_up(
            uv_mesh.vertices,
            source_coordinate_system=resolved_source_coordinate_system,
        ),
        faces=uv_mesh.faces,
        uvs=uv_mesh.uvs,
        stats=uv_mesh.stats,
    )
    glb = _timed_stage(
        diagnostics,
        "write_glb",
        lambda: write_textured_glb(
            glb_path,
            gltf_uv_mesh,
            base_color_rgba=baked.base_color_rgba,
            metallic_roughness=baked.metallic_roughness,
            generator=f"mlx-spatial SpatialKit {model_identity['label']}",
            mesh_name=f"{model_identity['asset_prefix']}_TexturedMesh",
            material_name=f"{model_identity['asset_prefix']}_PBR",
            metadata={
                "pipeline_type": decoded_metadata_value(diagnostics, "pipeline_type"),
                "shape_decoder_artifact": str(shape_path),
                "texture_decoder_artifact": str(texture_path),
                "texture_size": int(baked.texture_size),
                "target_faces": resolved_target_faces,
                "quality_preset": resolved_quality_preset,
                "uv_backend": resolved_uv_backend,
                "uv_stats_backend": str(uv_mesh.stats.get("backend")),
                "xatlas_parallel_chunks": resolved_xatlas_parallel_chunks,
                "chart_angle_degrees": resolved_chart_angle_degrees,
                "bake_backend": str(baked.stats.get("backend")),
                "coverage_ratio": float(baked.stats.get("coverage_ratio", 0.0)),
                "raw_coverage_ratio": float(baked.stats.get("raw_coverage_ratio", 0.0)),
                "simplifier_backend": quality["simplifier_backend"],
                "simplifier_quality_tier": quality["simplifier_quality_tier"],
                "production_quality_ready": bool(quality["production_quality_ready"]),
                "production_equivalence_ready": bool(quality["production_equivalence"]["ready"]),
                "source_coordinate_system": resolved_source_coordinate_system,
            },
        ),
        memory_monitor=memory_monitor,
    )
    diagnostics["stages"]["write_glb"]["artifact"] = glb.metadata
    glb_inspection = inspect_glb(glb.path)
    diagnostics["stages"]["write_glb"]["inspection"] = glb_inspection
    quality["glb_viewer_compatibility"] = _glb_viewer_compatibility_summary(glb_inspection)
    diagnostics["quality"] = quality
    sample("after_write_glb")
    del gltf_uv_mesh

    if reference is not None:
        reference_glb = _reference_glb_path(reference)
        if reference_glb is not None:
            visual_report = _timed_stage(
                diagnostics,
                "visual_compare",
                lambda: compare_textured_glbs(
                    glb.path,
                    reference_glb,
                    output_dir=glb.path.parent / "visual_parity",
                ),
                memory_monitor=memory_monitor,
            )
            diagnostics["visual_comparison"] = _visual_comparison_summary(
                visual_report,
                quality.get("upstream_export_settings"),
                texture_stats=baked.stats,
                export_metrics=post_metrics,
            )
            quality["rendered_visual_ready"] = bool(diagnostics["visual_comparison"]["rendered_visual_ready"])
            quality["production_equivalence"] = _production_equivalence_summary(
                quality,
                diagnostics["visual_comparison"],
            )
            diagnostics["quality"] = quality

    diagnostics["result"] = {
        "ready": bool(quality["artifact_ready"]),
        "artifact_ready": bool(quality["artifact_ready"]),
        "rendered_visual_ready": bool(quality.get("rendered_visual_ready", False)),
        "production_quality_ready": bool(quality["production_quality_ready"]),
        "production_equivalence_ready": bool(quality["production_equivalence"]["ready"]),
        "remaining_parity_boundaries": quality["production_equivalence"]["remaining_parity_boundaries"],
        "equivalence_blockers": quality["production_equivalence"]["blockers"],
        "quality_warnings": quality["warnings"],
        "model_glb": str(glb.path),
        "diagnostics_json": str(resolved_diagnostics_path),
        "bytes_written": int(glb.bytes_written),
    }
    manifest_path = glb.path.parent / "artifact-manifest.json"
    run_manifest = _build_ovoxel_run_manifest(
        decoded_dir=source_dir,
        shape_path=shape_path,
        texture_path=texture_path,
        glb=glb,
        diagnostics_path=resolved_diagnostics_path,
        diagnostics=diagnostics,
        fixture_manifest=fixture_manifest,
        reference=reference,
    )
    diagnostics["artifact_manifest"] = {
        "path": str(manifest_path),
        "lineage_id": run_manifest["lineage_id"],
        "roles": tuple(run_manifest["roles"]),
    }
    _write_json_atomic(manifest_path, run_manifest)
    memory_monitor.stop()
    diagnostics["memory"] = memory_monitor.summary()
    _write_json_atomic(resolved_diagnostics_path, diagnostics)
    return OVoxelGlbExportResult(
        glb=glb,
        diagnostics_path=resolved_diagnostics_path,
        diagnostics=diagnostics,
        raw_texture_inputs=raw_texture_inputs,
    )


def _mesh_shape(mesh: NativeMesh, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_vertices": int(mesh.vertices.shape[0]),
        f"{prefix}_faces": int(mesh.faces.shape[0]),
    }


def _uv_shape(mesh: NativeUvMesh) -> dict[str, Any]:
    return {
        "vertices_shape": tuple(int(dim) for dim in mesh.vertices.shape),
        "faces_shape": tuple(int(dim) for dim in mesh.faces.shape),
        "uvs_shape": tuple(int(dim) for dim in mesh.uvs.shape),
        "stats": mesh.stats,
    }


def _texture_shape(baked: Any) -> dict[str, Any]:
    return {
        "base_color_shape": tuple(int(dim) for dim in baked.base_color_rgba.shape),
        "metallic_roughness_shape": tuple(int(dim) for dim in baked.metallic_roughness.shape),
        "coverage_status_shape": tuple(int(dim) for dim in baked.coverage_status.shape),
        "stats": baked.stats,
    }


def _resolve_ovoxel_export_paths(output: str | Path, diagnostics_path: str | Path | None) -> tuple[Path, Path]:
    output_path = Path(output)
    glb_path = output_path if output_path.suffix.lower() == ".glb" else output_path / "model.glb"
    if diagnostics_path is None:
        diag_path = glb_path.with_name("diagnostics.json")
    else:
        diag_path = Path(diagnostics_path)
    if diag_path.suffix.lower() != ".json":
        raise ValueError("O-Voxel export diagnostics path must end with .json")
    return glb_path, diag_path


def _resolve_positive_int(*values: Any, default: int, name: str) -> int:
    for value in values:
        if value is None:
            continue
        resolved = int(value)
        if resolved <= 0:
            raise ValueError(f"{name} must be positive")
        return resolved
    return int(default)


def _resolve_ovoxel_export_settings(
    decoded_dir: Path,
    model_family: str,
    quality_preset: str,
    target_faces: int | None,
    *,
    fixture_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preset = _normalize_ovoxel_quality_preset(quality_preset)
    has_pixal3d_profile = model_family == "pixal3d"
    reference = (
        _load_pixal3d_reference_trace(decoded_dir, fixture_manifest=fixture_manifest)
        if has_pixal3d_profile
        else None
    )
    if target_faces is not None:
        resolved_target_faces = int(target_faces)
        target_source = "explicit"
    elif preset == "reference-target" and reference is not None and reference.get("final_faces") is not None:
        resolved_target_faces = int(reference["final_faces"])
        target_source = "reference_final_faces"
    elif preset == "reference-target" and has_pixal3d_profile:
        resolved_target_faces = PIXAL3D_REFERENCE_TARGET_FACES
        target_source = "reference_default"
    elif preset == "reference-target":
        resolved_target_faces = OVOXEL_DEFAULT_TARGET_FACES
        target_source = "ovoxel_production_default"
    else:
        resolved_target_faces = OVOXEL_PREVIEW_TARGET_FACES
        target_source = "preview_default"
    if resolved_target_faces <= 0:
        raise ValueError("target_faces must be positive")
    return {
        "quality_preset": preset,
        "target_faces": resolved_target_faces,
        "target_faces_source": target_source,
        "reference": reference,
        "reference_profile": "pixal3d-upstream" if has_pixal3d_profile else None,
    }


def _resolve_pixal3d_export_settings(
    decoded_dir: Path,
    quality_preset: str,
    target_faces: int | None,
    *,
    fixture_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for Pixal3D quality-evidence tests."""

    return _resolve_ovoxel_export_settings(
        decoded_dir,
        "pixal3d",
        quality_preset,
        target_faces,
        fixture_manifest=fixture_manifest,
    )


def _nested_get(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        tmp_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# Compatibility aliases for callers written before the decoded contract became
# model-neutral. New code should use the O-Voxel names above.
Pixal3DDecodedInputs = DecodedOVoxelInputs
Pixal3DGlbExportResult = OVoxelGlbExportResult
export_pixal3d_glb = export_decoded_ovoxel_glb
load_pixal3d_decoded_npz = load_decoded_ovoxel_npz
validate_pixal3d_decoded = validate_decoded_ovoxel


__all__ = [
    "DecodedOVoxelInputs",
    "NativeGlbArtifact",
    "NativeUvMesh",
    "OVoxelGlbExportResult",
    "Pixal3DGlbExportResult",
    "Pixal3DDecodedInputs",
    "backend_info",
    "export_decoded_ovoxel_glb",
    "export_pixal3d_glb",
    "load_decoded_ovoxel_npz",
    "load_pixal3d_decoded_npz",
    "make_face_atlas_uvs",
    "make_native_chart_uvs",
    "make_reference_uvs",
    "make_xatlas_uvs",
    "textured_glb_payload",
    "validate_decoded_ovoxel",
    "validate_pixal3d_decoded",
    "write_textured_glb",
]
