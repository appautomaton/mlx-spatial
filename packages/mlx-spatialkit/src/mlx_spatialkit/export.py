"""Thin Python entry points for native export functionality."""

from __future__ import annotations

import gc
import json
import os
import resource
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

from ._native import (
    backend_info,
    make_face_atlas_uvs as _make_face_atlas_uvs,
    make_native_chart_uvs as _make_native_chart_uvs,
    textured_glb_payload as _textured_glb_payload,
    validate_pixal3d_shape_fields,
    validate_pixal3d_texture_attributes,
)
from .glb_compare import compare_textured_glbs, inspect_glb
from .mesh import NativeMesh, clean_mesh, extract_flexi_dual_grid, mesh_metrics, simplify_mesh

_T = TypeVar("_T")

PIXAL3D_PREVIEW_TARGET_FACES = 50_000
PIXAL3D_REFERENCE_TARGET_FACES = 212_542
PIXAL3D_REFERENCE_FINAL_COVERAGE_THRESHOLD = 0.50
PIXAL3D_REFERENCE_FACE_RATIO_MIN = 0.80
PIXAL3D_REFERENCE_FACE_RATIO_MAX = 1.25
PIXAL3D_MEMORY_POLL_INTERVAL_SEC = 0.25
PIXAL3D_UPSTREAM_EXPORT_TARGET_FACES = 1_000_000
PIXAL3D_UPSTREAM_EXPORT_TEXTURE_SIZE = 4096
PIXAL3D_UPSTREAM_EXPORT_FACE_RETENTION_MIN = 0.60
PIXAL3D_CHART_UV_GLOBAL_COVERAGE_MIN = 0.50
PIXAL3D_CHART_UV_SURFACE_OCCUPANCY_MIN = 0.50
PIXAL3D_CHART_UV_SURFACE_VISIBLE_MIN = 0.50
PIXAL3D_FACE_ATLAS_TILE_PADDING = 0.08
PIXAL3D_NATIVE_CHART_TILE_PADDING = 0.02


@dataclass(frozen=True)
class Pixal3DDecodedInputs:
    """Decoded Pixal3D model-stage arrays validated at the native boundary."""

    shape_coordinates: np.ndarray
    shape_fields: np.ndarray
    texture_coordinates: np.ndarray
    texture_attributes: np.ndarray
    contracts: dict[str, Any]
    shape_metadata: dict[str, Any]
    texture_metadata: dict[str, Any]
    texture_spatial_shape: tuple[int, int, int] | None
    texture_batch_size: int | None
    texture_decode_resolution: int | None
    texture_voxel_size: float | None


@dataclass(frozen=True)
class NativeUvMesh:
    """UV-ready triangle mesh prepared by the native backend."""

    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    stats: dict[str, Any]


@dataclass(frozen=True)
class NativeGlbArtifact:
    """Written native GLB artifact metadata."""

    path: Path
    format: str
    bytes_written: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Pixal3DGlbExportResult:
    """End-to-end native Pixal3D GLB export result."""

    glb: NativeGlbArtifact
    diagnostics_path: Path
    diagnostics: dict[str, Any]


def validate_pixal3d_decoded(
    shape_coordinates: np.ndarray,
    shape_fields: np.ndarray,
    texture_coordinates: np.ndarray,
    texture_attributes: np.ndarray,
) -> dict[str, Any]:
    """Validate Pixal3D decoded arrays through native contract checks."""

    shape_contract = validate_pixal3d_shape_fields(shape_coordinates, shape_fields)
    texture_contract = validate_pixal3d_texture_attributes(texture_coordinates, texture_attributes)
    return {"shape": shape_contract, "texture": texture_contract}


def make_face_atlas_uvs(vertices: np.ndarray, faces: np.ndarray, *, tile_padding: float = 0.08) -> NativeUvMesh:
    """Create a deterministic native face-atlas UV mesh."""

    result = _make_face_atlas_uvs(vertices, faces, float(tile_padding))
    return NativeUvMesh(
        vertices=np.asarray(result["vertices"]),
        faces=np.asarray(result["faces"]),
        uvs=np.asarray(result["uvs"]),
        stats=dict(result["stats"]),
    )


def make_native_chart_uvs(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    chart_angle_degrees: float = 45.0,
    tile_padding: float = 0.04,
) -> NativeUvMesh:
    """Create a deterministic native chart UV mesh."""

    result = _make_native_chart_uvs(vertices, faces, float(chart_angle_degrees), float(tile_padding))
    return NativeUvMesh(
        vertices=np.asarray(result["vertices"]),
        faces=np.asarray(result["faces"]),
        uvs=np.asarray(result["uvs"]),
        stats=dict(result["stats"]),
    )


def textured_glb_payload(
    mesh: NativeUvMesh,
    *,
    base_color_rgba: np.ndarray,
    metallic_roughness: np.ndarray,
    generator: str = "mlx-spatialkit",
    mesh_name: str = "TexturedMesh",
    material_name: str = "PBRMaterial",
) -> bytes:
    """Build a native self-contained GLB 2.0 payload."""

    return bytes(
        _textured_glb_payload(
            mesh.vertices,
            mesh.faces,
            mesh.uvs,
            base_color_rgba,
            metallic_roughness,
            str(generator),
            str(mesh_name),
            str(material_name),
        )
    )


def write_textured_glb(
    path: str | Path,
    mesh: NativeUvMesh,
    *,
    base_color_rgba: np.ndarray,
    metallic_roughness: np.ndarray,
    generator: str = "mlx-spatialkit",
    mesh_name: str = "TexturedMesh",
    material_name: str = "PBRMaterial",
    metadata: dict[str, Any] | None = None,
) -> NativeGlbArtifact:
    """Write a native GLB payload to disk."""

    output = Path(path)
    if output.suffix.lower() != ".glb":
        raise ValueError("native textured exports require a .glb output path")
    payload = textured_glb_payload(
        mesh,
        base_color_rgba=base_color_rgba,
        metallic_roughness=metallic_roughness,
        generator=generator,
        mesh_name=mesh_name,
        material_name=material_name,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_name(f".{output.name}.tmp")
    try:
        tmp_path.write_bytes(payload)
        tmp_path.replace(output)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    payload_metadata = {
        "stage": "textured_glb",
        "format": "glb",
        "bytes_written": int(output.stat().st_size),
        "generator": str(generator),
        "mesh_name": str(mesh_name),
        "material_name": str(material_name),
        **(metadata or {}),
    }
    return NativeGlbArtifact(
        path=output,
        format="glb",
        bytes_written=int(output.stat().st_size),
        metadata=payload_metadata,
    )


def export_pixal3d_glb(
    decoded_dir: str | Path,
    output: str | Path,
    *,
    texture_size: int = 1024,
    target_faces: int | None = None,
    quality_preset: str = "preview",
    grid_size: int | None = None,
    min_component_faces: int = 32,
    uv_backend: str = "face-atlas",
    chart_angle_degrees: float = 45.0,
    tile_padding: float | None = None,
    max_texture_pixels: int | None = None,
    diagnostics_path: str | Path | None = None,
) -> Pixal3DGlbExportResult:
    """Convert decoded Pixal3D NPZ artifacts into a textured GLB through native hot paths."""

    from .texture import bake_pbr_texture

    source_dir = Path(decoded_dir)
    if not source_dir.is_dir():
        raise ValueError(f"decoded Pixal3D directory does not exist: {source_dir}")
    if texture_size <= 0:
        raise ValueError("texture_size must be positive")
    if grid_size is not None and grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if min_component_faces <= 0:
        raise ValueError("min_component_faces must be positive")
    if max_texture_pixels is not None and max_texture_pixels <= 0:
        raise ValueError("max_texture_pixels must be positive")
    resolved_uv_backend = _resolve_pixal3d_uv_backend(uv_backend)
    resolved_chart_angle_degrees = _resolve_chart_angle_degrees(chart_angle_degrees)
    resolved_tile_padding, tile_padding_source = _resolve_tile_padding(tile_padding, resolved_uv_backend)
    glb_path, resolved_diagnostics_path = _resolve_pixal3d_export_paths(output, diagnostics_path)
    shape_path = source_dir / "shape_decoder_fields.npz"
    texture_path = source_dir / "texture_decoder_pbr.npz"
    if not shape_path.exists():
        raise ValueError(f"missing decoded shape artifact: {shape_path}")
    if not texture_path.exists():
        raise ValueError(f"missing decoded texture artifact: {texture_path}")

    export_settings = _resolve_pixal3d_export_settings(source_dir, quality_preset, target_faces)
    reference = export_settings["reference"]
    resolved_quality_preset = str(export_settings["quality_preset"])
    resolved_target_faces = int(export_settings["target_faces"])
    requested_simplifier_backend = _simplifier_backend_for_quality_preset(resolved_quality_preset)

    diagnostics: dict[str, Any] = {
        "stage": "pixal3d_glb_export",
        "source_dir": str(source_dir),
        "output_path": str(glb_path),
        "diagnostics_path": str(resolved_diagnostics_path),
        "settings": {
            "quality_preset": resolved_quality_preset,
            "texture_size": int(texture_size),
            "target_faces": resolved_target_faces,
            "requested_simplifier_backend": requested_simplifier_backend,
            "requested_target_faces": int(target_faces) if target_faces is not None else None,
            "target_faces_source": export_settings["target_faces_source"],
            "reference_available": reference is not None,
            "reference_trace_path": str(reference["trace_path"]) if reference is not None else None,
            "reference_target_faces": reference.get("final_faces") if reference is not None else None,
            "reference_texture_size": reference.get("texture_size") if reference is not None else None,
            "reference_xatlas_face_guard": reference.get("xatlas_face_guard") if reference is not None else None,
            "grid_size": int(grid_size) if grid_size is not None else None,
            "min_component_faces": int(min_component_faces),
            "requested_uv_backend": str(uv_backend),
            "uv_backend": resolved_uv_backend,
            "chart_angle_degrees": resolved_chart_angle_degrees,
            "tile_padding": resolved_tile_padding,
            "tile_padding_source": tile_padding_source,
            "max_texture_pixels": int(max_texture_pixels) if max_texture_pixels is not None else None,
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
        lambda: load_pixal3d_decoded_npz(shape_path, texture_path),
        memory_monitor=memory_monitor,
    )
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

    simplified, simplify_stats = _timed_stage(
        diagnostics,
        "simplify_mesh",
        lambda: simplify_mesh(
            cleaned.vertices,
            cleaned.faces,
            target_faces=resolved_target_faces,
            min_component_faces=min_component_faces,
            backend=requested_simplifier_backend,
        ),
        memory_monitor=memory_monitor,
    )
    diagnostics["stages"]["simplify_mesh"].update(_mesh_shape(simplified, "simplified"))
    diagnostics["stages"]["simplify_mesh"]["stats"] = simplify_stats
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
        ),
        memory_monitor=memory_monitor,
    )
    diagnostics["stages"]["texture_bake"].update(_texture_shape(baked))
    del texture_coordinates, texture_attributes
    gc.collect()
    sample("after_texture_bake")

    if reference is not None:
        diagnostics["reference"] = reference
        diagnostics["reference_comparison"] = _reference_comparison(diagnostics, reference)

    quality = _export_quality_summary(
        simplify_stats,
        post_metrics,
        baked.stats,
        reference,
        quality_preset=resolved_quality_preset,
    )
    chart_uv_candidate = _native_chart_uv_candidate_status(
        uv_mesh.stats,
        baked.stats,
        resolved_uv_backend,
    )
    quality["native_chart_uv_candidate"] = chart_uv_candidate
    if chart_uv_candidate.get("status") == "quality_blocked":
        quality["warnings"] = tuple([*quality["warnings"], "native_chart_uv_candidate_quality_blocked"])
    quality["upstream_export_settings"] = _upstream_export_settings_summary(
        resolved_target_faces,
        texture_size,
        simplify_stats,
        baked.stats,
        quality,
    )
    diagnostics["quality"] = quality

    glb = _timed_stage(
        diagnostics,
        "write_glb",
        lambda: write_textured_glb(
            glb_path,
            uv_mesh,
            base_color_rgba=baked.base_color_rgba,
            metallic_roughness=baked.metallic_roughness,
            generator="mlx-spatialkit Pixal3D",
            mesh_name="Pixal3D_TexturedMesh",
            material_name="Pixal3D_PBR",
            metadata={
                "pipeline_type": decoded_metadata_value(diagnostics, "pipeline_type"),
                "shape_decoder_artifact": str(shape_path),
                "texture_decoder_artifact": str(texture_path),
                "texture_size": int(baked.texture_size),
                "target_faces": resolved_target_faces,
                "quality_preset": resolved_quality_preset,
                "uv_backend": resolved_uv_backend,
                "uv_stats_backend": str(uv_mesh.stats.get("backend")),
                "chart_angle_degrees": resolved_chart_angle_degrees,
                "bake_backend": str(baked.stats.get("backend")),
                "coverage_ratio": float(baked.stats.get("coverage_ratio", 0.0)),
                "raw_coverage_ratio": float(baked.stats.get("raw_coverage_ratio", 0.0)),
                "simplifier_backend": quality["simplifier_backend"],
                "simplifier_quality_tier": quality["simplifier_quality_tier"],
                "production_quality_ready": bool(quality["production_quality_ready"]),
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
            )

    diagnostics["result"] = {
        "ready": bool(quality["artifact_ready"]),
        "artifact_ready": bool(quality["artifact_ready"]),
        "production_quality_ready": bool(quality["production_quality_ready"]),
        "quality_warnings": quality["warnings"],
        "model_glb": str(glb.path),
        "diagnostics_json": str(resolved_diagnostics_path),
        "bytes_written": int(glb.bytes_written),
    }
    memory_monitor.stop()
    diagnostics["memory"] = memory_monitor.summary()
    _write_json_atomic(resolved_diagnostics_path, diagnostics)
    return Pixal3DGlbExportResult(glb=glb, diagnostics_path=resolved_diagnostics_path, diagnostics=diagnostics)


def _load_npz_array(payload: np.lib.npyio.NpzFile, key: str, path: Path) -> np.ndarray:
    if key not in payload.files:
        raise ValueError(f"{path} is missing required array {key!r}")
    return np.asarray(payload[key])


def _load_npz_metadata(payload: np.lib.npyio.NpzFile, path: Path) -> dict[str, Any]:
    if "metadata_json" not in payload.files:
        return {}
    raw = payload["metadata_json"]
    try:
        text = str(raw.item() if raw.shape == () else raw.tolist())
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} contains invalid metadata_json") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} metadata_json must decode to an object")
    return value


def _load_optional_scalar(payload: np.lib.npyio.NpzFile, key: str, path: Path) -> Any:
    if key not in payload.files:
        return None
    value = payload[key]
    if value.shape != ():
        raise ValueError(f"{path} optional scalar {key!r} must be rank 0")
    return value.item()


def load_pixal3d_decoded_npz(
    shape_decoder_path: str | Path,
    texture_decoder_path: str | Path,
) -> Pixal3DDecodedInputs:
    """Load Pixal3D decoded NPZ files and validate their native contracts."""

    shape_path = Path(shape_decoder_path)
    texture_path = Path(texture_decoder_path)
    with np.load(shape_path) as shape_payload:
        shape_coordinates = _load_npz_array(shape_payload, "coordinates", shape_path)
        shape_fields = _load_npz_array(shape_payload, "fields", shape_path)
        shape_metadata = _load_npz_metadata(shape_payload, shape_path)
    with np.load(texture_path) as texture_payload:
        texture_coordinates = _load_npz_array(texture_payload, "coordinates", texture_path)
        texture_attributes = _load_npz_array(texture_payload, "attributes", texture_path)
        texture_metadata = _load_npz_metadata(texture_payload, texture_path)
        texture_spatial_shape = (
            tuple(int(dim) for dim in _load_npz_array(texture_payload, "spatial_shape", texture_path))
            if "spatial_shape" in texture_payload.files
            else None
        )
        texture_batch_size = _load_optional_scalar(texture_payload, "batch_size", texture_path)
        texture_decode_resolution = _load_optional_scalar(texture_payload, "decode_resolution", texture_path)
        texture_voxel_size = _load_optional_scalar(texture_payload, "voxel_size", texture_path)
    contracts = validate_pixal3d_decoded(
        shape_coordinates,
        shape_fields,
        texture_coordinates,
        texture_attributes,
    )
    return Pixal3DDecodedInputs(
        shape_coordinates=shape_coordinates,
        shape_fields=shape_fields,
        texture_coordinates=texture_coordinates,
        texture_attributes=texture_attributes,
        contracts=contracts,
        shape_metadata=shape_metadata,
        texture_metadata=texture_metadata,
        texture_spatial_shape=texture_spatial_shape,
        texture_batch_size=int(texture_batch_size) if texture_batch_size is not None else None,
        texture_decode_resolution=(
            None
            if texture_decode_resolution is None or int(texture_decode_resolution) < 0
            else int(texture_decode_resolution)
        ),
        texture_voxel_size=(
            None
            if texture_voxel_size is None or not np.isfinite(float(texture_voxel_size))
            else float(texture_voxel_size)
        ),
    )


def _timed_stage(
    diagnostics: dict[str, Any],
    name: str,
    fn: Callable[[], _T],
    *,
    memory_monitor: _ProcessMemoryMonitor | None = None,
) -> _T:
    start = time.perf_counter()
    try:
        if memory_monitor is None:
            return fn()
        with memory_monitor.track_stage(name):
            return fn()
    except BaseException:
        if memory_monitor is not None:
            memory_monitor.stop()
        raise
    finally:
        elapsed = time.perf_counter() - start
        diagnostics["timings_sec"][name] = elapsed
        diagnostics["stages"].setdefault(name, {})["seconds"] = elapsed


class _MemoryStageScope:
    def __init__(self, monitor: _ProcessMemoryMonitor, stage: str):
        self._monitor = monitor
        self._stage = stage
        self._prior_stage = "idle"

    def __enter__(self) -> None:
        self._prior_stage = self._monitor._set_active_stage(self._stage)
        self._monitor._set_stage_boundary(self._stage, "start", self._monitor.sample(f"{self._stage}:start"))

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._monitor._set_stage_boundary(self._stage, "end", self._monitor.sample(f"{self._stage}:end"))
        self._monitor._set_active_stage(self._prior_stage)


class _ProcessMemoryMonitor:
    def __init__(
        self,
        *,
        poll_interval_sec: float = PIXAL3D_MEMORY_POLL_INTERVAL_SEC,
        sample_fn: Callable[[], dict[str, Any]] | None = None,
    ):
        if poll_interval_sec <= 0:
            raise ValueError("poll_interval_sec must be positive")
        self._poll_interval_sec = float(poll_interval_sec)
        self._sample_fn = sample_fn or _memory_sample
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopped = False
        self._active_stage = "idle"
        self._sample_count = 0
        self._peak_current_rss_bytes: int | None = None
        self._peak_current_rss_label: str | None = None
        self._peak_current_rss_stage: str | None = None
        self._peak_max_rss_bytes: int | None = None
        self._peak_max_rss_label: str | None = None
        self._peak_max_rss_stage: str | None = None
        self._last_sample: dict[str, Any] | None = None
        self._stage_peaks: dict[str, dict[str, Any]] = {}

    @property
    def poll_interval_sec(self) -> float:
        return self._poll_interval_sec

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        self.sample("monitor_start")
        thread = threading.Thread(
            target=self._poll_loop,
            name="mlx-spatialkit-memory-monitor",
            daemon=True,
        )
        with self._lock:
            self._thread = thread
        thread.start()

    def stop(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            thread = self._thread
        self._stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, self._poll_interval_sec * 4.0))
        self.sample("monitor_stop")

    def sample(self, label: str) -> dict[str, Any]:
        sample = self._sample_fn()
        self._record(label, sample)
        return sample

    def track_stage(self, stage: str) -> _MemoryStageScope:
        return _MemoryStageScope(self, stage)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            stage_peaks = {stage: dict(values) for stage, values in sorted(self._stage_peaks.items())}
            return {
                "source": "process RSS from ps; high-water RSS from resource.getrusage(RUSAGE_SELF).ru_maxrss",
                "poll_interval_sec": self._poll_interval_sec,
                "sample_count": self._sample_count,
                "peak_current_rss_bytes": self._peak_current_rss_bytes,
                "peak_current_rss_label": self._peak_current_rss_label,
                "peak_current_rss_stage": self._peak_current_rss_stage,
                "peak_max_rss_bytes": self._peak_max_rss_bytes,
                "peak_max_rss_label": self._peak_max_rss_label,
                "peak_max_rss_stage": self._peak_max_rss_stage,
                "last_sample": dict(self._last_sample) if self._last_sample is not None else None,
                "stage_peaks": stage_peaks,
            }

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self._poll_interval_sec):
            self.sample("poll")

    def _set_active_stage(self, stage: str) -> str:
        with self._lock:
            prior = self._active_stage
            self._active_stage = stage
            if stage != "idle":
                self._stage_peaks.setdefault(stage, self._empty_stage_record())
            return prior

    def _set_stage_boundary(self, stage: str, boundary: str, sample: dict[str, Any]) -> None:
        current_rss = _sample_int(sample, "current_rss_bytes")
        max_rss = _sample_int(sample, "max_rss_bytes")
        with self._lock:
            record = self._stage_peaks.setdefault(stage, self._empty_stage_record())
            record[f"{boundary}_current_rss_bytes"] = current_rss
            record[f"{boundary}_max_rss_bytes"] = max_rss

    def _record(self, label: str, sample: dict[str, Any]) -> None:
        current_rss = _sample_int(sample, "current_rss_bytes")
        max_rss = _sample_int(sample, "max_rss_bytes")
        with self._lock:
            self._sample_count += 1
            self._last_sample = dict(sample)
            stage = self._active_stage
            if current_rss is not None and (
                self._peak_current_rss_bytes is None or current_rss > self._peak_current_rss_bytes
            ):
                self._peak_current_rss_bytes = current_rss
                self._peak_current_rss_label = label
                self._peak_current_rss_stage = stage
            if max_rss is not None and (self._peak_max_rss_bytes is None or max_rss > self._peak_max_rss_bytes):
                self._peak_max_rss_bytes = max_rss
                self._peak_max_rss_label = label
                self._peak_max_rss_stage = stage
            if stage == "idle":
                return
            record = self._stage_peaks.setdefault(stage, self._empty_stage_record())
            record["sample_count"] += 1
            if current_rss is not None and (
                record["peak_current_rss_bytes"] is None or current_rss > record["peak_current_rss_bytes"]
            ):
                record["peak_current_rss_bytes"] = current_rss
                record["peak_current_rss_label"] = label
            if max_rss is not None and (
                record["peak_max_rss_bytes"] is None or max_rss > record["peak_max_rss_bytes"]
            ):
                record["peak_max_rss_bytes"] = max_rss
                record["peak_max_rss_label"] = label

    @staticmethod
    def _empty_stage_record() -> dict[str, Any]:
        return {
            "sample_count": 0,
            "start_current_rss_bytes": None,
            "end_current_rss_bytes": None,
            "peak_current_rss_bytes": None,
            "peak_current_rss_label": None,
            "start_max_rss_bytes": None,
            "end_max_rss_bytes": None,
            "peak_max_rss_bytes": None,
            "peak_max_rss_label": None,
        }


def _sample_int(sample: dict[str, Any], key: str) -> int | None:
    value = sample.get(key)
    return None if value is None else int(value)


def _memory_sample() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss = int(usage.ru_maxrss)
    max_rss_bytes = max_rss if sys.platform == "darwin" else max_rss * 1024
    return {
        "pid": os.getpid(),
        "current_rss_bytes": _current_rss_bytes(),
        "max_rss_bytes": max_rss_bytes,
        "source": "ps rss plus resource.getrusage(RUSAGE_SELF).ru_maxrss",
    }


def _current_rss_bytes() -> int | None:
    try:
        output = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return int(output.strip()) * 1024
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


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


def _resolve_pixal3d_export_paths(output: str | Path, diagnostics_path: str | Path | None) -> tuple[Path, Path]:
    output_path = Path(output)
    glb_path = output_path if output_path.suffix.lower() == ".glb" else output_path / "model.glb"
    if diagnostics_path is None:
        diag_path = glb_path.with_name("diagnostics.json")
    else:
        diag_path = Path(diagnostics_path)
    if diag_path.suffix.lower() != ".json":
        raise ValueError("Pixal3D export diagnostics path must end with .json")
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


def decoded_metadata_value(diagnostics: dict[str, Any], key: str) -> Any:
    source = diagnostics.get("source", {})
    for section in ("shape_decoder", "texture_decoder"):
        metadata = source.get(section, {}).get("metadata", {})
        if key in metadata:
            return metadata[key]
    return None


def _resolve_pixal3d_export_settings(
    decoded_dir: Path,
    quality_preset: str,
    target_faces: int | None,
) -> dict[str, Any]:
    preset = _normalize_quality_preset(quality_preset)
    reference = _load_pixal3d_reference_trace(decoded_dir)
    if target_faces is not None:
        resolved_target_faces = int(target_faces)
        target_source = "explicit"
    elif preset == "reference-target" and reference is not None and reference.get("final_faces") is not None:
        resolved_target_faces = int(reference["final_faces"])
        target_source = "reference_final_faces"
    elif preset == "reference-target":
        resolved_target_faces = PIXAL3D_REFERENCE_TARGET_FACES
        target_source = "reference_default"
    else:
        resolved_target_faces = PIXAL3D_PREVIEW_TARGET_FACES
        target_source = "preview_default"
    if resolved_target_faces <= 0:
        raise ValueError("target_faces must be positive")
    return {
        "quality_preset": preset,
        "target_faces": resolved_target_faces,
        "target_faces_source": target_source,
        "reference": reference,
    }


def _normalize_quality_preset(value: str) -> str:
    preset = str(value).strip().lower().replace("_", "-")
    if preset in ("production", "reference", "reference-target"):
        return "reference-target"
    if preset == "preview":
        return "preview"
    raise ValueError("quality_preset must be 'preview' or 'reference-target'")


def _resolve_pixal3d_uv_backend(value: str) -> str:
    backend = str(value).strip().lower().replace("_", "-")
    if backend in ("face-atlas", "native-chart"):
        return backend
    raise ValueError("uv_backend must be 'face-atlas' or 'native-chart'")


def _resolve_chart_angle_degrees(value: float) -> float:
    angle = float(value)
    if not np.isfinite(angle) or angle < 0.0 or angle > 180.0:
        raise ValueError("chart_angle_degrees must be finite and in [0, 180]")
    return angle


def _resolve_tile_padding(value: float | None, uv_backend: str) -> tuple[float, str]:
    backend = _resolve_pixal3d_uv_backend(uv_backend)
    if value is None:
        if backend == "native-chart":
            return PIXAL3D_NATIVE_CHART_TILE_PADDING, "backend_default:native-chart"
        return PIXAL3D_FACE_ATLAS_TILE_PADDING, "backend_default:face-atlas"
    padding = float(value)
    if not np.isfinite(padding) or padding < 0.0 or padding >= 0.45:
        raise ValueError("tile_padding must be finite and in [0, 0.45)")
    return padding, "explicit"


def _simplifier_backend_for_quality_preset(quality_preset: str) -> str:
    preset = _normalize_quality_preset(quality_preset)
    if preset == "reference-target":
        return "topology-aware"
    return "spatial-cluster"


def _export_quality_summary(
    simplify_stats: dict[str, Any],
    export_metrics: dict[str, Any],
    texture_stats: dict[str, Any] | None = None,
    reference: dict[str, Any] | None = None,
    *,
    quality_preset: str = "preview",
) -> dict[str, Any]:
    blockers = tuple(str(item) for item in export_metrics.get("export_blocking_reasons", ()))
    simplifier_quality = str(simplify_stats.get("quality_tier", "unknown"))
    simplifier_backend = str(simplify_stats.get("backend", "unknown"))
    preset = _normalize_quality_preset(quality_preset)
    thresholds = _production_thresholds(
        simplify_stats,
        export_metrics,
        texture_stats or {},
        reference,
        quality_preset=preset,
    )
    warnings: list[str] = []
    if preset == "preview":
        warnings.append("preview_quality_preset")
    if simplifier_quality != "production":
        warnings.append("preview_simplifier_quality_tier")
    if blockers:
        warnings.append("export_blocking_reasons_present")
    if not thresholds["all_passed"]:
        warnings.append("production_thresholds_failed")
    artifact_ready = len(blockers) == 0
    return {
        "artifact_ready": artifact_ready,
        "production_quality_ready": artifact_ready and bool(thresholds["all_passed"]),
        "quality_preset": preset,
        "simplifier_backend": simplifier_backend,
        "simplifier_quality_tier": simplifier_quality,
        "native_geometry_candidate": _native_geometry_candidate_status(simplify_stats, thresholds, preset),
        "export_blocking_reasons": blockers,
        "production_thresholds": thresholds,
        "warnings": tuple(warnings),
    }


def _native_geometry_candidate_status(
    simplify_stats: dict[str, Any],
    thresholds: dict[str, Any],
    quality_preset: str,
) -> dict[str, Any]:
    checks = thresholds.get("checks", {})
    backend_check = checks.get("backend_tier", {})
    face_check = checks.get("face_count_ratio", {})
    topology_check = checks.get("topology_exportability", {})
    if quality_preset != "reference-target":
        return {
            "status": "not_requested",
            "reason": "quality_preset_is_preview",
            "current_backend": simplify_stats.get("backend"),
            "current_quality_tier": simplify_stats.get("quality_tier"),
            "requested_backend": simplify_stats.get("requested_backend"),
            "backend_selection_status": simplify_stats.get("backend_selection_status"),
        }
    if bool(backend_check.get("passed")):
        return {
            "status": "candidate",
            "reason": "native_geometry_candidate_available",
            "current_backend": simplify_stats.get("backend"),
            "current_quality_tier": simplify_stats.get("quality_tier"),
            "requested_backend": simplify_stats.get("requested_backend"),
            "backend_selection_status": simplify_stats.get("backend_selection_status"),
            "face_count_ratio": face_check.get("actual"),
            "topology_exportability_passed": bool(topology_check.get("passed")),
        }
    return {
        "status": "blocked",
        "reason": "native_geometry_candidate_blocked",
        "detail": "reference-target export still uses a preview-tier native simplifier",
        "current_backend": simplify_stats.get("backend"),
        "current_quality_tier": simplify_stats.get("quality_tier"),
        "requested_backend": simplify_stats.get("requested_backend"),
        "backend_selection_status": simplify_stats.get("backend_selection_status"),
        "face_count_ratio": face_check.get("actual"),
        "topology_exportability_passed": bool(topology_check.get("passed")),
    }


def _native_chart_uv_candidate_status(
    uv_stats: dict[str, Any],
    texture_stats: dict[str, Any],
    uv_backend: str,
) -> dict[str, Any]:
    uv_stats_backend = str(uv_stats.get("backend", "unknown"))
    texture_backend = str(texture_stats.get("backend", "unknown"))
    if uv_backend != "native-chart":
        return {
            "status": "not_requested",
            "artifact_ready": None,
            "quality_ready": None,
            "requested_uv_backend": uv_backend,
            "uv_backend": uv_stats_backend,
            "texture_bake_backend": texture_backend,
            "checks": {},
            "quality_blockers": (),
            "xatlas_chart_parity": False,
        }
    chart_count = _maybe_int(uv_stats.get("chart_count"))
    sampled_texels = _maybe_int(texture_stats.get("sampled_texel_count"))
    uv_bin_references = _maybe_int(texture_stats.get("uv_bin_face_reference_count"))
    uv_bin_guard_passed = bool(texture_stats.get("uv_bin_guard_passed"))
    final_coverage = _maybe_float(texture_stats.get("coverage_ratio", texture_stats.get("final_visible_coverage_ratio")))
    uv_surface_visible = _maybe_float(texture_stats.get("uv_surface_final_visible_coverage_ratio"))
    texture_pixel_count = _maybe_int(texture_stats.get("texture_pixel_count"))
    uv_surface_texel_count = _maybe_int(texture_stats.get("uv_surface_texel_count"))
    uv_surface_occupancy = None
    if texture_pixel_count not in (None, 0) and uv_surface_texel_count is not None:
        uv_surface_occupancy = float(uv_surface_texel_count) / float(texture_pixel_count)

    checks = {
        "chart_backend": {
            "passed": uv_stats_backend == "native-chart-atlas",
            "actual": uv_stats_backend,
            "required": "native-chart-atlas",
        },
        "texture_backend": {
            "passed": texture_backend == "metal-uv-binned-nearest",
            "actual": texture_backend,
            "required": "metal-uv-binned-nearest",
        },
        "chart_count": {
            "passed": chart_count is not None and chart_count > 0,
            "actual": chart_count,
            "required": ">0",
        },
        "sampled_texels": {
            "passed": sampled_texels is not None and sampled_texels > 0,
            "actual": sampled_texels,
            "required": ">0",
        },
        "uv_bin_guard": {
            "passed": uv_bin_guard_passed,
            "actual": uv_bin_guard_passed,
            "required": True,
        },
        "uv_bin_references": {
            "passed": uv_bin_references is not None and uv_bin_references > 0,
            "actual": uv_bin_references,
            "required": ">0",
        },
        "global_coverage_floor": {
            "passed": final_coverage is not None and final_coverage >= PIXAL3D_CHART_UV_GLOBAL_COVERAGE_MIN,
            "actual": final_coverage,
            "required_min": PIXAL3D_CHART_UV_GLOBAL_COVERAGE_MIN,
        },
        "uv_surface_occupancy_floor": {
            "passed": uv_surface_occupancy is not None
            and uv_surface_occupancy >= PIXAL3D_CHART_UV_SURFACE_OCCUPANCY_MIN,
            "actual": uv_surface_occupancy,
            "required_min": PIXAL3D_CHART_UV_SURFACE_OCCUPANCY_MIN,
        },
        "uv_surface_visible_floor": {
            "passed": uv_surface_visible is not None and uv_surface_visible >= PIXAL3D_CHART_UV_SURFACE_VISIBLE_MIN,
            "actual": uv_surface_visible,
            "required_min": PIXAL3D_CHART_UV_SURFACE_VISIBLE_MIN,
        },
    }
    artifact_check_names = (
        "chart_backend",
        "texture_backend",
        "chart_count",
        "sampled_texels",
        "uv_bin_guard",
        "uv_bin_references",
    )
    quality_check_names = (
        "global_coverage_floor",
        "uv_surface_occupancy_floor",
        "uv_surface_visible_floor",
    )
    artifact_ready = all(bool(checks[name]["passed"]) for name in artifact_check_names)
    quality_ready = artifact_ready and all(bool(checks[name]["passed"]) for name in quality_check_names)
    quality_blockers = tuple(name for name in quality_check_names if not bool(checks[name]["passed"]))
    artifact_blockers = tuple(name for name in artifact_check_names if not bool(checks[name]["passed"]))
    if not artifact_ready:
        status = "artifact_blocked"
    elif quality_ready:
        status = "quality_ready"
    else:
        status = "quality_blocked"
    return {
        "status": status,
        "artifact_ready": artifact_ready,
        "quality_ready": quality_ready,
        "requested_uv_backend": uv_backend,
        "uv_backend": uv_stats_backend,
        "texture_bake_backend": texture_backend,
        "chart_count": _maybe_int(uv_stats.get("chart_count")),
        "output_vertices": _maybe_int(uv_stats.get("output_vertices")),
        "output_faces": _maybe_int(uv_stats.get("output_faces")),
        "duplicated_vertex_ratio": _maybe_float(uv_stats.get("duplicated_vertex_ratio")),
        "global_coverage_ratio": final_coverage,
        "uv_surface_occupancy_ratio": uv_surface_occupancy,
        "uv_surface_final_visible_coverage_ratio": uv_surface_visible,
        "uv_surface_texel_count": uv_surface_texel_count,
        "texture_pixel_count": texture_pixel_count,
        "uv_bin_face_reference_count": _maybe_int(texture_stats.get("uv_bin_face_reference_count")),
        "uv_bin_max_candidate_faces": _maybe_int(texture_stats.get("uv_bin_max_candidate_faces")),
        "checks": checks,
        "artifact_blockers": artifact_blockers,
        "quality_blockers": quality_blockers,
        "xatlas_chart_parity": False,
    }


def _production_thresholds(
    simplify_stats: dict[str, Any],
    export_metrics: dict[str, Any],
    texture_stats: dict[str, Any],
    reference: dict[str, Any] | None,
    *,
    quality_preset: str,
) -> dict[str, Any]:
    blockers = tuple(str(item) for item in export_metrics.get("export_blocking_reasons", ()))
    simplifier_quality = str(simplify_stats.get("quality_tier", "unknown"))
    final_faces = _maybe_int(simplify_stats.get("final_faces"))
    reference_faces = _maybe_int(reference.get("final_faces")) if reference is not None else None
    final_coverage = _maybe_float(texture_stats.get("coverage_ratio", texture_stats.get("final_visible_coverage_ratio")))
    reference_coverage = _maybe_float(reference.get("coverage_ratio")) if reference is not None else None
    raw_coverage = _maybe_float(texture_stats.get("raw_coverage_ratio"))
    reference_raw_coverage = _maybe_float(reference.get("raw_coverage_ratio")) if reference is not None else None

    face_ratio = None
    face_count_passed = False
    if final_faces is not None and reference_faces not in (None, 0):
        face_ratio = float(final_faces) / float(reference_faces)
        face_count_passed = PIXAL3D_REFERENCE_FACE_RATIO_MIN <= face_ratio <= PIXAL3D_REFERENCE_FACE_RATIO_MAX

    final_coverage_ratio = None
    coverage_passed = False
    if final_coverage is not None and reference_coverage not in (None, 0.0):
        final_coverage_ratio = final_coverage / reference_coverage
        coverage_passed = final_coverage_ratio >= PIXAL3D_REFERENCE_FINAL_COVERAGE_THRESHOLD

    raw_coverage_ratio = None
    if raw_coverage is not None and reference_raw_coverage not in (None, 0.0):
        raw_coverage_ratio = raw_coverage / reference_raw_coverage

    checks = {
        "reference_available": {
            "passed": reference is not None,
            "actual": bool(reference is not None),
            "required": True,
        },
        "quality_preset": {
            "passed": quality_preset == "reference-target",
            "actual": quality_preset,
            "required": "reference-target",
        },
        "backend_tier": {
            "passed": simplifier_quality == "production",
            "actual": simplifier_quality,
            "required": "production",
        },
        "topology_exportability": {
            "passed": len(blockers) == 0,
            "actual": blockers,
            "required": [],
        },
        "face_count_ratio": {
            "passed": face_count_passed,
            "actual": face_ratio,
            "required_min": PIXAL3D_REFERENCE_FACE_RATIO_MIN,
            "required_max": PIXAL3D_REFERENCE_FACE_RATIO_MAX,
            "spatialkit_final_faces": final_faces,
            "reference_final_faces": reference_faces,
        },
        "final_coverage_ratio": {
            "passed": coverage_passed,
            "actual": final_coverage_ratio,
            "required_min": PIXAL3D_REFERENCE_FINAL_COVERAGE_THRESHOLD,
            "spatialkit_final_coverage_ratio": final_coverage,
            "reference_final_coverage_ratio": reference_coverage,
        },
        "raw_coverage_ratio": {
            "passed": raw_coverage_ratio is not None,
            "actual": raw_coverage_ratio,
            "required": "reported",
            "spatialkit_raw_coverage_ratio": raw_coverage,
            "reference_raw_coverage_ratio": reference_raw_coverage,
        },
    }
    all_passed = all(bool(check["passed"]) for check in checks.values())
    return {
        "all_passed": all_passed,
        "checks": checks,
    }


def _load_pixal3d_reference_trace(decoded_dir: Path) -> dict[str, Any] | None:
    candidates = [
        decoded_dir.parent / "pixal3d-1024-cascade-glb-reference" / "trace.json",
        Path.cwd() / "inputs" / "mlx-spatialkit" / "pixal3d-1024-cascade-glb-reference" / "trace.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            trace = json.load(handle)
        metadata = trace.get("metadata", {})
        mesh_export = metadata.get("mesh_export", {})
        postprocess = mesh_export.get("postprocess_stats", {})
        artifact_metadata = metadata.get("textured_glb_artifact", {}).get("metadata", {})
        return {
            "trace_path": str(path),
            "model_glb_path": str(path.with_name("model.glb")) if path.with_name("model.glb").exists() else None,
            "final_faces": _maybe_int(postprocess.get("final_faces")),
            "final_vertices": _maybe_int(postprocess.get("final_vertices")),
            "raw_coverage_ratio": _maybe_float(mesh_export.get("raw_coverage_ratio", artifact_metadata.get("raw_coverage_ratio"))),
            "coverage_ratio": _maybe_float(mesh_export.get("coverage_ratio", artifact_metadata.get("coverage_ratio"))),
            "unwrap_backend": mesh_export.get("unwrap_backend", artifact_metadata.get("unwrap_backend")),
            "bake_backend": mesh_export.get("bake_backend", artifact_metadata.get("bake_backend")),
            "texture_size": _maybe_int(mesh_export.get("texture_size", artifact_metadata.get("texture_size"))),
            "xatlas_face_guard": _maybe_int(mesh_export.get("xatlas_face_guard", artifact_metadata.get("xatlas_face_guard"))),
            "unwrap_utilization": _maybe_float(mesh_export.get("unwrap_utilization", artifact_metadata.get("unwrap_utilization"))),
        }
    return None


def _upstream_export_settings_summary(
    target_faces: int,
    texture_size: int,
    simplify_stats: dict[str, Any],
    texture_stats: dict[str, Any],
    quality: dict[str, Any],
) -> dict[str, Any]:
    final_faces = _maybe_int(simplify_stats.get("final_faces"))
    face_retention = None
    if final_faces is not None and PIXAL3D_UPSTREAM_EXPORT_TARGET_FACES > 0:
        face_retention = float(final_faces) / float(PIXAL3D_UPSTREAM_EXPORT_TARGET_FACES)
    final_coverage = _maybe_float(texture_stats.get("coverage_ratio", texture_stats.get("final_visible_coverage_ratio")))
    backend_tier = str(simplify_stats.get("quality_tier", "unknown"))
    target_reached = bool(simplify_stats.get("target_reached"))
    artifact_ready = bool(quality.get("artifact_ready"))

    checks = {
        "target_faces": {
            "passed": int(target_faces) == PIXAL3D_UPSTREAM_EXPORT_TARGET_FACES,
            "actual": int(target_faces),
            "required": PIXAL3D_UPSTREAM_EXPORT_TARGET_FACES,
        },
        "texture_size": {
            "passed": int(texture_size) == PIXAL3D_UPSTREAM_EXPORT_TEXTURE_SIZE,
            "actual": int(texture_size),
            "required": PIXAL3D_UPSTREAM_EXPORT_TEXTURE_SIZE,
        },
        "backend_tier": {
            "passed": backend_tier == "production",
            "actual": backend_tier,
            "required": "production",
        },
        "target_reached": {
            "passed": target_reached,
            "actual": target_reached,
            "required": True,
        },
        "face_retention_ratio": {
            "passed": face_retention is not None
            and face_retention >= PIXAL3D_UPSTREAM_EXPORT_FACE_RETENTION_MIN
            and face_retention <= 1.0,
            "actual": face_retention,
            "required_min": PIXAL3D_UPSTREAM_EXPORT_FACE_RETENTION_MIN,
            "required_max": 1.0,
            "final_faces": final_faces,
        },
        "artifact_ready": {
            "passed": artifact_ready,
            "actual": artifact_ready,
            "required": True,
        },
        "final_coverage_ratio": {
            "passed": final_coverage is not None and final_coverage >= PIXAL3D_REFERENCE_FINAL_COVERAGE_THRESHOLD,
            "actual": final_coverage,
            "required_min": PIXAL3D_REFERENCE_FINAL_COVERAGE_THRESHOLD,
        },
    }
    return {
        "all_passed": all(bool(check["passed"]) for check in checks.values()),
        "reference": {
            "source": "vendored_pixal3d_inference_defaults",
            "decimation_target": PIXAL3D_UPSTREAM_EXPORT_TARGET_FACES,
            "texture_size": PIXAL3D_UPSTREAM_EXPORT_TEXTURE_SIZE,
            "remesh": True,
            "remesh_band": 1,
            "remesh_project": 0,
            "xatlas_chart_parity": False,
        },
        "checks": checks,
    }


def _glb_viewer_compatibility_summary(glb_summary: dict[str, Any]) -> dict[str, Any]:
    primitives = list(glb_summary.get("primitives", ()))
    large_mesh_threshold = 65_536
    all_have_normals = bool(primitives) and all(
        bool(primitive.get("has_normals"))
        and int(primitive.get("normal_count", 0)) == int(primitive.get("vertex_count", -1))
        for primitive in primitives
    )
    uint16_only = bool(primitives) and all(
        _maybe_int(primitive.get("indices_component_type")) == 5123 for primitive in primitives
    )
    local_indices_bounded = bool(primitives) and all(
        _primitive_indices_within_uint16(primitive) for primitive in primitives
    )
    triangle_indices = bool(primitives) and all(int(primitive.get("index_count", 0)) % 3 == 0 for primitive in primitives)
    total_vertices = _maybe_int(glb_summary.get("total_vertices")) or 0
    primitive_count = _maybe_int(glb_summary.get("primitive_count")) or 0
    chunking_required = total_vertices > large_mesh_threshold
    checks = {
        "glb_parseable": {
            "passed": bool(primitives),
            "actual": bool(primitives),
            "required": True,
        },
        "textured_material": {
            "passed": glb_summary.get("material_count", 0) >= 1
            and glb_summary.get("texture_count", 0) >= 2
            and glb_summary.get("image_count", 0) >= 2,
            "materials": glb_summary.get("material_count", 0),
            "textures": glb_summary.get("texture_count", 0),
            "images": glb_summary.get("image_count", 0),
            "required": "at_least_one_material_two_textures_two_images",
        },
        "normals": {
            "passed": all_have_normals,
            "actual": [
                {
                    "primitive_index": primitive.get("primitive_index"),
                    "has_normals": primitive.get("has_normals"),
                    "vertex_count": primitive.get("vertex_count"),
                    "normal_count": primitive.get("normal_count"),
                }
                for primitive in primitives
            ],
            "required": "NORMAL attribute with count matching POSITION for every primitive",
        },
        "uint16_indices": {
            "passed": uint16_only,
            "actual": [primitive.get("indices_component_type") for primitive in primitives],
            "required": 5123,
        },
        "local_index_bounds": {
            "passed": local_indices_bounded,
            "actual": [
                {
                    "primitive_index": primitive.get("primitive_index"),
                    "indices_min": primitive.get("indices_min"),
                    "indices_max": primitive.get("indices_max"),
                }
                for primitive in primitives
            ],
            "required_min": 0,
            "required_max": 65_535,
        },
        "triangle_indices": {
            "passed": triangle_indices,
            "actual": [primitive.get("index_count") for primitive in primitives],
            "required": "index_count divisible by 3",
        },
        "chunking_for_large_mesh": {
            "passed": not chunking_required or primitive_count > 1,
            "actual": primitive_count,
            "required": ">1 primitive when total_vertices > 65536",
            "total_vertices": total_vertices,
            "large_mesh_threshold": large_mesh_threshold,
        },
    }
    return {
        "all_passed": all(bool(check["passed"]) for check in checks.values()),
        "checks": checks,
    }


def _primitive_indices_within_uint16(primitive: dict[str, Any]) -> bool:
    min_values = primitive.get("indices_min")
    max_values = primitive.get("indices_max")
    if not isinstance(min_values, list) or not min_values:
        return False
    if not isinstance(max_values, list) or not max_values:
        return False
    min_index = _maybe_int(min_values[0])
    max_index = _maybe_int(max_values[0])
    return min_index is not None and max_index is not None and min_index >= 0 and max_index <= 65_535


def _reference_glb_path(reference: dict[str, Any]) -> Path | None:
    path = reference.get("model_glb_path")
    if path is None:
        return None
    reference_glb = Path(path)
    return reference_glb if reference_glb.exists() else None


def _visual_comparison_summary(
    report: dict[str, Any],
    upstream_export_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    deferred_boundaries = list(report["deferred_parity_boundaries"])
    if upstream_export_settings is not None and bool(upstream_export_settings.get("all_passed")):
        deferred_boundaries = [
            item for item in deferred_boundaries if item != "not_1m_face_export_setting_parity"
        ]
    return {
        "summary": report["summary"],
        "checks": report["checks"],
        "artifacts": report.get("artifacts", {}),
        "deferred_parity_boundaries": deferred_boundaries,
    }


def _reference_comparison(diagnostics: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    simplify_stats = diagnostics.get("stages", {}).get("simplify_mesh", {}).get("stats", {})
    texture_stats = diagnostics.get("stages", {}).get("texture_bake", {}).get("stats", {})
    final_faces = _maybe_int(simplify_stats.get("final_faces"))
    reference_faces = _maybe_int(reference.get("final_faces"))
    raw_coverage = _maybe_float(texture_stats.get("raw_coverage_ratio"))
    final_coverage = _maybe_float(texture_stats.get("coverage_ratio", texture_stats.get("final_visible_coverage_ratio")))
    reference_raw = _maybe_float(reference.get("raw_coverage_ratio"))
    reference_final = _maybe_float(reference.get("coverage_ratio"))
    comparison: dict[str, Any] = {
        "spatialkit_simplifier_backend": simplify_stats.get("backend"),
        "spatialkit_quality_tier": simplify_stats.get("quality_tier"),
        "reference_unwrap_backend": reference.get("unwrap_backend"),
        "reference_bake_backend": reference.get("bake_backend"),
        "spatialkit_final_faces": final_faces,
        "reference_final_faces": reference_faces,
        "spatialkit_raw_coverage_ratio": raw_coverage,
        "reference_raw_coverage_ratio": reference_raw,
        "spatialkit_final_coverage_ratio": final_coverage,
        "reference_final_coverage_ratio": reference_final,
    }
    if final_faces is not None and reference_faces not in (None, 0):
        comparison["final_face_count_ratio"] = float(final_faces) / float(reference_faces)
    if raw_coverage is not None and reference_raw not in (None, 0.0):
        comparison["raw_coverage_ratio_vs_reference"] = raw_coverage / reference_raw
    if final_coverage is not None and reference_final not in (None, 0.0):
        comparison["final_coverage_ratio_vs_reference"] = final_coverage / reference_final
    return comparison


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

__all__ = [
    "NativeGlbArtifact",
    "NativeUvMesh",
    "Pixal3DGlbExportResult",
    "Pixal3DDecodedInputs",
    "backend_info",
    "export_pixal3d_glb",
    "load_pixal3d_decoded_npz",
    "make_face_atlas_uvs",
    "make_native_chart_uvs",
    "textured_glb_payload",
    "validate_pixal3d_decoded",
    "write_textured_glb",
]
