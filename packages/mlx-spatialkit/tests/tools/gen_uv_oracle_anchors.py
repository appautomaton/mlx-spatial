#!/usr/bin/env python
"""Generate version-pinned UV-parity oracle anchors (tests/data/uv_oracle_anchors.json).

For both cached Pixal3D fixtures this script:
  1. Phase A (PROJECT venv, this process): rebuilds the QEM-decimated 50k mesh
     exactly as the heavy QEM proof tests do (extract grid=256 -> clean -> narrow-band
     remesh res=256 repair_nonmanifold=True -> simplify_mesh backend="qem"
     target=50_000), runs the native stage-A clusterer
     mlx_spatialkit._native.compute_uv_charts with PRODUCTION knobs, and dumps
     per-cluster submeshes plus the whole mesh to a /tmp scratch NPZ.
  2. Phase B (ORACLE venv subprocess, /tmp/uvoracle-venv/bin/python, pip
     xatlas 0.0.11): builds ONE xatlas.Atlas, add_mesh's each cluster submesh,
     generates charts+packing at the reference defaults (CuMesh uv_unwrap wrapper
     defaults == xatlas defaults), plus one whole-mesh sanity run, and dumps
     uvs/faces/vmapping/per-face chart ids back to a /tmp NPZ.
  3. Phase A again: computes UV quality metrics on the oracle output with
     mlx_spatialkit._native.uv_quality_metrics and writes the anchors JSON.

The PROJECT venv must NOT have xatlas installed; xatlas is only imported inside
the phase-B subprocess running under the oracle venv.

Usage (from packages/mlx-spatialkit, PROJECT venv):
    .venv/bin/python tests/tools/gen_uv_oracle_anchors.py
    .venv/bin/python tests/tools/gen_uv_oracle_anchors.py --reuse-cache
    .venv/bin/python tests/tools/gen_uv_oracle_anchors.py --reuse-cache --fixtures main

--reuse-cache reuses the cached QEM meshes under /tmp/uv-oracle-cache/ (the
QEM pipeline costs ~1-2 min per fixture); --fixtures lets you iterate on one
fixture (results are merged into an existing anchors JSON). The run log is
written to /tmp/uv-oracle-anchors-run.log; all heavy intermediates stay in /tmp.

Oracle venv bootstrap (one-time; the PROJECT venv must NOT have xatlas):
    python3 -m venv /tmp/uvoracle-venv && \\
        /tmp/uvoracle-venv/bin/pip install xatlas==0.0.11 numpy

Internal phase-B invocation (dispatched automatically, do not run by hand):
    /tmp/uvoracle-venv/bin/python tests/tools/gen_uv_oracle_anchors.py \
        --phase-b SCRATCH_IN.npz SCRATCH_OUT.npz
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ORACLE_PYTHON_DEFAULT = "/tmp/uvoracle-venv/bin/python"
ORACLE_XATLAS_VERSION = "0.0.11"
ORACLE_BOOTSTRAP_RECIPE = (
    "python3 -m venv /tmp/uvoracle-venv && "
    f"/tmp/uvoracle-venv/bin/pip install xatlas=={ORACLE_XATLAS_VERSION} numpy"
)
CACHE_DIR = Path("/tmp/uv-oracle-cache")
LOG_PATH = Path("/tmp/uv-oracle-anchors-run.log")

PKG_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
ANCHORS_PATH = PKG_ROOT / "tests" / "data" / "uv_oracle_anchors.json"

FIXTURES = {
    "main": REPO_ROOT / "inputs" / "mlx-spatialkit" / "pixal3d-1024-cascade-decoded-pbr",
    "violin_bow": REPO_ROOT / "inputs" / "mlx-spatialkit" / "violin-bow" / "pixal3d-1024-cascade-decoded-pbr",
}

# QEM mesh production, mirroring the heavy two-fixture proof tests in
# tests/test_real_pixal3d_export.py (_load_remeshed_mesh_from_fixture +
# simplify_mesh(backend="qem") at the preview-default 50k target).
QEM_PIPELINE = {
    "grid_size": 256,
    "remesh_resolution": 256,
    "remesh_repair_nonmanifold": True,
    "min_component_faces": 32,
    "simplify_backend": "qem",
    "target_faces": 50_000,
}

# PRODUCTION stage-A knobs (NOT the binding defaults, which are the slow
# CuMesh defaults: refine_iterations=100, global_iterations=3).
STAGE_A_KNOBS = {
    "threshold_cone_half_angle_rad": math.radians(90.0),
    "refine_iterations": 0,
    "global_iterations": 1,
    "smooth_strength": 1.0,
    "area_penalty_weight": 0.1,
    "perimeter_area_ratio_weight": 0.0001,
}

# Reference chart/pack options: CuMesh's uv_unwrap wrapper defaults
# (cumesh.py:408) equal the xatlas defaults. Keys are the CuMesh option
# names; values give the pip xatlas 0.0.11 attribute they map to and the
# reference value. Every CuMesh option has a direct pip equivalent
# (verified against dir(xatlas.ChartOptions()) / dir(xatlas.PackOptions())).
OPTION_MAPPING = {
    "chart_options": {
        "max_cost": {"pip_attr": "max_cost", "value": 2.0},
        "normal_deviation_weight": {"pip_attr": "normal_deviation_weight", "value": 2.0},
        "roundness": {"pip_attr": "roundness_weight", "value": 0.01},
        "straightness": {"pip_attr": "straightness_weight", "value": 6.0},
        "normal_seam": {"pip_attr": "normal_seam_weight", "value": 4.0},
        "texture_seam": {"pip_attr": "texture_seam_weight", "value": 0.5},
        "max_iterations": {"pip_attr": "max_iterations", "value": 1},
    },
    "pack_options": {
        "padding": {"pip_attr": "padding", "value": 0},
        "bilinear": {"pip_attr": "bilinear", "value": True},
        "rotate_charts": {"pip_attr": "rotate_charts", "value": True},
        "brute_force": {"pip_attr": "bruteForce", "value": False},
    },
    "notes": [
        "CuMesh uv_unwrap wrapper defaults equal xatlas defaults; options are set explicitly anyway.",
        "whole_mesh run uses xatlas.Atlas.generate with the reference options instead of "
        "xatlas.parametrize so chart_count/utilization are obtainable; parametrize is a "
        "default-options Atlas wrapper, and the run also executes xatlas.parametrize and "
        "records parametrize_matches_atlas (uvs allclose) as proof of equivalence.",
        "pip xatlas get_mesh/parametrize return uvs already normalized to [0,1] atlas space.",
        "uv_flipped_count is genuinely large for real xatlas output: xatlas mirrors roughly "
        "half its charts (verified per-chart: every chart is internally orientation-consistent, "
        "~50/50 all-positive vs all-negative UV signed area, even on a winding-consistent cube).",
        "xatlas assigns zero-area degenerate faces to no chart and gives them zero uvs; these "
        "are recorded per block as degenerate_unassigned_faces and excluded from the per-chart "
        "stretch summaries.",
        "uv_quality_metrics reports an exact 0.0 stretch sentinel for any chart with no "
        "measurable face (all faces degenerate or flipped; fully mirrored charts are ~half of "
        "real xatlas output), meaning 'no measurable faces', not 'zero distortion'. The "
        "per-chart stretch summaries exclude those sentinels as well; measured_chart_count vs "
        "total_chart_count records how many charts were actually measured.",
    ],
}

# pip-attribute -> value dicts actually applied in phase B.
PIP_CHART_OPTS = {m["pip_attr"]: m["value"] for m in OPTION_MAPPING["chart_options"].values()}
PIP_PACK_OPTS = {m["pip_attr"]: m["value"] for m in OPTION_MAPPING["pack_options"].values()}


def log(msg: str) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Phase B: runs ONLY under the oracle venv (pip xatlas 0.0.11 + numpy).
# Must not import mlx_spatialkit; must be the only place importing xatlas.
# ---------------------------------------------------------------------------

def run_phase_b(in_path: str, out_path: str) -> None:
    import xatlas  # noqa: PLC0415  (oracle venv only)

    if xatlas.__version__ != ORACLE_XATLAS_VERSION:
        raise SystemExit(
            f"oracle venv has xatlas {xatlas.__version__}, but the anchors are pinned to "
            f"xatlas {ORACLE_XATLAS_VERSION}. Rebuild the oracle venv: {ORACLE_BOOTSTRAP_RECIPE}"
        )

    data = np.load(in_path)
    out: dict[str, np.ndarray] = {
        "xatlas_version": np.array(xatlas.__version__),
        "numpy_version_oracle": np.array(np.__version__),
    }

    def make_options() -> tuple["xatlas.ChartOptions", "xatlas.PackOptions"]:
        chart_options = xatlas.ChartOptions()
        for attr, value in PIP_CHART_OPTS.items():
            setattr(chart_options, attr, value)
        pack_options = xatlas.PackOptions()
        for attr, value in PIP_PACK_OPTS.items():
            setattr(pack_options, attr, value)
        return chart_options, pack_options

    def face_chart_ids(atlas: "xatlas.Atlas", mesh_index: int, face_count: int) -> np.ndarray:
        ids = np.full(face_count, -1, dtype=np.int64)
        for j in range(atlas.get_mesh_chart_count(mesh_index)):
            chart = atlas.get_mesh_chart(mesh_index, j)
            ids[np.asarray(chart.faces, dtype=np.int64)] = j
        return ids

    # --- Composition: ONE atlas, add_mesh per stage-A cluster submesh -----
    cluster_count = int(data["cluster_count"])
    atlas = xatlas.Atlas()
    for k in range(cluster_count):
        atlas.add_mesh(
            np.ascontiguousarray(data[f"cluster_{k}_vertices"], dtype=np.float32),
            np.ascontiguousarray(data[f"cluster_{k}_faces"], dtype=np.uint32),
        )
    chart_options, pack_options = make_options()
    atlas.generate(chart_options, pack_options, verbose=False)
    out["composition_chart_count"] = np.int64(atlas.chart_count)
    out["composition_atlas_count"] = np.int64(atlas.atlas_count)
    out["composition_utilization"] = np.float64(atlas.utilization)
    out["composition_width"] = np.int64(atlas.width)
    out["composition_height"] = np.int64(atlas.height)
    for k in range(cluster_count):
        vmapping, indices, uvs = atlas.get_mesh(k)
        out[f"composition_{k}_vmapping"] = vmapping
        out[f"composition_{k}_faces"] = indices
        out[f"composition_{k}_uvs"] = uvs
        out[f"composition_{k}_chart_count"] = np.int64(atlas.get_mesh_chart_count(k))
        out[f"composition_{k}_chart_ids"] = face_chart_ids(atlas, k, indices.shape[0])

    # --- Whole-mesh sanity run --------------------------------------------
    whole_vertices = np.ascontiguousarray(data["whole_vertices"], dtype=np.float32)
    whole_faces = np.ascontiguousarray(data["whole_faces"], dtype=np.uint32)
    whole_atlas = xatlas.Atlas()
    whole_atlas.add_mesh(whole_vertices, whole_faces)
    chart_options, pack_options = make_options()
    whole_atlas.generate(chart_options, pack_options, verbose=False)
    vmapping, indices, uvs = whole_atlas.get_mesh(0)
    out["whole_chart_count"] = np.int64(whole_atlas.chart_count)
    out["whole_utilization"] = np.float64(whole_atlas.utilization)
    out["whole_width"] = np.int64(whole_atlas.width)
    out["whole_height"] = np.int64(whole_atlas.height)
    out["whole_vmapping"] = vmapping
    out["whole_faces_out"] = indices
    out["whole_uvs"] = uvs
    out["whole_chart_ids"] = face_chart_ids(whole_atlas, 0, indices.shape[0])

    # parametrize equivalence proof: parametrize is a default-options Atlas
    # wrapper, and the reference options equal the defaults.
    p_vmapping, p_indices, p_uvs = xatlas.parametrize(whole_vertices, whole_faces)
    matches = (
        p_vmapping.shape == vmapping.shape
        and p_indices.shape == indices.shape
        and bool(np.array_equal(p_vmapping, vmapping))
        and bool(np.array_equal(p_indices, indices))
        and bool(np.allclose(p_uvs, uvs, atol=1e-6))
    )
    out["whole_parametrize_matches_atlas"] = np.bool_(matches)

    np.savez_compressed(out_path, **out)


# ---------------------------------------------------------------------------
# Phase A: runs under the PROJECT venv (mlx_spatialkit, NO xatlas).
# ---------------------------------------------------------------------------

def _spatialkit_version() -> str:
    """Installed mlx-spatialkit version, for tagging the QEM mesh cache."""
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return version("mlx-spatialkit")
    except Exception:  # noqa: BLE001  (cache tag is best-effort)
        return "unknown"


def build_qem_mesh(name: str, fixture: Path, reuse_cache: bool) -> tuple[np.ndarray, np.ndarray]:
    """Produce the QEM-decimated mesh exactly as the heavy QEM proof tests do."""
    pkg_version = _spatialkit_version()
    cache_base = (
        f"qem_{name}_grid{QEM_PIPELINE['grid_size']}"
        f"_res{QEM_PIPELINE['remesh_resolution']}_t{QEM_PIPELINE['target_faces']}"
    )
    # The package version is part of the cache key: a QEM backend change would
    # otherwise invalidate cached geometry silently.
    cache_path = CACHE_DIR / f"{cache_base}_pkg{pkg_version}.npz"
    legacy_cache_path = CACHE_DIR / f"{cache_base}.npz"
    if reuse_cache:
        reuse_path = None
        if cache_path.exists():
            reuse_path = cache_path
        elif legacy_cache_path.exists():
            reuse_path = legacy_cache_path
            log(f"[{name}] WARNING: reusing legacy QEM cache WITHOUT a package-version tag "
                f"({legacy_cache_path}); if the QEM backend changed since it was written, "
                f"this geometry is silently stale. Delete the file to force a rebuild "
                f"under mlx-spatialkit {pkg_version}.")
        if reuse_path is not None:
            with np.load(reuse_path) as payload:
                vertices = np.asarray(payload["vertices"])
                faces = np.asarray(payload["faces"])
            log(f"[{name}] reused cached QEM mesh {reuse_path} "
                f"({vertices.shape[0]} vertices, {faces.shape[0]} faces)")
            return vertices, faces

    from mlx_spatialkit.mesh import (  # noqa: PLC0415  (project venv only)
        clean_mesh,
        extract_flexi_dual_grid,
        remesh_narrow_band,
        simplify_mesh,
    )

    shape_path = fixture / "shape_decoder_fields.npz"
    with np.load(shape_path) as payload:
        coordinates = np.asarray(payload["coordinates"])
        fields = np.asarray(payload["fields"])
    log(f"[{name}] extracting (grid_size={QEM_PIPELINE['grid_size']}) ...")
    raw = extract_flexi_dual_grid(coordinates, fields, grid_size=QEM_PIPELINE["grid_size"])
    cleaned, _ = clean_mesh(
        raw.vertices, raw.faces, min_component_faces=QEM_PIPELINE["min_component_faces"]
    )
    log(f"[{name}] remeshing (resolution={QEM_PIPELINE['remesh_resolution']}, "
        f"repair_nonmanifold=True) ...")
    remeshed, _ = remesh_narrow_band(
        cleaned.vertices,
        cleaned.faces,
        resolution=QEM_PIPELINE["remesh_resolution"],
        repair_nonmanifold=QEM_PIPELINE["remesh_repair_nonmanifold"],
    )
    log(f"[{name}] QEM simplify (backend=qem, target_faces={QEM_PIPELINE['target_faces']}) "
        f"from {remeshed.faces.shape[0]} faces ...")
    simplified, stats = simplify_mesh(
        remeshed.vertices,
        remeshed.faces,
        target_faces=QEM_PIPELINE["target_faces"],
        min_component_faces=QEM_PIPELINE["min_component_faces"],
        backend=QEM_PIPELINE["simplify_backend"],
    )
    assert stats["backend"] == "qem", f"expected qem backend, got {stats['backend']!r}"
    vertices = np.ascontiguousarray(simplified.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(simplified.faces, dtype=np.int64)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, vertices=vertices, faces=faces)
    log(f"[{name}] QEM mesh cached to {cache_path} "
        f"({vertices.shape[0]} vertices, {faces.shape[0]} faces)")
    return vertices, faces


def cluster_submeshes(
    vertices: np.ndarray, faces: np.ndarray, chart_ids: np.ndarray, chart_count: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split the mesh into per-cluster compacted submeshes (stage-A output)."""
    submeshes: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(chart_count):
        cluster_faces = faces[chart_ids == k]
        used, remapped = np.unique(cluster_faces, return_inverse=True)
        submeshes.append(
            (
                np.ascontiguousarray(vertices[used], dtype=np.float32),
                np.ascontiguousarray(remapped.reshape(cluster_faces.shape), dtype=np.int64),
            )
        )
    return submeshes


def _stretch_summary(values: object) -> dict | None:
    """Summarize per-chart stretch over MEASURABLE charts only.

    uv_quality_metrics reports an exact 0.0 sentinel for charts with no
    measurable (non-degenerate, non-flipped) face -- ~half of real xatlas
    charts are fully mirrored -- meaning "no measurable faces", not "zero
    distortion". Folding those sentinels into mean/p95 would understate the
    real per-chart stretch ~2x, so they are excluded here.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    measured = arr[np.isfinite(arr) & (arr != 0.0)]
    if measured.size == 0:
        return None
    return {
        "mean": float(measured.mean()),
        "p95": float(np.percentile(measured, 95)),
        "max": float(measured.max()),
        "measured_chart_count": int(measured.size),
        "total_chart_count": int(arr.size),
    }


def _metrics_block(
    name: str,
    label: str,
    vertices_3d: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    face_chart_ids: np.ndarray,
    chart_count: int,
    atlas_utilization: float,
    atlas_width: int,
    atlas_height: int,
    source_vertices: int,
) -> dict:
    from mlx_spatialkit._native import uv_quality_metrics  # noqa: PLC0415

    # xatlas leaves zero-area (degenerate) faces out of every chart; route
    # them to a synthetic chart id == chart_count so per-chart stretch stays
    # obtainable for the real charts, then drop the synthetic entry below.
    unassigned_faces = int((face_chart_ids < 0).sum())
    dense_chart_ids = np.where(face_chart_ids >= 0, face_chart_ids, chart_count)
    metrics = uv_quality_metrics(
        np.ascontiguousarray(vertices_3d, dtype=np.float32),
        np.ascontiguousarray(faces, dtype=np.int64),
        np.ascontiguousarray(uvs, dtype=np.float32),
        chart_ids=np.ascontiguousarray(dense_chart_ids, dtype=np.int64),
    )
    chart_stretch_l2 = list(metrics.get("chart_stretch_l2", []))
    chart_stretch_linf = list(metrics.get("chart_stretch_linf", []))
    present = list(metrics.get("chart_ids_present", []))
    if unassigned_faces and present and int(present[-1]) == int(chart_count):
        chart_stretch_l2 = chart_stretch_l2[:-1]
        chart_stretch_linf = chart_stretch_linf[:-1]

    # Scale-free per-chart stretch: l2_c * sqrt(Auv_c / A3d_c) >= 1, with 1 ==
    # isometric. This is the cross-pipeline parity signal: the raw l2 above
    # carries the atlas packing scale, which differs between the oracle's
    # packed [0,1] output and a chart-local native parameterization.
    uv_tris = np.asarray(uvs, dtype=np.float64)[np.asarray(faces)]
    uv_signed = 0.5 * (
        (uv_tris[:, 1, 0] - uv_tris[:, 0, 0]) * (uv_tris[:, 2, 1] - uv_tris[:, 0, 1])
        - (uv_tris[:, 2, 0] - uv_tris[:, 0, 0]) * (uv_tris[:, 1, 1] - uv_tris[:, 0, 1])
    )
    p_tris = np.asarray(vertices_3d, dtype=np.float64)[np.asarray(faces)]
    a3d = 0.5 * np.linalg.norm(
        np.cross(p_tris[:, 1] - p_tris[:, 0], p_tris[:, 2] - p_tris[:, 0]), axis=1
    )
    measurable = (uv_signed > 1e-12) & (a3d > 0.0)
    chart_stretch_l2_normalized = []
    for chart, raw_l2 in enumerate(chart_stretch_l2):
        in_chart = measurable & (np.asarray(face_chart_ids) == chart)
        auv = float(uv_signed[in_chart].sum())
        a3 = float(a3d[in_chart].sum())
        if raw_l2 > 0.0 and auv > 0.0 and a3 > 0.0:
            chart_stretch_l2_normalized.append(raw_l2 * math.sqrt(auv / a3))
        else:
            chart_stretch_l2_normalized.append(0.0)  # sentinel, excluded below
    output_vertices = int(vertices_3d.shape[0])
    block = {
        "chart_count": int(chart_count),
        "atlas_utilization": float(atlas_utilization),
        "atlas_width": int(atlas_width),
        "atlas_height": int(atlas_height),
        "uv_overlap_count": int(metrics["uv_overlap_count"]),
        "uv_flipped_count": int(metrics["uv_flipped_count"]),
        "uv_degenerate_count": int(metrics["uv_degenerate_count"]),
        "uv_stretch_l2": float(metrics["uv_stretch_l2"]),
        "uv_stretch_linf": float(metrics["uv_stretch_linf"]),
        "uv_bbox_utilization": float(metrics["uv_bbox_utilization"]),
        "uv_total_area": float(metrics["uv_total_area"]),
        "output_vertices": output_vertices,
        "duplicated_vertex_ratio": output_vertices / source_vertices,
        "degenerate_unassigned_faces": unassigned_faces,
        "chart_stretch_l2_summary": _stretch_summary(chart_stretch_l2),
        "chart_stretch_linf_summary": _stretch_summary(chart_stretch_linf),
        "chart_stretch_l2_normalized_summary": _stretch_summary(chart_stretch_l2_normalized),
    }
    log(f"[{name}] {label}: charts={block['chart_count']} "
        f"atlas_util={block['atlas_utilization']:.4f} "
        f"bbox_util={block['uv_bbox_utilization']:.4f} "
        f"stretch_l2={block['uv_stretch_l2']:.4f} "
        f"stretch_linf={block['uv_stretch_linf']:.4f} "
        f"overlaps={block['uv_overlap_count']} flipped={block['uv_flipped_count']} "
        f"out_verts={output_vertices} dup_ratio={block['duplicated_vertex_ratio']:.4f}")
    return block


def process_fixture(name: str, fixture: Path, oracle_python: str, reuse_cache: bool) -> dict:
    from mlx_spatialkit._native import compute_uv_charts  # noqa: PLC0415

    vertices, faces = build_qem_mesh(name, fixture, reuse_cache)
    source_vertices = int(vertices.shape[0])
    source_faces = int(faces.shape[0])

    log(f"[{name}] stage-A clustering (production knobs) ...")
    stage_a = compute_uv_charts(vertices, faces, **STAGE_A_KNOBS)
    stage_a_cluster_count = int(stage_a["chart_count"])
    stage_a_chart_ids = np.asarray(stage_a["chart_ids"], dtype=np.int64)
    log(f"[{name}] stage-A clusters: {stage_a_cluster_count}")
    assert (
        (stage_a_chart_ids >= 0) & (stage_a_chart_ids < stage_a_cluster_count)
    ).all(), (
        f"[{name}] stage-A chart_ids must cover all faces in "
        f"[0, {stage_a_cluster_count}) with no -1 before building cluster submeshes"
    )

    submeshes = cluster_submeshes(vertices, faces, stage_a_chart_ids, stage_a_cluster_count)

    scratch_in = Path(f"/tmp/uv-oracle-scratch-{name}-in.npz")
    scratch_out = Path(f"/tmp/uv-oracle-scratch-{name}-out.npz")
    payload: dict[str, np.ndarray] = {
        "cluster_count": np.int64(stage_a_cluster_count),
        "whole_vertices": vertices,
        "whole_faces": faces,
    }
    for k, (cluster_vertices, cluster_faces) in enumerate(submeshes):
        payload[f"cluster_{k}_vertices"] = cluster_vertices
        payload[f"cluster_{k}_faces"] = cluster_faces
    np.savez_compressed(scratch_in, **payload)

    log(f"[{name}] phase B: pip xatlas in oracle venv ({oracle_python}) ...")
    completed = subprocess.run(
        [oracle_python, str(Path(__file__).resolve()), "--phase-b", str(scratch_in), str(scratch_out)],
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        log(f"[{name}] phase B stdout: {completed.stdout.strip()}")
    if completed.stderr:
        log(f"[{name}] phase B stderr: {completed.stderr.strip()}")
    if completed.returncode != 0:
        raise RuntimeError(f"phase B failed for fixture {name!r} (rc={completed.returncode})")

    oracle = np.load(scratch_out)

    # Composition uv mesh: concatenate per-cluster xatlas outputs into the
    # shared [0,1] atlas domain (one atlas packed them together). 3D positions
    # are remapped by xatlas vmapping; faces index the remapped arrays.
    parts_v3: list[np.ndarray] = []
    parts_f: list[np.ndarray] = []
    parts_uv: list[np.ndarray] = []
    parts_cid: list[np.ndarray] = []
    vertex_offset = 0
    chart_offset = 0
    for k, (cluster_vertices, _cluster_faces) in enumerate(submeshes):
        vmapping = oracle[f"composition_{k}_vmapping"].astype(np.int64)
        out_faces = oracle[f"composition_{k}_faces"].astype(np.int64)
        out_uvs = oracle[f"composition_{k}_uvs"].astype(np.float32)
        chart_ids = oracle[f"composition_{k}_chart_ids"].astype(np.int64)
        parts_v3.append(cluster_vertices[vmapping])
        parts_f.append(out_faces + vertex_offset)
        parts_uv.append(out_uvs)
        parts_cid.append(np.where(chart_ids >= 0, chart_ids + chart_offset, -1))
        vertex_offset += vmapping.shape[0]
        chart_offset += int(oracle[f"composition_{k}_chart_count"])

    assert chart_offset == int(oracle["composition_chart_count"]), (
        f"[{name}] per-mesh chart-offset total {chart_offset} != composition atlas "
        f"chart_count {int(oracle['composition_chart_count'])}"
    )
    assert int(oracle["composition_atlas_count"]) == 1, (
        f"[{name}] composition must pack into exactly one atlas, got "
        f"{int(oracle['composition_atlas_count'])}"
    )

    composition = _metrics_block(
        name,
        "per_cluster_composition",
        np.concatenate(parts_v3),
        np.concatenate(parts_f),
        np.concatenate(parts_uv),
        np.concatenate(parts_cid),
        chart_count=int(oracle["composition_chart_count"]),
        atlas_utilization=float(oracle["composition_utilization"]),
        atlas_width=int(oracle["composition_width"]),
        atlas_height=int(oracle["composition_height"]),
        source_vertices=source_vertices,
    )

    whole_vmapping = oracle["whole_vmapping"].astype(np.int64)
    whole_mesh = _metrics_block(
        name,
        "whole_mesh",
        vertices[whole_vmapping],
        oracle["whole_faces_out"].astype(np.int64),
        oracle["whole_uvs"].astype(np.float32),
        oracle["whole_chart_ids"].astype(np.int64),
        chart_count=int(oracle["whole_chart_count"]),
        atlas_utilization=float(oracle["whole_utilization"]),
        atlas_width=int(oracle["whole_width"]),
        atlas_height=int(oracle["whole_height"]),
        source_vertices=source_vertices,
    )
    whole_mesh["parametrize_matches_atlas"] = bool(oracle["whole_parametrize_matches_atlas"])

    return {
        "fixture_path": str(fixture.relative_to(REPO_ROOT)),
        "source_vertices": source_vertices,
        "source_faces": source_faces,
        "stage_a_cluster_count": stage_a_cluster_count,
        "per_cluster_composition": composition,
        "whole_mesh": whole_mesh,
        "_oracle_versions": {
            "xatlas_version": str(oracle["xatlas_version"]),
            "numpy_version_oracle": str(oracle["numpy_version_oracle"]),
        },
    }


def run_phase_a(args: argparse.Namespace) -> None:
    from mlx_spatialkit import metal_device_available  # noqa: PLC0415

    if not metal_device_available():
        raise SystemExit("Metal device unavailable; the QEM fixture pipeline needs it.")

    if not Path(args.oracle_python).exists():
        raise SystemExit(
            f"oracle venv python not found: {args.oracle_python}\n"
            f"Bootstrap it first: {ORACLE_BOOTSTRAP_RECIPE}"
        )

    fixture_names = [n.strip() for n in args.fixtures.split(",") if n.strip()]
    unknown = sorted(set(fixture_names) - set(FIXTURES))
    if unknown:
        raise SystemExit(f"unknown fixtures {unknown}; available: {sorted(FIXTURES)}")

    log(f"=== gen_uv_oracle_anchors run start (fixtures={fixture_names}, "
        f"reuse_cache={args.reuse_cache}) ===")

    fixtures_payload: dict[str, dict] = {}
    existing_generated_with: dict | None = None
    if ANCHORS_PATH.exists() and set(fixture_names) != set(FIXTURES):
        # Partial regeneration: merge into the existing anchors file.
        existing = json.loads(ANCHORS_PATH.read_text())
        fixtures_payload = existing.get("fixtures", {})
        existing_generated_with = existing.get("generated_with", {})
        log(f"merging into existing anchors (kept: {sorted(fixtures_payload)})")

    oracle_versions: dict[str, str] = {}
    for name in fixture_names:
        fixture = FIXTURES[name]
        if not fixture.exists():
            raise SystemExit(f"fixture {name!r} not present: {fixture}")
        record = process_fixture(name, fixture, args.oracle_python, args.reuse_cache)
        oracle_versions = record.pop("_oracle_versions")
        if existing_generated_with is not None:
            # Partial merges must not mix provenance: the kept fixtures were
            # generated under the recorded versions, so the current run's
            # versions must match them exactly.
            current_versions = {
                "xatlas_version": oracle_versions.get("xatlas_version", "unknown"),
                "numpy_version_oracle": oracle_versions.get("numpy_version_oracle", "unknown"),
                "numpy_version_project": np.__version__,
            }
            mismatched = {
                key: {"recorded": existing_generated_with.get(key), "current": value}
                for key, value in current_versions.items()
                if existing_generated_with.get(key) != value
            }
            if mismatched:
                raise SystemExit(
                    "refusing partial-fixture merge: current run versions differ from the "
                    f"existing anchors' generated_with: {mismatched}. Regenerate ALL fixtures "
                    "(omit --fixtures) so the recorded provenance matches every fixture."
                )
        fixtures_payload[name] = record

    anchors = {
        "generated_with": {
            "xatlas_version": oracle_versions.get("xatlas_version", "unknown"),
            "numpy_version_oracle": oracle_versions.get("numpy_version_oracle", "unknown"),
            "numpy_version_project": np.__version__,
            "oracle_python": args.oracle_python,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "command": ".venv/bin/python tests/tools/gen_uv_oracle_anchors.py",
            "option_mapping": OPTION_MAPPING,
            "stage_a_knobs": STAGE_A_KNOBS,
            "qem_pipeline": QEM_PIPELINE,
        },
        "fixtures": fixtures_payload,
    }
    ANCHORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANCHORS_PATH.write_text(json.dumps(anchors, indent=2, sort_keys=True) + "\n")
    log(f"anchors written to {ANCHORS_PATH}")
    log("=== gen_uv_oracle_anchors run complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase-b",
        nargs=2,
        metavar=("IN_NPZ", "OUT_NPZ"),
        help="internal: run phase B (oracle venv) on a scratch NPZ",
    )
    parser.add_argument(
        "--fixtures",
        default=",".join(FIXTURES),
        help="comma-separated subset of fixtures to (re)generate "
        f"(default: {','.join(FIXTURES)})",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="reuse cached QEM meshes under /tmp/uv-oracle-cache/",
    )
    parser.add_argument(
        "--oracle-python",
        default=ORACLE_PYTHON_DEFAULT,
        help=f"oracle venv python with pip xatlas (default: {ORACLE_PYTHON_DEFAULT})",
    )
    args = parser.parse_args()
    if args.phase_b:
        run_phase_b(args.phase_b[0], args.phase_b[1])
        return
    if not [n.strip() for n in args.fixtures.split(",") if n.strip()]:
        parser.error("--fixtures must name at least one fixture")
    run_phase_a(args)


if __name__ == "__main__":
    main()
