#!/usr/bin/env python
"""Generate version-pinned Telea-inpaint parity oracle anchors.

Writes ``tests/data/inpaint_oracle_anchors.json`` (committed, small: hashes +
tolerances + tiny synthetic cv2 outputs) and a regenerable per-fixture cache
under ``/tmp/inpaint-oracle-cache/`` (raw channels + mask + cv2 outputs) that
the heavy parity test consumes.

For each cached Pixal3D fixture this script:
  1. Phase A (PROJECT venv, this process): runs ``export_pixal3d_glb`` with
     ``expose_raw_postprocess_inputs=True`` to obtain the raw pre-postprocess
     bake channels + coverage status (the Codex P0-1 contract), derives the
     reference inverse-coverage mask (status in {0,2,3}), runs our native
     ``telea_inpaint`` per reference channel group, and dumps raw channels +
     mask + our outputs to a /tmp scratch NPZ.
  2. Phase B (ORACLE venv subprocess, /tmp/inpaint-oracle-venv, pinned
     opencv-python-headless): runs ``cv2.inpaint(..., INPAINT_TELEA)`` per
     channel group and dumps the cv2 outputs to a /tmp NPZ.
  3. Phase A again: computes per-group parity stats (full mask + near-coverage
     band) of ours vs cv2, writes the combined cache + the anchors JSON.

The PROJECT venv must NOT have cv2 installed; cv2 is only imported inside the
phase-B subprocess running under the oracle venv.

Usage (from packages/mlx-spatialkit, PROJECT venv):
    .venv/bin/python tests/tools/gen_inpaint_oracle_anchors.py
    .venv/bin/python tests/tools/gen_inpaint_oracle_anchors.py --fixtures main
    .venv/bin/python tests/tools/gen_inpaint_oracle_anchors.py --texture-size 512

Oracle venv bootstrap (one-time; the PROJECT venv must NOT have cv2):
    python3 -m venv /tmp/inpaint-oracle-venv && \\
        /tmp/inpaint-oracle-venv/bin/pip install opencv-python-headless==4.10.0.84 numpy

Internal phase-B invocation (dispatched automatically, do not run by hand):
    /tmp/inpaint-oracle-venv/bin/python tests/tools/gen_inpaint_oracle_anchors.py \\
        --phase-b SCRATCH_IN.npz SCRATCH_OUT.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ORACLE_PYTHON_DEFAULT = "/tmp/inpaint-oracle-venv/bin/python"
ORACLE_CV2_VERSION = "4.10.0.84"
ORACLE_BOOTSTRAP_RECIPE = (
    "python3 -m venv /tmp/inpaint-oracle-venv && "
    f"/tmp/inpaint-oracle-venv/bin/pip install opencv-python-headless=={ORACLE_CV2_VERSION} numpy"
)
CACHE_DIR = Path("/tmp/inpaint-oracle-cache")
LOG_PATH = Path("/tmp/inpaint-oracle-anchors-run.log")

PKG_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
ANCHORS_PATH = PKG_ROOT / "tests" / "data" / "inpaint_oracle_anchors.json"

FIXTURES = {
    "main": REPO_ROOT / "inputs" / "mlx-spatialkit" / "pixal3d-1024-cascade-decoded-pbr",
    "violin_bow": REPO_ROOT / "inputs" / "mlx-spatialkit" / "violin-bow" / "pixal3d-1024-cascade-decoded-pbr",
}

# Reference channel groups (TRELLIS o-voxel postprocess.py): base_color RGB at
# radius 3; metallic / roughness / alpha as single channels at radius 1. Our
# packing carries alpha in base_color_rgba[...,3] and the metallic/roughness
# channels in metallic_roughness[...,0:3].
INVERSE_COVERAGE_STATUSES = (0, 2, 3)  # no_face, missing_surface, out_of_grid


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def build_groups(base_color_rgba: np.ndarray, metallic_roughness: np.ndarray) -> dict:
    """Return {group_name: {"image": uint8 (H,W[,3]), "radius": int}}."""
    return {
        "base_color_rgb": {"image": np.ascontiguousarray(base_color_rgba[:, :, :3]), "radius": 3},
        "alpha": {"image": np.ascontiguousarray(base_color_rgba[:, :, 3]), "radius": 1},
        "metallic_roughness_0": {"image": np.ascontiguousarray(metallic_roughness[:, :, 0]), "radius": 1},
        "metallic_roughness_1": {"image": np.ascontiguousarray(metallic_roughness[:, :, 1]), "radius": 1},
        "metallic_roughness_2": {"image": np.ascontiguousarray(metallic_roughness[:, :, 2]), "radius": 1},
    }


def near_coverage_band(mask: np.ndarray, covered: np.ndarray, rings: int = 2) -> np.ndarray:
    """Masked texels within `rings` 4-neighbour steps of a covered texel."""
    reach = covered.copy()
    for _ in range(rings):
        nxt = reach.copy()
        nxt[1:, :] |= reach[:-1, :]
        nxt[:-1, :] |= reach[1:, :]
        nxt[:, 1:] |= reach[:, :-1]
        nxt[:, :-1] |= reach[:, 1:]
        reach = nxt
    return mask & reach


def group_stats(ours: np.ndarray, ref: np.ndarray, mask: np.ndarray, band: np.ndarray) -> dict:
    if ours.ndim == 3:
        m_full = np.repeat(mask[:, :, None], ours.shape[2], axis=2)
        m_band = np.repeat(band[:, :, None], ours.shape[2], axis=2)
    else:
        m_full, m_band = mask, band
    diff = np.abs(ours.astype(np.int32) - ref.astype(np.int32))
    full = diff[m_full]
    bandvals = diff[m_band]

    def summary(vals: np.ndarray) -> dict:
        if vals.size == 0:
            return {"n": 0, "max": 0, "mean": 0.0, "p95": 0.0, "p99": 0.0}
        return {
            "n": int(vals.size),
            "max": int(vals.max()),
            "mean": float(round(float(vals.mean()), 4)),
            "p95": float(np.percentile(vals, 95)),
            "p99": float(np.percentile(vals, 99)),
        }

    return {"full_mask": summary(full), "near_coverage_band": summary(bandvals)}


# --------------------------------------------------------------------------
# Phase B: cv2 oracle (runs ONLY under the oracle venv)
# --------------------------------------------------------------------------
def run_phase_b(scratch_in: Path, scratch_out: Path) -> None:
    import cv2  # noqa: PLC0415 — only available in the oracle venv

    data = np.load(scratch_in)
    mask = data["mask"]
    out = {"cv2_version": np.array(cv2.__version__)}
    group_names = [k[len("img__"):] for k in data.files if k.startswith("img__")]
    for name in group_names:
        img = np.ascontiguousarray(data[f"img__{name}"])
        radius = int(data[f"radius__{name}"])
        ref = cv2.inpaint(img, np.ascontiguousarray(mask), radius, cv2.INPAINT_TELEA)
        out[f"cv2__{name}"] = ref
    np.savez(scratch_out, **out)


# --------------------------------------------------------------------------
# Phase A helpers
# --------------------------------------------------------------------------
def synthetic_cases() -> list[dict]:
    """Tiny deterministic image+mask cases for the non-heavy oracle test."""
    cases = []
    # 1) RGB ramp with a block hole, r=3
    rng = np.random.default_rng(11)
    cols = np.clip(20 + 4 * np.arange(24), 0, 255)
    ramp = np.repeat(cols[None, :], 18, axis=0)
    rgb = np.stack([ramp, 255 - ramp, ramp // 2], axis=-1).astype(np.uint8)
    m = np.zeros((18, 24), np.uint8); m[6:12, 9:15] = 1
    cases.append({"name": "ramp_rgb_r3", "image": rgb, "mask": m, "radius": 3})
    # 2) single-channel smooth field, thin border hole, r=1
    yy, xx = np.mgrid[0:20, 0:20]
    field = (120 + 50 * np.sin(xx / 4.0) + 30 * np.cos(yy / 5.0)).clip(0, 255).astype(np.uint8)
    m2 = np.zeros((20, 20), np.uint8); m2[9:11, 4:16] = 1
    cases.append({"name": "smooth_c1_r1", "image": field, "mask": m2, "radius": 1})
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-b", nargs=2, metavar=("IN", "OUT"))
    parser.add_argument("--oracle-python", default=ORACLE_PYTHON_DEFAULT)
    parser.add_argument("--fixtures", nargs="*", default=list(FIXTURES))
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--target-faces", type=int, default=50_000)
    args = parser.parse_args()

    if args.phase_b is not None:
        run_phase_b(Path(args.phase_b[0]), Path(args.phase_b[1]))
        return 0

    from mlx_spatialkit import export_pixal3d_glb, metal_device_available, telea_inpaint

    if not metal_device_available():
        print("Metal device unavailable; cannot bake fixtures.", file=sys.stderr)
        return 2
    oracle_python = Path(args.oracle_python)
    if not oracle_python.exists():
        print(f"oracle venv missing: {oracle_python}\nbootstrap: {ORACLE_BOOTSTRAP_RECIPE}", file=sys.stderr)
        return 2

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    anchors: dict = {
        "schema": 1,
        "oracle": {"cv2_version_pinned": ORACLE_CV2_VERSION, "method": "INPAINT_TELEA"},
        "inverse_coverage_statuses": list(INVERSE_COVERAGE_STATUSES),
        "texture_size": args.texture_size,
        "fixtures": {},
        "synthetic": [],
    }

    # --- synthetic cases (always emitted; consumed by the non-heavy test) ---
    syn_in = CACHE_DIR / "synthetic_in.npz"
    syn_payload = {"mask": np.zeros((1, 1), np.uint8)}  # placeholder; per-case below
    for case in synthetic_cases():
        case_in = CACHE_DIR / f"syn_{case['name']}_in.npz"
        np.savez(case_in, mask=case["mask"], **{f"img__{case['name']}": case["image"], f"radius__{case['name']}": case["radius"]})
        case_out = CACHE_DIR / f"syn_{case['name']}_out.npz"
        subprocess.run([str(oracle_python), str(Path(__file__).resolve()), "--phase-b", str(case_in), str(case_out)], check=True)
        cv2_out = np.load(case_out)[f"cv2__{case['name']}"]
        ours = telea_inpaint(case["image"], case["mask"], case["radius"])
        diff = np.abs(ours.astype(int) - cv2_out.astype(int))
        msk = case["mask"] != 0
        m = np.repeat(msk[:, :, None], ours.shape[2], 2) if ours.ndim == 3 else msk
        anchors["synthetic"].append({
            "name": case["name"],
            "radius": case["radius"],
            "image": case["image"].tolist(),
            "mask": case["mask"].tolist(),
            "cv2_output": cv2_out.tolist(),
            "measured_max_abs_err": int(diff[m].max()),
            "tolerance_max_abs_err": int(diff[m].max()) + 2,
        })
        print(f"[synthetic] {case['name']}: max|ours-cv2|={int(diff[m].max())}")

    # --- real fixtures ---
    for key in args.fixtures:
        fixture = FIXTURES[key]
        if not fixture.exists():
            print(f"[skip] fixture missing: {fixture}", file=sys.stderr)
            continue
        out_dir = Path("/tmp") / f"inpaint-oracle-export-{key}"
        result = export_pixal3d_glb(
            fixture, out_dir, texture_size=args.texture_size,
            target_faces=args.target_faces, expose_raw_postprocess_inputs=True,
        )
        raw = result.raw_texture_inputs
        assert raw is not None, "raw_texture_inputs not returned"
        base = raw["raw_base_color_rgba"]; mr = raw["raw_metallic_roughness"]; status = raw["raw_coverage_status"]
        mask = np.isin(status, INVERSE_COVERAGE_STATUSES).astype(np.uint8)
        covered = ~mask.astype(bool)
        band = near_coverage_band(mask.astype(bool), covered, rings=2)
        groups = build_groups(base, mr)

        # phase A: ours
        scratch_in = CACHE_DIR / f"{key}_in.npz"
        payload = {"mask": mask}
        ours_by_group = {}
        for name, g in groups.items():
            payload[f"img__{name}"] = g["image"]
            payload[f"radius__{name}"] = g["radius"]
            ours_by_group[name] = telea_inpaint(g["image"], mask, g["radius"])
        np.savez(scratch_in, **payload)
        # phase B: cv2
        scratch_out = CACHE_DIR / f"{key}_cv2.npz"
        subprocess.run([str(oracle_python), str(Path(__file__).resolve()), "--phase-b", str(scratch_in), str(scratch_out)], check=True)
        cv2_data = np.load(scratch_out)

        # stats + combined cache (raw channels + mask + cv2 outputs) for heavy test
        fixture_anchor = {
            "fixture_path": str(fixture),
            "raw_base_color_sha256": _sha(base),
            "raw_metallic_roughness_sha256": _sha(mr),
            "raw_coverage_status_sha256": _sha(status),
            "mask_sha256": _sha(mask),
            "masked_texel_count": int(mask.sum()),
            "near_band_texel_count": int(band.sum()),
            "groups": {},
        }
        combined = {"mask": mask, "band": band.astype(np.uint8)}
        for name, g in groups.items():
            ref = cv2_data[f"cv2__{name}"]
            stats = group_stats(ours_by_group[name], ref, mask.astype(bool), band)
            band_p99 = stats["near_coverage_band"]["p99"]
            full_p99 = stats["full_mask"]["p99"]
            fixture_anchor["groups"][name] = {
                "radius": g["radius"],
                "input_sha256": _sha(g["image"]),
                "cv2_output_sha256": _sha(ref),
                "measured": stats,
                # Tolerances: tight near coverage (what bilinear rendering uses),
                # looser full-mask (long-range gutter extrapolation; documented).
                "tolerance_near_band_p99": float(band_p99 + 3),
                "tolerance_full_mask_p99": float(full_p99 + 5),
            }
            combined[f"img__{name}"] = g["image"]
            combined[f"radius__{name}"] = np.array(g["radius"])
            combined[f"cv2__{name}"] = ref
            print(f"[{key}] {name}: band p99={band_p99} full p99={full_p99} full max={stats['full_mask']['max']}")
        np.savez(CACHE_DIR / f"{key}.npz", **combined)
        anchors["fixtures"][key] = fixture_anchor

    ANCHORS_PATH.write_text(json.dumps(anchors, indent=2, sort_keys=True))
    print(f"wrote {ANCHORS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
