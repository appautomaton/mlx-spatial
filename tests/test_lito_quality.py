from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from mlx_spatial.lito_real_backend import write_lito_gaussians_ply


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lito" / "inspect_quality.py"
_SPEC = importlib.util.spec_from_file_location("inspect_quality", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
inspect_quality = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(inspect_quality)


def _write_checkpoint_ply(path: Path, *, offset: float = 0.0) -> None:
    gaussians = {
        "xyz_w": np.array([[0.0 + offset, 0.1, 0.2], [0.3 + offset, 0.4, 0.5]], dtype=np.float32),
        "scaling": np.full((2, 3), 0.02, dtype=np.float32),
        "quaternion": np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
        "opacity": np.full((2, 1), 0.8, dtype=np.float32),
        "rgb_sh": np.arange(96, dtype=np.float32).reshape(2, 16, 3) / 100.0,
    }
    write_lito_gaussians_ply(path, gaussians)


def test_inspect_lito_ply_reports_checkpoint_schema_stats(tmp_path):
    path = tmp_path / "checkpoint.ply"
    _write_checkpoint_ply(path)

    report = inspect_quality.inspect_lito_ply(path)

    assert report["checkpoint_backed"] is True
    assert report["source_contract_smoke"] is False
    assert report["vertex_count"] == 2
    assert report["property_count"] == 62
    assert np.allclose(report["xyz"]["bbox_span"], [0.3, 0.3, 0.3])
    assert np.isclose(report["opacity_probability"]["median"], 0.8)
    assert np.isclose(report["scale_exp"]["median"], 0.02)
    assert np.isclose(report["quaternion_norm"]["median"], 1.0)
    assert report["failure_classification"] == "stats_sane_visual_review_required"
    assert report["flags"] == []


def test_inspect_lito_ply_marks_smoke_schema_only(tmp_path):
    path = tmp_path / "smoke.ply"
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "comment mlx-spatial LiTo source-contract smoke 3DGS export",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "property float scale_0",
                "property float scale_1",
                "property float scale_2",
                "property float rot_0",
                "property float rot_1",
                "property float rot_2",
                "property float rot_3",
                "property float opacity",
                "property float red",
                "property float green",
                "property float blue",
                "end_header",
                "0 0 0 0.1 0.1 0.1 0 0 0 1 1 1 1 1",
            ]
        )
        + "\n",
        encoding="ascii",
    )

    report = inspect_quality.inspect_lito_ply(path)

    assert report["checkpoint_backed"] is False
    assert report["source_contract_smoke"] is True
    assert report["failure_classification"] == "invalid_or_schema_only"
    assert {"not_checkpoint_backed", "source_contract_smoke", "bbox_collapsed"}.issubset(report["flags"])


def test_inspect_quality_cli_writes_json_and_compare(tmp_path):
    baseline = tmp_path / "baseline.ply"
    current = tmp_path / "current.ply"
    baseline_json = tmp_path / "baseline.json"
    current_json = tmp_path / "current.json"
    _write_checkpoint_ply(baseline)
    _write_checkpoint_ply(current, offset=0.2)

    subprocess.run(
        [sys.executable, str(_SCRIPT), str(baseline), "--json", str(baseline_json)],
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [sys.executable, str(_SCRIPT), str(current), "--compare", str(baseline_json), "--json", str(current_json)],
        check=True,
        text=True,
        capture_output=True,
    )

    report = json.loads(current_json.read_text(encoding="utf-8"))
    assert report["comparison"]["vertex_count_delta"] == 0
    assert np.allclose(report["comparison"]["bbox_span_delta"], [0.0, 0.0, 0.0], atol=1e-7)
