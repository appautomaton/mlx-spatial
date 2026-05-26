import os
import subprocess
import sys

import mlx.core as mx
import numpy as np

import mlx_spatial
from mlx_spatial.mapanything_parity import (
    MAPANYTHING_TORCH_PARITY_ENV,
    compare_mapanything_parity_tensors,
    load_mapanything_parity_bundle,
    mapanything_parity_report_to_dict,
    mapanything_parity_trace_metadata,
    require_mapanything_torch_parity_enabled,
    write_mapanything_parity_bundle,
)


def test_mapanything_parity_bundle_round_trips_metadata_and_arrays(tmp_path):
    bundle = write_mapanything_parity_bundle(
        tmp_path / "reference.npz",
        {
            "encoder.patch_embed": np.array([[1.0, 2.0]], dtype=np.float32),
            "encoder.block0": mx.array([[0.0, 1.0]], dtype=mx.float32),
        },
        metadata={"case": "tiny"},
    )

    reference = load_mapanything_parity_bundle(bundle)

    assert reference.metadata["case"] == "tiny"
    assert reference.metadata["source"].endswith("PyTorch")
    assert sorted(reference.tensors) == ["encoder.block0", "encoder.patch_embed"]
    np.testing.assert_allclose(reference.tensors["encoder.patch_embed"], [[1.0, 2.0]])


def test_mapanything_parity_compare_reports_pass_and_value_mismatch(tmp_path):
    reference = load_mapanything_parity_bundle(
        write_mapanything_parity_bundle(
            tmp_path / "reference.npz",
            {"a": np.array([1.0, 2.0], dtype=np.float32), "b": np.array([3.0], dtype=np.float32)},
        )
    )

    report = compare_mapanything_parity_tensors(
        {"a": mx.array([1.0, 2.00001], dtype=mx.float32), "b": np.array([9.0], dtype=np.float32)},
        reference,
        atol=1e-4,
        rtol=1e-4,
    )

    assert not report.passed
    assert [comparison.status for comparison in report.comparisons] == ["pass", "value-mismatch"]
    payload = mapanything_parity_report_to_dict(report)
    assert payload["passed"] is False
    assert payload["failed_names"] == ["b"]
    assert payload["parity_trace_metadata"]["numeric_parity_verified"] is False
    assert payload["comparisons"][1]["max_abs_error"] == 6.0

    passed_payload = mapanything_parity_report_to_dict(
        compare_mapanything_parity_tensors(
            {"a": np.array([1.0, 2.0], dtype=np.float32)},
            {"a": np.array([1.0, 2.0], dtype=np.float32)},
        )
    )
    assert passed_payload["parity_trace_metadata"]["numeric_parity_verified"] is True


def test_mapanything_parity_compare_reports_missing_and_shape_mismatch():
    report = compare_mapanything_parity_tensors(
        {"shape": np.zeros((2, 1), dtype=np.float32)},
        {
            "missing": np.zeros((1,), dtype=np.float32),
            "shape": np.zeros((2, 2), dtype=np.float32),
        },
    )

    assert not report.passed
    assert [comparison.status for comparison in report.comparisons] == [
        "missing-actual",
        "shape-mismatch",
    ]


def test_mapanything_parity_trace_metadata_defaults_to_unverified():
    metadata = mapanything_parity_trace_metadata()

    assert metadata["runtime_depends_on_torch"] is False
    assert metadata["numeric_parity_verified"] is False
    assert metadata["status"] == "unverified"
    assert metadata["dev_reference_env"] == MAPANYTHING_TORCH_PARITY_ENV


def test_mapanything_torch_reference_guard_requires_explicit_env(monkeypatch):
    monkeypatch.delenv(MAPANYTHING_TORCH_PARITY_ENV, raising=False)

    try:
        require_mapanything_torch_parity_enabled()
    except RuntimeError as error:
        assert MAPANYTHING_TORCH_PARITY_ENV in str(error)
    else:
        raise AssertionError("expected MapAnything Torch parity guard to block")

    monkeypatch.setenv(MAPANYTHING_TORCH_PARITY_ENV, "1")
    require_mapanything_torch_parity_enabled()


def test_mapanything_torch_reference_script_blocks_without_env(tmp_path):
    env = dict(os.environ)
    env.pop(MAPANYTHING_TORCH_PARITY_ENV, None)
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "tools/mapanything_dump_torch_reference.py",
            "weights/map-anything",
            "inputs/map-anything/desk",
            "--output",
            str(tmp_path / "reference.npz"),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert MAPANYTHING_TORCH_PARITY_ENV in result.stderr
    assert not (tmp_path / "reference.npz").exists()


def test_mapanything_torch_scene_reference_script_blocks_without_env(tmp_path):
    env = dict(os.environ)
    env.pop(MAPANYTHING_TORCH_PARITY_ENV, None)
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "tools/mapanything_dump_torch_scene_reference.py",
            "weights/map-anything",
            "inputs/map-anything/desk",
            "--output",
            str(tmp_path / "scene-reference.npz"),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert MAPANYTHING_TORCH_PARITY_ENV in result.stderr
    assert not (tmp_path / "scene-reference.npz").exists()


def test_mapanything_parity_helpers_are_public_exports():
    assert mlx_spatial.MAPANYTHING_TORCH_PARITY_ENV == MAPANYTHING_TORCH_PARITY_ENV
    assert mlx_spatial.write_mapanything_parity_bundle is write_mapanything_parity_bundle
    assert mlx_spatial.compare_mapanything_parity_tensors is compare_mapanything_parity_tensors
