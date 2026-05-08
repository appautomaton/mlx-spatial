import os
import subprocess
import sys

import mlx.core as mx
import numpy as np

from mlx_spatial.hyworld2_inference import HyWorld2InferencePipeline
from mlx_spatial.hyworld2 import main
from mlx_spatial.hyworld2_parity import (
    HYWORLD2_TORCH_PARITY_ENV,
    compare_hyworld2_parity_tensors,
    hyworld2_parity_trace_metadata,
    load_hyworld2_parity_bundle,
    parity_report_to_dict,
    require_hyworld2_torch_parity_enabled,
    write_hyworld2_parity_bundle,
)


def test_hyworld2_parity_bundle_round_trips_metadata_and_arrays(tmp_path):
    bundle = write_hyworld2_parity_bundle(
        tmp_path / "reference.npz",
        {
            "predictions.depth": np.array([[1.0, 2.0]], dtype=np.float32),
            "predictions.camera_params": mx.array([[0.0, 1.0]], dtype=mx.float32),
        },
        metadata={"case": "tiny"},
    )

    reference = load_hyworld2_parity_bundle(bundle)

    assert reference.metadata["case"] == "tiny"
    assert reference.metadata["source"].endswith("WorldMirror PyTorch")
    assert sorted(reference.tensors) == ["predictions.camera_params", "predictions.depth"]
    np.testing.assert_allclose(reference.tensors["predictions.depth"], [[1.0, 2.0]])


def test_hyworld2_parity_compare_reports_pass_and_value_mismatch(tmp_path):
    reference = load_hyworld2_parity_bundle(
        write_hyworld2_parity_bundle(
            tmp_path / "reference.npz",
            {"a": np.array([1.0, 2.0], dtype=np.float32), "b": np.array([3.0], dtype=np.float32)},
        )
    )

    report = compare_hyworld2_parity_tensors(
        {"a": mx.array([1.0, 2.00001], dtype=mx.float32), "b": np.array([9.0], dtype=np.float32)},
        reference,
        atol=1e-4,
        rtol=1e-4,
    )

    assert not report.passed
    assert [comparison.status for comparison in report.comparisons] == ["pass", "value-mismatch"]
    payload = parity_report_to_dict(report)
    assert payload["passed"] is False
    assert payload["failed_names"] == ["b"]
    assert payload["comparisons"][1]["max_abs_error"] == 6.0


def test_hyworld2_parity_compare_reports_missing_and_shape_mismatch():
    report = compare_hyworld2_parity_tensors(
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


def test_hyworld2_parity_trace_metadata_defaults_to_unverified():
    metadata = hyworld2_parity_trace_metadata()

    assert metadata["runtime_depends_on_torch"] is False
    assert metadata["numeric_parity_verified"] is False
    assert metadata["status"] == "unverified"
    assert metadata["dev_reference_env"] == HYWORLD2_TORCH_PARITY_ENV


def test_hyworld2_reconstruction_trace_includes_unverified_parity_metadata(tmp_path):
    result = HyWorld2InferencePipeline(tmp_path / "missing").reconstruct(
        tmp_path / "input",
        output_path="outputs/hyworld2/parity-missing-assets",
    )

    assert result.trace.metadata["parity"]["runtime_depends_on_torch"] is False
    assert result.trace.metadata["parity"]["numeric_parity_verified"] is False


def test_hyworld2_torch_reference_guard_requires_explicit_env(monkeypatch):
    monkeypatch.delenv(HYWORLD2_TORCH_PARITY_ENV, raising=False)

    try:
        require_hyworld2_torch_parity_enabled()
    except RuntimeError as error:
        assert HYWORLD2_TORCH_PARITY_ENV in str(error)
    else:
        raise AssertionError("expected HY-World Torch parity guard to block")

    monkeypatch.setenv(HYWORLD2_TORCH_PARITY_ENV, "1")
    require_hyworld2_torch_parity_enabled()


def test_hyworld2_torch_reference_script_blocks_without_env():
    env = dict(os.environ)
    env.pop(HYWORLD2_TORCH_PARITY_ENV, None)
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "tools/hyworld2_dump_torch_reference.py",
            "weights/hy-world-2",
            "inputs/hyworld2",
            "--output",
            "outputs/hyworld2/reference.npz",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert HYWORLD2_TORCH_PARITY_ENV in result.stderr


def test_hyworld2_cli_parity_compare_reports_mismatches(tmp_path, capsys):
    reference = write_hyworld2_parity_bundle(
        tmp_path / "reference.npz",
        {"tensor": np.array([1.0], dtype=np.float32)},
    )
    actual = write_hyworld2_parity_bundle(
        tmp_path / "actual.npz",
        {"tensor": np.array([2.0], dtype=np.float32)},
    )
    json_output = tmp_path / "report.json"

    assert main(
        [
            "parity-compare",
            str(reference),
            str(actual),
            "--json-output",
            str(json_output),
        ]
    ) == 1

    output = capsys.readouterr().out
    assert "passed=False" in output
    assert "mismatch tensor status=value-mismatch" in output
    assert '"passed": false' in json_output.read_text(encoding="utf-8")
