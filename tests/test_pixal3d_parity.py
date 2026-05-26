import os
import subprocess
import sys

import mlx.core as mx
import numpy as np

import mlx_spatial
from mlx_spatial.pixal3d_parity import (
    PIXAL3D_TORCH_PARITY_ENV,
    load_pixal3d_parity_bundle,
    pixal3d_parity_trace_metadata,
    require_pixal3d_torch_parity_enabled,
    write_pixal3d_parity_bundle,
)


def test_pixal3d_parity_bundle_round_trips_metadata_and_arrays(tmp_path):
    bundle = write_pixal3d_parity_bundle(
        tmp_path / "reference.npz",
        {
            "projection.global": np.array([[1.0, 2.0]], dtype=np.float32),
            "projection.grid": mx.array([[3.0]], dtype=mx.float32),
        },
        metadata={"case": "tiny"},
    )

    reference = load_pixal3d_parity_bundle(bundle)

    assert reference.metadata["case"] == "tiny"
    assert reference.metadata["source"].endswith("PyTorch reference")
    assert sorted(reference.tensors) == ["projection.global", "projection.grid"]
    np.testing.assert_allclose(reference.tensors["projection.global"], [[1.0, 2.0]])


def test_pixal3d_parity_trace_metadata_defaults_to_unverified():
    metadata = pixal3d_parity_trace_metadata()

    assert metadata["runtime_depends_on_torch"] is False
    assert metadata["numeric_parity_verified"] is False
    assert metadata["status"] == "unverified"
    assert metadata["dev_reference_env"] == PIXAL3D_TORCH_PARITY_ENV


def test_pixal3d_torch_reference_guard_requires_explicit_env(monkeypatch):
    monkeypatch.delenv(PIXAL3D_TORCH_PARITY_ENV, raising=False)

    try:
        require_pixal3d_torch_parity_enabled()
    except RuntimeError as error:
        assert PIXAL3D_TORCH_PARITY_ENV in str(error)
    else:
        raise AssertionError("expected Pixal3D Torch parity guard to block")

    monkeypatch.setenv(PIXAL3D_TORCH_PARITY_ENV, "1")
    require_pixal3d_torch_parity_enabled()


def test_pixal3d_torch_reference_script_blocks_without_env(tmp_path):
    env = dict(os.environ)
    env.pop(PIXAL3D_TORCH_PARITY_ENV, None)
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "tools/pixal3d_dump_torch_reference.py",
            "weights/pixal3d",
            "vendors/Pixal3D/assets/images/0_img.png",
            "--output",
            str(tmp_path / "reference.npz"),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert PIXAL3D_TORCH_PARITY_ENV in result.stderr
    assert not (tmp_path / "reference.npz").exists()


def test_pixal3d_parity_helpers_are_public_exports():
    assert mlx_spatial.PIXAL3D_TORCH_PARITY_ENV == PIXAL3D_TORCH_PARITY_ENV
    assert mlx_spatial.write_pixal3d_parity_bundle is write_pixal3d_parity_bundle
    assert mlx_spatial.load_pixal3d_parity_bundle is load_pixal3d_parity_bundle
