from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
from safetensors.numpy import save_file

from mlx_spatial.lito import main as lito_main


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_exposes_lito_script_entry():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert data["project"]["scripts"]["mlx-spatial-lito"] == "mlx_spatial.lito:main"


def test_cli_validate_returns_zero_on_valid_weights(tmp_path, capsys):
    root = _write_valid_weights(tmp_path / "weights")

    assert lito_main(["validate", str(root)]) == 0
    output = capsys.readouterr().out

    assert "ready=True" in output


def test_cli_generate_produces_ply(tmp_path, capsys):
    image = _write_synthetic_image(tmp_path / "input.png")
    output = tmp_path / "test.ply"

    status = lito_main(
        [
            "generate",
            str(image),
            "--weights-root",
            str(tmp_path / "weights"),
            "--output",
            str(output),
            "--format",
            "ply",
            "--memory-profile",
            "safe",
            "--resolution",
            "24",
            "--render-size",
            "12",
            "--source-contract-smoke",
        ]
    )

    assert status == 0
    assert "gaussians=64" in capsys.readouterr().out
    _assert_valid_ply(output, expected_vertices=64)
    assert output.with_suffix(".safetensors").is_file()


def test_cli_generate_honors_global_root_before_subcommand(tmp_path, monkeypatch):
    image = _write_synthetic_image(tmp_path / "input.png")
    output = tmp_path / "test.ply"
    observed: dict[str, object] = {}

    class RecordingPipeline:
        def __init__(
            self,
            weights_root,
            *,
            memory_profile,
            max_init_coords_per_batch="profile",
            source_contract_smoke=False,
        ):
            observed["weights_root"] = str(weights_root)
            observed["memory_profile"] = memory_profile
            observed["max_init_coords_per_batch"] = max_init_coords_per_batch
            observed["source_contract_smoke"] = source_contract_smoke

        def generate(self, image_path, **kwargs):
            observed["image_path"] = str(image_path)
            observed["output_path"] = str(kwargs["output_path"])
            observed["ply_storage"] = kwargs["ply_storage"]
            return type("Result", (), {"gaussians": {"xyz_w": np.zeros((1, 3), dtype=np.float32)}})()

    monkeypatch.setattr("mlx_spatial.lito.LitoInferencePipeline", RecordingPipeline)

    status = lito_main(
        [
            "--root",
            str(tmp_path / "global-weights"),
            "generate",
            str(image),
            "--output",
            str(output),
            "--memory-profile",
            "safe",
            "--max-init-coords-per-batch",
            "none",
            "--ply-storage",
            "ascii",
            "--source-contract-smoke",
        ]
    )

    assert status == 0
    assert observed == {
        "weights_root": str(tmp_path / "global-weights"),
        "memory_profile": "safe",
        "max_init_coords_per_batch": None,
        "source_contract_smoke": True,
        "image_path": str(image),
        "output_path": str(output),
        "ply_storage": "ascii",
    }


def test_cli_generate_format_safetensors(tmp_path):
    image = _write_synthetic_image(tmp_path / "input.png")
    output = tmp_path / "test.safetensors"

    assert (
        lito_main(
            [
                "generate",
                str(image),
                "--output",
                str(output),
                "--format",
                "safetensors",
                "--memory-profile",
                "safe",
                "--resolution",
                "24",
                "--render-size",
                "12",
                "--source-contract-smoke",
            ]
        )
        == 0
    )
    assert output.is_file()


def test_cli_generate_format_splat(tmp_path):
    image = _write_synthetic_image(tmp_path / "input.png")
    output = tmp_path / "test.splat"

    assert (
        lito_main(
            [
                "generate",
                str(image),
                "--output",
                str(output),
                "--format",
                "splat",
                "--memory-profile",
                "safe",
                "--resolution",
                "24",
                "--render-size",
                "12",
                "--source-contract-smoke",
            ]
        )
        == 0
    )
    data = output.read_bytes()
    assert data.startswith(b"LITO_SPLAT_SMOKE\0")
    assert output.with_suffix(".safetensors").is_file()


def test_cli_generate_print_metrics_logs_all_stages(tmp_path, capsys):
    image = _write_synthetic_image(tmp_path / "input.png")
    output = tmp_path / "test.ply"

    assert (
        lito_main(
            [
                "generate",
                str(image),
                "--output",
                str(output),
                "--format",
                "ply",
                "--memory-profile",
                "safe",
                "--resolution",
                "24",
                "--render-size",
                "12",
                "--print-metrics",
                "--source-contract-smoke",
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    for stage in ("preprocess", "condition", "tokenize", "dit", "decode", "render", "export"):
        assert f'"{stage}"' in text


def test_scripts_lito_generate_wrapper_works(tmp_path):
    image = _write_synthetic_image(tmp_path / "input.png")
    output = tmp_path / "wrapper.ply"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/lito/generate.py",
            str(image),
            "--weights-root",
            str(tmp_path / "weights"),
            "--output",
            str(output),
            "--memory-profile",
            "safe",
            "--resolution",
            "24",
            "--render-size",
            "12",
            "--source-contract-smoke",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    _assert_valid_ply(output, expected_vertices=64)


def test_cli_generate_fails_closed_without_smoke_flag(tmp_path, capsys):
    image = _write_synthetic_image(tmp_path / "input.png")
    output = tmp_path / "test.ply"

    status = lito_main(
        [
            "generate",
            str(image),
            "--weights-root",
            str(tmp_path / "missing-weights"),
            "--output",
            str(output),
            "--memory-profile",
            "safe",
        ]
    )

    assert status == 1
    assert not output.exists()
    assert "checkpoint-backed LiTo generation requires converted assets" in capsys.readouterr().out


def test_cli_generate_rejects_placeholder_weights_without_smoke_flag(tmp_path, capsys):
    root = _write_valid_weights(tmp_path / "weights")
    image = _write_synthetic_image(tmp_path / "input.png")
    output = tmp_path / "test.ply"

    status = lito_main(
        [
            "generate",
            str(image),
            "--weights-root",
            str(root),
            "--output",
            str(output),
            "--memory-profile",
            "safe",
        ]
    )

    assert status == 1
    assert not output.exists()
    assert "missing required real tensor keys" in capsys.readouterr().out


def _write_valid_weights(root: Path) -> Path:
    (root / "tokenizer").mkdir(parents=True, exist_ok=True)
    (root / "image_to_3d").mkdir(parents=True, exist_ok=True)
    save_file({"tokenizer.weight": np.array([1.0], dtype=np.float32)}, root / "tokenizer" / "lito_new.safetensors")
    save_file(
        {"dit.weight": np.array([1.0], dtype=np.float32)},
        root / "image_to_3d" / "lito_dit_rgba.safetensors",
    )
    return root


def _write_synthetic_image(path: Path) -> Path:
    rgba = np.zeros((36, 36, 4), dtype=np.uint8)
    rgba[..., 0] = 120
    rgba[..., 1] = np.arange(36, dtype=np.uint8)[None, :] * 5
    rgba[..., 2] = np.arange(36, dtype=np.uint8)[:, None] * 5
    rgba[6:30, 6:30, 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(path)
    return path


def _assert_valid_ply(path: Path, *, expected_vertices: int) -> None:
    ply = PlyData.read(path)
    assert ply.elements[0].count == expected_vertices

    lines = path.read_text(encoding="ascii").splitlines()
    assert lines[0] == "ply"
    assert lines[1] == "format ascii 1.0"
    assert f"element vertex {expected_vertices}" in lines
    header_end = lines.index("end_header")
    assert len(lines[header_end + 1 :]) == expected_vertices
    assert len(lines[header_end + 1].split()) == 14
