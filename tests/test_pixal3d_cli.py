import json

from mlx_spatial.pixal3d import main as pixal3d_main
from pixal3d_fixtures import write_fake_pixal3d_root


def test_pixal3d_cli_download_command(capsys):
    result = pixal3d_main(["download-command", "weights/pixal3d"])

    assert result == 0
    assert capsys.readouterr().out.strip() == "uv run hf download TencentARC/Pixal3D --local-dir weights/pixal3d"


def test_pixal3d_cli_validate_reports_ready_json(tmp_path, capsys):
    write_fake_pixal3d_root(tmp_path)

    result = pixal3d_main(["validate", str(tmp_path), "--json"])

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True
    assert payload["pipeline"]["default_pipeline_type"] == "1536_cascade"
    assert len(payload["pipeline"]["models"]) == 7


def test_pixal3d_cli_validate_reports_missing(tmp_path, capsys):
    result = pixal3d_main(["validate", str(tmp_path)])

    assert result == 1
    output = capsys.readouterr().out
    assert "ready=False" in output
    assert "missing pipeline.json" in output


def test_pixal3d_cli_generate_writes_trace_for_skeleton_boundary(tmp_path, capsys):
    root = write_fake_pixal3d_root(tmp_path / "weights")
    image = tmp_path / "input.png"
    image.write_bytes(b"not decoded yet")
    trace = tmp_path / "trace.json"

    result = pixal3d_main(
        [
            "generate",
            str(image),
            "--root",
            str(root),
            "--manual-fov",
            "0.2",
            "--output-dir",
            str(tmp_path / "out"),
            "--trace-output",
            str(trace),
        ]
    )

    assert result == 2
    output = capsys.readouterr().out
    assert "blocker_stage=input-preprocessing" in output
    payload = json.loads(trace.read_text(encoding="utf-8"))
    assert payload["completed_stages"] == ["input-image", "asset-validation", "pipeline-config"]
    assert payload["blocker"]["stage"] == "input-preprocessing"
    assert payload["metadata"]["stage_plan"]["actual_hr_resolution"] == 1024
