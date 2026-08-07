from pathlib import Path

import pytest

from mlx_spatial.ovoxel_export import export_ovoxel_glb


def test_export_ovoxel_glb_applies_one_production_policy(tmp_path: Path) -> None:
    calls = {}
    result = object()

    def exporter(decoded_dir, output, **kwargs):
        calls["decoded_dir"] = decoded_dir
        calls["output"] = output
        calls["kwargs"] = kwargs
        return result

    actual = export_ovoxel_glb(
        tmp_path / "decoded",
        tmp_path / "model.glb",
        texture_size=1024,
        target_faces=200_000,
        grid_size=512,
        diagnostics_path=tmp_path / "diagnostics.json",
        exporter=exporter,
    )

    assert actual is result
    assert calls == {
        "decoded_dir": tmp_path / "decoded",
        "output": tmp_path / "model.glb",
        "kwargs": {
            "texture_size": 1024,
            "target_faces": 200_000,
            "quality_preset": "reference-target",
            "grid_size": 512,
            "uv_backend": "xatlas-clustered",
            "remesh": True,
            "remesh_resolution": 512,
            "simplify_backend": "mlx-qem",
            "texture_postprocess": "telea",
            "diagnostics_path": tmp_path / "diagnostics.json",
        },
    }


@pytest.mark.parametrize(
    ("name", "overrides"),
    (
        ("texture_size", {"texture_size": 0}),
        ("target_faces", {"target_faces": 0}),
        ("grid_size", {"grid_size": 0}),
    ),
)
def test_export_ovoxel_glb_rejects_nonpositive_policy_values(tmp_path: Path, name: str, overrides: dict) -> None:
    values = {
        "texture_size": 1024,
        "target_faces": 200_000,
        "grid_size": 512,
        **overrides,
    }

    with pytest.raises(ValueError, match=rf"{name} must be positive"):
        export_ovoxel_glb(
            tmp_path / "decoded",
            tmp_path / "model.glb",
            exporter=lambda *_args, **_kwargs: object(),
            **values,
        )
