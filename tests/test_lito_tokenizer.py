from __future__ import annotations

import ast
import json
import logging
import re
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from safetensors import safe_open

from mlx_spatial.lito_tokenizer import LitoTokenizer


FIXTURE_ROOT = Path("tests/fixtures/lito")
GAP_MATRIX = Path(".agent/work/2026-05-22-lito-mlx-inference-pipeline/spec/gap-matrix.md")
TOKENIZER_SOURCE = Path("src/mlx_spatial/lito_tokenizer.py")
LOGGER = logging.getLogger(__name__)


def _load_safetensors(path: Path) -> dict[str, np.ndarray]:
    with safe_open(path, framework="numpy") as handle:
        return {name: handle.get_tensor(name) for name in handle.keys()}


def _tokenizer_tolerances() -> tuple[float, float]:
    text = GAP_MATRIX.read_text(encoding="utf-8")
    section = text.split("## LITO-C", maxsplit=1)[1].split("## LITO-D", maxsplit=1)[0]
    match = re.search(r"atol=([0-9.eE+-]+),\s*rtol=([0-9.eE+-]+)", section)
    assert match is not None
    return float(match.group(1)), float(match.group(2))


def _record_metrics(caplog, name: str, call):
    caplog.set_level(logging.INFO)
    _reset_peak_memory_if_available()
    start = time.perf_counter()
    out = call()
    mx.eval(out)
    if hasattr(mx, "synchronize"):
        mx.synchronize()
    elapsed = time.perf_counter() - start
    peak_gb = _peak_memory_gb_or_zero()
    LOGGER.info(
        "lito_tokenizer_metrics case=%s wall_time_s=%.6f peak_active_memory_gb=%.6f",
        name,
        elapsed,
        peak_gb,
    )
    return out, {"wall_time_s": elapsed, "peak_active_memory_gb": peak_gb}


def _reset_peak_memory_if_available() -> None:
    metal = getattr(mx, "metal", None)
    if metal is not None and hasattr(metal, "reset_peak_memory"):
        try:
            metal.reset_peak_memory()
        except RuntimeError:
            return


def _peak_memory_gb_or_zero() -> float:
    metal = getattr(mx, "metal", None)
    if metal is None or not hasattr(metal, "get_peak_memory"):
        return 0.0
    try:
        return metal.get_peak_memory() / (1024**3)
    except RuntimeError:
        return 0.0


@pytest.fixture(scope="module")
def tokenizer_manifest() -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))["tokenizer"]


def test_tokenizer_manifest_records_source_contract_metadata(tokenizer_manifest):
    assert tokenizer_manifest["backend"] == "source_contract_local"
    assert tokenizer_manifest["fixture_role"] == "shape_dtype_range_contract"
    assert tokenizer_manifest["dtype"] == "float16 output, float32 input"
    assert tokenizer_manifest["shape"]["latent_tokens"] == [1, 8192, 32]
    assert tokenizer_manifest["upstream_entry"].endswith("LightTokenizationTrainer.get_latents")
    assert tokenizer_manifest["upstream_sources"] == [
        "vendors/ml-lito/src/lito/trainers/lito_trainer.py",
        "vendors/ml-lito/src/lito/models/spoint_encoder.py",
    ]
    assert len(tokenizer_manifest["cases"]) >= 3


def test_manifest_records_no_cuda_no_vendor_runtime_contract():
    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["fixture_kind"] == "source_contract_local"
    assert manifest["no_cuda_contract"] == {
        "cuda_allowed": False,
        "torch_parity": "optional_cpu_mps_only",
        "vendor_runtime_imports": False,
    }


def test_gap_matrix_records_tokenizer_tolerances_and_no_cuda_contract():
    atol, rtol = _tokenizer_tolerances()
    assert atol == pytest.approx(2e-3)
    assert rtol == pytest.approx(2e-3)
    section = GAP_MATRIX.read_text(encoding="utf-8").split("## LITO-C", maxsplit=1)[1].split(
        "## LITO-D", maxsplit=1
    )[0]
    assert "Source-contract" in section
    assert "No-CUDA" in section
    assert "xformers" in section


@pytest.mark.parametrize("case_index", [0, 1, 2])
def test_tokenizer_matches_source_contract_fixture(case_index: int, caplog):
    inputs = _load_safetensors(FIXTURE_ROOT / f"tokenizer_input_{case_index}.safetensors")
    expected = _load_safetensors(FIXTURE_ROOT / f"tokenizer_output_{case_index}.safetensors")["latent_tokens"]
    tokenizer = LitoTokenizer.load(weights_root=None)

    actual, metrics = _record_metrics(
        caplog,
        f"fixture_{case_index}",
        lambda: tokenizer(
            mx.array(inputs["xyz_w"]),
            mx.array(inputs["rgb"]),
            mx.array(inputs["ray_origin_direction_w"]),
        ),
    )

    atol, rtol = _tokenizer_tolerances()
    assert actual.shape == expected.shape
    assert actual.dtype == mx.float16
    assert metrics["wall_time_s"] >= 0.0
    assert metrics["peak_active_memory_gb"] >= 0.0
    assert "lito_tokenizer_metrics" in caplog.text
    np.testing.assert_allclose(np.array(actual), expected, atol=atol, rtol=rtol)


def test_tokenizer_output_shape_and_dtype():
    inputs = _load_safetensors(FIXTURE_ROOT / "tokenizer_input_0.safetensors")
    result = LitoTokenizer()(
        mx.array(inputs["xyz_w"]),
        mx.array(inputs["rgb"]),
        mx.array(inputs["ray_origin_direction_w"]),
    )

    assert result.shape == (1, 8192, 32)
    assert result.dtype == mx.float16


def test_tokenizer_rejects_shape_mismatch():
    inputs = _load_safetensors(FIXTURE_ROOT / "tokenizer_input_0.safetensors")
    tokenizer = LitoTokenizer()

    with pytest.raises(ValueError, match="batch/point axes differ"):
        tokenizer(
            mx.array(inputs["xyz_w"]),
            mx.array(inputs["rgb"][:, :128, :]),
            mx.array(inputs["ray_origin_direction_w"]),
        )


def test_tokenizer_source_has_no_vendor_runtime_or_cuda_imports():
    tree = ast.parse(TOKENIZER_SOURCE.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])

    forbidden = {"torch", "xformers", "flash_attn", "gsplat", "plibs", "lito", "vendors", "cuda"}
    assert imported_roots.isdisjoint(forbidden)


def test_tokenizer_float32_uses_are_annotated():
    source = TOKENIZER_SOURCE.read_text(encoding="utf-8")
    float32_lines = [
        line.strip()
        for line in source.splitlines()
        if "mx.float32" in line
    ]

    assert float32_lines
    assert all("#" in line and "fixture" in line.lower() for line in float32_lines)
