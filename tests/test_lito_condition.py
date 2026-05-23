from __future__ import annotations

import ast
import json
import logging
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.numpy import load_file

from mlx_spatial.lito_condition import (
    LITO_CONDITION_ATOL,
    LITO_CONDITION_RTOL,
    LITO_CONDITION_SHAPE,
    LitoCondition,
)


FIXTURE_ROOT = Path("tests/fixtures/lito")
FORBIDDEN_IMPORT_ROOTS = {
    "cuda",
    "flash_attn",
    "gsplat",
    "torch",
    "torchvision",
    "vendors",
    "xformers",
}


def _condition_cases() -> list[dict[str, object]]:
    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    return list(manifest["condition"]["cases"])


def _load_case(index: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    case = _condition_cases()[index]
    inputs = load_file(FIXTURE_ROOT / str(case["input"]))
    expected = load_file(FIXTURE_ROOT / str(case["output"]))["cond_tokens"]
    return inputs, expected


def test_cond_output_shape() -> None:
    inputs, _ = _load_case(0)
    tokens = LitoCondition.load(weights_root="weights/lito")(
        inputs["straight_rgb"],
        inputs["alpha"],
    )

    assert tuple(tokens.shape) == LITO_CONDITION_SHAPE


def test_cond_uses_float16() -> None:
    inputs, _ = _load_case(0)
    tokens = LitoCondition.load()(inputs["straight_rgb"], inputs["alpha"])

    assert tokens.dtype == mx.float16


def test_cond_matches_source_contract_input_0(caplog) -> None:
    _assert_condition_matches_fixture(0, caplog)


def test_cond_matches_source_contract_input_1(caplog) -> None:
    _assert_condition_matches_fixture(1, caplog)


def test_cond_matches_source_contract_input_2(caplog) -> None:
    _assert_condition_matches_fixture(2, caplog)


def test_cond_batch_isolates_samples_when_other_sample_changes() -> None:
    inputs0, _ = _load_case(0)
    inputs1, _ = _load_case(1)
    conditioner = LitoCondition.load()
    baseline0 = np.array(conditioner(inputs0["straight_rgb"], inputs0["alpha"]))[0]

    batch_rgb = np.concatenate([inputs0["straight_rgb"], inputs1["straight_rgb"]], axis=0)
    batch_alpha = np.concatenate([inputs0["alpha"], inputs1["alpha"]], axis=0)
    original0 = np.array(conditioner(batch_rgb, batch_alpha))[0]

    changed_rgb = batch_rgb.copy()
    changed_alpha = batch_alpha.copy()
    changed_rgb[1] = 0.0
    changed_alpha[1] = 0.0
    changed0 = np.array(conditioner(changed_rgb, changed_alpha))[0]

    np.testing.assert_allclose(original0, baseline0, atol=LITO_CONDITION_ATOL, rtol=LITO_CONDITION_RTOL)
    np.testing.assert_allclose(changed0, baseline0, atol=LITO_CONDITION_ATOL, rtol=LITO_CONDITION_RTOL)


def test_cond_has_no_vendor_runtime_or_cuda_imports() -> None:
    source_path = Path("src/mlx_spatial/lito_condition.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots.isdisjoint(FORBIDDEN_IMPORT_ROOTS)


def _assert_condition_matches_fixture(index: int, caplog) -> None:
    inputs, expected = _load_case(index)
    tokens, metrics = _record_metrics(
        caplog,
        f"cond_input_{index}",
        lambda: LitoCondition.load()(inputs["straight_rgb"], inputs["alpha"]),
    )
    actual = np.array(tokens)

    assert actual.shape == expected.shape
    assert actual.dtype == np.float16
    assert metrics["wall_time_s"] >= 0.0
    assert metrics["peak_active_memory_gb"] >= 0.0
    np.testing.assert_allclose(
        actual,
        expected,
        atol=LITO_CONDITION_ATOL,
        rtol=LITO_CONDITION_RTOL,
    )


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
    logging.getLogger(__name__).info(
        "lito_condition_metrics case=%s wall_time_s=%.6f peak_active_memory_gb=%.6f",
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
