import json
import logging
import time
from pathlib import Path

import mlx.core as mx
import pytest
from safetensors.mlx import load_file

from mlx_spatial import lito_dit
from mlx_spatial.lito_dit import (
    LITO_DEFAULT_MEMORY_PROFILE,
    LITO_MEMORY_PROFILES,
    LITO_RECOMMENDED_NUM_STEPS,
    LitoDiT,
    memory_profile_config,
)


FIXTURE_ROOT = Path("tests/fixtures/lito")
ATOL = 2e-3
RTOL = 2e-3
SOFT_MEMORY_LIMIT_GB = 90.0


def _fixture(name):
    return load_file(FIXTURE_ROOT / name)


def _dit_input():
    return _fixture("dit_input_0.safetensors")


def _assert_allclose(actual, expected):
    mx.eval(actual, expected)
    assert bool(mx.allclose(actual, expected, atol=ATOL, rtol=RTOL))


def _record_metrics(caplog, name, call):
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
        "lito_dit_metrics case=%s wall_time_s=%.6f peak_active_memory_gb=%.6f",
        name,
        elapsed,
        peak_gb,
    )
    return out, {"wall_time_s": elapsed, "peak_active_memory_gb": peak_gb}


def _reset_peak_memory_if_available():
    metal = getattr(mx, "metal", None)
    if metal is not None and hasattr(metal, "reset_peak_memory"):
        metal.reset_peak_memory()


def _peak_memory_gb_or_zero():
    metal = getattr(mx, "metal", None)
    if metal is None or not hasattr(metal, "get_peak_memory"):
        return 0.0
    try:
        return metal.get_peak_memory() / (1024**3)
    except RuntimeError:
        return 0.0


def _require_metal_memory_api():
    metal = getattr(mx, "metal", None)
    required = ("is_available", "reset_peak_memory", "get_peak_memory", "get_active_memory")
    if metal is None or any(not hasattr(metal, name) for name in required):
        pytest.skip("MLX metal memory APIs are unavailable")
    try:
        if not metal.is_available():
            pytest.skip("MLX metal device is unavailable")
        metal.reset_peak_memory()
        probe = mx.array([0.0], dtype=mx.float16)
        mx.eval(probe)
        if hasattr(mx, "synchronize"):
            mx.synchronize()
        metal.get_active_memory()
        metal.get_peak_memory()
    except RuntimeError as error:
        pytest.skip(f"MLX metal memory APIs are unavailable: {error}")


def test_dit_manifest_records_source_contract_no_cuda():
    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    dit_manifest = manifest["dit"]

    assert manifest["no_cuda_contract"]["cuda_allowed"] is False
    assert manifest["no_cuda_contract"]["vendor_runtime_imports"] is False
    assert dit_manifest["backend"] == "source_contract_local"
    assert dit_manifest["fixture_role"] == "microtrajectory_contract"
    assert "vendors/ml-lito/src/lito/mlx/models/dit.py" in dit_manifest["upstream_sources"]


def test_memory_profiles_match_lito_contract():
    assert LITO_MEMORY_PROFILES == ("safe", "balanced", "large")
    assert LITO_DEFAULT_MEMORY_PROFILE == "balanced"
    assert memory_profile_config("balanced").name == "balanced"

    with pytest.raises(ValueError):
        memory_profile_config("default")


def test_dit_step_0_matches_source_contract(caplog):
    tensors = _dit_input()
    expected = _fixture("dit_step_0_0.safetensors")["latent"]
    dit = LitoDiT()

    actual, metrics = _record_metrics(
        caplog,
        "step_0",
        lambda: dit(tensors["latent"], tensors["cond_tokens"], mx.array(0.0, dtype=mx.float32)),
    )

    assert actual.dtype == mx.float16
    assert metrics["wall_time_s"] >= 0.0
    _assert_allclose(actual, expected)


def test_dit_trajectory_conditioning_is_batch_isolated():
    tensors = _dit_input()
    dit = LitoDiT()
    latent = mx.concatenate([tensors["latent"], tensors["latent"]], axis=0)
    cond = mx.concatenate([tensors["cond_tokens"], tensors["cond_tokens"]], axis=0)
    changed_cond = mx.concatenate(
        [tensors["cond_tokens"], tensors["cond_tokens"] + mx.array(10.0, dtype=mx.float16)],
        axis=0,
    )

    baseline = dit(latent, cond, mx.array(0.0, dtype=mx.float32))
    changed = dit(latent, changed_cond, mx.array(0.0, dtype=mx.float32))
    mx.eval(baseline, changed)

    _assert_allclose(changed[0:1], baseline[0:1])
    assert not bool(mx.allclose(changed[1:2], baseline[1:2], atol=ATOL, rtol=RTOL))


def test_dit_timestep_selection_is_batch_isolated():
    tensors = _dit_input()
    dit = LitoDiT()
    latent = mx.concatenate([tensors["latent"], tensors["latent"]], axis=0)
    cond = mx.concatenate([tensors["cond_tokens"], tensors["cond_tokens"]], axis=0)

    baseline = dit(latent, cond, mx.array([0.0, 0.0], dtype=mx.float32))
    changed = dit(latent, cond, mx.array([0.0, 1.0], dtype=mx.float32))
    mx.eval(baseline, changed)

    _assert_allclose(changed[0:1], baseline[0:1])
    assert not bool(mx.allclose(changed[1:2], baseline[1:2], atol=ATOL, rtol=RTOL))


def test_dit_forward_rejects_mismatched_latent_cond_batches():
    tensors = _dit_input()
    dit = LitoDiT()
    latent = mx.concatenate([tensors["latent"], tensors["latent"]], axis=0)

    with pytest.raises(ValueError, match="batch sizes must match"):
        dit(latent, tensors["cond_tokens"], mx.array(0.0, dtype=mx.float32))


def test_dit_sample_rejects_mismatched_initial_latent_cond_batches():
    tensors = _dit_input()
    dit = LitoDiT()
    latent = mx.concatenate([tensors["latent"], tensors["latent"]], axis=0)

    with pytest.raises(ValueError, match="batch sizes must match"):
        dit.sample(tensors["cond_tokens"], initial_latent=latent)


def test_dit_step_mid_matches_source_contract(caplog):
    tensors = _dit_input()
    expected = _fixture("dit_step_mid_0.safetensors")["latent"]
    dit = LitoDiT()

    actual, metrics = _record_metrics(
        caplog,
        "step_mid",
        lambda: dit(tensors["latent"], tensors["cond_tokens"], mx.array(0.5, dtype=mx.float32)),
    )

    assert actual.dtype == mx.float16
    assert metrics["peak_active_memory_gb"] >= 0.0
    _assert_allclose(actual, expected)


def test_dit_step_final_matches_source_contract(caplog):
    tensors = _dit_input()
    expected = _fixture("dit_step_final_0.safetensors")["latent"]
    dit = LitoDiT()

    actual, metrics = _record_metrics(
        caplog,
        "step_final",
        lambda: dit(tensors["latent"], tensors["cond_tokens"], mx.array(1.0, dtype=mx.float32)),
    )

    assert actual.dtype == mx.float16
    assert metrics["wall_time_s"] >= 0.0
    _assert_allclose(actual, expected)


def test_dit_full_trajectory_matches_source_contract(caplog):
    tensors = _dit_input()
    expected = _fixture("dit_step_final_0.safetensors")["latent"]
    dit = LitoDiT()

    result, metrics = _record_metrics(
        caplog,
        "trajectory",
        lambda: dit.sample(
            tensors["cond_tokens"],
            tensors["num_steps"],
            tensors["seed"],
            initial_latent=tensors["latent"],
            return_trajectory=True,
        ),
    )
    actual, trajectory = result

    assert actual.dtype == mx.float16
    assert metrics["wall_time_s"] >= 0.0
    assert len(trajectory) == 3
    _assert_allclose(actual, expected)


@pytest.mark.heavy
@pytest.mark.parametrize("profile", LITO_MEMORY_PROFILES)
def test_dit_memory_profiles_stay_under_90gb(profile):
    _require_metal_memory_api()
    tensors = _dit_input()
    dit = LitoDiT(memory_profile=profile)

    mx.metal.reset_peak_memory()
    out = dit.sample(
        tensors["cond_tokens"],
        num_steps=LITO_RECOMMENDED_NUM_STEPS,
        seed=42,
        initial_latent=tensors["latent"],
        memory_profile=profile,
    )
    mx.eval(out)
    if hasattr(mx, "synchronize"):
        mx.synchronize()
    peak_gb = mx.metal.get_peak_memory() / (1024**3)

    assert out.dtype == mx.float16
    assert peak_gb < SOFT_MEMORY_LIMIT_GB, f"{profile} profile peaked at {peak_gb:.1f} GB"


@pytest.mark.heavy
def test_dit_memory_safe_stays_well_under_threshold():
    _require_metal_memory_api()
    tensors = _dit_input()
    dit = LitoDiT(memory_profile="safe")

    mx.metal.reset_peak_memory()
    out = dit.sample(tensors["cond_tokens"], initial_latent=tensors["latent"], memory_profile="safe")
    mx.eval(out)
    if hasattr(mx, "synchronize"):
        mx.synchronize()
    peak_gb = mx.metal.get_peak_memory() / (1024**3)

    assert peak_gb < 1.0


def test_dit_module_does_not_depend_on_slice5_memory_exception():
    assert not hasattr(lito_dit, "LitoMemoryLimitExceeded")
