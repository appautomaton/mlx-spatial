"""SAM 3D Objects flow/shortcut sampler primitives."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


SAM3D_SLAT_STD = mx.array(
    [
        2.377650737762451,
        2.386378288269043,
        2.124418020248413,
        2.1748552322387695,
        2.663944721221924,
        2.371192216873169,
        2.6217446327209473,
        2.684523105621338,
    ],
    dtype=mx.float32,
)
SAM3D_SLAT_MEAN = mx.array(
    [
        -2.1687545776367188,
        -0.004347046371549368,
        -0.13352349400520325,
        -0.08418072760105133,
        -0.5271206498146057,
        0.7238689064979553,
        -1.1414450407028198,
        1.2039363384246826,
    ],
    dtype=mx.float32,
)


@dataclass(frozen=True)
class Sam3dFlowSchedule:
    t_seq: np.ndarray
    shortcut_d: float | None
    time_scale: float
    rescale_t: float


@dataclass(frozen=True)
class Sam3dShortcutParityReport:
    """Reference comparison for shortcut/fewer-step SAM3D outputs."""

    passed: bool
    tensor_count: int
    max_abs_error: float
    max_relative_error: float
    atol: float
    rtol: float
    tensor_errors: dict[str, dict[str, object]]


def sam3d_flow_time_sequence(
    steps: int,
    *,
    rescale_t: float = 3.0,
    reversed_timestamp: bool = False,
) -> np.ndarray:
    """Match official FlowMatching `_prepare_t` inference timestamps."""

    if steps <= 0:
        raise ValueError("SAM3D flow steps must be positive")
    t_seq = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)
    if rescale_t:
        t_seq = t_seq / (1.0 + (float(rescale_t) - 1.0) * (1.0 - t_seq))
    if reversed_timestamp:
        t_seq = 1.0 - t_seq
    return t_seq.astype(np.float32, copy=False)


def sam3d_shortcut_schedule(
    steps: int,
    *,
    rescale_t: float = 3.0,
    no_shortcut: bool = True,
    time_scale: float = 1000.0,
) -> Sam3dFlowSchedule:
    """Match official ShortCut `_prepare_t_and_d` for inference."""

    return Sam3dFlowSchedule(
        t_seq=sam3d_flow_time_sequence(steps, rescale_t=rescale_t),
        shortcut_d=0.0 if no_shortcut else 1.0 / float(steps),
        time_scale=float(time_scale),
        rescale_t=float(rescale_t),
    )


def compare_sam3d_shortcut_outputs(
    reference: Mapping[str, mx.array | np.ndarray],
    candidate: Mapping[str, mx.array | np.ndarray],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> Sam3dShortcutParityReport:
    """Compare fewer-step shortcut outputs against a reference output bundle."""

    reference_keys = set(reference)
    candidate_keys = set(candidate)
    if reference_keys != candidate_keys:
        missing = sorted(reference_keys - candidate_keys)
        extra = sorted(candidate_keys - reference_keys)
        raise ValueError(f"SAM3D shortcut parity keys differ: missing={missing}, extra={extra}")
    if not reference_keys:
        raise ValueError("SAM3D shortcut parity requires at least one tensor")

    tensor_errors: dict[str, dict[str, object]] = {}
    max_abs_error = 0.0
    max_relative_error = 0.0
    passed = True
    for key in sorted(reference_keys):
        ref = np.asarray(reference[key], dtype=np.float32)
        got = np.asarray(candidate[key], dtype=np.float32)
        if ref.shape != got.shape:
            raise ValueError(f"SAM3D shortcut parity tensor {key} shape differs: {ref.shape} != {got.shape}")
        abs_error = np.abs(got - ref)
        rel_error = abs_error / np.maximum(np.abs(ref), float(atol))
        tensor_max_abs = float(abs_error.max(initial=0.0))
        tensor_max_rel = float(rel_error.max(initial=0.0))
        tensor_passed = bool(np.all(abs_error <= float(atol) + float(rtol) * np.abs(ref)))
        tensor_errors[key] = {
            "shape": tuple(int(value) for value in ref.shape),
            "max_abs_error": tensor_max_abs,
            "max_relative_error": tensor_max_rel,
            "passed": tensor_passed,
        }
        max_abs_error = max(max_abs_error, tensor_max_abs)
        max_relative_error = max(max_relative_error, tensor_max_rel)
        passed = passed and tensor_passed

    return Sam3dShortcutParityReport(
        passed=passed,
        tensor_count=len(reference_keys),
        max_abs_error=float(max_abs_error),
        max_relative_error=float(max_relative_error),
        atol=float(atol),
        rtol=float(rtol),
        tensor_errors=tensor_errors,
    )


def sam3d_seeded_normal(shape: tuple[int, ...], *, seed: int, dtype=mx.float32) -> mx.array:
    """Deterministic NumPy-backed normal noise for SAM3D exact-mode tests and guards."""

    rng = np.random.default_rng(int(seed))
    return mx.array(rng.standard_normal(shape).astype(np.float32), dtype=dtype)


def sam3d_classifier_free_guidance(
    conditional: mx.array,
    unconditional: mx.array,
    *,
    strength: float,
    interval: tuple[float, float] | None,
    t_scaled: float,
) -> mx.array:
    """Apply official CFG inference blend for a single tensor."""

    active_strength = float(strength) if interval is not None and interval[0] <= t_scaled <= interval[1] else 0.0
    if active_strength == 0.0:
        return conditional
    return (1.0 + active_strength) * conditional - active_strength * unconditional


def sam3d_euler_solve(
    x0: mx.array,
    dynamics: Callable[[mx.array, float], mx.array],
    *,
    t_seq: np.ndarray,
) -> mx.array:
    """Run the official Euler ODE update `x = x + dt * v(x,t)`."""

    x = x0
    for index in range(len(t_seq) - 1):
        t0 = float(t_seq[index])
        t1 = float(t_seq[index + 1])
        x = x + (t1 - t0) * dynamics(x, t0)
        mx.eval(x)
    return x


def denormalize_sam3d_slat(features: mx.array) -> mx.array:
    """Apply official `slat * std + mean` channel denormalization."""

    return features * SAM3D_SLAT_STD.astype(features.dtype) + SAM3D_SLAT_MEAN.astype(features.dtype)
