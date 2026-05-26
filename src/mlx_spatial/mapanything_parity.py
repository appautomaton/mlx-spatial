"""Dev-only MapAnything PyTorch reference bundle and MLX parity helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import mlx.core as mx
import numpy as np


MAPANYTHING_TORCH_PARITY_ENV = "MAPANYTHING_TORCH_REF"
MAPANYTHING_PARITY_BUNDLE_VERSION = 1
MAPANYTHING_PARITY_REFERENCE_SOURCE = "vendored MapAnything PyTorch"
MAPANYTHING_PARITY_DEFAULT_RTOL = 1e-4
MAPANYTHING_PARITY_DEFAULT_ATOL = 1e-4


@dataclass(frozen=True)
class MapAnythingParityReference:
    """Loaded MapAnything reference tensor bundle."""

    path: Path
    tensors: dict[str, np.ndarray]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MapAnythingParityTensorComparison:
    """One tensor comparison result."""

    name: str
    status: str
    expected_shape: tuple[int, ...] | None
    actual_shape: tuple[int, ...] | None
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    max_rel_error: float | None = None
    atol: float = MAPANYTHING_PARITY_DEFAULT_ATOL
    rtol: float = MAPANYTHING_PARITY_DEFAULT_RTOL
    reason: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "pass"


@dataclass(frozen=True)
class MapAnythingParityReport:
    """Aggregate parity comparison summary."""

    comparisons: tuple[MapAnythingParityTensorComparison, ...]
    reference_path: Path | None = None

    @property
    def passed(self) -> bool:
        return all(comparison.passed for comparison in self.comparisons)

    @property
    def failed_names(self) -> tuple[str, ...]:
        return tuple(comparison.name for comparison in self.comparisons if not comparison.passed)


def mapanything_parity_trace_metadata(
    *,
    reference_path: str | Path | None = None,
    report: MapAnythingParityReport | None = None,
) -> dict[str, object]:
    """Return trace metadata that keeps MapAnything parity claims explicit."""

    resolved_reference_path = (
        str(reference_path or report.reference_path)
        if reference_path is not None or (report is not None and report.reference_path is not None)
        else None
    )
    return {
        "official_reference": MAPANYTHING_PARITY_REFERENCE_SOURCE,
        "runtime_depends_on_torch": False,
        "dev_reference_env": MAPANYTHING_TORCH_PARITY_ENV,
        "numeric_parity_verified": bool(report.passed) if report is not None else False,
        "reference_path": resolved_reference_path,
        "checked_tensors": len(report.comparisons) if report is not None else 0,
        "failed_tensors": list(report.failed_names) if report is not None else [],
        "status": "verified" if report is not None and report.passed else "unverified",
    }


def require_mapanything_torch_parity_enabled() -> None:
    """Raise unless the user explicitly enabled dev-only Torch reference work."""

    if os.environ.get(MAPANYTHING_TORCH_PARITY_ENV) != "1":
        raise RuntimeError(
            f"MapAnything PyTorch reference dumping is dev-only. Set {MAPANYTHING_TORCH_PARITY_ENV}=1 "
            "to run it. The shipped MLX runtime remains Torch/CUDA-free."
        )


def write_mapanything_parity_bundle(
    path: str | Path,
    tensors: Mapping[str, object],
    *,
    metadata: Mapping[str, object] | None = None,
    compressed: bool = True,
) -> Path:
    """Write a portable NumPy reference bundle for later MLX parity checks."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        _validate_tensor_name(name): _to_numpy(value).astype(np.float32, copy=False)
        for name, value in tensors.items()
    }
    payload_metadata = {
        "bundle_version": MAPANYTHING_PARITY_BUNDLE_VERSION,
        "source": MAPANYTHING_PARITY_REFERENCE_SOURCE,
        "tensor_names": sorted(arrays),
        **dict(metadata or {}),
    }
    arrays["__metadata_json__"] = np.array(json.dumps(payload_metadata, sort_keys=True))
    writer = np.savez_compressed if compressed else np.savez
    writer(output_path, **arrays)
    return output_path


def load_mapanything_parity_bundle(path: str | Path) -> MapAnythingParityReference:
    """Load a reference bundle written by :func:`write_mapanything_parity_bundle`."""

    bundle_path = Path(path)
    with np.load(bundle_path, allow_pickle=False) as data:
        metadata_raw = str(data["__metadata_json__"]) if "__metadata_json__" in data else "{}"
        metadata = json.loads(metadata_raw)
        tensors = {
            name: np.asarray(data[name])
            for name in data.files
            if name != "__metadata_json__"
        }
    return MapAnythingParityReference(path=bundle_path, tensors=tensors, metadata=metadata)


def compare_mapanything_parity_tensors(
    actual_tensors: Mapping[str, object],
    reference: MapAnythingParityReference | Mapping[str, object],
    *,
    names: Sequence[str] | None = None,
    atol: float = MAPANYTHING_PARITY_DEFAULT_ATOL,
    rtol: float = MAPANYTHING_PARITY_DEFAULT_RTOL,
) -> MapAnythingParityReport:
    """Compare MLX/NumPy tensors against a MapAnything PyTorch reference bundle."""

    expected_tensors = reference.tensors if isinstance(reference, MapAnythingParityReference) else reference
    reference_path = reference.path if isinstance(reference, MapAnythingParityReference) else None
    target_names = tuple(names) if names is not None else tuple(sorted(expected_tensors))
    comparisons = []
    for name in target_names:
        if name not in expected_tensors:
            comparisons.append(
                MapAnythingParityTensorComparison(
                    name=name,
                    status="missing-reference",
                    expected_shape=None,
                    actual_shape=None,
                    reason="reference bundle does not contain this tensor",
                    atol=atol,
                    rtol=rtol,
                )
            )
            continue
        expected = _to_numpy(expected_tensors[name]).astype(np.float32, copy=False)
        if name not in actual_tensors:
            comparisons.append(
                MapAnythingParityTensorComparison(
                    name=name,
                    status="missing-actual",
                    expected_shape=tuple(int(dim) for dim in expected.shape),
                    actual_shape=None,
                    reason="MLX output did not provide this tensor",
                    atol=atol,
                    rtol=rtol,
                )
            )
            continue
        actual = _to_numpy(actual_tensors[name]).astype(np.float32, copy=False)
        if actual.shape != expected.shape:
            comparisons.append(
                MapAnythingParityTensorComparison(
                    name=name,
                    status="shape-mismatch",
                    expected_shape=tuple(int(dim) for dim in expected.shape),
                    actual_shape=tuple(int(dim) for dim in actual.shape),
                    reason="tensor shapes differ",
                    atol=atol,
                    rtol=rtol,
                )
            )
            continue
        diff = np.abs(actual - expected)
        rel = diff / np.maximum(np.abs(expected), atol)
        max_abs = float(np.max(diff)) if diff.size else 0.0
        mean_abs = float(np.mean(diff)) if diff.size else 0.0
        max_rel = float(np.max(rel)) if rel.size else 0.0
        passed = bool(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True))
        comparisons.append(
            MapAnythingParityTensorComparison(
                name=name,
                status="pass" if passed else "value-mismatch",
                expected_shape=tuple(int(dim) for dim in expected.shape),
                actual_shape=tuple(int(dim) for dim in actual.shape),
                max_abs_error=max_abs,
                mean_abs_error=mean_abs,
                max_rel_error=max_rel,
                reason=None if passed else "tensor values exceed configured tolerance",
                atol=atol,
                rtol=rtol,
            )
        )
    return MapAnythingParityReport(comparisons=tuple(comparisons), reference_path=reference_path)


def mapanything_parity_report_to_dict(report: MapAnythingParityReport) -> dict[str, object]:
    """Return a JSON-safe parity report payload."""

    return {
        "passed": report.passed,
        "reference_path": str(report.reference_path) if report.reference_path is not None else None,
        "failed_names": list(report.failed_names),
        "parity_trace_metadata": mapanything_parity_trace_metadata(report=report),
        "comparisons": [
            {
                "name": comparison.name,
                "status": comparison.status,
                "expected_shape": list(comparison.expected_shape)
                if comparison.expected_shape is not None
                else None,
                "actual_shape": list(comparison.actual_shape)
                if comparison.actual_shape is not None
                else None,
                "max_abs_error": comparison.max_abs_error,
                "mean_abs_error": comparison.mean_abs_error,
                "max_rel_error": comparison.max_rel_error,
                "atol": comparison.atol,
                "rtol": comparison.rtol,
                "reason": comparison.reason,
            }
            for comparison in report.comparisons
        ],
    }


def _validate_tensor_name(name: str) -> str:
    tensor_name = str(name)
    if not tensor_name or tensor_name == "__metadata_json__":
        raise ValueError(f"invalid MapAnything parity tensor name: {name!r}")
    return tensor_name


def _to_numpy(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, mx.array):
        mx.eval(value)
        return np.asarray(value)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)
