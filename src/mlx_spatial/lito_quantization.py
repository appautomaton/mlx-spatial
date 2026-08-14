"""Selective MLX affine quantization for checkpoint-backed LiTo inference."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import mlx.core as mx

from .lito_assets import (
    LITO_TRELLIS_BUNDLE_PATH,
    LITO_TRELLIS_METADATA_FILES,
    LITO_TRELLIS_REQUIRED_FILES,
)
from .safetensors_io import (
    SafetensorHeader,
    inspect_safetensors,
    load_mlx_safetensors,
    read_safetensors_metadata,
    save_safetensors,
)


LITO_QUANTIZATION_FORMAT = "mlx-spatial-lito-runtime-affine-v1"
LITO_QUANTIZATION_METADATA_KEY = "mlx_spatial.lito.quantization"
LITO_RUNTIME_FORMAT = "mlx-spatial-lito-runtime-v1"
LITO_RUNTIME_METADATA_KEY = "mlx_spatial.lito.runtime"
LITO_QUANTIZED_CHECKPOINTS = (
    Path("image_to_3d/lito_dit_rgba.safetensors"),
    Path("tokenizer/lito_new.safetensors"),
)

_QWEIGHT_SUFFIX = ".__mlx_qweight__"
_SCALES_SUFFIX = ".__mlx_scales__"
_BIASES_SUFFIX = ".__mlx_biases__"
_QUANTIZED_SUFFIXES = (_QWEIGHT_SUFFIX, _SCALES_SUFFIX, _BIASES_SUFFIX)
_FLOAT_DTYPES = frozenset(("F16", "BF16", "F32"))
_AFFINE_GROUP_SIZES = frozenset((32, 64, 128))
_RUNTIME_PREFIXES = (
    "velocity_estimator_ema.module.",
    "patch_encoder.",
    "gs_decoder.",
    "voxel_decoder.",
)

_DIT_INTERNAL_WEIGHT = re.compile(r"^velocity_estimator_ema\.module\.blocks\.\d+\..*\.weight$")
_GAUSSIAN_ATTENTION_WEIGHT = re.compile(
    r"^gs_decoder\.perceiver\.blocks\.\d+\.(?:ca_layer|sa_layers\.\d+)\.linear_(?:q|kv|qkv|out)\.weight$"
)
_GAUSSIAN_MLP_WEIGHT = re.compile(
    r"^gs_decoder\.perceiver\.blocks\.\d+\.(?:ca_mlp|mlp_layers\.\d+)\.(?:w12|w3)\.weight$"
)
_VOXEL_ATTENTION_WEIGHT = re.compile(
    r"^voxel_decoder\.net\.encoder\.blocks\.\d+\.(?:ca_layer|sa_layers\.\d+)\.linear_(?:q|kv|qkv|out)\.weight$"
)
_VOXEL_MLP_WEIGHT = re.compile(
    r"^voxel_decoder\.net\.encoder\.blocks\.\d+\.(?:ca_mlp|mlp_layers\.\d+)\.fc[12]\.weight$"
)


@dataclass(frozen=True)
class LitoQuantizedTensorInfo:
    """Logical shape and dtype of one packed checkpoint matrix."""

    shape: tuple[int, int]
    dtype: str


@dataclass(frozen=True)
class LitoQuantizationSpec:
    """Validated per-checkpoint affine quantization metadata."""

    bits: int
    group_size: int
    mode: str
    tensors: Mapping[str, LitoQuantizedTensorInfo]
    policy: str


@dataclass(frozen=True)
class LitoQuantizedMatrix:
    """Packed MLX affine matrix consumed directly by ``mx.quantized_matmul``."""

    qweight: mx.array
    scales: mx.array
    biases: mx.array
    original_shape: tuple[int, int]
    original_dtype: str
    bits: int
    group_size: int
    mode: str = "affine"

    @property
    def shape(self) -> tuple[int, int]:
        return self.original_shape

    def slice_rows(self, start: int, stop: int) -> LitoQuantizedMatrix:
        """Split independent output rows without dequantizing the matrix."""

        if start < 0 or stop < start or stop > self.original_shape[0]:
            raise ValueError(f"invalid quantized row slice [{start}:{stop}] for shape {self.original_shape}")
        return LitoQuantizedMatrix(
            qweight=self.qweight[start:stop],
            scales=self.scales[start:stop],
            biases=self.biases[start:stop],
            original_shape=(stop - start, self.original_shape[1]),
            original_dtype=self.original_dtype,
            bits=self.bits,
            group_size=self.group_size,
            mode=self.mode,
        )


@dataclass(frozen=True)
class LitoCheckpointQuantizationResult:
    """Summary for one quantized LiTo checkpoint."""

    source: Path
    output: Path
    source_tensor_count: int
    runtime_tensor_count: int
    removed_tensor_count: int
    quantized_tensor_count: int
    source_payload_bytes: int
    runtime_source_bytes: int
    removed_source_bytes: int
    quantized_source_bytes: int


@dataclass(frozen=True)
class LitoCheckpointPruningResult:
    """Summary for one runtime-only full-precision LiTo checkpoint."""

    source: Path
    output: Path
    source_tensor_count: int
    runtime_tensor_count: int
    removed_tensor_count: int
    source_payload_bytes: int
    runtime_payload_bytes: int
    removed_payload_bytes: int


def is_lito_runtime_tensor(name: str) -> bool:
    """Return whether the checkpoint tensor is read by the main LiTo inference path."""

    return name.startswith(_RUNTIME_PREFIXES)


def should_quantize_lito_tensor(
    name: str,
    shape: Sequence[int],
    dtype: str,
    *,
    group_size: int = 64,
) -> bool:
    """Select internal transformer matrices while preserving sensitive boundaries."""

    if group_size not in _AFFINE_GROUP_SIZES:
        raise ValueError(f"affine group_size must be one of {sorted(_AFFINE_GROUP_SIZES)}, got {group_size}")
    if (
        not is_lito_runtime_tensor(name)
        or dtype not in _FLOAT_DTYPES
        or len(shape) != 2
        or int(shape[1]) % group_size
    ):
        return False
    return any(
        pattern.fullmatch(name) is not None
        for pattern in (
            _DIT_INTERNAL_WEIGHT,
            _GAUSSIAN_ATTENTION_WEIGHT,
            _GAUSSIAN_MLP_WEIGHT,
            _VOXEL_ATTENTION_WEIGHT,
            _VOXEL_MLP_WEIGHT,
        )
    )


def read_lito_quantization_spec(path: str | Path) -> LitoQuantizationSpec | None:
    """Read and validate LiTo quantization metadata, if present."""

    metadata = read_safetensors_metadata(path)
    raw = metadata.get(LITO_QUANTIZATION_METADATA_KEY)
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid LiTo quantization metadata in {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("format") != LITO_QUANTIZATION_FORMAT:
        raise ValueError(f"unsupported LiTo quantization format in {path}")

    bits = payload.get("bits")
    group_size = payload.get("group_size")
    mode = payload.get("mode")
    policy = payload.get("policy")
    raw_tensors = payload.get("tensors")
    if not isinstance(bits, int) or not 2 <= bits <= 8:
        raise ValueError(f"invalid affine bit width in {path}: {bits!r}")
    if group_size not in _AFFINE_GROUP_SIZES:
        raise ValueError(f"invalid affine group size in {path}: {group_size!r}")
    if mode != "affine":
        raise ValueError(f"unsupported LiTo quantization mode in {path}: {mode!r}")
    if not isinstance(policy, str) or not policy:
        raise ValueError(f"missing LiTo quantization policy in {path}")
    if not isinstance(raw_tensors, dict) or not raw_tensors:
        raise ValueError(f"missing quantized tensor inventory in {path}")

    tensors: dict[str, LitoQuantizedTensorInfo] = {}
    for name, value in raw_tensors.items():
        if not isinstance(name, str) or not name or not isinstance(value, dict):
            raise ValueError(f"invalid quantized tensor entry in {path}")
        shape = value.get("shape")
        dtype = value.get("dtype")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(not isinstance(dim, int) or dim <= 0 for dim in shape)
            or shape[1] % group_size
            or dtype not in _FLOAT_DTYPES
        ):
            raise ValueError(f"invalid logical quantized tensor descriptor for {name!r} in {path}")
        tensors[name] = LitoQuantizedTensorInfo(shape=(shape[0], shape[1]), dtype=dtype)
    return LitoQuantizationSpec(
        bits=bits,
        group_size=group_size,
        mode=mode,
        tensors=tensors,
        policy=policy,
    )


def inspect_logical_lito_safetensors(path: str | Path) -> tuple[SafetensorHeader, ...]:
    """Inspect original logical tensors in either full-precision or packed LiTo files."""

    physical = {header.name: header for header in inspect_safetensors(path)}
    spec = read_lito_quantization_spec(path)
    if spec is None:
        return tuple(sorted(physical.values(), key=lambda header: header.name))

    logical = dict(physical)
    for name, info in spec.tensors.items():
        if name in logical:
            raise ValueError(f"quantized LiTo checkpoint contains both packed and full-precision tensor {name!r}")
        _validate_physical_quantized_headers(path, name, info, spec, logical)
        for suffix in _QUANTIZED_SUFFIXES:
            logical.pop(name + suffix)
        logical[name] = SafetensorHeader(name=name, shape=info.shape, dtype=info.dtype)
    _raise_orphaned_quantized_tensors(path, logical)
    return tuple(sorted(logical.values(), key=lambda header: header.name))


def load_logical_lito_safetensors(path: str | Path) -> dict[str, mx.array | LitoQuantizedMatrix]:
    """Load full-precision tensors and reconstruct packed matrices as logical weights."""

    physical = load_mlx_safetensors(path)
    spec = read_lito_quantization_spec(path)
    if spec is None:
        return physical

    logical: dict[str, mx.array | LitoQuantizedMatrix] = dict(physical)
    for name, info in spec.tensors.items():
        if name in logical:
            raise ValueError(f"quantized LiTo checkpoint contains both packed and full-precision tensor {name!r}")
        try:
            qweight = logical.pop(name + _QWEIGHT_SUFFIX)
            scales = logical.pop(name + _SCALES_SUFFIX)
            biases = logical.pop(name + _BIASES_SUFFIX)
        except KeyError as error:
            raise ValueError(f"quantized LiTo checkpoint is missing packed tensor {error.args[0]!r}") from error
        _validate_loaded_quantized_tensors(path, name, info, spec, qweight, scales, biases)
        logical[name] = LitoQuantizedMatrix(
            qweight=qweight,
            scales=scales,
            biases=biases,
            original_shape=info.shape,
            original_dtype=info.dtype,
            bits=spec.bits,
            group_size=spec.group_size,
            mode=spec.mode,
        )
    _raise_orphaned_quantized_tensors(path, logical)
    return logical


def quantize_lito_checkpoint(
    source: str | Path,
    output: str | Path,
    *,
    bits: int = 8,
    group_size: int = 64,
    overwrite: bool = False,
) -> LitoCheckpointQuantizationResult:
    """Quantize selected matrices in one LiTo safetensors checkpoint."""

    source_path = Path(source)
    output_path = Path(output)
    _validate_quantization_options(bits, group_size)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("LiTo quantization output must differ from the full-precision source")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"LiTo quantization output already exists: {output_path}")
    if read_lito_quantization_spec(source_path) is not None:
        raise ValueError(f"LiTo source checkpoint is already quantized: {source_path}")

    headers = inspect_safetensors(source_path)
    runtime_headers = [header for header in headers if is_lito_runtime_tensor(header.name)]
    selected = [
        header
        for header in runtime_headers
        if should_quantize_lito_tensor(header.name, header.shape, header.dtype, group_size=group_size)
    ]
    if not runtime_headers:
        raise ValueError(f"LiTo checkpoint contains no tensors used by the inference runtime: {source_path}")
    if not selected:
        raise ValueError(
            f"LiTo checkpoint contains no tensors selected by the inference quantization policy: {source_path}"
        )

    loaded = load_mlx_safetensors(source_path)
    output_tensors: dict[str, mx.array] = {header.name: loaded[header.name] for header in runtime_headers}
    del loaded
    logical_tensors: dict[str, dict[str, Any]] = {}
    for header in selected:
        source_weight = output_tensors.pop(header.name).astype(mx.float32)
        qweight, scales, biases = mx.quantize(
            source_weight,
            group_size=group_size,
            bits=bits,
            mode="affine",
        )
        mx.eval(qweight, scales, biases)
        output_tensors[header.name + _QWEIGHT_SUFFIX] = qweight
        output_tensors[header.name + _SCALES_SUFFIX] = scales
        output_tensors[header.name + _BIASES_SUFFIX] = biases
        logical_tensors[header.name] = {"shape": list(header.shape), "dtype": header.dtype}
        mx.clear_cache()

    metadata = read_safetensors_metadata(source_path)
    metadata[LITO_QUANTIZATION_METADATA_KEY] = json.dumps(
        {
            "format": LITO_QUANTIZATION_FORMAT,
            "bits": bits,
            "group_size": group_size,
            "mode": "affine",
            "policy": "runtime-only-selective-internal-linear-v1",
            "tensors": logical_tensors,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors(output_path, output_tensors, metadata=metadata)
    mx.clear_cache()

    source_payload_bytes = sum(_tensor_nbytes(header) for header in headers)
    runtime_source_bytes = sum(_tensor_nbytes(header) for header in runtime_headers)
    quantized_source_bytes = sum(_tensor_nbytes(header) for header in selected)
    return LitoCheckpointQuantizationResult(
        source=source_path,
        output=output_path,
        source_tensor_count=len(headers),
        runtime_tensor_count=len(runtime_headers),
        removed_tensor_count=len(headers) - len(runtime_headers),
        quantized_tensor_count=len(selected),
        source_payload_bytes=source_payload_bytes,
        runtime_source_bytes=runtime_source_bytes,
        removed_source_bytes=source_payload_bytes - runtime_source_bytes,
        quantized_source_bytes=quantized_source_bytes,
    )


def quantize_lito_weights(
    source_root: str | Path,
    output_root: str | Path,
    *,
    bits: int = 8,
    group_size: int = 64,
    overwrite: bool = False,
) -> tuple[LitoCheckpointQuantizationResult, ...]:
    """Create a selectively quantized LiTo checkpoint root."""

    source = Path(source_root)
    output = Path(output_root)
    _validate_quantization_options(bits, group_size)
    if source.resolve() == output.resolve():
        raise ValueError("LiTo quantization output root must differ from the full-precision source root")
    dependency_pairs = _lito_bundle_dependency_pairs(source, output, overwrite=overwrite)
    for relative_path in LITO_QUANTIZED_CHECKPOINTS:
        source_path = source / relative_path
        output_path = output / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(f"LiTo source checkpoint not found: {source_path}")
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"LiTo quantization output already exists: {output_path}")
    results = tuple(
        quantize_lito_checkpoint(
            source / relative_path,
            output / relative_path,
            bits=bits,
            group_size=group_size,
            overwrite=overwrite,
        )
        for relative_path in LITO_QUANTIZED_CHECKPOINTS
    )
    _copy_lito_bundle_dependencies(dependency_pairs)
    return results


def prune_lito_checkpoint(
    source: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> LitoCheckpointPruningResult:
    """Write only tensors consumed by the main LiTo inference runtime."""

    source_path = Path(source)
    output_path = Path(output)
    if not source_path.is_file():
        raise FileNotFoundError(f"LiTo source checkpoint not found: {source_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"LiTo runtime checkpoint output already exists: {output_path}")
    if read_lito_quantization_spec(source_path) is not None:
        raise ValueError(f"full-precision LiTo pruning does not accept a quantized source: {source_path}")

    headers = inspect_safetensors(source_path)
    runtime_headers = [header for header in headers if is_lito_runtime_tensor(header.name)]
    if not runtime_headers:
        raise ValueError(f"LiTo checkpoint contains no tensors used by the inference runtime: {source_path}")

    loaded = load_mlx_safetensors(source_path)
    output_tensors = {header.name: loaded[header.name] for header in runtime_headers}
    del loaded
    source_payload_bytes = sum(_tensor_nbytes(header) for header in headers)
    runtime_payload_bytes = sum(_tensor_nbytes(header) for header in runtime_headers)
    metadata = read_safetensors_metadata(source_path)
    metadata[LITO_RUNTIME_METADATA_KEY] = json.dumps(
        {
            "format": LITO_RUNTIME_FORMAT,
            "policy": "main-inference-prefixes-v1",
            "source_tensors": len(headers),
            "runtime_tensors": len(runtime_headers),
            "removed_tensors": len(headers) - len(runtime_headers),
            "source_payload_bytes": source_payload_bytes,
            "runtime_payload_bytes": runtime_payload_bytes,
            "removed_payload_bytes": source_payload_bytes - runtime_payload_bytes,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors(output_path, output_tensors, metadata=metadata)
    mx.clear_cache()
    return LitoCheckpointPruningResult(
        source=source_path,
        output=output_path,
        source_tensor_count=len(headers),
        runtime_tensor_count=len(runtime_headers),
        removed_tensor_count=len(headers) - len(runtime_headers),
        source_payload_bytes=source_payload_bytes,
        runtime_payload_bytes=runtime_payload_bytes,
        removed_payload_bytes=source_payload_bytes - runtime_payload_bytes,
    )


def prune_lito_weights(
    source_root: str | Path,
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[LitoCheckpointPruningResult, ...]:
    """Create a runtime-only full-precision LiTo checkpoint root."""

    source = Path(source_root)
    output = Path(output_root)
    dependency_pairs = _lito_bundle_dependency_pairs(source, output, overwrite=overwrite)
    for relative_path in LITO_QUANTIZED_CHECKPOINTS:
        source_path = source / relative_path
        output_path = output / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(f"LiTo source checkpoint not found: {source_path}")
        if read_lito_quantization_spec(source_path) is not None:
            raise ValueError(f"full-precision LiTo pruning does not accept a quantized source: {source_path}")
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"LiTo runtime checkpoint output already exists: {output_path}")
    results = tuple(
        prune_lito_checkpoint(
            source / relative_path,
            output / relative_path,
            overwrite=overwrite,
        )
        for relative_path in LITO_QUANTIZED_CHECKPOINTS
    )
    _copy_lito_bundle_dependencies(dependency_pairs)
    return results


def _lito_bundle_dependency_pairs(
    source: Path,
    output: Path,
    *,
    overwrite: bool,
) -> tuple[tuple[Path, Path], ...]:
    relative_paths = tuple(
        LITO_TRELLIS_BUNDLE_PATH / relative_path
        for relative_path in (*LITO_TRELLIS_REQUIRED_FILES, *LITO_TRELLIS_METADATA_FILES)
    )
    missing = [source / relative_path for relative_path in relative_paths if not (source / relative_path).is_file()]
    if missing:
        raise FileNotFoundError(f"LiTo source bundle is missing embedded dependency: {missing[0]}")
    if source.resolve() == output.resolve():
        return ()

    pairs = tuple((source / relative_path, output / relative_path) for relative_path in relative_paths)
    if not overwrite:
        existing = [destination for _, destination in pairs if destination.exists()]
        if existing:
            raise FileExistsError(f"LiTo bundle dependency output already exists: {existing[0]}")
    return pairs


def _copy_lito_bundle_dependencies(pairs: tuple[tuple[Path, Path], ...]) -> None:
    for source, destination in pairs:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _validate_quantization_options(bits: int, group_size: int) -> None:
    if not isinstance(bits, int) or not 2 <= bits <= 8:
        raise ValueError(f"affine bits must be an integer from 2 through 8, got {bits!r}")
    if group_size not in _AFFINE_GROUP_SIZES:
        raise ValueError(f"affine group_size must be one of {sorted(_AFFINE_GROUP_SIZES)}, got {group_size}")


def _validate_physical_quantized_headers(
    path: str | Path,
    name: str,
    info: LitoQuantizedTensorInfo,
    spec: LitoQuantizationSpec,
    headers: Mapping[str, SafetensorHeader],
) -> None:
    expected = _expected_quantized_shapes(info.shape, spec.bits, spec.group_size)
    for suffix, expected_shape, expected_dtype in (
        (_QWEIGHT_SUFFIX, expected[0], "U32"),
        (_SCALES_SUFFIX, expected[1], "F32"),
        (_BIASES_SUFFIX, expected[1], "F32"),
    ):
        header = headers.get(name + suffix)
        if header is None or header.shape != expected_shape or header.dtype != expected_dtype:
            raise ValueError(
                f"invalid packed tensor {name + suffix!r} in {path}; "
                f"expected {expected_dtype} {expected_shape}"
            )


def _validate_loaded_quantized_tensors(
    path: str | Path,
    name: str,
    info: LitoQuantizedTensorInfo,
    spec: LitoQuantizationSpec,
    qweight: mx.array,
    scales: mx.array,
    biases: mx.array,
) -> None:
    qweight_shape, group_shape = _expected_quantized_shapes(info.shape, spec.bits, spec.group_size)
    if tuple(qweight.shape) != qweight_shape or qweight.dtype != mx.uint32:
        raise ValueError(f"invalid packed qweight for {name!r} in {path}")
    if tuple(scales.shape) != group_shape or scales.dtype != mx.float32:
        raise ValueError(f"invalid affine scales for {name!r} in {path}")
    if tuple(biases.shape) != group_shape or biases.dtype != mx.float32:
        raise ValueError(f"invalid affine biases for {name!r} in {path}")


def _expected_quantized_shapes(
    shape: tuple[int, int],
    bits: int,
    group_size: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    rows, columns = shape
    packed_values = columns * bits
    if packed_values % 32:
        raise ValueError(f"matrix shape {shape} cannot be packed into uint32 values at {bits} bits")
    return (rows, packed_values // 32), (rows, columns // group_size)


def _raise_orphaned_quantized_tensors(path: str | Path, tensors: Mapping[str, Any]) -> None:
    orphaned = sorted(name for name in tensors if name.endswith(_QUANTIZED_SUFFIXES))
    if orphaned:
        raise ValueError(f"LiTo checkpoint contains packed tensors absent from its metadata: {orphaned[0]!r} in {path}")


def _tensor_nbytes(header: SafetensorHeader) -> int:
    byte_widths = {"F16": 2, "BF16": 2, "F32": 4}
    return math.prod(header.shape) * byte_widths.get(header.dtype, 0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Selectively quantize LiTo inference weights with MLX affine packing")
    parser.add_argument("source", help="full-precision LiTo weights root")
    parser.add_argument("output", help="new quantized LiTo weights root")
    parser.add_argument("--bits", type=int, default=8, choices=range(2, 9))
    parser.add_argument("--group-size", type=int, default=64, choices=sorted(_AFFINE_GROUP_SIZES))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _build_prune_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remove tensors unused by the main LiTo inference runtime")
    parser.add_argument("source", help="full LiTo MLX weights root")
    parser.add_argument("output", help="runtime-only LiTo MLX weights root; may equal source with --overwrite")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    results = quantize_lito_weights(
        args.source,
        args.output,
        bits=args.bits,
        group_size=args.group_size,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "format": LITO_QUANTIZATION_FORMAT,
                "bits": args.bits,
                "group_size": args.group_size,
                "checkpoints": [
                    {
                        "output": str(result.output),
                        "source_tensors": result.source_tensor_count,
                        "runtime_tensors": result.runtime_tensor_count,
                        "removed_tensors": result.removed_tensor_count,
                        "quantized_tensors": result.quantized_tensor_count,
                        "source_payload_bytes": result.source_payload_bytes,
                        "runtime_source_bytes": result.runtime_source_bytes,
                        "removed_source_bytes": result.removed_source_bytes,
                        "quantized_source_bytes": result.quantized_source_bytes,
                        "output_bytes": result.output.stat().st_size,
                    }
                    for result in results
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def prune_main(argv: Sequence[str] | None = None) -> int:
    args = _build_prune_parser().parse_args(argv)
    results = prune_lito_weights(args.source, args.output, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "format": LITO_RUNTIME_FORMAT,
                "checkpoints": [
                    {
                        "output": str(result.output),
                        "source_tensors": result.source_tensor_count,
                        "runtime_tensors": result.runtime_tensor_count,
                        "removed_tensors": result.removed_tensor_count,
                        "source_payload_bytes": result.source_payload_bytes,
                        "runtime_payload_bytes": result.runtime_payload_bytes,
                        "removed_payload_bytes": result.removed_payload_bytes,
                        "output_bytes": result.output.stat().st_size,
                    }
                    for result in results
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
