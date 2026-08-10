"""Selective MLX affine quantization for SAM 3D Objects checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import mlx.core as mx

from .checkpoint import CheckpointTensorInfo
from .safetensors_io import (
    SafetensorHeader,
    inspect_safetensors,
    load_mlx_safetensors,
    read_safetensors_metadata,
    save_safetensors,
)


SAM3D_QUANTIZATION_FORMAT = "mlx-spatial-sam3d-runtime-affine-v1"
SAM3D_QUANTIZATION_METADATA_KEY = "mlx_spatial.sam3d.quantization"
SAM3D_QUANTIZATION_POLICY = "selective-internal-transformer-linear-v1"
SAM3D_CHECKPOINT_PATHS = (
    Path("checkpoints/ss_generator.safetensors"),
    Path("checkpoints/ss_decoder.safetensors"),
    Path("checkpoints/slat_generator.safetensors"),
    Path("checkpoints/slat_decoder_gs.safetensors"),
    Path("checkpoints/slat_decoder_gs_4.safetensors"),
    Path("checkpoints/slat_decoder_mesh.safetensors"),
    Path("moge/model.safetensors"),
)

_QWEIGHT_SUFFIX = ".__mlx_qweight__"
_SCALES_SUFFIX = ".__mlx_scales__"
_BIASES_SUFFIX = ".__mlx_biases__"
_QUANTIZED_SUFFIXES = (_QWEIGHT_SUFFIX, _SCALES_SUFFIX, _BIASES_SUFFIX)
_FLOAT_DTYPES = frozenset(("F16", "BF16", "F32"))
_AFFINE_GROUP_SIZES = frozenset((32, 64, 128))

_CONDITION_DINO_BLOCK_WEIGHT = re.compile(
    r"^_base_models\.condition_embedder\.module_list\.\d+\.backbone\.blocks\.\d+\..*\.weight$"
)
_CONDITION_POINT_BLOCK_WEIGHT = re.compile(
    r"^_base_models\.condition_embedder\.module_list\.2\.blocks\.\d+\..*\.weight$"
)
_GENERATOR_BLOCK_WEIGHT = re.compile(
    r"^_base_models\.generator\.reverse_fn\.backbone\.blocks\.\d+\..*\.weight$"
)
_DECODER_BLOCK_WEIGHT = re.compile(r"^blocks\.\d+\..*\.weight$")
_MOGE_BLOCK_WEIGHT = re.compile(r"^backbone\.blocks\.\d+\..*\.weight$")


@dataclass(frozen=True)
class Sam3dQuantizedTensorInfo:
    """Logical shape and dtype of one packed SAM3D matrix."""

    shape: tuple[int, int]
    dtype: str


@dataclass(frozen=True)
class Sam3dQuantizationSpec:
    """Validated affine quantization metadata for one SAM3D checkpoint."""

    bits: int
    group_size: int
    mode: str
    tensors: Mapping[str, Sam3dQuantizedTensorInfo]
    policy: str


@dataclass(frozen=True)
class Sam3dQuantizedMatrix:
    """Packed affine matrix consumed directly by ``mx.quantized_matmul``."""

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


Sam3dWeight = mx.array | Sam3dQuantizedMatrix


@dataclass(frozen=True)
class Sam3dCheckpointQuantizationResult:
    """Summary for one output checkpoint."""

    checkpoint: Path
    source: Path
    output: Path
    source_tensor_count: int
    quantized_tensor_count: int
    source_payload_bytes: int
    quantized_source_bytes: int


def should_quantize_sam3d_tensor(
    checkpoint: str | Path,
    name: str,
    shape: Sequence[int],
    dtype: str,
    *,
    group_size: int = 64,
) -> bool:
    """Select block-internal matrices while preserving model boundaries."""

    if group_size not in _AFFINE_GROUP_SIZES:
        raise ValueError(f"affine group_size must be one of {sorted(_AFFINE_GROUP_SIZES)}, got {group_size}")
    if dtype not in _FLOAT_DTYPES or len(shape) != 2 or int(shape[1]) % group_size:
        return False

    relative_path = _normalize_checkpoint_path(checkpoint)
    if relative_path.name in {
        "slat_decoder_gs.safetensors",
        "slat_decoder_gs_4.safetensors",
        "slat_decoder_mesh.safetensors",
    }:
        return _DECODER_BLOCK_WEIGHT.fullmatch(name) is not None
    if relative_path.name in {"ss_generator.safetensors", "slat_generator.safetensors"}:
        return any(
            pattern.fullmatch(name) is not None
            for pattern in (
                _CONDITION_DINO_BLOCK_WEIGHT,
                _CONDITION_POINT_BLOCK_WEIGHT,
                _GENERATOR_BLOCK_WEIGHT,
            )
        )
    if relative_path.as_posix().endswith("moge/model.safetensors"):
        return _MOGE_BLOCK_WEIGHT.fullmatch(name) is not None
    return False


def read_sam3d_quantization_spec(path: str | Path) -> Sam3dQuantizationSpec | None:
    """Read and validate SAM3D quantization metadata, if present."""

    raw = read_safetensors_metadata(path).get(SAM3D_QUANTIZATION_METADATA_KEY)
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid SAM3D quantization metadata in {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("format") != SAM3D_QUANTIZATION_FORMAT:
        raise ValueError(f"unsupported SAM3D quantization format in {path}")

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
        raise ValueError(f"unsupported SAM3D quantization mode in {path}: {mode!r}")
    if policy != SAM3D_QUANTIZATION_POLICY:
        raise ValueError(f"unsupported SAM3D quantization policy in {path}: {policy!r}")
    if not isinstance(raw_tensors, dict) or not raw_tensors:
        raise ValueError(f"missing quantized tensor inventory in {path}")

    tensors: dict[str, Sam3dQuantizedTensorInfo] = {}
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
        tensors[name] = Sam3dQuantizedTensorInfo(shape=(shape[0], shape[1]), dtype=dtype)
    return Sam3dQuantizationSpec(
        bits=bits,
        group_size=group_size,
        mode=mode,
        tensors=tensors,
        policy=policy,
    )


def inspect_logical_sam3d_safetensors(path: str | Path) -> tuple[SafetensorHeader, ...]:
    """Expose original logical names for full-precision and packed checkpoints."""

    physical = {header.name: header for header in inspect_safetensors(path)}
    spec = read_sam3d_quantization_spec(path)
    if spec is None:
        return tuple(sorted(physical.values(), key=lambda header: header.name))

    logical = dict(physical)
    for name, info in spec.tensors.items():
        if name in logical:
            raise ValueError(f"quantized SAM3D checkpoint contains both packed and full-precision tensor {name!r}")
        _validate_physical_quantized_headers(path, name, info, spec, logical)
        for suffix in _QUANTIZED_SUFFIXES:
            logical.pop(name + suffix)
        logical[name] = SafetensorHeader(name=name, shape=info.shape, dtype=info.dtype)
    _raise_orphaned_quantized_tensors(path, logical)
    return tuple(sorted(logical.values(), key=lambda header: header.name))


def load_logical_sam3d_safetensors(path: str | Path) -> dict[str, Sam3dWeight]:
    """Load tensors while reconstructing packed matrices under original names."""

    physical = load_mlx_safetensors(path)
    spec = read_sam3d_quantization_spec(path)
    if spec is None:
        return physical

    logical: dict[str, Sam3dWeight] = dict(physical)
    for name, info in spec.tensors.items():
        if name in logical:
            raise ValueError(f"quantized SAM3D checkpoint contains both packed and full-precision tensor {name!r}")
        try:
            qweight = logical.pop(name + _QWEIGHT_SUFFIX)
            scales = logical.pop(name + _SCALES_SUFFIX)
            biases = logical.pop(name + _BIASES_SUFFIX)
        except KeyError as error:
            raise ValueError(f"quantized SAM3D checkpoint is missing packed tensor {error.args[0]!r}") from error
        if not isinstance(qweight, mx.array) or not isinstance(scales, mx.array) or not isinstance(biases, mx.array):
            raise TypeError(f"invalid packed SAM3D matrix for {name!r} in {path}")
        _validate_loaded_quantized_tensors(path, name, info, spec, qweight, scales, biases)
        logical[name] = Sam3dQuantizedMatrix(
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


def inspect_sam3d_checkpoint(
    path: str | Path,
    *,
    names: Iterable[str] | None = None,
    prefixes: Iterable[str] | None = None,
) -> tuple[CheckpointTensorInfo, ...]:
    """Inspect a SAM3D checkpoint using logical tensor names and shapes."""

    checkpoint_path = _validate_checkpoint_path(path)
    exact_names, name_prefixes, has_filter = _normalize_filters(names, prefixes)
    infos = tuple(
        CheckpointTensorInfo(
            name=header.name,
            shape=header.shape,
            dtype=header.dtype,
            source=str(checkpoint_path),
        )
        for header in inspect_logical_sam3d_safetensors(checkpoint_path)
        if not has_filter or _matches_filter(header.name, exact_names, name_prefixes)
    )
    if has_filter and not infos:
        raise ValueError("checkpoint filters matched no tensors")
    return infos


def load_sam3d_checkpoint_tensors(
    path: str | Path,
    *,
    names: Iterable[str] | None = None,
    prefixes: Iterable[str] | None = None,
) -> dict[str, Sam3dWeight]:
    """Load selected SAM3D tensors with transparent packed-matrix support."""

    checkpoint_path = _validate_checkpoint_path(path)
    exact_names, name_prefixes, has_filter = _normalize_filters(names, prefixes)
    if not has_filter:
        raise ValueError("loading checkpoint tensors requires names or prefixes")
    tensors = load_logical_sam3d_safetensors(checkpoint_path)
    loaded = {
        name: tensors[name]
        for name in sorted(tensors)
        if _matches_filter(name, exact_names, name_prefixes)
    }
    if not loaded:
        raise ValueError("checkpoint filters matched no tensors")
    if exact_names:
        missing = sorted(exact_names.difference(loaded))
        if missing:
            raise ValueError(f"checkpoint is missing requested tensors: {missing}")
    return loaded


def sam3d_linear(values: mx.array, weight: Sam3dWeight, bias: mx.array | None) -> mx.array:
    """Apply a full-precision or packed affine SAM3D linear weight."""

    if isinstance(weight, Sam3dQuantizedMatrix):
        if int(values.shape[-1]) != weight.shape[1]:
            raise ValueError(
                f"SAM3D quantized linear input has {int(values.shape[-1])} channels, expected {weight.shape[1]}"
            )
        input_shape = tuple(int(dim) for dim in values.shape)
        flat = mx.reshape(values, (-1, input_shape[-1]))
        output = mx.quantized_matmul(
            flat,
            weight.qweight,
            weight.scales,
            weight.biases,
            transpose=True,
            group_size=weight.group_size,
            bits=weight.bits,
            mode=weight.mode,
        )
        output = mx.reshape(output, input_shape[:-1] + (weight.shape[0],))
    else:
        output = values @ mx.transpose(weight.astype(values.dtype))
    if bias is not None:
        output = output + bias.astype(output.dtype)
    return output


def quantize_sam3d_checkpoint(
    source: str | Path,
    output: str | Path,
    *,
    checkpoint: str | Path,
    bits: int = 8,
    group_size: int = 64,
    overwrite: bool = False,
) -> Sam3dCheckpointQuantizationResult:
    """Quantize selected matrices while retaining every logical tensor."""

    source_path = Path(source)
    output_path = Path(output)
    relative_path = _normalize_checkpoint_path(checkpoint)
    _validate_quantization_options(bits, group_size)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("SAM3D quantization output must differ from the full-precision source")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"SAM3D quantization output already exists: {output_path}")
    if read_sam3d_quantization_spec(source_path) is not None:
        raise ValueError(f"SAM3D source checkpoint is already quantized: {source_path}")

    headers = inspect_safetensors(source_path)
    selected = [
        header
        for header in headers
        if should_quantize_sam3d_tensor(
            relative_path,
            header.name,
            header.shape,
            header.dtype,
            group_size=group_size,
        )
    ]
    source_payload_bytes = sum(_tensor_nbytes(header) for header in headers)
    quantized_source_bytes = sum(_tensor_nbytes(header) for header in selected)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not selected:
        shutil.copy2(source_path, output_path)
        return Sam3dCheckpointQuantizationResult(
            checkpoint=relative_path,
            source=source_path,
            output=output_path,
            source_tensor_count=len(headers),
            quantized_tensor_count=0,
            source_payload_bytes=source_payload_bytes,
            quantized_source_bytes=0,
        )

    output_tensors: dict[str, mx.array] = load_mlx_safetensors(source_path)
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
    metadata[SAM3D_QUANTIZATION_METADATA_KEY] = json.dumps(
        {
            "format": SAM3D_QUANTIZATION_FORMAT,
            "bits": bits,
            "group_size": group_size,
            "mode": "affine",
            "policy": SAM3D_QUANTIZATION_POLICY,
            "checkpoint": relative_path.as_posix(),
            "tensors": logical_tensors,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    save_safetensors(output_path, output_tensors, metadata=metadata)
    mx.clear_cache()
    return Sam3dCheckpointQuantizationResult(
        checkpoint=relative_path,
        source=source_path,
        output=output_path,
        source_tensor_count=len(headers),
        quantized_tensor_count=len(selected),
        source_payload_bytes=source_payload_bytes,
        quantized_source_bytes=quantized_source_bytes,
    )


def quantize_sam3d_weights(
    source_root: str | Path,
    output_root: str | Path,
    *,
    bits: int = 8,
    group_size: int = 64,
    overwrite: bool = False,
    progress: Callable[[Path], None] | None = None,
) -> tuple[Sam3dCheckpointQuantizationResult, ...]:
    """Create a complete same-layout selectively quantized SAM3D root."""

    source = Path(source_root)
    output = Path(output_root)
    _validate_quantization_options(bits, group_size)
    if source.resolve() == output.resolve():
        raise ValueError("SAM3D quantization output root must differ from the full-precision source root")
    if not source.is_dir():
        raise FileNotFoundError(f"SAM3D source weights root not found: {source}")
    for relative_path in SAM3D_CHECKPOINT_PATHS:
        source_path = source / relative_path
        output_path = output / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(f"SAM3D source checkpoint not found: {source_path}")
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"SAM3D quantization output already exists: {output_path}")

    _copy_non_checkpoint_assets(source, output, overwrite=overwrite)
    results = []
    for relative_path in SAM3D_CHECKPOINT_PATHS:
        if progress is not None:
            progress(relative_path)
        results.append(
            quantize_sam3d_checkpoint(
                source / relative_path,
                output / relative_path,
                checkpoint=relative_path,
                bits=bits,
                group_size=group_size,
                overwrite=overwrite,
            )
        )
    return tuple(results)


def _copy_non_checkpoint_assets(source: Path, output: Path, *, overwrite: bool) -> None:
    checkpoints = {path.as_posix() for path in SAM3D_CHECKPOINT_PATHS}
    for source_path in sorted(source.rglob("*")):
        if not source_path.is_file():
            continue
        relative_path = source_path.relative_to(source)
        if ".cache" in relative_path.parts or relative_path.as_posix() in checkpoints:
            continue
        output_path = output / relative_path
        if output_path.exists() and not overwrite:
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)


def _normalize_checkpoint_path(checkpoint: str | Path) -> Path:
    value = Path(checkpoint)
    normalized = value.as_posix()
    for relative_path in SAM3D_CHECKPOINT_PATHS:
        if normalized.endswith(relative_path.as_posix()):
            return relative_path
    return value


def _validate_checkpoint_path(path: str | Path) -> Path:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {checkpoint_path}")
    if checkpoint_path.suffix != ".safetensors":
        raise ValueError(f"unsupported checkpoint format: {checkpoint_path.suffix or '<none>'}")
    return checkpoint_path


def _normalize_filters(
    names: Iterable[str] | None,
    prefixes: Iterable[str] | None,
) -> tuple[set[str], tuple[str, ...], bool]:
    exact_names = _normalize_string_filter(names, "names")
    name_prefixes = tuple(sorted(_normalize_string_filter(prefixes, "prefixes")))
    return exact_names, name_prefixes, bool(exact_names or name_prefixes)


def _normalize_string_filter(values: Iterable[str] | None, label: str) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        raise ValueError(f"{label} must be an iterable of non-empty strings")
    normalized = set(values)
    if not normalized:
        raise ValueError(f"{label} must not be empty when provided")
    if any(not isinstance(value, str) or not value for value in normalized):
        raise ValueError(f"{label} must contain only non-empty strings")
    return normalized


def _matches_filter(name: str, exact_names: set[str], prefixes: tuple[str, ...]) -> bool:
    return name in exact_names or any(name.startswith(prefix) for prefix in prefixes)


def _validate_quantization_options(bits: int, group_size: int) -> None:
    if not isinstance(bits, int) or not 2 <= bits <= 8:
        raise ValueError(f"affine bits must be an integer from 2 through 8, got {bits!r}")
    if group_size not in _AFFINE_GROUP_SIZES:
        raise ValueError(f"affine group_size must be one of {sorted(_AFFINE_GROUP_SIZES)}, got {group_size}")


def _validate_physical_quantized_headers(
    path: str | Path,
    name: str,
    info: Sam3dQuantizedTensorInfo,
    spec: Sam3dQuantizationSpec,
    headers: Mapping[str, SafetensorHeader],
) -> None:
    qweight_shape, group_shape = _expected_quantized_shapes(info.shape, spec.bits, spec.group_size)
    for suffix, expected_shape, expected_dtype in (
        (_QWEIGHT_SUFFIX, qweight_shape, "U32"),
        (_SCALES_SUFFIX, group_shape, "F32"),
        (_BIASES_SUFFIX, group_shape, "F32"),
    ):
        header = headers.get(name + suffix)
        if header is None or header.shape != expected_shape or header.dtype != expected_dtype:
            raise ValueError(
                f"invalid packed tensor {name + suffix!r} in {path}; expected {expected_dtype} {expected_shape}"
            )


def _validate_loaded_quantized_tensors(
    path: str | Path,
    name: str,
    info: Sam3dQuantizedTensorInfo,
    spec: Sam3dQuantizationSpec,
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
        raise ValueError(f"SAM3D checkpoint contains packed tensors absent from metadata: {orphaned[0]!r} in {path}")


def _tensor_nbytes(header: SafetensorHeader) -> int:
    byte_widths = {"F16": 2, "BF16": 2, "F32": 4}
    return math.prod(header.shape) * byte_widths.get(header.dtype, 0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Selectively quantize SAM3D inference weights with MLX affine packing")
    parser.add_argument("source", help="full-precision SAM3D MLX weights root")
    parser.add_argument("output", help="new same-layout quantized SAM3D weights root")
    parser.add_argument("--bits", type=int, default=8, choices=range(2, 9))
    parser.add_argument("--group-size", type=int, default=64, choices=sorted(_AFFINE_GROUP_SIZES))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    results = quantize_sam3d_weights(
        args.source,
        args.output,
        bits=args.bits,
        group_size=args.group_size,
        overwrite=args.overwrite,
        progress=lambda path: print(f"quantizing {path.as_posix()}", flush=True),
    )
    print(
        json.dumps(
            {
                "format": SAM3D_QUANTIZATION_FORMAT,
                "bits": args.bits,
                "group_size": args.group_size,
                "source": str(Path(args.source)),
                "output": str(Path(args.output)),
                "checkpoints": [
                    {
                        "checkpoint": result.checkpoint.as_posix(),
                        "source_tensors": result.source_tensor_count,
                        "quantized_tensors": result.quantized_tensor_count,
                        "source_payload_bytes": result.source_payload_bytes,
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
