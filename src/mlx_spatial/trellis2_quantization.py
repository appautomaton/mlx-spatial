"""Selective MLX affine quantization for TRELLIS.2 inference weights."""

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

from .safetensors_io import (
    SafetensorHeader,
    inspect_safetensors,
    load_mlx_safetensors,
    read_safetensors_metadata,
    save_safetensors,
)


TRELLIS2_QUANTIZATION_FORMAT = "mlx-spatial-trellis2-runtime-affine-v1"
TRELLIS2_QUANTIZATION_METADATA_KEY = "mlx_spatial.trellis2.quantization"
TRELLIS2_QUANTIZATION_POLICY = "selective-internal-linear-v1"
TRELLIS2_MAIN_CHECKPOINT_PATHS = (
    Path("ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"),
    Path("microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"),
    Path("ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors"),
    Path("ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors"),
    Path("ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors"),
    Path("ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors"),
    Path("ckpts/shape_dec_next_dc_f16c32_fp16.safetensors"),
    Path("ckpts/shape_enc_next_dc_f16c32_fp16.safetensors"),
    Path("ckpts/tex_dec_next_dc_f16c32_fp16.safetensors"),
    Path("ckpts/tex_enc_next_dc_f16c32_fp16.safetensors"),
)
TRELLIS2_DINOV3_CHECKPOINT_PATH = Path("dinov3/model.safetensors")
TRELLIS2_RMBG_CHECKPOINT_PATH = Path("rmbg/model.safetensors")

_QWEIGHT_SUFFIX = ".__mlx_qweight__"
_SCALES_SUFFIX = ".__mlx_scales__"
_BIASES_SUFFIX = ".__mlx_biases__"
_QUANTIZED_SUFFIXES = (_QWEIGHT_SUFFIX, _SCALES_SUFFIX, _BIASES_SUFFIX)
_FLOAT_DTYPES = frozenset(("F16", "BF16", "F32"))
_AFFINE_GROUP_SIZES = frozenset((32, 64, 128))

_DIT_CHECKPOINT_NAMES = frozenset(
    (
        "ss_flow_img_dit_1_3B_64_bf16.safetensors",
        "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
        "slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
        "slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
        "slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
    )
)
_CONVNEXT_CHECKPOINT_NAMES = frozenset(
    (
        "shape_dec_next_dc_f16c32_fp16.safetensors",
        "shape_enc_next_dc_f16c32_fp16.safetensors",
        "tex_dec_next_dc_f16c32_fp16.safetensors",
        "tex_enc_next_dc_f16c32_fp16.safetensors",
    )
)
_DIT_BLOCK_WEIGHT = re.compile(
    r"^blocks\.\d+\.(?:"
    r"self_attn\.(?:to_qkv|to_out)|"
    r"cross_attn\.(?:to_q|to_kv|to_out)|"
    r"mlp\.mlp\.(?:0|2)"
    r")\.weight$"
)
_CONVNEXT_BLOCK_MLP_WEIGHT = re.compile(r"^blocks\.\d+\.\d+\.mlp\.(?:0|2)\.weight$")
_DINOV3_BLOCK_WEIGHT = re.compile(
    r"^layer\.\d+\.(?:"
    r"attention\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:up_proj|down_proj)"
    r")\.weight$"
)
_RMBG_SWIN_BLOCK_WEIGHT = re.compile(
    r"^bb\.layers\.\d+\.blocks\.\d+\.(?:"
    r"attn\.(?:qkv|proj)|"
    r"mlp\.(?:fc1|fc2)"
    r")\.weight$"
)


@dataclass(frozen=True)
class Trellis2QuantizedTensorInfo:
    """Logical descriptor for one packed affine matrix."""

    shape: tuple[int, int]
    dtype: str


@dataclass(frozen=True)
class Trellis2QuantizationSpec:
    """Validated affine quantization metadata for one checkpoint."""

    bits: int
    group_size: int
    mode: str
    policy: str
    checkpoint: str
    tensors: Mapping[str, Trellis2QuantizedTensorInfo]


@dataclass(frozen=True)
class Trellis2QuantizedMatrix:
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


Trellis2Weight = mx.array | Trellis2QuantizedMatrix


@dataclass(frozen=True)
class Trellis2CheckpointQuantizationResult:
    """Summary for one output checkpoint."""

    checkpoint: Path
    source: Path
    output: Path
    source_tensor_count: int
    quantized_tensor_count: int
    source_payload_bytes: int
    quantized_source_bytes: int


def should_quantize_trellis2_tensor(
    checkpoint: str | Path,
    name: str,
    shape: Sequence[int],
    dtype: str,
    *,
    group_size: int = 64,
) -> bool:
    """Select internal matrices while retaining convolutional and model boundaries."""

    if group_size not in _AFFINE_GROUP_SIZES:
        raise ValueError(f"affine group_size must be one of {sorted(_AFFINE_GROUP_SIZES)}, got {group_size}")
    if dtype not in _FLOAT_DTYPES or len(shape) != 2 or int(shape[1]) % group_size:
        return False

    relative_path = _normalize_checkpoint_path(checkpoint)
    if relative_path.name in _DIT_CHECKPOINT_NAMES:
        return _DIT_BLOCK_WEIGHT.fullmatch(name) is not None
    if relative_path.name in _CONVNEXT_CHECKPOINT_NAMES:
        return _CONVNEXT_BLOCK_MLP_WEIGHT.fullmatch(name) is not None
    if relative_path == TRELLIS2_DINOV3_CHECKPOINT_PATH:
        return _DINOV3_BLOCK_WEIGHT.fullmatch(name) is not None
    if relative_path == TRELLIS2_RMBG_CHECKPOINT_PATH:
        return _RMBG_SWIN_BLOCK_WEIGHT.fullmatch(name) is not None
    return False


def read_trellis2_quantization_spec(path: str | Path) -> Trellis2QuantizationSpec | None:
    """Read and validate TRELLIS.2 affine metadata, if present."""

    raw = read_safetensors_metadata(path).get(TRELLIS2_QUANTIZATION_METADATA_KEY)
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid TRELLIS.2 quantization metadata in {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("format") != TRELLIS2_QUANTIZATION_FORMAT:
        raise ValueError(f"unsupported TRELLIS.2 quantization format in {path}")

    bits = payload.get("bits")
    group_size = payload.get("group_size")
    mode = payload.get("mode")
    policy = payload.get("policy")
    checkpoint = payload.get("checkpoint")
    raw_tensors = payload.get("tensors")
    if not isinstance(bits, int) or not 2 <= bits <= 8:
        raise ValueError(f"invalid affine bit width in {path}: {bits!r}")
    if group_size not in _AFFINE_GROUP_SIZES:
        raise ValueError(f"invalid affine group size in {path}: {group_size!r}")
    if mode != "affine":
        raise ValueError(f"unsupported TRELLIS.2 quantization mode in {path}: {mode!r}")
    if policy != TRELLIS2_QUANTIZATION_POLICY:
        raise ValueError(f"unsupported TRELLIS.2 quantization policy in {path}: {policy!r}")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(f"missing TRELLIS.2 checkpoint identity in {path}")
    if not isinstance(raw_tensors, dict) or not raw_tensors:
        raise ValueError(f"missing quantized tensor inventory in {path}")

    tensors: dict[str, Trellis2QuantizedTensorInfo] = {}
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
        tensors[name] = Trellis2QuantizedTensorInfo(shape=(shape[0], shape[1]), dtype=dtype)
    return Trellis2QuantizationSpec(
        bits=bits,
        group_size=group_size,
        mode=mode,
        policy=policy,
        checkpoint=checkpoint,
        tensors=tensors,
    )


def inspect_logical_trellis2_safetensors(path: str | Path) -> tuple[SafetensorHeader, ...]:
    """Expose original logical names for full-precision and packed checkpoints."""

    physical = {header.name: header for header in inspect_safetensors(path)}
    spec = read_trellis2_quantization_spec(path)
    if spec is None:
        return tuple(sorted(physical.values(), key=lambda header: header.name))

    logical = dict(physical)
    for name, info in spec.tensors.items():
        if name in logical:
            raise ValueError(f"TRELLIS.2 checkpoint contains packed and full-precision tensor {name!r}")
        _validate_physical_quantized_headers(path, name, info, spec, logical)
        for suffix in _QUANTIZED_SUFFIXES:
            logical.pop(name + suffix)
        logical[name] = SafetensorHeader(name=name, shape=info.shape, dtype=info.dtype)
    _raise_orphaned_quantized_tensors(path, logical)
    return tuple(sorted(logical.values(), key=lambda header: header.name))


def load_logical_trellis2_safetensors(path: str | Path) -> dict[str, Trellis2Weight]:
    """Load tensors while reconstructing packed matrices under original names."""

    physical = load_mlx_safetensors(path)
    spec = read_trellis2_quantization_spec(path)
    if spec is None:
        return physical

    logical: dict[str, Trellis2Weight] = dict(physical)
    for name, info in spec.tensors.items():
        if name in logical:
            raise ValueError(f"TRELLIS.2 checkpoint contains packed and full-precision tensor {name!r}")
        try:
            qweight = logical.pop(name + _QWEIGHT_SUFFIX)
            scales = logical.pop(name + _SCALES_SUFFIX)
            biases = logical.pop(name + _BIASES_SUFFIX)
        except KeyError as error:
            raise ValueError(f"TRELLIS.2 checkpoint is missing packed tensor {error.args[0]!r}") from error
        if not isinstance(qweight, mx.array) or not isinstance(scales, mx.array) or not isinstance(biases, mx.array):
            raise TypeError(f"invalid packed TRELLIS.2 matrix for {name!r} in {path}")
        _validate_loaded_quantized_tensors(path, name, info, spec, qweight, scales, biases)
        logical[name] = Trellis2QuantizedMatrix(
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


def trellis2_linear(values: mx.array, weight: Trellis2Weight, bias: mx.array | None) -> mx.array:
    """Apply a full-precision or packed affine TRELLIS.2 linear weight."""

    if isinstance(weight, Trellis2QuantizedMatrix):
        if int(values.shape[-1]) != weight.shape[1]:
            raise ValueError(
                f"TRELLIS.2 quantized linear input has {int(values.shape[-1])} channels, "
                f"expected {weight.shape[1]}"
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


def quantize_trellis2_checkpoint(
    source: str | Path,
    output: str | Path,
    *,
    checkpoint: str | Path,
    bits: int = 8,
    group_size: int = 64,
    overwrite: bool = False,
) -> Trellis2CheckpointQuantizationResult:
    """Quantize selected matrices while retaining every logical tensor."""

    source_path = Path(source)
    output_path = Path(output)
    relative_path = _normalize_checkpoint_path(checkpoint)
    _validate_quantization_options(bits, group_size)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("TRELLIS.2 quantization output must differ from the full-precision source")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"TRELLIS.2 quantization output already exists: {output_path}")
    if read_trellis2_quantization_spec(source_path) is not None:
        raise ValueError(f"TRELLIS.2 source checkpoint is already quantized: {source_path}")

    headers = inspect_safetensors(source_path)
    selected = [
        header
        for header in headers
        if should_quantize_trellis2_tensor(
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
        return Trellis2CheckpointQuantizationResult(
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
    metadata[TRELLIS2_QUANTIZATION_METADATA_KEY] = json.dumps(
        {
            "format": TRELLIS2_QUANTIZATION_FORMAT,
            "bits": bits,
            "group_size": group_size,
            "mode": "affine",
            "policy": TRELLIS2_QUANTIZATION_POLICY,
            "checkpoint": relative_path.as_posix(),
            "tensors": logical_tensors,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    save_safetensors(output_path, output_tensors, metadata=metadata)
    mx.clear_cache()
    return Trellis2CheckpointQuantizationResult(
        checkpoint=relative_path,
        source=source_path,
        output=output_path,
        source_tensor_count=len(headers),
        quantized_tensor_count=len(selected),
        source_payload_bytes=source_payload_bytes,
        quantized_source_bytes=quantized_source_bytes,
    )


def quantize_trellis2_weights(
    source_root: str | Path,
    output_root: str | Path,
    *,
    dinov3_root: str | Path | None = None,
    rmbg_root: str | Path | None = None,
    bits: int = 8,
    group_size: int = 64,
    overwrite: bool = False,
    progress: Callable[[Path], None] | None = None,
) -> tuple[Trellis2CheckpointQuantizationResult, ...]:
    """Create a complete same-layout selectively quantized TRELLIS.2 root."""

    source = Path(source_root)
    output = Path(output_root)
    _validate_quantization_options(bits, group_size)
    if source.resolve() == output.resolve():
        raise ValueError("TRELLIS.2 quantization output root must differ from the source root")
    if not source.is_dir():
        raise FileNotFoundError(f"TRELLIS.2 source weights root not found: {source}")

    targets: list[tuple[Path, Path, Path]] = [
        (source / relative_path, output / relative_path, relative_path)
        for relative_path in TRELLIS2_MAIN_CHECKPOINT_PATHS
    ]
    auxiliary_roots: list[tuple[Path, Path, Path]] = []
    if dinov3_root is not None:
        dino_source = Path(dinov3_root)
        auxiliary_roots.append((dino_source, output / "dinov3", TRELLIS2_DINOV3_CHECKPOINT_PATH))
        targets.append(
            (
                dino_source / "model.safetensors",
                output / TRELLIS2_DINOV3_CHECKPOINT_PATH,
                TRELLIS2_DINOV3_CHECKPOINT_PATH,
            )
        )
    if rmbg_root is not None:
        rmbg_source = Path(rmbg_root)
        auxiliary_roots.append((rmbg_source, output / "rmbg", TRELLIS2_RMBG_CHECKPOINT_PATH))
        targets.append(
            (
                rmbg_source / "model.safetensors",
                output / TRELLIS2_RMBG_CHECKPOINT_PATH,
                TRELLIS2_RMBG_CHECKPOINT_PATH,
            )
        )

    for source_path, output_path, _ in targets:
        if not source_path.is_file():
            raise FileNotFoundError(f"TRELLIS.2 source checkpoint not found: {source_path}")
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"TRELLIS.2 quantization output already exists: {output_path}")

    _copy_non_checkpoint_assets(
        source,
        output,
        checkpoint_paths=TRELLIS2_MAIN_CHECKPOINT_PATHS,
        overwrite=overwrite,
    )
    for auxiliary_source, auxiliary_output, _ in auxiliary_roots:
        _copy_non_checkpoint_assets(
            auxiliary_source,
            auxiliary_output,
            checkpoint_paths=(Path("model.safetensors"),),
            overwrite=overwrite,
        )

    results = []
    for source_path, output_path, relative_path in targets:
        if progress is not None:
            progress(relative_path)
        results.append(
            quantize_trellis2_checkpoint(
                source_path,
                output_path,
                checkpoint=relative_path,
                bits=bits,
                group_size=group_size,
                overwrite=overwrite,
            )
        )
    return tuple(results)


def _copy_non_checkpoint_assets(
    source: Path,
    output: Path,
    *,
    checkpoint_paths: Iterable[Path],
    overwrite: bool,
) -> None:
    checkpoints = {path.as_posix() for path in checkpoint_paths}
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
    known = TRELLIS2_MAIN_CHECKPOINT_PATHS + (
        TRELLIS2_DINOV3_CHECKPOINT_PATH,
        TRELLIS2_RMBG_CHECKPOINT_PATH,
    )
    for relative_path in known:
        if normalized.endswith(relative_path.as_posix()):
            return relative_path
    return value


def _validate_quantization_options(bits: int, group_size: int) -> None:
    if not isinstance(bits, int) or not 2 <= bits <= 8:
        raise ValueError(f"affine bits must be an integer from 2 through 8, got {bits!r}")
    if group_size not in _AFFINE_GROUP_SIZES:
        raise ValueError(f"affine group_size must be one of {sorted(_AFFINE_GROUP_SIZES)}, got {group_size}")


def _validate_physical_quantized_headers(
    path: str | Path,
    name: str,
    info: Trellis2QuantizedTensorInfo,
    spec: Trellis2QuantizationSpec,
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
    info: Trellis2QuantizedTensorInfo,
    spec: Trellis2QuantizationSpec,
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
        raise ValueError(f"TRELLIS.2 checkpoint has packed tensors absent from metadata: {orphaned[0]!r} in {path}")


def _tensor_nbytes(header: SafetensorHeader) -> int:
    byte_widths = {"F16": 2, "BF16": 2, "F32": 4}
    return math.prod(header.shape) * byte_widths.get(header.dtype, 0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Selectively quantize TRELLIS.2 inference weights with MLX affine packing"
    )
    parser.add_argument("source", help="full-precision TRELLIS.2 MLX weights root")
    parser.add_argument("output", help="new same-layout quantized TRELLIS.2 weights root")
    parser.add_argument("--dinov3-root", help="optional DINOv3 weights root to bundle and quantize")
    parser.add_argument("--rmbg-root", help="optional RMBG-2.0 weights root to bundle and quantize")
    parser.add_argument("--bits", type=int, default=8, choices=range(2, 9))
    parser.add_argument("--group-size", type=int, default=64, choices=sorted(_AFFINE_GROUP_SIZES))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    results = quantize_trellis2_weights(
        args.source,
        args.output,
        dinov3_root=args.dinov3_root,
        rmbg_root=args.rmbg_root,
        bits=args.bits,
        group_size=args.group_size,
        overwrite=args.overwrite,
        progress=lambda path: print(f"quantizing {path.as_posix()}", flush=True),
    )
    print(
        json.dumps(
            {
                "format": TRELLIS2_QUANTIZATION_FORMAT,
                "policy": TRELLIS2_QUANTIZATION_POLICY,
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
