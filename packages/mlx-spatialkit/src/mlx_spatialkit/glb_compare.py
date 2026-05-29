"""GLB inspection and visual-comparison helpers."""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GlbPayload:
    document: dict[str, Any]
    bin_blob: bytes


@dataclass(frozen=True)
class PngPixels:
    width: int
    height: int
    channels: int
    rows: list[bytes]


@dataclass(frozen=True)
class PngCoverage:
    width: int
    height: int
    channels: int
    pixel_count: int
    alpha_nonzero_count: int
    rgb_nonzero_count: int
    visible_rgb_nonzero_count: int

    @property
    def alpha_coverage_ratio(self) -> float:
        return self.alpha_nonzero_count / float(self.pixel_count)

    @property
    def rgb_coverage_ratio(self) -> float:
        return self.rgb_nonzero_count / float(self.pixel_count)

    @property
    def visible_rgb_coverage_ratio(self) -> float:
        return self.visible_rgb_nonzero_count / float(self.pixel_count)

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "pixel_count": self.pixel_count,
            "alpha_nonzero_count": self.alpha_nonzero_count,
            "rgb_nonzero_count": self.rgb_nonzero_count,
            "visible_rgb_nonzero_count": self.visible_rgb_nonzero_count,
            "alpha_coverage_ratio": self.alpha_coverage_ratio,
            "rgb_coverage_ratio": self.rgb_coverage_ratio,
            "visible_rgb_coverage_ratio": self.visible_rgb_coverage_ratio,
        }


def inspect_glb(path: str | Path) -> dict[str, Any]:
    """Inspect GLB mesh/material/image structure and embedded PNG coverage."""

    glb_path = Path(path)
    payload = parse_glb(glb_path.read_bytes())
    document = payload.document
    primitive_summaries = []
    total_faces = 0
    total_indices = 0
    total_vertices = 0
    for mesh_index, mesh in enumerate(document.get("meshes", [])):
        for primitive_index, primitive in enumerate(mesh.get("primitives", [])):
            attributes = primitive.get("attributes", {})
            position_accessor = _accessor(document, attributes.get("POSITION"))
            normal_accessor = _accessor(document, attributes.get("NORMAL"))
            texcoord_accessor = _accessor(document, attributes.get("TEXCOORD_0"))
            indices_accessor = _accessor(document, primitive.get("indices"))
            vertex_count = int(position_accessor.get("count", 0)) if position_accessor is not None else 0
            normal_count = int(normal_accessor.get("count", 0)) if normal_accessor is not None else 0
            texcoord_count = int(texcoord_accessor.get("count", 0)) if texcoord_accessor is not None else 0
            index_count = int(indices_accessor.get("count", 0)) if indices_accessor is not None else 0
            face_count = index_count // 3
            total_vertices += vertex_count
            total_indices += index_count
            total_faces += face_count
            primitive_summaries.append(
                {
                    "mesh_index": mesh_index,
                    "primitive_index": primitive_index,
                    "material": primitive.get("material"),
                    "vertex_count": vertex_count,
                    "normal_count": normal_count,
                    "texcoord_count": texcoord_count,
                    "index_count": index_count,
                    "face_count": face_count,
                    "indices_component_type": indices_accessor.get("componentType")
                    if indices_accessor is not None
                    else None,
                    "indices_min": indices_accessor.get("min") if indices_accessor is not None else None,
                    "indices_max": indices_accessor.get("max") if indices_accessor is not None else None,
                    "has_normals": normal_accessor is not None,
                    "attributes": sorted(str(name) for name in attributes),
                }
            )

    images = _image_summaries(document, payload.bin_blob)
    textures = document.get("textures", [])
    return {
        "path": str(glb_path),
        "bytes": int(glb_path.stat().st_size),
        "mesh_count": len(document.get("meshes", [])),
        "material_count": len(document.get("materials", [])),
        "texture_count": len(textures),
        "image_count": len(document.get("images", [])),
        "primitive_count": len(primitive_summaries),
        "total_vertices": total_vertices,
        "total_indices": total_indices,
        "total_faces": total_faces,
        "primitives": primitive_summaries,
        "images": images,
    }


def compare_textured_glbs(
    candidate_path: str | Path,
    reference_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compare a generated textured GLB against a reference textured GLB."""

    candidate = inspect_glb(candidate_path)
    reference = inspect_glb(reference_path)
    candidate_base = _image_by_name(candidate, "baseColorTexture")
    reference_base = _image_by_name(reference, "baseColorTexture")
    candidate_mr = _image_by_name(candidate, "metallicRoughnessTexture")
    reference_mr = _image_by_name(reference, "metallicRoughnessTexture")
    face_ratio = _ratio(candidate["total_faces"], reference["total_faces"])
    vertex_ratio = _ratio(candidate["total_vertices"], reference["total_vertices"])
    alpha_ratio = _coverage_ratio(candidate_base, reference_base, "alpha_coverage_ratio")
    rgb_ratio = _coverage_ratio(candidate_base, reference_base, "visible_rgb_coverage_ratio")
    raw_rgb_ratio = _coverage_ratio(candidate_base, reference_base, "rgb_coverage_ratio")
    candidate_base_stats = candidate_base.get("stats", {})
    reference_base_stats = reference_base.get("stats", {})
    candidate_mr_stats = candidate_mr.get("stats", {})
    reference_mr_stats = reference_mr.get("stats", {})
    candidate_dark_ratio = candidate_base_stats.get("visible_dark_rgb_ratio")
    reference_dark_ratio = reference_base_stats.get("visible_dark_rgb_ratio")
    candidate_gray_ratio = candidate_base_stats.get("visible_grayish_rgb_ratio")
    reference_gray_ratio = reference_base_stats.get("visible_grayish_rgb_ratio")
    roughness_mean_ratio = _ratio(
        candidate_mr_stats.get("channel_1_mean"),
        reference_mr_stats.get("channel_1_mean"),
    )
    candidate_roughness_low_ratio = candidate_mr_stats.get("channel_1_low_51_ratio")
    reference_roughness_low_ratio = reference_mr_stats.get("channel_1_low_51_ratio")
    texture_resolution_match = (
        candidate_base.get("coverage", {}).get("width") == reference_base.get("coverage", {}).get("width")
        and candidate_base.get("coverage", {}).get("height") == reference_base.get("coverage", {}).get("height")
    )
    dark_ratio_max = _relative_ratio_ceiling(reference_dark_ratio, absolute=0.15, delta=0.15)
    gray_ratio_max = _relative_ratio_ceiling(reference_gray_ratio, absolute=0.35, delta=0.25)
    roughness_low_max = _relative_ratio_ceiling(reference_roughness_low_ratio, absolute=0.05, delta=0.20)
    roughness_reference_mean = reference_mr_stats.get("channel_1_mean")
    checks = {
        "glb_parseable": {"passed": True, "required": True, "actual": True},
        "textured_mesh": {
            "passed": candidate["primitive_count"] > 0 and reference["primitive_count"] > 0,
            "candidate_primitives": candidate["primitive_count"],
            "reference_primitives": reference["primitive_count"],
            "required": "both_have_mesh_primitives",
        },
        "face_count_ratio": {
            "passed": face_ratio is not None and 0.80 <= face_ratio <= 1.25,
            "actual": face_ratio,
            "required_min": 0.80,
            "required_max": 1.25,
        },
        "texture_resolution_match": {
            "passed": texture_resolution_match,
            "candidate": _image_size(candidate_base),
            "reference": _image_size(reference_base),
            "required": "same_base_color_dimensions",
        },
        "base_color_alpha_coverage_ratio": {
            "passed": alpha_ratio is not None and alpha_ratio >= 0.95,
            "actual": alpha_ratio,
            "required_min": 0.95,
        },
        "base_color_rgb_coverage_ratio": {
            "passed": rgb_ratio is not None and rgb_ratio >= 0.95,
            "actual": rgb_ratio,
            "metric": "visible_rgb_coverage_ratio",
            "required_min": 0.95,
        },
        "base_color_visible_dark_ratio": {
            "passed": candidate_dark_ratio is not None
            and dark_ratio_max is not None
            and candidate_dark_ratio <= dark_ratio_max,
            "actual": candidate_dark_ratio,
            "reference": reference_dark_ratio,
            "required_max": dark_ratio_max,
        },
        "base_color_visible_grayish_ratio": {
            "passed": candidate_gray_ratio is not None
            and gray_ratio_max is not None
            and candidate_gray_ratio <= gray_ratio_max,
            "actual": candidate_gray_ratio,
            "reference": reference_gray_ratio,
            "required_max": gray_ratio_max,
        },
        "roughness_mean_ratio": {
            "passed": _passes_ratio_when_reference_present(roughness_mean_ratio, roughness_reference_mean, 0.75),
            "actual": roughness_mean_ratio,
            "candidate_mean": candidate_mr_stats.get("channel_1_mean"),
            "reference_mean": roughness_reference_mean,
            "required_min": 0.75,
            "channel": "metallicRoughnessTexture.G",
        },
        "roughness_low_ratio": {
            "passed": (
                _reference_absent(roughness_reference_mean)
                or (
                    candidate_roughness_low_ratio is not None
                    and roughness_low_max is not None
                    and candidate_roughness_low_ratio <= roughness_low_max
                )
            ),
            "actual": candidate_roughness_low_ratio,
            "reference": reference_roughness_low_ratio,
            "required_max": roughness_low_max,
            "channel": "metallicRoughnessTexture.G",
        },
    }
    report: dict[str, Any] = {
        "candidate": candidate,
        "reference": reference,
        "summary": {
            "all_passed": all(bool(check["passed"]) for check in checks.values()),
            "comparison_kind": "coarse_glb_structure_texture_heuristic",
            "spatial_proof_ready": False,
            "face_count_ratio": face_ratio,
            "vertex_count_ratio": vertex_ratio,
            "base_color_alpha_coverage_ratio": alpha_ratio,
            "base_color_rgb_coverage_ratio": rgb_ratio,
            "base_color_raw_rgb_coverage_ratio": raw_rgb_ratio,
            "base_color_visible_dark_ratio": candidate_dark_ratio,
            "base_color_reference_visible_dark_ratio": reference_dark_ratio,
            "base_color_visible_grayish_ratio": candidate_gray_ratio,
            "base_color_reference_visible_grayish_ratio": reference_gray_ratio,
            "roughness_mean_ratio": roughness_mean_ratio,
            "roughness_candidate_mean": candidate_mr_stats.get("channel_1_mean"),
            "roughness_reference_mean": roughness_reference_mean,
            "roughness_low_ratio": candidate_roughness_low_ratio,
            "roughness_reference_low_ratio": reference_roughness_low_ratio,
            "texture_resolution_match": texture_resolution_match,
        },
        "checks": checks,
        "comparison_scope": {
            "kind": "coarse_glb_structure_texture_heuristic",
            "spatial_proof": False,
            "viewer_render_proof": False,
            "texture_registration_proof": False,
            "reason": (
                "Checks compare GLB structure and embedded texture statistics only; they are not "
                "spatially registered UV, rendered-view, or per-surface proof."
            ),
        },
        "deferred_parity_boundaries": [
            "not_xatlas_chart_parity",
            "not_1m_face_export_setting_parity",
        ],
    }
    if output_dir is not None:
        _write_visual_report_artifacts(
            report,
            candidate_path=Path(candidate_path),
            reference_path=Path(reference_path),
            output_dir=Path(output_dir),
        )
    return report


def parse_glb(payload: bytes) -> GlbPayload:
    """Parse a GLB 2.0 payload into JSON and BIN chunks."""

    if len(payload) < 20:
        raise ValueError("GLB payload is too small")
    magic, version, total_length = struct.unpack_from("<III", payload, 0)
    if magic != 0x46546C67:
        raise ValueError("GLB magic is invalid")
    if version != 2:
        raise ValueError("only GLB version 2 is supported")
    if total_length != len(payload):
        raise ValueError("GLB total length does not match payload size")
    json_length, json_type = struct.unpack_from("<I4s", payload, 12)
    if json_type != b"JSON":
        raise ValueError("GLB first chunk must be JSON")
    json_start = 20
    json_end = json_start + json_length
    if json_end > len(payload):
        raise ValueError("GLB JSON chunk extends past payload")
    document = json.loads(payload[json_start:json_end].rstrip(b" ").decode("utf-8"))
    if json_end == len(payload):
        return GlbPayload(document=document, bin_blob=b"")
    if json_end + 8 > len(payload):
        raise ValueError("GLB BIN chunk header is truncated")
    bin_length, bin_type = struct.unpack_from("<I4s", payload, json_end)
    if bin_type != b"BIN\x00":
        raise ValueError("GLB second chunk must be BIN")
    bin_start = json_end + 8
    if bin_start + bin_length != len(payload):
        raise ValueError("GLB BIN chunk length does not match payload size")
    return GlbPayload(document=document, bin_blob=payload[bin_start:])


def glb_image_payload(payload: GlbPayload, image_index: int) -> bytes:
    """Return an embedded GLB image payload by image index."""

    images = payload.document.get("images", [])
    if image_index < 0 or image_index >= len(images):
        raise ValueError(f"GLB image index out of range: {image_index}")
    image = images[image_index]
    if image.get("mimeType") != "image/png":
        raise ValueError("only embedded PNG images are supported")
    view_index = image.get("bufferView")
    views = payload.document.get("bufferViews", [])
    if not isinstance(view_index, int) or view_index < 0 or view_index >= len(views):
        raise ValueError("GLB image bufferView is invalid")
    view = views[view_index]
    start = int(view.get("byteOffset", 0))
    end = start + int(view["byteLength"])
    if start < 0 or end > len(payload.bin_blob):
        raise ValueError("GLB image bufferView extends past BIN chunk")
    return payload.bin_blob[start:end]


def glb_named_image_payload(path: str | Path, image_name: str) -> bytes:
    """Return an embedded GLB PNG payload by image name."""

    payload = parse_glb(Path(path).read_bytes())
    for index, image in enumerate(payload.document.get("images", [])):
        if image.get("name") == image_name:
            return glb_image_payload(payload, index)
    raise ValueError(f"GLB image not found: {image_name}")


def png_coverage(png: bytes) -> PngCoverage:
    """Measure nonzero alpha/RGB coverage for 8-bit RGB or RGBA PNGs."""

    pixels = _decode_png(png)
    alpha_nonzero = 0
    rgb_nonzero = 0
    visible_rgb_nonzero = 0
    for row in pixels.rows:
        for pixel in range(0, len(row), pixels.channels):
            r = row[pixel]
            g = row[pixel + 1]
            b = row[pixel + 2]
            a = row[pixel + 3] if pixels.channels == 4 else 255
            if a:
                alpha_nonzero += 1
            if r or g or b:
                rgb_nonzero += 1
                if a:
                    visible_rgb_nonzero += 1
    return PngCoverage(
        width=pixels.width,
        height=pixels.height,
        channels=pixels.channels,
        pixel_count=pixels.width * pixels.height,
        alpha_nonzero_count=alpha_nonzero,
        rgb_nonzero_count=rgb_nonzero,
        visible_rgb_nonzero_count=visible_rgb_nonzero,
    )


def _decode_png(png: bytes) -> PngPixels:
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("PNG signature is invalid")
    pos = 8
    width = 0
    height = 0
    color_type = -1
    idat = bytearray()
    while pos < len(png):
        if pos + 8 > len(png):
            raise ValueError("PNG chunk header is truncated")
        length = struct.unpack(">I", png[pos : pos + 4])[0]
        pos += 4
        chunk_type = png[pos : pos + 4]
        pos += 4
        chunk = png[pos : pos + length]
        pos += length
        if pos + 4 > len(png):
            raise ValueError("PNG chunk crc is truncated")
        pos += 4
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
            if bit_depth != 8 or color_type not in (2, 6) or compression != 0 or filter_method != 0 or interlace != 0:
                raise ValueError("only non-interlaced 8-bit RGB/RGBA PNGs are supported")
        elif chunk_type == b"IDAT":
            idat.extend(chunk)
        elif chunk_type == b"IEND":
            break
    if width <= 0 or height <= 0:
        raise ValueError("PNG IHDR is missing")
    channels = 4 if color_type == 6 else 3
    rows = _decode_png_rows(zlib.decompress(bytes(idat)), width, height, channels)
    return PngPixels(
        width=width,
        height=height,
        channels=channels,
        rows=rows,
    )


def _png_stats(png: bytes) -> dict[str, Any]:
    pixels = _decode_png(png)
    sums = [0.0] * pixels.channels
    low_51 = [0] * pixels.channels
    visible_rgb_sum = [0.0, 0.0, 0.0]
    visible_count = 0
    visible_dark = 0
    visible_grayish = 0
    pixel_count = pixels.width * pixels.height
    for row in pixels.rows:
        for pixel in range(0, len(row), pixels.channels):
            values = [row[pixel + channel] for channel in range(pixels.channels)]
            for channel, value in enumerate(values):
                sums[channel] += float(value)
                if value < 51:
                    low_51[channel] += 1
            alpha = values[3] if pixels.channels == 4 else 255
            if alpha:
                rgb = values[:3]
                visible_count += 1
                for channel in range(3):
                    visible_rgb_sum[channel] += float(rgb[channel])
                if sum(rgb) / 3.0 < 16.0:
                    visible_dark += 1
                if max(rgb) - min(rgb) <= 16:
                    visible_grayish += 1
    result: dict[str, Any] = {
        "visible_texel_count": visible_count,
        "visible_dark_rgb_ratio": _ratio(visible_dark, visible_count) if visible_count else None,
        "visible_grayish_rgb_ratio": _ratio(visible_grayish, visible_count) if visible_count else None,
    }
    if visible_count:
        result["visible_rgb_mean"] = [value / float(visible_count) for value in visible_rgb_sum]
    else:
        result["visible_rgb_mean"] = None
    for channel, total in enumerate(sums):
        result[f"channel_{channel}_mean"] = total / float(pixel_count)
        result[f"channel_{channel}_low_51_ratio"] = low_51[channel] / float(pixel_count)
    return result


def _decode_png_rows(raw: bytes, width: int, height: int, channels: int) -> list[bytes]:
    row_bytes = width * channels
    rows: list[bytes] = []
    offset = 0
    prior = bytes(row_bytes)
    for _ in range(height):
        if offset >= len(raw):
            raise ValueError("PNG decompressed data ended early")
        filter_type = raw[offset]
        offset += 1
        row = bytearray(raw[offset : offset + row_bytes])
        if len(row) != row_bytes:
            raise ValueError("PNG row is truncated")
        offset += row_bytes
        _unfilter_row(filter_type, row, prior, channels)
        current = bytes(row)
        rows.append(current)
        prior = current
    return rows


def _unfilter_row(filter_type: int, row: bytearray, prior: bytes, bpp: int) -> None:
    if filter_type == 0:
        return
    if filter_type == 1:
        for idx, value in enumerate(row):
            left = row[idx - bpp] if idx >= bpp else 0
            row[idx] = (value + left) & 0xFF
        return
    if filter_type == 2:
        for idx, value in enumerate(row):
            row[idx] = (value + prior[idx]) & 0xFF
        return
    if filter_type == 3:
        for idx, value in enumerate(row):
            left = row[idx - bpp] if idx >= bpp else 0
            up = prior[idx]
            row[idx] = (value + ((left + up) >> 1)) & 0xFF
        return
    if filter_type == 4:
        for idx, value in enumerate(row):
            left = row[idx - bpp] if idx >= bpp else 0
            up = prior[idx]
            up_left = prior[idx - bpp] if idx >= bpp else 0
            row[idx] = (value + _paeth(left, up, up_left)) & 0xFF
        return
    raise ValueError(f"unsupported PNG filter type: {filter_type}")


def _paeth(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    left_distance = abs(estimate - left)
    up_distance = abs(estimate - up)
    up_left_distance = abs(estimate - up_left)
    if left_distance <= up_distance and left_distance <= up_left_distance:
        return left
    if up_distance <= up_left_distance:
        return up
    return up_left


def _accessor(document: dict[str, Any], accessor_index: Any) -> dict[str, Any] | None:
    if not isinstance(accessor_index, int):
        return None
    accessors = document.get("accessors", [])
    if accessor_index < 0 or accessor_index >= len(accessors):
        raise ValueError(f"GLB accessor index out of range: {accessor_index}")
    return accessors[accessor_index]


def _write_visual_report_artifacts(
    report: dict[str, Any],
    *,
    candidate_path: Path,
    reference_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_png = output_dir / "candidate_base_color.png"
    reference_png = output_dir / "reference_base_color.png"
    candidate_png.write_bytes(glb_named_image_payload(candidate_path, "baseColorTexture"))
    reference_png.write_bytes(glb_named_image_payload(reference_path, "baseColorTexture"))
    report_path = output_dir / "visual_parity.json"
    html_path = output_dir / "index.html"
    report["artifacts"] = {
        "report_json": str(report_path),
        "preview_html": str(html_path),
        "candidate_base_color_png": str(candidate_png),
        "reference_base_color_png": str(reference_png),
    }
    _write_json_atomic(report_path, report)
    html_path.write_text(_visual_report_html(report), encoding="utf-8")


def _visual_report_html(report: dict[str, Any]) -> str:
    summary = report["summary"]
    artifacts = report["artifacts"]
    status = "PASS" if summary["all_passed"] else "FAIL"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>mlx-spatialkit visual parity report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #151515; }}
    main {{ max-width: 1080px; margin: 0 auto; }}
    .textures {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    img {{ width: 100%; image-rendering: auto; border: 1px solid #bbb; background: #eee; }}
    code {{ background: #f1f1f1; padding: 2px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
<main>
  <h1>mlx-spatialkit Visual Parity: {status}</h1>
  <p>Face ratio: <code>{summary["face_count_ratio"]:.6f}</code></p>
  <p>Base-color alpha coverage ratio: <code>{summary["base_color_alpha_coverage_ratio"]:.6f}</code></p>
  <p>Base-color visible RGB coverage ratio: <code>{summary["base_color_rgb_coverage_ratio"]:.6f}</code></p>
  <p>Base-color raw RGB footprint ratio: <code>{summary["base_color_raw_rgb_coverage_ratio"]:.6f}</code></p>
  <section class="textures">
    <figure>
      <img src="{escape(Path(artifacts["candidate_base_color_png"]).name)}" alt="candidate base color texture">
      <figcaption>candidate base color</figcaption>
    </figure>
    <figure>
      <img src="{escape(Path(artifacts["reference_base_color_png"]).name)}" alt="reference base color texture">
      <figcaption>reference base color</figcaption>
    </figure>
  </section>
  <p>Detailed machine-readable report: <a href="{escape(Path(artifacts["report_json"]).name)}">visual_parity.json</a></p>
</main>
</body>
</html>
"""


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _image_by_name(summary: dict[str, Any], name: str) -> dict[str, Any]:
    for image in summary.get("images", []):
        if image.get("name") == name:
            return image
    raise ValueError(f"GLB image summary not found: {name}")


def _ratio(numerator: Any, denominator: Any) -> float | None:
    if denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _coverage_ratio(candidate_image: dict[str, Any], reference_image: dict[str, Any], key: str) -> float | None:
    candidate = candidate_image.get("coverage", {}).get(key)
    reference = reference_image.get("coverage", {}).get(key)
    return _ratio(candidate, reference)


def _reference_absent(reference_value: Any) -> bool:
    return reference_value in (None, 0, 0.0)


def _passes_ratio_when_reference_present(
    actual_ratio: float | None,
    reference_value: Any,
    required_min: float,
) -> bool:
    if _reference_absent(reference_value):
        return True
    return actual_ratio is not None and actual_ratio >= required_min


def _relative_ratio_ceiling(reference_ratio: Any, *, absolute: float, delta: float) -> float | None:
    if reference_ratio is None:
        return None
    return max(absolute, float(reference_ratio) + delta)


def _image_size(image: dict[str, Any]) -> dict[str, int | None]:
    coverage = image.get("coverage", {})
    return {
        "width": coverage.get("width"),
        "height": coverage.get("height"),
    }


def _image_summaries(document: dict[str, Any], bin_blob: bytes) -> list[dict[str, Any]]:
    payload = GlbPayload(document=document, bin_blob=bin_blob)
    images = []
    for index, image in enumerate(document.get("images", [])):
        summary: dict[str, Any] = {
            "index": index,
            "name": image.get("name"),
            "mime_type": image.get("mimeType"),
            "buffer_view": image.get("bufferView"),
        }
        if image.get("mimeType") == "image/png" and isinstance(image.get("bufferView"), int):
            image_payload = glb_image_payload(payload, index)
            coverage = png_coverage(image_payload)
            summary["coverage"] = coverage.as_dict()
            summary["stats"] = _png_stats(image_payload)
        images.append(summary)
    return images
