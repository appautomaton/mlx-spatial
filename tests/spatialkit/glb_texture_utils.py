from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass


@dataclass(frozen=True)
class PngCoverage:
    width: int
    height: int
    channels: int
    pixel_count: int
    alpha_nonzero_count: int
    rgb_nonzero_count: int

    @property
    def alpha_coverage_ratio(self) -> float:
        return self.alpha_nonzero_count / float(self.pixel_count)

    @property
    def rgb_coverage_ratio(self) -> float:
        return self.rgb_nonzero_count / float(self.pixel_count)


def glb_document_and_bin(payload: bytes) -> tuple[dict, bytes]:
    magic, version, total_length = struct.unpack_from("<III", payload, 0)
    assert magic == 0x46546C67
    assert version == 2
    assert total_length == len(payload)
    json_length, json_type = struct.unpack_from("<I4s", payload, 12)
    assert json_type == b"JSON"
    json_start = 20
    json_end = json_start + json_length
    document = json.loads(payload[json_start:json_end].rstrip(b" ").decode("utf-8"))
    bin_length, bin_type = struct.unpack_from("<I4s", payload, json_end)
    assert bin_type == b"BIN\x00"
    bin_start = json_end + 8
    assert bin_start + bin_length == len(payload)
    return document, payload[bin_start:]


def glb_image_payload(payload: bytes, image_name: str) -> bytes:
    document, bin_blob = glb_document_and_bin(payload)
    for image in document["images"]:
        if image.get("name") != image_name:
            continue
        view = document["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        end = start + view["byteLength"]
        return bin_blob[start:end]
    raise AssertionError(f"GLB image not found: {image_name}")


def png_coverage(png: bytes) -> PngCoverage:
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pos = 8
    width = 0
    height = 0
    color_type = -1
    idat = bytearray()
    while pos < len(png):
        length = struct.unpack(">I", png[pos : pos + 4])[0]
        pos += 4
        chunk_type = png[pos : pos + 4]
        pos += 4
        payload = png[pos : pos + length]
        pos += length + 4
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, *_ = struct.unpack(">IIBBBBB", payload)
            assert bit_depth == 8
            assert color_type in (2, 6)
        elif chunk_type == b"IDAT":
            idat.extend(payload)
        elif chunk_type == b"IEND":
            break
    channels = 4 if color_type == 6 else 3
    raw = zlib.decompress(bytes(idat))
    row_bytes = width * channels
    alpha_nonzero = 0
    rgb_nonzero = 0
    offset = 0
    for _ in range(height):
        filter_type = raw[offset]
        offset += 1
        assert filter_type == 0
        row = raw[offset : offset + row_bytes]
        offset += row_bytes
        for pixel in range(0, len(row), channels):
            r = row[pixel]
            g = row[pixel + 1]
            b = row[pixel + 2]
            a = row[pixel + 3] if channels == 4 else 255
            if a:
                alpha_nonzero += 1
            if r or g or b:
                rgb_nonzero += 1
    return PngCoverage(
        width=width,
        height=height,
        channels=channels,
        pixel_count=width * height,
        alpha_nonzero_count=alpha_nonzero,
        rgb_nonzero_count=rgb_nonzero,
    )
