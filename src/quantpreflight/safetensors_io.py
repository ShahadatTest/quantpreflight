"""Zero-weight Safetensors binary header and shard-index parser."""

from __future__ import annotations

import json
import posixpath
import struct
from dataclasses import dataclass
from typing import Any, Callable

MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024  # 64 MiB safety ceiling

SAFETENSORS_DTYPE_BITS: dict[str, float] = {
    "F64": 64.0,
    "I64": 64.0,
    "U64": 64.0,
    "F32": 32.0,
    "I32": 32.0,
    "U32": 32.0,
    "F16": 16.0,
    "BF16": 16.0,
    "I16": 16.0,
    "U16": 16.0,
    "F8_E4M3": 8.0,
    "F8_E5M2": 8.0,
    "F8_E8M0": 8.0,
    "I8": 8.0,
    "U8": 8.0,
    "F4": 4.0,
    "BOOL": 8.0,
}


class SafetensorsFormatError(ValueError):
    """Raised when a Safetensors header or shard index is malformed or unsafe."""


@dataclass(frozen=True)
class RawSafetensorEntry:
    """Parsed entry from a single `.safetensors` JSON header."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    byte_size: int


@dataclass(frozen=True)
class ParsedSafetensorsHeader:
    """Complete parsed result of a single `.safetensors` file header."""

    tensors: dict[str, RawSafetensorEntry]
    metadata: dict[str, str]
    header_byte_length: int


def validate_shard_filename(shard_name: str) -> str:
    """Ensure a shard filename from `index.json` stays confined to the model root."""
    if not shard_name or not isinstance(shard_name, str):
        raise SafetensorsFormatError("Empty or non-string shard filename in index.json.")
    if "://" in shard_name or shard_name.startswith(("/", "\\")):
        raise SafetensorsFormatError(f"Unsafe non-relative shard path: {shard_name!r}")
    if ":" in shard_name:
        raise SafetensorsFormatError(f"Drive or scheme prefix not allowed in shard path: {shard_name!r}")

    normalized = shard_name.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    if not parts or any(p in (".", "..") for p in parts):
        raise SafetensorsFormatError(f"Path traversal segment in shard path: {shard_name!r}")

    clean = posixpath.join(*parts)
    if not clean.endswith(".safetensors"):
        raise SafetensorsFormatError(f"Shard path does not end with .safetensors: {shard_name!r}")
    return clean


def parse_safetensors_header_bytes(length_prefix: bytes, header_bytes: bytes) -> ParsedSafetensorsHeader:
    """Parse an 8-byte little-endian prefix and its UTF-8 JSON header buffer."""
    if len(length_prefix) != 8:
        raise SafetensorsFormatError(
            f"Expected 8-byte Safetensors header prefix, got {len(length_prefix)} bytes."
        )
    (header_len,) = struct.unpack("<Q", length_prefix)
    if header_len == 0:
        raise SafetensorsFormatError("Safetensors header length cannot be 0.")
    if header_len > MAX_SAFETENSORS_HEADER_BYTES:
        raise SafetensorsFormatError(
            f"Safetensors header length {header_len} exceeds safety ceiling ({MAX_SAFETENSORS_HEADER_BYTES} bytes)."
        )
    if len(header_bytes) != header_len:
        raise SafetensorsFormatError(
            f"Truncated Safetensors header: expected {header_len} bytes, got {len(header_bytes)}."
        )

    try:
        raw_obj = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsFormatError(f"Invalid UTF-8 JSON in Safetensors header: {exc}") from exc

    if not isinstance(raw_obj, dict):
        raise SafetensorsFormatError("Safetensors JSON header root must be an object.")

    metadata: dict[str, str] = {}
    tensors: dict[str, RawSafetensorEntry] = {}

    for key, val in raw_obj.items():
        if key == "__metadata__":
            if isinstance(val, dict):
                metadata = {str(k): str(v) for k, v in val.items()}
            continue

        if not isinstance(val, dict):
            raise SafetensorsFormatError(f"Tensor entry {key!r} must be a JSON object.")

        dtype = val.get("dtype")
        shape_raw = val.get("shape")
        offsets_raw = val.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in SAFETENSORS_DTYPE_BITS:
            raise SafetensorsFormatError(f"Unsupported or missing Safetensors dtype {dtype!r} on {key!r}.")
        if not isinstance(shape_raw, list) or not all(isinstance(d, int) and d >= 0 for d in shape_raw):
            raise SafetensorsFormatError(f"Invalid shape {shape_raw!r} on tensor {key!r}.")
        if (
            not isinstance(offsets_raw, list)
            or len(offsets_raw) != 2
            or not all(isinstance(o, int) and o >= 0 for o in offsets_raw)
            or offsets_raw[1] < offsets_raw[0]
        ):
            raise SafetensorsFormatError(f"Invalid data_offsets {offsets_raw!r} on tensor {key!r}.")

        shape = tuple(shape_raw)
        start, end = offsets_raw[0], offsets_raw[1]
        numel = 1
        for dim in shape:
            numel *= dim
        expected_bits = numel * SAFETENSORS_DTYPE_BITS[dtype]
        expected_bytes = int((expected_bits + 7) // 8)
        byte_size = (end - start) if (end > start) else expected_bytes

        tensors[key] = RawSafetensorEntry(
            name=key,
            dtype=dtype,
            shape=shape,
            data_offsets=(start, end),
            byte_size=byte_size,
        )

    return ParsedSafetensorsHeader(
        tensors=tensors,
        metadata=metadata,
        header_byte_length=8 + header_len,
    )


def read_safetensors_header_via_range(
    fetch_range: Callable[[int, int], bytes],
    initial_chunk_bytes: int = 65536,
) -> ParsedSafetensorsHeader:
    """Read a Safetensors header in 1 roundtrip when <=64 KiB, or 2 roundtrips if larger."""
    first = fetch_range(0, initial_chunk_bytes - 1)
    if len(first) < 8:
        raise SafetensorsFormatError(
            f"Safetensors resource shorter than 8 bytes ({len(first)} bytes received)."
        )
    (header_len,) = struct.unpack("<Q", first[:8])
    if header_len == 0 or header_len > MAX_SAFETENSORS_HEADER_BYTES:
        raise SafetensorsFormatError(
            f"Invalid or oversized Safetensors header length: {header_len} bytes."
        )

    needed_total = 8 + header_len
    if len(first) >= needed_total:
        return parse_safetensors_header_bytes(first[:8], first[8:needed_total])

    rest = fetch_range(len(first), needed_total - 1)
    combined = first + rest
    if len(combined) < needed_total:
        raise SafetensorsFormatError(
            f"Incomplete Safetensors header fetch: expected {needed_total} bytes, got {len(combined)}."
        )
    return parse_safetensors_header_bytes(combined[:8], combined[8:needed_total])


def parse_safetensors_index(index_obj: dict[str, Any]) -> tuple[dict[str, str], int | None]:
    """Validate `model.safetensors.index.json` and return `(weight_map, total_size)`."""
    if not isinstance(index_obj, dict):
        raise SafetensorsFormatError("model.safetensors.index.json must be a JSON object.")
    weight_map_raw = index_obj.get("weight_map")
    if not isinstance(weight_map_raw, dict) or not weight_map_raw:
        raise SafetensorsFormatError("model.safetensors.index.json missing non-empty 'weight_map'.")

    clean_map: dict[str, str] = {}
    for tensor_name, shard_file in weight_map_raw.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise SafetensorsFormatError("Empty tensor name in weight_map.")
        clean_map[tensor_name] = validate_shard_filename(shard_file)

    metadata = index_obj.get("metadata")
    total_size: int | None = None
    if isinstance(metadata, dict) and isinstance(metadata.get("total_size"), int):
        total_size = metadata["total_size"]

    return clean_map, total_size
