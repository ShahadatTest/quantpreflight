"""Zero-weight GGUF v2/v3 binary metadata and tensor-info parser."""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from typing import Any, Callable

GGUF_MAGIC = b"GGUF"
MAX_GGUF_HEADER_BYTES = 32 * 1024 * 1024  # 32 MiB ceiling for huge vocab GGUF headers
MAX_GGUF_TENSORS = 200_000
MAX_GGUF_KV_COUNT = 100_000
MAX_GGUF_ARRAY_LEN = 1_000_000
MAX_GGUF_STRING_BYTES = 4 * 1024 * 1024
MAX_GGUF_SPLIT_PARTS = 512


class GGUFFormatError(ValueError):
    """Raised when a GGUF header is malformed, unsupported, or exceeds safety limits."""


# (type_name, block_elements, block_bytes, logical_bits_per_weight)
GGML_TYPES: dict[int, tuple[str, int, int, float]] = {
    0: ("F32", 1, 4, 32.0),
    1: ("F16", 1, 2, 16.0),
    2: ("Q4_0", 32, 18, 4.0),
    3: ("Q4_1", 32, 20, 4.0),
    6: ("Q5_0", 32, 22, 5.0),
    7: ("Q5_1", 32, 24, 5.0),
    8: ("Q8_0", 32, 34, 8.0),
    9: ("Q8_1", 32, 36, 8.0),
    10: ("Q2_K", 256, 84, 2.0),
    11: ("Q3_K", 256, 110, 3.0),
    12: ("Q4_K", 256, 144, 4.0),
    13: ("Q5_K", 256, 176, 5.0),
    14: ("Q6_K", 256, 210, 6.0),
    15: ("Q8_K", 256, 292, 8.0),
    16: ("IQ2_XXS", 256, 66, 2.0),
    17: ("IQ2_XS", 256, 74, 2.0),
    18: ("IQ3_XXS", 256, 98, 3.0),
    19: ("IQ1_S", 256, 50, 1.58),
    20: ("IQ4_NL", 32, 18, 4.0),
    21: ("IQ3_S", 256, 110, 3.0),
    22: ("IQ2_S", 256, 82, 2.0),
    23: ("IQ4_XS", 256, 136, 4.0),
    24: ("I8", 1, 1, 8.0),
    25: ("I16", 1, 2, 16.0),
    26: ("I32", 1, 4, 32.0),
    27: ("I64", 1, 8, 64.0),
    28: ("F64", 1, 8, 64.0),
    29: ("IQ1_M", 256, 56, 1.75),
    30: ("BF16", 1, 2, 16.0),
    34: ("TQ1_0", 256, 54, 1.58),
    35: ("TQ2_0", 256, 66, 2.0),
    39: ("MXFP4", 32, 17, 4.0),
}

GGUF_FILE_TYPE_NAMES: dict[int, str] = {
    0: "ALL_F32",
    1: "MOSTLY_F16",
    2: "MOSTLY_Q4_0",
    3: "MOSTLY_Q4_1",
    7: "MOSTLY_Q8_0",
    8: "MOSTLY_Q5_0",
    9: "MOSTLY_Q5_1",
    10: "MOSTLY_Q2_K",
    11: "MOSTLY_Q3_K_S",
    12: "MOSTLY_Q3_K_M",
    13: "MOSTLY_Q3_K_L",
    14: "MOSTLY_Q4_K_S",
    15: "MOSTLY_Q4_K_M",
    16: "MOSTLY_Q5_K_S",
    17: "MOSTLY_Q5_K_M",
    18: "MOSTLY_Q6_K",
    19: "MOSTLY_IQ2_XXS",
    20: "MOSTLY_IQ2_XS",
    21: "MOSTLY_Q2_K_S",
    22: "MOSTLY_IQ3_XS",
    23: "MOSTLY_IQ3_XXS",
    24: "MOSTLY_IQ1_S",
    25: "MOSTLY_IQ4_NL",
    26: "MOSTLY_IQ3_S",
    27: "MOSTLY_IQ3_M",
    28: "MOSTLY_IQ2_S",
    29: "MOSTLY_IQ2_M",
    30: "MOSTLY_IQ4_XS",
    31: "MOSTLY_IQ1_M",
    32: "MOSTLY_BF16",
    36: "MOSTLY_TQ1_0",
    37: "MOSTLY_TQ2_0",
}

_SCALAR_FORMATS: dict[int, tuple[str, int]] = {
    0: ("<B", 1),
    1: ("<b", 1),
    2: ("<H", 2),
    3: ("<h", 2),
    4: ("<I", 4),
    5: ("<i", 4),
    6: ("<f", 4),
    7: ("<?", 1),
    10: ("<Q", 8),
    11: ("<q", 8),
    12: ("<d", 8),
}


@dataclass(frozen=True)
class RawGGUFTensorEntry:
    """Header-only representation of a GGUF tensor."""

    name: str
    ggml_type_id: int
    dtype_name: str
    shape: tuple[int, ...]
    offset: int
    byte_size: int
    effective_bits: float


@dataclass(frozen=True)
class ParsedGGUFHeader:
    """Complete parsed GGUF v2/v3 metadata and tensor table."""

    version: int
    metadata: dict[str, Any]
    tensors: dict[str, RawGGUFTensorEntry]
    header_bytes_consumed: int


class _AdaptiveRangeReader:
    """Streams contiguous bytes from a callback using expanding window fetches."""

    def __init__(self, fetch_range: Callable[[int, int], bytes], initial_window: int = 262144) -> None:
        self._fetch_range = fetch_range
        self._buf = bytearray()
        self._pos = 0
        self._next_window = initial_window
        self._eof = False

    @property
    def bytes_fetched(self) -> int:
        return len(self._buf)

    @property
    def position(self) -> int:
        return self._pos

    def read_exact(self, n: int) -> bytes:
        if n < 0:
            raise GGUFFormatError("Negative read length.")
        if self._pos + n > MAX_GGUF_HEADER_BYTES:
            raise GGUFFormatError(
                f"GGUF header exceeded safety ceiling of {MAX_GGUF_HEADER_BYTES} bytes."
            )
        while len(self._buf) < self._pos + n and not self._eof:
            need = (self._pos + n) - len(self._buf)
            chunk_len = max(self._next_window, need)
            start = len(self._buf)
            end = min(start + chunk_len, MAX_GGUF_HEADER_BYTES) - 1
            chunk = self._fetch_range(start, end)
            if not chunk:
                self._eof = True
                break
            self._buf.extend(chunk)
            if len(chunk) < (end - start + 1):
                self._eof = True
            self._next_window = min(self._next_window * 2, 4 * 1024 * 1024)

        if len(self._buf) < self._pos + n:
            raise GGUFFormatError(
                f"Unexpected EOF in GGUF header at byte {self._pos} (needed {n} more bytes)."
            )
        out = bytes(self._buf[self._pos : self._pos + n])
        self._pos += n
        return out

    def unpack(self, fmt: str, size: int) -> tuple[Any, ...]:
        return struct.unpack(fmt, self.read_exact(size))

    def read_string(self) -> str:
        (slen,) = self.unpack("<Q", 8)
        if slen > MAX_GGUF_STRING_BYTES:
            raise GGUFFormatError(f"GGUF string length {slen} exceeds safety limit.")
        raw = self.read_exact(slen)
        return raw.decode("utf-8", errors="replace")

    def read_value(self, vtype: int, depth: int = 0) -> Any:
        if depth > 4:
            raise GGUFFormatError("GGUF nested array depth > 4 is not permitted.")
        if vtype in _SCALAR_FORMATS:
            fmt, sz = _SCALAR_FORMATS[vtype]
            return self.unpack(fmt, sz)[0]
        if vtype == 8:
            return self.read_string()
        if vtype == 9:
            elem_type, count = self.unpack("<IQ", 12)
            if count > MAX_GGUF_ARRAY_LEN:
                raise GGUFFormatError(f"GGUF array length {count} exceeds limit.")
            if elem_type in _SCALAR_FORMATS:
                fmt_char = _SCALAR_FORMATS[elem_type][0][1:]
                sz = _SCALAR_FORMATS[elem_type][1]
                raw = self.read_exact(sz * count)
                if count <= 32:
                    return list(struct.unpack(f"<{count}{fmt_char}", raw))
                # Summarize large numeric token-score/type tables without building giant Python lists
                return {"__gguf_array_len__": count, "elem_type": elem_type}
            if elem_type == 8:
                # Keep token string table count and boundary strings for vocab bounds verification
                last_token: str | None = None
                sample_head: list[str] = []
                for idx in range(count):
                    s = self.read_string()
                    if idx < 8:
                        sample_head.append(s)
                    last_token = s
                return {
                    "__gguf_string_table_len__": count,
                    "head": sample_head,
                    "last": last_token,
                }
            return [self.read_value(elem_type, depth + 1) for _ in range(count)]
        raise GGUFFormatError(f"Unknown GGUF metadata value type: {vtype}")


def compute_ggml_tensor_bytes(ggml_type_id: int, dims: tuple[int, ...]) -> tuple[str, int, float]:
    """Compute physical byte footprint and logical bit-width for a GGUF tensor."""
    # A type newer than this table shouldn't stop the scan, so assume one byte per element
    fallback = (f"GGML_TYPE_{ggml_type_id}", 1, 1, 8.0)
    type_name, block_elems, block_bytes, logical_bits = GGML_TYPES.get(ggml_type_id, fallback)
    numel = 1
    for d in dims:
        numel *= d
    if numel == 0:
        return type_name, 0, logical_bits
    blocks = (numel + block_elems - 1) // block_elems
    return type_name, blocks * block_bytes, logical_bits


def split_gguf_names(name: str) -> list[str]:
    """Return every part of a split GGUF file name, or just `name` when it isn't split."""
    match = re.fullmatch(r"(.*-)(\d{5})-of-(\d{5})(\.gguf)", name, re.IGNORECASE)
    if match is None:
        return [name]
    prefix, _, total, suffix = match.groups()
    if not 1 <= int(total) <= MAX_GGUF_SPLIT_PARTS:
        return [name]
    return [f"{prefix}{part:05d}-of-{total}{suffix}" for part in range(1, int(total) + 1)]


def merge_gguf_parts(parts: list[ParsedGGUFHeader]) -> ParsedGGUFHeader:
    """Combine the headers of a split GGUF file. Metadata lives in the first part."""
    tensors: dict[str, RawGGUFTensorEntry] = {}
    for part in parts:
        tensors.update(part.tensors)
    return ParsedGGUFHeader(
        version=parts[0].version,
        metadata=parts[0].metadata,
        tensors=tensors,
        header_bytes_consumed=sum(part.header_bytes_consumed for part in parts),
    )


def read_gguf_header_via_range(fetch_range: Callable[[int, int], bytes]) -> ParsedGGUFHeader:
    """Parse GGUF v2/v3 metadata KV pairs and tensor headers without reading weight data."""
    reader = _AdaptiveRangeReader(fetch_range)
    magic = reader.read_exact(4)
    if magic != GGUF_MAGIC:
        raise GGUFFormatError(f"Invalid GGUF magic {magic!r}; expected b'GGUF'.")

    (version,) = reader.unpack("<I", 4)
    if version not in (2, 3):
        raise GGUFFormatError(f"Unsupported GGUF version {version}; supported versions are 2 and 3.")

    tensor_count, kv_count = reader.unpack("<QQ", 16)
    if tensor_count > MAX_GGUF_TENSORS:
        raise GGUFFormatError(f"GGUF tensor_count {tensor_count} exceeds limit.")
    if kv_count > MAX_GGUF_KV_COUNT:
        raise GGUFFormatError(f"GGUF kv_count {kv_count} exceeds limit.")

    metadata: dict[str, Any] = {}
    for _ in range(kv_count):
        key = reader.read_string()
        (vtype,) = reader.unpack("<I", 4)
        metadata[key] = reader.read_value(vtype)

    tensors: dict[str, RawGGUFTensorEntry] = {}
    for _ in range(tensor_count):
        tname = reader.read_string()
        (n_dims,) = reader.unpack("<I", 4)
        if n_dims > 8:
            raise GGUFFormatError(f"GGUF tensor {tname!r} has invalid rank {n_dims}.")
        ggml_dims = tuple(reader.unpack(f"<{n_dims}Q", 8 * n_dims))
        # GGML stores ne[0] = inner (in_features), ne[1] = outer (out_features).
        # Reverse 2D+ shapes to PyTorch convention [out_features, in_features] for unified analysis.
        pytorch_shape = tuple(reversed(ggml_dims)) if len(ggml_dims) >= 2 else ggml_dims
        ggml_type_id, offset = reader.unpack("<IQ", 12)
        dtype_name, byte_size, logical_bits = compute_ggml_tensor_bytes(ggml_type_id, ggml_dims)
        tensors[tname] = RawGGUFTensorEntry(
            name=tname,
            ggml_type_id=ggml_type_id,
            dtype_name=dtype_name,
            shape=pytorch_shape,
            offset=offset,
            byte_size=byte_size,
            effective_bits=logical_bits,
        )

    return ParsedGGUFHeader(
        version=version,
        metadata=metadata,
        tensors=tensors,
        header_bytes_consumed=reader.bytes_fetched,
    )
