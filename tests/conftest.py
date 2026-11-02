"""Shared binary fixture builders for Safetensors and GGUF header tests."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any


def build_safetensors_header_file(
    tensors: dict[str, tuple[str, list[int]]],
    metadata: dict[str, str] | None = None,
) -> bytes:
    """Build a valid header-only Safetensors binary envelope (zero weight payload needed)."""
    header_obj: dict[str, Any] = {}
    if metadata is not None:
        header_obj["__metadata__"] = metadata

    offset = 0
    dtype_bytes = {
        "F64": 8,
        "I64": 8,
        "F32": 4,
        "I32": 4,
        "U32": 4,
        "F16": 2,
        "BF16": 2,
        "I16": 2,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "I8": 1,
        "U8": 1,
    }
    for name, (dtype, shape) in tensors.items():
        numel = 1
        for d in shape:
            numel *= d
        bsize = numel * dtype_bytes.get(dtype, 2)
        header_obj[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + bsize],
        }
        offset += bsize

    raw_json = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(raw_json)) + raw_json


def _pack_gguf_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def build_gguf_header_bytes(
    *,
    version: int = 3,
    kv_strings: dict[str, str] | None = None,
    kv_u32: dict[str, int] | None = None,
    tokens: list[str] | None = None,
    tensors: list[tuple[str, tuple[int, ...], int]] | None = None,
) -> bytes:
    """Build a spec-compliant GGUF v2/v3 header buffer for offline testing."""
    kv_parts: list[bytes] = []
    for k, v in (kv_strings or {}).items():
        kv_parts.append(_pack_gguf_str(k) + struct.pack("<I", 8) + _pack_gguf_str(v))
    for k, v in (kv_u32 or {}).items():
        kv_parts.append(_pack_gguf_str(k) + struct.pack("<II", 4, v))
    if tokens is not None:
        arr_hdr = _pack_gguf_str("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, len(tokens))
        arr_body = b"".join(_pack_gguf_str(t) for t in tokens)
        kv_parts.append(arr_hdr + arr_body)

    t_list = tensors or []
    t_parts: list[bytes] = []
    cur_off = 0
    for tname, ggml_dims, ggml_type in t_list:
        part = _pack_gguf_str(tname) + struct.pack("<I", len(ggml_dims))
        for d in ggml_dims:
            part += struct.pack("<Q", d)
        part += struct.pack("<IQ", ggml_type, cur_off)
        t_parts.append(part)
        cur_off += 256

    header = (
        b"GGUF"
        + struct.pack("<IQQ", version, len(t_list), len(kv_parts))
        + b"".join(kv_parts)
        + b"".join(t_parts)
    )
    return header


def write_checkpoint_dir(
    root: Path,
    *,
    config: dict[str, Any],
    tensors: dict[str, tuple[str, list[int]]],
    generation_config: dict[str, Any] | None = None,
    tokenizer_config: dict[str, Any] | None = None,
    tokenizer_json: dict[str, Any] | None = None,
) -> Path:
    """Write a complete synthetic local Safetensors checkpoint directory."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if generation_config is not None:
        (root / "generation_config.json").write_text(json.dumps(generation_config), encoding="utf-8")
    if tokenizer_config is not None:
        (root / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    if tokenizer_json is not None:
        (root / "tokenizer.json").write_text(json.dumps(tokenizer_json), encoding="utf-8")
    (root / "model.safetensors").write_bytes(build_safetensors_header_file(tensors))
    return root
