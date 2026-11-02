"""Tests for Safetensors and GGUF binary header parsers and security bounds."""

from __future__ import annotations

import struct
import pytest

from conftest import build_gguf_header_bytes, build_safetensors_header_file
from quantpreflight.gguf_io import GGUFFormatError, read_gguf_header_via_range
from quantpreflight.safetensors_io import (
    MAX_SAFETENSORS_HEADER_BYTES,
    SafetensorsFormatError,
    parse_safetensors_index,
    read_safetensors_header_via_range,
    validate_shard_filename,
)


def test_safetensors_single_and_two_roundtrip_range_reads() -> None:
    raw = build_safetensors_header_file(
        {
            "model.embed_tokens.weight": ("BF16", [32000, 4096]),
            "model.layers.0.mlp.down_proj.qweight": ("I32", [11008, 512]),
        },
        metadata={"format": "pt"},
    )
    calls: list[tuple[int, int]] = []

    def small_fetch(start: int, end: int) -> bytes:
        calls.append((start, end))
        return raw[start : end + 1]

    # Force a 2-roundtrip read with a tiny 32-byte initial window
    parsed = read_safetensors_header_via_range(small_fetch, initial_chunk_bytes=32)
    assert len(calls) == 2
    assert parsed.metadata == {"format": "pt"}
    assert parsed.tensors["model.embed_tokens.weight"].shape == (32000, 4096)
    assert parsed.tensors["model.embed_tokens.weight"].byte_size == 32000 * 4096 * 2
    assert parsed.tensors["model.layers.0.mlp.down_proj.qweight"].dtype == "I32"


def test_safetensors_oversized_header_dos_rejected() -> None:
    malicious = struct.pack("<Q", MAX_SAFETENSORS_HEADER_BYTES + 1) + b"{}"
    with pytest.raises(SafetensorsFormatError, match="Invalid or oversized"):
        read_safetensors_header_via_range(lambda s, e: malicious[s : e + 1])


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "",
        "../secret.safetensors",
        "shards/../../etc/passwd.safetensors",
        "/abs/model.safetensors",
        "C:\\Windows\\model.safetensors",
        "https://evil.example/model.safetensors",
        "weights.bin",
    ],
)
def test_validate_shard_filename_blocks_traversal_and_schemes(unsafe_path: str) -> None:
    with pytest.raises(SafetensorsFormatError):
        validate_shard_filename(unsafe_path)


def test_parse_safetensors_index_clean() -> None:
    wmap, total = parse_safetensors_index(
        {
            "metadata": {"total_size": 41600000000},
            "weight_map": {
                "model.layers.0.q_proj.qweight": "model-00001-of-00002.safetensors",
                "lm_head.weight": "sub/model-00002-of-00002.safetensors",
            },
        }
    )
    assert total == 41600000000
    assert wmap["lm_head.weight"] == "sub/model-00002-of-00002.safetensors"


def test_gguf_v3_header_and_quant_types() -> None:
    buf = build_gguf_header_bytes(
        version=3,
        kv_strings={"general.architecture": "llama"},
        kv_u32={
            "general.file_type": 15,
            "llama.embedding_length": 4096,
            "llama.block_count": 32,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
            "tokenizer.ggml.eos_token_id": 2,
        },
        tokens=["<unk>", "<s>", "</s>", "hello"],
        tensors=[
            # GGML dims [in_features=4096, out_features=4] -> reversed PyTorch shape (4, 4096)
            ("token_embd.weight", (4096, 4), 12),   # Q4_K (4-bit)
            ("blk.0.attn_norm.weight", (4096,), 0), # F32 (32-bit)
            ("blk.0.ffn_down.weight", (11008, 4096), 16),  # IQ2_XXS (2-bit)
        ],
    )
    parsed = read_gguf_header_via_range(lambda s, e: buf[s : e + 1])
    assert parsed.version == 3
    assert parsed.metadata["general.architecture"] == "llama"
    assert parsed.metadata["tokenizer.ggml.tokens"]["__gguf_string_table_len__"] == 4
    assert parsed.tensors["token_embd.weight"].shape == (4, 4096)
    assert parsed.tensors["token_embd.weight"].dtype_name == "Q4_K"
    assert parsed.tensors["token_embd.weight"].effective_bits == 4.0
    assert parsed.tensors["blk.0.ffn_down.weight"].dtype_name == "IQ2_XXS"
    assert parsed.tensors["blk.0.ffn_down.weight"].effective_bits == 2.0


def test_gguf_rejects_bad_magic_and_version() -> None:
    with pytest.raises(GGUFFormatError, match="Invalid GGUF magic"):
        read_gguf_header_via_range(lambda s, e: b"NOTG" + b"\x00" * 32)

    bad_ver = b"GGUF" + struct.pack("<IQQ", 99, 0, 0)
    with pytest.raises(GGUFFormatError, match="Unsupported GGUF version"):
        read_gguf_header_via_range(lambda s, e: bad_ver[s : e + 1])
