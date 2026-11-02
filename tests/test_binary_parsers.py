"""Byte-level GGUF and Safetensors parsing, including every safety limit."""

from __future__ import annotations

import json
import struct
from typing import Any

import pytest

from conftest import _pack_gguf_str, build_gguf_header_bytes, build_safetensors_header_file
from quantpreflight import gguf_io
from quantpreflight.gguf_io import (
    GGML_TYPES,
    GGUF_MAGIC,
    GGUFFormatError,
    MAX_GGUF_ARRAY_LEN,
    MAX_GGUF_HEADER_BYTES,
    MAX_GGUF_KV_COUNT,
    MAX_GGUF_SPLIT_PARTS,
    MAX_GGUF_STRING_BYTES,
    MAX_GGUF_TENSORS,
    _AdaptiveRangeReader,
    compute_ggml_tensor_bytes,
    merge_gguf_parts,
    read_gguf_header_via_range,
    split_gguf_names,
)
from quantpreflight.safetensors_io import (
    MAX_SAFETENSORS_HEADER_BYTES,
    SafetensorsFormatError,
    parse_safetensors_header_bytes,
    parse_safetensors_index,
    read_safetensors_header_via_range,
    validate_shard_filename,
)


def _ranged(blob: bytes) -> Any:
    def fetch(start: int, end: int) -> bytes:
        return blob[start : end + 1]

    return fetch


def _gguf_kv(key: str, vtype: int, payload: bytes) -> bytes:
    return b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + _pack_gguf_str(key) + struct.pack("<I", vtype) + payload


def _read_kv(vtype: int, payload: bytes) -> Any:
    return read_gguf_header_via_range(_ranged(_gguf_kv("k", vtype, payload))).metadata["k"]


# GGUF values


@pytest.mark.parametrize(
    ("vtype", "fmt", "value"),
    [
        (0, "<B", 200),
        (1, "<b", -100),
        (2, "<H", 60000),
        (3, "<h", -30000),
        (4, "<I", 4000000000),
        (5, "<i", -2000000000),
        (7, "<?", True),
        (10, "<Q", 2**63),
        (11, "<q", -(2**62)),
    ],
)
def test_gguf_integer_values(vtype: int, fmt: str, value: Any) -> None:
    assert _read_kv(vtype, struct.pack(fmt, value)) == value


@pytest.mark.parametrize(("vtype", "fmt"), [(6, "<f"), (12, "<d")])
def test_gguf_float_values(vtype: int, fmt: str) -> None:
    assert _read_kv(vtype, struct.pack(fmt, 0.5)) == 0.5


def test_gguf_string_value() -> None:
    assert _read_kv(8, _pack_gguf_str("héllo")) == "héllo"


def test_gguf_string_with_bad_utf8_is_replaced_not_fatal() -> None:
    assert _read_kv(8, struct.pack("<Q", 2) + b"\xff\xfe") == "\ufffd\ufffd"


def test_gguf_small_numeric_array_is_returned() -> None:
    assert _read_kv(9, struct.pack("<IQ", 4, 3) + struct.pack("<3I", 1, 2, 3)) == [1, 2, 3]


@pytest.mark.parametrize(("count", "summarized"), [(32, False), (33, True)])
def test_gguf_numeric_arrays_are_summarized_past_32(count: int, summarized: bool) -> None:
    value = _read_kv(9, struct.pack("<IQ", 0, count) + bytes(count))
    assert isinstance(value, dict) is summarized
    if summarized:
        assert value == {"__gguf_array_len__": count, "elem_type": 0}


def test_gguf_string_table_keeps_count_head_and_last() -> None:
    tokens = [f"t{i}" for i in range(12)]
    payload = struct.pack("<IQ", 8, len(tokens)) + b"".join(_pack_gguf_str(t) for t in tokens)
    assert _read_kv(9, payload) == {"__gguf_string_table_len__": 12, "head": tokens[:8], "last": "t11"}


def test_gguf_nested_array() -> None:
    inner = struct.pack("<IQ", 0, 1) + b"\x07"
    assert _read_kv(9, struct.pack("<IQ", 9, 2) + inner + inner) == [[7], [7]]


def _nested(depth: int) -> bytes:
    payload = struct.pack("<IQ", 0, 0)
    for _ in range(depth):
        payload = struct.pack("<IQ", 9, 1) + payload
    return payload


def test_gguf_array_depth_limit() -> None:
    assert _read_kv(9, _nested(4))
    with pytest.raises(GGUFFormatError, match="depth"):
        _read_kv(9, _nested(5))


def test_gguf_unknown_value_type() -> None:
    with pytest.raises(GGUFFormatError, match="Unknown GGUF metadata value type"):
        _read_kv(13, b"")


# GGUF limits


@pytest.mark.parametrize(
    ("header", "message"),
    [
        (b"", "Unexpected EOF"),
        (b"GGU", "Unexpected EOF"),
        (b"NOPE" + bytes(32), "Invalid GGUF magic"),
        (b"GGUF" + struct.pack("<I", 1), "Unsupported GGUF version"),
        (b"GGUF" + struct.pack("<I", 4), "Unsupported GGUF version"),
        (b"GGUF" + struct.pack("<IQQ", 3, MAX_GGUF_TENSORS + 1, 0), "tensor_count"),
        (b"GGUF" + struct.pack("<IQQ", 3, 0, MAX_GGUF_KV_COUNT + 1), "kv_count"),
        (_gguf_kv("k", 8, struct.pack("<Q", MAX_GGUF_STRING_BYTES + 1)), "string length"),
        (_gguf_kv("k", 9, struct.pack("<IQ", 0, MAX_GGUF_ARRAY_LEN + 1)), "array length"),
    ],
)
def test_gguf_rejects_bad_headers(header: bytes, message: str) -> None:
    with pytest.raises(GGUFFormatError, match=message):
        read_gguf_header_via_range(_ranged(header))


@pytest.mark.parametrize(("tensors", "kv"), [(MAX_GGUF_TENSORS, 0), (0, MAX_GGUF_KV_COUNT)])
def test_gguf_counts_at_the_limit_get_past_the_guard(tensors: int, kv: int) -> None:
    header = b"GGUF" + struct.pack("<IQQ", 3, tensors, kv)
    with pytest.raises(GGUFFormatError, match="Unexpected EOF"):
        read_gguf_header_via_range(_ranged(header))


def test_gguf_string_at_the_limit_gets_past_the_guard() -> None:
    with pytest.raises(GGUFFormatError, match="Unexpected EOF"):
        _read_kv(8, struct.pack("<Q", MAX_GGUF_STRING_BYTES))


def test_gguf_array_at_the_limit_gets_past_the_guard() -> None:
    with pytest.raises(GGUFFormatError, match="Unexpected EOF"):
        _read_kv(9, struct.pack("<IQ", 0, MAX_GGUF_ARRAY_LEN))


def test_gguf_header_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = _AdaptiveRangeReader(lambda start, end: bytes(end - start + 1))
    with pytest.raises(GGUFFormatError, match="safety ceiling"):
        reader.read_exact(MAX_GGUF_HEADER_BYTES + 1)
    assert reader.bytes_fetched == 0


def test_gguf_reader_rejects_negative_reads() -> None:
    with pytest.raises(GGUFFormatError, match="Negative"):
        _AdaptiveRangeReader(lambda start, end: b"").read_exact(-1)


def test_gguf_tensor_rank_limit() -> None:
    def header(rank: int) -> bytes:
        info = _pack_gguf_str("t") + struct.pack("<I", rank) + struct.pack(f"<{rank}Q", *([2] * rank)) + struct.pack("<IQ", 0, 0)
        return b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + info

    assert read_gguf_header_via_range(_ranged(header(8))).tensors["t"].shape == (2,) * 8
    with pytest.raises(GGUFFormatError, match="invalid rank"):
        read_gguf_header_via_range(_ranged(header(9)))


# GGUF tensors


@pytest.mark.parametrize("version", [2, 3])
def test_gguf_supported_versions(version: int) -> None:
    assert read_gguf_header_via_range(_ranged(build_gguf_header_bytes(version=version))).version == version


def test_gguf_reverses_matrix_shapes_only() -> None:
    blob = build_gguf_header_bytes(tensors=[("m", (32, 64), 0), ("v", (32,), 0)])
    tensors = read_gguf_header_via_range(_ranged(blob)).tensors
    assert tensors["m"].shape == (64, 32)
    assert tensors["v"].shape == (32,)


def test_gguf_records_offset_and_type() -> None:
    blob = build_gguf_header_bytes(tensors=[("a", (32,), 0), ("b", (32,), 12)])
    tensors = read_gguf_header_via_range(_ranged(blob)).tensors
    assert (tensors["b"].offset, tensors["b"].ggml_type_id, tensors["b"].dtype_name) == (256, 12, "Q4_K")
    assert tensors["b"].effective_bits == 4.0


@pytest.mark.parametrize(("type_id", "row"), sorted(GGML_TYPES.items()))
def test_ggml_type_table(type_id: int, row: tuple[str, int, int, float]) -> None:
    name, block_elems, block_bytes, bits = row
    assert compute_ggml_tensor_bytes(type_id, (block_elems,)) == (name, block_bytes, bits)
    assert compute_ggml_tensor_bytes(type_id, (block_elems * 3 + 1,))[1] == block_bytes * 4
    assert compute_ggml_tensor_bytes(type_id, (2, block_elems))[1] == block_bytes * 2
    assert compute_ggml_tensor_bytes(type_id, (0, block_elems))[1] == 0


@pytest.mark.parametrize(("type_id", "row"), sorted(GGML_TYPES.items()))
def test_ggml_block_size_stays_close_to_declared_bits(type_id: int, row: tuple[str, int, int, float]) -> None:
    _, block_elems, block_bytes, bits = row
    assert bits - 0.1 <= block_bytes * 8 / block_elems <= bits + 3.2


def test_ggml_type_newer_than_the_table_does_not_stop_a_scan() -> None:
    assert compute_ggml_tensor_bytes(4, (8,)) == ("GGML_TYPE_4", 8, 8.0)
    blob = build_gguf_header_bytes(tensors=[("t", (32,), 40)])
    assert read_gguf_header_via_range(_ranged(blob)).tensors["t"].dtype_name == "GGML_TYPE_40"


def test_gguf_magic_constant() -> None:
    assert GGUF_MAGIC == b"GGUF"


# Adaptive reader


def test_reader_doubles_its_window_and_stops_at_the_cap() -> None:
    requests: list[int] = []

    def fetch(start: int, end: int) -> bytes:
        requests.append(end - start + 1)
        return bytes(end - start + 1)

    reader = _AdaptiveRangeReader(fetch, initial_window=4)
    for _ in range(5):
        reader.read_exact(reader.bytes_fetched + 1)
    assert requests == sorted(requests)
    assert requests[:3] == [4, 8, 16]

    capped = _AdaptiveRangeReader(fetch, initial_window=4 * 1024 * 1024)
    capped.read_exact(1)
    capped.read_exact(4 * 1024 * 1024)
    assert requests[-1] == 4 * 1024 * 1024


def test_reader_requests_more_than_its_window_when_needed() -> None:
    requests: list[int] = []

    def fetch(start: int, end: int) -> bytes:
        requests.append(end - start + 1)
        return bytes(end - start + 1)

    _AdaptiveRangeReader(fetch, initial_window=4).read_exact(100)
    assert requests == [100]


def test_reader_tracks_position_and_stops_after_a_short_read() -> None:
    calls = []

    def fetch(start: int, end: int) -> bytes:
        calls.append(start)
        return b"abc"[start:]

    reader = _AdaptiveRangeReader(fetch)
    assert reader.read_exact(2) == b"ab"
    assert reader.position == 2
    assert reader.bytes_fetched == 3
    with pytest.raises(GGUFFormatError):
        reader.read_exact(2)
    assert calls == [0]


def test_reader_needs_no_second_fetch_for_buffered_bytes() -> None:
    calls = []

    def fetch(start: int, end: int) -> bytes:
        calls.append(start)
        return bytes(end - start + 1)

    reader = _AdaptiveRangeReader(fetch, initial_window=8)
    reader.read_exact(4)
    reader.read_exact(4)
    assert calls == [0]


def test_reader_clamps_its_range_to_the_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gguf_io, "MAX_GGUF_HEADER_BYTES", 10)
    ends = []

    def fetch(start: int, end: int) -> bytes:
        ends.append(end)
        return bytes(end - start + 1)

    _AdaptiveRangeReader(fetch, initial_window=64).read_exact(10)
    assert ends == [9]


# Safetensors headers


def _header(tensors: Any) -> tuple[bytes, bytes]:
    raw = json.dumps(tensors).encode()
    return struct.pack("<Q", len(raw)), raw


def _entry(**overrides: Any) -> dict[str, Any]:
    return {"t": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8], **overrides}}


def test_safetensors_parses_entries_and_metadata() -> None:
    parsed = parse_safetensors_header_bytes(*_header({"__metadata__": {"format": "pt", "n": 1}, **_entry()}))
    entry = parsed.tensors["t"]
    assert (entry.name, entry.dtype, entry.shape, entry.data_offsets, entry.byte_size) == ("t", "F16", (2, 2), (0, 8), 8)
    assert parsed.metadata == {"format": "pt", "n": "1"}
    assert parsed.header_byte_length == 8 + len(_header({"__metadata__": {"format": "pt", "n": 1}, **_entry()})[1])


def test_safetensors_zero_sized_dimension_is_valid() -> None:
    assert parse_safetensors_header_bytes(*_header(_entry(shape=[0, 8], data_offsets=[0, 0]))).tensors["t"].shape == (0, 8)


def test_safetensors_header_at_the_ceiling_passes_the_size_check() -> None:
    with pytest.raises(SafetensorsFormatError, match="Truncated"):
        parse_safetensors_header_bytes(struct.pack("<Q", MAX_SAFETENSORS_HEADER_BYTES), b"")


def test_safetensors_metadata_that_is_not_an_object_is_ignored() -> None:
    assert parse_safetensors_header_bytes(*_header({"__metadata__": "x"})).metadata == {}


def test_safetensors_empty_span_uses_the_shape_for_its_size() -> None:
    parsed = parse_safetensors_header_bytes(*_header(_entry(data_offsets=[4, 4])))
    assert parsed.tensors["t"].byte_size == 8


def test_safetensors_real_span_wins_over_the_shape() -> None:
    parsed = parse_safetensors_header_bytes(*_header(_entry(data_offsets=[0, 100])))
    assert parsed.tensors["t"].byte_size == 100


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"t": 5}, "must be a JSON object"),
        (_entry(dtype=None), "dtype"),
        (_entry(dtype="F128"), "dtype"),
        (_entry(dtype=5), "dtype"),
        (_entry(shape=None), "Invalid shape"),
        (_entry(shape=[-1]), "Invalid shape"),
        (_entry(shape=["2"]), "Invalid shape"),
        (_entry(data_offsets=None), "data_offsets"),
        (_entry(data_offsets=[0]), "data_offsets"),
        (_entry(data_offsets=[0, 1, 2]), "data_offsets"),
        (_entry(data_offsets=[-1, 8]), "data_offsets"),
        (_entry(data_offsets=[8, 0]), "data_offsets"),
        (_entry(data_offsets=["0", 8]), "data_offsets"),
    ],
)
def test_safetensors_rejects_bad_entries(entry: dict[str, Any], message: str) -> None:
    with pytest.raises(SafetensorsFormatError, match=message):
        parse_safetensors_header_bytes(*_header(entry))


@pytest.mark.parametrize("root", [[], 5, "x", None])
def test_safetensors_root_must_be_an_object(root: Any) -> None:
    with pytest.raises(SafetensorsFormatError, match="root must be an object"):
        parse_safetensors_header_bytes(*_header(root))


@pytest.mark.parametrize(
    ("prefix", "body", "message"),
    [
        (b"", b"", "Expected 8-byte"),
        (b"1234567", b"", "Expected 8-byte"),
        (struct.pack("<Q", 0), b"", "cannot be 0"),
        (struct.pack("<Q", MAX_SAFETENSORS_HEADER_BYTES + 1), b"", "safety ceiling"),
        (struct.pack("<Q", 4), b"{}", "Truncated"),
        (struct.pack("<Q", 2), b"\xff\xfe", "Invalid UTF-8 JSON"),
        (struct.pack("<Q", 2), b"{x", "Invalid UTF-8 JSON"),
    ],
)
def test_safetensors_rejects_bad_envelopes(prefix: bytes, body: bytes, message: str) -> None:
    with pytest.raises(SafetensorsFormatError, match=message):
        parse_safetensors_header_bytes(prefix, body)


@pytest.mark.parametrize(
    ("blob", "message"),
    [
        (b"", "shorter than 8 bytes"),
        (b"1234567", "shorter than 8 bytes"),
        (struct.pack("<Q", 0), "Invalid or oversized"),
        (struct.pack("<Q", MAX_SAFETENSORS_HEADER_BYTES + 1), "Invalid or oversized"),
        (struct.pack("<Q", 100) + b"{}", "Incomplete"),
    ],
)
def test_safetensors_range_read_rejects_bad_files(blob: bytes, message: str) -> None:
    with pytest.raises(SafetensorsFormatError, match=message):
        read_safetensors_header_via_range(_ranged(blob))


def test_safetensors_header_at_the_ceiling_is_accepted_by_the_guard() -> None:
    with pytest.raises(SafetensorsFormatError, match="Incomplete"):
        read_safetensors_header_via_range(_ranged(struct.pack("<Q", MAX_SAFETENSORS_HEADER_BYTES)))


@pytest.mark.parametrize("extra", [0, 1, 10])
def test_safetensors_small_header_takes_one_round_trip(extra: int) -> None:
    blob = build_safetensors_header_file({"a": ("F16", [2, 2])}) + bytes(extra)
    calls = []

    def fetch(start: int, end: int) -> bytes:
        calls.append((start, end))
        return blob[start : end + 1]

    assert "a" in read_safetensors_header_via_range(fetch).tensors
    assert calls == [(0, 65535)]


def test_safetensors_header_filling_the_first_chunk_exactly() -> None:
    blob = build_safetensors_header_file({"a": ("F16", [2, 2])})
    calls = []

    def fetch(start: int, end: int) -> bytes:
        calls.append((start, end))
        return blob[start : end + 1]

    read_safetensors_header_via_range(fetch, initial_chunk_bytes=len(blob))
    assert len(calls) == 1


def test_safetensors_large_header_takes_a_second_round_trip() -> None:
    tensors = {f"layer.{i}.weight": ("F16", [2, 2]) for i in range(2000)}
    blob = build_safetensors_header_file(tensors)
    calls = []

    def fetch(start: int, end: int) -> bytes:
        calls.append((start, end))
        return blob[start : end + 1]

    assert len(read_safetensors_header_via_range(fetch).tensors) == 2000
    assert calls == [(0, 65535), (65536, len(blob) - 1)]


# Shard names and the index


@pytest.mark.parametrize(
    ("name", "clean"),
    [
        ("model-00001-of-00002.safetensors", "model-00001-of-00002.safetensors"),
        ("sub/model.safetensors", "sub/model.safetensors"),
        ("sub\\model.safetensors", "sub/model.safetensors"),
        ("sub//model.safetensors", "sub/model.safetensors"),
    ],
)
def test_valid_shard_names(name: str, clean: str) -> None:
    assert validate_shard_filename(name) == clean


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("", "Empty or non-string"),
        (None, "Empty or non-string"),
        (5, "Empty or non-string"),
        ("https://evil/model.safetensors", "non-relative"),
        ("/etc/model.safetensors", "non-relative"),
        ("\\\\host\\model.safetensors", "non-relative"),
        ("C:\\model.safetensors", "Drive or scheme"),
        ("../model.safetensors", "traversal"),
        ("a/../../model.safetensors", "traversal"),
        ("./model.safetensors", "traversal"),
        ("a\\..\\model.safetensors", "traversal"),
        ("model.bin", "does not end with"),
        ("model.safetensors.exe", "does not end with"),
    ],
)
def test_unsafe_shard_names(name: Any, message: str) -> None:
    with pytest.raises(SafetensorsFormatError, match=message):
        validate_shard_filename(name)


def test_index_returns_clean_map_and_total_size() -> None:
    index = {"weight_map": {"a": "sub\\m.safetensors"}, "metadata": {"total_size": 123}}
    assert parse_safetensors_index(index) == ({"a": "sub/m.safetensors"}, 123)


@pytest.mark.parametrize("metadata", [None, {}, {"total_size": "x"}, 5])
def test_index_total_size_is_optional(metadata: Any) -> None:
    assert parse_safetensors_index({"weight_map": {"a": "m.safetensors"}, "metadata": metadata})[1] is None


@pytest.mark.parametrize("index", [None, [], {}, {"weight_map": None}, {"weight_map": {}}, {"weight_map": []}, {"weight_map": {"": "m.safetensors"}}])
def test_index_rejects_bad_shapes(index: Any) -> None:
    with pytest.raises(SafetensorsFormatError):
        parse_safetensors_index(index)


# Split names


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("m.gguf", ["m.gguf"]),
        ("m-1-of-2.gguf", ["m-1-of-2.gguf"]),
        ("m-00001-of-00002.safetensors", ["m-00001-of-00002.safetensors"]),
        ("m-00002-of-00002.gguf", ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"]),
        ("sub/M-00001-of-00002.GGUF", ["sub/M-00001-of-00002.GGUF", "sub/M-00002-of-00002.GGUF"]),
        ("m-00001-of-00000.gguf", ["m-00001-of-00000.gguf"]),
        (f"m-00001-of-{MAX_GGUF_SPLIT_PARTS + 1:05d}.gguf", [f"m-00001-of-{MAX_GGUF_SPLIT_PARTS + 1:05d}.gguf"]),
    ],
)
def test_split_gguf_names(name: str, expected: list[str]) -> None:
    assert split_gguf_names(name) == expected


def test_split_gguf_name_limit() -> None:
    assert len(split_gguf_names(f"m-00001-of-{MAX_GGUF_SPLIT_PARTS:05d}.gguf")) == MAX_GGUF_SPLIT_PARTS


def test_merge_keeps_the_first_metadata_and_adds_sizes() -> None:
    first = read_gguf_header_via_range(_ranged(build_gguf_header_bytes(kv_strings={"a": "1"}, tensors=[("x", (32,), 0)])))
    second = read_gguf_header_via_range(_ranged(build_gguf_header_bytes(kv_strings={"a": "2"}, tensors=[("y", (32,), 0)])))
    merged = merge_gguf_parts([first, second])
    assert merged.metadata == {"a": "1"}
    assert set(merged.tensors) == {"x", "y"}
    assert merged.header_bytes_consumed == first.header_bytes_consumed + second.header_bytes_consumed
    assert merged.version == first.version
