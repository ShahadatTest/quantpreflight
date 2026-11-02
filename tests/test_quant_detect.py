"""Quantization config parsing, packed tensor shapes, and GGUF metadata normalization."""

from __future__ import annotations

from typing import Any

import pytest

from quantpreflight.gguf_io import ParsedGGUFHeader, RawGGUFTensorEntry
from quantpreflight.models import QuantSpec
from quantpreflight.quant_detect import (
    _bits,
    _block_size,
    _str_tuple,
    build_safetensors_specs,
    normalize_gguf_metadata,
    parse_quant_spec_from_config,
)
from quantpreflight.safetensors_io import RawSafetensorEntry


def _parse(**qcfg: Any) -> QuantSpec:
    return parse_quant_spec_from_config({"quantization_config": qcfg})


def _fields(spec: QuantSpec, expected: dict[str, Any]) -> dict[str, Any]:
    return {key: getattr(spec, key) for key in expected}


def _ct_group(**weights: Any) -> dict[str, Any]:
    return {"group_0": {"weights": weights}}


@pytest.mark.parametrize("config", [{}, {"quantization_config": None}, {"quantization_config": {}}, {"quantization_config": "x"}])
def test_no_quantization_config_means_none(config: dict[str, Any]) -> None:
    assert parse_quant_spec_from_config(config).method == "none"


@pytest.mark.parametrize(
    ("qcfg", "expected"),
    [
        (
            {"quant_method": "awq", "bits": 4, "group_size": 128, "zero_point": True, "version": "GEMM"},
            {"method": "awq", "bits": 4.0, "group_size": 128, "sym": False, "zero_point": True, "format_version": "gemm"},
        ),
        (
            {"quant_method": "autoawq", "w_bit": 8, "q_group_size": 64, "zero_point": False},
            {"method": "awq", "bits": 8.0, "group_size": 64, "sym": True, "format_version": "gemm"},
        ),
        ({"quant_method": "awq"}, {"method": "awq", "bits": 4.0, "group_size": None, "sym": None, "zero_point": True}),
        (
            {"quant_method": "awq", "modules_to_not_convert": ["lm_head", 5]},
            {"ignored_layers": ("lm_head",), "activation_dtype": "fp16", "activation_bits": 16.0},
        ),
    ],
)
def test_awq(qcfg: dict[str, Any], expected: dict[str, Any]) -> None:
    assert _fields(_parse(**qcfg), expected) == expected


@pytest.mark.parametrize(
    ("qcfg", "expected"),
    [
        (
            {"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": True, "desc_act": True},
            {"method": "gptq", "bits": 4.0, "group_size": 128, "sym": True, "desc_act": True, "format_version": "gptq"},
        ),
        ({"quant_method": "gptq_v2", "bits": 8, "sym": False}, {"bits": 8.0, "sym": False, "zero_point": True}),
        ({"quant_method": "gptq", "group_size": -1}, {"group_size": None, "sym": True, "zero_point": False}),
        ({"quant_method": "gptq", "checkpoint_format": "marlin"}, {"format_version": "marlin"}),
        ({"quant_method": "gptq", "is_marlin_format": True}, {"format_version": "marlin"}),
        ({"bits": 4, "group_size": 32}, {"method": "gptq", "group_size": 32}),
        ({"bits": 4, "damp_percent": 0.01}, {"method": "gptq", "group_size": None}),
    ],
)
def test_gptq(qcfg: dict[str, Any], expected: dict[str, Any]) -> None:
    assert _fields(_parse(**qcfg), expected) == expected


@pytest.mark.parametrize("name", ["auto-round", "autoround", "inc", "AutoRound", " inc "])
def test_autoround_aliases(name: str) -> None:
    spec = _parse(quant_method=name, bits=4, group_size=128, sym=False, packing_format="Auto_GPTQ")
    expected = {"method": "autoround", "bits": 4.0, "group_size": 128, "sym": False, "format_version": "auto_gptq"}
    assert _fields(spec, expected) == expected


def test_autoround_defaults() -> None:
    spec = _parse(quant_method="auto-round", backend="ipex", group_size=0, desc_act=True)
    expected = {"bits": 4.0, "group_size": None, "sym": True, "desc_act": True, "format_version": "ipex"}
    assert _fields(spec, expected) == expected
    assert _parse(quant_method="auto-round").format_version == "auto_round"


@pytest.mark.parametrize("name", ["fp8", "finegrained_fp8", "fbgemm_fp8"])
def test_fp8_aliases_and_block_size(name: str) -> None:
    plain = _parse(quant_method=name, activation_scheme="STATIC", ignored_layers=["lm_head"])
    expected = {
        "method": "fp8",
        "bits": 8.0,
        "activation_dtype": "fp8",
        "format_version": "static",
        "weight_block_size": None,
        "ignored_layers": ("lm_head",),
    }
    assert _fields(plain, expected) == expected

    blocked = _parse(quant_method=name, weight_block_size=[128, 128])
    assert blocked.weight_block_size == (128, 128)
    assert blocked.format_version == "block-fp8"
    assert _parse(quant_method=name).format_version == "dynamic"


@pytest.mark.parametrize(
    ("qcfg", "expected"),
    [
        (
            {"quant_method": "compressed-tensors", "config_groups": _ct_group(num_bits=4, group_size=128, symmetric=True)},
            {"method": "compressed-tensors", "bits": 4.0, "group_size": 128, "sym": True, "zero_point": False},
        ),
        ({"config_groups": _ct_group(num_bits=8, symmetric=False)}, {"bits": 8.0, "sym": False, "zero_point": True}),
        ({"config_groups": _ct_group(num_bits=4, type="float")}, {"method": "nvfp4", "format_version": "nvfp4-pack-quantized"}),
        ({"format": "nvfp4-pack-quantized", "config_groups": _ct_group(num_bits=4)}, {"method": "nvfp4"}),
        ({"config_groups": _ct_group(num_bits=8, type="float", block_structure=[128, 128])}, {"weight_block_size": (128, 128)}),
        ({"config_groups": _ct_group(num_bits=8, group_size=0)}, {"group_size": None}),
        ({"config_groups": {"g": {"weights": {"num_bits": 8}, "input_activations": {"num_bits": 8, "type": "float"}}}}, {"activation_dtype": "fp8", "activation_bits": 8.0}),
        ({"config_groups": {"g": {"input_activations": {"num_bits": 8, "type": "int"}}}}, {"activation_dtype": "int8"}),
        ({"config_groups": {"g": {"input_activations": {"num_bits": 4}}}}, {"activation_dtype": "fp4"}),
        ({"config_groups": {"g": {"input_activations": {"num_bits": 16}}}}, {"activation_dtype": None, "activation_bits": 16.0}),
        ({"config_groups": {"g": None, "h": {"weights": {"num_bits": 8}}}}, {"bits": 8.0}),
        ({"quant_method": "compressed-tensors"}, {"format_version": "compressed-tensors", "bits": None}),
        ({"quant_method": "compressed-tensors", "ignore": ["lm_head", None]}, {"ignored_layers": ("lm_head",)}),
    ],
)
def test_compressed_tensors(qcfg: dict[str, Any], expected: dict[str, Any]) -> None:
    assert _fields(_parse(**qcfg), expected) == expected


def test_compressed_tensors_reads_only_the_first_config_group() -> None:
    groups = {"a": {"weights": {"num_bits": 4}}, "b": {"weights": {"num_bits": 8}}}
    assert _parse(config_groups=groups).bits == 4.0


@pytest.mark.parametrize(
    ("qcfg", "expected"),
    [
        ({"producer": {"name": "ModelOpt"}, "quantization": {"quant_algo": "FP8"}}, {"method": "fp8", "bits": 8.0, "group_size": None}),
        ({"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "NVFP4"}}, {"method": "nvfp4", "bits": 4.0, "group_size": 16}),
        ({"quantization": {"quant_algo": "MXFP4"}}, {"method": "mxfp4", "bits": 4.0, "group_size": 32}),
        ({"quant_algo": "FP8", "group_size": 7}, {"method": "fp8", "group_size": 7, "format_version": "fp8"}),
        ({"quant_method": "nvfp4"}, {"method": "nvfp4", "activation_dtype": "fp4"}),
        ({"quant_method": "mxfp4"}, {"method": "mxfp4"}),
        ({"quant_algo": "FP8", "exclude_modules": ["lm_head"]}, {"ignored_layers": ("lm_head",), "activation_dtype": "fp8"}),
        ({"quantization": {"quant_algo": "FP8", "exclude_modules": ["a"]}, "exclude_modules": ["b"]}, {"ignored_layers": ("a",)}),
        ({"quantization": {"quant_algo": "FP8", "modules_to_not_convert": ["c"]}}, {"ignored_layers": ("c",)}),
    ],
)
def test_modelopt(qcfg: dict[str, Any], expected: dict[str, Any]) -> None:
    assert _fields(_parse(**qcfg), expected) == expected


@pytest.mark.parametrize(
    ("qcfg", "expected"),
    [
        ({"quant_method": "bitsandbytes", "load_in_4bit": True}, {"method": "bitsandbytes", "bits": 4.0, "format_version": "nf4", "activation_bits": 16.0}),
        ({"quant_method": "bnb", "load_in_8bit": True}, {"bits": 8.0, "format_version": "llm.int8", "activation_bits": 8.0}),
        ({"load_in_4bit": True, "bnb_4bit_quant_type": "fp4"}, {"method": "bitsandbytes", "format_version": "fp4"}),
        ({"load_in_8bit": True}, {"method": "bitsandbytes", "bits": 8.0}),
        ({"quant_method": "bitsandbytes", "bnb_4bit_quant_type": "nf4"}, {"bits": 4.0}),
    ],
)
def test_bitsandbytes(qcfg: dict[str, Any], expected: dict[str, Any]) -> None:
    assert _fields(_parse(**qcfg), expected) == expected


@pytest.mark.parametrize("name", ["bitnet", "1bit", "ternary"])
def test_bitnet_aliases(name: str) -> None:
    spec = _parse(quant_method=name)
    assert (spec.method, spec.bits, spec.format_version, spec.activation_dtype) == ("bitnet", 1.58, name, "int8")


def test_unknown_methods_keep_their_name_and_numbers() -> None:
    spec = _parse(quant_method="NewThing", bits=6, group_size=32)
    assert (spec.method, spec.bits, spec.group_size) == ("newthing", 6.0, 32)
    assert _parse(w_bit=3).bits == 3.0
    assert _parse(something=1).method == "custom"
    assert _parse(something=1).bits is None


def test_raw_config_is_kept_for_reports() -> None:
    assert _parse(quant_method="awq", bits=4).raw_config == {"quant_method": "awq", "bits": 4}


# Packed tensor shapes


def _entry(name: str, dtype: str, shape: tuple[int, ...]) -> tuple[RawSafetensorEntry, str]:
    return RawSafetensorEntry(name, dtype, shape, (0, 64), 64), "model.safetensors"


def _specs(quant: QuantSpec, *items: tuple[str, str, tuple[int, ...]]) -> dict[str, Any]:
    raw = {name: _entry(name, dtype, shape) for name, dtype, shape in items}
    return build_safetensors_specs(raw, quant)


@pytest.mark.parametrize(
    ("bits", "packed_cols", "logical"),
    [(4.0, 512, (4096, 1024)), (8.0, 1024, (4096, 1024)), (2.0, 256, (4096, 1024))],
)
def test_awq_qweight_unpacks_along_out_features(bits: float, packed_cols: int, logical: tuple[int, int]) -> None:
    quant = QuantSpec(method="awq", bits=bits)
    spec = _specs(quant, ("x.qweight", "I32", (1024, packed_cols)))["x.qweight"]
    assert spec.unpacked_shape == logical
    assert spec.effective_bits == bits


def test_awq_prefers_the_scales_tensor_for_out_features() -> None:
    quant = QuantSpec(method="awq", bits=4.0)
    specs = _specs(quant, ("x.qweight", "I32", (1024, 512)), ("x.scales", "F16", (16, 4093)))
    assert specs["x.qweight"].unpacked_shape == (4093, 1024)


@pytest.mark.parametrize("method", ["gptq", "autoround", "compressed-tensors"])
@pytest.mark.parametrize(("bits", "packed_rows"), [(4.0, 128), (8.0, 256)])
def test_gptq_family_qweight_unpacks_along_in_features(method: str, bits: float, packed_rows: int) -> None:
    quant = QuantSpec(method=method, bits=bits)
    spec = _specs(quant, ("x.qweight", "U32", (packed_rows, 4096)))["x.qweight"]
    assert spec.unpacked_shape == (4096, 1024)
    assert spec.effective_bits == bits


@pytest.mark.parametrize(("dtype", "shape", "logical", "bits"), [("I32", (64, 128), (64, 1024), 4.0), ("U32", (64, 128), (64, 1024), 4.0), ("U8", (64, 512), (64, 1024), 4.0), ("I8", (64, 512), (64, 1024), 4.0)])
def test_weight_packed(dtype: str, shape: tuple[int, int], logical: tuple[int, int], bits: float) -> None:
    spec = _specs(QuantSpec(method="nvfp4", bits=4.0), ("x.weight_packed", dtype, shape))["x.weight_packed"]
    assert spec.unpacked_shape == logical
    assert spec.effective_bits == bits


def test_weight_packed_u8_at_8_bits_is_not_expanded() -> None:
    spec = _specs(QuantSpec(method="compressed-tensors", bits=8.0), ("x.weight_packed", "U8", (64, 1024)))["x.weight_packed"]
    assert spec.unpacked_shape == (64, 1024)
    assert spec.effective_bits == 8.0


@pytest.mark.parametrize(("method", "bits", "cols"), [("nvfp4", 4.0, 2048), ("mxfp4", 4.0, 2048), ("bitnet", 1.58, 4096)])
def test_byte_weights_of_sub_byte_methods(method: str, bits: float, cols: int) -> None:
    spec = _specs(QuantSpec(method=method, bits=bits), ("x.weight", "U8", (64, 1024)))["x.weight"]
    assert spec.unpacked_shape == (64, cols)
    assert spec.effective_bits == bits


def test_byte_weights_of_other_methods_stay_as_stored() -> None:
    spec = _specs(QuantSpec(method="fp8", bits=8.0), ("x.weight", "U8", (64, 1024)))["x.weight"]
    assert spec.logical_shape is None
    assert spec.effective_bits == 8.0


@pytest.mark.parametrize(
    ("name", "dtype", "shape"),
    [("x.qweight", "F16", (8, 8)), ("x.qweight", "I32", (8,)), ("x.weight_packed", "F16", (8, 8)), ("x.weight_packed", "I32", (8,)), ("x.weight", "I32", (8, 8))],
)
def test_tensors_that_are_not_packed_keep_their_shape(name: str, dtype: str, shape: tuple[int, ...]) -> None:
    assert _specs(QuantSpec(method="awq", bits=4.0), (name, dtype, shape))[name].logical_shape is None


def test_spec_keeps_the_raw_header_fields() -> None:
    spec = _specs(QuantSpec(), ("x.weight", "F32", (2, 2)))["x.weight"]
    assert (spec.shard, spec.byte_size, spec.data_offsets, spec.effective_bits) == ("model.safetensors", 64, (0, 64), 32.0)


def test_unknown_quant_bits_fall_back_to_four_for_packing() -> None:
    spec = _specs(QuantSpec(method="awq"), ("x.qweight", "I32", (1024, 512)))["x.qweight"]
    assert spec.effective_bits == 4.0
    assert spec.unpacked_shape == (4096, 1024)


# GGUF metadata


def _gguf(metadata: dict[str, Any], tensors: dict[str, tuple[tuple[int, ...], float, int]] | None = None) -> ParsedGGUFHeader:
    entries = {
        name: RawGGUFTensorEntry(name, 1, "Q4_K", shape, 0, size, bits)
        for name, (shape, bits, size) in (tensors or {}).items()
    }
    return ParsedGGUFHeader(version=3, metadata=metadata, tensors=entries, header_bytes_consumed=64)


def test_gguf_maps_architecture_keys_to_config() -> None:
    md = {
        "general.architecture": "Llama",
        "llama.embedding_length": 512,
        "llama.feed_forward_length": 2048,
        "llama.block_count": 8,
        "llama.attention.head_count": 8,
        "llama.attention.head_count_kv": 2,
        "llama.context_length": 4096,
        "llama.expert_count": 4,
    }
    config, _, _ = normalize_gguf_metadata(_gguf(md))
    expected = {
        "model_type": "llama",
        "architectures": ["llama"],
        "hidden_size": 512,
        "intermediate_size": 2048,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "max_position_embeddings": 4096,
        "num_local_experts": 4,
    }
    assert {key: config[key] for key in expected} == expected


def test_gguf_kv_heads_fall_back_to_head_count_and_experts_stay_optional() -> None:
    config, _, _ = normalize_gguf_metadata(_gguf({"general.architecture": "x", "x.attention.head_count": 8, "x.expert_count": 0}))
    assert config["num_key_value_heads"] == 8
    assert "num_local_experts" not in config


@pytest.mark.parametrize(
    ("metadata", "vocab"),
    [
        ({"general.vocab_size": 7}, 7),
        ({"general.architecture": "x", "x.vocab_size": 9}, 9),
        ({"tokenizer.ggml.tokens": {"__gguf_string_table_len__": 11}}, 11),
        ({"tokenizer.ggml.tokens": ["a", "b"]}, 2),
        ({}, None),
    ],
)
def test_gguf_vocab_size_sources(metadata: dict[str, Any], vocab: int | None) -> None:
    assert normalize_gguf_metadata(_gguf(metadata))[0]["vocab_size"] == vocab


def test_gguf_unknown_architecture_label() -> None:
    assert normalize_gguf_metadata(_gguf({}))[0]["model_type"] == "unknown"


@pytest.mark.parametrize(("ftype", "label"), [(15, "MOSTLY_Q4_K_M"), (0, "ALL_F32"), (999, "ftype_999"), ("x", "gguf"), (None, "gguf")])
def test_gguf_file_type_label(ftype: Any, label: str) -> None:
    assert normalize_gguf_metadata(_gguf({"general.file_type": ftype}))[1].format_version == label


def test_gguf_dominant_bits_ignore_embeddings_and_output() -> None:
    tensors = {
        "token_embd.weight": ((8, 8), 16.0, 9000),
        "output.weight": ((8, 8), 16.0, 9000),
        "blk.0.attn_q.weight": ((8, 8), 4.0, 100),
        "blk.0.ffn_up.weight": ((8, 8), 8.0, 50),
    }
    assert normalize_gguf_metadata(_gguf({}, tensors))[1].bits == 4.0


def test_gguf_dominant_bits_fall_back_to_all_tensors() -> None:
    tensors = {"token_embd.weight": ((8, 8), 16.0, 10), "blk.0.norm.weight": ((8,), 32.0, 500)}
    assert normalize_gguf_metadata(_gguf({}, tensors))[1].bits == 16.0


def test_gguf_without_tensors_has_no_bit_width() -> None:
    quant = normalize_gguf_metadata(_gguf({}))[1]
    assert quant.method == "gguf"
    assert quant.bits is None


def test_gguf_specs_expose_the_python_shape() -> None:
    spec = normalize_gguf_metadata(_gguf({}, {"a.weight": ((4, 8), 4.0, 12)}))[2]["a.weight"]
    assert (spec.shard, spec.storage_dtype, spec.unpacked_shape, spec.byte_size) == ("gguf", "Q4_K", (4, 8), 12)


@pytest.mark.parametrize("value", [0, -4, float("inf"), float("nan"), True, None, "4", [4], 10**400])
def test_unusable_bit_widths_fall_back(value: Any) -> None:
    assert _bits(value, 4.0) == 4.0


@pytest.mark.parametrize("value", [1, 2.5, 8])
def test_usable_bit_widths_are_kept(value: Any) -> None:
    assert _bits(value, 4.0) == float(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([128, 128], (128, 128)),
        ([1, 2], (1, 2)),
        ([0, 128], None),
        ([128, -1], None),
        ([True, True], None),
        (["a", 1], None),
        ([128], None),
        ("128", None),
    ],
)
def test_block_sizes(value: Any, expected: Any) -> None:
    assert _block_size(value) == expected


def test_str_tuple_uses_the_first_non_empty_list() -> None:
    assert _str_tuple(None, [], ["a", 1], ["b"]) == ("a",)
    assert _str_tuple("x", 5) == ()


@pytest.mark.parametrize(
    ("qcfg", "method"),
    [
        ({"producer": {"name": "ModelOpt"}}, "fp8"),
        ({"producer": {"name": "other"}}, "custom"),
        ({"producer": {"name": None}}, "custom"),
        ({"producer": "ModelOpt"}, "custom"),
        ({"group_size": 32}, "custom"),
        ({"damp_percent": 0.1}, "custom"),
        ({"bits": 4}, "custom"),
    ],
)
def test_method_detection_without_a_quant_method(qcfg: dict[str, Any], method: str) -> None:
    assert _parse(**qcfg).method == method


@pytest.mark.parametrize(
    "config",
    [
        {"quantization": {"bits": 4, "group_size": 64}},
        {"quantization": {"bits": 4, "group_size": 64}, "quantization_config": {"bits": 4, "group_size": 64}},
    ],
)
def test_mlx_quantization_is_recognised(config: dict[str, Any]) -> None:
    spec = parse_quant_spec_from_config(config)
    assert (spec.method, spec.bits, spec.group_size) == ("mlx", 4.0, 64)


def test_a_real_method_wins_over_a_stray_quantization_block() -> None:
    config = {"quantization": {"bits": 4}, "quantization_config": {"quant_method": "awq", "bits": 4}}
    assert parse_quant_spec_from_config(config).method == "awq"


def test_quantization_block_without_bits_is_ignored() -> None:
    assert parse_quant_spec_from_config({"quantization": {"quant_algo": "FP8"}}).method == "none"
