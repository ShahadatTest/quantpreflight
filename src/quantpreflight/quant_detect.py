"""Normalize quantization metadata and compute per-tensor bit-widths and logical shapes."""

from __future__ import annotations

import math
from typing import Any

from .gguf_io import GGUF_FILE_TYPE_NAMES, ParsedGGUFHeader
from .models import QuantSpec, TensorSpec
from .safetensors_io import RawSafetensorEntry, SAFETENSORS_DTYPE_BITS


def _bits(value: Any, default: float) -> float:
    """Return a usable bit-width from an untrusted config value, or `default`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    try:
        bits = float(value)
    except OverflowError:
        return default
    return bits if math.isfinite(bits) and bits > 0 else default


def _str_tuple(*candidates: Any) -> tuple[str, ...]:
    """Return the string items of the first non-empty list among `candidates`."""
    for value in candidates:
        if isinstance(value, list) and value:
            return tuple(x for x in value if isinstance(x, str))
    return ()


def _block_size(value: Any) -> tuple[int, int] | None:
    """Return a `(height, width)` FP8 block shape, or `None` unless both sides are positive."""
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in value)
    ):
        return value[0], value[1]
    return None


def _parse_compressed_tensors_config(qcfg: dict[str, Any]) -> QuantSpec:
    fmt = str(qcfg.get("format") or "compressed-tensors")
    ignored = _str_tuple(qcfg.get("ignore"))
    groups = qcfg.get("config_groups") or {}

    bits: float | None = None
    group_size: int | None = None
    sym: bool | None = None
    act_bits: float | None = None
    act_dtype: str | None = None
    block_size: tuple[int, int] | None = None

    if isinstance(groups, dict):
        for grp in groups.values():
            if not isinstance(grp, dict):
                continue
            weights = grp.get("weights")
            if isinstance(weights, dict):
                num_bits = _bits(weights.get("num_bits"), 0.0)
                if num_bits:
                    bits = num_bits
                if isinstance(weights.get("group_size"), int) and weights["group_size"] > 0:
                    group_size = weights["group_size"]
                if isinstance(weights.get("symmetric"), bool):
                    sym = weights["symmetric"]
                wtype = str(weights.get("type") or "").lower()
                if fmt == "nvfp4-pack-quantized" or (bits == 4.0 and wtype == "float"):
                    fmt = "nvfp4-pack-quantized"
                block_size = _block_size(weights.get("block_structure")) or block_size

            acts = grp.get("input_activations")
            act_num_bits = _bits(acts.get("num_bits"), 0.0) if isinstance(acts, dict) else 0.0
            if act_num_bits:
                act_bits = act_num_bits
                atype = str(acts.get("type") or "").lower()
                if act_bits == 8.0:
                    act_dtype = "fp8" if atype == "float" else "int8"
                elif act_bits == 4.0:
                    act_dtype = "fp4"
            break

    method = "nvfp4" if fmt == "nvfp4-pack-quantized" else "compressed-tensors"
    return QuantSpec(
        method=method,
        bits=bits,
        group_size=group_size,
        sym=sym,
        desc_act=False,
        zero_point=(not sym) if sym is not None else None,
        format_version=fmt,
        activation_bits=act_bits,
        activation_dtype=act_dtype,
        weight_block_size=block_size,
        ignored_layers=ignored,
        raw_config=qcfg,
    )


def parse_quant_spec_from_config(config: dict[str, Any]) -> QuantSpec:
    """Extract normalized `QuantSpec` from a Hugging Face `config.json` dictionary."""
    qcfg = config.get("quantization_config")
    mlx = config.get("quantization")
    is_mlx = isinstance(mlx, dict) and "bits" in mlx
    if not isinstance(qcfg, dict) or not qcfg:
        if not is_mlx:
            return QuantSpec(method="none")
        qcfg = mlx
    if is_mlx and "quant_method" not in qcfg and "quant_type" not in qcfg:
        group_size = qcfg.get("group_size")
        return QuantSpec(
            method="mlx",
            bits=_bits(qcfg.get("bits"), 4.0),
            group_size=group_size if isinstance(group_size, int) else None,
            raw_config=qcfg,
        )

    raw_method = str(qcfg.get("quant_method") or qcfg.get("quant_type") or "").strip().lower()
    if not raw_method:
        prod = qcfg.get("producer")
        if isinstance(prod, dict) and str(prod.get("name") or "").lower() == "modelopt":
            raw_method = "modelopt"
        elif "quantization" in qcfg and isinstance(qcfg["quantization"], dict) and "quant_algo" in qcfg["quantization"]:
            raw_method = "modelopt"
        elif "quant_algo" in qcfg:
            raw_method = "modelopt"
        elif "bits" in qcfg and ("group_size" in qcfg or "damp_percent" in qcfg):
            raw_method = "gptq"

    if raw_method == "compressed-tensors" or "config_groups" in qcfg:
        return _parse_compressed_tensors_config(qcfg)

    if raw_method in ("awq", "autoawq"):
        bits = _bits(qcfg.get("bits") or qcfg.get("w_bit"), 4.0)
        group_size = qcfg.get("group_size")
        if group_size is None:
            group_size = qcfg.get("q_group_size")
        zp = qcfg.get("zero_point")
        version = str(qcfg.get("version") or "gemm").lower()
        ignored = _str_tuple(qcfg.get("modules_to_not_convert"))
        return QuantSpec(
            method="awq",
            bits=bits,
            group_size=int(group_size) if isinstance(group_size, int) else None,
            sym=(not zp) if isinstance(zp, bool) else None,
            desc_act=False,
            zero_point=zp if isinstance(zp, bool) else True,
            format_version=version,
            activation_bits=16.0,
            activation_dtype="fp16",
            ignored_layers=ignored,
            raw_config=qcfg,
        )

    if raw_method in ("gptq", "gptq_v2"):
        bits = _bits(qcfg.get("bits"), 4.0)
        group_size = qcfg.get("group_size")
        sym = qcfg.get("sym")
        desc_act = bool(qcfg.get("desc_act", False))
        is_marlin = bool(qcfg.get("checkpoint_format") == "marlin" or qcfg.get("is_marlin_format"))
        return QuantSpec(
            method="gptq",
            bits=bits,
            group_size=int(group_size) if isinstance(group_size, int) and group_size > 0 else None,
            sym=sym if isinstance(sym, bool) else True,
            desc_act=desc_act,
            zero_point=(not sym) if isinstance(sym, bool) else False,
            format_version="marlin" if is_marlin else "gptq",
            activation_bits=16.0,
            activation_dtype="fp16",
            raw_config=qcfg,
        )

    if raw_method in ("auto-round", "autoround", "inc"):
        bits = _bits(qcfg.get("bits"), 4.0)
        group_size = qcfg.get("group_size")
        sym = qcfg.get("sym")
        packing = str(qcfg.get("packing_format") or qcfg.get("backend") or "auto_round").lower()
        return QuantSpec(
            method="autoround",
            bits=bits,
            group_size=int(group_size) if isinstance(group_size, int) and group_size > 0 else None,
            sym=sym if isinstance(sym, bool) else True,
            desc_act=bool(qcfg.get("desc_act", False)),
            format_version=packing,
            activation_bits=16.0,
            activation_dtype="fp16",
            raw_config=qcfg,
        )

    if raw_method in ("fp8", "finegrained_fp8", "fbgemm_fp8"):
        block_tuple = _block_size(qcfg.get("weight_block_size"))
        act_scheme = str(qcfg.get("activation_scheme") or "dynamic").lower()
        ignored = _str_tuple(qcfg.get("modules_to_not_convert"), qcfg.get("ignored_layers"))
        return QuantSpec(
            method="fp8",
            bits=8.0,
            sym=True,
            format_version="block-fp8" if block_tuple else act_scheme,
            activation_bits=8.0,
            activation_dtype="fp8",
            weight_block_size=block_tuple,
            ignored_layers=ignored,
            raw_config=qcfg,
        )

    if raw_method in ("modelopt", "nvfp4", "mxfp4") or "quant_algo" in qcfg:
        q_sub = qcfg.get("quantization") if isinstance(qcfg.get("quantization"), dict) else qcfg
        algo = str(q_sub.get("quant_algo") or qcfg.get("quant_algo") or raw_method).lower()
        is_fp4 = "fp4" in algo or raw_method in ("nvfp4", "mxfp4")
        method = "mxfp4" if "mxfp4" in algo or raw_method == "mxfp4" else ("nvfp4" if is_fp4 else "fp8")
        bits = 4.0 if is_fp4 else 8.0
        group_size = q_sub.get("group_size") or (16 if method == "nvfp4" else (32 if method == "mxfp4" else None))
        ignored = _str_tuple(
            q_sub.get("exclude_modules"),
            q_sub.get("modules_to_not_convert"),
            qcfg.get("exclude_modules"),
            qcfg.get("modules_to_not_convert"),
        )
        return QuantSpec(
            method=method,
            bits=bits,
            group_size=int(group_size) if isinstance(group_size, int) else None,
            sym=True,
            format_version=algo,
            activation_bits=bits,
            activation_dtype="fp4" if is_fp4 else "fp8",
            ignored_layers=ignored,
            raw_config=qcfg,
        )

    if raw_method in ("bitsandbytes", "bnb") or qcfg.get("load_in_4bit") or qcfg.get("load_in_8bit"):
        is_4bit = bool(qcfg.get("load_in_4bit")) or str(qcfg.get("bnb_4bit_quant_type") or "") != ""
        bits = 4.0 if is_4bit else 8.0
        qtype = str(qcfg.get("bnb_4bit_quant_type") or ("nf4" if is_4bit else "llm.int8"))
        return QuantSpec(
            method="bitsandbytes",
            bits=bits,
            format_version=qtype,
            activation_bits=16.0 if is_4bit else 8.0,
            raw_config=qcfg,
        )

    if raw_method in ("bitnet", "1bit", "ternary"):
        return QuantSpec(
            method="bitnet",
            bits=1.58,
            format_version=raw_method,
            activation_bits=8.0,
            activation_dtype="int8",
            raw_config=qcfg,
        )

    bits_val = _bits(qcfg.get("bits") or qcfg.get("w_bit"), 0.0)
    return QuantSpec(
        method=raw_method or "custom",
        bits=bits_val or None,
        group_size=qcfg.get("group_size") if isinstance(qcfg.get("group_size"), int) else None,
        raw_config=qcfg,
    )


def build_safetensors_specs(
    raw_entries: dict[str, tuple[RawSafetensorEntry, str]],
    quant: QuantSpec,
) -> dict[str, TensorSpec]:
    """Convert raw Safetensors header entries into `TensorSpec` objects with unpacked shapes."""
    specs: dict[str, TensorSpec] = {}
    cfg_bits = quant.bits or 4.0

    for name, (entry, shard_name) in raw_entries.items():
        dtype = entry.dtype
        shape = entry.shape
        eff_bits = SAFETENSORS_DTYPE_BITS.get(dtype, 16.0)
        logical_shape: tuple[int, ...] | None = None

        if name.endswith(".qweight") and dtype in ("I32", "U32") and len(shape) == 2:
            pack_factor = max(1, int(round(32.0 / cfg_bits)))
            eff_bits = cfg_bits
            prefix = name[: -len(".qweight")]
            scales_item = raw_entries.get(f"{prefix}.scales")
            if quant.method == "awq":
                # AWQ GEMM packs along out_features: qweight is [in_features, out_features // pack_factor]
                in_f = shape[0]
                out_f = scales_item[0].shape[1] if (scales_item and len(scales_item[0].shape) == 2) else shape[1] * pack_factor
                logical_shape = (out_f, in_f)
            else:
                # GPTQ / AutoRound pack along in_features: qweight is [in_features // pack_factor, out_features]
                in_f = shape[0] * pack_factor
                out_f = shape[1]
                logical_shape = (out_f, in_f)

        elif name.endswith(".weight_packed") and len(shape) == 2:
            if dtype in ("I32", "U32"):
                pack_factor = max(1, int(round(32.0 / cfg_bits)))
                eff_bits = cfg_bits
                logical_shape = (shape[0], shape[1] * pack_factor)
            elif dtype in ("U8", "I8"):
                # NVFP4 / MXFP4 packs two 4-bit E2M1 values per U8 byte
                eff_bits = 4.0 if cfg_bits <= 4.0 else cfg_bits
                pack_factor = max(1, int(round(8.0 / eff_bits)))
                logical_shape = (shape[0], shape[1] * pack_factor)

        elif (
            name.endswith(".weight")
            and dtype in ("U8", "I8")
            and len(shape) == 2
            and quant.method in ("nvfp4", "mxfp4", "bitnet")
        ):
            eff_bits = 1.58 if quant.method == "bitnet" else 4.0
            pack_factor = 4 if quant.method == "bitnet" else 2
            logical_shape = (shape[0], shape[1] * pack_factor)

        specs[name] = TensorSpec(
            name=name,
            storage_dtype=dtype,
            shape=shape,
            shard=shard_name,
            byte_size=entry.byte_size,
            effective_bits=eff_bits,
            logical_shape=logical_shape,
            data_offsets=entry.data_offsets,
        )

    return specs


def normalize_gguf_metadata(parsed: ParsedGGUFHeader) -> tuple[dict[str, Any], QuantSpec, dict[str, TensorSpec]]:
    """Map GGUF KV metadata and tensors into unified `config`, `QuantSpec`, and `TensorSpec` objects."""
    md = parsed.metadata
    arch = str(md.get("general.architecture") or "unknown").lower()

    def _get_arch_key(suffix: str) -> Any:
        return md.get(f"{arch}.{suffix}")

    vocab_size: int | None = md.get("general.vocab_size") or _get_arch_key("vocab_size")
    tok_table = md.get("tokenizer.ggml.tokens")
    if vocab_size is None:
        if isinstance(tok_table, dict) and "__gguf_string_table_len__" in tok_table:
            vocab_size = int(tok_table["__gguf_string_table_len__"])
        elif isinstance(tok_table, list):
            vocab_size = len(tok_table)

    config: dict[str, Any] = {
        "model_type": arch,
        "architectures": [arch],
        "hidden_size": _get_arch_key("embedding_length"),
        "intermediate_size": _get_arch_key("feed_forward_length"),
        "num_hidden_layers": _get_arch_key("block_count"),
        "num_attention_heads": _get_arch_key("attention.head_count"),
        "num_key_value_heads": _get_arch_key("attention.head_count_kv") or _get_arch_key("attention.head_count"),
        "max_position_embeddings": _get_arch_key("context_length"),
        "vocab_size": vocab_size,
    }
    experts = _get_arch_key("expert_count")
    if isinstance(experts, int) and experts > 0:
        config["num_local_experts"] = experts

    specs: dict[str, TensorSpec] = {}
    block_bit_counts: dict[float, int] = {}
    all_bit_counts: dict[float, int] = {}
    for tname, entry in parsed.tensors.items():
        specs[tname] = TensorSpec(
            name=tname,
            storage_dtype=entry.dtype_name,
            shape=entry.shape,
            shard="gguf",
            byte_size=entry.byte_size,
            effective_bits=entry.effective_bits,
            logical_shape=entry.shape,
        )
        if len(entry.shape) >= 2:
            all_bit_counts[entry.effective_bits] = all_bit_counts.get(entry.effective_bits, 0) + entry.byte_size
            if not tname.startswith(("token_embd.", "output.")):
                block_bit_counts[entry.effective_bits] = (
                    block_bit_counts.get(entry.effective_bits, 0) + entry.byte_size
                )

    chosen_counts = block_bit_counts or all_bit_counts
    dominant_bits = max(chosen_counts.items(), key=lambda kv: kv[1])[0] if chosen_counts else None
    ftype = md.get("general.file_type")
    ftype_label = GGUF_FILE_TYPE_NAMES.get(ftype, f"ftype_{ftype}") if isinstance(ftype, int) else "gguf"

    quant = QuantSpec(
        method="gguf",
        bits=dominant_bits,
        format_version=ftype_label,
        raw_config={"file_type": ftype, "quantization_version": md.get("general.quantization_version")},
    )
    return config, quant, specs
