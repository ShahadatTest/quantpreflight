"""Check a pasted config.json or a fetched model, in the site build and in the browser."""

from __future__ import annotations

import json
import struct
from typing import Any

from quantpreflight.cli import SCAN_ERRORS
from quantpreflight.loader import _extract_tokenizer_meta
from quantpreflight.models import CheckpointMeta, TargetSpec
from quantpreflight.quant_detect import build_safetensors_specs, parse_quant_spec_from_config
from quantpreflight.rules import analyze_checkpoint
from quantpreflight.safetensors_io import parse_safetensors_header_bytes, parse_safetensors_index

MODEL_KEYS = ("architectures", "model_type", "quantization_config", "hidden_size", "text_config")


def scan_text(text: str, engine: str = "vllm", gpu: str = "", tp: int = 1, weights_gib: float = 0.0) -> str:
    """Return the check of one `config.json` as JSON for the page."""
    try:
        config = json.loads(text)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"This isn't valid JSON near line {exc.lineno}."})
    except RecursionError:
        return json.dumps({"error": "This JSON is nested too deeply to read."})

    if not isinstance(config, dict) or not any(key in config for key in MODEL_KEYS):
        return json.dumps({"error": "This doesn't look like a model config.json."})

    meta = CheckpointMeta(
        source="config.json",
        container_format="safetensors",
        config=config,
        total_weight_bytes=int(weights_gib * 1024**3),
        quant=parse_quant_spec_from_config(config),
    )
    return json.dumps(_summary(meta, engine, gpu, tp))


def _summary(meta: CheckpointMeta, engine: str, gpu: str, tp: int) -> dict[str, Any]:
    report = analyze_checkpoint(meta, TargetSpec(engine=engine, gpu=gpu or None, tp=tp))
    return {
        "quant": report.quant_summary.replace("unquantized (unknown)", "unquantized"),
        "vram_gib": report.estimated_vram_per_gpu_gb,
        "findings": [finding.to_dict() for finding in report.findings],
    }


def _json_file(files: dict[str, str | None], name: str) -> dict[str, Any]:
    text = files.get(name)
    if not text:
        return {}
    value = json.loads(text)
    return value if isinstance(value, dict) else {}


def _build_meta(payload: dict[str, Any]) -> CheckpointMeta:
    files, shards, siblings = payload["files"], payload["shards"], payload["siblings"]
    config = _json_file(files, "config.json")
    if not config.get("quantization_config"):
        external = _json_file(files, "hf_quant_config.json") or _json_file(files, "quantize_config.json")
        if external:
            config["quantization_config"] = external
    tokenizer_config = _json_file(files, "tokenizer_config.json")
    index = _json_file(files, "model.safetensors.index.json")
    weight_map, total_size = parse_safetensors_index(index) if index else (None, None)

    raw_entries: dict[str, Any] = {}
    duplicates: list[str] = []
    missing: list[str] = []
    for name, header in shards.items():
        if header is None:
            missing.append(name)
            continue
        raw = header.encode("utf-8")
        parsed = parse_safetensors_header_bytes(struct.pack("<Q", len(raw)), raw)
        for tensor, entry in parsed.tensors.items():
            if tensor in raw_entries:
                duplicates.append(tensor)
            raw_entries[tensor] = (entry, name)

    dangling = [
        tensor for tensor, name in (weight_map or {}).items() if name not in missing and tensor not in raw_entries
    ]
    quant = parse_quant_spec_from_config(config)
    specs = build_safetensors_specs(raw_entries, quant)
    calculated = sum(spec.byte_size for spec in specs.values())
    legacy = [
        name
        for name in siblings
        if name.lower().endswith((".bin", ".pt", ".pth"))
        and any(word in name.lower() for word in ("model", "pytorch", "weight", "consolidated"))
    ]
    return CheckpointMeta(
        source=payload["repo"],
        container_format="safetensors",
        config=config,
        generation_config=_json_file(files, "generation_config.json"),
        tokenizer_config=tokenizer_config,
        tokenizer_meta=_extract_tokenizer_meta(None, tokenizer_config, config_vocab_size=config.get("vocab_size")),
        tensors=specs,
        shard_files=tuple(shards),
        missing_shards=tuple(missing),
        dangling_index_tensors=tuple(sorted(dangling)),
        duplicate_tensors=tuple(sorted(set(duplicates))),
        legacy_weight_files=tuple(sorted(legacy)),
        custom_code_files=tuple(sorted(name for name in siblings if name.lower().endswith(".py"))),
        total_weight_bytes=total_size if (total_size and missing) else calculated,
        quant=quant,
    )


def scan_repo(payload: str, engine: str = "vllm", gpu: str = "", tp: int = 1) -> str:
    """Check files the page fetched from Hugging Face. The page sends them as JSON."""
    try:
        meta = _build_meta(json.loads(payload))
        result = _summary(meta, engine, gpu, tp)
    except SCAN_ERRORS as exc:
        return json.dumps({"error": f"This model couldn't be read: {exc}"})
    result["weights_gib"] = round(meta.total_weight_bytes / 1024**3, 2)
    result["tensors"] = len(meta.tensors)
    result["config"] = json.dumps(meta.config, indent=2)
    return json.dumps(result)
