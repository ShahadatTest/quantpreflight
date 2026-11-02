"""Verification suite for the 20 real closed/documented vLLM deployment issues."""

from __future__ import annotations

import json
from pathlib import Path

from quantpreflight.models import CheckpointMeta, TargetSpec, TokenizerMeta
from quantpreflight.quant_detect import parse_quant_spec_from_config
from quantpreflight.rules import RULE_CATALOG, analyze_checkpoint

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmark"


def test_vllm_20_issues_catalog_integrity() -> None:
    catalog_path = BENCHMARK_DIR / "vllm_20_issues.json"
    assert catalog_path.is_file()
    issues = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert len(issues) == 20

    for item in issues:
        assert item["issue_id"].startswith("vllm#")
        assert item["url"].startswith("https://github.com/vllm-project/vllm/issues/")
        assert len(item["target_rules"]) >= 1
        for rule_id in item["target_rules"]:
            assert rule_id in RULE_CATALOG


def test_vllm_issue_58241_block_fp8_on_sm89() -> None:
    """vllm#58241: cutlass_scaled_mm_supports_block_fp8 rejects SM89 (L4/L40S/RTX 4090)."""
    cfg = {
        "architectures": ["Qwen2ForCausalLM"],
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    }
    meta = CheckpointMeta(
        source="mock/qwen-fp8-block",
        container_format="safetensors",
        config=cfg,
        quant=parse_quant_spec_from_config(cfg),
    )
    report = analyze_checkpoint(meta, TargetSpec(engine="vllm", gpu="L40S", tp=1))
    fired = {f.rule_id for f in report.findings}
    assert "QPF004" in fired
    assert any("58241" in ref for f in report.findings for ref in f.references)


def test_vllm_issue_49546_marlin_fp8_gb10_corruption() -> None:
    """vllm#49546: VLLM_MARLIN_INPUT_DTYPE=fp8 silently corrupts output on GB10/sm_121a."""
    cfg = {
        "architectures": ["Qwen2ForCausalLM"],
        "quantization_config": {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
        },
    }
    meta = CheckpointMeta(
        source="mock/qwen-awq-4bit",
        container_format="safetensors",
        config=cfg,
        quant=parse_quant_spec_from_config(cfg),
    )
    report = analyze_checkpoint(
        meta,
        TargetSpec(engine="vllm", gpu="GB10", tp=1, marlin_input_dtype="fp8"),
    )
    fired = {f.rule_id for f in report.findings}
    assert "QPF005" in fired
    assert any("49546" in ref for f in report.findings for ref in f.references)


def test_vllm_issue_48904_autoround_marlin_int8_ignored() -> None:
    """vllm#48904: VLLM_MARLIN_INPUT_DTYPE=int8 silently ignored on auto-round checkpoints."""
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "quantization_config": {
            "quant_method": "auto-round",
            "bits": 4,
            "group_size": 128,
        },
    }
    meta = CheckpointMeta(
        source="mock/llama-autoround",
        container_format="safetensors",
        config=cfg,
        quant=parse_quant_spec_from_config(cfg),
    )
    report = analyze_checkpoint(
        meta,
        TargetSpec(engine="vllm", gpu="A100", tp=1, marlin_input_dtype="int8"),
    )
    fired = {f.rule_id for f in report.findings}
    assert "QPF005" in fired
    assert any("48904" in ref for f in report.findings for ref in f.references)


def test_vllm_issue_54195_oob_stop_token_id() -> None:
    """vllm#54195: Out-of-vocab stop_token_ids crashes the whole engine."""
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": 32000,
        "eos_token_id": 35000,
    }
    from quantpreflight.models import TensorSpec

    embed = TensorSpec(
        name="model.embed_tokens.weight",
        storage_dtype="BF16",
        shape=(32000, 4096),
        shard="model.safetensors",
        byte_size=32000 * 4096 * 2,
        effective_bits=16.0,
    )
    meta = CheckpointMeta(
        source="mock/llama-bad-eos",
        container_format="safetensors",
        config=cfg,
        tensors={"model.embed_tokens.weight": embed},
    )
    report = analyze_checkpoint(meta, TargetSpec(engine="vllm", gpu="A100", tp=1))
    fired = {f.rule_id for f in report.findings}
    assert "QPF002" in fired


def test_vllm_issue_43888_desc_act_tp2_penalty() -> None:
    """vllm#43888: GPTQ desc_act=true causes activation gather and kernel overhead on TP > 1."""
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "quantization_config": {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "desc_act": True,
        },
    }
    meta = CheckpointMeta(
        source="mock/gptq-desc-act",
        container_format="safetensors",
        config=cfg,
        quant=parse_quant_spec_from_config(cfg),
    )
    report = analyze_checkpoint(meta, TargetSpec(engine="vllm", gpu="A100", tp=2))
    assert "QPF008" in {f.rule_id for f in report.findings}


def test_vllm_issue_45922_gqa_kv_heads_indivisible() -> None:
    """vllm#45922: GQA num_key_value_heads=8 cannot be partitioned across TP=3 ranks."""
    cfg = {
        "architectures": ["Qwen2ForCausalLM"],
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
    }
    meta = CheckpointMeta(source="mock/qwen-tp3", container_format="safetensors", config=cfg)
    report = analyze_checkpoint(meta, TargetSpec(engine="vllm", gpu="A100", tp=3))
    assert "QPF006" in {f.rule_id for f in report.findings}


def test_vllm_nvfp4_on_hopper_h100_rejected() -> None:
    """NVFP4 requires Blackwell sm_100+ tensor cores; Hopper H100 sm_90 is incompatible."""
    cfg = {
        "architectures": ["Qwen2ForCausalLM"],
        "quantization_config": {"quant_method": "nvfp4"},
    }
    meta = CheckpointMeta(
        source="mock/nvfp4",
        container_format="safetensors",
        config=cfg,
        quant=parse_quant_spec_from_config(cfg),
    )
    report = analyze_checkpoint(meta, TargetSpec(engine="vllm", gpu="H100", tp=1))
    assert "QPF004" in {f.rule_id for f in report.findings}


def test_vllm_safetensors_on_llamacpp_rejected() -> None:
    """llama.cpp requires GGUF format; passing .safetensors directly is rejected."""
    cfg = {"architectures": ["LlamaForCausalLM"]}
    meta = CheckpointMeta(source="mock/llama", container_format="safetensors", config=cfg)
    report = analyze_checkpoint(meta, TargetSpec(engine="llamacpp", gpu="L4", tp=1))
    assert "QPF011" in {f.rule_id for f in report.findings}
