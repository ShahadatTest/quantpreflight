"""Unit and adversarial regression tests for pre-flight rules QPF001 - QPF012."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import build_gguf_header_bytes, build_safetensors_header_file, write_checkpoint_dir
from quantpreflight.loader import load_checkpoint_metadata
from quantpreflight.models import TargetSpec
from quantpreflight.rules import analyze_checkpoint


def test_adversarial_1_sharded_hybrid_moe_router_and_ssm_quantized(tmp_path: Path) -> None:
    """Adversarial Test 1: Sharded Qwen3-Next hybrid model with quantized MoE gate & A_log."""
    ckpt = tmp_path / "qwen3_next_bad_fp8"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3NextForCausalLM"],
                "model_type": "qwen3_next",
                "vocab_size": 32000,
                "hidden_size": 2048,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic"},
            }
        ),
        encoding="utf-8",
    )
    shard1 = "model-00001-of-00002.safetensors"
    shard2 = "model-00002-of-00002.safetensors"
    (ckpt / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": shard1,
                    "model.layers.0.mlp.experts.0.gate_proj.weight": shard1,
                    "model.layers.0.mlp.gate.weight": shard2,
                    "model.layers.0.linear_attn.A_log": shard2,
                    "lm_head.weight": shard2,
                }
            }
        ),
        encoding="utf-8",
    )
    (ckpt / shard1).write_bytes(
        build_safetensors_header_file(
            {
                "model.embed_tokens.weight": ("BF16", [32000, 2048]),
                "model.layers.0.mlp.experts.0.gate_proj.weight": ("F8_E4M3", [1024, 2048]),
            }
        )
    )
    (ckpt / shard2).write_bytes(
        build_safetensors_header_file(
            {
                "model.layers.0.mlp.gate.weight": ("F8_E4M3", [64, 2048]),
                "model.layers.0.linear_attn.A_log": ("F8_E4M3", [16]),
                "lm_head.weight": ("BF16", [32000, 2048]),
            }
        )
    )

    meta = load_checkpoint_metadata(str(ckpt))
    report = analyze_checkpoint(meta, TargetSpec(engine="vllm", gpu="H100", tp=1))
    qpf001 = [f for f in report.findings if f.rule_id == "QPF001"]
    assert len(qpf001) == 2
    flagged_tensors = {t for f in qpf001 for t in f.tensors}
    assert "model.layers.0.mlp.gate.weight" in flagged_tensors
    assert "model.layers.0.linear_attn.A_log" in flagged_tensors
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" not in flagged_tensors


def test_adversarial_2_off_by_one_added_tokens_cuda_assert(tmp_path: Path) -> None:
    """Adversarial Test 2: CVE-2026-115916 pattern where added_tokens ID == embed_tokens.shape[0]."""
    ckpt = write_checkpoint_dir(
        tmp_path / "oob_tok",
        config={
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        },
        tensors={
            "model.embed_tokens.weight": ("BF16", [32000, 4096]),
            "lm_head.weight": ("BF16", [32000, 4096]),
        },
        tokenizer_json={
            "model": {"vocab": {"<s>": 1, "</s>": 2}},
            "added_tokens": [{"id": 32000, "content": "<tool_call>"}],
        },
    )
    meta = load_checkpoint_metadata(str(ckpt))
    report = analyze_checkpoint(meta, TargetSpec(engine="vllm", gpu="A100", tp=1))
    qpf002 = [f for f in report.findings if f.rule_id == "QPF002"]
    assert len(qpf002) >= 1
    assert any("<tool_call>" in f.message and "32000" in f.message for f in qpf002)


def test_adversarial_3_marlin_and_tp4_gqa_head_and_shape_crash(tmp_path: Path) -> None:
    """Adversarial Test 3: AWQ 4-bit with 6 KV heads and intermediate_size=11016 on A100 TP=4."""
    ckpt = write_checkpoint_dir(
        tmp_path / "awq_tp4_bad",
        config={
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11016,
            "num_attention_heads": 32,
            "num_key_value_heads": 6,
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "version": "gemm",
            },
        },
        tensors={
            "model.embed_tokens.weight": ("F16", [32000, 4096]),
            # AWQ gate_proj: in=4096, out=11016 -> packed [4096, 11016 // 8] is not integer, use scales to prove exact 11016 or row-parallel down_proj [11016, 512]
            "model.layers.0.mlp.down_proj.qweight": ("I32", [11016, 512]),
            "model.layers.0.mlp.down_proj.scales": ("F16", [86, 4096]),
        },
    )
    meta = load_checkpoint_metadata(str(ckpt))
    report = analyze_checkpoint(meta, TargetSpec(engine="vllm", gpu="A100", tp=4))
    rule_ids = {f.rule_id for f in report.findings}
    assert "QPF006" in rule_ids  # 6 % 4 != 0
    assert "QPF007" in rule_ids  # 11016 / 4 = 2754, violates group_size=128 and Marlin 128-alignment
    assert any("2754" in f.message for f in report.findings if f.rule_id == "QPF007")


def test_dangling_index_tensor_and_missing_shard_caught(tmp_path: Path) -> None:
    ckpt = tmp_path / "broken_sharded"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}), encoding="utf-8")
    (ckpt / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "model.ghost_layer.weight": "model-00001-of-00002.safetensors",
                    "lm_head.weight": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (ckpt / "model-00001-of-00002.safetensors").write_bytes(
        build_safetensors_header_file({"model.embed_tokens.weight": ("BF16", [1000, 512])})
    )
    # Note: shard 2 is missing AND model.ghost_layer.weight is missing from shard 1!
    meta = load_checkpoint_metadata(str(ckpt))
    report = analyze_checkpoint(meta)
    qpf009 = [f for f in report.findings if f.rule_id == "QPF009"]
    assert len(qpf009) == 2


def test_gguf_iq2_xxs_on_vllm_flags_qpf001_and_qpf011(tmp_path: Path) -> None:
    gguf_file = tmp_path / "tiny-iq2.gguf"
    gguf_file.write_bytes(
        build_gguf_header_bytes(
            kv_strings={"general.architecture": "llama"},
            kv_u32={
                "general.file_type": 19,
                "llama.embedding_length": 2048,
                "llama.block_count": 16,
                "llama.attention.head_count": 16,
                "llama.attention.head_count_kv": 4,
            },
            tokens=["<unk>", "<s>", "</s>"],
            tensors=[
                ("token_embd.weight", (2048, 3), 16),  # IQ2_XXS (2-bit embedding!)
                ("blk.0.ffn_down.weight", (4096, 2048), 16),
            ],
        )
    )
    meta = load_checkpoint_metadata(str(gguf_file))
    report = analyze_checkpoint(meta, TargetSpec(engine="vllm", gpu="A100", tp=1))
    fired = {f.rule_id for f in report.findings}
    assert "QPF001" in fired  # 2-bit token_embd < 4-bit minimum
    assert "QPF011" in fired  # IQ2_XXS GGUF on vLLM
