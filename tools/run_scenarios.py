"""Check scenarios of broken, misconfigured, or incompatible checkpoints against live models."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import build_safetensors_header_file, write_checkpoint_dir
from quantpreflight.loader import load_checkpoint_metadata
from quantpreflight.models import CheckpointMeta, TargetSpec, TensorSpec
from quantpreflight.quant_detect import parse_quant_spec_from_config
from quantpreflight.rules import analyze_checkpoint


SCENARIOS = [
    # --- Group 1: Hardware & Kernel Incompatibilities (QPF004 & QPF005) ---
    {
        "id": "defect_01_nvfp4_on_hopper_h100",
        "description": "NVIDIA ModelOpt NVFP4 checkpoint targeted at H100 (sm_90 lacks NVFP4 tensor cores; requires sm_100+)",
        "target": "nvidia/Qwen3-8B-FP4",
        "engine": "vllm", "gpu": "H100", "tp": 1,
        "must_catch": ["QPF004"],
    },
    {
        "id": "defect_02_nvfp4_on_ampere_a100",
        "description": "NVIDIA ModelOpt NVFP4 checkpoint targeted at A100 (sm_80 lacks NVFP4 tensor cores)",
        "target": "nvidia/Qwen3-8B-FP4",
        "engine": "vllm", "gpu": "A100", "tp": 1,
        "must_catch": ["QPF004"],
    },
    {
        "id": "defect_03_nvfp4_on_turing_t4",
        "description": "NVIDIA ModelOpt NVFP4 checkpoint targeted at T4 (sm_75 lacks NVFP4 tensor cores)",
        "target": "nvidia/Qwen3-8B-FP4",
        "engine": "vllm", "gpu": "T4", "tp": 1,
        "must_catch": ["QPF004"],
    },
    {
        "id": "defect_04_fp8_w8a8_on_turing_t4",
        "description": "NVIDIA ModelOpt FP8 checkpoint targeted at T4 (sm_75 lacks FP8 tensor cores)",
        "target": "nvidia/Llama-3.1-8B-Instruct-FP8",
        "engine": "vllm", "gpu": "T4", "tp": 1,
        "must_catch": ["QPF004"],
    },
    {
        "id": "defect_05_fp8_block_on_sm89_l40s",
        "description": "Qwen3-8B-FP8 (weight_block_size=[128,128]) on L40S (vllm#58241: cutlass_scaled_mm rejects block FP8 on SM89)",
        "target": "Qwen/Qwen3-8B-FP8",
        "engine": "vllm", "gpu": "L40S", "tp": 1,
        "must_catch": ["QPF004"],
    },
    {
        "id": "defect_06_bf16_on_turing_t4",
        "description": "TinyLlama BF16 checkpoint on T4, which has no BF16 and runs it as FP16 (a warning, verified on real T4)",
        "target": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "engine": "vllm", "gpu": "T4", "tp": 1,
        "must_catch": ["QPF004"],
    },
    {
        "id": "defect_07_bf16_granite_on_turing_t4",
        "description": "IBM Granite-3.1-8B BF16 checkpoint on T4, which runs it as FP16 (a warning)",
        "target": "ibm-granite/granite-3.1-8b-instruct",
        "engine": "vllm", "gpu": "T4", "tp": 1,
        "must_catch": ["QPF004"],
    },
    {
        "id": "defect_08_marlin_fp8_on_gb10_sm121",
        "description": "Qwen2.5 AWQ checkpoint on GB10 with VLLM_MARLIN_INPUT_DTYPE=fp8 (vllm#49546 silent corruption)",
        "target": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "engine": "vllm", "gpu": "GB10", "tp": 1,
        "marlin_input_dtype": "fp8",
        "must_catch": ["QPF005"],
    },
    {
        "id": "defect_09_marlin_int8_ignored_autoround",
        "description": "AutoRound W8A8 checkpoint on A100 with VLLM_MARLIN_INPUT_DTYPE=int8 (vllm#48904 ignored flag)",
        "target": "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
        "engine": "vllm", "gpu": "A100", "tp": 1,
        "marlin_input_dtype": "int8",
        "must_catch": ["QPF005"],
    },
    {
        "id": "defect_10_fp8_dynamic_on_turing_t4",
        "description": "Llama-3.1 FP8 Dynamic on T4 (sm_75 lacks FP8 tensor cores)",
        "target": "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",
        "engine": "vllm", "gpu": "T4", "tp": 1,
        "must_catch": ["QPF004"],
    },

    # --- Group 2: Tokenizer Vocabulary Out-of-Bounds (QPF002) & Stop-Token Drift (QPF003) ---
    {
        "id": "defect_12_mistral_q2k_vocab_oob",
        "description": "Mistral-7B Q2_K GGUF with the token embedding quantized below 4-bit",
        "target": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q2_K.gguf",
        "engine": "vllm", "gpu": "L4", "tp": 1,
        "must_catch": ["QPF001"],
    },
    {
        "id": "defect_13_llama31_iq2m_vocab_oob",
        "description": "Llama-3.1 IQ2_M 2-bit GGUF checkpoint served on vLLM, which handles 2-bit I-quants poorly",
        "target": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf",
        "engine": "vllm", "gpu": "A100", "tp": 1,
        "must_catch": ["QPF001", "QPF011"],
    },
    {
        "id": "defect_14_olmo_chat_eos_drift",
        "description": "OLMo checkpoint with missing generation eos_token_id causing potential runaway token generation",
        "target": "allenai/OLMo-1B-hf",
        "engine": "vllm", "gpu": "A100", "tp": 1,
        "must_catch": ["QPF003"],
    },
    {
        "id": "defect_15_added_tokens_oob_cve_2026",
        "description": "Adversarial CVE-2026-115916 pattern: tokenizer added_token_id 32005 exceeds 32000-row embedding matrix",
        "type": "synthetic_oob_tokens",
        "must_catch": ["QPF002"],
    },
    {
        "id": "defect_16_generation_eos_disjoint_config",
        "description": "generation_config.json specifies eos_token_id=2 while config.json specifies eos_token_id=151645 (drift)",
        "type": "synthetic_eos_drift",
        "must_catch": ["QPF003"],
    },
    {
        "id": "defect_17_chat_template_stop_missing",
        "description": "Tokenizer chat_template expects <|im_end|> (151645) but generation_config only has ID 2",
        "type": "synthetic_chat_eos_missing",
        "must_catch": ["QPF003"],
    },
    {
        "id": "defect_18_oob_stop_token_id_vllm_crash",
        "description": "vllm#54195: Stop token ID 50000 exceeds model vocabulary size 32000, crashing server on boot",
        "type": "synthetic_stop_token_oob",
        "must_catch": ["QPF002"],
    },

    # --- Group 3: Sensitive Layer Quantization & Sub-4-Bit Collapse (QPF001) ---
    {
        "id": "defect_19_mistral_q2k_sub4bit_embed",
        "description": "Mistral-7B Q2_K GGUF sub-4-bit extreme token embedding quantization degrades output generation",
        "target": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q2_K.gguf",
        "engine": "vllm", "gpu": "L4", "tp": 1,
        "must_catch": ["QPF001"],
    },
    {
        "id": "defect_20_llama31_iq2m_sub3bit_embed",
        "description": "Llama-3.1 IQ2_M 2-bit GGUF checkpoint quantizes embedding down to 2-bit",
        "target": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf",
        "engine": "vllm", "gpu": "A100", "tp": 1,
        "must_catch": ["QPF001"],
    },
    {
        "id": "defect_21_deepseek_router_gate_quantized",
        "description": "DeepSeek MoE mlp.gate.weight router quantized to 4-bit, which can distort routing (a quality warning)",
        "type": "synthetic_moe_router_quantized",
        "must_catch": ["QPF001"],
    },
    {
        "id": "defect_22_mamba_ssm_decay_quantized",
        "description": "Mamba hybrid recurrent decay parameters (A_log, dt_bias) quantized to INT8, which risks state divergence",
        "type": "synthetic_mamba_ssm_quantized",
        "must_catch": ["QPF001"],
    },
    {
        "id": "defect_23_layernorm_rmsnorm_quantized",
        "description": "Model with input_layernorm and post_attention_layernorm quantized below 16-bit",
        "type": "synthetic_norm_quantized",
        "must_catch": ["QPF001"],
    },
    {
        "id": "defect_24_vision_patch_quantized_extreme",
        "description": "Multimodal VLM vision tower linear projection quantized below 8-bit precision",
        "type": "synthetic_vision_quantized",
        "must_catch": ["QPF001"],
    },

    # --- Group 4: Tensor Parallelism Head & Expert Divisibility (QPF006) ---
    {
        "id": "defect_25_qwen14b_tp3_gqa_indivisible",
        "description": "Qwen2.5-14B (8 KV heads) deployed on TP=3, which cannot be partitioned evenly across ranks",
        "target": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "engine": "vllm", "gpu": "A100", "tp": 3,
        "must_catch": ["QPF006"],
    },
    {
        "id": "defect_26_qwen14b_tp5_gqa_indivisible",
        "description": "Qwen2.5-14B (8 KV heads) deployed on TP=5, violating TP divisibility",
        "target": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "engine": "vllm", "gpu": "A100", "tp": 5,
        "must_catch": ["QPF006"],
    },
    {
        "id": "defect_27_qwen14b_tp6_gqa_indivisible",
        "description": "Qwen2.5-14B (8 KV heads) deployed on TP=6, violating TP divisibility",
        "target": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "engine": "vllm", "gpu": "A100", "tp": 6,
        "must_catch": ["QPF006"],
    },
    {
        "id": "defect_28_qwen14b_tp7_gqa_indivisible",
        "description": "Qwen2.5-14B (8 KV heads) deployed on TP=7, violating TP divisibility",
        "target": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "engine": "vllm", "gpu": "A100", "tp": 7,
        "must_catch": ["QPF006"],
    },
    {
        "id": "defect_29_qwen57b_tp7_moe_indivisible",
        "description": "Qwen2-57B-A14B MoE (64 routed experts) deployed on TP=7, violating expert sharding divisibility",
        "target": "Qwen/Qwen2-57B-A14B-Instruct",
        "engine": "vllm", "gpu": "H100", "tp": 7,
        "must_catch": ["QPF006"],
    },
    {
        "id": "defect_30_qwen57b_tp9_moe_indivisible",
        "description": "Qwen2-57B-A14B MoE (64 routed experts) deployed on TP=9, violating expert sharding divisibility",
        "target": "Qwen/Qwen2-57B-A14B-Instruct",
        "engine": "vllm", "gpu": "H100", "tp": 9,
        "must_catch": ["QPF006"],
    },
    {
        "id": "defect_31_tinyllama_tp8_gqa_indivisible",
        "description": "TinyLlama (4 KV heads) deployed on TP=8, where ranks exceed available KV head groups",
        "target": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "engine": "vllm", "gpu": "A100", "tp": 8,
        "must_catch": ["QPF006"],
    },
    {
        "id": "defect_32_gemma2_2b_tp3_indivisible",
        "description": "Gemma-2-2B (4 KV heads, 8 attn heads) deployed on TP=3",
        "target": "unsloth/gemma-2-2b-it-bnb-4bit",
        "engine": "vllm", "gpu": "A100", "tp": 3,
        "must_catch": ["QPF006"],
    },

    # --- Group 5: Quantization Alignment, Marlin Tiles & GPTQ desc_act (QPF007 & QPF008) ---
    {
        "id": "defect_33_wizardlm_desc_act_tp2",
        "description": "WizardLM-7B 8-bit GPTQ with desc_act=true on vLLM, which rejects it (verified on real T4)",
        "target": "Squish42/WizardLM-7B-Uncensored-GPTQ-act_order-8bit",
        "engine": "vllm", "gpu": "A100", "tp": 2,
        "must_catch": ["QPF008"],
    },
    {
        "id": "defect_34_wizardlm_desc_act_tp4",
        "description": "WizardLM-7B 8-bit GPTQ with desc_act=true on vLLM across four GPUs",
        "target": "Squish42/WizardLM-7B-Uncensored-GPTQ-act_order-8bit",
        "engine": "vllm", "gpu": "A100", "tp": 4,
        "must_catch": ["QPF008"],
    },
    {
        "id": "defect_35_marlin_tile_unaligned_dim",
        "description": "AWQ checkpoint post-TP out_features dimension (e.g. 1000) not divisible by Marlin tile (128x64)",
        "type": "synthetic_marlin_tile_unaligned",
        "must_catch": ["QPF007"],
    },
    {
        "id": "defect_36_group_size_indivisible_after_tp",
        "description": "Sharded linear dimension not divisible by group_size=128",
        "type": "synthetic_group_size_unaligned",
        "must_catch": ["QPF007"],
    },
    {
        "id": "defect_37_block_fp8_tile_indivisible",
        "description": "FP8 block scale shape not divisible by weight_block_size=[128, 128]",
        "type": "synthetic_block_fp8_unaligned",
        "must_catch": ["QPF007"],
    },

    # --- Group 6: Engine Container & Sub-3-Bit Format Compatibility (QPF011) ---
    {
        "id": "defect_38_safetensors_on_llamacpp",
        "description": "Safetensors checkpoint passed directly to llama.cpp without GGUF conversion",
        "target": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "engine": "llamacpp", "gpu": "L4", "tp": 1,
        "must_catch": ["QPF011"],
    },
    {
        "id": "defect_39_gguf_on_tensorrt_llm",
        "description": "GGUF checkpoint passed to TensorRT-LLM container, which requires Safetensors engine build",
        "target": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "engine": "trtllm", "gpu": "H100", "tp": 1,
        "must_catch": ["QPF011"],
    },
    {
        "id": "defect_40_iq2xxs_on_vllm",
        "description": "Sub-3-bit GGUF (IQ2_M) served on vLLM, falling back to slow dequantization",
        "target": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf",
        "engine": "vllm", "gpu": "A100", "tp": 1,
        "must_catch": ["QPF011"],
    },
    {
        "id": "defect_41_iq1s_on_sglang",
        "description": "1-bit GGUF (IQ1_S) served on SGLang",
        "type": "synthetic_iq1s_on_sglang",
        "must_catch": ["QPF011"],
    },

    # --- Group 7: VRAM Over-Allocation & OOM Budget Crashes (QPF012) ---
    {
        "id": "defect_42_qwen57b_moe_on_rtx4090",
        "description": "Qwen2-57B in BF16 on one 24 GiB RTX 4090",
        "target": "Qwen/Qwen2-57B-A14B-Instruct",
        "engine": "vllm", "gpu": "RTX-4090", "tp": 1,
        "must_catch": ["QPF012"],
    },
    {
        "id": "defect_43_qwen57b_moe_on_single_l4",
        "description": "Qwen2-57B deployed on a single 24 GiB L4 GPU (TP=1)",
        "target": "Qwen/Qwen2-57B-A14B-Instruct",
        "engine": "vllm", "gpu": "L4", "tp": 1,
        "must_catch": ["QPF012"],
    },
    {
        "id": "defect_44_qwen14b_128k_ctx_on_t4",
        "description": "Qwen2.5-14B with --context-len 131072 on one T4",
        "target": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "engine": "vllm", "gpu": "T4", "tp": 1,
        "context_len": 131072,
        "must_catch": ["QPF012"],
    },
    {
        "id": "defect_45_qwen72b_on_single_a100_40gb",
        "description": "72B parameter model on a single 40 GiB A100 GPU (TP=1)",
        "type": "synthetic_vram_overflow_72b",
        "must_catch": ["QPF012"],
    },

    # --- Group 8: Shard Integrity, Binary Offset Corruptions & Supply-Chain Hazards (QPF009, QPF010, QPF013, QPF014) ---
    {
        "id": "defect_46_safetensors_overlapping_spans",
        "description": "Safetensors shard with overlapping tensor data offsets causing memory span corruption (QPF013)",
        "type": "synthetic_overlapping_spans",
        "must_catch": ["QPF013"],
    },
    {
        "id": "defect_47_safetensors_truncated_file_eof",
        "description": "Safetensors shard truncated prematurely where tensor end offset exceeds physical file size (QPF013)",
        "type": "synthetic_truncated_file_eof",
        "must_catch": ["QPF013"],
    },
    {
        "id": "defect_48_legacy_pickle_ace_hazard",
        "description": "Checkpoint relying solely on legacy PyTorch pickle serialization (arbitrary code execution supply-chain hazard)",
        "target": "prajjwal1/bert-tiny",
        "engine": "vllm", "gpu": "L4", "tp": 1,
        "must_catch": ["QPF014"],
    },
    {
        "id": "defect_49_redundant_pickle_in_moe_repo",
        "description": "Official Mixtral-8x7B repository contains lingering legacy pickle weights alongside Safetensors",
        "target": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "engine": "vllm", "gpu": "H100", "tp": 2,
        "must_catch": ["QPF014"],
    },
    {
        "id": "defect_50_unsandboxed_trust_remote_code",
        "description": "Phi-3 checkpoint requiring trust_remote_code=True for unreviewed custom Python modeling scripts",
        "target": "microsoft/Phi-3-mini-4k-instruct",
        "engine": "vllm", "gpu": "A100", "tp": 1,
        "must_catch": ["QPF014"],
    },
]


def _build_synthetic_checkpoint(stype: str) -> CheckpointMeta:
    from quantpreflight.loader import _extract_tokenizer_meta

    if stype == "synthetic_oob_tokens":
        cfg = {"architectures": ["LlamaForCausalLM"], "vocab_size": 32000}
        tok_cfg = {"added_tokens_decoder": {"32005": {"content": "<|extra|>"}}}
        tmeta = _extract_tokenizer_meta(None, tok_cfg, config_vocab_size=32000)
        return CheckpointMeta(
            source="synthetic_oob_tokens",
            container_format="safetensors",
            config=cfg,
            tokenizer_config=tok_cfg,
            tokenizer_meta=tmeta,
            tensors={
                "model.embed_tokens.weight": TensorSpec(
                    name="model.embed_tokens.weight", storage_dtype="BF16",
                    shape=(32000, 4096), shard="model.safetensors",
                    byte_size=32000 * 4096 * 2, effective_bits=16.0,
                )
            },
        )
    elif stype == "synthetic_eos_drift":
        cfg = {"architectures": ["LlamaForCausalLM"], "vocab_size": 32000, "eos_token_id": 151645}
        return CheckpointMeta(
            source="synthetic_eos_drift", container_format="safetensors",
            config=cfg, generation_config={"eos_token_id": 2},
        )
    elif stype == "synthetic_chat_eos_missing":
        cfg = {"architectures": ["LlamaForCausalLM"], "vocab_size": 32000}
        tok_cfg = {
            "chat_template": "Hello <|im_end|>",
            "eos_token": "<|im_end|>",
            "added_tokens_decoder": {"151645": {"content": "<|im_end|>"}},
        }
        tmeta = _extract_tokenizer_meta(None, tok_cfg, config_vocab_size=32000)
        return CheckpointMeta(
            source="synthetic_chat_eos_missing", container_format="safetensors",
            config=cfg, generation_config={"eos_token_id": 2},
            tokenizer_config=tok_cfg,
            tokenizer_meta=tmeta,
        )
    elif stype == "synthetic_stop_token_oob":
        cfg = {"architectures": ["LlamaForCausalLM"], "vocab_size": 32000, "eos_token_id": 50000}
        return CheckpointMeta(
            source="synthetic_stop_token_oob", container_format="safetensors",
            config=cfg,
            tensors={
                "model.embed_tokens.weight": TensorSpec(
                    name="model.embed_tokens.weight", storage_dtype="BF16",
                    shape=(32000, 4096), shard="model.safetensors",
                    byte_size=32000 * 4096 * 2, effective_bits=16.0,
                )
            },
        )
    elif stype == "synthetic_moe_router_quantized":
        cfg = {
            "architectures": ["DeepseekV2ForCausalLM"], "model_type": "deepseek_v2",
            "quantization_config": {"quant_method": "awq", "bits": 4},
        }
        return CheckpointMeta(
            source="synthetic_moe_router_quantized", container_format="safetensors",
            config=cfg, quant=parse_quant_spec_from_config(cfg),
            tensors={
                "model.layers.0.mlp.gate.weight": TensorSpec(
                    name="model.layers.0.mlp.gate.weight", storage_dtype="I32",
                    shape=(64, 512), shard="model.safetensors",
                    byte_size=64 * 512 * 4, effective_bits=4.0,
                )
            },
        )
    elif stype == "synthetic_mamba_ssm_quantized":
        cfg = {
            "architectures": ["MambaForCausalLM"], "model_type": "mamba",
            "quantization_config": {"quant_method": "gptq", "bits": 8},
        }
        return CheckpointMeta(
            source="synthetic_mamba_ssm_quantized", container_format="safetensors",
            config=cfg, quant=parse_quant_spec_from_config(cfg),
            tensors={
                "backbone.layers.0.mixer.A_log": TensorSpec(
                    name="backbone.layers.0.mixer.A_log", storage_dtype="I8",
                    shape=(16, 64), shard="model.safetensors",
                    byte_size=1024, effective_bits=8.0,
                )
            },
        )
    elif stype == "synthetic_norm_quantized":
        cfg = {
            "architectures": ["LlamaForCausalLM"],
            "quantization_config": {"quant_method": "awq", "bits": 4},
        }
        return CheckpointMeta(
            source="synthetic_norm_quantized", container_format="safetensors",
            config=cfg, quant=parse_quant_spec_from_config(cfg),
            tensors={
                "model.layers.0.input_layernorm.weight": TensorSpec(
                    name="model.layers.0.input_layernorm.weight", storage_dtype="I32",
                    shape=(4096, 512), shard="model.safetensors",
                    byte_size=4096 * 512 * 4, effective_bits=4.0,
                )
            },
        )
    elif stype == "synthetic_vision_quantized":
        cfg = {
            "architectures": ["Qwen2VLForConditionalGeneration"], "model_type": "qwen2_vl",
            "quantization_config": {"quant_method": "awq", "bits": 4},
        }
        return CheckpointMeta(
            source="synthetic_vision_quantized", container_format="safetensors",
            config=cfg, quant=parse_quant_spec_from_config(cfg),
            tensors={
                "visual.patch_embed.proj.weight": TensorSpec(
                    name="visual.patch_embed.proj.weight", storage_dtype="I32",
                    shape=(1152, 128), shard="model.safetensors",
                    byte_size=1152 * 128 * 4, effective_bits=4.0,
                )
            },
        )
    elif stype == "synthetic_marlin_tile_unaligned":
        cfg = {
            "architectures": ["LlamaForCausalLM"],
            "quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128},
        }
        return CheckpointMeta(
            source="synthetic_marlin_tile_unaligned", container_format="safetensors",
            config=cfg, quant=parse_quant_spec_from_config(cfg),
            tensors={
                "model.layers.0.self_attn.o_proj.qweight": TensorSpec(
                    name="model.layers.0.self_attn.o_proj.qweight", storage_dtype="I32",
                    shape=(4096, 125), shard="model.safetensors",  # 125 * 8 = 1000 (not divisible by 64)
                    byte_size=4096 * 125 * 4, effective_bits=4.0,
                    logical_shape=(1000, 4096),
                )
            },
        )
    elif stype == "synthetic_group_size_unaligned":
        cfg = {
            "architectures": ["LlamaForCausalLM"],
            "quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128},
        }
        return CheckpointMeta(
            source="synthetic_group_size_unaligned", container_format="safetensors",
            config=cfg, quant=parse_quant_spec_from_config(cfg),
            tensors={
                "model.layers.0.mlp.down_proj.qweight": TensorSpec(
                    name="model.layers.0.mlp.down_proj.qweight", storage_dtype="I32",
                    shape=(100, 4096), shard="model.safetensors",
                    byte_size=100 * 4096 * 4, effective_bits=4.0,
                    logical_shape=(4096, 800),
                )
            },
        )
    elif stype == "synthetic_block_fp8_unaligned":
        cfg = {
            "architectures": ["Qwen2ForCausalLM"],
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            },
        }
        return CheckpointMeta(
            source="synthetic_block_fp8_unaligned", container_format="safetensors",
            config=cfg, quant=parse_quant_spec_from_config(cfg),
            tensors={
                "model.layers.0.mlp.gate_proj.weight": TensorSpec(
                    name="model.layers.0.mlp.gate_proj.weight", storage_dtype="F8_E4M3",
                    shape=(4000, 1000), shard="model.safetensors",  # Not divisible by 128
                    byte_size=4000 * 1000, effective_bits=8.0,
                )
            },
        )
    elif stype == "synthetic_iq1s_on_sglang":
        return CheckpointMeta(
            source="synthetic_iq1s", container_format="gguf",
            tensors={
                "blk.0.attn_q.weight": TensorSpec(
                    name="blk.0.attn_q.weight", storage_dtype="IQ1_S",
                    shape=(4096, 4096), shard="model.gguf",
                    byte_size=4096 * 512, effective_bits=1.5,
                )
            },
        )
    elif stype == "synthetic_vram_overflow_72b":
        cfg = {
            "architectures": ["Qwen2ForCausalLM"], "vocab_size": 152064,
            "hidden_size": 8192, "num_hidden_layers": 80,
            "num_attention_heads": 64, "num_key_value_heads": 8,
        }
        return CheckpointMeta(
            source="synthetic_vram_72b", container_format="safetensors",
            config=cfg, total_weight_bytes=72 * (1024**3),
        )
    elif stype == "synthetic_overlapping_spans":
        return CheckpointMeta(
            source="synthetic_overlapping_spans", container_format="safetensors",
            config={"architectures": ["LlamaForCausalLM"]},
            shard_files=("model.safetensors",),
            tensors={
                "t1": TensorSpec(name="t1", storage_dtype="F16", shape=(10, 10), shard="model.safetensors", byte_size=200, effective_bits=16.0, data_offsets=(0, 200)),
                "t2": TensorSpec(name="t2", storage_dtype="F16", shape=(10, 10), shard="model.safetensors", byte_size=200, effective_bits=16.0, data_offsets=(100, 300)),
            },
        )
    elif stype == "synthetic_truncated_file_eof":
        return CheckpointMeta(
            source="synthetic_truncated_file_eof", container_format="safetensors",
            config={"architectures": ["LlamaForCausalLM"]},
            shard_files=("model.safetensors",),
            shard_file_sizes={"model.safetensors": 500},
            shard_header_lengths={"model.safetensors": 100},
            tensors={
                "t1": TensorSpec(name="t1", storage_dtype="F16", shape=(20, 20), shard="model.safetensors", byte_size=800, effective_bits=16.0, data_offsets=(0, 800)),
            },
        )
    raise ValueError(f"Unknown synthetic type {stype}")


def run_scenarios() -> int:
    print("=" * 80)
    print("RUNNING 50-DEFECT VALIDATION MATRIX (50 Real & Adversarial Failure Scenarios)")
    print("=" * 80)

    passed = 0
    failed = 0
    results = []
    _cached_metas: dict[str, CheckpointMeta] = {}

    total = len(SCENARIOS)
    for idx, case in enumerate(SCENARIOS, 1):
        cid = case["id"]
        desc = case["description"]
        must_catch = case["must_catch"]
        eng = case.get("engine", "vllm")
        gpu = case.get("gpu", "A100")
        tp = case.get("tp", 1)
        ctx = case.get("context_len")
        marlin_dtype = case.get("marlin_input_dtype")

        t0 = time.perf_counter()
        try:
            if "type" in case:
                meta = _build_synthetic_checkpoint(case["type"])
            else:
                tgt = case["target"]
                if tgt not in _cached_metas:
                    _cached_metas[tgt] = load_checkpoint_metadata(tgt, timeout=25.0)
                meta = _cached_metas[tgt]

            rep = analyze_checkpoint(
                meta,
                TargetSpec(
                    engine=eng,
                    gpu=gpu,
                    tp=tp,
                    context_len=ctx,
                    marlin_input_dtype=marlin_dtype,
                ),
            )
            dt = time.perf_counter() - t0

            caught_rules = {f.rule_id for f in rep.findings}
            missing = [r for r in must_catch if r not in caught_rules]

            if not missing:
                passed += 1
                findings_summary = [f"{f.rule_id} [{f.severity.value}]: {f.title}" for f in rep.findings if f.rule_id in must_catch]
                results.append({
                    "index": idx,
                    "id": cid,
                    "status": "PASS",
                    "sec": round(dt, 2),
                    "description": desc,
                    "caught": findings_summary,
                })
                print(f"[{idx:02d}/{total} PASS {dt:4.1f}s] {cid:<40} caught: {', '.join(must_catch)}")
                for f in rep.findings:
                    if f.rule_id in must_catch:
                        print(f"       -> {f.rule_id} [{f.severity.value}]: {f.title}")
            else:
                failed += 1
                results.append({
                    "index": idx,
                    "id": cid,
                    "status": "FAIL",
                    "sec": round(dt, 2),
                    "description": desc,
                    "expected": must_catch,
                    "missing": missing,
                    "caught": list(caught_rules),
                })
                print(f"[{idx:02d}/{total} FAIL {dt:4.1f}s] {cid:<40} missing: {missing}")

        except Exception as exc:
            dt = time.perf_counter() - t0
            failed += 1
            results.append({
                "index": idx,
                "id": cid,
                "status": "ERROR",
                "sec": round(dt, 2),
                "error": str(exc),
            })
            print(f"[{idx:02d}/{total} ERR  {dt:4.1f}s] {cid:<40} error: {exc}")

    out_file = ROOT / "benchmark" / "scenario_results.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"SCENARIOS: {passed}/{total} caught (failed: {failed}).")
    print(f"Results saved to {out_file}")
    print("=" * 80)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_scenarios())
