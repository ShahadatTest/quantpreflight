"""Every bundled profile, and every way a contributed profile can be wrong."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from quantpreflight import profiles
from quantpreflight.profiles import (
    ProfileValidationError,
    load_architecture_profiles,
    load_engine_profiles,
    load_gpu_profiles,
    resolve_architecture_profile,
    resolve_gpu_profile,
)

PACKAGE = Path(profiles.__file__).parent

GOOD_RULE = """
[[sensitive_layers]]
name = "n"
pattern = "x"
min_bits = 8
severity = "high"
reason = "r"
"""

GOOD_ARCH = f'id = "a"\nname = "A"\n{GOOD_RULE}'
DEFAULT_ARCH = f'id = "default"\nname = "D"\n{GOOD_RULE}'


def _arch_dir(tmp_path: Path, **files: str) -> Path:
    for name, text in files.items():
        (tmp_path / f"{name}.toml").write_text(text, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('name = "A"', "'id'"),
        ('id = ""\nname = "A"', "'id'"),
        ('id = 5\nname = "A"', "'id'"),
        ('id = "a"', "'name'"),
        ('id = "a"\nname = ""', "'name'"),
        ('id = "a"\nname = "A"\nmatch_architectures = "x"', "match_architectures"),
        ('id = "a"\nname = "A"\nmatch_architectures = [1]', "match_architectures"),
        ('id = "a"\nname = "A"\nmatch_model_types = "x"', "match_model_types"),
        ('id = "a"\nname = "A"\nmatch_model_types = [1]', "match_model_types"),
        ('id = "a"\nname = "A"\ncol_parallel_suffixes = "x"', "col_parallel_suffixes"),
        ('id = "a"\nname = "A"\ncol_parallel_suffixes = [1]', "col_parallel_suffixes"),
        ('id = "a"\nname = "A"\nrow_parallel_suffixes = "x"', "row_parallel_suffixes"),
        ('id = "a"\nname = "A"\nrow_parallel_suffixes = [1]', "row_parallel_suffixes"),
        ('id = "a"\nname = "A"', "sensitive_layers"),
        ('id = "a"\nname = "A"\nsensitive_layers = 5', "sensitive_layers"),
        ('id = "a"\nname = "A"\nsensitive_layers = [1]', "must be a table"),
        ("this is not toml", "Failed to parse"),
    ],
)
def test_bad_architecture_files(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ProfileValidationError, match=message):
        load_architecture_profiles(_arch_dir(tmp_path, bad=text))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('name = "n"', 'name = ""', "missing 'name'"),
        ('pattern = "x"', 'pattern = ""', "missing 'pattern'"),
        ('pattern = "x"', 'pattern = "("', "invalid regex"),
        ('pattern = "x"', 'pattern = "x"\nexclude_pattern = 5', "exclude_pattern"),
        ('pattern = "x"', 'pattern = "x"\nexclude_pattern = "("', "invalid regex"),
        ("min_bits = 8", 'min_bits = "8"', "min_bits"),
        ("min_bits = 8", "min_bits = 0", "min_bits"),
        ("min_bits = 8", "min_bits = -1", "min_bits"),
        ("min_bits = 8", "min_bits = 65", "min_bits"),
        ('severity = "high"', 'severity = "fatal"', "severity"),
        ('reason = "r"', 'reason = ""', "missing 'reason'"),
    ],
)
def test_bad_sensitive_rules(tmp_path: Path, old: str, new: str, message: str) -> None:
    text = GOOD_ARCH.replace(old, new)
    with pytest.raises(ProfileValidationError, match=message):
        load_architecture_profiles(_arch_dir(tmp_path, bad=text, default=DEFAULT_ARCH))


def test_good_architecture_file_is_normalized(tmp_path: Path) -> None:
    text = GOOD_ARCH.replace('name = "A"', 'name = "A"\nmatch_architectures = ["FooForCausalLM"]\nmatch_model_types = ["FOO"]\ncol_parallel_suffixes = ["q"]\nrow_parallel_suffixes = ["o"]', 1).replace('severity = "high"', 'severity = "HIGH"').replace("min_bits = 8", "min_bits = 64")
    loaded = load_architecture_profiles(_arch_dir(tmp_path, a=text, default=DEFAULT_ARCH))["a"]
    assert loaded.match_architectures == ("fooforcausallm",)
    assert loaded.match_model_types == ("foo",)
    assert (loaded.col_parallel_suffixes, loaded.row_parallel_suffixes) == (("q",), ("o",))
    rule = loaded.sensitive_layers[0]
    assert (rule.min_bits, rule.severity, rule.exclude_pattern, rule.compiled_exclude) == (64.0, "high", None, None)
    assert rule.compiled_regex.search("axb")


def test_architecture_folder_rules(tmp_path: Path) -> None:
    with pytest.raises(ProfileValidationError, match="No architecture TOML"):
        load_architecture_profiles(tmp_path)
    with pytest.raises(ProfileValidationError, match="Missing required 'default'"):
        load_architecture_profiles(_arch_dir(tmp_path, a=GOOD_ARCH))
    with pytest.raises(ProfileValidationError, match="Duplicate architecture"):
        load_architecture_profiles(_arch_dir(tmp_path, a=GOOD_ARCH, b=GOOD_ARCH, default=DEFAULT_ARCH))


def test_custom_folders_do_not_replace_the_bundled_cache(tmp_path: Path) -> None:
    bundled = load_architecture_profiles()
    assert load_architecture_profiles(_arch_dir(tmp_path, default=DEFAULT_ARCH)).keys() == {"default"}
    assert load_architecture_profiles() is bundled


GOOD_GPU = """
[[gpus]]
id = "X1"
aliases = ["Other"]
vendor = "ACME"
arch = "arch_1"
compute_capability = 10
vram_gb = 8
supports_bf16 = true
supports_fp8_w8a8 = false
supports_fp8_block_scale = false
supports_nvfp4 = false
supports_marlin = false
"""


def _gpus(tmp_path: Path, text: str) -> dict[str, Any]:
    (tmp_path / "gpus.toml").write_text(text, encoding="utf-8")
    return load_gpu_profiles(tmp_path)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "non-empty"),
        ("gpus = 5", "non-empty"),
        ("gpus = []", "non-empty"),
        ("gpus = [1]", "must be a table"),
        (GOOD_GPU.replace('id = "X1"', 'id = ""'), "missing 'id'"),
        (GOOD_GPU.replace('aliases = ["Other"]', 'aliases = "x"'), "aliases"),
        (GOOD_GPU.replace('aliases = ["Other"]', "aliases = [1]"), "aliases"),
        (GOOD_GPU.replace("supports_bf16 = true", ""), "supports_bf16"),
        (GOOD_GPU.replace("supports_fp8_w8a8 = false", 'supports_fp8_w8a8 = "no"'), "supports_fp8_w8a8"),
        (GOOD_GPU.replace("supports_fp8_block_scale = false", ""), "supports_fp8_block_scale"),
        (GOOD_GPU.replace("supports_nvfp4 = false", ""), "supports_nvfp4"),
        (GOOD_GPU.replace("supports_marlin = false", ""), "supports_marlin"),
        (GOOD_GPU.replace("vram_gb = 8", ""), "vram_gb"),
        (GOOD_GPU.replace("vram_gb = 8", "vram_gb = 0"), "vram_gb"),
        (GOOD_GPU.replace("vram_gb = 8", 'vram_gb = "8"'), "vram_gb"),
        (GOOD_GPU.replace("compute_capability = 10", ""), "compute_capability"),
        (GOOD_GPU.replace("compute_capability = 10", "compute_capability = 1.5"), "compute_capability"),
    ],
)
def test_bad_gpu_files(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ProfileValidationError, match=message):
        _gpus(tmp_path, text)


def test_good_gpu_file_is_normalized(tmp_path: Path) -> None:
    gpu = _gpus(tmp_path, GOOD_GPU.replace('id = "X1"', 'id = "x1"'))["X1"]
    assert (gpu.id, gpu.aliases, gpu.vendor, gpu.arch, gpu.compute_capability, gpu.vram_gb) == ("x1", ("other",), "acme", "arch_1", 10, 8.0)
    assert gpu.supports_bf16 is True
    assert gpu.supports_marlin is False


def test_gpu_vendor_defaults_to_nvidia(tmp_path: Path) -> None:
    assert _gpus(tmp_path, GOOD_GPU.replace('vendor = "ACME"', ""))["X1"].vendor == "nvidia"


GOOD_ENGINE = """
[[engines]]
id = "E"
name = "Engine"
supported_containers = ["SafeTensors"]
supported_quant_methods = ["AWQ"]
fp8_block_restricted_arches = ["SM_89"]
"""


def _engines(tmp_path: Path, text: str) -> dict[str, Any]:
    (tmp_path / "engines.toml").write_text(text, encoding="utf-8")
    return load_engine_profiles(tmp_path)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "non-empty"),
        ("engines = []", "non-empty"),
        ("engines = [1]", "must be a table"),
        (GOOD_ENGINE.replace('id = "E"', 'id = ""'), "missing 'id'"),
        (GOOD_ENGINE.replace('supported_containers = ["SafeTensors"]', ""), "supported_containers"),
        (GOOD_ENGINE.replace('supported_containers = ["SafeTensors"]', "supported_containers = []"), "supported_containers"),
        (GOOD_ENGINE.replace('supported_quant_methods = ["AWQ"]', ""), "supported_quant_methods"),
    ],
)
def test_bad_engine_files(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ProfileValidationError, match=message):
        _engines(tmp_path, text)


def test_good_engine_file_is_normalized(tmp_path: Path) -> None:
    engine = _engines(tmp_path, GOOD_ENGINE)["e"]
    assert (engine.id, engine.name, engine.supported_containers, engine.supported_quant_methods) == ("e", "Engine", ("safetensors",), ("awq",))
    assert (engine.marlin_in_align, engine.marlin_out_align) == (128, 64)
    assert engine.fp8_block_restricted_arches == ("sm_89",)


def test_engine_name_defaults_to_its_id(tmp_path: Path) -> None:
    assert _engines(tmp_path, GOOD_ENGINE.replace('name = "Engine"', ""))["e"].name == "E"


def test_bad_toml_names_the_file(tmp_path: Path) -> None:
    with pytest.raises(ProfileValidationError, match="gpus.toml"):
        _gpus(tmp_path, "[[")


def test_validate_all_counts_every_bundled_file() -> None:
    assert profiles.validate_all_profiles() == (
        len(list((PACKAGE / "architectures").glob("*.toml"))),
        len(load_gpu_profiles()),
        4,
    )


# Bundled data

GPUS = sorted(load_gpu_profiles().values(), key=lambda gpu: gpu.id)
ARCHES = sorted(load_architecture_profiles().values(), key=lambda arch: arch.id)
UNIQUE_ARCH_GPUS = [gpu for gpu in GPUS if sum(other.arch == gpu.arch for other in GPUS) == 1]


@pytest.mark.parametrize("gpu", GPUS, ids=lambda gpu: gpu.id)
def test_gpu_resolves_by_id_in_any_case(gpu: Any) -> None:
    assert resolve_gpu_profile(gpu.id.lower()) is gpu
    assert resolve_gpu_profile(f"  {gpu.id.upper()} ") is gpu


@pytest.mark.parametrize(("gpu", "alias"), [(gpu, alias) for gpu in GPUS for alias in gpu.aliases], ids=str)
def test_gpu_resolves_by_alias(gpu: Any, alias: str) -> None:
    assert resolve_gpu_profile(alias) is gpu


@pytest.mark.parametrize("gpu", UNIQUE_ARCH_GPUS, ids=lambda gpu: gpu.id)
def test_gpu_resolves_by_arch(gpu: Any) -> None:
    assert resolve_gpu_profile(gpu.arch) is gpu


def test_gpu_aliases_never_collide() -> None:
    names = [gpu.id.lower() for gpu in GPUS] + [alias for gpu in GPUS for alias in gpu.aliases]
    assert len(names) == len(set(names))


def test_unknown_gpu_lists_the_known_ones() -> None:
    with pytest.raises(ProfileValidationError, match="Known GPUs: .*H100"):
        resolve_gpu_profile("H9000")


@pytest.mark.parametrize("gpu", GPUS, ids=lambda gpu: gpu.id)
def test_gpu_capabilities_follow_hardware_generations(gpu: Any) -> None:
    if gpu.supports_nvfp4:
        assert gpu.supports_fp8_w8a8 and gpu.supports_bf16
    if gpu.vendor == "amd":
        assert not gpu.supports_marlin
        assert not gpu.supports_nvfp4
    if gpu.vendor == "nvidia" and gpu.compute_capability < 80:
        assert not gpu.supports_bf16
    if gpu.vendor == "nvidia" and gpu.compute_capability < 89:
        assert not gpu.supports_fp8_w8a8


@pytest.mark.parametrize("arch", [arch for arch in ARCHES if arch.id != "default"], ids=lambda arch: arch.id)
def test_each_architecture_matches_by_every_declared_name(arch: Any) -> None:
    for name in arch.match_architectures:
        assert resolve_architecture_profile({"architectures": [name]}).id == arch.id
    for name in arch.match_model_types:
        assert resolve_architecture_profile({"model_type": name}).id == arch.id
        assert resolve_architecture_profile({"text_config": {"model_type": name.upper()}}).id == arch.id


@pytest.mark.parametrize("arch", ARCHES, ids=lambda arch: arch.id)
def test_each_architecture_covers_both_parallel_axes(arch: Any) -> None:
    assert arch.col_parallel_suffixes
    assert arch.row_parallel_suffixes
    assert not set(arch.col_parallel_suffixes) & set(arch.row_parallel_suffixes)


@pytest.mark.parametrize("arch", ARCHES, ids=lambda arch: arch.id)
def test_each_architecture_guards_norms(arch: Any) -> None:
    rules = [rule for rule in arch.sensitive_layers if rule.compiled_regex.search("model.layers.0.input_layernorm.weight")]
    assert rules
    assert max(rule.min_bits for rule in rules) >= 16.0


@pytest.mark.parametrize("arch", ARCHES, ids=lambda arch: arch.id)
def test_no_architecture_treats_plain_attention_as_sensitive(arch: Any) -> None:
    for name in ("model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.qweight"):
        for rule in arch.sensitive_layers:
            excluded = rule.compiled_exclude and rule.compiled_exclude.search(name)
            assert excluded or not rule.compiled_regex.search(name), f"{rule.name} matched {name}"


@pytest.mark.parametrize("config", [{}, {"architectures": None}, {"architectures": ["Unknown"]}, {"model_type": "unknown"}, {"text_config": "x"}])
def test_unknown_models_use_the_default_profile(config: dict[str, Any]) -> None:
    assert resolve_architecture_profile(config).id == "default"


@pytest.mark.parametrize(
    ("name", "rule"),
    [
        ("model.layers.0.mlp.gate.weight", "moe_router_gate"),
        ("model.layers.0.block_sparse_moe.gate.qweight", "moe_router_gate"),
        ("blk.0.ffn_gate_inp.weight", "moe_router_gate"),
        ("model.layers.0.mixer.A_log", "ssm_recurrent_state"),
        ("model.layers.0.mamba.conv1d.weight", "ssm_recurrent_state"),
        ("model.norm.weight", "normalization_weights"),
        ("visual.blocks.0.attn.proj.weight", "multimodal_vision_or_projector"),
        ("multi_modal_projector.linear_1.weight", "multimodal_vision_or_projector"),
        ("model.embed_tokens.weight", "token_embedding_or_lm_head_extreme"),
        ("lm_head.qweight", "token_embedding_or_lm_head_extreme"),
    ],
)
def test_default_sensitive_patterns(name: str, rule: str) -> None:
    default = load_architecture_profiles()["default"]
    matches = {r.name for r in default.sensitive_layers if r.compiled_regex.search(name)}
    assert rule in matches


@pytest.mark.parametrize("engine_id", ["vllm", "sglang", "trtllm", "llamacpp"])
def test_engine_tables_are_consistent(engine_id: str) -> None:
    engine = load_engine_profiles()[engine_id]
    assert engine.supported_containers
    assert "none" in engine.supported_quant_methods or engine_id == "llamacpp"
    if engine_id == "vllm":
        assert "bitsandbytes" not in engine.supported_quant_methods
    assert ("gguf" in engine.supported_containers) == ("gguf" in engine.supported_quant_methods)
    assert ("pytorch_bin" in engine.supported_containers) == (engine_id in ("vllm", "sglang"))


def test_gguf_only_engine_has_no_marlin_alignment() -> None:
    llamacpp = load_engine_profiles()["llamacpp"]
    assert (llamacpp.marlin_in_align, llamacpp.marlin_out_align) == (1, 1)


@pytest.mark.parametrize(
    ("config", "profile"),
    [
        ({"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": {"model_type": "qwen3_5_text"}}, "qwen_vl"),
        ({"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe"}, "qwen_vl"),
        ({"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp"}, "qwen_vl"),
        ({"architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text"}, "qwen_moe_hybrid"),
        ({"architectures": ["Phi3ForCausalLM"], "model_type": "phi3"}, "phi"),
        ({"architectures": ["PhiMoEForCausalLM"], "model_type": "phimoe"}, "phi"),
        ({"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4"}, "deepseek_moe"),
        ({"architectures": ["Glm4MoeForCausalLM"], "model_type": "glm4_moe"}, "glm"),
        ({"architectures": ["Gemma4UnifiedForConditionalGeneration"], "model_type": "gemma4_unified"}, "gemma"),
        ({"architectures": ["Mistral3ForConditionalGeneration"], "model_type": "mistral3"}, "mistral_mixtral"),
    ],
)
def test_newer_model_families(config: dict[str, Any], profile: str) -> None:
    assert resolve_architecture_profile(config).id == profile


@pytest.mark.parametrize(
    ("arch", "name", "matches"),
    [
        ("qwen_vl", "model.layers.0.linear_attn.conv1d.weight", True),
        ("qwen_vl", "model.visual.blocks.0.attn.qkv.weight", True),
        ("qwen_vl", "model.layers.0.linear_attn.in_proj_qkv.weight", False),
        ("phi", "model.layers.0.block_sparse_moe.gate.weight", True),
        ("phi", "model.layers.0.self_attn.qkv_proj.weight", False),
        ("phi", "model.final_layernorm.weight", True),
        ("glm", "model.layers.0.mlp.gate.weight", True),
        ("glm", "model.layers.0.mlp.gate_proj.weight", False),
        ("deepseek_moe", "layers.0.ffn.gate.weight", True),
        ("deepseek_moe", "layers.0.attn.wkv.weight", False),
        ("gemma", "vision_embedder.patch_dense.weight", True),
    ],
)
def test_sensitive_patterns_on_real_tensor_names(arch: str, name: str, matches: bool) -> None:
    rules = load_architecture_profiles()[arch].sensitive_layers
    hit = any(rule.compiled_regex.search(name) and not (rule.compiled_exclude and rule.compiled_exclude.search(name)) for rule in rules)
    assert hit is matches


@pytest.mark.parametrize(
    ("arch", "module", "axis"),
    [
        ("qwen_vl", "in_proj_qkv", "col"),
        ("qwen_vl", "out_proj", "row"),
        ("phi", "qkv_proj", "col"),
        ("phi", "dense", "row"),
        ("deepseek_moe", "wkv", "col"),
        ("deepseek_moe", "wo_b", "row"),
    ],
)
def test_parallel_axes_of_newer_families(arch: str, module: str, axis: str) -> None:
    profile = load_architecture_profiles()[arch]
    assert module in (profile.col_parallel_suffixes if axis == "col" else profile.row_parallel_suffixes)
