"""Validate all bundled TOML architecture, GPU, and engine profiles."""

from __future__ import annotations

import pytest

from quantpreflight.profiles import (
    ProfileValidationError,
    load_architecture_profiles,
    load_engine_profiles,
    load_gpu_profiles,
    resolve_architecture_profile,
    resolve_gpu_profile,
    validate_all_profiles,
)


def test_all_bundled_profiles_validate_cleanly() -> None:
    n_arch, n_gpu, n_eng = validate_all_profiles()
    assert n_arch >= 8
    assert n_gpu >= 12
    assert n_eng == 4


def test_moe_router_regex_never_matches_swiglu_gate_proj() -> None:
    profiles = load_architecture_profiles()
    swiglu_names = [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.gate_proj.qweight",
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.0.mlp.experts.3.gate_proj.weight",
    ]
    router_names = [
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.gate.qweight",
    ]
    for prof in profiles.values():
        for rule in prof.sensitive_layers:
            if "router" in rule.name or "moe_gate" in rule.name:
                for sname in swiglu_names:
                    matched = bool(rule.compiled_regex.search(sname)) and not (
                        rule.compiled_exclude and rule.compiled_exclude.search(sname)
                    )
                    assert not matched, f"{prof.id}:{rule.name} falsely matched SwiGLU {sname}"

    qwen_prof = resolve_architecture_profile({"architectures": ["Qwen3MoeForCausalLM"]}, profiles)
    for rname in router_names:
        assert any(
            r.compiled_regex.search(rname) and not (r.compiled_exclude and r.compiled_exclude.search(rname))
            for r in qwen_prof.sensitive_layers
        )


def test_gpu_alias_resolution_and_unknown_error() -> None:
    gpus = load_gpu_profiles()
    assert resolve_gpu_profile("sm_89", gpus).id in ("L4", "L40S")
    assert resolve_gpu_profile("rtx4090", gpus).id == "L40S"
    assert resolve_gpu_profile("dgx-spark", gpus).id == "GB10"
    assert resolve_gpu_profile("gfx942", gpus).id == "MI300X"

    with pytest.raises(ProfileValidationError, match="Unknown GPU target"):
        resolve_gpu_profile("Voodoo3DFX", gpus)


def test_engines_present() -> None:
    engines = load_engine_profiles()
    assert set(engines.keys()) == {"vllm", "sglang", "trtllm", "llamacpp"}
