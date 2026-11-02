"""Load and validate declarative TOML architecture and hardware capability profiles."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found]


class ProfileValidationError(ValueError):
    """Raised when an architecture or hardware TOML profile violates schema rules."""


@dataclass(frozen=True)
class SensitiveLayerRule:
    """Declarative minimum-bit-width requirement for a sensitive tensor pattern."""

    name: str
    pattern: str
    exclude_pattern: str | None
    min_bits: float
    severity: str
    reason: str
    compiled_regex: re.Pattern[str]
    compiled_exclude: re.Pattern[str] | None


@dataclass(frozen=True)
class ArchitectureProfile:
    """Per-family sensitive layer, parallel sharding, and MTP projection rules."""

    id: str
    name: str
    match_architectures: tuple[str, ...]
    match_model_types: tuple[str, ...]
    sensitive_layers: tuple[SensitiveLayerRule, ...]
    col_parallel_suffixes: tuple[str, ...]
    row_parallel_suffixes: tuple[str, ...]


@dataclass(frozen=True)
class GPUProfile:
    """Hardware compute capability, VRAM, and native numeric dtype limits."""

    id: str
    aliases: tuple[str, ...]
    vendor: str
    arch: str
    compute_capability: int  # e.g. 75, 80, 86, 89, 90, 100, 120, 121; AMD gfx90a=900, gfx942=942
    vram_gb: float
    supports_bf16: bool
    supports_fp8_w8a8: bool
    supports_fp8_block_scale: bool
    supports_nvfp4: bool
    supports_marlin: bool


@dataclass(frozen=True)
class EngineProfile:
    """Serving engine container support, alignment rules, and known kernel hazards."""

    id: str
    name: str
    supported_containers: tuple[str, ...]
    supported_quant_methods: tuple[str, ...]
    marlin_in_align: int
    marlin_out_align: int
    fp8_block_restricted_arches: tuple[str, ...]


def _pkg_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception as exc:
        raise ProfileValidationError(f"Failed to parse TOML profile {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ProfileValidationError(f"TOML root in {path} must be a table.")
    return data


def _parse_arch_profile(path: Path) -> ArchitectureProfile:
    raw = _load_toml(path)
    pid = raw.get("id")
    pname = raw.get("name")
    if not isinstance(pid, str) or not pid:
        raise ProfileValidationError(f"{path.name}: missing non-empty string 'id'.")
    if not isinstance(pname, str) or not pname:
        raise ProfileValidationError(f"{path.name}: missing non-empty string 'name'.")

    match_arches = raw.get("match_architectures", [])
    match_types = raw.get("match_model_types", [])
    if not isinstance(match_arches, list) or not all(isinstance(x, str) for x in match_arches):
        raise ProfileValidationError(f"{path.name}: 'match_architectures' must be a list of strings.")
    if not isinstance(match_types, list) or not all(isinstance(x, str) for x in match_types):
        raise ProfileValidationError(f"{path.name}: 'match_model_types' must be a list of strings.")

    col_suf = raw.get("col_parallel_suffixes", [])
    row_suf = raw.get("row_parallel_suffixes", [])
    if not isinstance(col_suf, list) or not all(isinstance(x, str) for x in col_suf):
        raise ProfileValidationError(f"{path.name}: 'col_parallel_suffixes' must be a list of strings.")
    if not isinstance(row_suf, list) or not all(isinstance(x, str) for x in row_suf):
        raise ProfileValidationError(f"{path.name}: 'row_parallel_suffixes' must be a list of strings.")

    rules_raw = raw.get("sensitive_layers", [])
    if not isinstance(rules_raw, list) or not rules_raw:
        raise ProfileValidationError(f"{path.name}: 'sensitive_layers' must be a non-empty array of tables.")

    rules: list[SensitiveLayerRule] = []
    for idx, item in enumerate(rules_raw):
        if not isinstance(item, dict):
            raise ProfileValidationError(f"{path.name}: sensitive_layers[{idx}] must be a table.")
        rname = item.get("name")
        pat = item.get("pattern")
        exc_pat = item.get("exclude_pattern")
        min_bits = item.get("min_bits")
        sev = str(item.get("severity") or "").lower()
        reason = item.get("reason")
        if not isinstance(rname, str) or not rname:
            raise ProfileValidationError(f"{path.name}: sensitive_layers[{idx}] missing 'name'.")
        if not isinstance(pat, str) or not pat:
            raise ProfileValidationError(f"{path.name}: sensitive_layers[{idx}] missing 'pattern'.")
        if exc_pat is not None and not isinstance(exc_pat, str):
            raise ProfileValidationError(f"{path.name}: sensitive_layers[{idx}] 'exclude_pattern' must be string.")
        if not isinstance(min_bits, (int, float)) or min_bits <= 0 or min_bits > 64:
            raise ProfileValidationError(f"{path.name}: sensitive_layers[{idx}] invalid 'min_bits'.")
        if sev not in ("critical", "high", "medium", "low"):
            raise ProfileValidationError(f"{path.name}: sensitive_layers[{idx}] invalid severity {sev!r}.")
        if not isinstance(reason, str) or not reason:
            raise ProfileValidationError(f"{path.name}: sensitive_layers[{idx}] missing 'reason'.")

        try:
            cregex = re.compile(pat)
            cexc = re.compile(exc_pat) if exc_pat else None
        except re.error as exc:
            raise ProfileValidationError(f"{path.name}: invalid regex in {rname!r}: {exc}") from exc

        rules.append(
            SensitiveLayerRule(
                name=rname,
                pattern=pat,
                exclude_pattern=exc_pat,
                min_bits=float(min_bits),
                severity=sev,
                reason=reason,
                compiled_regex=cregex,
                compiled_exclude=cexc,
            )
        )

    return ArchitectureProfile(
        id=pid,
        name=pname,
        match_architectures=tuple(x.lower() for x in match_arches),
        match_model_types=tuple(x.lower() for x in match_types),
        sensitive_layers=tuple(rules),
        col_parallel_suffixes=tuple(col_suf),
        row_parallel_suffixes=tuple(row_suf),
    )


_DEFAULT_ARCH_PROFILES: dict[str, ArchitectureProfile] | None = None
_DEFAULT_GPU_PROFILES: dict[str, GPUProfile] | None = None
_DEFAULT_ENGINE_PROFILES: dict[str, EngineProfile] | None = None
_DEFAULT_ADVISORIES: dict[str, Any] | None = None


def load_architecture_profiles(arch_dir: Path | None = None) -> dict[str, ArchitectureProfile]:
    """Load and validate all `architectures/*.toml` files."""
    global _DEFAULT_ARCH_PROFILES
    if arch_dir is None and _DEFAULT_ARCH_PROFILES is not None:
        return _DEFAULT_ARCH_PROFILES

    folder = arch_dir or (_pkg_dir() / "architectures")
    files = sorted(folder.glob("*.toml"))
    if not files:
        raise ProfileValidationError(f"No architecture TOML profiles found in {folder}")

    profiles: dict[str, ArchitectureProfile] = {}
    for fp in files:
        prof = _parse_arch_profile(fp)
        if prof.id in profiles:
            raise ProfileValidationError(f"Duplicate architecture profile id {prof.id!r} in {fp.name}")
        profiles[prof.id] = prof

    if "default" not in profiles:
        raise ProfileValidationError("Missing required 'default' architecture profile.")

    if arch_dir is None:
        _DEFAULT_ARCH_PROFILES = profiles
    return profiles


def resolve_architecture_profile(
    config: dict[str, Any],
    profiles: dict[str, ArchitectureProfile] | None = None,
) -> ArchitectureProfile:
    """Match a checkpoint's `config.json` against specific architecture profiles, with default fallback."""
    loaded = profiles or load_architecture_profiles()
    raw_arches = config.get("architectures")
    arches = [a.lower() for a in raw_arches if isinstance(a, str)] if isinstance(raw_arches, list) else []
    mtype = str(config.get("model_type") or "").lower()
    text_cfg = config.get("text_config")
    sub_type = str(text_cfg.get("model_type") or "").lower() if isinstance(text_cfg, dict) else ""

    for pid, prof in loaded.items():
        if pid == "default":
            continue
        if any(a in prof.match_architectures for a in arches):
            return prof
        if mtype and mtype in prof.match_model_types:
            return prof
        if sub_type and sub_type in prof.match_model_types:
            return prof
    return loaded["default"]


def load_gpu_profiles(hw_dir: Path | None = None) -> dict[str, GPUProfile]:
    """Load and validate all GPU definitions from `hardware/gpus.toml`."""
    global _DEFAULT_GPU_PROFILES
    if hw_dir is None and _DEFAULT_GPU_PROFILES is not None:
        return _DEFAULT_GPU_PROFILES

    path = (hw_dir or (_pkg_dir() / "hardware")) / "gpus.toml"
    raw = _load_toml(path)
    gpus_raw = raw.get("gpus")
    if not isinstance(gpus_raw, list) or not gpus_raw:
        raise ProfileValidationError("hardware/gpus.toml must define a non-empty [[gpus]] list.")

    gpus: dict[str, GPUProfile] = {}
    for idx, item in enumerate(gpus_raw):
        if not isinstance(item, dict):
            raise ProfileValidationError(f"gpus.toml: gpus[{idx}] must be a table.")
        gid = item.get("id")
        if not isinstance(gid, str) or not gid:
            raise ProfileValidationError(f"gpus.toml: gpus[{idx}] missing 'id'.")
        aliases = item.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
            raise ProfileValidationError(f"gpus.toml: {gid} 'aliases' must be string list.")

        for req_bool in (
            "supports_bf16",
            "supports_fp8_w8a8",
            "supports_fp8_block_scale",
            "supports_nvfp4",
            "supports_marlin",
        ):
            if not isinstance(item.get(req_bool), bool):
                raise ProfileValidationError(f"gpus.toml: {gid} missing boolean {req_bool!r}.")

        if not isinstance(item.get("vram_gb"), (int, float)) or item["vram_gb"] <= 0:
            raise ProfileValidationError(f"gpus.toml: {gid} invalid 'vram_gb'.")
        if not isinstance(item.get("compute_capability"), int):
            raise ProfileValidationError(f"gpus.toml: {gid} invalid integer 'compute_capability'.")

        prof = GPUProfile(
            id=gid,
            aliases=tuple(a.lower() for a in aliases),
            vendor=str(item.get("vendor") or "nvidia").lower(),
            arch=str(item.get("arch") or ""),
            compute_capability=item["compute_capability"],
            vram_gb=float(item["vram_gb"]),
            supports_bf16=item["supports_bf16"],
            supports_fp8_w8a8=item["supports_fp8_w8a8"],
            supports_fp8_block_scale=item["supports_fp8_block_scale"],
            supports_nvfp4=item["supports_nvfp4"],
            supports_marlin=item["supports_marlin"],
        )
        gpus[gid.upper()] = prof

    if hw_dir is None:
        _DEFAULT_GPU_PROFILES = gpus
    return gpus


def resolve_gpu_profile(gpu_query: str, gpus: dict[str, GPUProfile] | None = None) -> GPUProfile:
    """Resolve a user GPU name or SM arch (e.g. `H100`, `L40S`, `sm_89`, `GB10`) to a `GPUProfile`."""
    loaded = gpus or load_gpu_profiles()
    q = gpu_query.strip().lower()
    for prof in loaded.values():
        if prof.id.lower() == q or q in prof.aliases:
            return prof
    for prof in loaded.values():
        if prof.arch.lower() == q:
            return prof
    known = ", ".join(sorted(loaded.keys()))
    raise ProfileValidationError(f"Unknown GPU target {gpu_query!r}. Known GPUs: {known}")


def load_engine_profiles(hw_dir: Path | None = None) -> dict[str, EngineProfile]:
    """Load and validate serving engine definitions from `hardware/engines.toml`."""
    global _DEFAULT_ENGINE_PROFILES
    if hw_dir is None and _DEFAULT_ENGINE_PROFILES is not None:
        return _DEFAULT_ENGINE_PROFILES

    path = (hw_dir or (_pkg_dir() / "hardware")) / "engines.toml"
    raw = _load_toml(path)
    eng_raw = raw.get("engines")
    if not isinstance(eng_raw, list) or not eng_raw:
        raise ProfileValidationError("hardware/engines.toml must define a non-empty [[engines]] list.")

    engines: dict[str, EngineProfile] = {}
    for idx, item in enumerate(eng_raw):
        if not isinstance(item, dict):
            raise ProfileValidationError(f"engines.toml: engines[{idx}] must be a table.")
        eid = item.get("id")
        if not isinstance(eid, str) or not eid:
            raise ProfileValidationError(f"engines.toml: engines[{idx}] missing 'id'.")
        containers = item.get("supported_containers", [])
        qmethods = item.get("supported_quant_methods", [])
        restricted = item.get("fp8_block_restricted_arches", [])
        if not isinstance(containers, list) or not containers:
            raise ProfileValidationError(f"engines.toml: {eid} missing 'supported_containers'.")
        if not isinstance(qmethods, list) or not qmethods:
            raise ProfileValidationError(f"engines.toml: {eid} missing 'supported_quant_methods'.")

        engines[eid.lower()] = EngineProfile(
            id=eid.lower(),
            name=str(item.get("name") or eid),
            supported_containers=tuple(str(c).lower() for c in containers),
            supported_quant_methods=tuple(str(m).lower() for m in qmethods),
            marlin_in_align=int(item.get("marlin_in_align", 128)),
            marlin_out_align=int(item.get("marlin_out_align", 64)),
            fp8_block_restricted_arches=tuple(str(a).lower() for a in restricted),
        )

    if hw_dir is None:
        _DEFAULT_ENGINE_PROFILES = engines
    return engines


def load_advisories() -> dict[str, Any]:
    """Load the bundled engine advisories, a snapshot of the OSV database."""
    global _DEFAULT_ADVISORIES
    if _DEFAULT_ADVISORIES is None:
        text = (_pkg_dir() / "hardware" / "advisories.json").read_text(encoding="utf-8")
        _DEFAULT_ADVISORIES = json.loads(text)
    return _DEFAULT_ADVISORIES


def validate_all_profiles() -> tuple[int, int, int]:
    """Validate all bundled architecture, GPU, and engine TOML profiles."""
    arches = load_architecture_profiles()
    gpus = load_gpu_profiles()
    engines = load_engine_profiles()
    return len(arches), len(gpus), len(engines)
