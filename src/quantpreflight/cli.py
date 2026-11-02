"""Command-line interface for `quantpreflight`."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Sequence

from . import __version__
from .formatters import format_json_report, format_sarif_report, format_text_report
from .gguf_io import GGUFFormatError
from .loader import CheckpointLoadError, load_checkpoint_metadata
from .models import Severity, TargetSpec
from .profiles import (
    ProfileValidationError,
    load_gpu_profiles,
    validate_all_profiles,
)
from .rules import RULE_CATALOG, analyze_checkpoint
from .safetensors_io import SafetensorsFormatError
from .suppress import load_project_config, resolve_ignored_rules


# What a malformed checkpoint or a failed fetch can raise while scanning
SCAN_ERRORS = (
    CheckpointLoadError,
    SafetensorsFormatError,
    GGUFFormatError,
    ProfileValidationError,
    ValueError,
    OSError,
    RecursionError,
)


def _parse_fail_on(raw: str) -> Severity | None:
    low = raw.strip().lower()
    if low == "none":
        return None
    return Severity(low)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantpreflight",
        description=(
            "Zero-GPU pre-flight scanner for 32/16/8/4/2-bit quantized AI checkpoints "
            "(Safetensors & GGUF)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"quantpreflight {__version__}")

    sub = parser.add_subparsers(dest="command")

    for cmd_name in ("check", "scan"):
        chk = sub.add_parser(
            cmd_name,
            help="Scan a Hugging Face repo ID, HTTPS URL, or local checkpoint path.",
        )
        chk.add_argument(
            "targets",
            nargs="+",
            help="Hugging Face repo (org/model), URL, or local directory/file path(s).",
        )
        chk.add_argument(
            "--gpu",
            default=None,
            help="Target GPU profile or SM arch (e.g. H100, A100, L40S, T4, B200, GB10, MI300X, sm_89).",
        )
        chk.add_argument(
            "--engine",
            default="vllm",
            choices=["vllm", "sglang", "trtllm", "llamacpp"],
            help="Target inference engine (default: vllm).",
        )
        chk.add_argument(
            "--tp",
            type=int,
            default=1,
            help="Tensor parallel degree (default: 1).",
        )
        chk.add_argument(
            "--engine-version",
            default=None,
            help="Installed engine version, checked against published advisories (vLLM and SGLang).",
        )
        chk.add_argument(
            "--concurrency",
            type=int,
            default=1,
            help="Requests served at once, for the KV-cache estimate (default: 1).",
        )
        chk.add_argument(
            "--context-len",
            type=int,
            default=None,
            help="Context length for smoke KV-cache VRAM estimation.",
        )
        chk.add_argument(
            "--vram-gb",
            type=float,
            default=None,
            help="Override per-GPU physical VRAM in GiB.",
        )
        chk.add_argument(
            "--gpu-memory-utilization",
            type=float,
            default=0.90,
            help="Usable GPU memory fraction (default: 0.90).",
        )
        chk.add_argument(
            "--marlin-input-dtype",
            choices=["fp16", "bf16", "fp8", "int8"],
            default=os.environ.get("VLLM_MARLIN_INPUT_DTYPE"),
            help="Simulate VLLM_MARLIN_INPUT_DTYPE kernel flag (fp8 or int8).",
        )
        chk.add_argument(
            "--revision",
            default="main",
            help="Hugging Face git revision / branch / commit (default: main).",
        )
        chk.add_argument(
            "--token",
            default=None,
            help="Hugging Face token (defaults to HF_TOKEN environment variable).",
        )
        chk.add_argument(
            "--ignore",
            action="append",
            default=[],
            help="Comma-separated rule IDs to suppress (e.g. QPF003,QPF012).",
        )
        chk.add_argument(
            "--config",
            type=pathlib.Path,
            default=None,
            help="Path to .quantpreflight.toml configuration file.",
        )
        chk.add_argument(
            "--fail-on",
            choices=["critical", "high", "medium", "low", "none"],
            default="high",
            help="Exit with code 1 when any finding meets or exceeds this severity (default: high).",
        )
        chk.add_argument(
            "--format",
            choices=["text", "json", "sarif"],
            default="text",
            help="Output report format (default: text).",
        )
        chk.add_argument(
            "--output",
            type=pathlib.Path,
            default=None,
            help="Optional file path to write the rendered output.",
        )

    explain = sub.add_parser("explain", help="Show what a rule means.")
    explain.add_argument("rule", help="Rule ID, such as QPF006.")

    sub.add_parser("list-gpus", help="List bundled GPU hardware profiles.")
    sub.add_parser("list-rules", help="List all pre-flight rules.")
    sub.add_parser("validate-profiles", help="Validate all TOML architecture and hardware profiles.")

    return parser


def _cmd_list_gpus() -> int:
    gpus = load_gpu_profiles()
    print(f"{'GPU':<12} {'Vendor':<8} {'Arch':<8} {'VRAM':<10} {'BF16':<6} {'FP8':<6} {'BlkFP8':<8} {'NVFP4':<6}")
    print("-" * 72)
    for gid in sorted(gpus.keys()):
        g = gpus[gid]
        print(
            f"{g.id:<12} {g.vendor:<8} {g.arch:<8} {g.vram_gb:>5.0f} GiB  "
            f"{str(g.supports_bf16):<6} {str(g.supports_fp8_w8a8):<6} "
            f"{str(g.supports_fp8_block_scale):<8} {str(g.supports_nvfp4):<6}"
        )
    return 0


def _cmd_list_rules() -> int:
    for rid, (sev, title, desc) in sorted(RULE_CATALOG.items()):
        print(f"{rid} [{sev.value.upper():<8}] {title}")
        print(f"    {desc}")
    return 0


def _cmd_explain(rule: str) -> int:
    rule_id = rule.strip().upper()
    if rule_id not in RULE_CATALOG:
        known = ", ".join(sorted(RULE_CATALOG))
        print(f"Unknown rule {rule!r}. Known rules: {known}", file=sys.stderr)
        return 2
    severity, title, description = RULE_CATALOG[rule_id]
    print(f"{rule_id}: {title}")
    print(f"Highest severity: {severity.value}")
    print(description)
    print(f"More: https://layer8labs8.github.io/quantpreflight/#{rule_id}")
    return 0


def _cmd_validate_profiles() -> int:
    try:
        n_arch, n_gpu, n_eng = validate_all_profiles()
    except ProfileValidationError as exc:
        print(f"Profile validation error: {exc}", file=sys.stderr)
        return 2
    print(f"Validated {n_arch} architecture profiles, {n_gpu} GPU profiles, and {n_eng} engine profiles.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "list-gpus":
        return _cmd_list_gpus()
    if args.command == "list-rules":
        return _cmd_list_rules()
    if args.command == "explain":
        return _cmd_explain(args.rule)
    if args.command == "validate-profiles":
        return _cmd_validate_profiles()

    if args.tp < 1:
        print("Error: --tp must be >= 1.", file=sys.stderr)
        return 2
    if args.concurrency < 1:
        print("Error: --concurrency must be >= 1.", file=sys.stderr)
        return 2
    if not (0.05 <= args.gpu_memory_utilization <= 1.0):
        print("Error: --gpu-memory-utilization must be between 0.05 and 1.0.", file=sys.stderr)
        return 2

    try:
        proj_cfg = load_project_config(args.config)
        ignored = resolve_ignored_rules(args.ignore, proj_cfg)
        fail_threshold = _parse_fail_on(args.fail_on)
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    target_spec = TargetSpec(
        engine=args.engine,
        gpu=args.gpu or proj_cfg.get("gpu"),
        tp=args.tp,
        context_len=args.context_len,
        vram_gb_override=args.vram_gb,
        gpu_memory_utilization=args.gpu_memory_utilization,
        marlin_input_dtype=args.marlin_input_dtype,
        engine_version=args.engine_version,
        concurrency=args.concurrency,
    )

    # Deduplicate targets when pre-commit passes both config.json and index.json from one directory
    seen_sources: set[str] = set()
    reports = []
    for raw_target in args.targets:
        p = pathlib.Path(raw_target)
        dedupe_key = str(p.parent.resolve()) if p.is_file() and p.name in ("config.json", "model.safetensors.index.json") else raw_target
        if dedupe_key in seen_sources:
            continue
        seen_sources.add(dedupe_key)

        try:
            meta = load_checkpoint_metadata(
                raw_target,
                revision=args.revision,
                token=args.token,
                prefer_gguf=args.engine == "llamacpp",
            )
            rep = analyze_checkpoint(meta, target_spec, ignored_rules=ignored)
            reports.append(rep)
        except SCAN_ERRORS as exc:
            print(f"Error scanning {raw_target}: {exc}", file=sys.stderr)
            return 2

    if args.format == "json":
        rendered = format_json_report(reports[0]) if len(reports) == 1 else (
            "[\n" + ",\n".join(format_json_report(r) for r in reports) + "\n]"
        )
    elif args.format == "sarif":
        # Merge findings if multiple targets were passed
        primary = reports[0]
        for extra in reports[1:]:
            primary.findings.extend(extra.findings)
        rendered = format_sarif_report(primary)
    else:
        rendered = "\n\n".join(format_text_report(r) for r in reports)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)

    any_failed = any(not r.passed(fail_threshold) for r in reports)
    return 1 if any_failed else 0
