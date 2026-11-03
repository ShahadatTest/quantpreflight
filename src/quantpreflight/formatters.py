"""Human text, JSON, and SARIF 2.1.0 output formatters for `quantpreflight`."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from . import __version__
from .models import ScanReport, Severity, TargetSpec
from .profiles import ProfileValidationError, load_engine_profiles, resolve_gpu_profile
from .rules import RULE_CATALOG

if TYPE_CHECKING:
    from .suggest import Candidate


def _fmt_bytes(num_bytes: int) -> str:
    if num_bytes >= 1024**3:
        return f"{num_bytes / (1024**3):.2f} GiB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / (1024**2):.2f} MiB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KiB"
    return f"{num_bytes} B"


def _gpu_name(query: str) -> str:
    """Name the GPU profile an alias such as `sm_89` resolved to."""
    try:
        gpu = resolve_gpu_profile(query)
    except ProfileValidationError:
        return query
    return gpu.id if gpu.id.lower() == query.lower() else f"{gpu.id} (from {query})"


def _setup_lines(report: ScanReport) -> list[str]:
    plan = report.setup
    if plan is None:
        return []
    if plan.status == "blocked":
        rules = ", ".join(dict.fromkeys(f.rule_id for f in plan.blocking))
        return [
            "Setup: no --tp or context length makes this start.",
            *(f"  {reason}" for reason in plan.changes),
            f"  {rules} needs a different checkpoint, GPU, or engine.",
            f"  Find one that works: quantpreflight suggest {report.source}"
            + (f" --gpu {report.target.gpu}" if report.target.gpu else ""),
        ]
    lines = ["Setup: ready as asked." if plan.status == "ready" else "Setup: it starts with these changes."]
    lines += [f"  - {change}" for change in plan.changes]
    if plan.max_context is not None and plan.context_len is not None:
        per = "per request" if report.target.concurrency == 1 else f"for each of {report.target.concurrency} requests"
        if plan.max_context > plan.context_len:
            lines.append(f"  Runs with {plan.context_len} tokens {per}, and up to {plan.max_context} would fit at tp={plan.tp}.")
        else:
            lines.append(f"  Fits {plan.context_len} tokens {per} at tp={plan.tp}.")
    if plan.command:
        lines.append(f"  Run:  {plan.command}")
    return lines


def format_text_report(report: ScanReport) -> str:
    """Render a concise terminal summary and prioritized finding list."""
    tgt_parts = [f"engine={report.target.engine}", f"tp={report.target.tp}"]
    if report.target.gpu:
        tgt_parts.insert(1, f"gpu={_gpu_name(report.target.gpu)}")

    lines = [
        f"quantpreflight v{__version__}",
        f"  Source:       {report.source}",
        f"  Architecture: {report.architecture} ({report.container_format})",
        f"  Quantization: {report.quant_summary}",
        f"  Weights:      {_fmt_bytes(report.total_weight_bytes)} across {report.shard_count} file(s), {report.tensor_count} tensors",
        f"  Headers read: {_fmt_bytes(report.header_bytes_fetched)} (zero weight bytes downloaded)",
        f"  Target:       {', '.join(tgt_parts)}",
    ]
    if report.estimated_vram_per_gpu_gb is not None:
        lines.append(f"  Est. VRAM:    {report.estimated_vram_per_gpu_gb:.2f} GiB / GPU (weights, runtime, and KV cache)")

    lines.append("")
    setup = _setup_lines(report)
    if not report.findings:
        lines.append("Result: PASS (0 findings)")
        return "\n".join(lines + ([""] + setup if setup else []))

    counts: dict[str, int] = {}
    for f in report.findings:
        counts[f.severity.value] = counts.get(f.severity.value, 0) + 1
    summary_str = ", ".join(f"{v} {k}" for k, v in counts.items())
    lines.append(f"Findings: {len(report.findings)} ({summary_str})")
    lines.append("-" * 72)

    for idx, f in enumerate(report.findings, start=1):
        lines.append(f"[{idx}] {f.severity.value.upper()} {f.rule_id}: {f.title}")
        lines.append(f"    Location: {f.location}")
        lines.append(f"    Detail:   {f.message}")
        lines.append(f"    Fix:      {f.remediation}")
        if f.references:
            lines.append(f"    Ref:      {', '.join(f.references)}")
        lines.append("")

    return "\n".join(lines + setup).rstrip()


def format_json_report(report: ScanReport) -> str:
    """Serialize `ScanReport` as deterministic JSON."""
    payload: dict[str, Any] = {
        "version": __version__,
        "source": report.source,
        "container_format": report.container_format,
        "architecture": report.architecture,
        "quant_summary": report.quant_summary,
        "total_weight_bytes": report.total_weight_bytes,
        "header_bytes_fetched": report.header_bytes_fetched,
        "tensor_count": report.tensor_count,
        "shard_count": report.shard_count,
        "estimated_vram_per_gpu_gb": report.estimated_vram_per_gpu_gb,
        "target": {
            "engine": report.target.engine,
            "gpu": report.target.gpu,
            "tp": report.target.tp,
            "context_len": report.target.context_len,
            "marlin_input_dtype": report.target.marlin_input_dtype,
        },
        "findings": [f.to_dict() for f in report.findings],
    }
    if report.setup is not None:
        payload["setup"] = report.setup.to_dict()
    return json.dumps(payload, indent=2)


def _sarif_level(sev: Severity) -> str:
    if sev in (Severity.CRITICAL, Severity.HIGH):
        return "error"
    if sev == Severity.MEDIUM:
        return "warning"
    return "note"


def format_sarif_report(report: ScanReport) -> str:
    """Serialize `ScanReport` as GitHub Code Scanning compatible SARIF 2.1.0."""
    rules_list: list[dict[str, Any]] = []
    for rid, (default_sev, short_desc, full_desc) in sorted(RULE_CATALOG.items()):
        rules_list.append(
            {
                "id": rid,
                "name": short_desc.replace(" ", ""),
                "shortDescription": {"text": short_desc},
                "fullDescription": {"text": full_desc},
                "defaultConfiguration": {"level": _sarif_level(default_sev)},
            }
        )

    results: list[dict[str, Any]] = []
    for f in report.findings:
        results.append(
            {
                "ruleId": f.rule_id,
                "level": _sarif_level(f.severity),
                "message": {"text": f"{f.title}: {f.message} Fix: {f.remediation}"},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": f.location.replace("\\", "/")},
                            "region": {"startLine": 1},
                        }
                    }
                ],
                "properties": {
                    "severity": f.severity.value,
                    "tensors": list(f.tensors),
                    "references": list(f.references),
                },
            }
        )

    sarif: dict[str, Any] = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "quantpreflight",
                        "version": __version__,
                        "informationUri": "https://github.com/layer8labs8/quantpreflight",
                        "rules": rules_list,
                    }
                },
                "results": results,
            }
        ],
    }
    return json.dumps(sarif, indent=2)


def _target_label(target: TargetSpec) -> str:
    engines = load_engine_profiles()
    engine = engines[target.engine].name if target.engine in engines else target.engine
    where = f"on {_gpu_name(target.gpu)}" if target.gpu else "with no GPU named"
    return f"{engine} {where}, tp={target.tp}"


def _candidate_detail(item: "Candidate") -> str:
    if item.plan is None:
        return item.error or ""
    if item.plan.status == "blocked":
        return ", ".join(dict.fromkeys(f.rule_id for f in item.plan.blocking))
    parts = [f"tp={item.plan.tp}"]
    if item.plan.context_len:
        parts.append(f"{item.plan.context_len} tokens")
    return ", ".join(parts)


def format_suggestions(base: str, target: TargetSpec, candidates: list["Candidate"]) -> str:
    """Render the ranked versions of a model as a table, with the best pick and its command."""
    lines = [f"quantpreflight v{__version__}", f"Versions of {base} for {_target_label(target)}:", ""]
    rows = [("Setup", "Model", "Format", "Weights", "Details")]
    for item in candidates:
        status = item.plan.status if item.plan else "skipped"
        weights = f"{item.weights_gib:.2f} GiB" if item.weights_gib is not None else ""
        rows.append((status, item.repo, item.quant, weights, _candidate_detail(item)))
    widths = [max(len(row[col]) for row in rows) for col in range(4)]
    for row in rows:
        lines.append("  " + "  ".join(cell.ljust(widths[col]) for col, cell in enumerate(row[:4])) + "  " + row[4])

    best = next((item for item in candidates if item.starts), None)
    lines.append("")
    if best is None:
        lines.append("None of these start on this target. A GPU with more memory or newer kernels, or another engine, may help.")
        return "\n".join(lines)
    lines.append(f"Best pick: {best.repo}, the highest precision that starts.")
    for change in best.plan.changes if best.plan else ():
        lines.append(f"  - {change}")
    if best.plan and best.plan.command:
        lines.append(f"  Run:  {best.plan.command}")
    return "\n".join(lines)


def format_suggestions_json(base: str, target: TargetSpec, candidates: list["Candidate"]) -> str:
    payload = {
        "version": __version__,
        "base_model": base,
        "target": {"engine": target.engine, "gpu": target.gpu, "tp": target.tp, "context_len": target.context_len},
        "candidates": [item.to_dict() for item in candidates],
    }
    return json.dumps(payload, indent=2)
