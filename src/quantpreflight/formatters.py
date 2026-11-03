"""Human text, JSON, and SARIF 2.1.0 output formatters for `quantpreflight`."""

from __future__ import annotations

import json
from typing import Any

from . import __version__
from .models import ScanReport, Severity
from .profiles import ProfileValidationError, resolve_gpu_profile
from .rules import RULE_CATALOG


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
    if not report.findings:
        lines.append("Result: PASS (0 findings)")
        return "\n".join(lines)

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

    return "\n".join(lines).rstrip()


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
