"""Refresh the bundled engine advisories from the OSV database."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Any
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "src" / "quantpreflight" / "hardware" / "advisories.json"
PACKAGES = {"vllm": "vllm", "sglang": "sglang"}
OSV_QUERY = "https://api.osv.dev/v1/query"


def _query(package: str) -> list[dict[str, Any]]:
    vulns: list[dict[str, Any]] = []
    body: dict[str, Any] = {"package": {"name": package, "ecosystem": "PyPI"}}
    while True:
        request = urllib.request.Request(
            OSV_QUERY, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            page = json.load(response)
        vulns += page.get("vulns", [])
        if not page.get("next_page_token"):
            return vulns
        body["page_token"] = page["next_page_token"]


def _ranges(vuln: dict[str, Any], package: str) -> list[dict[str, str | None]]:
    ranges: list[dict[str, str | None]] = []
    for affected in vuln.get("affected", []):
        if str(affected.get("package", {}).get("name", "")).lower() != package:
            continue
        for item in affected.get("ranges", []):
            if item.get("type") != "ECOSYSTEM":
                continue
            events: dict[str, str] = {}
            for event in item.get("events", []):
                events.update(event)
            ranges.append(
                {
                    "introduced": events.get("introduced", "0"),
                    "fixed": events.get("fixed"),
                    "last_affected": events.get("last_affected"),
                }
            )
    return ranges


def merge(package: str, vulns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse OSV records that describe the same CVE into one entry."""
    merged: dict[str, dict[str, Any]] = {}
    for vuln in vulns:
        ranges = _ranges(vuln, package)
        if not ranges:
            continue
        cves = sorted(alias for alias in vuln.get("aliases", []) if alias.startswith("CVE-"))
        key = vuln["id"] if vuln["id"].startswith("CVE-") else (cves[0] if cves else vuln["id"])
        entry = merged.setdefault(key, {"id": key, "published": vuln.get("published", "")[:10], "ranges": []})
        entry["published"] = max(entry["published"], vuln.get("published", "")[:10])
        for item in ranges:
            if item not in entry["ranges"]:
                entry["ranges"].append(item)
    return sorted(merged.values(), key=lambda entry: (entry["published"], entry["id"]), reverse=True)


def _today() -> str:
    return date.today().isoformat()


def main() -> None:
    engines = {engine: merge(package, _query(package)) for engine, package in PACKAGES.items()}
    current = json.loads(TARGET.read_text(encoding="utf-8")) if TARGET.is_file() else {}
    # A new date alone shouldn't count as a change worth a pull request
    updated = current["updated"] if current.get("engines") == engines else _today()
    data = {"source": "https://osv.dev", "updated": updated, "engines": engines}
    TARGET.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print({engine: len(items) for engine, items in engines.items()})


if __name__ == "__main__":
    main()
