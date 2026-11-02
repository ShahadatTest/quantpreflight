"""The advisory refresh tool and the bundled data it writes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import update_advisories
from update_advisories import merge

from quantpreflight.profiles import load_advisories
from quantpreflight.rules import _parse_version


def _vuln(vuln_id: str, *events: list[dict[str, str]], **fields: Any) -> dict[str, Any]:
    return {
        "id": vuln_id,
        "published": "2026-01-02T03:04:05Z",
        "affected": [
            {"package": {"name": "vllm"}, "ranges": [{"type": "ECOSYSTEM", "events": item} for item in events]},
        ],
        **fields,
    }


def test_merge_reads_each_kind_of_range() -> None:
    entries = merge(
        "vllm",
        [
            _vuln("GHSA-1", [{"introduced": "0.1.0"}, {"fixed": "0.2.0"}]),
            _vuln("GHSA-2", [{"introduced": "0.3.0"}, {"last_affected": "0.4.0"}]),
            _vuln("GHSA-3", [{"introduced": "0"}]),
        ],
    )
    ranges = {entry["id"]: entry["ranges"] for entry in entries}
    assert ranges["GHSA-1"] == [{"introduced": "0.1.0", "fixed": "0.2.0", "last_affected": None}]
    assert ranges["GHSA-2"] == [{"introduced": "0.3.0", "fixed": None, "last_affected": "0.4.0"}]
    assert ranges["GHSA-3"] == [{"introduced": "0", "fixed": None, "last_affected": None}]


def test_merge_joins_records_of_one_cve() -> None:
    first = _vuln("GHSA-1", [{"introduced": "0"}, {"fixed": "1.0"}], aliases=["CVE-2026-1"])
    second = _vuln("PYSEC-1", [{"introduced": "2.0"}, {"fixed": "3.0"}], aliases=["CVE-2026-1"], published="2026-05-05T00:00:00Z")
    entries = merge("vllm", [first, second])
    assert [entry["id"] for entry in entries] == ["CVE-2026-1"]
    assert len(entries[0]["ranges"]) == 2
    assert entries[0]["published"] == "2026-05-05"


def test_merge_keeps_one_copy_of_a_repeated_range() -> None:
    first = _vuln("GHSA-1", [{"introduced": "0"}, {"fixed": "1.0"}], aliases=["CVE-2026-1"])
    second = _vuln("PYSEC-1", [{"introduced": "0"}, {"fixed": "1.0"}], aliases=["CVE-2026-1"])
    assert len(merge("vllm", [first, second])[0]["ranges"]) == 1


def test_merge_skips_other_packages_and_git_ranges() -> None:
    other = {"id": "GHSA-9", "affected": [{"package": {"name": "torch"}, "ranges": [{"type": "ECOSYSTEM", "events": []}]}]}
    git = {"id": "GHSA-8", "affected": [{"package": {"name": "vllm"}, "ranges": [{"type": "GIT", "events": []}]}]}
    assert merge("vllm", [other, git]) == []


def test_merge_sorts_newest_first() -> None:
    old = _vuln("GHSA-1", [{"introduced": "0"}], published="2025-01-01T00:00:00Z")
    new = _vuln("GHSA-2", [{"introduced": "0"}], published="2026-01-01T00:00:00Z")
    assert [entry["id"] for entry in merge("vllm", [old, new])] == ["GHSA-2", "GHSA-1"]


def _refresh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, vulns: list[dict[str, Any]], today: str) -> dict[str, Any]:
    target = tmp_path / "advisories.json"
    monkeypatch.setattr(update_advisories, "TARGET", target)
    monkeypatch.setattr(update_advisories, "PACKAGES", {"vllm": "vllm"})
    monkeypatch.setattr(update_advisories, "_query", lambda package: vulns)
    monkeypatch.setattr(update_advisories, "_today", lambda: today)
    update_advisories.main()
    return json.loads(target.read_text(encoding="utf-8"))


def test_refresh_keeps_the_date_when_nothing_changed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    vulns = [_vuln("GHSA-1", [{"introduced": "0"}])]
    assert _refresh(monkeypatch, tmp_path, vulns, "2026-01-01")["updated"] == "2026-01-01"
    assert _refresh(monkeypatch, tmp_path, vulns, "2026-02-02")["updated"] == "2026-01-01"


def test_refresh_changes_the_date_with_new_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _refresh(monkeypatch, tmp_path, [_vuln("GHSA-1", [{"introduced": "0"}])], "2026-01-01")
    updated = _refresh(monkeypatch, tmp_path, [_vuln("GHSA-2", [{"introduced": "0"}])], "2026-02-02")
    assert updated["updated"] == "2026-02-02"


def test_bundled_data_is_well_formed() -> None:
    data = load_advisories()
    assert set(data["engines"]) == {"vllm", "sglang"}
    assert data["source"] == "https://osv.dev"
    for entries in data["engines"].values():
        assert entries
        for entry in entries:
            assert entry["id"]
            for item in entry["ranges"]:
                _parse_version(item["introduced"])
                for key in ("fixed", "last_affected"):
                    assert item[key] is None or _parse_version(item[key])
