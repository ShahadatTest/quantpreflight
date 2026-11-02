"""The labelled benchmark and the demo checkpoints must keep their results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from quantpreflight.cli import main
from quantpreflight.loader import load_checkpoint_metadata
from quantpreflight.rules import RULE_CATALOG, analyze_checkpoint
from run_benchmark import GAPS, Score, load_cases, run, score_case

FIXTURES = Path(__file__).parent / "fixtures"
CASES = load_cases()
ALL_CASES = [case for items in CASES.values() for case in items]
SCORED = [case for case in ALL_CASES if case["group"] != GAPS]


@pytest.mark.parametrize("case", SCORED, ids=lambda case: case["id"])
def test_case_gives_the_labelled_result(case: dict[str, Any]) -> None:
    score = score_case(case)
    assert score.missed == []
    assert score.false_alarms == []


@pytest.mark.parametrize("case", CASES[GAPS], ids=lambda case: case["id"])
def test_known_gaps_are_still_gaps(case: dict[str, Any]) -> None:
    assert score_case(case).missed, "This gap is closed. Move the case to a scored group."


def test_every_case_is_well_formed() -> None:
    ids = [case["id"] for case in ALL_CASES]
    assert len(ids) == len(set(ids))
    for case in ALL_CASES:
        assert case["description"]
        assert set(case["must_fire"]) <= set(RULE_CATALOG)
        assert set(case["must_not_fire"]) <= set(RULE_CATALOG)


def test_every_rule_is_exercised() -> None:
    labelled = {rule for case in SCORED for rule in case["must_fire"]}
    assert labelled == set(RULE_CATALOG)


def test_scores_add_up() -> None:
    total = Score()
    for score in run(CASES).values():
        total.add(score)
    assert total.found == sum(len(case["must_fire"]) for case in SCORED)
    assert len(total.missed) == len(CASES[GAPS])
    assert total.false_alarms == []
    assert total.precision == 1.0
    assert 0.9 < total.recall < 1.0


def test_score_ratios_handle_empty_groups() -> None:
    assert Score().recall == 1.0
    assert Score().precision == 1.0
    assert Score(found=3, missed=["a"]).recall == 0.75
    assert Score(found=1, false_alarms=["a"]).precision == 0.5


def test_quiet_case_counts_any_finding_as_a_false_alarm() -> None:
    case = dict(next(case for case in CASES["parallel"] if case["id"] == "tp_heads_not_divisible"))
    case["must_fire"] = []
    assert score_case(case).false_alarms == [f"{case['id']}: QPF006"]


def test_good_demo_checkpoint_is_clean(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(FIXTURES / "good-model"), "--gpu", "H100", "--tp", "4"]) == 0
    assert "Result: PASS (0 findings)" in capsys.readouterr().out


def test_broken_demo_checkpoint_shows_many_problems(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(FIXTURES / "broken-model"), "--gpu", "H100", "--tp", "4", "--format", "json"]) == 1
    fired = {finding["rule_id"] for finding in json.loads(capsys.readouterr().out)["findings"]}
    assert fired == {"QPF001", "QPF002", "QPF003", "QPF006"}


def test_demo_checkpoints_hold_no_weight_data() -> None:
    for folder in ("good-model", "broken-model"):
        meta = load_checkpoint_metadata(str(FIXTURES / folder))
        assert (FIXTURES / folder / "model.safetensors").stat().st_size < 4096
        assert meta.total_weight_bytes > 100 * 1024**2
        assert not [f for f in analyze_checkpoint(meta).findings if f.rule_id == "QPF013"]
