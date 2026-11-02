"""The README, site text, and project files must agree with the code."""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from build_site import BENCHMARK_GROUPS, INSTALL, QUESTIONS, SUPPORT, _evidence
from quantpreflight import __version__
from quantpreflight.gguf_io import MAX_GGUF_HEADER_BYTES
from quantpreflight.loader import USER_AGENT
from quantpreflight.profiles import load_engine_profiles, load_gpu_profiles
from quantpreflight.rules import RULE_CATALOG
from quantpreflight.safetensors_io import MAX_SAFETENSORS_HEADER_BYTES
from run_benchmark import Score, load_cases, run

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
ACCURACY = (ROOT / "docs" / "accuracy.md").read_text(encoding="utf-8")
USAGE = (ROOT / "docs" / "usage.md").read_text(encoding="utf-8")


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("rule_id", sorted(RULE_CATALOG))
def test_readme_explains_every_rule(rule_id: str) -> None:
    question, why = QUESTIONS[rule_id]
    assert f"| {question} | {why} | `{rule_id}` |" in README


@pytest.mark.parametrize("gpu", sorted(gpu.id for gpu in load_gpu_profiles().values()))
def test_readme_lists_every_gpu(gpu: str) -> None:
    assert gpu in README


@pytest.mark.parametrize("engine", sorted(load_engine_profiles()))
def test_readme_lists_every_engine(engine: str) -> None:
    assert load_engine_profiles()[engine].name in README


def test_benchmark_numbers_are_current() -> None:
    cases = load_cases()
    total = Score()
    for group, score in run(cases).items():
        total.add(score)
        row = f"| {BENCHMARK_GROUPS[group]} | {len(cases[group])} | {score.found} | {len(score.missed)} | {len(score.false_alarms)} |"
        assert row in ACCURACY
    count = sum(len(items) for items in cases.values())
    assert f"{total.recall:.1%} recall and {total.precision:.0%} precision" in ACCURACY
    assert f"{count} small checkpoints" in ACCURACY
    assert f"| Labelled benchmark | {total.recall:.1%} recall, {total.precision:.0%} precision on {count} cases |" in README


def test_real_results_in_the_docs_are_current() -> None:
    numbers = _evidence()
    assert f"{numbers['gpu_match']} of {numbers['gpu_total']} predicted correctly" in README
    assert f"| Public Hugging Face models scanned | {numbers['public_count']}, with 0 crashes |" in README
    assert f"{numbers['gpu_match']} of {numbers['gpu_total']}" in ACCURACY
    assert numbers["public_count"] in ACCURACY


def test_every_cli_option_is_documented() -> None:
    options = set(re.findall(r'"(--[a-z-]+)"', _text("src/quantpreflight/cli.py"))) - {"--version"}
    assert options
    assert all(f"`{option}`" in USAGE for option in options), options


def test_readme_install_command_matches_the_site() -> None:
    assert INSTALL in README


def test_readme_links_point_at_real_files() -> None:
    for path in re.findall(r"github\.com/layer8labs8/quantpreflight/(?:blob|tree)/main/([^)\"\s]+)", README):
        assert (ROOT / path).exists(), path
    for path in re.findall(r"quantpreflight/main/(docs/assets/[^\"]+)", README):
        assert (ROOT / path).exists(), path
    for page in (README, ACCURACY, USAGE):
        for path in re.findall(r"\]\((?!https?:|#)([^)#]+)", page):
            base = ROOT if page is README else ROOT / "docs"
            assert (base / path).exists(), path


def test_readme_never_lists_wallets_that_differ_from_the_site() -> None:
    for _, address in SUPPORT:
        assert (address in README) is False
    assert "/#support" in README


def test_version_is_the_same_everywhere() -> None:
    assert f'version = "{__version__}"' in _text("pyproject.toml")
    assert f"## [{__version__}]" in _text("CHANGELOG.md")
    assert USER_AGENT == f"quantpreflight/{__version__}"


def test_security_policy_states_the_real_limits() -> None:
    policy = _text("SECURITY.md")
    assert f"`{MAX_SAFETENSORS_HEADER_BYTES // 2**20} MiB` max Safetensors JSON header" in policy
    assert MAX_GGUF_HEADER_BYTES == 32 * 2**20


def test_funding_points_at_the_support_section() -> None:
    assert "layer8labs8.github.io/quantpreflight/#support" in _text(".github/FUNDING.yml")


def test_action_exposes_the_options_the_readme_uses() -> None:
    action = _text("action.yml")
    for name in ("target", "engine", "gpu", "tp", "engine-version", "fail-on", "sarif-file"):
        assert f"  {name}:" in action


def test_pre_commit_hook_runs_the_cli() -> None:
    hooks = _text(".pre-commit-hooks.yaml")
    assert "id: quantpreflight" in hooks
    assert "entry: quantpreflight check" in hooks


def test_issue_templates_exist() -> None:
    names = {path.name for path in (ROOT / ".github" / "ISSUE_TEMPLATE").iterdir()}
    assert {"detection_problem.yml", "new_profile.yml"} <= names
