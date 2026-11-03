"""Tests for the project website."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
import zipfile

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "website"))

from build_site import ACTION_REF, BENCHMARK_GROUPS, EXAMPLES, INSTALL, QUESTIONS, SUPPORT, build
from quantpreflight import __version__
from quantpreflight.profiles import load_engine_profiles, load_gpu_profiles
from quantpreflight.rules import RULE_CATALOG
from run_benchmark import load_cases
from site_scan import scan_repo, scan_text


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("site") / "_site"
    build(out)
    return out


@pytest.fixture(scope="module")
def page(site: Path) -> str:
    return (site / "index.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def data(page: str) -> dict[str, Any]:
    raw = page.split('<script id="site-data" type="application/json">')[1].split("</script>")[0]
    return json.loads(raw)


def _example(data: dict[str, Any], slug: str) -> dict[str, Any]:
    return next(example for example in data["examples"] if example["slug"] == slug)


def _rules(example: dict[str, Any]) -> set[str]:
    return {finding["rule_id"] for finding in example["result"]["findings"]}


def test_every_rule_has_a_question() -> None:
    assert set(QUESTIONS) == set(RULE_CATALOG)


def test_every_benchmark_group_has_a_label() -> None:
    assert set(BENCHMARK_GROUPS) == set(load_cases())


def test_page_has_no_unfilled_tokens(page: str) -> None:
    assert "<!-- " not in page


def test_page_lists_rules_and_version(page: str) -> None:
    assert all(f'<tr id="{rule_id}">' in page for rule_id in RULE_CATALOG)
    assert f"quantpreflight v{__version__}" in page
    assert f"layer8labs8/quantpreflight@{ACTION_REF}" in page


def test_page_shows_each_support_address(page: str) -> None:
    assert page.count('id="support"') == 1
    assert all(f'data-copy="{address}"' in page for _, address in SUPPORT)


def test_page_shows_the_benchmark(page: str) -> None:
    assert '<th scope="row">Splitting across GPUs</th>' in page
    assert "% recall" in page
    assert "<td>3</td><td>0</td><td>3</td><td>0</td>" in page


def test_page_shows_the_install_command(page: str) -> None:
    assert page.count(INSTALL) >= 3


def test_page_lists_all_engines_and_gpus(page: str, data: dict[str, Any]) -> None:
    assert {gpu["id"] for gpu in data["gpus"]} == {gpu.id for gpu in load_gpu_profiles().values()}
    assert {engine["id"] for engine in data["engines"]} == set(load_engine_profiles())
    assert all(f"<li>{engine['name']}</li>" in page for engine in data["engines"])
    assert all(f"<li>{gpu['id']}</li>" in page for gpu in data["gpus"])


def test_examples_match_what_they_show(data: dict[str, Any]) -> None:
    assert _rules(_example(data, "fits")) == set()
    assert _rules(_example(data, "too-big")) == {"QPF012"}
    assert _rules(_example(data, "wrong-gpu")) == {"QPF004"}
    assert _rules(_example(data, "bad-split")) == {"QPF006", "QPF008"}


def test_example_tabs_are_complete(data: dict[str, Any]) -> None:
    assert [example["slug"] for example in data["examples"]] == [slug for slug, _, _ in EXAMPLES]
    for example in data["examples"]:
        assert json.loads(example["text"])["quantization_config"]
        assert example["target"]["gpu"] in {gpu["id"] for gpu in data["gpus"]}


def test_scanner_archive_is_complete(site: Path) -> None:
    with zipfile.ZipFile(site / "scanner.zip") as archive:
        names = set(archive.namelist())
    assert {"site_scan.py", "quantpreflight/rules.py", "quantpreflight/hardware/gpus.toml", "quantpreflight/architectures/default.toml"} <= names
    assert not any("__pycache__" in name for name in names)


def test_source_files_stay_out_of_the_site(site: Path) -> None:
    assert not (site / "examples").exists()
    assert not (site / "site_scan.py").exists()
    assert {path.name for path in site.iterdir()} >= {"index.html", "style.css", "app.js", "favicon.svg", "scanner.zip", "social.png"}


def test_page_runs_only_scripts_it_allows(page: str) -> None:
    assert "script-src 'self' https://cdn.jsdelivr.net 'wasm-unsafe-eval'" in page
    assert "onclick" not in page


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{", "valid JSON near line 1"),
        ('{\n\n"a": }', "valid JSON near line 3"),
        ("", "valid JSON"),
        ("[]", "doesn't look like"),
        ('{"hello": 1}', "doesn't look like"),
        ("5", "doesn't look like"),
        ("[" * 200_000, "nested too deeply"),
    ],
    # The deep case would make an ID longer than Windows allows in an environment variable
    ids=["open-brace", "missing-value", "empty", "list", "no-model-keys", "number", "deep-nesting"],
)
def test_text_that_cannot_be_read_says_why(text: str, message: str) -> None:
    assert message in json.loads(scan_text(text))["error"]


def test_scan_text_reports_the_quantization() -> None:
    result = json.loads(scan_text('{"model_type": "llama", "quantization_config": {"quant_method": "awq", "bits": 4}}'))
    assert result["quant"] == "awq 4-bit gemm"
    assert result["findings"] == []


def test_scan_text_reads_a_plain_config() -> None:
    assert json.loads(scan_text('{"model_type": "llama"}'))["quant"] == "unquantized"


def test_scan_text_passes_the_target_on() -> None:
    config = '{"num_attention_heads": 8, "model_type": "llama"}'
    assert json.loads(scan_text(config, tp=2))["findings"] == []
    assert json.loads(scan_text(config, tp=3))["findings"][0]["rule_id"] == "QPF006"
    assert json.loads(scan_text(config, engine="llamacpp"))["findings"][0]["rule_id"] == "QPF011"


def test_scan_text_uses_the_weight_size_for_memory() -> None:
    config = '{"model_type": "llama"}'
    assert json.loads(scan_text(config, weights_gib=0))["vram_gib"] is None
    assert json.loads(scan_text(config, gpu="L4", weights_gib=1))["findings"] == []
    assert json.loads(scan_text(config, gpu="L4", weights_gib=40))["findings"][0]["rule_id"] == "QPF012"


def test_scan_text_puts_the_worst_finding_first() -> None:
    config = '{"model_type": "llama", "num_attention_heads": 8, "num_key_value_heads": 2, "num_local_experts": 3, "quantization_config": {"quant_method": "nvfp4"}}'
    severities = [finding["severity"] for finding in json.loads(scan_text(config, gpu="H100", tp=4))["findings"]]
    order = ["critical", "high", "medium", "low"]
    assert severities == sorted(severities, key=order.index)


# Models fetched by the page


def _repo_payload(**overrides: Any) -> str:
    header = json.dumps(
        {
            "model.embed_tokens.weight": {"dtype": "F16", "shape": [8, 8], "data_offsets": [0, 128]},
            "lm_head.weight": {"dtype": "F16", "shape": [8, 8], "data_offsets": [128, 256]},
        }
    )
    payload: dict[str, Any] = {
        "repo": "org/model",
        "files": {"config.json": '{"model_type": "llama", "vocab_size": 8}'},
        "shards": {"model.safetensors": header},
        "siblings": ["config.json", "model.safetensors"],
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_scan_repo_reads_tensor_headers() -> None:
    result = json.loads(scan_repo(_repo_payload()))
    assert result["findings"] == []
    assert result["tensors"] == 2
    assert result["quant"] == "unquantized (F16)"
    assert json.loads(result["config"])["model_type"] == "llama"


def test_scan_repo_runs_the_tensor_rules() -> None:
    files = {"config.json": '{"model_type": "llama", "vocab_size": 9}'}
    assert json.loads(scan_repo(_repo_payload(files=files)))["findings"][0]["rule_id"] == "QPF002"


def test_scan_repo_uses_an_external_quant_config() -> None:
    files = {"config.json": "{}", "hf_quant_config.json": '{"quant_algo": "NVFP4"}'}
    result = json.loads(scan_repo(_repo_payload(files=files), gpu="H100"))
    assert {finding["rule_id"] for finding in result["findings"]} >= {"QPF004"}


def test_scan_repo_reports_missing_shards_and_dangling_tensors() -> None:
    index = json.dumps({"weight_map": {"ghost": "model.safetensors", "x": "gone.safetensors"}, "metadata": {"total_size": 5 * 1024**3}})
    payload = _repo_payload(
        files={"config.json": "{}", "model.safetensors.index.json": index},
        shards={"model.safetensors": json.dumps({"a.weight": {"dtype": "F16", "shape": [8, 8], "data_offsets": [0, 128]}}), "gone.safetensors": None},
    )
    result = json.loads(scan_repo(payload))
    titles = {finding["title"] for finding in result["findings"]}
    assert "Missing Safetensors shard file(s) referenced by index.json" in titles
    assert "Tensors declared in weight_map are missing from shard headers" in titles
    assert result["weights_gib"] == 5.0


def test_scan_repo_finds_duplicates_across_shards() -> None:
    shard = json.dumps({"a.weight": {"dtype": "F16", "shape": [8, 8], "data_offsets": [0, 128]}})
    result = json.loads(scan_repo(_repo_payload(files={"config.json": "{}"}, shards={"a.safetensors": shard, "b.safetensors": shard})))
    assert any(finding["title"].startswith("Duplicate tensor") for finding in result["findings"])


def test_scan_repo_flags_pickle_and_custom_code() -> None:
    files = {"config.json": '{"auto_map": {"AutoModel": "m.X"}}'}
    payload = _repo_payload(files=files, siblings=["pytorch_model.bin", "training_args.bin", "modeling_m.py"])
    titles = [finding["title"] for finding in json.loads(scan_repo(payload))["findings"]]
    assert any("Redundant legacy pickle" in title for title in titles)
    assert any("custom Python" in title for title in titles)


def test_scan_repo_reads_added_tokens() -> None:
    files = {"config.json": "{}", "tokenizer_config.json": '{"added_tokens_decoder": {"9": {"content": "<x>"}}}'}
    assert json.loads(scan_repo(_repo_payload(files=files)))["findings"][0]["rule_id"] == "QPF002"


@pytest.mark.parametrize(
    "payload",
    [
        _repo_payload(files={"config.json": "{"}),
        _repo_payload(shards={"model.safetensors": "{"}),
        _repo_payload(files={"config.json": "{}", "model.safetensors.index.json": '{"weight_map": {"a": "../x.safetensors"}}'}),
    ],
)
def test_scan_repo_reports_unreadable_files(payload: str) -> None:
    assert "couldn't be read" in json.loads(scan_repo(payload))["error"]


def test_scan_repo_treats_a_null_json_file_as_absent() -> None:
    payload = _repo_payload(files={"config.json": "[]", "generation_config.json": None})
    assert json.loads(scan_repo(payload))["quant"] == "unquantized (F16)"


def test_page_may_only_reach_hugging_face_and_the_scanner_cdn(page: str) -> None:
    assert "connect-src 'self' https://cdn.jsdelivr.net https://huggingface.co https://*.huggingface.co https://*.hf.co;" in page
