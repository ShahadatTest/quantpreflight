"""Finding the quantized versions of a model and ranking the ones that start, against an in-memory hub."""

from __future__ import annotations

import json
from typing import Any
import urllib.error

import pytest

from conftest import build_safetensors_header_file
from quantpreflight import loader, suggest as suggest_module
from quantpreflight.cli import main
from quantpreflight.models import Finding, SetupPlan, Severity, TargetSpec
from quantpreflight.suggest import Candidate, find_base_model, find_variants, rank, suggest

HF = "https://huggingface.co"
LLAMA = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "hidden_size": 128,
    "max_position_embeddings": 4096,
    "vocab_size": 8,
}
PLAIN = {"model.embed_tokens.weight": ("F16", [8, 128]), "lm_head.weight": ("F16", [8, 128])}
AWQ = {
    **PLAIN,
    "model.layers.0.mlp.down_proj.qweight": ("I32", [128, 16]),
    "model.layers.0.mlp.down_proj.qzeros": ("I32", [1, 16]),
    "model.layers.0.mlp.down_proj.scales": ("F16", [1, 128]),
}


class FakeHub:
    """Serves Hub API answers and repository files by URL."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.listing: list[dict[str, Any]] = []
        self.denied: set[str] = set()
        self.queries: list[str] = []

    def repo(self, repo: str, *, config: dict[str, Any] | None = None, tensors: dict[str, Any] | None = None,
             tags: tuple[str, ...] = (), downloads: int = 0) -> None:
        api = {"id": repo, "downloads": downloads, "tags": list(tags), "siblings": [{"rfilename": "config.json"}, {"rfilename": "model.safetensors"}]}
        self.files[f"{HF}/api/models/{repo}"] = json.dumps(api).encode()
        self.files[f"{HF}/{repo}/resolve/main/config.json"] = json.dumps(config or LLAMA).encode()
        self.files[f"{HF}/{repo}/resolve/main/model.safetensors"] = build_safetensors_header_file(tensors or PLAIN)

    def __call__(self, url: str, *, byte_range: tuple[int, int] | None = None, **_: Any) -> bytes:
        if url.startswith(f"{HF}/api/models?"):
            self.queries.append(url)
            return json.dumps(self.listing).encode()
        if any(f"/{repo}/" in url or url.endswith(f"/{repo}") for repo in self.denied):
            raise urllib.error.HTTPError(url, 401, "gated", None, None)  # type: ignore[arg-type]
        if url not in self.files:
            raise urllib.error.HTTPError(url, 404, "missing", None, None)  # type: ignore[arg-type]
        data = self.files[url]
        return data if byte_range is None else data[byte_range[0] : byte_range[1] + 1]


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    fake = FakeHub()
    monkeypatch.setattr(loader, "_http_fetch_bytes", fake)
    quantized = "base_model:quantized:org/base"
    fake.repo("org/base", downloads=500)
    fake.repo("org/base-AWQ", tensors=AWQ, tags=(quantized,), downloads=900,
              config={**LLAMA, "quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128, "version": "gemm", "zero_point": True}})
    fake.repo("org/base-bnb", tags=(quantized,), downloads=50,
              config={**LLAMA, "quantization_config": {"quant_method": "bitsandbytes", "load_in_4bit": True}})
    fake.repo("org/base-copy", tags=(quantized,), downloads=10)
    fake.listing = [
        {"id": "org/base-AWQ", "downloads": 900},
        {"id": "org/base-GGUF", "downloads": 800},
        {"id": "org/base-bnb", "downloads": 50},
        {"id": "org/base-copy", "downloads": 10},
        {"id": "gated/base-FP8", "downloads": 5},
        {"id": "org/base", "downloads": 500},
        {"downloads": 1},
    ]
    fake.denied.add("gated/base-FP8")
    return fake


def test_a_quantized_repo_leads_back_to_its_base(hub: FakeHub) -> None:
    assert find_base_model("org/base-AWQ") == ("org/base", 500)
    assert find_base_model("org/base") == ("org/base", 500)


def test_variants_are_listed_by_downloads(hub: FakeHub) -> None:
    variants = find_variants("org/base", limit=5)
    assert variants[0] == ("org/base-AWQ", 900)
    assert "filter=base_model%3Aquantized%3Aorg%2Fbase" in hub.queries[0]
    assert "limit=5" in hub.queries[0]


def test_versions_that_start_come_first_highest_precision_first(hub: FakeHub) -> None:
    base, found = suggest("org/base-AWQ", TargetSpec(gpu="H100"))
    assert base == "org/base"
    order = [(item.repo, item.plan.status if item.plan else item.error) for item in found]
    assert order[:3] == [("org/base", "ready"), ("org/base-AWQ", "ready"), ("org/base-bnb", "blocked")]
    skipped = dict(order[3:])
    assert skipped["org/base-copy"] == "not quantized in a format the tool knows"
    assert "gated" in skipped["gated/base-FP8"]
    # GGUF repos can't suit vLLM, so they aren't read at all
    assert "org/base-GGUF" not in skipped
    awq = next(item for item in found if item.repo == "org/base-AWQ")
    assert awq.bits == 4 and awq.quant == "awq 4-bit g128 gemm" and awq.starts


def test_llamacpp_only_looks_at_gguf_repos(hub: FakeHub) -> None:
    _, found = suggest("org/base", TargetSpec(engine="llamacpp", gpu="H100"))
    assert [item.repo for item in found] == ["org/base-GGUF"]
    assert found[0].error


def test_the_limit_counts_only_versions_that_suit_the_engine(hub: FakeHub) -> None:
    _, found = suggest("org/base", TargetSpec(gpu="H100"), limit=1)
    assert [item.repo for item in found] == ["org/base", "org/base-AWQ"]
    assert "limit=3" in hub.queries[0]


def test_ranking_breaks_ties_by_downloads() -> None:
    ready = SetupPlan(status="ready", tp=1)
    blocked = SetupPlan(status="blocked", tp=1, blocking=(Finding("QPF004", Severity.CRITICAL, "t", "m", "r"),))
    items = [
        Candidate("a", downloads=1, bits=4, plan=ready),
        Candidate("b", downloads=9, bits=4, plan=ready),
        Candidate("c", downloads=99, bits=16, plan=blocked),
        Candidate("d", error="gated"),
        Candidate("e", downloads=0, bits=8, plan=SetupPlan(status="changed", tp=2)),
    ]
    assert [item.repo for item in rank(items)] == ["e", "b", "a", "c", "d"]


def test_suggest_on_the_command_line(hub: FakeHub, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["suggest", "org/base-AWQ", "--gpu", "H100"]) == 0
    out = capsys.readouterr().out
    assert "Versions of org/base for vLLM on H100, tp=1:" in out
    assert "Best pick: org/base, the highest precision that starts." in out
    assert "  Run:  vllm serve org/base" in out
    assert "QPF011" in out


def test_suggest_as_json(hub: FakeHub, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["suggest", "org/base", "--gpu", "H100", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["base_model"] == "org/base"
    assert payload["candidates"][0]["setup"]["status"] == "ready"
    assert payload["target"]["gpu"] == "H100"


def test_suggest_exits_1_when_nothing_starts(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    blocked = SetupPlan(status="blocked", tp=1, blocking=(Finding("QPF012", Severity.CRITICAL, "t", "m", "r"),))
    monkeypatch.setattr("quantpreflight.cli.suggest", lambda *a, **k: ("org/base", [Candidate("org/base", plan=blocked)]))
    assert main(["suggest", "org/base", "--gpu", "T4"]) == 1
    assert "None of these start on this target." in capsys.readouterr().out


def test_suggest_reports_a_missing_model(hub: FakeHub, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["suggest", "org/nothing"]) == 2
    assert "Error finding versions of org/nothing" in capsys.readouterr().err


@pytest.mark.parametrize("args", [["--limit", "0"], ["--tp", "0"], ["--concurrency", "0"]])
def test_suggest_rejects_bad_options(args: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["suggest", "org/base", *args]) == 2
    assert "must be >= 1" in capsys.readouterr().err


def test_a_scan_that_fails_after_loading_is_skipped(hub: FakeHub, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: Any, **kwargs: Any) -> None:
        raise ValueError("bad profile")

    monkeypatch.setattr(suggest_module, "plan_setup", broken)
    _, found = suggest("org/base", TargetSpec(gpu="H100"))
    assert all(item.plan is None for item in found)
    assert next(item for item in found if item.repo == "org/base").error == "bad profile"
