"""Find the quantized versions of a model on Hugging Face and rank the ones that start on your GPUs."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
import json
from typing import Any
import urllib.parse

from . import loader
from .errors import SCAN_ERRORS
from .models import SetupPlan, TargetSpec
from .planner import plan_setup
from .rules import format_quant_summary

HUB_API = "https://huggingface.co/api/models"
QUANTIZED_TAG = "base_model:quantized:"
# Ready and changed setups both start, so they rank together
STATUS_ORDER = {"ready": 0, "changed": 0, "blocked": 1}


@dataclass
class Candidate:
    """One version of the model and what the scan made of it."""

    repo: str
    downloads: int = 0
    quant: str = ""
    bits: float = 16.0
    weights_gib: float | None = None
    plan: SetupPlan | None = None
    error: str | None = None

    @property
    def starts(self) -> bool:
        return self.plan is not None and self.plan.status != "blocked"

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "downloads": self.downloads,
            "quant": self.quant,
            "bits": self.bits,
            "weights_gib": self.weights_gib,
            "setup": self.plan.to_dict() if self.plan else None,
            "error": self.error,
        }


def _hub_json(url: str, token: str | None) -> Any:
    return json.loads(loader._http_fetch_bytes(url, token=loader._resolve_hf_token(token)).decode("utf-8"))


def find_base_model(repo: str, token: str | None = None) -> tuple[str, int]:
    """Return the model a quantized repo was made from, or the repo itself, with its download count."""
    info = _hub_json(f"{HUB_API}/{repo}", token)
    for tag in info.get("tags") or []:
        if isinstance(tag, str) and tag.startswith(QUANTIZED_TAG):
            base = tag[len(QUANTIZED_TAG):]
            base_info = _hub_json(f"{HUB_API}/{base}", token)
            return base, int(base_info.get("downloads") or 0)
    return repo, int(info.get("downloads") or 0)


def find_variants(base: str, token: str | None = None, limit: int = 15) -> list[tuple[str, int]]:
    """Return the most downloaded quantized versions of `base`, with their download counts."""
    query = urllib.parse.urlencode(
        {"filter": f"{QUANTIZED_TAG}{base}", "sort": "downloads", "direction": -1, "limit": limit}
    )
    items = _hub_json(f"{HUB_API}?{query}", token)
    return [(str(item["id"]), int(item.get("downloads") or 0)) for item in items if isinstance(item, dict) and item.get("id")]


def _scan(
    repo: str,
    base: str,
    downloads: int,
    target: TargetSpec,
    token: str | None,
    ignored: set[str] | None,
    gpu_count: int | None,
) -> Candidate:
    candidate = Candidate(repo=repo, downloads=downloads)
    try:
        meta = loader.load_checkpoint_metadata(repo, token=token, prefer_gguf=target.engine == "llamacpp")
    except SCAN_ERRORS as exc:
        candidate.error = str(exc).split("\n")[0][:160]
        return candidate
    if (meta.container_format == "gguf") != (target.engine == "llamacpp"):
        candidate.error = "GGUF, which only llama.cpp serves well" if meta.container_format == "gguf" else "not GGUF"
        return candidate
    if repo != base and meta.quant.method == "none":
        candidate.error = "not quantized in a format the tool knows"
        return candidate
    candidate.quant = format_quant_summary(meta)
    candidate.bits = meta.quant.bits or 16.0
    candidate.weights_gib = round(meta.total_weight_bytes / 1024**3, 2)
    try:
        candidate.plan = plan_setup(meta, target, ignored_rules=ignored, gpu_count=gpu_count)
    except SCAN_ERRORS as exc:
        candidate.error = str(exc).split("\n")[0][:160]
    return candidate


def rank(candidates: list[Candidate]) -> list[Candidate]:
    """Put the versions that start first, highest precision first, then the most downloaded."""

    def key(item: Candidate) -> tuple[int, float, int]:
        order = STATUS_ORDER[item.plan.status] if item.plan else 2
        return order, -item.bits, -item.downloads

    return sorted(candidates, key=key)


def suggest(
    repo: str,
    target: TargetSpec,
    *,
    token: str | None = None,
    limit: int = 15,
    workers: int = 4,
    ignored_rules: set[str] | None = None,
    gpu_count: int | None = None,
) -> tuple[str, list[Candidate]]:
    """Scan the model behind `repo` and its quantized versions against `target`."""
    base, base_downloads = find_base_model(repo, token)
    gguf = target.engine == "llamacpp"
    # The name says GGUF often enough to skip the repos that can't suit this engine without reading them
    found = [item for item in find_variants(base, token, limit * 3) if item[0] != base and ("gguf" in item[0].lower()) == gguf]
    variants = ([] if gguf else [(base, base_downloads)]) + found[:limit]
    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
        scanned = list(pool.map(lambda item: _scan(item[0], base, item[1], target, token, ignored_rules, gpu_count), variants))
    return base, rank(scanned)
