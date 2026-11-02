"""Scan popular public checkpoints and save what quantpreflight says about each one.

This never downloads weights, but it does make many requests to Hugging Face, so it takes a while.
Run it with --search to pick models from Hugging Face again. Otherwise it rescans the saved list.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quantpreflight.cli import SCAN_ERRORS  # noqa: E402
from quantpreflight.loader import load_checkpoint_metadata  # noqa: E402
from quantpreflight.models import TargetSpec  # noqa: E402
from quantpreflight.rules import analyze_checkpoint  # noqa: E402

RESULTS = ROOT / "benchmark" / "public_models.json"
SEARCH_TERMS = [
    "awq", "gptq", "fp8", "nvfp4", "w4a16", "w8a8", "autoround", "bnb-4bit", "4bit", "gguf",
    "compressed-tensors", "int4", "int8", "marlin", "qwen", "llama", "mistral", "deepseek", "gemma", "phi",
]


def search(term: str) -> list[str]:
    """Return the most downloaded public, ungated models for a search term."""
    query = urllib.parse.urlencode({"search": term, "sort": "downloads", "direction": -1, "limit": 30})
    with urllib.request.urlopen(f"https://huggingface.co/api/models?{query}", timeout=30) as response:
        return [item["id"] for item in json.load(response) if not item.get("gated") and not item.get("private")]


def scan(repo: str) -> dict[str, Any]:
    low = repo.lower()
    engine = "llamacpp" if "gguf" in low else "vllm"
    gpu = "B200" if "nvfp4" in low or "fp4" in low else "H100"
    for attempt in range(4):
        try:
            meta = load_checkpoint_metadata(repo, timeout=20, prefer_gguf=engine == "llamacpp")
            # The scan looks for file and format problems, so memory is out of scope here
            report = analyze_checkpoint(meta, TargetSpec(engine=engine, gpu=gpu, vram_gb_override=100_000))
            break
        except SCAN_ERRORS as exc:
            if "429" in str(exc) and attempt < 3:
                time.sleep(60)
                continue
            return {"repo": repo, "status": "unreadable", "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
        except Exception:
            return {"repo": repo, "status": "crash", "error": traceback.format_exc()[-400:]}
    return {
        "repo": repo,
        "status": "scanned",
        "engine": engine,
        "gpu": gpu,
        "container": meta.container_format,
        "quant": report.quant_summary,
        "weights_gib": round(meta.total_weight_bytes / 1024**3, 2),
        "findings": [[f.rule_id, f.severity.value, f.title] for f in report.findings],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", action="store_true", help="Pick the models from Hugging Face again.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-list", type=Path, default=None, help="Take repo names from another results file.")
    args = parser.parse_args()

    if args.search:
        repos = list(dict.fromkeys(repo for term in SEARCH_TERMS for repo in search(term)))
    else:
        source = args.seed_list or RESULTS
        repos = [item["repo"] for item in json.loads(source.read_text(encoding="utf-8"))]

    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(scan, repos))

    RESULTS.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(collections.Counter(item["status"] for item in results))
    print(collections.Counter(item["error"][:70] for item in results if item["status"] == "unreadable").most_common(8))
    print(collections.Counter(finding[0] for item in results for finding in item.get("findings", [])))


if __name__ == "__main__":
    main()
