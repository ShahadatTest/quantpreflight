"""Check the setups quantpreflight suggests against real vLLM starts on the GPUs of this machine.

For each case it asks `quantpreflight check --gpu auto` whether the setup starts as asked, then starts
vLLM with it and compares. The contexts were picked so that some fit on a Tesla T4 and some don't. Like tools/gpu_check.py, it downloads real weights into its own
cache folder and empties it between models. See docs/gpu-testing.md.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("QPF_GPU_CHECK_CACHE", Path.home() / ".cache" / "quantpreflight-gpu-check"))
TIMEOUT_SECONDS = 25 * 60

CHILD = """
import json, sys
from vllm import LLM, SamplingParams
kwargs = dict(model=sys.argv[1], tensor_parallel_size=int(sys.argv[2]), gpu_memory_utilization=0.9)
if sys.argv[3] != "default":
    kwargs["max_model_len"] = int(sys.argv[3])
llm = LLM(**kwargs)
out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0))
print("QPF_OUTPUT " + json.dumps(out[0].outputs[0].text), flush=True)
"""

# (repo, GPUs, max_model_len or "default" for the model's own limit)
CASES: list[tuple[str, int, str]] = [
    ("Qwen/Qwen2.5-14B-Instruct-AWQ", 1, "default"),
    ("Qwen/Qwen2.5-14B-Instruct-AWQ", 1, "5120"),
    ("Qwen/Qwen2.5-14B-Instruct-AWQ", 1, "10240"),
    ("hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", 1, "default"),
    ("hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", 1, "46080"),
    ("Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4", 1, "default"),
    ("Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4", 1, "5120"),
    ("Qwen/Qwen3-8B-AWQ", 1, "default"),
    ("Qwen/Qwen3-8B-AWQ", 1, "37888"),
    ("Qwen/Qwen3-14B-AWQ", 1, "default"),
    ("Qwen/Qwen3-14B-AWQ", 1, "6144"),
    ("casperhansen/mistral-nemo-instruct-2407-awq", 1, "default"),
    ("casperhansen/mistral-nemo-instruct-2407-awq", 1, "17408"),
    ("Qwen/Qwen2.5-7B-Instruct-AWQ", 1, "default"),
    ("Qwen/Qwen2.5-Coder-7B-Instruct-AWQ", 1, "default"),
    ("Qwen/Qwen3-4B-AWQ", 1, "default"),
    ("TheBloke/Mistral-7B-Instruct-v0.2-AWQ", 1, "default"),
    ("solidrust/Mistral-7B-Instruct-v0.3-AWQ", 1, "default"),
    ("Qwen/Qwen2.5-7B-Instruct", 1, "default"),
    ("Qwen/Qwen2.5-7B-Instruct", 2, "default"),
]
REAL_LIMIT = re.compile(r"estimated maximum model length is (\d+)")
KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")


def predict(repo: str, tp: int, context: str, gpu: str) -> dict[str, Any]:
    """Return the setup `quantpreflight check` gives for one case."""
    command = [sys.executable, "-m", "quantpreflight", "check", repo, "--gpu", gpu, "--tp", str(tp), "--format", "json"]
    if context != "default":
        command += ["--context-len", context]
    # Use this checkout's code when there is one, like tools/gpu_check.py does
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")} if (ROOT / "src").is_dir() else None
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    return json.loads(result.stdout)["setup"]


def start(repo: str, tp: int, context: str) -> dict[str, Any]:
    """Start vLLM in a new process and generate a few tokens."""
    env = {**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_CACHE": str(CACHE)}
    command = [sys.executable, "-c", CHILD, repo, str(tp), context]
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, env=env)
        log = process.stdout + process.stderr
    except subprocess.TimeoutExpired as exc:
        log = str(exc.stdout or "") + str(exc.stderr or "") + "\nTIMED OUT"
    output = re.search(r"QPF_OUTPUT (.*)", log)
    limit = REAL_LIMIT.search(log)
    tokens = KV_TOKENS.search(log)
    errors = [line for line in log.splitlines() if re.search(r"error|exception|not enough|out of memory", line, re.I)]
    return {
        "started": output is not None,
        "vllm_max_model_len": int(limit.group(1)) if limit else None,
        "kv_cache_tokens": int(tokens.group(1).replace(",", "")) if tokens else None,
        "error": None if output else errors[-4:],
    }


def run(out: Path, only: list[str], gpu: str, dry_run: bool) -> None:
    results: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"gpu": gpu}
    if not dry_run:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
        meta |= {"gpus": smi.stdout.splitlines(), "vllm": metadata.version("vllm")}
    previous = None
    for repo, tp, context in CASES:
        if only and not any(part in repo for part in only):
            continue
        if repo != previous:
            shutil.rmtree(CACHE, ignore_errors=True)
            previous = repo
        setup = predict(repo, tp, context, gpu)
        # Ready means it starts exactly as asked, with this GPU count and context
        expected = setup["status"] == "ready"
        actual = {"started": None} if dry_run else start(repo, tp, context)
        entry = {"repo": repo, "tp": tp, "max_model_len": context, "setup": setup, "expected_start": expected, **actual}
        entry["agrees"] = actual["started"] == expected
        results.append(entry)
        print(f"=== {repo} tp={tp} max_model_len={context} expected={expected} {actual}", flush=True)
        out.write_text(json.dumps({"meta": meta, "results": results}, indent=1), encoding="utf-8")
    shutil.rmtree(CACHE, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("setup_check_results.json"))
    parser.add_argument("--only", nargs="*", default=[], help="Only run repos containing one of these words.")
    parser.add_argument("--dry-run", action="store_true", help="Print the predictions only. Needs no GPU and no vLLM.")
    parser.add_argument("--gpu", default="auto", help="GPU profile to predict for, auto by default.")
    args = parser.parse_args()
    run(args.out, args.only, args.gpu, args.dry_run)


if __name__ == "__main__":
    main()
