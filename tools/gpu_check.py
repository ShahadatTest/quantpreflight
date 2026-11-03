"""Compare quantpreflight's predictions with real vLLM loads on the GPUs of this machine.

Each case is scanned first, then loaded in its own process. Results go to a JSON file after every
case, so a stopped run keeps what it finished. It downloads real weights into its own cache folder,
which it empties after each case, so your Hugging Face cache is left alone. See docs/gpu-testing.md.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if (ROOT / "src").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

from quantpreflight import __version__  # noqa: E402
from quantpreflight.cli import SCAN_ERRORS  # noqa: E402
from quantpreflight.loader import load_checkpoint_metadata  # noqa: E402
from quantpreflight.models import TargetSpec  # noqa: E402
from quantpreflight.detect import match_gpu_name  # noqa: E402
from quantpreflight.rules import analyze_checkpoint  # noqa: E402

CONTEXT = 4096
TIMEOUT_SECONDS = 25 * 60
SECURITY_RULES = {"QPF014", "QPF015"}
CACHE = Path(os.environ.get("QPF_GPU_CHECK_CACHE", Path.home() / ".cache" / "quantpreflight-gpu-check"))

# (repo, GPUs per replica, what the case tests)
CASES: list[tuple[str, int, str]] = [
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 1, "small plain model"),
    ("Qwen/Qwen2.5-1.5B-Instruct", 1, "small plain model"),
    ("Qwen/Qwen2.5-Coder-7B-Instruct-AWQ", 1, "AWQ"),
    ("casperhansen/llama-3-8b-instruct-awq", 1, "AWQ from a third party"),
    ("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8", 1, "8-bit GPTQ"),
    ("TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ", 1, "GPTQ from a third party"),
    ("RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8", 1, "Compressed-Tensors W8A8"),
    ("RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic", 1, "Compressed-Tensors FP8"),
    ("NousResearch/Hermes-3-Llama-3.1-8B", 2, "plain 15 GiB model across two GPUs"),
    ("Qwen/Qwen2.5-VL-3B-Instruct", 1, "vision model"),
    ("Qwen/Qwen2.5-0.5B-Instruct", 1, "small plain model"),
    ("Qwen/Qwen2.5-7B-Instruct-AWQ", 1, "AWQ"),
    ("Qwen/Qwen2.5-7B-Instruct-AWQ", 2, "AWQ across two GPUs"),
    ("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", 1, "GPTQ"),
    ("RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16", 1, "Compressed-Tensors W4A16"),
    ("unsloth/Llama-3.2-1B-Instruct-bnb-4bit", 1, "bitsandbytes"),
    ("Qwen/Qwen3-8B-FP8", 1, "FP8"),
    ("nvidia/Llama-3.1-8B-Instruct-FP8", 1, "ModelOpt FP8"),
    ("nvidia/Qwen3-8B-FP4", 1, "NVFP4"),
    ("Qwen/Qwen2.5-14B-Instruct-AWQ", 1, "close to the memory limit of a 16 GB GPU"),
    ("Qwen/Qwen2.5-32B-Instruct-AWQ", 1, "too big for one 16 GB GPU"),
    ("Qwen/Qwen2.5-32B-Instruct-AWQ", 2, "the same model across two GPUs"),
    ("HuggingFaceTB/SmolLM2-135M-Instruct", 1, "small plain model"),
    ("HuggingFaceTB/SmolLM2-135M-Instruct", 2, "heads that may not split across two GPUs"),
    ("facebook/opt-125m", 1, "pickle weights only"),
    ("Zerul/Qwen3.5-0.8B-W4A16-AutoRound-AWQ", 1, "Qwen3.5 family"),
    ("RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16", 2, "MoE router quantized to 4-bit"),
]

ENV_CHECK = """
import torch, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
print("gpu math", (torch.ones(4).cuda() * 2).sum().item())
print("vllm", vllm.__version__)
"""

CHILD = """
import json, sys
from vllm import LLM, SamplingParams
llm = LLM(model=sys.argv[1], tensor_parallel_size=int(sys.argv[2]), max_model_len=int(sys.argv[3]), gpu_memory_utilization=0.9)
outputs = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0))
print("QPF_OUTPUT " + json.dumps(outputs[0].outputs[0].text), flush=True)
"""

LOG_PATTERNS = {
    "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
    "max_concurrency": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
    "kv_available_gib": r"Available KV cache memory: ([\d.]+) ?GiB",
    "weights_gib": r"model weights take ([\d.]+) ?GiB",
}
GPU_QUERY = """
import json, torch
gpus = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    # CUDA builds also have gcnArchName, so only trust it on ROCm
    arch = p.gcnArchName.split(":")[0] if torch.version.hip else f"sm_{p.major}{p.minor}"
    gpus.append({"name": p.name, "memory_gib": round(p.total_memory / 1024**3, 2), "arch": arch})
print(json.dumps(gpus))
"""


def gpu_table() -> list[dict[str, Any]]:
    """Return each GPU's name, memory, and architecture as PyTorch sees them."""
    text = subprocess.run([sys.executable, "-c", GPU_QUERY], capture_output=True, text=True, check=True).stdout
    return json.loads(text.strip().splitlines()[-1])


def profile_for(name: str, arch: str) -> str | None:
    """Match a device name such as `NVIDIA L40S` to a GPU profile, falling back to its architecture."""
    return match_gpu_name(name, arch)


def predict(repo: str, tp: int, gpu: str | None, vram_gib: float) -> dict[str, Any]:
    """Run the scan that would happen before any download."""
    for attempt in range(4):
        try:
            meta = load_checkpoint_metadata(repo)
            limit = meta.config.get("max_position_embeddings")
            context = min(limit, CONTEXT) if isinstance(limit, int) and limit > 0 else CONTEXT
            target = TargetSpec(gpu=gpu, tp=tp, context_len=context, vram_gb_override=vram_gib)
            report = analyze_checkpoint(meta, target)
            break
        except SCAN_ERRORS as exc:
            if attempt == 3:
                return {"scan_error": str(exc)[:300]}
            time.sleep(15)
    # Security rules say a model is risky, not that it won't load
    blocking = [f.rule_id for f in report.findings if f.severity.rank >= 3 and f.rule_id not in SECURITY_RULES]
    return {
        "verdict": "no-go" if blocking else "go",
        "context": context,
        "findings": [[f.rule_id, f.severity.value, f.title] for f in report.findings],
        "quant": report.quant_summary,
        "weights_gib": round(meta.total_weight_bytes / 1024**3, 2),
        "estimated_vram_gib": report.estimated_vram_per_gpu_gb,
    }


class MemoryWatcher(threading.Thread):
    """Record the highest memory use of any GPU while a load runs."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peak_mib = 0.0
        self.stopping = threading.Event()

    def run(self) -> None:
        query = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        if shutil.which(query[0]) is None:
            return
        while not self.stopping.wait(2):
            result = subprocess.run(query, capture_output=True, text=True)
            values = [float(row) for row in result.stdout.split() if row.strip()]
            if values:
                self.peak_mib = max(self.peak_mib, *values)


def load(repo: str, tp: int, context: int) -> dict[str, Any]:
    """Load the model with vLLM in a new process and generate a few tokens."""
    env = {**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_CACHE": str(CACHE)}
    watcher = MemoryWatcher()
    watcher.start()
    started = time.time()
    command = [sys.executable, "-c", CHILD, repo, str(tp), str(context)]
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, env=env)
        code, log = process.returncode, process.stdout + process.stderr
    except subprocess.TimeoutExpired as exc:
        code, log = -1, str(exc.stdout or "") + str(exc.stderr or "") + "\nTIMED OUT"
    watcher.stopping.set()
    watcher.join()

    output = re.search(r"QPF_OUTPUT (.*)", log)
    result: dict[str, Any] = {
        "loaded": output is not None,
        "exit_code": code,
        "seconds": round(time.time() - started),
        "peak_gib": round(watcher.peak_mib / 1024, 2),
        "text": json.loads(output.group(1)) if output else None,
    }
    for key, pattern in LOG_PATTERNS.items():
        match = re.search(pattern, log)
        result[key] = float(match.group(1).replace(",", "")) if match else None
    if output is None:
        lines = [line for line in log.splitlines() if line.strip()]
        errors = [line for line in lines if re.search(r"error|exception|not supported|out of memory|assert", line, re.I)]
        result["error"] = (errors or lines)[-6:]
    result["log_tail"] = log[-1500:]
    return result


def run(out: Path, only: list[str], dry_run: bool, gpu: str | None, vram: float) -> None:
    results: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"quantpreflight": __version__, "context": CONTEXT}
    if not dry_run:
        check = subprocess.run([sys.executable, "-c", ENV_CHECK], capture_output=True, text=True)
        print(check.stdout, flush=True)
        if check.returncode != 0:
            raise SystemExit(f"This machine can't run the checks:\n{check.stderr[-2000:]}")
        gpus = gpu_table()
        gpu = profile_for(gpus[0]["name"], gpus[0]["arch"])
        vram = gpus[0]["memory_gib"]
        import vllm  # type: ignore[import-not-found]

        meta |= {"gpus": gpus, "profile": gpu, "vllm": vllm.__version__}
    print(meta, flush=True)

    for repo, tp, note in CASES:
        if only and not any(part in repo for part in only):
            continue
        if not dry_run and tp > len(meta["gpus"]):
            continue
        entry: dict[str, Any] = {"repo": repo, "tp": tp, "note": note, "prediction": predict(repo, tp, gpu, vram)}
        print(f"\n=== {repo} tp={tp}: {note}\nPREDICTED {entry['prediction']}", flush=True)
        if not dry_run:
            entry["actual"] = load(repo, tp, entry["prediction"].get("context", CONTEXT))
            actual = entry["actual"]
            print(f"ACTUAL loaded={actual['loaded']} peak={actual['peak_gib']} GiB text={actual['text']!r} error={actual.get('error')}", flush=True)
            shutil.rmtree(CACHE, ignore_errors=True)
        results.append(entry)
        out.write_text(json.dumps({"meta": meta, "results": results}, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("gpu_check_results.json"))
    parser.add_argument("--only", nargs="*", default=[], help="Only run repos containing one of these words.")
    parser.add_argument("--dry-run", action="store_true", help="Scan only. Needs no GPU and no vLLM.")
    parser.add_argument("--gpu", default="T4", help="GPU profile for --dry-run.")
    parser.add_argument("--vram", type=float, default=15.0, help="VRAM per GPU for --dry-run.")
    args = parser.parse_args()
    run(args.out, args.only, args.dry_run, args.gpu, args.vram)


if __name__ == "__main__":
    main()
