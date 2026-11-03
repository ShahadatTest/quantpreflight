"""Build the project website into _site."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "website"
PACKAGE = ROOT / "src" / "quantpreflight"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools"), str(SOURCE)]

from quantpreflight import __version__  # noqa: E402
from quantpreflight.profiles import load_engine_profiles, load_gpu_profiles  # noqa: E402
from quantpreflight.rules import RULE_CATALOG  # noqa: E402
from run_benchmark import Score, load_cases, run  # noqa: E402
from site_scan import scan_text  # noqa: E402

INSTALL = "pip install quantpreflight"
ACTION_REF = f"v{__version__}"

FORMATS = ["AWQ", "GPTQ", "Compressed-Tensors", "FP8", "NVFP4", "MXFP4", "Marlin", "AutoRound", "bitsandbytes", "BitNet", "MLX", "GGUF", "Safetensors"]

EXAMPLES = [
    ("fits", "Fits", {"engine": "vllm", "gpu": "H100", "tp": 4, "weights_gib": 38.74}),
    ("too-big", "Too big", {"engine": "vllm", "gpu": "L4", "tp": 1, "weights_gib": 38.74}),
    ("wrong-gpu", "Wrong GPU", {"engine": "vllm", "gpu": "H100", "tp": 1, "weights_gib": 5.2}),
    ("bad-split", "Bad split", {"engine": "vllm", "gpu": "A100", "tp": 3, "weights_gib": 12.5}),
]

QUESTIONS = {
    "QPF001": ("Were sensitive layers quantized too hard?", "Norms and vision towers lose too much at low precision, and a quantized router can hurt quality silently."),
    "QPF002": ("Can a token fall off the end of the embedding table?", "A token ID past the last row ends in a GPU assert. This is the checkpoint-side relative of CVE-2026-115916."),
    "QPF003": ("Will the model know when to stop?", "If the stop token differs between files, it can generate until max_tokens."),
    "QPF004": ("Does your GPU have the kernels for this format?", "NVFP4 needs Blackwell, ModelOpt needs sm_80, block FP8 is limited on sm_89, and FP8 may fail on GPUs without FP8 cores."),
    "QPF005": ("Does this mix trigger a known silent bug?", "Some engine, GPU, and format mixes corrupt output or ignore your flags with no error."),
    "QPF006": ("Can the heads and experts split across your GPUs?", "Attention heads, KV heads, and MoE experts must divide evenly by --tp."),
    "QPF007": ("Do the shards still fit the kernel after splitting?", "Per-GPU sizes must stay multiples of group_size, the 128x64 Marlin tile, and the FP8 block."),
    "QPF008": ("Does this GPTQ model use desc_act?", "vLLM rejects it outright, and other engines pay for extra gathers across GPUs."),
    "QPF009": ("Is every shard really there?", "Missing shards, dangling tensors, duplicates, and wrong MTP draft shapes fail at load."),
    "QPF010": ("Does the config match the weights?", "Packed tensors with no quantization_config, or the reverse, load wrongly or waste memory."),
    "QPF011": ("Can this engine read this format?", "Safetensors on llama.cpp and GGUF on TensorRT-LLM fail to load, and 1-bit or 2-bit GGUF on vLLM or SGLang is often unsupported or slow."),
    "QPF012": ("Will it fit in GPU memory?", "Weights, runtime overhead measured on real GPUs, and the KV cache, with DeepSeek MLA, sliding windows, and Mamba state counted properly."),
    "QPF013": ("Are the byte offsets in the file sane?", "Overlapping, inverted, or truncated spans mean a corrupt or incomplete file."),
    "QPF014": ("Will loading it run someone else's code?", "Pickle weights and trust_remote_code can execute arbitrary code on your host."),
    "QPF015": ("Does your engine version have known vulnerabilities?", "With --engine-version, your vLLM or SGLang version is checked against published advisories."),
    "QPF016": ("Is the context longer than the model allows?", "vLLM refuses a --context-len above max_position_embeddings."),
}

BENCHMARK_GROUPS = {
    "safe": "Safe setups",
    "sensitive": "Sensitive layers",
    "tokens": "Tokens and stop IDs",
    "hardware": "GPU support",
    "kernels": "Silent kernel bugs",
    "parallel": "Splitting across GPUs",
    "files": "Broken files",
    "engines": "Engine and format",
    "memory": "GPU memory",
    "gaps": "Known gaps",
}

SUPPORT = [
    ("Ethereum (ERC20), BNB Chain (BEP20), Base, Arbitrum, Optimism, Polygon", "0x7C524d172F2bf750e3f7b4Df0339B3fFb09F9ADb"),
    ("Bitcoin", "bc1qcvdafsre6tlfpsdnv2s4ltwvy9tlgkjj6ag4gu"),
    ("Solana", "Fv9k74qakg1L7ehMaJ3SBXxVbTKg2WuM5AjW2KLM3nVH"),
    ("Tron (TRC20)", "TJVfPm5wJnKjD5ZzBRhMXtzB1aF17Kbyuz"),
]


def _rules() -> str:
    rows = []
    for rule_id, (severity, _, _) in sorted(RULE_CATALOG.items()):
        question, why = QUESTIONS[rule_id]
        rows.append(
            f'<tr id="{rule_id}"><th scope="row"><code>{rule_id}</code></th>'
            f'<td><span class="tag {severity.value}">{severity.value.title()}</span></td>'
            f"<td><strong>{escape(question)}</strong><span>{escape(why)}</span></td></tr>"
        )
    return "\n".join(rows)


def _items(names: list[str]) -> str:
    return "\n".join(f"<li>{escape(name)}</li>" for name in names)


def _gpus() -> list[dict[str, str]]:
    gpus = sorted(load_gpu_profiles().values(), key=lambda gpu: (gpu.vendor != "nvidia", gpu.compute_capability, gpu.id))
    return [{"id": gpu.id, "label": f"{gpu.id} ({gpu.vram_gb:g} GiB)"} for gpu in gpus]


def _engines() -> list[dict[str, str]]:
    return [{"id": engine.id, "name": engine.name} for engine in load_engine_profiles().values()]


def _support() -> str:
    return "\n".join(
        f'<div class="wallet"><p>{escape(network)}</p>'
        f'<div><code>{address}</code><button type="button" class="copy" data-copy="{address}">Copy</button></div></div>'
        for network, address in SUPPORT
    )


def _benchmark() -> dict[str, str]:
    cases = load_cases()
    total = Score()
    rows = []
    for group, score in run(cases).items():
        total.add(score)
        rows.append(
            f'<tr><th scope="row">{escape(BENCHMARK_GROUPS[group])}</th><td>{score.cases}</td>'
            f"<td>{score.passed}</td><td>{len(score.missed)}</td><td>{len(score.false_alarms)}</td></tr>"
        )
    return {
        "benchmark": "\n".join(rows),
        "case_count": str(sum(len(items) for items in cases.values())),
        "recall": f"{total.recall:.1%}",
        "precision": f"{total.precision:.0%}",
    }


def _evidence() -> dict[str, str]:
    public = json.loads((ROOT / "benchmark" / "public_models.json").read_text(encoding="utf-8"))
    scanned = [item for item in public if item["status"] == "scanned"]
    blocking = [item for item in scanned if any(finding[1] in ("critical", "high") for finding in item["findings"])]
    runs = json.loads((ROOT / "benchmark" / "gpu_check_t4.json").read_text(encoding="utf-8"))["results"]
    right = [run for run in runs if (run["prediction"].get("verdict") == "go") == bool(run["actual"]["loaded"])]
    return {
        "public_count": str(len(scanned)),
        "public_clean": str(sum(not item["findings"] for item in scanned)),
        "public_blocking": str(len(blocking)),
        "gpu_total": str(len(runs)),
        "gpu_match": str(len(right)),
    }


def _data() -> str:
    examples = []
    for slug, label, target in EXAMPLES:
        text = (SOURCE / "examples" / f"{slug}.json").read_text(encoding="utf-8")
        result = scan_text(text, target["engine"], target["gpu"], target["tp"], target["weights_gib"])
        examples.append({"slug": slug, "label": label, "file": "config.json", "text": text, "target": target, "result": json.loads(result)})
    data = {"examples": examples, "gpus": _gpus(), "engines": _engines()}
    return json.dumps(data).replace("</", "<\\/")


def _write_scanner(path: Path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(SOURCE / "site_scan.py", "site_scan.py")
        for file in sorted(PACKAGE.rglob("*")):
            if file.suffix in {".py", ".toml"}:
                archive.write(file, file.relative_to(PACKAGE.parent).as_posix())


def build(out: Path) -> None:
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(SOURCE, out, ignore=shutil.ignore_patterns("__pycache__", "examples", "site_scan.py"))
    page = (SOURCE / "index.html").read_text(encoding="utf-8")
    for token, value in {
        "version": __version__,
        "install": INSTALL,
        "ref": ACTION_REF,
        "rules": _rules(),
        "formats": _items(FORMATS),
        "engines": _items([engine["name"] for engine in _engines()]),
        "gpus": _items([gpu["id"] for gpu in _gpus()]),
        "support": _support(),
        "data": _data(),
        **_benchmark(),
        **_evidence(),
    }.items():
        page = page.replace(f"<!-- {token} -->", value)
    (out / "index.html").write_text(page, encoding="utf-8")
    _write_scanner(out / "scanner.zip")


if __name__ == "__main__":
    build(ROOT / "_site")
