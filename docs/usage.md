# Usage

## Scan a model

```bash
quantpreflight check <target> [options]
```

`<target>` can be a Hugging Face repo ID, a direct URL, a local folder, or a single `.safetensors` or `.gguf` file. You can pass several targets.

With `--engine llamacpp`, a repository that has both Safetensors and GGUF files is scanned as GGUF. Every part of a split GGUF file is read, and a missing part is reported.

## Options

| Option | What it does | Default |
| :--- | :--- | :--- |
| `--engine` | `vllm`, `sglang`, `trtllm`, or `llamacpp` | `vllm` |
| `--gpu` | Target GPU, alias, or architecture | None |
| `--tp` | GPUs per replica | `1` |
| `--engine-version` | Installed engine version, checked against advisories | None |
| `--context-len` | Context length for the cache estimate | Model limit, up to 4096 |
| `--concurrency` | Requests served at once, for the cache estimate | `1` |
| `--vram-gb` | Override VRAM per GPU, in GiB | From the GPU table |
| `--gpu-memory-utilization` | Share of VRAM the engine may use | `0.90` |
| `--marlin-input-dtype` | Simulate `VLLM_MARLIN_INPUT_DTYPE` | From the environment |
| `--revision` | Hugging Face branch, tag, or commit | `main` |
| `--token` | Hugging Face token | `HF_TOKEN` |
| `--ignore` | Rule IDs to skip, separated by commas | None |
| `--config` | Load a settings file | `.quantpreflight.toml` if present |
| `--fail-on` | Lowest severity that fails the run, or `none` | `high` |
| `--format` | `text`, `json`, or `sarif` | `text` |
| `--output` | Write the report to a file | Print it |

Exit codes: `0` means nothing reached the `--fail-on` level, `1` means something did, and `2` means the scan itself failed, for example on a corrupt file or a missing repository.

## Other commands

| Command | What it does |
| :--- | :--- |
| `quantpreflight explain QPF006` | Explain a rule |
| `quantpreflight list-rules` | List every rule |
| `quantpreflight list-gpus` | List GPUs and their capabilities |
| `quantpreflight validate-profiles` | Check the bundled TOML profiles |

You can pass a GPU alias or architecture such as `rtx4090` or `sm_89`.

## Severity

| Severity | What it means | What to do |
| :--- | :--- | :--- |
| **Critical** | It will crash, corrupt output, or expose your host. | Fix before you deploy. |
| **High** | Serious, and it may not start. | Fix soon. |
| **Medium** | It works, with a real cost. | Plan a fix. |
| **Low** | Good hygiene. | Fix when convenient. |

## Check your engine version

Pass the version you run, and it's compared with published vLLM and SGLang advisories from [OSV](https://osv.dev). They're bundled with the tool, so this needs no extra network access, and a weekly job refreshes them:

```bash
quantpreflight check Qwen/Qwen2.5-7B-Instruct-AWQ --engine-version 0.31.0
```

```text
[1] HIGH QPF015: vLLM 0.31.0 has 10 published advisories
    Fix:      Upgrade vLLM to 0.32.0 or newer.
```

`QPF002` catches token IDs the checkpoint itself carries. CVE-2026-115916 is an off-by-one in vLLM that a crafted request can trigger, so only `--engine-version` can flag that one.

## Estimate memory for real traffic

The estimate covers the weights, runtime overhead, and the KV cache for one request. Pass your own numbers to see the cost of real traffic:

```bash
quantpreflight check Qwen/Qwen2.5-72B-Instruct-AWQ --gpu H100 --tp 4 --context-len 8192 --concurrency 32
```

## Ignore a finding you've reviewed

Add `.quantpreflight.toml` to your repository root. It can also set your usual GPU:

```toml
gpu = "H100"
ignore = ["QPF008"]
```

Or pass `--ignore QPF003,QPF012` on the command line.

## Use it in CI

Add this as `.github/workflows/model-preflight.yml`:

```yaml
name: Model preflight
on: [push, pull_request]

permissions:
  contents: read
  security-events: write

jobs:
  preflight:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: layer8labs8/quantpreflight@main
        with:
          target: ./exported-checkpoint
          engine: vllm
          engine-version: "0.33.0"
          gpu: H100
          tp: "4"
          fail-on: high
          sarif-file: quantpreflight.sarif
      - uses: github/codeql-action/upload-sarif@v4
        if: always()
        with:
          sarif_file: quantpreflight.sarif
```

The upload step adds findings to the Security tab, which on private repositories can require GitHub Advanced Security. Without it, remove that step and `security-events: write`.

To run it as a [pre-commit](https://pre-commit.com) hook instead:

```yaml
repos:
  - repo: https://github.com/layer8labs8/quantpreflight
    rev: main
    hooks:
      - id: quantpreflight
```
