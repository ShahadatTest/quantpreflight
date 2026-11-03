# Accuracy

This page collects the evidence behind `quantpreflight`, along with its limits. Everything here can be reproduced from this repository.

| Check | Result |
| :--- | ---: |
| Real vLLM 0.33.0 loads on Tesla T4 | 26 of 27 predicted correctly |
| Suggested setups on Tesla T4 | 9 of 9 predicted correctly |
| Public Hugging Face models scanned | 505, with 0 crashes |
| Scenarios against live models | 49 of 49 caught |
| Labelled benchmark | 96.9% recall, 100% precision |

## Real GPUs

Each model was scanned, then loaded with vLLM 0.33.0 on Tesla T4 GPUs on Kaggle. A model counts as loaded if it finished starting and generated text. The results are in [`gpu_check_t4.json`](../benchmark/gpu_check_t4.json), and [`tools/gpu_check.py`](../tools/gpu_check.py) repeats the check on your own GPUs.

| Model | GPUs | Prediction | Real result | Correct |
| :--- | ---: | :--- | :--- | :--- |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1 | Go | Loaded | Yes |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1 | Go | Loaded | Yes |
| `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` | 1 | Go | Loaded | Yes |
| `casperhansen/llama-3-8b-instruct-awq` | 1 | Go | Loaded | Yes |
| `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8` | 1 | Go | Loaded | Yes |
| `TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ` | 1 | No-go (QPF008) | Failed | Yes |
| `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8` | 1 | Go | Loaded | Yes |
| `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic` | 1 | No-go (QPF004) | Failed | Yes |
| `NousResearch/Hermes-3-Llama-3.1-8B` | 2 | Go | Loaded | Yes |
| `Qwen/Qwen2.5-VL-3B-Instruct` | 1 | Go | Loaded | Yes |
| `Qwen/Qwen2.5-0.5B-Instruct` | 1 | Go | Loaded | Yes |
| `Qwen/Qwen2.5-7B-Instruct-AWQ` | 1 | Go | Loaded | Yes |
| `Qwen/Qwen2.5-7B-Instruct-AWQ` | 2 | Go | Loaded | Yes |
| `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` | 1 | Go | Loaded | Yes |
| `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16` | 1 | No-go (QPF008) | Failed | Yes |
| `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` | 1 | No-go (QPF011) | Failed | Yes |
| `Qwen/Qwen3-8B-FP8` | 1 | No-go (QPF004) | Loaded | **No** |
| `nvidia/Llama-3.1-8B-Instruct-FP8` | 1 | No-go (QPF004) | Failed | Yes |
| `nvidia/Qwen3-8B-FP4` | 1 | No-go (QPF004) | Failed | Yes |
| `Qwen/Qwen2.5-14B-Instruct-AWQ` | 1 | Go | Loaded | Yes |
| `Qwen/Qwen2.5-32B-Instruct-AWQ` | 1 | No-go (QPF012) | Failed | Yes |
| `Qwen/Qwen2.5-32B-Instruct-AWQ` | 2 | Go | Loaded | Yes |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | 1 | Go | Loaded | Yes |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | 2 | No-go (QPF006) | Failed | Yes |
| `facebook/opt-125m` | 1 | Go | Loaded | Yes |
| `Zerul/Qwen3.5-0.8B-W4A16-AutoRound-AWQ` | 1 | Go | Loaded | Yes |
| `RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16` | 2 | Go | Loaded | Yes |

**Read this number with care.** These runs were also used to fix the rules, so 26 of 27 isn't a blind test. Before the fixes, the same kind of run showed wrong answers on BF16, FP8, `desc_act`, and `bitsandbytes`.

The one miss is FP8 on a GPU without FP8 cores. Of the three FP8 models tried, one loaded and two failed, so the tool warns on all of them. Security rules such as `QPF014` aren't counted here, because they say a model is risky, not that it won't load.

Only T4 has been tested. Rules for L4, H100, Blackwell, and AMD GPUs rest on vendor docs and engine issues. [GPU testing](gpu-testing.md) shows how to add results from your own GPUs.

## Memory

The estimate per GPU is `weights × 1.18 + 0.65 GiB + KV cache`. The first two terms were fitted to 16 of these runs, and the estimate landed within 5% of the memory vLLM reported on 12 of them, with a worst case of 16% on the smallest model. The KV cache formula is not fitted, and it matched vLLM's own figure closely.

The fit comes from one GPU type. It assumes one request at a time unless you pass `--concurrency`.

## Setups

Each report ends with a setup that should start, such as a shorter context when the model's full one won't fit. [`tools/setup_check.py`](../tools/setup_check.py) asks for the setup on the GPUs of the machine, starts vLLM with it, and compares. The results are in [`setup_check_t4.json`](../benchmark/setup_check_t4.json). The last column is the longest context vLLM said would fit, from its own error message.

| Model | GPUs | Context | Setup says | Real result | vLLM's longest | Correct |
| :--- | ---: | ---: | :--- | :--- | ---: | :--- |
| `Qwen/Qwen3-8B-AWQ` | 1 | model default | Won't start | Failed | 39,584 | Yes |
| `Qwen/Qwen3-8B-AWQ` | 1 | 37,888 | Starts | Started |  | Yes |
| `Qwen/Qwen3-14B-AWQ` | 1 | model default | Won't start | Failed | 8,656 | Yes |
| `Qwen/Qwen3-14B-AWQ` | 1 | 6,144 | Starts | Started |  | Yes |
| `casperhansen/mistral-nemo-instruct-2407-awq` | 1 | model default | Won't start | Failed | 19,136 | Yes |
| `casperhansen/mistral-nemo-instruct-2407-awq` | 1 | 17,408 | Starts | Started |  | Yes |
| `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` | 1 | model default | Starts | Started |  | Yes |
| `Qwen/Qwen3-4B-AWQ` | 1 | model default | Starts | Started |  | Yes |
| `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` | 1 | model default | Starts | Started |  | Yes |

That's 9 of 9 on Tesla T4. It took three rounds. The first, with no spare memory, got 7 of 8 right, and the second got 6 of 7. In both, one suggested context was 1 to 2% longer than what vLLM could hold, because the estimate gave the cache between 0.12 and 0.31 GiB more room than vLLM did. A suggested context now leaves 0.5 GiB spare. On some models that's cautious: Qwen3-14B fits 8,656 tokens, and the setup suggests 6,144. The earlier rounds also started Qwen2.5-14B-AWQ with 7,168 tokens, where the setup now suggests 5,120.

The models in the last round that weren't in the first two are a fresh test, and all of them agreed. Like the memory fit, these results come from one GPU type.

## Public models

[`tools/scan_public_models.py`](../tools/scan_public_models.py) scanned popular models from Hugging Face. The results are in [`public_models.json`](../benchmark/public_models.json).

| Result | Models |
| :--- | ---: |
| Scanned | 505 |
| Nothing to report | 359 |
| Couldn't be read, such as weights kept in subfolders | 11 |
| Crashes | 0 |

| Rule | What it means | Models |
| :--- | :--- | ---: |
| `QPF014` | Insecure legacy weight format or unpinned custom modeling code | 58 |
| `QPF011` | Serving engine container or quant format mismatch | 53 |
| `QPF003` | Stop-token (eos_token_id) drift across configs | 20 |
| `QPF007` | Quantized tensor group-size or Marlin tile alignment mismatch | 10 |
| `QPF009` | Broken shard index or speculative MTP projection shape mismatch | 6 |
| `QPF001` | Bit-starved sensitive layer | 4 |
| `QPF010` | Quantization config vs. tensor header contradiction | 3 |
| `QPF002` | Tokenizer or vocabulary ID out of embedding bounds | 2 |
| `QPF008` | GPTQ desc_act (act_order) is rejected or slow | 1 |

These are risky patterns, not proven failures. Nobody loaded these 505 models, and each one was checked against one assumed target. This scan found real bugs in the tool, including a false alarm on GGUF models that hit 33 of 73.

## Labelled benchmark

Every change runs against [`cases.json`](../benchmark/cases.json), 132 small checkpoints labelled with what must fire and what must stay quiet.

| Group | Cases | Passed | Findings missed | False alarms |
| :--- | ---: | ---: | ---: | ---: |
| Safe setups | 36 | 36 | 0 | 0 |
| Sensitive layers | 6 | 6 | 0 | 0 |
| Tokens and stop IDs | 13 | 13 | 0 | 0 |
| GPU support | 16 | 16 | 0 | 0 |
| Silent kernel bugs | 8 | 8 | 0 | 0 |
| Splitting across GPUs | 15 | 15 | 0 | 0 |
| Broken files | 17 | 17 | 0 | 0 |
| Engine and format | 12 | 12 | 0 | 0 |
| GPU memory | 6 | 6 | 0 | 0 |
| Known gaps | 3 | 0 | 3 | 0 |

A case passes when every finding it expects fires and nothing it forbids does. That's 96.9% recall and 100% precision on these cases. We wrote them ourselves, so treat this as a regression check and not a promise about your model.

The 3 misses are real gaps: NaNs in the weights, running out of memory under real traffic when `--concurrency` isn't set, and bugs that only a live request can trigger when `--engine-version` isn't set.

## Tests

Checkpoints are untrusted input, so the tests aim at both wrong answers and bad files.

| Tests | What they prove |
| :--- | :--- |
| [Rule boundaries](../tests/test_rule_boundaries.py) | Each rule fires on the bad input and stays quiet on the one next to it. |
| [Compatibility matrix](../tests/test_compatibility_matrix.py) | Results match answers worked out separately from the GPU tables and plain arithmetic. |
| [Hostile inputs](../tests/test_hostile_inputs.py) | Malformed values in every config field end in a report or a clean exit, never a crash. |
| [Binary parsers](../tests/test_binary_parsers.py) | Every GGUF and Safetensors safety limit holds, byte by byte. |
| [Remote loading](../tests/test_loader_remote.py) | Hugging Face paths, retries, and token handling work, with no network needed. |
| [Benchmark](../tests/test_benchmark.py) | Every labelled case keeps its result, and known gaps stay listed. |

Most of the roughly 7,000 tests come from tables of inputs, so the count says less than the coverage does. Line coverage is 99%. We also break the code on purpose, one comparison at a time, to confirm that a test notices.
