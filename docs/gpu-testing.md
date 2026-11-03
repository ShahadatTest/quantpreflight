# GPU testing

`quantpreflight` predicts whether a model will load before anyone downloads it. The only way to know if those predictions are right is to load the models for real. So far that has been done on Tesla T4 only, and [the results](accuracy.md#real-gpus) changed several rules. Every other GPU in the table rests on vendor docs and engine issues.

If you can run vLLM on another GPU, one run gives us the evidence we can't get ourselves.

## What it does

[`tools/gpu_check.py`](../tools/gpu_check.py) goes through a list of 27 public models. For each one it:

1. scans the model the way `quantpreflight check` would, from headers only,
2. loads it with vLLM in a separate process and asks for a few tokens,
3. records whether it loaded, the peak memory, and the error if it failed.

The results go to a JSON file after every model, so you can stop at any time and keep what finished.

## What you need

- Linux with an NVIDIA GPU and a working vLLM install. AMD GPUs with ROCm work too, but peak memory isn't recorded there yet.
- About 40 GB of free disk. Models download into their own cache folder, which is emptied after each one, so your Hugging Face cache is left alone.
- One to three hours. Cases that need two GPUs are skipped on a single GPU.

No Hugging Face token is needed. Every model in the list is public.

## Run it

```bash
git clone https://github.com/layer8labs8/quantpreflight
cd quantpreflight
pip install -e . vllm
python tools/gpu_check.py --out gpu_check_results.json
```

If you already have vLLM, skip `vllm` in the install so your version stays. To run a few models first, pass words from their names:

```bash
python tools/gpu_check.py --only Qwen2.5-7B TinyLlama --out gpu_check_results.json
```

Set `QPF_GPU_CHECK_CACHE` to download somewhere other than `~/.cache/quantpreflight-gpu-check`.

## Share the results

[Open a GPU results issue](https://github.com/layer8labs8/quantpreflight/issues/new?template=gpu_results.yml) and attach `gpu_check_results.json`. Partial runs help too.

The file holds the GPU names, memory, the vLLM version, and the last lines of each load's log. Look through the log lines before you post them, in case a path shows a user name you'd rather keep private.

## What we learn from it

| GPU family | What a run would confirm |
| :--- | :--- |
| Ampere: A100, A10G, RTX 3090 | Whether FP8 checkpoints load through vLLM's weight-only fallback, and the ModelOpt limit at compute capability 80 |
| Ada: L4, L40S, RTX 4090 | FP8 on GPUs with FP8 cores, and the block FP8 limit on sm_89 |
| Hopper: H100, H200 | FP8 and block FP8, and whether the memory estimate holds on large GPUs |
| Blackwell: B200, RTX 5090, DGX Spark | NVFP4, which is the main reason these GPUs are in the table |
| AMD: MI250X, MI300X, MI325X | Every AMD rule, none of which has been checked yet |

A result that shows a wrong prediction is the most useful kind. Please share those first.
