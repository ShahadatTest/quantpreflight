# Contributing to `quantpreflight`

`quantpreflight` is designed so contributors can add support for a **new model architecture** or **new GPU/accelerator** by editing a single declarative TOML file—without needing a GPU or downloading multi-gigabyte models.

## Add a Model Architecture in 5 Minutes

1. Create `src/quantpreflight/architectures/<family>.toml`:

```toml
id = "example_moe"
name = "Example-MoE Hybrid"
match_architectures = ["examplemoeforcausallm"]
match_model_types = ["example_moe"]

col_parallel_suffixes = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
row_parallel_suffixes = ["o_proj", "down_proj"]

[[sensitive_layers]]
name = "example_moe_router"
pattern = '(^|\.)mlp\.gate\.(weight|qweight|weight_packed)$'
exclude_pattern = 'gate_(up_)?proj'
min_bits = 16.0
severity = "critical"
reason = "Router gate collapses expert balance if quantized below 16-bit."

[[sensitive_layers]]
name = "example_norms"
pattern = '(^|\.)([a-z0-9_]*norm[a-z0-9_]*)\.(weight|qweight|weight_packed)$'
min_bits = 16.0
severity = "critical"
reason = "RMSNorm weights must remain >=16-bit."
```

2. Validate profiles and run tests:

```bash
python -m pip install -e ".[dev]"
quantpreflight validate-profiles
python -m pytest -q
python tools/run_benchmark.py
```

3. Add one clean case and one broken case in `benchmark/cases.json` and open a Pull Request linking to the upstream model's `modeling_*.py` or `config.json`.

## Test Your Change

Tests run with no GPU and no network. Each kind has its own file in `tests/`:

| File | Add a test here when |
| :--- | :--- |
| `test_rule_boundaries.py` | A rule changes. Cover both the input that fires and the one next to it that doesn't. |
| `test_compatibility_matrix.py` | A rule follows a table or a formula. Work out the answer separately from the rule. |
| `test_hostile_inputs.py` | A new config key is read. Add it to the list so malformed values get tested. |
| `benchmark/cases.json` | You find a real checkpoint that was flagged wrongly or missed. |

Run the website locally with `python tools/build_site.py`, then open `_site/index.html` through any local web server.

## Add a New GPU or Engine Rule

Edit `src/quantpreflight/hardware/gpus.toml` or `src/quantpreflight/hardware/engines.toml`, run `quantpreflight validate-profiles`, and include a reference link to the vendor architecture spec or engine issue/PR.

## Check Results on Real GPUs

`python tools/gpu_check.py` compares each prediction with a real vLLM load on the GPUs of your machine. `python tools/build_gpu_notebook.py` packs the same check into a Kaggle notebook. Results from other GPUs are very welcome, because only T4 has been tested so far.

## Report a False Alarm or Missed Crash

Open an issue with the checkpoint's `config.json` (or public Hugging Face repo ID) and target `--engine` / `--gpu` / `--tp` flags. No model weight files are needed.
