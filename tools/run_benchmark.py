"""Run labelled pre-flight cases and report what was caught, missed, and wrongly flagged."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quantpreflight.gguf_io import ParsedGGUFHeader, RawGGUFTensorEntry, compute_ggml_tensor_bytes  # noqa: E402
from quantpreflight.models import CheckpointMeta, TargetSpec, TokenizerMeta  # noqa: E402
from quantpreflight.quant_detect import (  # noqa: E402
    build_safetensors_specs,
    normalize_gguf_metadata,
    parse_quant_spec_from_config,
)
from quantpreflight.rules import analyze_checkpoint  # noqa: E402
from quantpreflight.safetensors_io import RawSafetensorEntry, SAFETENSORS_DTYPE_BITS  # noqa: E402

CASES_PATH = ROOT / "benchmark" / "cases.json"
GAPS = "gaps"


@dataclass
class Score:
    """Results of one group of cases."""

    found: int = 0
    missed: list[str] = field(default_factory=list)
    false_alarms: list[str] = field(default_factory=list)
    cases: int = 0
    passed: int = 0

    def add(self, other: "Score") -> None:
        self.cases += other.cases
        self.passed += other.passed
        self.found += other.found
        self.missed += other.missed
        self.false_alarms += other.false_alarms

    @property
    def recall(self) -> float:
        total = self.found + len(self.missed)
        return self.found / total if total else 1.0

    @property
    def precision(self) -> float:
        total = self.found + len(self.false_alarms)
        return self.found / total if total else 1.0


def _build_meta_from_case(case: dict[str, Any]) -> CheckpointMeta:
    if case["container"] == "gguf":
        arch = case.get("gguf_arch", "llama")
        md: dict[str, Any] = {
            "general.architecture": arch,
            "general.file_type": case.get("gguf_file_type", 15),
            f"{arch}.embedding_length": 4096,
            f"{arch}.block_count": 32,
            f"{arch}.attention.head_count": 32,
            f"{arch}.attention.head_count_kv": 8,
            "tokenizer.ggml.tokens": {
                "__gguf_string_table_len__": case.get("gguf_vocab", 32000),
                "head": ["<unk>"],
                "last": "<eof>",
            },
        }
        gtensors: dict[str, RawGGUFTensorEntry] = {}
        for item in case.get("gguf_tensors", []):
            dims = tuple(item["dims"])
            py_shape = tuple(reversed(dims)) if len(dims) >= 2 else dims
            dtype_name, bsize, lbits = compute_ggml_tensor_bytes(item["ggml_type"], dims)
            gtensors[item["name"]] = RawGGUFTensorEntry(
                name=item["name"],
                ggml_type_id=item["ggml_type"],
                dtype_name=dtype_name,
                shape=py_shape,
                offset=0,
                byte_size=bsize,
                effective_bits=lbits,
            )
        parsed = ParsedGGUFHeader(version=3, metadata=md, tensors=gtensors, header_bytes_consumed=4096)
        cfg, quant, specs = normalize_gguf_metadata(parsed)
        return CheckpointMeta(
            source=case["id"],
            container_format="gguf",
            config=cfg,
            tokenizer_meta=TokenizerMeta(
                vocab_max_id=case.get("gguf_vocab", 32000) - 1,
                max_token_id=case.get("gguf_vocab", 32000) - 1,
                max_token_str="<eof>",
                total_tokens=case.get("gguf_vocab", 32000),
            ),
            tensors=specs,
            shard_files=("model.gguf",),
            missing_shards=tuple(case.get("missing_shards", ())),
            gguf_metadata=md,
            header_bytes_fetched=4096,
            total_weight_bytes=sum(s.byte_size for s in specs.values()),
            quant=quant,
        )

    config = case.get("config", {})
    quant = parse_quant_spec_from_config(config)
    raw_entries: dict[str, tuple[RawSafetensorEntry, str]] = {}
    running_offset = 0
    for tname, tinfo in case.get("tensors", {}).items():
        shape = tuple(tinfo["shape"])
        dtype = tinfo["dtype"]
        numel = 1
        for d in shape:
            numel *= d
        bsize = tinfo.get("byte_size_override") or int((numel * SAFETENSORS_DTYPE_BITS[dtype] + 7) // 8)
        start_off = tinfo.get("offset_override", running_offset)
        end_off = start_off + bsize
        raw_entries[tname] = (
            RawSafetensorEntry(
                name=tname,
                dtype=dtype,
                shape=shape,
                data_offsets=(start_off, end_off),
                byte_size=bsize,
            ),
            tinfo.get("shard", "model.safetensors"),
        )
        running_offset = end_off

    specs = build_safetensors_specs(raw_entries, quant)
    max_id = case.get("tokenizer_max_id")
    eos_id = case.get("tokenizer_eos_id")
    tok_cfg = case.get("tokenizer_config", {})
    tok_meta = TokenizerMeta(
        vocab_max_id=max_id,
        max_token_id=max_id,
        max_token_str="<special>" if max_id is not None else None,
        total_tokens=(max_id + 1) if max_id is not None else None,
        eos_token=tok_cfg.get("eos_token"),
        eos_token_ids=(eos_id,) if isinstance(eos_id, int) else (),
    )
    return CheckpointMeta(
        source=case["id"],
        container_format=case.get("container", "safetensors"),
        config=config,
        generation_config=case.get("generation_config", {}),
        tokenizer_config=tok_cfg,
        tokenizer_meta=tok_meta,
        tensors=specs,
        shard_files=tuple(case.get("shard_files", ("model.safetensors",))),
        missing_shards=tuple(case.get("missing_shards", ())),
        dangling_index_tensors=tuple(case.get("dangling_index_tensors", ())),
        duplicate_tensors=tuple(case.get("duplicate_tensors", ())),
        legacy_weight_files=tuple(case.get("legacy_weight_files", ())),
        custom_code_files=tuple(case.get("custom_code_files", ())),
        shard_file_sizes=case.get("shard_file_sizes", {}),
        shard_header_lengths=case.get("shard_header_lengths", {}),
        header_bytes_fetched=2048,
        total_weight_bytes=case.get("total_weight_bytes", sum(s.byte_size for s in specs.values())),
        quant=quant,
    )


def load_cases() -> dict[str, list[dict[str, Any]]]:
    """Return the labelled cases by group, in the order they first appear."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for case in json.loads(CASES_PATH.read_text(encoding="utf-8")):
        groups.setdefault(case["group"], []).append(case)
    return groups


def score_case(case: dict[str, Any]) -> Score:
    """Compare one case's findings with its labels."""
    t_raw = case.get("target", {})
    target = TargetSpec(
        engine=t_raw.get("engine", "vllm"),
        gpu=t_raw.get("gpu"),
        tp=t_raw.get("tp", 1),
        marlin_input_dtype=t_raw.get("marlin_input_dtype"),
        engine_version=t_raw.get("engine_version"),
        context_len=t_raw.get("context_len"),
    )
    report = analyze_checkpoint(_build_meta_from_case(case), target)
    fired = {f.rule_id for f in report.findings}
    must_fire = case.get("must_fire", [])

    score = Score(found=sum(rule in fired for rule in must_fire))
    score.missed = [f"{case['id']}: {rule}" for rule in must_fire if rule not in fired]
    # A case with nothing to find must stay completely quiet
    unwanted = fired if not must_fire else fired & set(case.get("must_not_fire", []))
    score.false_alarms = [f"{case['id']}: {rule}" for rule in sorted(unwanted)]
    score.cases = 1
    score.passed = int(not score.missed and not score.false_alarms)
    return score


def run(cases: dict[str, list[dict[str, Any]]]) -> dict[str, Score]:
    """Score every group."""
    results: dict[str, Score] = {}
    for group, items in cases.items():
        results[group] = Score()
        for case in items:
            results[group].add(score_case(case))
    return results


def main() -> int:
    cases = load_cases()
    results = run(cases)
    total = Score()
    failed = False

    print(f"{'Group':<12} {'Cases':>5} {'Passed':>7} {'Missed':>7} {'False alarms':>13}")
    for group, score in results.items():
        total.add(score)
        print(f"{group:<12} {score.cases:>5} {score.passed:>7} {len(score.missed):>7} {len(score.false_alarms):>13}")
        if group != GAPS and (score.missed or score.false_alarms):
            failed = True
            for line in score.missed:
                print(f"  MISSED {line}", file=sys.stderr)
            for line in score.false_alarms:
                print(f"  FALSE ALARM {line}", file=sys.stderr)

    count = sum(len(items) for items in cases.values())
    print(f"Benchmark: {count} cases, {total.recall:.1%} recall, {total.precision:.0%} precision.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
