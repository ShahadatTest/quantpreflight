"""Core data structures for checkpoint metadata, target hardware, and findings."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """Finding severity levels ordered from highest to lowest impact."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def rank(self) -> int:
        return {
            Severity.CRITICAL: 4,
            Severity.HIGH: 3,
            Severity.MEDIUM: 2,
            Severity.LOW: 1,
        }[self]


@dataclass(frozen=True)
class TensorSpec:
    """Header-only representation of a single checkpoint tensor."""

    name: str
    storage_dtype: str
    shape: tuple[int, ...]
    shard: str
    byte_size: int
    effective_bits: float
    logical_shape: tuple[int, ...] | None = None
    data_offsets: tuple[int, int] | None = None

    @property
    def unpacked_shape(self) -> tuple[int, ...]:
        return self.logical_shape if self.logical_shape is not None else self.shape


@dataclass(frozen=True)
class QuantSpec:
    """Normalized quantization configuration extracted from config and headers."""

    method: str = "none"
    bits: float | None = None
    group_size: int | None = None
    sym: bool | None = None
    desc_act: bool = False
    zero_point: bool | None = None
    format_version: str | None = None
    activation_bits: float | None = None
    activation_dtype: str | None = None
    weight_block_size: tuple[int, int] | None = None
    ignored_layers: tuple[str, ...] = ()
    raw_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenizerMeta:
    """Token ID bounds and stop-token metadata parsed without loading tokenizers."""

    vocab_max_id: int | None = None
    max_token_id: int | None = None
    max_token_str: str | None = None
    total_tokens: int | None = None
    eos_token: str | None = None
    eos_token_ids: tuple[int, ...] = ()


@dataclass
class CheckpointMeta:
    """Complete zero-weight metadata snapshot of a local or remote checkpoint."""

    source: str
    container_format: str
    config: dict[str, Any] = field(default_factory=dict)
    generation_config: dict[str, Any] = field(default_factory=dict)
    tokenizer_config: dict[str, Any] = field(default_factory=dict)
    tokenizer_meta: TokenizerMeta = field(default_factory=TokenizerMeta)
    tensors: dict[str, TensorSpec] = field(default_factory=dict)
    shard_files: tuple[str, ...] = ()
    missing_shards: tuple[str, ...] = ()
    dangling_index_tensors: tuple[str, ...] = ()
    duplicate_tensors: tuple[str, ...] = ()
    legacy_weight_files: tuple[str, ...] = ()
    custom_code_files: tuple[str, ...] = ()
    shard_file_sizes: dict[str, int] = field(default_factory=dict)
    shard_header_lengths: dict[str, int] = field(default_factory=dict)
    safetensors_metadata: dict[str, str] = field(default_factory=dict)
    gguf_metadata: dict[str, Any] = field(default_factory=dict)
    header_bytes_fetched: int = 0
    total_weight_bytes: int = 0
    quant: QuantSpec = field(default_factory=QuantSpec)


@dataclass(frozen=True)
class TargetSpec:
    """Deployment target parameters supplied via CLI flags or config."""

    engine: str = "vllm"
    gpu: str | None = None
    tp: int = 1
    context_len: int | None = None
    vram_gb_override: float | None = None
    gpu_memory_utilization: float = 0.90
    marlin_input_dtype: str | None = None
    engine_version: str | None = None
    concurrency: int = 1


@dataclass(frozen=True)
class Finding:
    """A single pre-flight compatibility, precision, or crash finding."""

    rule_id: str
    severity: Severity
    title: str
    message: str
    remediation: str
    tensors: tuple[str, ...] = ()
    location: str = "config.json"
    references: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "remediation": self.remediation,
            "location": self.location,
            "tensors": list(self.tensors),
            "references": list(self.references),
        }


@dataclass
class SetupPlan:
    """A setup that should start, worked out from the scan."""

    # "ready" as asked, "changed" to start, or "blocked" when no setting helps
    status: str
    tp: int
    context_len: int | None = None
    max_context: int | None = None
    estimated_vram_per_gpu_gb: float | None = None
    changes: tuple[str, ...] = ()
    blocking: tuple[Finding, ...] = ()
    command: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "tp": self.tp,
            "context_len": self.context_len,
            "max_context": self.max_context,
            "estimated_vram_per_gpu_gb": self.estimated_vram_per_gpu_gb,
            "changes": list(self.changes),
            "blocking": [f.rule_id for f in self.blocking],
            "command": self.command,
        }


@dataclass
class ScanReport:
    """Structured scan summary and findings list."""

    source: str
    container_format: str
    architecture: str
    quant_summary: str
    total_weight_bytes: int
    header_bytes_fetched: int
    tensor_count: int
    shard_count: int
    estimated_vram_per_gpu_gb: float | None
    target: TargetSpec
    findings: list[Finding] = field(default_factory=list)
    setup: SetupPlan | None = None

    def highest_severity(self) -> Severity | None:
        if not self.findings:
            return None
        return max(self.findings, key=lambda f: f.severity.rank).severity

    def passed(self, fail_on: Severity | None) -> bool:
        if fail_on is None:
            return True
        return all(f.severity.rank < fail_on.rank for f in self.findings)
