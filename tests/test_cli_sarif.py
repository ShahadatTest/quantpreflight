"""CLI command, exit code, JSON, SARIF 2.1.0, and `.quantpreflight.toml` tests."""

from __future__ import annotations

import json
from pathlib import Path
import random
import struct

import pytest

from conftest import write_checkpoint_dir
from quantpreflight.cli import main


def _small_ckpt(path: Path) -> Path:
    return write_checkpoint_dir(
        path,
        config={"architectures": ["LlamaForCausalLM"], "vocab_size": 10, "hidden_size": 8},
        tensors={"model.embed_tokens.weight": ("F16", [10, 8]), "lm_head.weight": ("F16", [10, 8])},
    )


def test_cli_rejects_deeply_nested_config_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ckpt = _small_ckpt(tmp_path / "deep_cfg")
    (ckpt / "config.json").write_text("[" * 200_000, encoding="utf-8")

    assert main(["check", str(ckpt)]) == 2
    assert "Error scanning" in capsys.readouterr().err


def test_cli_rejects_deeply_nested_safetensors_header(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ckpt = _small_ckpt(tmp_path / "deep_hdr")
    raw = b"[" * 200_000
    (ckpt / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)

    assert main(["check", str(ckpt)]) == 2
    assert "Error scanning" in capsys.readouterr().err


def test_cli_survives_seeded_header_corruption(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rng = random.Random(7)
    for i in range(200):
        ckpt = _small_ckpt(tmp_path / f"fuzz_{i}")
        victim = ckpt / rng.choice(["model.safetensors", "config.json"])
        blob = bytearray(victim.read_bytes())
        for _ in range(rng.randint(1, 12)):
            blob[rng.randrange(len(blob))] = rng.randrange(256)
        victim.write_bytes(bytes(blob[: rng.randint(1, len(blob))]))

        assert main(["check", str(ckpt), "--gpu", "H100"]) in (0, 1, 2)
    capsys.readouterr()


def test_cli_utility_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate-profiles"]) == 0
    out_val = capsys.readouterr().out
    assert "Validated" in out_val

    assert main(["list-gpus"]) == 0
    out_gpus = capsys.readouterr().out
    assert "H100" in out_gpus and "GB10" in out_gpus and "MI300X" in out_gpus

    assert main(["list-rules"]) == 0
    out_rules = capsys.readouterr().out
    assert "QPF001" in out_rules and "QPF012" in out_rules and "QPF013" in out_rules and "QPF014" in out_rules


def test_cli_sarif_and_json_and_ignore(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ckpt = write_checkpoint_dir(
        tmp_path / "bad_bf16_t4",
        config={
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 1000,
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
        },
        tensors={"model.embed_tokens.weight": ("BF16", [1000, 512])},
    )

    sarif_out = tmp_path / "report.sarif"
    rc = main(["check", str(ckpt), "--gpu", "T4", "--format", "sarif", "--output", str(sarif_out), "--fail-on", "medium"])
    assert rc == 1
    sarif_obj = json.loads(sarif_out.read_text(encoding="utf-8"))
    assert sarif_obj["version"] == "2.1.0"
    results = sarif_obj["runs"][0]["results"]
    assert any(r["ruleId"] == "QPF004" for r in results)

    # Suppress QPF004 via .quantpreflight.toml -> exit code becomes 0
    cfg_toml = tmp_path / ".quantpreflight.toml"
    cfg_toml.write_text('ignore = ["QPF004"]\n', encoding="utf-8")
    capsys.readouterr()
    rc_suppressed = main(
        ["check", str(ckpt), "--gpu", "T4", "--config", str(cfg_toml), "--format", "json"]
    )
    assert rc_suppressed == 0
    json_payload = json.loads(capsys.readouterr().out)
    assert json_payload["findings"] == []
