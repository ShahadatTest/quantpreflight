"""Load `.quantpreflight.toml` rule suppressions and default deployment targets."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found]


def load_project_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load `.quantpreflight.toml` from `config_path` or current working directory if present."""
    candidate = config_path or (Path.cwd() / ".quantpreflight.toml")
    if not candidate.is_file():
        return {}
    with candidate.open("rb") as fh:
        data = tomllib.load(fh)
    return data if isinstance(data, dict) else {}


def resolve_ignored_rules(cli_ignores: list[str] | None, cfg: dict[str, Any]) -> set[str]:
    """Merge CLI `--ignore` rule IDs with `ignore = [...]` in `.quantpreflight.toml`."""
    ignored: set[str] = set()
    cfg_list = cfg.get("ignore")
    if isinstance(cfg_list, list):
        for item in cfg_list:
            if isinstance(item, str) and item.strip():
                ignored.add(item.strip().upper())
    if cli_ignores:
        for raw in cli_ignores:
            for part in raw.split(","):
                if part.strip():
                    ignored.add(part.strip().upper())
    return ignored
