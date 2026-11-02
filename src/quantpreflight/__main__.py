"""CLI module entry point for `python -m quantpreflight`."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
