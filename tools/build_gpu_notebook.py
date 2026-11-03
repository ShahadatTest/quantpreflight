"""Build a self-contained Kaggle notebook that runs tools/gpu_check.py.

The package wheel is embedded, so the notebook tests your local copy, changes included.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "kaggle_gpu_check.ipynb"
ARGS = " ".join(sys.argv[1:])

INTRO = """# quantpreflight GPU check

Compares quantpreflight's predictions with real vLLM loads.

1. In the right panel, set **Accelerator** to **GPU T4 x2** and turn **Internet** on.
2. Choose **Save Version**, then **Save & Run All (Commit)**, so it keeps running after you close the tab.
3. When it finishes, open the version's **Output** tab and download `gpu_check_results.json`.

It takes about 1 to 3 hours and only uses public models. No token or password is needed.
"""


def _wheel() -> bytes:
    with tempfile.TemporaryDirectory() as folder:
        command = [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", folder, str(ROOT)]
        subprocess.run(command, check=True, capture_output=True, cwd=folder)
        return next(Path(folder).glob("*.whl")).read_bytes()


def _code(number: int, source: str) -> dict[str, object]:
    return {
        "id": f"cell-{number}",
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    wheel = base64.b64encode(_wheel()).decode()
    script = (ROOT / "tools" / "gpu_check.py").read_text(encoding="utf-8")
    cells = [
        {"id": "intro", "cell_type": "markdown", "metadata": {}, "source": INTRO.splitlines(keepends=True)},
        _code(1, "!nvidia-smi\n!df -h /kaggle/working | tail -1\n"),
        _code(
            2,
            "import base64, pathlib\n"
            f'pathlib.Path("quantpreflight-0-py3-none-any.b64").write_text("{wheel}")\n'
            'wheel = pathlib.Path("quantpreflight-0.2.0-py3-none-any.whl")\n'
            'wheel.write_bytes(base64.b64decode(pathlib.Path("quantpreflight-0-py3-none-any.b64").read_text()))\n'
            # The torch that vLLM pulls in must match torchvision and torchaudio
            "!pip install -q -U vllm torchvision torchaudio ./quantpreflight-0.2.0-py3-none-any.whl\n"
            "!python -c 'import torchaudio' || pip uninstall -y -q torchaudio\n"
        ),
        _code(3, f"pathlib.Path('gpu_check.py').write_text({script!r})\n"),
        _code(4, f"!python gpu_check.py {ARGS} --out /kaggle/working/gpu_check_results.json\n"),
    ]
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"{OUT} ({OUT.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
