"""Errors a scan can end in, which the command line reports instead of a traceback."""

from __future__ import annotations

from .gguf_io import GGUFFormatError
from .loader import CheckpointLoadError
from .profiles import ProfileValidationError
from .safetensors_io import SafetensorsFormatError

# What a malformed checkpoint or a failed fetch can raise while scanning
SCAN_ERRORS = (
    CheckpointLoadError,
    SafetensorsFormatError,
    GGUFFormatError,
    ProfileValidationError,
    ValueError,
    OSError,
    RecursionError,
)
