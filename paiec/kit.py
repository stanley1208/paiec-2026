"""Helpers imported unchanged from the official starting kit, copied into third_party/paiec_starting_kit/."""

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1] / "third_party" / "paiec_starting_kit" / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.append(str(_TOOLS))

from prediction_workers import prediction_input_key  # noqa: E402,F401
from streaming_ingestion import AcquisitionHook  # noqa: E402,F401
