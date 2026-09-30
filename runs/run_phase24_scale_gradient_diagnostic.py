#!/usr/bin/env python3
"""Run the matched Phase 24 feature-scale gradient diagnostic."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.phase24_scale_gradient_diagnostic import run


if __name__ == "__main__":
    run()
