#!/usr/bin/env python3
"""Run the read-only Phase 24 surrogate-fit audit."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.phase24_fit_audit import run


if __name__ == "__main__":
    run()
