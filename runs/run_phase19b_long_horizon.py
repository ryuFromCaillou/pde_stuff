#!/usr/bin/env python3
"""Run or resume the original Phase 19B long-horizon control."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from utils.phase19b_long_horizon import run
if __name__ == "__main__": run()
