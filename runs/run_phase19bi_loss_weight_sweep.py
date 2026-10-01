#!/usr/bin/env python3
"""Run or reuse the Phase 19B-i scale-transfer PDE-weight sweep."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from utils.phase19bi_loss_weight_sweep import run
if __name__ == "__main__": run()
