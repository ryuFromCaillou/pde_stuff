#!/usr/bin/env python3
"""Execute or validate/reuse Phase 24. See runs/PHASE24.md for the output contract."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.burgers_recoverability import OUT, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == '__main__':
    main()
