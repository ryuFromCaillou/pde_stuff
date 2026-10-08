#!/usr/bin/env python3
"""Validate parameter L1, then run the matched Paul-path pilot."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paul_path_l1 import main
if __name__ == '__main__':
    main()
