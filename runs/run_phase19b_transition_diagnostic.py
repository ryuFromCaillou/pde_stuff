#!/usr/bin/env python3
"""Post-hoc measurements and figures only; see PHASE19B_TRANSITION.md."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.phase19b_transition import run, DEFAULT_OUT
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    run(args.out)
    from utils.phase19b_transition_validation import validate
    validate(args.out)
    from utils.phase19b_transition_plotting import render
    render(args.out)
    from utils.phase19b_transition_report import render_report
    render_report(args.out)
