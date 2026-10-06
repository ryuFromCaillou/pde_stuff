#!/usr/bin/env python3
"""Exact diagnostic replay; protocol in PHASE19B_TRANSITION_INSTRUMENTED.md."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.phase19b_instrumented_replay import OUT, replay
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--stage',choices=['all','replay','analyze'],default='all')
    args=p.parse_args()
    if args.stage in ['all','replay']:replay(args.out)
    if args.stage in ['all','analyze']:
        from utils.phase19b_instrumented_analysis import analyze
        analyze(args.out)
        from utils.phase19b_instrumented_plotting import render
        render(args.out)
        from utils.phase19b_instrumented_report import report
        report(args.out)
        from utils.transition_instrumented_io import finalize
        finalize(args.out)
