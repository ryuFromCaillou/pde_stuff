"""Run the predeclared frozen 8000/8750 intervention in a NEW directory."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.phase19b_recombination import run, ROOT
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'run_results/phase19b_recombination_intervention')
    args=p.parse_args();run(args.out)
