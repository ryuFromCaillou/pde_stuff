import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hpc_api.tabulate_sweep import tabulate_sweep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_root", type=str, help="Path containing sweep_results.csv")
    ap.add_argument("--out-dir", type=str, default=None, help="Output directory (default: sweep_root)")
    ap.add_argument("--basename", type=str, default="sweep_table", help="Output basename (csv/md)")
    args = ap.parse_args()

    df = tabulate_sweep(Path(args.sweep_root), out_dir=args.out_dir, basename=args.basename)
    print(df.to_markdown(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

