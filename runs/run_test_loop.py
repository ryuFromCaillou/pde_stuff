from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.output_validation import find_latest_run_dir, load_json, validate_run_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a command repeatedly and validate the latest run_results output against a spec."
    )
    p.add_argument("--spec", type=str, required=True, help="Path to JSON validation spec.")
    p.add_argument("--root", type=str, default=str(_REPO_ROOT / "run_results"), help="run_results root folder.")
    p.add_argument("--interval-sec", type=float, default=0.0, help="Sleep between iterations (seconds).")
    p.add_argument("--max-iters", type=int, default=1, help="Maximum iterations (0 = infinite).")
    p.add_argument("--stop-on-pass", action="store_true", help="Stop as soon as validation passes.")
    p.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command to run. Use `--` before the command, e.g. `python runs/run_test_loop.py ... -- python runs/run_tv_lambda_sweep.py ...`",
    )
    return p.parse_args()


def _trim_leading_double_dash(cmd: List[str]) -> List[str]:
    if cmd and cmd[0] == "--":
        return cmd[1:]
    return cmd


def main() -> int:
    args = _parse_args()
    spec = load_json(Path(args.spec))
    root = Path(args.root)

    cmd = _trim_leading_double_dash(list(args.cmd))
    if not cmd:
        print("No command provided. Pass it after `--`.", file=sys.stderr)
        return 2

    it = 0
    while True:
        it += 1
        print(f"[iter {it}] running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=str(_REPO_ROOT))
        if proc.returncode != 0:
            print(f"[iter {it}] command failed with exit code {proc.returncode}", file=sys.stderr)
            if args.max_iters and it >= int(args.max_iters):
                return proc.returncode
        else:
            latest = find_latest_run_dir(root)
            if latest is None:
                print(f"[iter {it}] no summary.json found under {root}", file=sys.stderr)
                ok = False
            else:
                result = validate_run_dir(latest, spec)
                ok = result.ok
                if ok:
                    print(f"[iter {it}] PASS: {latest}")
                else:
                    print(f"[iter {it}] FAIL: {latest}", file=sys.stderr)
                    for e in result.errors:
                        print(f"- {e}", file=sys.stderr)

            if ok and args.stop_on_pass:
                return 0

        if args.max_iters and it >= int(args.max_iters):
            return 0

        if float(args.interval_sec) > 0:
            time.sleep(float(args.interval_sec))


if __name__ == "__main__":
    raise SystemExit(main())

