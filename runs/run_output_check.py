from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.output_validation import find_latest_run_dir, load_json, validate_run_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a run_results/* run directory against an expectation spec.")
    p.add_argument("--spec", type=str, required=True, help="Path to a JSON spec file.")
    p.add_argument("--run-dir", type=str, default=None, help="Explicit run directory to validate.")
    p.add_argument(
        "--latest",
        action="store_true",
        help="Validate the latest run directory under --root (by summary.json mtime).",
    )
    p.add_argument(
        "--root",
        type=str,
        default=str(_REPO_ROOT / "run_results"),
        help="Root folder to search when using --latest.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    spec_path = Path(args.spec)
    spec = load_json(spec_path)

    run_dir = Path(args.run_dir) if args.run_dir else None
    if args.latest:
        latest = find_latest_run_dir(Path(args.root))
        if latest is None:
            print(f"No run dirs found under {args.root}", file=sys.stderr)
            return 2
        run_dir = latest

    if run_dir is None:
        print("Provide --run-dir or use --latest", file=sys.stderr)
        return 2

    result = validate_run_dir(run_dir, spec)
    if result.ok:
        print(f"OK: {run_dir}")
        return 0

    print(f"FAILED: {run_dir}", file=sys.stderr)
    for e in result.errors:
        print(f"- {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

