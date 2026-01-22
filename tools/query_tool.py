#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
from datetime import datetime

def coerce(s: str):
    # try int, float, bool, else string
    sl = s.lower()
    if sl in ("true", "false"):
        return sl == "true"
    try:
        if re.fullmatch(r"[+-]?\d+", s):  # int
            return int(s)
        return float(s)                  # float
    except ValueError:
        return s

def iter_meta(root: Path):
    for p in root.rglob("meta.json"):
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        params = obj.get("params", {})
        yield {
            "meta_path": p,
            "run_dir": p.parent,
            "params": params,
            "saved_at": obj.get("saved_at"),
            "a_hat": obj.get("a_hat"),
        }

def get_nested(d, key):
    # params are flat in your code, but keep this anyway
    cur = d
    for k in key.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def parse_eq(expr: str):
    # key=value
    if "=" not in expr:
        raise ValueError(f"--eq must look like key=value, got: {expr}")
    k, v = expr.split("=", 1)
    return k.strip(), coerce(v.strip())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="runs", help="runs root")
    ap.add_argument("--eq", action="append", default=[], help="Exact match: key=value (repeatable)")
    ap.add_argument("--re", action="append", default=[], help="Regex match: key=/pattern/ (repeatable)")
    ap.add_argument("--has", action="append", default=[], help="List contains: key=item (repeatable)")
    ap.add_argument("--sort", type=str, default="saved_at", choices=["saved_at","run_dir"], help="sort key")
    ap.add_argument("--tail", type=int, default=50, help="show last N matches")
    args = ap.parse_args()

    eq_filters = [parse_eq(x) for x in args.eq]

    re_filters = []
    for x in args.re:
        if "=" not in x:
            raise ValueError(f"--re must look like key=/pat/, got: {x}")
        k, pat = x.split("=", 1)
        pat = pat.strip()
        if pat.startswith("/") and pat.endswith("/"):
            pat = pat[1:-1]
        re_filters.append((k.strip(), re.compile(pat)))

    has_filters = [parse_eq(x) for x in args.has]  # key=item, item coerced

    root = Path(args.root)
    rows = list(iter_meta(root))

    def match(row):
        p = row["params"]

        for k, v in eq_filters:
            if get_nested(p, k) != v:
                return False

        for k, rgx in re_filters:
            val = get_nested(p, k)
            if val is None or rgx.search(str(val)) is None:
                return False

        for k, item in has_filters:
            val = get_nested(p, k)
            if not isinstance(val, list) or item not in val:
                return False

        return True

    out = [r for r in rows if match(r)]

    def parse_dt(s):
        if not s:
            return datetime.min
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d-%H%M%S"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                pass
        return datetime.min

    if args.sort == "saved_at":
        out.sort(key=lambda r: parse_dt(r.get("saved_at")), reverse=True)
    else:
        out.sort(key=lambda r: str(r["run_dir"]), reverse=True)

    out = out[: max(0, args.tail)]

    for r in out:
        p = r["params"]
        print(r["run_dir"].as_posix())
        print(f"  saved_at={r.get('saved_at')}  a_hat={r.get('a_hat')}")
        keys = ["epochs","batch_size","lr","noise","stride_t","stride_x","part_num","which_part",
                "lam_tv","tv_type","lam_reg","lam_pde","lam_data","selected_derivs"]
        for k in keys:
            if k in p:
                print(f"  {k}={p[k]}")
        print()


if __name__ == "__main__":
    main()
