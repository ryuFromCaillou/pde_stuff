from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


class ValidationError(RuntimeError):
    pass


Number = Union[int, float]


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_required_file_matches(run_dir: Path, pattern: str) -> Iterable[Path]:
    rel = Path(pattern)
    # If the pattern includes glob characters, use glob; otherwise exact path.
    if any(ch in pattern for ch in ["*", "?", "["]):
        for p in run_dir.glob(pattern):
            if p.is_file():
                yield p
        return

    p = run_dir / rel
    if p.is_file():
        yield p


def _validate_required_files(run_dir: Path, required_files: List[Any], errors: List[str]) -> None:
    """
    required_files entries can be:
      - "relative/path.ext" (optionally with glob chars)
      - {"any_of": ["a.json", "b.json"]} meaning at least one must exist
    """
    for entry in required_files:
        if isinstance(entry, str):
            patterns = [entry]
            label = entry
            require_any = False
        elif isinstance(entry, dict) and isinstance(entry.get("any_of"), list):
            patterns = [str(x) for x in entry.get("any_of") or []]
            label = "any_of(" + ", ".join(patterns) + ")"
            require_any = True
        else:
            errors.append("Invalid required_files entry (expected string or {any_of:[...]}).")
            continue

        found = False
        for pattern in patterns:
            matches = list(_iter_required_file_matches(run_dir, pattern))
            if matches:
                found = True
                if not require_any:
                    break

        if not found:
            errors.append(f"Missing required file(s): {label}")


def _check_constraint(key: str, value: Any, constraint: Dict[str, Any], errors: List[str]) -> None:
    if "eq" in constraint:
        expected = constraint["eq"]
        if value != expected:
            errors.append(f"summary[{key}] expected eq {expected!r}, got {value!r}")
        return

    if "approx" in constraint:
        approx = constraint["approx"] or {}
        target = approx.get("value", None)
        if target is None or not _is_number(target) or not _is_number(value):
            errors.append(f"summary[{key}] approx requires numeric value; got {value!r} vs {target!r}")
            return
        atol = float(approx.get("atol", 0.0))
        rtol = float(approx.get("rtol", 0.0))
        diff = abs(float(value) - float(target))
        tol = atol + rtol * abs(float(target))
        if diff > tol:
            errors.append(
                f"summary[{key}] expected approx {target} (atol={atol}, rtol={rtol}), got {value} (diff={diff})"
            )
        return

    if "min" in constraint:
        if not _is_number(value):
            errors.append(f"summary[{key}] expected numeric for min, got {value!r}")
        else:
            mn = float(constraint["min"])
            if float(value) < mn:
                errors.append(f"summary[{key}] expected >= {mn}, got {value}")

    if "max" in constraint:
        if not _is_number(value):
            errors.append(f"summary[{key}] expected numeric for max, got {value!r}")
        else:
            mx = float(constraint["max"])
            if float(value) > mx:
                errors.append(f"summary[{key}] expected <= {mx}, got {value}")


def validate_run_dir(run_dir: Union[str, Path], spec: Dict[str, Any]) -> ValidationResult:
    run_dir = Path(run_dir)
    errors: List[str] = []

    if not run_dir.exists():
        return ValidationResult(ok=False, errors=[f"Run dir does not exist: {str(run_dir)}"])
    if not run_dir.is_dir():
        return ValidationResult(ok=False, errors=[f"Run dir is not a directory: {str(run_dir)}"])

    required_files = spec.get("required_files") or []
    if not isinstance(required_files, list):
        errors.append("spec.required_files must be a list")
    else:
        _validate_required_files(run_dir, required_files, errors)

    summary_spec = spec.get("summary") or {}
    if summary_spec:
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            errors.append("Missing summary.json for summary checks")
        else:
            try:
                summary = load_json(summary_path)
            except Exception as e:
                errors.append(f"Failed to parse summary.json: {e}")
                summary = {}

            required_keys = summary_spec.get("required_keys") or []
            if required_keys:
                for k in required_keys:
                    if k not in summary:
                        errors.append(f"summary missing key: {k}")

            constraints = summary_spec.get("constraints") or {}
            if constraints:
                if not isinstance(constraints, dict):
                    errors.append("spec.summary.constraints must be an object/dict")
                else:
                    for k, c in constraints.items():
                        if k not in summary:
                            errors.append(f"summary missing key for constraint: {k}")
                            continue
                        if not isinstance(c, dict):
                            errors.append(f"constraint for {k} must be a dict")
                            continue
                        _check_constraint(str(k), summary.get(k), c, errors)

    return ValidationResult(ok=(len(errors) == 0), errors=errors)


def find_latest_run_dir(root: Union[str, Path], *, glob_pattern: str = "**/summary.json") -> Optional[Path]:
    """
    Returns the directory containing the most-recently-modified summary.json under root.
    """
    root = Path(root)
    if not root.exists():
        return None
    candidates: List[Tuple[float, Path]] = []
    for p in root.glob(glob_pattern):
        if p.is_file() and p.name == "summary.json":
            try:
                candidates.append((p.stat().st_mtime, p.parent))
            except OSError:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]
