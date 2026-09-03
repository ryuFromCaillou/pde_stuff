from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path


def notebook_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def as_text(value) -> str:
    if value is None:
        return ""
    if hasattr(value, "to_string"):
        try:
            return value.to_string()
        except Exception:
            pass
    return repr(value)


def make_stream(name: str, text: str) -> dict:
    return {"output_type": "stream", "name": name, "text": [text]}


def make_execute_result(text: str) -> dict:
    return {
        "output_type": "execute_result",
        "metadata": {},
        "data": {"text/plain": [text]},
        "execution_count": None,
    }


def make_error(exc: BaseException) -> dict:
    return {
        "output_type": "error",
        "ename": type(exc).__name__,
        "evalue": str(exc),
        "traceback": traceback.format_exc().splitlines(),
    }


def run_notebook(notebook_path: Path, update_from_cell: int) -> None:
    notebook = json.loads(notebook_path.read_text())
    env: dict = {"__name__": "__main__"}
    display_buffer: list = []
    repo_root = Path.cwd()

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    def display(obj=None):
        display_buffer.append(obj)

    env["display"] = display

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

    exec_count = 1
    for idx, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue

        stdout = io.StringIO()
        stderr = io.StringIO()
        display_buffer.clear()
        cell_outputs: list[dict] = []
        code = notebook_source(cell)

        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(code, f"{notebook_path.name}:cell_{idx}", "exec"), env)
        except Exception as exc:
            if idx >= update_from_cell:
                out = stdout.getvalue()
                err = stderr.getvalue()
                if out:
                    cell_outputs.append(make_stream("stdout", out))
                if err:
                    cell_outputs.append(make_stream("stderr", err))
                cell_outputs.extend(
                    make_execute_result(as_text(obj)) for obj in display_buffer if as_text(obj)
                )
                cell_outputs.append(make_error(exc))
                cell["outputs"] = cell_outputs
                cell["execution_count"] = exec_count
                notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")
            raise

        if idx >= update_from_cell:
            out = stdout.getvalue()
            err = stderr.getvalue()
            if out:
                cell_outputs.append(make_stream("stdout", out))
            if err:
                cell_outputs.append(make_stream("stderr", err))
            cell_outputs.extend(
                make_execute_result(as_text(obj)) for obj in display_buffer if as_text(obj)
            )
            cell["outputs"] = cell_outputs
            cell["execution_count"] = exec_count

        exec_count += 1

    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--update-from-cell", type=int, default=0)
    args = parser.parse_args()
    run_notebook(args.notebook, update_from_cell=args.update_from_cell)


if __name__ == "__main__":
    main()
