from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import sys
import traceback
from io import BytesIO
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


def make_display_data_png(png_bytes: bytes, text: str) -> dict:
    return {
        "output_type": "display_data",
        "metadata": {},
        "data": {
            "image/png": base64.b64encode(png_bytes).decode("ascii"),
            "text/plain": [text],
        },
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
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None

    exec_count = 1
    for idx, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue

        stdout = io.StringIO()
        stderr = io.StringIO()
        display_buffer.clear()
        cell_outputs: list[dict] = []
        code = notebook_source(cell)
        pre_fig_nums = set(plt.get_fignums()) if plt is not None else set()

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
                if plt is not None:
                    new_fig_nums = [num for num in plt.get_fignums() if num not in pre_fig_nums]
                    for fig_num in new_fig_nums:
                        fig = plt.figure(fig_num)
                        buf = BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight")
                        cell_outputs.append(make_display_data_png(buf.getvalue(), f"<Figure size {fig.get_size_inches()[0]:.0f}x{fig.get_size_inches()[1]:.0f}>"))
                    plt.close("all")
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
            if plt is not None:
                new_fig_nums = [num for num in plt.get_fignums() if num not in pre_fig_nums]
                for fig_num in new_fig_nums:
                    fig = plt.figure(fig_num)
                    buf = BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight")
                    cell_outputs.append(make_display_data_png(buf.getvalue(), f"<Figure size {fig.get_size_inches()[0]:.0f}x{fig.get_size_inches()[1]:.0f}>"))
                plt.close("all")
            cell["outputs"] = cell_outputs
            cell["execution_count"] = exec_count
        elif plt is not None:
            plt.close("all")

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
