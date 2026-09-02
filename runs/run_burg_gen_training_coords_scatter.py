from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runs.run_eql_joint_training import RunConfig, load_dataset
from utils.data_prep_utils import PDETrainDataset
from utils.dataset_sampling_plotting import save_space_time_coordinate_scatter


@dataclass
class ScatterConfig:
    dataset_name: str = "burg_gen"
    noise_level: float = 0.0
    seed: int = 0
    stride_t: int = 2
    stride_x: int = 2
    output_root: str | None = None
    output_tag: str | None = None


def _extract_json_block(markdown_text: str) -> dict:
    start_token = "```json"
    end_token = "```"
    start = markdown_text.find(start_token)
    if start < 0:
        raise ValueError("Could not find JSON config block in research_state.md")
    start += len(start_token)
    end = markdown_text.find(end_token, start)
    if end < 0:
        raise ValueError("Could not find end of JSON config block in research_state.md")
    return json.loads(markdown_text[start:end].strip())


def load_active_scatter_config() -> ScatterConfig:
    research_state_path = ROOT / "research_state.md"
    if research_state_path.exists():
        cfg_data = _extract_json_block(research_state_path.read_text())
        if cfg_data.get("dataset_name") == "burg_gen":
            return ScatterConfig(
                dataset_name="burg_gen",
                noise_level=float(cfg_data.get("noise_level", 0.0)),
                seed=int(cfg_data.get("seed", 0)),
                stride_t=int(cfg_data.get("stride_t", RunConfig.stride_t)),
                stride_x=int(cfg_data.get("stride_x", RunConfig.stride_x)),
            )

    return ScatterConfig(
        dataset_name="burg_gen",
        stride_t=int(RunConfig.stride_t),
        stride_x=int(RunConfig.stride_x),
    )


def default_output_dir(cfg: ScatterConfig) -> Path:
    if cfg.output_root:
        return Path(cfg.output_root)

    tag = cfg.output_tag
    if not tag:
        tag = "training_coords_scatter_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    return ROOT / "Datasets" / "data" / "processed" / "burg_gen" / "artifacts" / tag


def build_sampling_summary(
    *,
    cfg: ScatterConfig,
    dataset_loader: str,
    t_grid: np.ndarray,
    x_grid: np.ndarray,
    dataset: PDETrainDataset,
    png_path: Path,
    pdf_path: Path,
) -> dict:
    sampled_t_count = int(np.arange(t_grid.size)[:: dataset.stride_t].size)
    sampled_x_count = int(np.arange(x_grid.size)[:: dataset.stride_x].size)
    return {
        "artifact_type": "burg_gen_training_coordinate_scatter",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "dataset_loader": dataset_loader,
        "coordinate_space": "physical",
        "original_grid": {
            "Nt": int(t_grid.size),
            "Nx": int(x_grid.size),
        },
        "sampling": {
            "stride_t": int(dataset.stride_t),
            "stride_x": int(dataset.stride_x),
            "sampled_temporal_count": sampled_t_count,
            "sampled_spatial_count": sampled_x_count,
            "total_sampled_observations": int(dataset.t_train.size),
        },
        "artifacts": {
            "scatter_png": str(png_path),
            "scatter_pdf": str(pdf_path),
            "summary_json": str(png_path.parent / "summary.json"),
            "research_state_md": str(png_path.parent / "research_state.md"),
        },
    }


def write_research_state(out_dir: Path, summary: dict) -> None:
    cfg = summary["config"]
    sampling = summary["sampling"]
    lines = [
        "# Research State",
        "",
        "## Experiment objective",
        "Generate a presentation-ready scatter plot of the actual burg_gen training coordinates after applying the active training stride.",
        "",
        "## Exact configuration",
        "```json",
        json.dumps(cfg, indent=2),
        "```",
        "",
        "## Dataset",
        f"- dataset: `{cfg['dataset_name']}`",
        f"- loader: `{summary['dataset_loader']}`",
        "- coordinate space used for plotting: physical x/t coordinates",
        "",
        "## Sampling summary",
        f"- original spatial grid size Nx: {summary['original_grid']['Nx']}",
        f"- original temporal grid size Nt: {summary['original_grid']['Nt']}",
        f"- stride_x: {sampling['stride_x']}",
        f"- stride_t: {sampling['stride_t']}",
        f"- sampled spatial count: {sampling['sampled_spatial_count']}",
        f"- sampled temporal count: {sampling['sampled_temporal_count']}",
        f"- total sampled (x,t) observations: {sampling['total_sampled_observations']}",
        "",
        "## Artifact directory",
        f"`{out_dir}`",
        "",
        "## Summary file",
        f"- summary: `{out_dir / 'summary.json'}`",
    ]
    (out_dir / "research_state.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a burg_gen training-coordinate scatter artifact.")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--output-tag", type=str, default=None)
    args = parser.parse_args()

    cfg = load_active_scatter_config()
    if args.output_root is not None:
        cfg.output_root = args.output_root
    if args.output_tag is not None:
        cfg.output_tag = args.output_tag

    if cfg.dataset_name != "burg_gen":
        raise ValueError(f"This script only supports burg_gen, got '{cfg.dataset_name}'.")

    dataset_loader = "runs/run_eql_joint_training.py::load_dataset -> Datasets/data/processed/burg_gen/burg_gen.py::solve_burgers"
    print(f"Dataset loader: {dataset_loader}")
    print(f"Applying strides: stride_t={cfg.stride_t}, stride_x={cfg.stride_x}")

    t_grid, x_grid, u_grid, _derivative_mode, _metadata = load_dataset(cfg.dataset_name)
    dataset = PDETrainDataset(
        t_grid=t_grid,
        x_grid=x_grid,
        u_grid=u_grid,
        stride_t=cfg.stride_t,
        stride_x=cfg.stride_x,
        noise_level=cfg.noise_level,
        seed=cfg.seed,
        normalize=True,
    )

    out_dir = default_output_dir(cfg)
    png_path = out_dir / "space_time_training_coords_scatter.png"
    pdf_path = out_dir / "space_time_training_coords_scatter.pdf"

    save_space_time_coordinate_scatter(
        x_coords=dataset.x_train,
        t_coords=dataset.t_train,
        png_path=png_path,
        vector_path=pdf_path,
        dpi=300,
    )

    summary = build_sampling_summary(
        cfg=cfg,
        dataset_loader=dataset_loader,
        t_grid=np.asarray(t_grid),
        x_grid=np.asarray(x_grid),
        dataset=dataset,
        png_path=png_path,
        pdf_path=pdf_path,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_research_state(out_dir, summary)

    sampling = summary["sampling"]
    print(f"original spatial grid size Nx: {summary['original_grid']['Nx']}")
    print(f"original temporal grid size Nt: {summary['original_grid']['Nt']}")
    print(f"stride_x: {sampling['stride_x']}")
    print(f"stride_t: {sampling['stride_t']}")
    print(f"sampled spatial count: {sampling['sampled_spatial_count']}")
    print(f"sampled temporal count: {sampling['sampled_temporal_count']}")
    print(f"total sampled (x,t) observations: {sampling['total_sampled_observations']}")
    print(f"artifact png: {png_path}")
    print(f"artifact pdf: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
