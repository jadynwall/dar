from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


STAGES = [
    ("time_full_depth", "Depth"),
    ("time_object_prep", "Object Prep"),
    ("time_scene_masks", "Scene Masks"),
    ("time_mpi_build", "MPI Build"),
    ("time_null_text", "Null Text"),
    ("time_diffusion", "Diffusion"),
]


def sort_key(path: Path) -> tuple[int, object]:
    name = path.name
    return (0, int(name)) if name.isdigit() else (1, name)


def iter_metric_files(run_dirs: Iterable[Path]) -> Iterable[Path]:
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        for child in sorted(run_dir.iterdir(), key=sort_key):
            metrics_path = child / "metrics.json"
            if metrics_path.is_file():
                yield metrics_path


def load_stage_means(run_dir: Path) -> Dict[str, float]:
    values: Dict[str, List[float]] = {key: [] for key, _ in STAGES}
    for metrics_path in iter_metric_files([run_dir]):
        try:
            data = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        for key, _label in STAGES:
            val = data.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                values[key].append(float(val))
    return {key: float(np.mean(vals)) for key, vals in values.items() if vals}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot average runtime breakdown per abs_depth_* run."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/object_placement"),
        help="Directory containing abs_depth_*",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/report/runtime_breakdown.png"),
        help="Where to save the plot",
    )
    args = parser.parse_args()

    runs = sorted(
        [p for p in args.results_root.glob("abs_depth_*") if p.is_dir()], reverse=True
    )
    if not runs:
        raise RuntimeError(f"No abs_depth_* runs found under {args.results_root}")

    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.35,
        }
    )

    run_labels = [f"Run {idx}" for idx in range(1, len(runs) + 1)]
    stage_means_per_run = [load_stage_means(run) for run in runs]

    x = np.arange(len(run_labels))
    width = 0.65
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)

    colors = mpl.cm.viridis(np.linspace(0.1, 0.9, len(STAGES)))
    bottoms = np.zeros(len(run_labels))

    for (key, label), color in zip(STAGES, colors):
        heights = np.array(
            [run_means.get(key, 0.0) for run_means in stage_means_per_run]
        )
        ax.bar(
            x,
            heights,
            width,
            bottom=bottoms,
            label=label,
            color=color,
            edgecolor="white",
        )
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels(run_labels)
    ax.set_ylabel("Time (s)")
    ax.set_title("Runtime Breakdown per Run (mean per sample)")
    ax.legend(title="Stage", ncol=3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved runtime plot to {args.output}")


if __name__ == "__main__":
    main()
