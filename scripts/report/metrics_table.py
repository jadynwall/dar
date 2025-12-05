from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results/object_placement"

# Rows of the output table -> list of result directories to aggregate.
CONFIGS: Dict[str, List[Path]] = {
    "disparity": [RESULTS_ROOT / "disparity"],
    "no scaling": [RESULTS_ROOT / "no_scale"],
    "abs depth": [
        RESULTS_ROOT / "abs_depth_1",
        RESULTS_ROOT / "abs_depth_2",
        RESULTS_ROOT / "abs_depth_3",
    ],
}

TARGET_METRICS = ["fid", "ssim", "mask_iou"]
TABLE_COLUMNS = [
    ("FID", "mean"),
    ("FID", "std"),
    ("SSIM", "mean"),
    ("SSIM", "std"),
    ("Mask IOU", "mean"),
    ("Mask IOU", "std"),
]


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def sort_key(path: Path) -> tuple[int, object]:
    name = path.name
    return (0, int(name)) if name.isdigit() else (1, name)


def iter_metric_files(run_dirs: Iterable[Path]) -> Iterable[Path]:
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            print(f"Skipping {run_dir} (not a directory)")
            continue
        for child in sorted(run_dir.iterdir(), key=sort_key):
            metrics_path = child / "metrics.json"
            if metrics_path.is_file():
                yield metrics_path


def load_metric_records(run_dirs: Iterable[Path]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for metrics_path in iter_metric_files(run_dirs):
        try:
            data = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            print(f"Skipping {metrics_path} (invalid JSON)")
            continue

        if not isinstance(data, dict) or not data:
            continue

        record: Dict[str, float] = {}
        missing = False
        for metric in TARGET_METRICS:
            value = data.get(metric)
            if not is_number(value):
                missing = True
                break
            record[metric] = float(value)

        if missing:
            print(f"Skipping {metrics_path} (missing target metrics)")
            continue

        rows.append(record)

    return pd.DataFrame(rows)


def summarize_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        raise ValueError("No valid metrics available to summarize.")

    summary: Dict[str, float] = {}
    for metric in TARGET_METRICS:
        summary[f"{metric}_mean"] = float(df[metric].mean())
        summary[f"{metric}_std"] = float(df[metric].std())
    return summary


def build_table() -> pd.DataFrame:
    records: Dict[str, Dict[str, float]] = {}
    for row_label, run_dirs in CONFIGS.items():
        df = load_metric_records(run_dirs)
        if df.empty:
            print(f"No valid metrics found for {row_label}; leaving row empty.")
            records[row_label] = {
                f"{metric}_{stat}": float("nan")
                for metric in TARGET_METRICS
                for stat in ("mean", "std")
            }
            continue
        records[row_label] = summarize_metrics(df)

    # Arrange columns as requested (metric -> mean/std).
    column_tuples = TABLE_COLUMNS
    columns = pd.MultiIndex.from_tuples(column_tuples, names=["metric", "stat"])

    table = pd.DataFrame(index=records.keys(), columns=columns, dtype=float)
    for row_label, metrics in records.items():
        for metric in TARGET_METRICS:
            table.loc[
                row_label,
                (metric.upper() if metric != "mask_iou" else "Mask IOU", "mean"),
            ] = metrics.get(f"{metric}_mean")
            table.loc[
                row_label,
                (metric.upper() if metric != "mask_iou" else "Mask IOU", "std"),
            ] = metrics.get(f"{metric}_std")

    return table


def main() -> None:
    table = build_table()
    rounded = table.round(3)
    print("\nAggregate metrics (mean ± std):\n")
    print(rounded.to_string())
    latex = build_simple_latex(rounded, decimals=3)
    print("\nLaTeX table:\n")
    print(latex)


def build_simple_latex(df: pd.DataFrame, decimals: int = 3) -> str:
    """Return a LaTeX tabular string that does not require booktabs."""
    metrics = ["FID", "SSIM", "Mask IOU"]
    # @{} removes the default left/right padding that can offset the table.
    header = r"\begin{tabular}{@{}lrrrrrr@{}}"
    header_line1 = (
        "metric & "
        + " & ".join([r"\multicolumn{2}{c}{" + metric + "}" for metric in metrics])
        + r" \\"
    )
    header_line2 = "& " + " & ".join(["mean & std"] * len(metrics)) + r" \\"

    lines = [header, r"\hline", header_line1, header_line2, r"\hline"]

    for row_label in df.index:
        row_label_tex = str(row_label).replace("_", r"\_")
        values = []
        for metric in metrics:
            values.append(df.loc[row_label, (metric, "mean")])
            values.append(df.loc[row_label, (metric, "std")])
        formatted = " & ".join(
            f"{v:.{decimals}f}" if pd.notna(v) else "--" for v in values
        )
        lines.append(f"{row_label_tex} & {formatted} \\\\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
