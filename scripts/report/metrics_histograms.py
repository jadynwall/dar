from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

pd.set_option("display.max_columns", None)
plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

results_root = Path("results/object_placement")
out_dir = Path("results/report/histograms")
out_dir.mkdir(exist_ok=True)

# Aggregate metrics across multiple runs of the same dataset and average them.
runs = ["abs_depth_1", "abs_depth_2", "abs_depth_3"]
samples = 35

# sample index -> [metrics.json, ...]
sample_to_metrics: dict[str, list[dict]] = {}
invalid_samples: set[str] = set()

for run in runs:
    base = results_root / run
    if not base.exists():
        print(f"Skipping {base} (missing directory)")
        continue

    sample_paths = sorted(
        [p for p in base.iterdir() if p.is_dir()], key=lambda p: int(p.name)
    )
    sample_paths = sample_paths[:samples]
    for sample_index in sample_paths:
        metrics_path = sample_index / "metrics.json"
        if not metrics_path.exists():
            print(f"Skipping {sample_index} (no metrics.json)")
            continue
        with metrics_path.open() as f:
            data = json.load(f)
        if not isinstance(data, dict) or not data:
            print(
                f"Skipping {sample_index} (empty or invalid metrics); dropping sample from all runs"
            )
            invalid_samples.add(sample_index.name)
            continue
        if sample_index.name in invalid_samples:
            # Already marked invalid from another run.
            continue
        sample_to_metrics.setdefault(sample_index.name, []).append(data)

if not sample_to_metrics:
    raise RuntimeError("No metrics found in abs_depth runs.")

# Remove any samples marked invalid across runs.
for key in list(sample_to_metrics.keys()):
    if key in invalid_samples:
        sample_to_metrics.pop(key, None)

records = []
for sample_index in sorted(sample_to_metrics.keys(), key=lambda name: int(name)):
    measurements = sample_to_metrics[sample_index]
    metrics_df = pd.DataFrame(measurements)
    averaged = metrics_df.mean(numeric_only=True).to_dict()
    averaged["sample"] = sample_index
    averaged["sources"] = len(measurements)
    if len(measurements) < len(runs):
        missing = len(runs) - len(measurements)
        print(
            f"Sample {sample_index}: missing {missing} source(s); averaging over {len(measurements)}"
        )
    records.append(averaged)

print(f"Total valid samples: {len(records)}")

df = (
    pd.DataFrame(records)
    .set_index("sample")
    .sort_index(key=lambda idx: idx.astype(int))
)
df.index.name = "sample"

metrics = [col for col in df.columns.tolist() if col != "sources"]
lime_color = "#a4eb34"


def pretty_label(name: str) -> str:
    label_map = {
        "fid": "FID",
        "ssim": "SSIM",
        "iou": "IOU",
    }
    key = name.lower()
    return label_map.get(key, name.replace("_", " ").title())


def safe_name(name: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return keep.strip("_").lower() or "metric"


for idx, metric in enumerate(metrics):
    series = df[metric].astype(float)
    bins = min(12, max(5, len(series) // 3))

    fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
    ax.hist(series.values, bins=bins, color=lime_color, edgecolor="white", alpha=0.85)

    ax.set_title(f"{pretty_label(metric)} Distribution")
    ax.set_xlabel(pretty_label(metric))
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    filename = out_dir / f"{safe_name(metric)}_hist.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {filename}")
