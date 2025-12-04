from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from PIL import Image


def sort_key(path: Path) -> tuple[int, object]:
    name = path.name
    return (0, int(name)) if name.isdigit() else (1, name)


def load_points(path: Path) -> Mapping[str, dict]:
    with path.open() as f:
        return json.load(f)


def load_entries(results_root: Path) -> list[dict]:
    entries: list[dict] = []
    for run_dir in sorted(results_root.glob("abs_depth_1")):
        if not run_dir.is_dir():
            continue
        for sample_dir in sorted(run_dir.iterdir(), key=sort_key):
            metrics_path = sample_dir / "metrics.json"
            result_path = sample_dir / "result.png"
            if not (metrics_path.is_file() and result_path.is_file()):
                continue
            with metrics_path.open() as f:
                data = json.load(f)
            fid = data.get("fid")
            if fid is None:
                continue
            entries.append(
                {
                    "run": run_dir.name,
                    "sample": sample_dir.name,
                    "fid": float(fid),
                    "image": result_path,
                }
            )
    if not entries:
        raise RuntimeError(
            f"No entries with fid found under {results_root}/abs_depth_*."
        )
    return entries


def crop_center(
    image: Image.Image, center: tuple[float, float], size: int = 400
) -> Image.Image:
    cx, cy = center
    half = size // 2
    left = int(round(cx - half))
    top = int(round(cy - half))
    right = left + size
    bottom = top + size

    # Clamp to image bounds.
    left = max(0, left)
    top = max(0, top)
    right = min(image.width, right)
    bottom = min(image.height, bottom)

    # If clamping shrinks the crop, shift to maintain size where possible.
    if right - left < size:
        shift = size - (right - left)
        left = max(0, left - shift)
        right = min(image.width, left + size)
    if bottom - top < size:
        shift = size - (bottom - top)
        top = max(0, top - shift)
        bottom = min(image.height, top + size)

    return image.crop((left, top, right, bottom))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export highest FID crops centered on target points."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/object_placement"),
        help="Base directory containing abs_depth_*",
    )
    parser.add_argument(
        "--points-path",
        type=Path,
        default=Path("datasets/csc2529/points.json"),
        help="JSON with target points keyed by sample id",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/report/fid_zoom"),
        help="Where to save the crops",
    )
    parser.add_argument(
        "--num", type=int, default=8, help="Number of lowest-FID samples to export"
    )
    parser.add_argument(
        "--crop-size", type=int, default=400, help="Square crop size in pixels"
    )
    args = parser.parse_args()

    entries = load_entries(args.results_root)
    points = load_points(args.points_path)

    entries_sorted = sorted(entries, key=lambda e: e["fid"], reverse=True)
    selected = entries_sorted[: max(1, args.num)]

    for entry in selected:
        point = points.get(entry["sample"])
        if not point:
            print(f"Skipping {entry['sample']}: no point found in {args.points_path}")
            continue
        target = (point["target_x"], point["target_y"])
        image = Image.open(entry["image"])
        crop = crop_center(image, target, size=args.crop_size)

        out_name = f"{entry['run']}_{entry['sample']}_fid_{entry['fid']:.3f}.png"
        out_path = args.output_dir / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out_path)
        print(f"Saved crop to {out_path}")


if __name__ == "__main__":
    main()
