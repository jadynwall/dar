from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image, ImageDraw


def sort_key(path: Path) -> tuple[int, object]:
    name = path.name
    return (0, int(name)) if name.isdigit() else (1, name)


def load_entries(results_root: Path) -> list[dict]:
    """Collect entries from abs_depth_1 with mask_iou available."""
    run_dir = results_root / "abs_depth_1"
    if not run_dir.is_dir():
        raise RuntimeError(f"Missing directory: {run_dir}")

    entries: list[dict] = []
    for sample_dir in sorted(run_dir.iterdir(), key=sort_key):
        metrics_path = sample_dir / "metrics.json"
        result_path = sample_dir / "result.png"
        target_path = sample_dir / "debug" / "target.png"
        preview_path = sample_dir / "debug" / "input_preview.png"
        if not (
            metrics_path.is_file()
            and result_path.is_file()
            and target_path.is_file()
            and preview_path.is_file()
        ):
            continue

        with metrics_path.open() as f:
            data = json.load(f)
        mask_iou = data.get("mask_iou")
        if mask_iou is None:
            continue

        entries.append(
            {
                "sample": sample_dir.name,
                "mask_iou": float(mask_iou),
                "result": result_path,
                "target": target_path,
                "preview": preview_path,
            }
        )
    return entries


def make_pair_collage(
    images: Iterable[Image.Image], output_path: Path, pad: int = 12
) -> None:
    a, b = list(images)
    left = a.convert("RGB")
    right = b.convert("RGB")

    max_h = max(left.height, right.height)
    width = pad * 3 + left.width + right.width
    height = pad * 2 + max_h
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))

    y_left = pad + (max_h - left.height) // 2
    y_right = pad + (max_h - right.height) // 2
    canvas.paste(left, (pad, y_left))
    canvas.paste(right, (pad * 2 + left.width, y_right))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Saved collage to {output_path}")


def draw_arrow(
    image: Image.Image, source: tuple[float, float], target: tuple[float, float]
) -> Image.Image:
    img = image.convert("RGBA")
    draw = ImageDraw.Draw(img)
    color = (0, 255, 0, 255)
    width = max(3, round(min(img.size) * 0.005))

    draw.line([source, target], fill=color, width=width)

    dx, dy = target[0] - source[0], target[1] - source[1]
    length = max((dx**2 + dy**2) ** 0.5, 1e-6)
    ux, uy = dx / length, dy / length
    head_len = min(30, max(12, length * 0.08))
    head_width = head_len * 0.6
    base_x = target[0] - ux * head_len
    base_y = target[1] - uy * head_len
    perp_x, perp_y = -uy, ux
    p1 = (target[0], target[1])
    p2 = (base_x + perp_x * head_width * 0.5, base_y + perp_y * head_width * 0.5)
    p3 = (base_x - perp_x * head_width * 0.5, base_y - perp_y * head_width * 0.5)
    draw.polygon([p1, p2, p3], fill=color)
    return img.convert("RGB")


def load_points(path: Path) -> Mapping[str, dict]:
    with path.open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export pairs with mask IoU below a threshold."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/object_placement"),
        help="Directory containing abs_depth_1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/report/mask_iou_pairs"),
        help="Where to save the side-by-side outputs",
    )
    parser.add_argument(
        "--max-mask-iou",
        type=float,
        default=1.0,
        help="Include samples whose mask_iou is strictly less than this value",
    )
    parser.add_argument(
        "--points-path",
        type=Path,
        default=Path("datasets/csc2529/points.json"),
        help="JSON file with source/target pixel coordinates keyed by sample id",
    )
    args = parser.parse_args()

    entries = load_entries(args.results_root)
    selected = [e for e in entries if e["mask_iou"] < args.max_mask_iou]
    if not selected:
        raise RuntimeError(f"No entries found with mask_iou < {args.max_mask_iou}")
    # Choose lowest and highest IoU among the filtered set.
    selected = sorted(selected, key=lambda e: e["mask_iou"])
    selected = [selected[0], selected[-1]] if len(selected) > 1 else [selected[0]]

    points = load_points(args.points_path)

    for entry in selected:
        sample_id = entry["sample"]
        point = points.get(sample_id)
        if not point:
            print(f"Skipping {sample_id}: no points found in {args.points_path}")
            continue

        source = (point["source_x"], point["source_y"])
        target = (point["target_x"], point["target_y"])
        target_img = draw_arrow(Image.open(entry["preview"]), source, target)
        result_img = Image.open(entry["result"])
        out_name = f"{entry['sample']}_mask_iou_{entry['mask_iou']:.3f}.png"
        out_path = args.output_dir / out_name
        make_pair_collage([target_img, result_img], out_path)


if __name__ == "__main__":
    main()
