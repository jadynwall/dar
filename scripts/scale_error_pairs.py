from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from PIL import Image, ImageDraw


def load_entries(results_root: Path) -> list[dict]:
    entries: list[dict] = []
    for run_dir in sorted(results_root.glob("abs_depth_3")):
        if not run_dir.is_dir():
            continue
        for sample_dir in sorted(
            run_dir.iterdir(),
            key=lambda p: (0, int(p.name)) if p.name.isdigit() else (1, p.name),
        ):
            metrics_path = sample_dir / "metrics.json"
            image_path = sample_dir / "result.png"
            bg_path = sample_dir / "debug" / "input_preview.png"
            if (
                not metrics_path.exists()
                or not image_path.exists()
                or not bg_path.exists()
            ):
                continue
            with metrics_path.open() as f:
                data = json.load(f)
            scale_error = data.get("scale_error")
            if scale_error is None:
                continue
            entries.append(
                {
                    "run": run_dir.name,
                    "sample": sample_dir.name,
                    "scale_error": float(scale_error),
                    "image": image_path,
                    "background": bg_path,
                }
            )
    print(len(entries))
    return entries


def load_points(path: Path) -> Mapping[str, dict]:
    with path.open() as f:
        return json.load(f)


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


def make_pair_collage(
    background: Image.Image, result: Image.Image, output_path: Path, pad: int = 12
) -> None:
    bg = background.convert("RGB")
    res = result.convert("RGB")

    max_h = max(bg.height, res.height)
    width = pad * 3 + bg.width + res.width
    height = pad * 2 + max_h
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))

    y_bg = pad + (max_h - bg.height) // 2
    y_res = pad + (max_h - res.height) // 2
    canvas.paste(bg, (pad, y_bg))
    canvas.paste(res, (pad * 2 + bg.width, y_res))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Saved collage to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create paired background/result images for abs_depth_1 samples above a scale_error threshold."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/object_placement"),
        help="Base directory containing abs_depth_*",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/report/scale_error_pairs"),
        help="Directory to store per-sample collages",
    )
    parser.add_argument(
        "--min-scale-error",
        type=float,
        default=1.0,
        help="Include samples whose scale_error is strictly greater than this value",
    )
    parser.add_argument(
        "--points-path",
        type=Path,
        default=Path("datasets/csc2529/points.json"),
        help="JSON with source/target pixel coordinates keyed by sample id",
    )
    args = parser.parse_args()

    entries = load_entries(args.results_root)
    selected = sorted(
        (e for e in entries if e["scale_error"] > args.min_scale_error),
        key=lambda e: e["scale_error"],
        reverse=True,
    )
    if not selected:
        raise RuntimeError(
            f"No entries in abs_depth_1 have scale_error greater than {args.min_scale_error}."
        )

    points = load_points(args.points_path)

    for entry in selected:
        sample_id = entry["sample"]
        point = points.get(sample_id)
        if not point:
            print(f"Skipping {sample_id}: no points found in {args.points_path}")
            continue

        source = (point["source_x"], point["source_y"])
        target = (point["target_x"], point["target_y"])
        background = draw_arrow(Image.open(entry["background"]), source, target)
        result = Image.open(entry["image"])

        out_name = f"scale_error_{sample_id}_{entry['scale_error']:.3f}.png"
        output_path = args.output_dir / out_name
        make_pair_collage(background, result, output_path)


if __name__ == "__main__":
    main()
