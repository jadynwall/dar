"""Generate a collage of abs_depth runs whose metrics.json files are empty."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image, ImageDraw, ImageFont


DEFAULT_ABS_DEPTH_DIR = Path("results/object_placement/abs_depth_1")
DEFAULT_OUTPUT = Path("results/report/no_target_aggregation_collage.png")
POINTS_PATH = Path("datasets/csc2529/points.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a collage of empty metrics runs (no target aggregation)."
    )
    parser.add_argument(
        "--abs-depth-dir",
        type=Path,
        default=DEFAULT_ABS_DEPTH_DIR,
        help=f"path to the abs_depth results directory (default: {DEFAULT_ABS_DEPTH_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output image path for the collage (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=2,
        help="number of rows to arrange the collage into (default: 2)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="margin between tiles/pairs inside the collage (default: 20)",
    )
    return parser.parse_args()


def find_empty_metric_runs(abs_depth_dir: Path) -> List[Path]:
    def sort_key(p: Path):
        name = p.name
        return (0, int(name)) if name.isdigit() else (1, name)

    samples: List[Path] = []
    for child in sorted(abs_depth_dir.iterdir(), key=sort_key):
        if not child.is_dir():
            continue
        metrics_path = child / "metrics.json"
        if not metrics_path.is_file():
            continue
        data = json.loads(metrics_path.read_text())
        if isinstance(data, dict) and not data:
            samples.append(child)
    return samples


def _measure_text(font: ImageFont.ImageFont, text: str) -> tuple[int, int]:
    if hasattr(font, "getbbox"):
        box = font.getbbox(text)
        return box[2] - box[0], box[3] - box[1]
    return font.getsize(text)


def load_points(points_path: Path = POINTS_PATH) -> Dict[str, Dict[str, int]]:
    if not points_path.is_file():
        return {}
    data = json.loads(points_path.read_text())
    points: Dict[str, Dict[str, int]] = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        points[str(key)] = {k: int(v) for k, v in value.items()}
    return points


def _center_crop(
    image: Image.Image, width: int, height: int
) -> tuple[Image.Image, int, int]:
    """Crop the image around its center to the given width/height."""
    left = max(0, (image.width - width) // 2)
    top = max(0, (image.height - height) // 2)
    right = left + width
    bottom = top + height
    return image.crop((left, top, right, bottom)), left, top


def _draw_arrow(
    draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float]
) -> None:
    color = (0, 200, 0)
    width = 6
    head_length = 18
    head_width = 12

    draw.line([start, end], fill=color, width=width)

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return

    ux, uy = dx / length, dy / length
    hx = end[0] - head_length * ux
    hy = end[1] - head_length * uy
    left = (hx + head_width * uy, hy - head_width * ux)
    right = (hx - head_width * uy, hy + head_width * ux)
    draw.polygon([end, left, right], fill=color)


def build_pair_tile(sample_dir: Path, points: Dict[str, Dict[str, int]]) -> Image.Image:
    preview_path = sample_dir / "debug" / "input_preview.png"
    result_path = sample_dir / "result.png"
    if not preview_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"Missing preview/result images for {sample_dir}")

    preview = Image.open(preview_path).convert("RGB")
    result = Image.open(result_path).convert("RGB")

    target_width = min(preview.width, result.width)
    target_height = min(preview.height, result.height)
    preview, preview_left, preview_top = _center_crop(
        preview, target_width, target_height
    )
    result, _, _ = _center_crop(result, target_width, target_height)

    point = points.get(sample_dir.name)
    if point is None and sample_dir.name.isdigit():
        point = points.get(str(int(sample_dir.name)))
    if point:
        source = (point["source_x"] - preview_left, point["source_y"] - preview_top)
        target = (point["target_x"] - preview_left, point["target_y"] - preview_top)
        draw = ImageDraw.Draw(preview)
        _draw_arrow(draw, source, target)

    pair_height = max(preview.height, result.height)
    pair_width = preview.width + result.width
    pair_image = Image.new("RGB", (pair_width, pair_height), "white")

    pair_image.paste(preview, (0, (pair_height - preview.height) // 2))
    pair_image.paste(result, (preview.width, (pair_height - result.height) // 2))

    font = ImageFont.load_default()
    text_width, text_height = _measure_text(font, sample_dir.name)
    label_padding = 4
    label_height = text_height + 2 * label_padding

    labeled = Image.new(
        "RGB", (pair_image.width, pair_image.height + label_height), "white"
    )
    labeled.paste(pair_image, (0, label_height))

    draw = ImageDraw.Draw(labeled)
    text_x = (labeled.width - text_width) // 2
    text_y = (label_height - text_height) // 2
    draw.text((text_x, text_y), sample_dir.name, fill="black", font=font)

    return labeled


def create_collage(
    tiles: Iterable[Image.Image], columns: int, rows: int, padding: int
) -> Image.Image:
    tiles = list(tiles)
    if not tiles:
        raise ValueError("No tiles provided for the collage.")

    tile_width = max(tile.width for tile in tiles)
    tile_height = max(tile.height for tile in tiles)

    collage_width = columns * tile_width + (columns + 1) * padding
    collage_height = rows * tile_height + (rows + 1) * padding
    collage = Image.new("RGB", (collage_width, collage_height), "white")

    for idx, tile in enumerate(tiles):
        row = idx // columns
        col = idx % columns
        if row >= rows:
            break
        x = padding + col * (tile_width + padding) + (tile_width - tile.width) // 2
        y = padding + row * (tile_height + padding) + (tile_height - tile.height) // 2
        collage.paste(tile, (x, y))
    return collage


def main() -> None:
    args = parse_args()
    abs_depth_dir = args.abs_depth_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    samples = find_empty_metric_runs(abs_depth_dir)
    if len(samples) < 2:
        raise SystemExit(f"Need at least 2 empty metrics runs in {abs_depth_dir}")

    samples = random.sample(samples, 2)

    points = load_points()
    tiles = [build_pair_tile(sample, points) for sample in samples]

    rows = max(1, min(args.rows, len(tiles)))
    columns = math.ceil(len(tiles) / rows)

    collage = create_collage(tiles, columns, rows, args.padding)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    collage.save(output_path)
    print(f"Saved collage with {len(tiles)} pairs to {output_path}")


if __name__ == "__main__":
    main()
