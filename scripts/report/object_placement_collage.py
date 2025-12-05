"""Create a four-column collage for object placement results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Sequence

from PIL import Image, ImageDraw, ImageFont


DEFAULT_ABS_DEPTH_DIR = Path("results/object_placement/abs_depth_1")
DEFAULT_NO_SCALE_DIR = Path("results/object_placement/no_scale")
DEFAULT_DISPARITY_DIR = Path("results/object_placement/disparity")
DEFAULT_OUTPUT = Path("results/report/object_placement_collage.png")
DEFAULT_PADDING = 8
CAPTION_PADDING = 6
RESIZE_DIM = (300, 300)
FONT_SIZE = 12
UPSCALE_FACTOR = 2
POINTS_PATH = Path("datasets/csc2529/points.json")

COLUMN_LABELS = ["Input", "No Scaling", "Disparity", "Absolute Depth"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a conference-style collage of object placement results."
    )
    parser.add_argument(
        "--indices",
        type=str,
        nargs="+",
        help="explicit list of sample indices to include; random sampling is used when omitted",
    )
    return parser.parse_args()


def load_points(points_path: Path = POINTS_PATH) -> dict[str, dict[str, int]]:
    if not points_path.is_file():
        return {}
    data = json.loads(points_path.read_text())
    points: dict[str, dict[str, int]] = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        points[str(key)] = {k: int(v) for k, v in value.items()}
    return points


def _find_cm_font() -> Path | None:
    try:
        import matplotlib  # type: ignore

        fonts_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        for name in ("cmr10.ttf", "cmss10.ttf", "cmb10.ttf"):
            candidate = fonts_dir / name
            if candidate.is_file():
                return candidate
    except Exception:
        return None
    return None


def load_font() -> ImageFont.ImageFont:
    cm_font = _find_cm_font()
    if cm_font:
        try:
            return ImageFont.truetype(str(cm_font), FONT_SIZE)
        except OSError:
            pass
    return ImageFont.load_default()


def _draw_arrow(
    draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float]
) -> None:
    color = (0, 200, 0)
    width = 2
    head_length = 8
    head_width = 4

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


def load_metrics(run_dir: Path, sample_idx: str) -> dict[str, float]:
    metrics_path = run_dir / sample_idx / "metrics.json"
    if not metrics_path.is_file():
        return {}
    try:
        return json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return {}


def _format_metric_value(value: float | None) -> str:
    if value is None:
        return "-"
    try:
        if math.isnan(value):
            return "NaN"
    except TypeError:
        pass
    return f"{value:.3f}"


def format_metrics(metrics: dict[str, float]) -> str:
    fid = _format_metric_value(metrics.get("fid"))
    ssim = _format_metric_value(metrics.get("ssim"))
    mask_iou = _format_metric_value(metrics.get("mask_iou"))
    return f"FID: {fid} | SSIM: {ssim} | MaskIoU: {mask_iou}"


def _sort_key(name: str) -> tuple[int, int | str]:
    return (0, int(name)) if name.isdigit() else (1, name)


def _required_paths(
    abs_depth_dir: Path, no_scale_dir: Path, disparity_dir: Path, sample_idx: str
) -> list[Path]:
    return [
        abs_depth_dir / sample_idx / "debug" / "input_preview.png",
        no_scale_dir / sample_idx / "result.png",
        disparity_dir / sample_idx / "result.png",
        abs_depth_dir / sample_idx / "result.png",
    ]


def find_common_samples(
    abs_depth_dir: Path, no_scale_dir: Path, disparity_dir: Path
) -> List[str]:
    """Find sample ids that have all required images across runs."""
    samples: List[str] = []
    for child in sorted(abs_depth_dir.iterdir(), key=lambda p: _sort_key(p.name)):
        if not child.is_dir():
            continue
        sample_idx = child.name
        if all(
            path.is_file()
            for path in _required_paths(
                abs_depth_dir, no_scale_dir, disparity_dir, sample_idx
            )
        ):
            samples.append(sample_idx)
    return samples


def _annotate_input_with_arrow(
    image: Image.Image,
    sample_idx: str,
    points: dict[str, dict[str, int]],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Image.Image:
    """Overlay a source-to-target arrow on the input image when available."""
    point = points.get(sample_idx)
    if point is None and sample_idx.isdigit():
        point = points.get(str(int(sample_idx)))
    if not point:
        return image

    start = (point["source_x"] * scale_x, point["source_y"] * scale_y)
    end = (point["target_x"] * scale_x, point["target_y"] * scale_y)
    draw = ImageDraw.Draw(image)
    _draw_arrow(draw, start, end)
    return image


def load_sample_images(
    abs_depth_dir: Path,
    no_scale_dir: Path,
    disparity_dir: Path,
    sample_idx: str,
    points: dict[str, dict[str, int]],
) -> List[tuple[Image.Image, str]]:
    """Load the four images (input + three outputs) and captions for a sample."""
    input_path, no_scale_path, disparity_path, abs_depth_path = _required_paths(
        abs_depth_dir, no_scale_dir, disparity_dir, sample_idx
    )
    metrics_no_scale = format_metrics(load_metrics(no_scale_dir, sample_idx))
    metrics_disparity = format_metrics(load_metrics(disparity_dir, sample_idx))
    metrics_abs = format_metrics(load_metrics(abs_depth_dir, sample_idx))

    def _load_and_resize(path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img.resize(RESIZE_DIM, Image.LANCZOS)

    input_img = Image.open(input_path).convert("RGB")
    orig_w, orig_h = input_img.size
    input_img = input_img.resize(RESIZE_DIM, Image.LANCZOS)
    input_img = _annotate_input_with_arrow(
        input_img,
        sample_idx,
        points,
        RESIZE_DIM[0] / max(1, orig_w),
        RESIZE_DIM[1] / max(1, orig_h),
    )

    images = [
        (input_img, ""),
        (_load_and_resize(no_scale_path), metrics_no_scale),
        (_load_and_resize(disparity_path), metrics_disparity),
        (_load_and_resize(abs_depth_path), metrics_abs),
    ]
    return images


def _measure_text(font: ImageFont.ImageFont, text: str) -> tuple[int, int]:
    if hasattr(font, "getbbox"):
        box = font.getbbox(text)
        return box[2] - box[0], box[3] - box[1]
    return font.getsize(text)


def _build_tile(
    image: Image.Image,
    caption: str,
    tile_width: int,
    image_area_height: int,
    caption_height: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    tile_height = image_area_height + caption_height + CAPTION_PADDING
    tile = Image.new("RGB", (tile_width, tile_height), "white")

    x = (tile_width - image.width) // 2
    y = (image_area_height - image.height) // 2
    tile.paste(image, (x, y))

    text_w, text_h = _measure_text(font, caption)
    text_x = (tile_width - text_w) // 2
    text_y = image_area_height + CAPTION_PADDING // 2 + (caption_height - text_h) // 2
    draw = ImageDraw.Draw(tile)
    draw.text((text_x, text_y), caption, fill="black", font=font)
    return tile


def build_collage(
    samples: Sequence[str],
    images_by_sample: dict[str, List[tuple[Image.Image, str]]],
    padding: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    if not samples:
        raise ValueError("No samples provided for the collage.")

    tile_width = max(
        image.width for imgs in images_by_sample.values() for image, _ in imgs
    )
    image_area_height = max(
        image.height for imgs in images_by_sample.values() for image, _ in imgs
    )
    caption_height = max(
        _measure_text(font, caption)[1]
        for imgs in images_by_sample.values()
        for _, caption in imgs
    )

    header_height = max(_measure_text(font, label)[1] for label in COLUMN_LABELS)
    columns = len(COLUMN_LABELS)
    rows = len(samples)

    tile_height = image_area_height + caption_height + CAPTION_PADDING
    collage_width = columns * tile_width + (columns + 1) * padding
    collage_height = header_height + rows * tile_height + (rows + 2) * padding
    collage = Image.new("RGB", (collage_width, collage_height), "white")
    draw = ImageDraw.Draw(collage)

    # Column headers
    for col, label in enumerate(COLUMN_LABELS):
        text_w, text_h = _measure_text(font, label)
        x = padding + col * (tile_width + padding) + (tile_width - text_w) // 2
        y = padding
        draw.text((x, y), label, fill="black", font=font)

    start_y = header_height + 2 * padding
    for row, sample_idx in enumerate(samples):
        row_y = start_y + row * (tile_height + padding)
        for col, (image, caption) in enumerate(images_by_sample[sample_idx]):
            tile = _build_tile(
                image, caption, tile_width, image_area_height, caption_height, font
            )

            col_x = padding + col * (tile_width + padding)
            collage.paste(tile, (col_x, row_y))

    return collage


def main() -> None:
    args = parse_args()

    abs_depth_dir = DEFAULT_ABS_DEPTH_DIR.expanduser().resolve()
    no_scale_dir = DEFAULT_NO_SCALE_DIR.expanduser().resolve()
    disparity_dir = DEFAULT_DISPARITY_DIR.expanduser().resolve()
    output_path = DEFAULT_OUTPUT.expanduser().resolve()
    points = load_points()

    available = find_common_samples(abs_depth_dir, no_scale_dir, disparity_dir)
    if not available:
        raise SystemExit(
            f"No samples found with all required images under {abs_depth_dir} / {no_scale_dir} / {disparity_dir}"
        )

    if not args.indices:
        raise SystemExit("Please provide --indices to select samples.")

    missing = [s for s in args.indices if s not in available]
    if missing:
        raise SystemExit(
            f"Requested indices not found or missing required files: {missing}"
        )

    selected = list(args.indices)

    images_by_sample: dict[str, List[Image.Image]] = {}
    for sample_idx in selected:
        images_by_sample[sample_idx] = load_sample_images(
            abs_depth_dir, no_scale_dir, disparity_dir, sample_idx, points
        )

    font = load_font()
    collage = build_collage(selected, images_by_sample, DEFAULT_PADDING, font)
    if UPSCALE_FACTOR > 1:
        collage = collage.resize(
            (
                int(collage.width * UPSCALE_FACTOR),
                int(collage.height * UPSCALE_FACTOR),
            ),
            Image.LANCZOS,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    collage.save(output_path)
    print(f"Saved collage with {len(selected)} samples to {output_path}")


if __name__ == "__main__":
    main()
