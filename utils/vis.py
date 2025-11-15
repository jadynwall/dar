import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont
from typing import Optional


def draw_depth_pts(
    image: npt.NDArray,
    source_px: npt.NDArray,
    target_px: npt.NDArray,
    source_depth: Optional[float] = None,
    target_depth: Optional[float] = None,
) -> npt.NDArray:
    image_copy = image.copy()
    radius = 3

    # Set source point to blue
    sx, sy = source_px
    image_copy[sy - radius : sy + radius, sx - radius : sx + radius, :] = 0
    image_copy[sy - radius : sy + radius, sx - radius : sx + radius, 2] = 255

    # Set target point to green
    tx, ty = target_px
    image_copy[ty - radius : ty + radius, tx - radius : tx + radius, :] = 0
    image_copy[ty - radius : ty + radius, tx - radius : tx + radius, 1] = 255

    if source_depth is not None or target_depth is not None:
        pil_image = Image.fromarray(image_copy)
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.load_default()
        height, width = image_copy.shape[:2]

        def _safe_coords(x: int, y: int) -> tuple[int, int]:
            return (
                max(0, min(width - 1, x)),
                max(0, min(height - 1, y)),
            )

        if source_depth is not None:
            sx_text = sx + radius + 5
            sy_text = sy - radius - 10
            if sy_text < 0:
                sy_text = sy + radius + 5
            draw.text(
                _safe_coords(sx_text, sy_text),
                f"S:{float(source_depth):.2f}",
                font=font,
                fill=(0, 0, 255),
            )

        if target_depth is not None:
            tx_text = tx + radius + 5
            ty_text = ty - radius - 10
            if ty_text < 0:
                ty_text = ty + radius + 5
            draw.text(
                _safe_coords(tx_text, ty_text),
                f"T:{float(target_depth):.2f}",
                font=font,
                fill=(0, 255, 0),
            )

        image_copy = np.array(pil_image)

    return image_copy
