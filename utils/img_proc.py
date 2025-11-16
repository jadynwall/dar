import numpy as np
import numpy.typing as npt
from transformers import Pipeline
from PIL import Image as PILImageModule
from PIL.Image import Image


def get_object_mask(
    image: npt.NDArray, px: npt.NDArray, sam_pipeline: Pipeline
) -> npt.NDArray:
    """Returns the first mask from the SAM pipeline that intersects px."""

    outputs = sam_pipeline(PILImageModule.fromarray(image), points_per_batch=64)
    masks = outputs["masks"]
    x, y = px
    for mask in masks:
        if mask[y, x] == 1:
            return mask
    else:
        raise RuntimeError("SAM couldn't find an object at the given point")


def extract_masked_object(image: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    assert image.shape[:2] == mask.shape
    assert image.shape[2] == 3
    assert mask.ndim == 2

    pil_mask_array = (mask > 0).astype(np.uint8) * 255
    pil_mask: Image = PILImageModule.fromarray(pil_mask_array, mode="L")
    extracted: Image = PILImageModule.fromarray(image).convert("RGBA")

    bbox = pil_mask.getbbox()

    extracted.putalpha(pil_mask)
    extracted = extracted.crop(bbox)
    return np.array(extracted, dtype=np.uint8)


def calc_target_mask(
    image: npt.NDArray,
    source_object: npt.NDArray,
    target_px: npt.NDArray,
    source_depth: float,
    target_depth: float,
) -> npt.NDArray:
    """Returns a mask with shape == image.shape with a box of True values of shape = source_object.shape

    The box is horizontally centered with the target_px and sits directly above it.
    The size of the box is scaled based on the depth.
    """

    scale_factor = source_depth / target_depth
    h, w = (np.array(source_object.shape[:2]) * scale_factor).astype(np.uint8)
    tx, ty = target_px
    target_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    b = 5  # buffer
    target_mask[ty - h - b : ty + b, tx - w // 2 - b : tx + w // 2 + b] = 255
    return target_mask


def get_mask_depth(
    depth_map: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8]
) -> tuple[float, float]:
    """Returns the mean and std of all the depth values inside the mask."""

    masked_depth = depth_map[mask > 0]
    mean_depth = float(masked_depth.mean())
    std_depth = float(masked_depth.std())
    return mean_depth, std_depth
