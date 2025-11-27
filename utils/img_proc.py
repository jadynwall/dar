import numpy as np
import numpy.typing as npt
from transformers import Pipeline
from PIL import Image as PILImageModule
from PIL.Image import Image


def get_object_mask(
    image: npt.NDArray[np.uint8], px: npt.NDArray[np.intp], sam_pipeline: Pipeline
) -> npt.NDArray[np.bool_]:
    """Returns the first mask from the SAM pipeline that intersects px."""

    outputs = sam_pipeline(PILImageModule.fromarray(image), points_per_batch=64)
    masks = outputs["masks"]  # type: ignore
    x, y = px
    largest_mask = None
    for mask in masks:
        if mask[y, x] == 1 and (
            largest_mask is None or np.sum(mask) > np.sum(largest_mask)
        ):
            largest_mask = mask

    if largest_mask is not None:
        return largest_mask  # type: ignore
    else:
        raise RuntimeError("SAM couldn't find an object at the given point")


def extract_masked_object(
    image: npt.NDArray[np.uint8], mask: npt.NDArray[np.bool_]
) -> npt.NDArray[np.uint8]:
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


def calc_target_bbox_mask(
    image: npt.NDArray[np.uint8],
    source_object: npt.NDArray[np.uint8],
    target_px: npt.NDArray[np.uint8],
    scale_factor: float,
) -> npt.NDArray[np.bool_]:
    """Returns a mask with shape == image.shape with a bbox of True values of shape = source_object.shape

    The box is horizontally centered with the target_px and sits directly above it.
    The size of the box is scaled based on the depth.
    """

    # Use signed ints so negative offsets clamp correctly near image borders.
    h, w = (np.array(source_object.shape[:2]) * scale_factor).astype(np.int64)
    tx, ty = target_px
    target_mask = np.zeros(image.shape[:2], dtype=np.bool_)
    b = 5  # buffer

    bbox_top = max(0, int(ty - h - b))
    bbox_bot = min(target_mask.shape[0], int(ty + b))
    bbox_left = max(0, int(tx - w // 2 - b))
    bbox_right = min(target_mask.shape[1], int(tx + w // 2 + b))
    target_mask[bbox_top:bbox_bot, bbox_left:bbox_right] = True
    return target_mask
