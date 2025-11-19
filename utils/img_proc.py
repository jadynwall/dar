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
    for mask in masks:
        if mask[y, x] == 1:
            return mask  # type: ignore
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


def calc_target_mask(
    image: npt.NDArray[np.uint8],
    source_object: npt.NDArray[np.uint8],
    target_px: npt.NDArray[np.uint8],
    source_depth: float,
    target_depth: float,
) -> npt.NDArray[np.bool_]:
    """Returns a mask with shape == image.shape with a box of True values of shape = source_object.shape

    The box is horizontally centered with the target_px and sits directly above it.
    The size of the box is scaled based on the depth.
    """

    scale_factor = source_depth / target_depth
    h, w = (np.array(source_object.shape[:2]) * scale_factor).astype(np.uint8)
    tx, ty = target_px
    target_mask = np.zeros(image.shape[:2], dtype=np.bool_)
    b = 5  # buffer
    target_mask[ty - h - b : ty + b, tx - w // 2 - b : tx + w // 2 + b] = True
    return target_mask
