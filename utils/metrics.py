import numpy.typing as npt
import numpy as np
from PIL import Image as PILImageModule
from PIL.Image import Image
import torch
import tempfile
from pathlib import Path
from torch_fidelity import calculate_metrics as fidelity_calculate_metrics
from transformers import Pipeline
from ldm.modules.image_degradation.utils_image import calculate_ssim
from utils.img_proc import get_object_mask
import warnings

# torch-fidelity still uses torch.TypedStorage internally; silence the deprecation noise.
warnings.filterwarnings(
    "ignore", message="TypedStorage is deprecated", category=UserWarning
)


def get_mask_depth(
    depth_map: npt.NDArray[np.float32], mask: npt.NDArray[np.uint8]
) -> tuple[float, float]:
    """Returns the mean and std of all the depth values inside the mask."""

    masked_disparity = depth_map[mask > 0]
    if masked_disparity.size == 0:
        return float("nan"), float("nan")
    mean_depth = 1.0 / float(masked_disparity.mean())
    std_depth = 1.0 / float(masked_disparity.std())
    return mean_depth, std_depth


def compute_fid_score(
    generated_image: npt.NDArray[np.uint8],
    reference_image: npt.NDArray[np.uint8],
) -> float:
    """Persist images to disk and compute FID via torch-fidelity."""

    with tempfile.TemporaryDirectory(
        prefix="fid_gen_", dir="/tmp"
    ) as gen_dir, tempfile.TemporaryDirectory(prefix="fid_ref_", dir="/tmp") as ref_dir:
        gen_path = Path(gen_dir) / "0.png"
        ref_path = Path(ref_dir) / "0.png"
        PILImageModule.fromarray(generated_image).save(gen_path)
        PILImageModule.fromarray(reference_image).save(ref_path)

        fid_metrics = fidelity_calculate_metrics(
            input1=str(gen_dir),
            input2=str(ref_dir),
            cuda=torch.cuda.is_available(),
            fid=True,
            kid=False,
            isc=False,
            verbose=False,
        )
    return float(fid_metrics["frechet_inception_distance"])


def calc_mask_iou(
    source_px: npt.NDArray[np.intp],
    source_image: npt.NDArray[np.uint8],
    target_px: npt.NDArray[np.intp],
    result_image: npt.NDArray[np.uint8],
    sam_pipeline: Pipeline,
) -> float:
    """Calculates the IOU of the source and observed objects.
    Uses the SAM model to segment the observed object at the center of the target mask.
    """

    try:
        source_mask = get_object_mask(source_image, source_px, sam_pipeline)
        observed_mask = get_object_mask(result_image, target_px, sam_pipeline)
    except RuntimeError:
        return float("nan")  # No object was found at the pixel by SAM

    # Get the mask bounding boxes
    source_pil_mask = PILImageModule.fromarray(
        source_mask.astype(np.uint8) * 255, mode="L"
    )
    source_bbox = source_pil_mask.getbbox()
    source_crop = source_pil_mask.crop(source_bbox)

    observed_pil_mask = PILImageModule.fromarray(
        observed_mask.astype(np.uint8) * 255, mode="L"
    )
    observed_bbox = observed_pil_mask.getbbox()
    observed_crop = observed_pil_mask.crop(observed_bbox)

    # Resize the source crop
    source_crop = source_crop.resize(observed_crop.size)

    # Compute IOU
    source_crop_np = np.asarray(source_crop)
    observed_crop_np = np.asarray(observed_crop)
    intersection = np.logical_and(source_crop_np, observed_crop_np).sum()
    union = np.logical_or(source_crop_np, observed_crop_np).sum()
    return float(intersection / union) if union > 0 else float("nan")


def calc_metrics(
    background_image: npt.NDArray[np.uint8],
    result_image: npt.NDArray[np.uint8],
    depth_map: Image,
    source_mask: npt.NDArray[np.bool_],
    target_mask: npt.NDArray[np.bool_],
    scale_applied: float,
    timings: dict[str, float],
    sam_pipeline: Pipeline,
    source_px: npt.NDArray[np.intp],
) -> dict[str, float]:
    """Metrics metrics metrics."""

    depth_np = np.asarray(depth_map, dtype=np.float32)
    if depth_np.shape != source_mask.shape or depth_np.shape != target_mask.shape:
        raise ValueError("Depth map and masks must share the same spatial dimensions.")

    source_mask_u8 = source_mask.astype(np.uint8)
    target_mask_u8 = target_mask.astype(np.uint8)

    z_mean_before, z_std_before = get_mask_depth(depth_np, source_mask_u8)
    z_mean_after, z_std_after = get_mask_depth(depth_np, target_mask_u8)
    z_mean_delta = (
        float(z_mean_after - z_mean_before)
        if np.isfinite(z_mean_before) and np.isfinite(z_mean_after)
        else float("nan")
    )

    if z_mean_after == 0 or not np.isfinite(z_mean_after):
        depth_ratio = float("nan")
    else:
        depth_ratio = float(z_mean_before / z_mean_after)

    area_before = float(np.count_nonzero(source_mask))
    area_after = float(np.count_nonzero(target_mask))
    if area_before == 0:
        scale_observed = float("nan")
    else:
        scale_observed = float(np.sqrt(area_after / area_before))

    if np.isnan(scale_observed) or np.isnan(depth_ratio):
        scale_error = float("nan")
        scale_error_rel = float("nan")
    else:
        scale_error = float(np.abs(scale_observed - depth_ratio))
        scale_error_rel = (
            float(scale_error / depth_ratio) if depth_ratio != 0 else float("nan")
        )

    fid_score = compute_fid_score(result_image, background_image)
    ssim_score = float(calculate_ssim(result_image, background_image))  # type: ignore

    x, y = np.where(np.array(target_mask, dtype=np.bool_))
    target_mask_center = np.floor(np.array(list(zip(y, x))).mean(axis=0)).astype(
        np.intp
    )
    mask_iou = calc_mask_iou(
        source_px=source_px,
        source_image=background_image,
        target_px=target_mask_center,
        result_image=result_image,
        sam_pipeline=sam_pipeline,
    )

    metrics: dict[str, float] = {
        "z_mean_before": z_mean_before,
        "z_std_before": z_std_before,
        "z_mean_after": z_mean_after,
        "z_std_after": z_std_after,
        "z_mean_delta": z_mean_delta,
        "depth_ratio": depth_ratio,
        "area_before": area_before,
        "area_after": area_after,
        "scale_applied": scale_applied,
        "scale_observed": scale_observed,
        "scale_error": scale_error,
        "scale_error_rel": scale_error_rel,
        "fid": fid_score,
        "ssim": ssim_score,
        "mask_iou": mask_iou,
    }
    metrics.update({key: float(value) for key, value in timings.items()})
    return metrics
