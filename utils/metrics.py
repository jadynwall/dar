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
    depth_map: npt.NDArray[np.float32], mask: npt.NDArray[np.bool_]
) -> tuple[float, float]:
    """Returns the mean and std of all the depth values inside the mask."""

    masked_disparity = depth_map[mask]
    if masked_disparity.size == 0:
        return float("nan"), float("nan")
    if masked_disparity.mean() == 0 or masked_disparity.std() == 0:
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
    source_mask: npt.NDArray[np.bool_],
    result_mask: npt.NDArray[np.bool_],
) -> float:
    """Calculates the IOU of the source and observed objects."""

    # Get the mask bounding boxes
    source_pil_mask = PILImageModule.fromarray(
        source_mask.astype(np.uint8) * 255, mode="L"
    )
    source_bbox = source_pil_mask.getbbox()
    source_crop = source_pil_mask.crop(source_bbox)

    result_pil_mask = PILImageModule.fromarray(
        result_mask.astype(np.uint8) * 255, mode="L"
    )
    result_bbox = result_pil_mask.getbbox()
    result_crop = result_pil_mask.crop(result_bbox)

    # Resize the source crop
    source_crop = source_crop.resize(result_crop.size)

    # Compute IOU
    source_crop_np = np.asarray(source_crop)
    result_crop_np = np.asarray(result_crop)
    intersection = np.logical_and(source_crop_np, result_crop_np).sum()
    union = np.logical_or(source_crop_np, result_crop_np).sum()
    return float(intersection / union) if union > 0 else float("nan")


def calc_metrics(
    background_image: npt.NDArray[np.uint8],
    result_image: npt.NDArray[np.uint8],
    source_mask: npt.NDArray[np.bool_],
    target_bbox_mask: npt.NDArray[np.bool_],
    scale_applied: float,
    timings: dict[str, float],
    sam_pipeline: Pipeline,
    depth_pipeline: Pipeline,
) -> dict[str, float]:
    """Metrics metrics metrics."""

    x, y = np.where(np.array(target_bbox_mask, dtype=np.bool_))
    target_center_px = np.floor(np.array(list(zip(y, x))).mean(axis=0)).astype(np.intp)
    target_mask: npt.NDArray[np.bool_] = get_object_mask(
        result_image, target_center_px, sam_pipeline
    )

    bg_disparity: Image = depth_pipeline(  # type: ignore
        PILImageModule.fromarray(background_image)
    )["depth"]
    res_disparity: Image = depth_pipeline(  # type: ignore
        PILImageModule.fromarray(result_image)
    )["depth"]

    z_mean_before, z_std_before = get_mask_depth(np.asarray(bg_disparity), source_mask)
    z_mean_after, z_std_after = get_mask_depth(np.asarray(res_disparity), target_mask)
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
    mask_iou = calc_mask_iou(source_mask, target_mask)

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
