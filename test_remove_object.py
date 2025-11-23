from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import einops
import numpy as np
from datasets.csc2529 import AnyDoorCollage, CSC2529Dataset, create_anydoor_collage
from torch.utils.data import DataLoader
from datetime import datetime
from pytorch_lightning import seed_everything
import os
import time
from utils.mpi.preprocess import get_depth_and_sam_mask
from cldm.ddim_hacked_mpi_featguidance import DDIMSampler
from utils.mpi.mpi import get_mpi_rgb_and_alpha
from cldm.model import create_model, load_state_dict
from PIL import Image as PILImageModule
from PIL.Image import Image
import torch
from omegaconf import OmegaConf
from src.featglac import FeatureGuidance
import numpy.typing as npt
from diffusers.image_processor import VaeImageProcessor
import attrs
from typing import Any
from transformers import pipeline
import pickle
from utils.vis import draw_depth_pts
from utils.img_proc import (
    get_object_mask,
    extract_masked_object,
    calc_target_mask,
)
import json
import warnings
from utils.metrics import calc_metrics
from diffusers import AutoPipelineForInpainting

def get_bbox_from_point(
    image: npt.NDArray[np.uint8],
    point: npt.NDArray[np.intp],
    box_size: int,
) -> tuple[int, int, int, int]:
    """Get bounding box (y1, y2, x1, x2) of size box_size centered at point."""
    H, W, _ = image.shape
    half_size = box_size // 2
    x, y = point
    y1 = max(0, y - half_size)
    y2 = min(H, y + half_size)
    x1 = max(0, x - half_size)
    x2 = min(W, x + half_size)

    # Adjust box if it goes out of image bounds
    if y2 - y1 < box_size:
        if y1 == 0:
            y2 = min(H, y1 + box_size)
        else:
            y1 = max(0, y2 - box_size)
    if x2 - x1 < box_size:
        if x1 == 0:
            x2 = min(W, x1 + box_size)
        else:
            x1 = max(0, x2 - box_size)

    return (y1, y2, x1, x2)


def crop_with_bbox(
    image: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.uint8],
    bbox: tuple[int, int, int, int],
) -> npt.NDArray[np.uint8]:
    """Crop a square region around a given point in the image."""
    y1, y2, x1, x2 = bbox

    cropped_image = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    return cropped_image, cropped_mask


def uncrop_with_bbox(
    cropped_image: npt.NDArray[np.uint8],
    original_image: npt.NDArray[np.uint8],
    bbox: tuple[int, int, int, int],
) -> npt.NDArray[np.uint8]:
    """Paste cropped region back to original image."""
    y1, y2, x1, x2 = bbox
    original_image[y1:y2, x1:x2] = cropped_image
    return original_image


def remove_object(
    background_image,
    object_mask
) -> npt.NDArray[np.uint8]:
    """Remove object from background image using inpainting."""
    sam_mask = object_mask.astype(np.uint8)*255
    kernel = np.ones((10,10), np.uint8)
    sam_mask = cv2.dilate(sam_mask.copy(), kernel, iterations=1)
    sam_mask = cv2.GaussianBlur(sam_mask.copy(), (5, 5), 0)
    mask_pil = PILImageModule.fromarray(sam_mask).convert("L")
    init_image = PILImageModule.fromarray(background_image)

    pipeline = AutoPipelineForInpainting.from_pretrained(
        "weights/stable-diffusion-2-1-base", torch_dtype=torch.float16, variant="fp16", local_files_only=True
    )

    pipeline = pipeline.to(0)
    pipeline.enable_model_cpu_offload()

    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device)
    image = pipeline(
        prompt="high quality, high resolution, indoor scene",
        negative_prompt="object",
        image=init_image,
        mask_image=mask_pil,
        generator=generator
    ).images[0]
    return np.array(image)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Place an object in scene.")
    parser.add_argument(
        "--dataset-base-dir",
        default="datasets/csc2529",
        help="Path to the images",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Outputs intermediated debug outputs results/",
    )
    parser.add_argument(
        "--skip-diffusion",
        action="store_true",
        help="Just generate the collage and skip generating final image",
    )
    parser.add_argument(
        "--sample-indexes",
        nargs="+",
        type=int,
        default=None,
        help="When provided, only run these dataset indices.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    RESULTS_BASE_DIR = "results/remove_object/{}"

    timestamped_results_dir: Path = Path(
        RESULTS_BASE_DIR.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    )

    args = parse_args()
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset: CSC2529Dataset = CSC2529Dataset(Path(args.dataset_base_dir))
    loader = DataLoader(dataset, collate_fn=lambda x: x[0])  # type: ignore

    # load SAM
    sam_pipeline = pipeline("mask-generation", model="facebook/sam-vit-huge", device=0)

    filter_indices = set(args.sample_indexes) if args.sample_indexes else None

    for dataset_idx, background_image, source_px, target_px in loader:
        if filter_indices and dataset_idx not in filter_indices:
            continue
        results_dir = timestamped_results_dir / str(dataset_idx)
        os.makedirs(results_dir)

        object_mask = get_object_mask(background_image, source_px, sam_pipeline)

        '''
        Remove object
        '''
        # Get bbox around source point
        bbox = get_bbox_from_point(
            background_image, source_px, box_size=512
        )
        print("Bounding box for inpainting:", bbox)

        # Crop image and mask
        background_image_cropped, object_mask_cropped = crop_with_bbox(
            background_image, object_mask, bbox
        )
        
        cv2.imwrite(str(results_dir / "cropped_image.png"), background_image_cropped)
        cv2.imwrite(str(results_dir / "cropped_mask.png"), object_mask_cropped * 255)

        # Remove object from cropped image
        background_image_cropped = remove_object(
            background_image_cropped.copy(), object_mask_cropped.copy()
        )
        cv2.imwrite(str(results_dir / "cropped_inpainted_result.png"), background_image_cropped)
        
        # Paste cropped image back to original image
        result_image = uncrop_with_bbox(
            background_image_cropped,
            background_image,
            bbox
        )
        result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(results_dir / "inpainted_result.png"), result_bgr)