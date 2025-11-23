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

# HF pipelines emit a noisy FutureWarning about clean_up_tokenization_spaces; silence it.
warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set",
    category=FutureWarning,
)

# Results are stored in a timestamped folder:
RESULTS_BASE_DIR = "results/new_object_placement/{}"


def load_anydoor() -> torch.nn.Module:
    config = OmegaConf.load("./configs/inference.yaml")
    model = create_model(config.config_file)
    model.load_state_dict(load_state_dict(config.pretrained_model))
    return model

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
    sam_mask = object_mask.astype(np.uint8) * 255
    kernel = np.ones((10,10), np.uint8)
    sam_mask = cv2.dilate(sam_mask.copy(), kernel, iterations=1)
    sam_mask = cv2.GaussianBlur(sam_mask.copy(), (5, 5), 0)
    sam_mask = cv2.dilate(sam_mask.copy(), kernel, iterations=1)
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
        generator=generator,
    ).images[0]
    return np.array(image)


def build_mpi_masks(
    background_crop: npt.NDArray[np.uint8],
    depth_image: PILImageModule.Image,
    target_depth: float,
    device: torch.device,
    debug_dir: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[npt.NDArray[np.uint8]]]:
    """Generate background/foreground MPI masks along with the high-res originals."""

    depth_partition: list[tuple[float, float]] = [
        (0, target_depth),
        (target_depth, 300),
    ]
    _, layered_alpha = get_mpi_rgb_and_alpha(
        np.array(background_crop), np.array(depth_image), depth_partition
    )

    if debug_dir is not None:
        bg_np = np.array(background_crop)
        cv2.imwrite(
            str(debug_dir / "mpi_foreground_alpha.png"),
            bg_np * layered_alpha[1][:, :, None],
        )
        cv2.imwrite(
            str(debug_dir / "mpi_background_alpha.png"),
            bg_np * layered_alpha[0][:, :, None],
        )

    background_alpha = layered_alpha[0]
    foreground_alpha = layered_alpha[1]
    background_alpha = 1 - foreground_alpha

    # Preserve the pre-resize masks for downstream compositing.
    mpi_orig_mask = [background_alpha.copy(), foreground_alpha.copy()]

    foreground_alpha = cv2.resize(
        foreground_alpha, (64, 64), interpolation=cv2.INTER_NEAREST
    )
    background_alpha = cv2.resize(
        background_alpha, (64, 64), interpolation=cv2.INTER_NEAREST
    )

    foreground_alpha_tensor = (
        torch.tensor(foreground_alpha, dtype=torch.float16)
        .to(device)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    background_alpha_tensor = (
        torch.tensor(background_alpha, dtype=torch.float16)
        .to(device)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    return background_alpha_tensor, foreground_alpha_tensor, mpi_orig_mask


def crop_back(pred, tar_image, extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1, y2, x1, x2 = tar_box_yyxx_crop
    pred = cv2.resize(pred, (W2, H2))
    m = 10  # margin_pixel

    if W1 == H1:
        tar_image[y1 + m : y2 - m, x1 + m : x2 - m, :] = pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:, pad1:-pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1:-pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1 + m : y2 - m, x1 + m : x2 - m, :] = pred[m:-m, m:-m]
    return gen_image


def inference_single_image(
    mpi_data_dict: dict[str, Any],
    alignment_result: AnyDoorCollage,
    background_image: npt.NDArray[np.uint8],
    anydoor: torch.nn.Module,
    ddim_sampler: DDIMSampler,
    guidance_scale: float = 5.0,
    save_memory: bool = True,
) -> npt.NDArray[np.float64]:
    object_crop = alignment_result.object
    collage = alignment_result.collage

    ref = object_crop * 255
    hint = collage * 127.5 + 127.5

    hint_mask = collage[:, :, -1] * 255
    hint_mask = np.stack([hint_mask, hint_mask, hint_mask], -1)
    ref = cv2.resize(ref.astype(np.uint8), (512, 512))

    if save_memory:
        anydoor.low_vram_shift(is_diffusing=False)

    ref = object_crop
    hint = collage
    num_samples = 1

    control = torch.from_numpy(hint.copy()).float().cuda()
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, "b h w c -> b c h w").clone()

    clip_input = torch.from_numpy(ref.copy()).float().cuda()
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, "b h w c -> b c h w").clone()

    guess_mode = False
    H, W = 512, 512

    cond = {
        "c_concat": [control],
        "c_crossattn": [anydoor.get_learned_conditioning(clip_input)],
    }
    un_cond = {
        "c_concat": None if guess_mode else [control],
        "c_crossattn": [
            anydoor.get_learned_conditioning(
                [torch.zeros((1, 3, 224, 224))] * num_samples
            )
        ],
    }
    shape = (4, H // 8, W // 8)

    if save_memory:
        anydoor.low_vram_shift(is_diffusing=True)

    num_samples = 1
    strength = 1
    guess_mode = False
    ddim_steps = 50
    scale = guidance_scale
    eta = 0.0

    anydoor.control_scales = (
        [strength * (0.825 ** float(12 - i)) for i in range(13)]
        if guess_mode
        else ([strength] * 13)
    )
    samples, intermediates = ddim_sampler.sample(
        ddim_steps,
        num_samples,
        shape,
        cond,
        verbose=False,
        eta=eta,
        mpi_data=mpi_data_dict,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond,
        log_every_t=1,
    )

    mpi_data_dict["object_latents"] = intermediates["x_inter"]

    if save_memory:
        anydoor.low_vram_shift(is_diffusing=False)

    x_samples = anydoor.decode_first_stage(samples)
    x_samples = (
        (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
        .cpu()
        .numpy()
    )

    pred = x_samples[0]
    pred = np.clip(pred, 0, 255)[:, :, :]

    orig_pred = pred.copy()

    pred_anydoor = orig_pred[1:, :, :]
    sizes = alignment_result.extra_sizes
    tar_box_yyxx_crop = alignment_result.target_box_yyxx_crop
    gen_image_anydoor = crop_back(
        pred_anydoor, background_image.copy(), sizes, tar_box_yyxx_crop
    )

    return gen_image_anydoor


@attrs.define
class NullTextInvertResults:
    ddim_latents: torch.Tensor
    # Activations from the first diffusion inference pass (from layers 1-3 of the decoder of the UNet as a list with 3 entries).
    activations: list[torch.Tensor]
    latent_image: torch.Tensor


def null_text_invert(
    image: npt.NDArray[np.uint8],
    depth: npt.NDArray[np.uint8],
    diff_handles: FeatureGuidance,
) -> NullTextInvertResults:
    inv_prompt = "a photo of an indoor scene"
    device = diff_handles.device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ten_img3 = (
        torch.from_numpy(np.array(PILImageModule.fromarray(image)))
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
        / 255.0
    )

    depth_fore = (
        torch.tensor(depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    )
    null_text_emb, ddim_latents = diff_handles.invert_input_image(
        ten_img3, depth_fore, prompt=inv_prompt
    )
    init_noise = ddim_latents[-1]

    _, _, activations, latent_image = diff_handles.generate_input_image(
        depth=depth_fore,
        prompt=inv_prompt,
        null_text_emb=null_text_emb,
        init_noise=init_noise,
    )
    return NullTextInvertResults(
        ddim_latents=ddim_latents,
        activations=activations,
        latent_image=latent_image,
    )


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
    timestamped_results_dir: Path = Path(
        RESULTS_BASE_DIR.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    )
    args = parse_args()

    cache_dir = RESULTS_BASE_DIR.format("cache")
    os.makedirs(cache_dir, exist_ok=True)

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # collate_fn keeps types as numpy instead of torch and avoids adding a batch dim
    dataset: CSC2529Dataset = CSC2529Dataset(Path(args.dataset_base_dir))
    loader = DataLoader(dataset, collate_fn=lambda x: x[0])  # type: ignore

    # Load Models
    sam_pipeline = pipeline("mask-generation", model="facebook/sam-vit-huge", device=0)
    depth_pipeline = pipeline(
        task="depth-estimation", model="LiheYoung/depth-anything-small-hf"
    )
    if not args.skip_diffusion:
        diff_handles = FeatureGuidance()
        anydoor = load_anydoor()
        ddim_sampler = DDIMSampler(anydoor)

    filter_indices = set(args.sample_indexes) if args.sample_indexes else None

    for dataset_idx, background_image, source_px, target_px in loader:
        if filter_indices is not None and dataset_idx not in filter_indices:
            continue
        iter_start = time.perf_counter()
        timings: dict[str, float] = {}
        # Create output directories
        results_dir = timestamped_results_dir / str(dataset_idx)
        debug_dir = results_dir / "debug"
        os.makedirs(results_dir)
        if args.debug:
            os.makedirs(debug_dir)

        # Calculate the depth of the source and target points
        depth_start = time.perf_counter()
        full_depth_image: Image = depth_pipeline(  # type: ignore
            PILImageModule.fromarray(background_image)
        )["depth"]
        source_depth = float(full_depth_image.getpixel(tuple(source_px)))
        target_depth = float(full_depth_image.getpixel(tuple(target_px)))
        scale_applied = (
            target_depth / source_depth if target_depth != 0 else float("nan")
        )
        timings["time_full_depth"] = time.perf_counter() - depth_start

        # Draw the source and target point on the background image.
        if args.debug:
            pts_preview = draw_depth_pts(
                background_image.copy(),
                source_px,
                target_px,
                source_depth=source_depth,
                target_depth=target_depth,
            )
            PILImageModule.fromarray(pts_preview).save(debug_dir / "input_preview.png")

        # Extract the SAM mask and corresponding sub-image that intersect source_px
        obj_prep_start = time.perf_counter()
        object_mask = get_object_mask(background_image, source_px, sam_pipeline)
        cropped_object_rgba = extract_masked_object(background_image, object_mask)

        # Remove object from image
        # Operate on 512 by 512 pixels centered at source_px
        bbox = get_bbox_from_point(background_image, source_px, box_size=512)
        background_image_cropped, object_mask_cropped = crop_with_bbox(
            background_image, object_mask, bbox
        )
        # Remove object from cropped image
        background_image_cropped = remove_object(
            background_image_cropped.copy(), object_mask_cropped.copy()
        )
        # Paste cropped image back to original image
        background_image = uncrop_with_bbox(
            background_image_cropped,
            background_image,
            bbox
        )

        # Calculate target mask based on depth scaling
        target_mask: npt.NDArray[np.bool_] = calc_target_mask(
            image=background_image,
            source_object=cropped_object_rgba,
            target_px=target_px,
            source_depth=source_depth,
            target_depth=target_depth,
        )

        anydoor_collage = create_anydoor_collage(
            bg_image=background_image,
            object_image=cropped_object_rgba[:, :, :3],
            object_mask=cropped_object_rgba[:, :, -1],
            target_mask=target_mask,
        )
        if args.debug:
            PILImageModule.fromarray(target_mask.astype(np.uint8) * 255, mode="L").save(
                debug_dir / "target_mask.png"
            )
            anydoor_collage.save(debug_dir)

        # Resize sam mask to 512 because the object_bbox_for_sam that create_anydoor_collage
        # returns is still expressed in the crop’s native resolution, i.e., before
        # the crop is padded and resized.
        y1, y2, x1, x2 = anydoor_collage.target_box_yyxx_crop
        oy1, oy2, ox1, ox2 = anydoor_collage.object_bbox_for_sam
        r = 512 / (y2 - y1)
        anydoor_collage.object_bbox_for_sam = np.array(
            [oy1 * r, oy2 * r, ox1 * r, ox2 * r]
        )
        bg_image_cropped = ((anydoor_collage.background * 127.5) + 127.5).astype(
            np.uint8
        )
        timings["time_object_prep"] = time.perf_counter() - obj_prep_start

        # Calculate the depth and sam masks of scene objects
        scene_masks_start = time.perf_counter()
        depth, sam_mask = get_depth_and_sam_mask(
            PILImageModule.fromarray(bg_image_cropped),
            depth_pipe=depth_pipeline,
            sam_pipe=sam_pipeline,
            is_relative_depth=True,
        )
        timings["time_scene_masks"] = time.perf_counter() - scene_masks_start

        if args.debug:
            depth.save(f"{debug_dir}/depth.png")

        mpi_start = time.perf_counter()
        (
            mpi_background_alpha,
            mpi_foreground_alpha,
            mpi_orig_mask,
        ) = build_mpi_masks(
            background_crop=bg_image_cropped,
            depth_image=depth,
            target_depth=target_depth,
            device=device,
            debug_dir=debug_dir,
        )
        timings["time_mpi_build"] = time.perf_counter() - mpi_start

        if args.skip_diffusion:
            continue

        # Move Diffusion Handles to GPU for null-text inversion
        null_text_start = time.perf_counter()
        diff_handles.to(device)
        null_text_cache_file = f"{cache_dir}/null_text_{dataset_idx}.pickle"
        try:
            with open(null_text_cache_file, "rb") as nti_pickle_file:
                nti_result = pickle.load(nti_pickle_file)
                print("Loaded null-text inversion from cache")
        except FileNotFoundError:
            with torch.enable_grad():
                nti_result = null_text_invert(
                    bg_image_cropped, np.array(depth), diff_handles
                )
            with open(null_text_cache_file, "wb") as nti_pickle_file:
                pickle.dump(nti_result, nti_pickle_file)
                print("Saved null-text inversion to cache")
        timings["time_null_text"] = time.perf_counter() - null_text_start

        # Reconstruct image to debug if inversion is correct
        if args.debug:
            latent_image = nti_result.latent_image
            with torch.no_grad():
                latent_image = diff_handles.diffuser.vae.decode(
                    latent_image / diff_handles.diffuser.vae.config.scaling_factor,
                    return_dict=False,
                )[0]
                latent_image = VaeImageProcessor(
                    vae_scale_factor=diff_handles.diffuser.vae.config.scaling_factor
                ).postprocess(latent_image, output_type="pt")
                latent_image = latent_image.permute(0, 2, 3, 1).squeeze().cpu().numpy()
                latent_image = (latent_image * 255).astype(np.uint8)
                cv2.imwrite(
                    str(debug_dir / "recon_fg.jpg"),
                    cv2.cvtColor(latent_image, cv2.COLOR_RGB2BGR),
                )

        if torch.cuda.is_available():
            # Unload Diffusion Handles from GPU
            diff_handles.to("cpu")
            torch.cuda.empty_cache()

            # Move AnyDoor to GPU for inpainting
            anydoor.to(device)
            ddim_sampler.model = anydoor

        mpi_data_dict: dict[str, Any] = {
            "ddim_latents": nti_result.ddim_latents,
            "mpi_masks": [mpi_background_alpha, mpi_foreground_alpha],
            "do_mpi": True,
            "mpi_orig_mask": mpi_orig_mask,
            "fg_object_latents": None,
            "activation_fore": nti_result.activations,
        }

        diffusion_start = time.perf_counter()
        result_image = inference_single_image(
            mpi_data_dict=mpi_data_dict,
            alignment_result=anydoor_collage,
            background_image=background_image.copy(),
            anydoor=anydoor,
            ddim_sampler=ddim_sampler,
            guidance_scale=5.0,
            save_memory=True,
        )
        timings["time_diffusion"] = time.perf_counter() - diffusion_start

        # Unload AnyDoor from GPU before next iteration
        anydoor.to("cpu")
        ddim_sampler.model = anydoor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result_image = np.clip(result_image, 0, 255).astype(np.uint8)
        cv2.imwrite(
            str(results_dir / "result.png"),
            cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR),
        )

        timings["time_iteration_total"] = time.perf_counter() - iter_start
        metrics = calc_metrics(
            background_image=background_image,
            result_image=result_image,
            depth_map=full_depth_image,
            source_mask=object_mask,
            target_mask=target_mask,
            scale_applied=scale_applied,
            timings=timings,
            sam_pipeline=sam_pipeline,
            source_px=source_px,
        )

        with open(results_dir / "metrics.json", "w") as file:
            json.dump(metrics, file, indent=4)
