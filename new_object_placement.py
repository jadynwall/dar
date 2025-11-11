from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import einops
import numpy as np
from datasets.csc2529 import AlignmentResult, CSC2529Dataset, align_images
from torch.utils.data import DataLoader
from datetime import datetime
from pytorch_lightning import seed_everything
import os
from utils.mpi.preprocess import get_depth_and_sam_mask
from cldm.ddim_hacked_mpi_featguidance import DDIMSampler
from utils.mpi.mpi import get_mpi_rgb_and_alpha
from cldm.model import create_model, load_state_dict
from PIL import Image as PILImageModule
import torch
from omegaconf import OmegaConf
from src.featglac import FeatureGuidance
import numpy.typing as npt
from diffusers.image_processor import VaeImageProcessor
import attrs
from typing import Any

import pickle

# Results are stored in a timestamped folder:
RESULTS_BASE_DIR = "results/new_object_placement/{}"


def load_anydoor() -> torch.nn.Module:
    config = OmegaConf.load("./configs/inference.yaml")
    model = create_model(config.config_file)
    model.load_state_dict(load_state_dict(config.pretrained_model))
    return model


def load_diffusion_handles() -> FeatureGuidance:
    diff_handles = FeatureGuidance()
    return diff_handles


def build_mpi_masks(
    background_crop: npt.NDArray[np.uint8],
    depth_image: PILImageModule.Image,
    target_depth: int,
    device: torch.device,
    debug_dir: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[npt.NDArray[np.uint8]]]:
    """Generate background/foreground MPI masks along with the high-res originals."""

    depth_partition: list[tuple[int, int]] = [
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
    m = 10  # maigin_pixel 5

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
    alignment_result: AlignmentResult,
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
    loader = DataLoader(dataset, collate_fn=lambda x: x[0])

    # Load Models
    diff_handles = load_diffusion_handles()
    anydoor = load_anydoor()
    ddim_sampler = DDIMSampler(anydoor)

    for (
        dataset_idx,
        background_image,
        object_image,
        object_mask,
        target_mask,
        target_depth,
    ) in loader:
        # Create output directories
        results_dir = timestamped_results_dir / str(dataset_idx)
        os.makedirs(results_dir)

        debug_dir = None
        if args.debug:
            debug_dir = results_dir / "debug"
            os.makedirs(debug_dir)

        alignment_result = align_images(
            background_image, object_image, object_mask, target_mask
        )
        if debug_dir:
            alignment_result.save(debug_dir)

        # Resize sam mask to 512 because the object_bbox_for_sam that align_images
        # returns is still expressed in the crop’s native resolution, i.e., before
        # the crop is padded and resized.
        y1, y2, x1, x2 = alignment_result.target_box_yyxx_crop
        oy1, oy2, ox1, ox2 = alignment_result.object_bbox_for_sam
        r = 512 / (y2 - y1)
        alignment_result.object_bbox_for_sam = np.array(
            [oy1 * r, oy2 * r, ox1 * r, ox2 * r]
        )

        bg_image_cropped = ((alignment_result.background * 127.5) + 127.5).astype(
            np.uint8
        )

        # Calculate scene depth
        depth, sam_mask = get_depth_and_sam_mask(
            PILImageModule.fromarray(bg_image_cropped), is_relative_depth=True
        )
        if args.debug:
            depth.save(f"{debug_dir}/depth.png")

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

        # Move Diffusion Handles to GPU for null-text inversion
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
                    f"{debug_dir}/recon_fg.jpg",
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

        gen_image = inference_single_image(
            mpi_data_dict=mpi_data_dict,
            alignment_result=alignment_result,
            background_image=background_image.copy(),
            anydoor=anydoor,
            ddim_sampler=ddim_sampler,
            guidance_scale=5.0,
            save_memory=True,
        )

        # Unload AnyDoor from GPU before next iteration
        anydoor.to("cpu")
        ddim_sampler.model = anydoor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cv2.imwrite(str(results_dir / "result.png"), gen_image[:, :, ::-1])
