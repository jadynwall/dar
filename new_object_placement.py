from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np
from datasets.csc2529 import CSC2529Dataset, align_images
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

# Results are stored in a timestamped folder:
RESULTS_DIR_TMPL = "results/new_object_placement/{}"


def load_anydoor() -> torch.nn.Module:
    config = OmegaConf.load("./configs/inference.yaml")
    model = create_model(config.config_file)
    model = model.to(torch.device("cpu"))
    model.load_state_dict(load_state_dict(config.pretrained_model, location="cpu"))
    return model


def load_diffusion_handles() -> FeatureGuidance:
    diff_handles = FeatureGuidance()
    diff_handles.to(torch.device("cpu"))
    return diff_handles


def inference_single_image(
    ref_image,
    ref_mask,
    tar_image,
    tar_mask,
    mpi_data_dict,
    item=None,
    guidance_scale=5.0,
    curr_save_dir=None,
    save_memory=False,
    ddim_sampler=None,
    model=None,
):
    if item is None:
        item = process_pairs(ref_image, ref_mask, tar_image, tar_mask)
    ref = item["ref"] * 255
    tar = item["jpg"] * 127.5 + 127.5
    hint = item["hint"] * 127.5 + 127.5

    # cv2.imwrite("ref_image.png", ref[:, :, ::-1])
    hint_image = hint[:, :, :-1]
    hint_mask = item["hint"][:, :, -1] * 255
    hint_mask = np.stack([hint_mask, hint_mask, hint_mask], -1)
    ref = cv2.resize(ref.astype(np.uint8), (512, 512))

    seed = random.randint(0, 65535)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    ref = item["ref"]
    tar = item["jpg"]
    hint = item["hint"]
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
        "c_crossattn": [model.get_learned_conditioning(clip_input)],
    }
    un_cond = {
        "c_concat": None if guess_mode else [control],
        "c_crossattn": [
            model.get_learned_conditioning(
                [torch.zeros((1, 3, 224, 224))] * num_samples
            )
        ],
    }
    shape = (4, H // 8, W // 8)

    # amodal conditioning

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    # ====
    num_samples = 1  # gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    image_resolution = 512  # gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
    strength = 1  # gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
    guess_mode = False  # gr.Checkbox(label='Guess Mode', value=False)
    # detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
    ddim_steps = (
        50  # gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
    )
    scale = guidance_scale  # gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    seed = (
        -1
    )  # gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    eta = 0.0  # gr.Number(label="eta (DDIM)", value=0.0)

    model.control_scales = (
        [strength * (0.825 ** float(12 - i)) for i in range(13)]
        if guess_mode
        else ([strength] * 13)
    )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
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
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (
        (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
        .cpu()
        .numpy()
    )  # .clip(0, 255).astype(np.uint8)

    # result = x_samples[0][:,:,::-1]
    # result = np.clip(result,0,255)

    pred = x_samples[0]
    pred = np.clip(pred, 0, 255)[:, :, :]

    tag = "w_mpi" if mpi_data_dict["do_mpi"] else "wo_mpi"
    # cv2.imwrite(
    #     os.path.join(curr_save_dir, f"anydoor_orig_gen_{tag}.png"), pred[:, :, ::-1]
    # )
    orig_pred = pred.copy()

    ## saving ours anydoor results
    pred_anydoor = orig_pred[1:, :, :]

    sizes = item["extra_sizes"]
    tar_box_yyxx_crop = item["tar_box_yyxx_crop"]
    gen_image_anydoor = crop_back(pred_anydoor, tar_image, sizes, tar_box_yyxx_crop)
    # cv2.imwrite(os.path.join(curr_save_dir, f"anydoor_gen_{tag}.png"), gen_image_anydoor[:,:,::-1])

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
    results_dir: Path = Path(
        RESULTS_DIR_TMPL.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    )
    args = parse_args()

    debug_dir, alignment_debug_dir = None, None
    if args.debug:
        debug_dir = results_dir / "debug"
        alignment_debug_dir = debug_dir / "alignment"
        os.makedirs(alignment_debug_dir)

    dataset: CSC2529Dataset = CSC2529Dataset(Path(args.dataset_base_dir))

    # collate_fin keeps types as numpy instead of torch and avoids adding a batch dim
    loader = DataLoader(dataset, collate_fn=lambda x: x[0])

    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diff_handles = load_diffusion_handles()

    anydoor = load_anydoor()
    if torch.cuda.is_available():
        anydoor.to("cuda")

    ddim_sampler = DDIMSampler(anydoor)
    ddim_sampler.model = anydoor

    for (
        background_image,
        object_image,
        object_mask,
        target_mask,
        target_depth,
    ) in loader:
        alignment_result = align_images(
            background_image, object_image, object_mask, target_mask
        )

        if alignment_debug_dir:
            alignment_result.save(alignment_debug_dir)

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

        depth, sam_mask = get_depth_and_sam_mask(
            PILImageModule.fromarray(bg_image_cropped), is_relative_depth=True
        )
        if args.debug:
            depth.save(f"{results_dir}/debug/depth.png")

        # Get layered depth mask
        depth_partition: list[tuple[int, int]] = [
            (0, target_depth),
            (target_depth, 300),
        ]
        mpi_foreground_rgb, mpi_foreground_alpha = get_mpi_rgb_and_alpha(
            np.array(bg_image_cropped), np.array(depth), depth_partition
        )

        if args.debug:
            cv2.imwrite(
                "{}/mpi_foreground_alpha.png".format(debug_dir),
                np.array(bg_image_cropped) * mpi_foreground_alpha[1][:, :, None],
            )
            cv2.imwrite(
                "{}/mpi_background_alpha.png".format(debug_dir),
                np.array(bg_image_cropped) * mpi_foreground_alpha[0][:, :, None],
            )

        mpi_background_alpha, mpi_foreground_alpha = (
            mpi_foreground_alpha[0],
            mpi_foreground_alpha[1],
        )
        mpi_background_alpha = 1 - mpi_foreground_alpha

        mpi_orig_mask = [mpi_background_alpha, mpi_foreground_alpha]
        mpi_foreground_alpha = cv2.resize(
            mpi_foreground_alpha, (64, 64), interpolation=cv2.INTER_NEAREST
        )

        mpi_background_alpha = cv2.resize(
            mpi_background_alpha, (64, 64), interpolation=cv2.INTER_NEAREST
        )
        mpi_foreground_alpha = (
            torch.tensor(mpi_foreground_alpha, dtype=torch.float16)
            .to(device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        mpi_background_alpha = (
            torch.tensor(mpi_background_alpha, dtype=torch.float16)
            .to(device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        diff_handles.to(device)
        nti_result = null_text_invert(bg_image_cropped, np.array(depth), diff_handles)

        # reconstruct image to check if inversion is correct
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

        fg_object_latents = None
        do_mpi = True
        mpi_data_dict = {
            "ddim_latents": nti_result.ddim_latents,
            "mpi_masks": [mpi_background_alpha, mpi_foreground_alpha],
            "do_mpi": do_mpi,
            "mpi_orig_mask": mpi_orig_mask,
            "fg_object_latents": fg_object_latents,
            "activation_fore": nti_result.activations,
        }

        # offload feature guidance weights before AnyDoor sampling
        diff_handles.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gen_image = inference_single_image(
            ref_image=object_image,
            ref_mask=target_mask,
            tar_image=alignment_result.object.copy(),
            tar_mask=alignment_result.target_mpi_mask,
            mpi_data_dict=mpi_data_dict,
            aligned_items=alignment_result,
            guidance_scale=5.0,
            curr_save_dir=results_dir,
            save_memory=True,
            ddim_sampler=ddim_sampler,
            model=anydoor,
        )
        cv2.imwrite(str(results_dir / "result.png"), gen_image[:, :, ::-1])
