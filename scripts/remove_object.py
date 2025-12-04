import os
import sys
from pathlib import Path
from datetime import datetime
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
from diffusers import StableDiffusionInpaintPipeline

# Allow running the script directly without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.img_proc import get_object_mask

RESULTS_BASE_DIR = "results/remove_object/{}"

timestamped_results_dir: Path = Path(
    RESULTS_BASE_DIR.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
)
results_dir = timestamped_results_dir
os.makedirs(results_dir)

# load image
image_path = "datasets/csc2529/background/0.png"
init_image_bgr = cv2.imread(image_path)
if init_image_bgr is None:
    raise FileNotFoundError(f"Image not found at {image_path}")
init_image_rgb = cv2.cvtColor(init_image_bgr, cv2.COLOR_BGR2RGB)
init_image_np = init_image_rgb.copy()
init_image = Image.fromarray(init_image_rgb)

# get object mask
sam_pipeline = pipeline("mask-generation", model="facebook/sam-vit-huge", device=0)
# object_mask = get_object_mask(init_image_np, [270,600], sam_pipeline) # for 0.png
object_mask = get_object_mask(init_image_np, [240, 250], sam_pipeline)  # for 1.png
object_mask = object_mask.astype(np.uint8)
mask_pil = Image.fromarray(object_mask * 255).convert("L")
cv2.imwrite(str(results_dir / "sam_mask.png"), object_mask * 255)

pipeline = StableDiffusionInpaintPipeline.from_single_file(
    "weights/sd-v1-5-inpainting.ckpt",
    torch_dtype=torch.float16,
)
pipeline = pipeline.to(0)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
# pipeline.enable_xformers_memory_efficient_attention()

generator_device = "cuda" if torch.cuda.is_available() else "cpu"
generator = torch.Generator(device=generator_device).manual_seed(42)
image = pipeline(
    prompt="destroy the bag",
    image=init_image,
    mask_image=mask_pil,
    generator=generator,
).images[0]
result_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
cv2.imwrite(str(results_dir / "removed_object.png"), result_bgr)
