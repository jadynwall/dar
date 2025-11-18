import os
from pathlib import Path
from datetime import datetime
import torch
import cv2
import numpy as np
from transformers import pipeline
from diffusers import AutoPipelineForInpainting
from utils.img_proc import get_object_mask

RESULTS_BASE_DIR = "results/remove_object/{}"

timestamped_results_dir: Path = Path(
    RESULTS_BASE_DIR.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
)
results_dir = timestamped_results_dir
os.makedirs(results_dir)

# load image
# image_path = 'datasets/csc2529/background/0.png'
image_path = 'datasets/csc2529/background/1.png'
init_image = cv2.imread(image_path)

# get object mask
sam_pipeline = pipeline("mask-generation", model="facebook/sam-vit-huge", device=0)
# object_mask = get_object_mask(init_image, [270,600], sam_pipeline) # for 0.png
object_mask = get_object_mask(init_image, [240,250], sam_pipeline) # for 1.png
object_mask = object_mask.astype(np.uint8)
cv2.imwrite(str(results_dir / 'sam_mask.png'), object_mask*255)

pipeline = AutoPipelineForInpainting.from_pretrained(
    # "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
    "weights/stable-diffusion-2-1-base", torch_dtype=torch.float16, variant="fp16", local_files_only=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
# pipeline.enable_xformers_memory_efficient_attention()

image = pipeline(prompt="high quality", image=init_image, mask_image=object_mask).images[0]
cv2.imwrite(str(results_dir / 'removed_object.png'), np.array(image))