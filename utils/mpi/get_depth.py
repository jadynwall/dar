import sys
from pathlib import Path
from typing import Tuple

import torch

repo_root = Path(__file__).resolve().parents[2]
depth_anything_path = repo_root / "src/Depth-Anything/metric_depth"
if str(depth_anything_path) not in sys.path:
    sys.path.append(str(depth_anything_path))
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


FL: float = 715.0873
FY: float = 256 * 0.6
FX: float = 256 * 0.6
NYU_DATASET: str = "nyu"
model_name: str = "zoedepth"
pretrained_resource: str = "local::./weights/depth_anything_metric_depth_indoor.pt"
# pretrained_resource = "local::/mnt/data/rishubh/sachi/AnyDoor/weights/depth_anything_metric_depth_indoor.pt"

config = get_config(model_name, "eval", NYU_DATASET)
config.pretrained_resource = pretrained_resource
depth_model: torch.nn.Module = build_model(config).to(
    "cuda" if torch.cuda.is_available() else "cpu"
)
depth_model.eval()


def get_depth_map(image: Image.Image) -> Tuple[np.ndarray, Image.Image]:
    input_image = image
    W, H = input_image.size
    image_tensor = (
        transforms.ToTensor()(input_image)
        .unsqueeze(0)
        .to("cuda" if torch.cuda.is_available() else "cpu")
    )
    pred = depth_model(image_tensor, dataset=NYU_DATASET)
    if isinstance(pred, dict):
        pred = pred.get("metric_depth", pred.get("out"))
    elif isinstance(pred, (list, tuple)):
        pred = pred[-1]
    pred = pred.squeeze().detach().cpu().numpy()
    pred = cv2.resize(pred, (W, H))

    visualise_pred = (pred - pred.min()) / (pred.max() - pred.min())
    visualise_pred = Image.fromarray((visualise_pred * 255).astype(np.uint8))
    return pred, visualise_pred
