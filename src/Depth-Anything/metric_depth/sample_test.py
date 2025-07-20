# from depth_anything.dpt import DepthAnything
import torch
import sys
sys.path.append("/mnt/data/rishubh/sachi/AnyDoor/src/Depth-Anything/")
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms


FL = 715.0873
FY = 256 * 0.6
FX = 256 * 0.6
DATASET = 'nyu'
model_name = 'zoedepth'
pretrained_resource = "local::../../../weights/depth_anything_metric_depth_indoor.pt"

config = get_config(model_name, "eval", DATASET)
config.pretrained_resource = pretrained_resource
model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

input_image = Image.open("../complex_dining.jpg")
W, H = input_image.size
image_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

pred = model(image_tensor, dataset=DATASET)
if isinstance(pred, dict):
    print("oits a dict, ",pred.keys())
    pred = pred.get('metric_depth', pred.get('out'))
elif isinstance(pred, (list, tuple)):
    pred = pred[-1]
pred = pred.squeeze().detach().cpu().numpy()
pred = cv2.resize(pred, (W, H))
cv2.imwrite("./actual_depth.jpg", pred)
print("max: ", pred.max(), "min: ", pred.min())
pred = (pred - pred.min()) / (pred.max() - pred.min())
print("max: ", pred.max(), "min: ", pred.min())
pred = Image.fromarray((pred*255).astype(np.uint8))
pred.save("sample_test_depth.jpg")
print("orig shape: ", W, H, "pred shape: ", pred.size)



