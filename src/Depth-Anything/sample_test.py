from depth_anything.dpt import DepthAnything
import torch
import sys
sys.path.append("/mnt/data/rishubh/sachi/AnyDoor/src/Depth-Anything/")
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

FL = 715.0873
FY = 256 * 0.6
FX = 256 * 0.6
DATASET = 'nyu'
model_name = 'zoedepth'
pretrained_resource = "../../weights/depth_anything_metric_depth_indoor.pt"

config = get_config(model_name, "eval", DATASET)
config.pretrained_resource = pretrained_resource
model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()