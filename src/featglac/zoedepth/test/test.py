import torch
import torchvision
from PIL import Image

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

def test():
    conf = get_config("zoedepth_nk", "infer")
    model = build_model(conf)

    input_img_path = 'data/sunflower.png'
    
    input_img = load_image(input_img_path).unsqueeze(dim=0)
    
    depth = model.infer(input_img)

    save_image((depth/depth.max()).squeeze(dim=0), 'results/sunflower_depth.png')

def load_image(path: str) -> torch.Tensor:
    img = Image.open(path)
    img = img.convert('RGB')
    img = torchvision.transforms.functional.pil_to_tensor(img)
    img = img / 255.0
    return img

def save_image(img: torch.Tensor, path: str):
    img = torchvision.transforms.functional.to_pil_image(img)
    img.save(path)

if __name__ == '__main__':
    test()
    print('Done!')
