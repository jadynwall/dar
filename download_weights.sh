#!/bin/bash

cd weights
# Downloading weights for depth anything model
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth

# Downloading weights for Dinov2 backbone
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth

# Downloading anydoor weights
wget https://huggingface.co/spaces/xichenhku/AnyDoor/resolve/main/epoch%3D1-step%3D8687.ckpt

