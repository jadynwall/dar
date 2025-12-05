## Depth-Aware Object Edit

This project stitches together depth estimation, SAM masking, Stable Diffusion inpainting, and feature-guided pipelines for single-image object edits with consistent depth cues.

## Setup

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda with GPU-capable PyTorch drivers (CUDA 11.7+).
2. Create the environment from the lockfile:
   ```bash
   conda env create -f environment.yml
   conda activate depthedit
   ```
   (The env installs this repo in editable mode via: `pip -e .`)
3. Install the Hugging Face CLI and fetch model weights (~62GB):
   ```bash
   pip install -U "huggingface_hub[cli]"
   bash download_weights.sh
   ```
4. Run scripts from the repo root, e.g.:
   ```bash
   python scripts/object_placement.py
   ```
   
#### System Requirements

- RAM: 32GB
- VRAM: 24GB
- Disk Space: 80GB

## Object Placement
Run the object placement pipeline on the dataset under `datasets/csc2529/`:
```bash
python scripts/object_placement.py --sample-indexes 10 11 14
```

## Repository Layout

- `scripts/`: Entry-point scripts (e.g., object removal/placement) that tie together pipelines.
- `datasets/`: Dataset helpers and sample assets.
- `weights/`: Downloaded model checkpoints (populated by `download_weights.sh`).
- `configs/`: YAML configs for training/inference.
- `utils/`: Shared utilities (masking, metrics, visualization, MPI helpers).
- `cldm/`, `ldm/`, `dinov2/`, `iseg/`: Third-party or adapted model code.
- `src/featglac/`: Feature guidance and ZoeDepth components used in pipelines.
- `src/Depth-Anything/`: Depth-Anything sources and dependencies.
- `results/`: Generated outputs and caches from runs.
