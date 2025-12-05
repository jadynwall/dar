from pathlib import Path

from setuptools import find_namespace_packages, setup

ROOT = Path(__file__).parent

# Packages living at repo root (implicit namespace packages).
base_packages = find_namespace_packages(
    where=str(ROOT),
    include=("utils*", "cldm*", "ldm*", "iseg*", "dinov2*", "datasets*"),
)

# Packages under src/featglac.
featglac_packages = find_namespace_packages(
    where=str(ROOT / "src"),
    include=("featglac*",),
)

# Packages under src/Depth-Anything.
depth_anything_packages = find_namespace_packages(
    where=str(ROOT / "src" / "Depth-Anything"),
    include=("depth_anything*", "metric_depth*", "semseg*", "controlnet*", "torchhub*"),
)

setup(
    name="depthedit",
    version="0.1.0",
    description="Depth-aware editing utilities and scripts.",
    packages=base_packages + featglac_packages + depth_anything_packages,
    package_dir={
        "": ".",
        "featglac": "src/featglac",
        "depth_anything": "src/Depth-Anything/depth_anything",
        "metric_depth": "src/Depth-Anything/metric_depth",
        "semseg": "src/Depth-Anything/semseg",
        "controlnet": "src/Depth-Anything/controlnet",
        "torchhub": "src/Depth-Anything/torchhub",
    },
    include_package_data=True,
    python_requires=">=3.9,<3.10",
    zip_safe=False,
)
