from __future__ import annotations

from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image
from datasets.data_utils import (
    expand_bbox,
    expand_image_mask,
    get_bbox_from_mask,
    new_box2squre,
    pad_to_square,
    sobel,
    box_in_box,
)
import json
import attrs

NPImage = npt.NDArray[np.uint8]
NPPixel = npt.NDArray[np.int32]  # x,y


class CSC2529Dataset(Dataset[tuple[int, NPImage, NPPixel, NPPixel]]):
    def __init__(self, base_dir: Path):
        self.background_image_dir = base_dir / "background"
        self.target_mask_dir = base_dir / "target_mask"
        self.object_dir = base_dir / "object"
        self.num_backgrounds = len(list(self.background_image_dir.glob("*.png")))

        with open(base_dir / "depth.json") as depth_file:
            depth_json = json.load(depth_file)
            self.idx_to_pts: dict[int, tuple[NPPixel, NPPixel]] = {
                int(idx): CSC2529Dataset.points_to_array(entry)
                for idx, entry in depth_json.items()
            }

    @staticmethod
    def points_to_array(json: dict[str, int]) -> tuple[NPPixel, NPPixel]:
        """Parses json dict of source pixel and target pixel to to numpy arrays."""
        return (
            np.array([json["source_x"], json["source_y"]], dtype=np.int32),
            np.array([json["target_x"], json["target_y"]], dtype=np.int32),
        )

    def __len__(self) -> int:
        return self.num_backgrounds

    def __getitem__(self, idx: int) -> tuple[int, NPImage, NPPixel, NPPixel]:
        background_image = cv2.imread(f"{self.background_image_dir}/{idx}.png")
        background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
        source, target = self.idx_to_pts[idx]
        return (idx, np.array(background_image), source, target)


@attrs.define
class AnyDoorCollage:
    """
    Contains the tensors produced by `create_collage`.
    Collage referece to the image composite referenced in the depth-ant
    """

    # Float32 HxWx3 RGB crop of the reference object (224x224) normalized to [0, 1].
    object: npt.NDArray[np.float32]
    # Float32 512x512x3 RGB crop of the target scene normalized to [-1, 1].
    background: npt.NDArray[np.float32]
    # Float32 512x512x4 tensor: collage RGB channels in [-1, 1] plus binary mask.
    collage: npt.NDArray[np.float32]
    # UInt8 512x512x3 mask aligned with MPI preprocessing.
    target_mpi_mask: NPImage

    # Float32 array [H1, W1, H2, W2] tracking sizes before/after padding.
    extra_sizes: npt.NDArray[np.float32]
    # Float32 array [y1, y2, x1, x2] for the expanded crop box in the background image.
    target_box_yyxx_crop: npt.NDArray[np.float32]
    # Float32 array [y1, y2, x1, x2] locating the object inside the cropped target.
    object_bbox_for_sam: npt.NDArray[np.float32]

    def save(self, output_dir: Path) -> None:
        """Persist all image tensors to ``output_dir`` for debugging."""

        output_dir.mkdir(parents=True, exist_ok=True)

        object_img = (np.clip(self.object, 0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(object_img).save(output_dir / "object.png")

        target_img = ((np.clip(self.background, -1.0, 1.0) + 1.0) * 127.5).astype(
            np.uint8
        )
        Image.fromarray(target_img).save(output_dir / "target.png")

        collage_rgb = (
            (np.clip(self.collage[:, :, :3], -1.0, 1.0) + 1.0) * 127.5
        ).astype(np.uint8)
        Image.fromarray(collage_rgb).save(output_dir / "collage.png")

        collage_mask = (np.clip(self.collage[:, :, 3], 0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(collage_mask).save(output_dir / "collage_mask.png")

        Image.fromarray(self.target_mpi_mask.astype(np.uint8)).save(
            output_dir / "target_mpi_mask.png"
        )


def create_anydoor_collage(
    bg_image: NPImage,
    object_image: NPImage,
    object_mask: NPImage,
    target_mask: NPImage,
    shape_control: bool = False,
) -> AnyDoorCollage:
    """
    Crops and aligns the reference object and target region so that the
    diffusion model can condition on them consistently.
    """

    # ref expand
    object_mask = (object_mask > 0).astype(np.uint8)
    object_box_yyxx = get_bbox_from_mask(object_mask)

    # ref filter mask
    object_mask_3 = np.stack([object_mask, object_mask, object_mask], -1)
    masked_object_image = object_image * object_mask_3 + np.ones_like(
        object_image
    ) * 255 * (1 - object_mask_3)

    y1, y2, x1, x2 = object_box_yyxx
    masked_object_image = masked_object_image[y1:y2, x1:x2, :]
    object_mask = object_mask[y1:y2, x1:x2]

    ratio = np.random.randint(11, 12) / 10
    masked_object_image, object_mask = expand_image_mask(
        masked_object_image, object_mask, ratio=ratio
    )
    object_mask_3 = np.stack([object_mask, object_mask, object_mask], -1)

    ### added
    masked_obj_img_for_collage = masked_object_image.copy()
    obj_mask_for_collage = object_mask.copy()

    # to square and resize
    masked_object_image = pad_to_square(
        masked_object_image, pad_value=255, random=False
    )
    masked_object_image = cv2.resize(masked_object_image, (224, 224)).astype(np.uint8)

    object_mask_3 = pad_to_square(object_mask_3 * 255, pad_value=0, random=False)
    object_mask_3 = cv2.resize(object_mask_3, (224, 224)).astype(np.uint8)
    object_mask = object_mask_3[:, :, 0]

    # ref aug
    masked_obj_image_aug = masked_object_image  # aug_data(masked_ref_image)

    ## changed
    masked_obj_image_compose, obj_mask_compose = (
        masked_obj_img_for_collage,
        obj_mask_for_collage * 255,
    )
    object_mask_3 = np.stack([obj_mask_compose, obj_mask_compose, obj_mask_compose], -1)
    obj_image_collage = sobel(masked_obj_image_compose, obj_mask_compose / 255)

    # ========= Target ===========
    tar_box_yyxx = get_bbox_from_mask(target_mask)
    tar_box_yyxx = expand_bbox(target_mask, tar_box_yyxx, ratio=[1.1, 1.2])  # 1.1,1.2

    # crop
    tar_box_yyxx_crop = expand_bbox(
        bg_image, tar_box_yyxx, ratio=[1.5, 3]
    )  # 1.5, 3   #1.2 1.6
    tar_box_yyxx_crop = new_box2squre(bg_image, tar_box_yyxx_crop)  # crop box
    y1, y2, x1, x2 = tar_box_yyxx_crop

    # transforming mask for mpi according to tar_image
    tar_mask_mpi = target_mask.copy()
    tar_mask_mpi_cropped = tar_mask_mpi[y1:y2, x1:x2]
    tar_mask_mpi_cropped = cv2.cvtColor(tar_mask_mpi_cropped, cv2.COLOR_GRAY2RGB)

    cropped_bg_image = bg_image[y1:y2, x1:x2, :]
    cropped_tar_mask = target_mask[y1:y2, x1:x2]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1, y2, x1, x2 = tar_box_yyxx

    # collage
    obj_image_collage = cv2.resize(obj_image_collage, (x2 - x1, y2 - y1))
    obj_mask_compose = cv2.resize(obj_mask_compose.astype(np.uint8), (x2 - x1, y2 - y1))
    obj_mask_compose = (obj_mask_compose > 128).astype(np.uint8)

    collage = cropped_bg_image.copy()
    collage[y1:y2, x1:x2, :] = obj_image_collage

    collage_mask = cropped_bg_image.copy() * 0.0
    collage_mask[y1:y2, x1:x2, :] = 1.0

    if shape_control:
        collage_mask = np.stack(
            [cropped_tar_mask, cropped_tar_mask, cropped_tar_mask], -1
        )

    # the size before pad
    H1, W1 = collage.shape[0], collage.shape[1]
    cropped_bg_image = pad_to_square(
        cropped_bg_image, pad_value=0, random=False
    ).astype(np.uint8)
    collage = pad_to_square(collage, pad_value=0, random=False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value=-1, random=False).astype(
        np.uint8
    )

    # transforming mask for mpi according to tar_image
    tar_mask_mpi_cropped = pad_to_square(
        tar_mask_mpi_cropped, pad_value=0, random=False
    ).astype(np.uint8)
    tar_mask_mpi_cropped = cv2.resize(
        tar_mask_mpi_cropped, (512, 512), interpolation=cv2.INTER_NEAREST
    ).astype(np.uint8)

    # the size after pad
    H2, W2 = collage.shape[0], collage.shape[1]
    cropped_bg_image = cv2.resize(cropped_bg_image, (512, 512)).astype(np.float32)
    collage = cv2.resize(collage, (512, 512)).astype(np.float32)
    collage_mask = (
        cv2.resize(collage_mask, (512, 512)).astype(np.float32) > 0.5
    ).astype(np.float32)

    masked_obj_image_aug = masked_obj_image_aug / 255
    cropped_bg_image = cropped_bg_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0
    collage = np.concatenate([collage, collage_mask[:, :, :1]], -1)

    return AnyDoorCollage(
        object=masked_obj_image_aug.copy(),
        background=cropped_bg_image.copy(),
        collage=collage.copy(),
        extra_sizes=np.array([H1, W1, H2, W2]),
        target_box_yyxx_crop=np.array(tar_box_yyxx_crop),
        target_mpi_mask=tar_mask_mpi_cropped,
        object_bbox_for_sam=np.array(tar_box_yyxx),
    )
