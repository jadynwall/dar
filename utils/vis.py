import numpy.typing as npt


def draw_pts(
    image: npt.NDArray, source_px: npt.NDArray, target_px: npt.NDArray
) -> npt.NDArray:
    image_copy = image.copy()
    radius = 3

    # Set source point to blue
    sx, sy = source_px
    image_copy[sy - radius : sy + radius, sx - radius : sx + radius, :] = 0
    image_copy[sy - radius : sy + radius, sx - radius : sx + radius, 2] = 255

    # Set target point to green
    tx, ty = target_px
    image_copy[ty - radius : ty + radius, tx - radius : tx + radius, :] = 0
    image_copy[ty - radius : ty + radius, tx - radius : tx + radius, 1] = 255

    return image_copy
