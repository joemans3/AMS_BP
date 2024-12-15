import numpy as np


def mask_maker(
    total_img_dims: np.ndarray, submask_coordinates: np.ndarray, mask_val: int
) -> np.ndarray:
    img = np.zeros(total_img_dims)
    img[
        submask_coordinates[1][0] : submask_coordinates[1][1],
        submask_coordinates[0][0] : submask_coordinates[0][1],
    ] = mask_val
    return img
