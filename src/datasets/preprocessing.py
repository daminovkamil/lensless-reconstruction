import numpy as np
import torch

from lensless_helpers.preprocessor import (
    convert_image_to_float,
    force_rgb,
    get_cropped_lensed,
)


def load_measurement(image):
    image = convert_image_to_float(force_rgb(np.array(image)))
    return torch.rot90(torch.from_numpy(image), dims=(-3, -2), k=2)


def crop_gt(image, lensless):
    image = convert_image_to_float(force_rgb(np.array(image)))
    return torch.from_numpy(get_cropped_lensed(image, lensless))
