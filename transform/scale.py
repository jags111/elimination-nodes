import torch
from torchvision import transforms
from typing import Tuple
from utils.tensor_utils import TensorImgUtils


class ImageScaler:
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        image = image * self.scale_factor
        return {"image": image, "landmarks": landmarks}

    def scale_side(
        self, image: torch.Tensor, target_size: int, axis: int
    ) -> torch.Tensor:
        """
        Scales the given image tensor along the specified axis to the target size.

        Args:
            image (torch.Tensor): The input image tensor.
            target_size (int): The desired size of the scaled side.
            axis (int): The axis along which to scale the image.
                        If axis is 2, the height will be scaled to target_size and the width will be adjusted proportionally.
                        If axis is not 2, the width will be scaled to target_size and the height will be adjusted proportionally.

        Returns:
            torch.Tensor: The scaled image tensor.
        """
        h, w = TensorImgUtils.height_width(image)
        if axis == 2:
            new_h = target_size
            new_w = int((w / h) * target_size)
        else:
            new_w = target_size
            new_h = int((h / w) * target_size)

        return transforms.Resize((new_h, new_w))(image)
