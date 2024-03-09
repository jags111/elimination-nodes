import torch
from typing import Tuple


class TensorImgUtils:
    @staticmethod
    def test_squeeze_batch(tensor: torch.Tensor, strict=False) -> torch.Tensor:
        # Check if the tensor has a batch dimension (size 4)
        if tensor.dim() == 4:
            if tensor.size(0) == 1 or not strict:
                # If it has a batch dimension with size 1, remove it. It represents a single image.
                return tensor.squeeze(0)
            else:
                raise ValueError(
                    f"This is not a single image. It's a batch of {tensor.size(0)} images."
                )
        else:
            # Otherwise, it doesn't have a batch dimension, so just return the tensor as is.
            return tensor

    @staticmethod
    def height_width(image: torch.Tensor) -> Tuple[int, int]:
        """Like torchvision.transforms methods, this method assumes Tensor to
        have [..., H, W] shape, where ... means an arbitrary number of leading
        dimensions
        """
        return image.shape[-2:]

    @staticmethod
    def smaller_axis(image: torch.Tensor) -> int:
        h, w = TensorImgUtils.height_width(image)
        return 2 if h < w else 3

    @staticmethod
    def larger_axis(image: torch.Tensor) -> int:
        h, w = TensorImgUtils.height_width(image)
        return 2 if h > w else 3
