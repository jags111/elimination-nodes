import torch
from torchvision import transforms
from typing import Tuple
from utils.tensor_utils import TensorImgUtils


class SizeMatcher:
    def __init__(self, mode: str):
        self.mode = mode

    def crop_to_match(
        self, image: torch.Tensor, target_dimensions, center=False
    ) -> torch.Tensor:
        """
        Crop the input image to match the specified target dimensions.

        Args:
            image (torch.Tensor): The input image tensor.
            target_dimensions (tuple): The target dimensions (height, width) to crop the image to.
            center (bool, optional): If True, the image will be cropped from the center. Defaults to False.

        Returns:
            torch.Tensor: The cropped image tensor.
        """
        if center:
            return transforms.CenterCrop(target_dimensions)(image)
        return image[:, :, : target_dimensions[0], : target_dimensions[1]]

    def fit_maintain_pad(
        self, image_1: torch.Tensor, image_2: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Scale the smaller of the two images to fit inside dimensions of the larger
        image as much as possible while maintaining aspect ratio, then pad the
        smaller image to match the dimensions of the larger image so the tensors
        have the same shape.

        Args:
            image_1 (torch.Tensor): The first input image.
            image_2 (torch.Tensor): The second input image.

        Returns:
            Tuple[torch.Tensor]: A tuple containing the resized images.
        """
        h1, w1 = TensorImgUtils.height_width(image_1)
        h2, w2 = TensorImgUtils.height_width(image_2)
        if h1 * w1 > h2 * w2:
            target_axis = TensorImgUtils.smaller_axis(image_2)
            image_2 = self.scale_side(image_2, image_1.shape[target_axis], target_axis)
        else:
            target_axis = TensorImgUtils.smaller_axis(image_1)
            image_1 = self.scale_side(image_1, image_2.shape[target_axis], target_axis)

        return (image_1, image_2)

    def cover_distort(
        self, image_1: torch.Tensor, image_2: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Resizes the smaller of the two images to cover the dimensions of the larger image.
        Scale and distort the smaller image to match the larger image's dimensions.

        Args:
            image_1 (torch.Tensor): The first input image.
            image_2 (torch.Tensor): The second input image.

        Returns:
            Tuple[torch.Tensor]: A tuple containing the resized images.
        """
        h1, w1 = TensorImgUtils.height_width(image_1)
        h2, w2 = TensorImgUtils.height_width(image_2)
        if h1 * w1 > h2 * w2:
            image_2 = transforms.Resize((h1, w1))(image_2)
        else:
            image_1 = transforms.Resize((h2, w2))(image_1)

        return (image_1, image_2)

    def crop_larger_to_match(
        self, image_1: torch.Tensor, image_2: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Crop the larger of the two images to match the dimensions of the smaller image.

        Args:
            image_1 (torch.Tensor): The first input image.
            image_2 (torch.Tensor): The second input image.

        Returns:
            Tuple[torch.Tensor]: A tuple containing the resized images.
        """
        h1, w1 = TensorImgUtils.height_width(image_1)
        h2, w2 = TensorImgUtils.height_width(image_2)
        if h1 * w1 > h2 * w2:
            image_1 = self.crop_to_match(image_1, (h2, w2), center=True)
        else:
            image_2 = self.crop_to_match(image_2, (h1, w1), center=True)

        return (image_1, image_2)

    def cover_maintain(
        self, image_1: torch.Tensor, image_2: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Resizes the smaller of the two images to cover the dimensions of the larger image.
        Scale while maintaining aspect ratio, then crop to match the larger image's dimensions
        starting from the top-left corner.

        Args:
            image_1 (torch.Tensor): The first input image.
            image_2 (torch.Tensor): The second input image.

        Returns:
            Tuple[torch.Tensor]: A tuple containing the resized images.
        """
        h1, w1 = TensorImgUtils.height_width(image_1)
        h2, w2 = TensorImgUtils.height_width(image_2)
        if h1 * w1 > h2 * w2:
            target_axis = TensorImgUtils.smaller_axis(image_2)
            image_2 = self.scale_side(image_2, image_1.shape[target_axis], target_axis)
            image_2 = self.crop_to_match(image_2, (h1, w1), center=True)
        else:
            target_axis = TensorImgUtils.smaller_axis(image_1)
            image_1 = self.scale_side(image_1, image_2.shape[target_axis], target_axis)
            image_1 = self.crop_to_match(image_1, (h2, w2), center=True)

        return (image_1, image_2)
