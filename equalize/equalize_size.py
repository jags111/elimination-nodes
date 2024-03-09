import torch
from torchvision import transforms
from typing import Tuple
from utils.tensor_utils import TensorImgUtils
from transform.scale import ImageScaler
from termcolor import colored


class SizeMatcher:
    def __init__(self):
        self.scale = ImageScaler()

    def pad_smaller(
        self, image_1: torch.Tensor, image_2: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Pads the smaller image to match the dimensions of the larger image.

        Args:
            image_1 (torch.Tensor): The first input image.
            image_2 (torch.Tensor): The second input image.

        Returns:
            Tuple[torch.Tensor]: A tuple containing the resized images.
        """
        h1, w1 = TensorImgUtils.height_width(image_1)
        h2, w2 = TensorImgUtils.height_width(image_2)
        if h1 * w1 > h2 * w2:
            top_pad = (h1 - h2) // 2
            bot_pad = h1 - h2 - top_pad
            left_pad = (w1 - w2) // 2
            right_pad = w1 - w2 - left_pad
            image_2 = transforms.Pad((left_pad, top_pad, right_pad, bot_pad))(image_2)
        else:
            top_pad = (h2 - h1) // 2
            bot_pad = h2 - h1 - top_pad
            left_pad = (w2 - w1) // 2
            right_pad = w2 - w1 - left_pad
            image_1 = transforms.Pad((left_pad, top_pad, right_pad, bot_pad))(image_1)

        return (image_1, image_2)

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
        diff_h = image.shape[2] - target_dimensions[0]
        diff_w = image.shape[3] - target_dimensions[1]
        # Calculate margins for center cropping
        top_margin = diff_h // 2
        bot_margin = top_margin + target_dimensions[0]
        left_margin = diff_w // 2
        right_margin = left_margin + target_dimensions[1]

        if diff_h < 0 and diff_w < 0:
            return image
        elif diff_h < 0 and diff_w >= 0:
            if center:
                return image[:, :, :, left_margin:right_margin]
            else:
                return image[:, :, :, : target_dimensions[1]]
        elif diff_w < 0 and diff_h >= 0:
            if center:
                return image[:, :, top_margin:bot_margin, :]
            else:
                return image[:, :, : target_dimensions[0], :]
        else:
            if center:
                return image[:, :, top_margin:bot_margin, left_margin:right_margin]
            else:
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
            max_scale_factor = min(h1 / h2, w1 / w2)
            image_2 = self.scale.by_side(image_2, int(h2 * max_scale_factor), 2)
            image_2 = transforms.Pad(
                (0, 0, w1 - image_2.shape[3], h1 - image_2.shape[2])
            )(image_2)
        else:
            max_scale_factor = min(h2 / h1, w2 / w1)
            image_1 = self.scale.by_side(image_1, int(h1 * max_scale_factor), 2)
            image_1 = transforms.Pad(
                (0, 0, w2 - image_1.shape[3], h2 - image_1.shape[2])
            )(image_1)

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
        self, image_1: torch.Tensor, image_2: torch.Tensor, center=True
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
        # If both sides are not larger, revert to cover_maintain
        if h1 * w1 > h2 * w2:
            if h1 < h2 or w1 < w2:
                return self.cover_maintain(image_1, image_2)
            image_1 = self.crop_to_match(image_1, (h2, w2), center=center)
        else:
            if h2 < h1 or w2 < w1:
                return self.cover_maintain(image_1, image_2)
            image_2 = self.crop_to_match(image_2, (h1, w1), center=center)

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
            image_2_ret = self.scale.by_side(
                image_2, image_1.shape[target_axis], target_axis
            )
            image_2_ret = self.crop_to_match(image_2_ret, (h1, w1), center=True)
            if image_2_ret.shape[-2:] != image_1.shape[-2:]:
                target_axis = TensorImgUtils.larger_axis(image_2)
                image_2_ret = self.scale.by_side(
                    image_2, image_1.shape[target_axis], target_axis
                )
                image_2_ret = self.crop_to_match(image_2_ret, (h1, w1), center=True)

            return (image_1, image_2_ret)
        else:
            target_axis = TensorImgUtils.smaller_axis(image_1)
            image_1_ret = self.scale.by_side(
                image_1, image_2.shape[target_axis], target_axis
            )
            image_1_ret = self.crop_to_match(image_1_ret, (h2, w2), center=True)
            if image_1_ret.shape[-2:] != image_2.shape[-2:]:
                target_axis = TensorImgUtils.larger_axis(image_1)
                image_1_ret = self.scale.by_side(
                    image_1, image_2.shape[target_axis], target_axis
                )
                image_1_ret = self.crop_to_match(image_1_ret, (h2, w2), center=True)

            return (image_1_ret, image_2)
