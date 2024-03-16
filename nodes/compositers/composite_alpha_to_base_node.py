"""Method signatures automatically generated

pyenv local 3.10.6"""

import torch
from typing import Tuple


try:
    from ...utils.tensor_utils import TensorImgUtils
    from ...equalize.equalize_size import SizeMatcher
    from ...segment.chromakey import ChromaKey
except ImportError:
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.tensor_utils import TensorImgUtils
    from equalize.equalize_size import SizeMatcher
    from segment.chromakey import ChromaKey


class CompositeCutoutOnBaseNode:
    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "cutout": ("IMAGE",),
                "cutout_alpha": ("MASK",),
                "size_matching_method": (
                    [
                        "cover_crop_center",
                        "cover_crop",
                        "center_dont_resize",
                        "fill",
                        "fit_center",
                        "crop_larger_center",
                        "crop_larger_topleft",
                    ],
                ),
                "invert_cutout": (
                    "BOOLEAN",
                    {"default": False, "label_on": "enabled", "label_off": "disabled"},
                ),
            },
        }

    def main(
        self,
        base_image: torch.Tensor,  # [Batch_n, H, W, 3-channel]
        cutout: torch.Tensor,  # [Batch_n, H, W, 3-channel]
        cutout_alpha: torch.Tensor,  # [H, W, 1-channel]
        size_matching_method: str,
        invert_cutout: bool,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Main method for performing composite alpha-to-base operation.

        If any of the tensors have a batch dimension, it iterates over each batch element (i) and applies the self.main() function to corresponding slices of each tensor along the batch dimension.
        The result of each iteration is concatenated along the batch dimension using torch.cat() to form a new batch.
        Finally, the tuple of concatenated tensors is returned.
        This approach ensures that if any of the input tensors are batches of images, the processing is applied batch-wise across all images in the batch. This is a common strategy when dealing with deep learning models that are designed to handle batches of data efficiently.

        Args:
            base_image (torch.Tensor): The base image tensor with shape [Batch_n, H, W, 3-channel].
            cutout (torch.Tensor): The cutout image tensor with shape [Batch_n, H, W, 3-channel].
            cutout_alpha (torch.Tensor): The cutout alpha channel tensor with shape [H, W, 1-channel].
            size_matching_method (str, optional): The method for matching the size of base_image and alpha_cutout.
                Defaults to "cover_crop_center".
            invert_cutout (str, optional): Whether to invert the cutout alpha channel. Defaults to "False".

        Returns:
            Tuple[torch.Tensor, ...]: A tuple containing the resulting composite image tensor with shape [Batch_n, H, W, 3-channel].

        Note:
            - The base_image, cutout, and cutout_alpha tensors can have an additional batch dimension.
            - The base_image and cutout tensors are expected to have shape [Batch_n, H, W, 3-channel].
            - The cutout_alpha tensor is expected to have shape [H, W, 1-channel].
            - The size_matching_method determines how the size of base_image and alpha_cutout is matched.
        """
        # Handle when alpha layer is in shape [H, W]
        if cutout_alpha.dim() == 2:
            cutout_alpha = cutout_alpha.unsqueeze(0)

        # Handles batches by recursing over batch dimension when present (comfy default is BHWC)
        if base_image.dim() == 4 or cutout.dim() == 4 or cutout_alpha.dim() == 4:
            return (
                torch.cat(
                    tuple(
                        self.main(
                            base_image[i] if base_image.dim() == 4 else base_image,
                            cutout[i] if cutout.dim() == 4 else cutout,
                            (
                                cutout_alpha[i]
                                if cutout_alpha.dim() == 4
                                else cutout_alpha
                            ),
                            size_matching_method,
                            invert_cutout,
                        )
                        for i in range(base_image.size(0))
                    ),
                    dim=0,  # Concat along batch dimension
                ),  # Include comma to force tuple return type even when single element
            )

        base_image = TensorImgUtils.convert_to_type(base_image, "CHW")
        cutout = TensorImgUtils.convert_to_type(cutout, "CHW")

        # If base_image is rgba for some reason, remove alpha channel
        if base_image.size(0) == 4:
            base_image = base_image[:3, :, :]

        # Comfy creates a default 64x64 mask if rgb image was loaded, so we check for size mismatch to know the image was rgb and didn't have an alpha channel at load time
        if cutout_alpha.size(1) != cutout.size(1) or cutout_alpha.size(
            2
        ) != cutout.size(2):
            print(
                f"Cutout alpha size {cutout_alpha.size()} does not match cutout size {cutout.size()}. Inferring alpha channel automatically."
            )
            _, cutout_alpha, _ = ChromaKey().infer_bg_and_remove(cutout)

        if invert_cutout:
            cutout_alpha = 1 - cutout_alpha

        alpha_cutout = self.recombine_alpha(
            cutout, cutout_alpha
        )  # recombine just so resize is easier
        base_image, alpha_cutout = self.match_size(
            base_image, alpha_cutout, size_matching_method
        )

        return TensorImgUtils.convert_to_type(
            self.composite(base_image, alpha_cutout), "BHWC"
        )

    def composite(self, base_image: torch.Tensor, cutout: torch.Tensor) -> torch.Tensor:
        """
        Composites the cutout image onto the base image using the alpha channel.

        Args:
            base_image (torch.Tensor): The base image onto which the cutout will be composited.
            cutout (torch.Tensor): The cutout image to be composited onto the base image.

        Returns:
            torch.Tensor: The composited image.

        """
        cutout = TensorImgUtils.test_squeeze_batch(cutout)
        base_image = TensorImgUtils.test_squeeze_batch(base_image)

        # Extract the alpha channel from the cutout
        alpha_only = cutout[3, :, :]

        # All pixels that are not transparent should be from the cutout
        composite = cutout[:3, :, :] * alpha_only + base_image * (1 - alpha_only)

        return composite

    def match_size(
        self,
        base_image: torch.Tensor,
        alpha_cutout: torch.Tensor,
        size_matching_method: str,
    ) -> Tuple[torch.Tensor]:
        """
        Matches the size of the base image and alpha cutout according to the specified size matching method.

        Args:
            base_image (torch.Tensor): The base image tensor.
            alpha_cutout (torch.Tensor): The alpha cutout tensor.
            size_matching_method (str): The method to use for size matching. Available options are:
                - "fit_center": Fits the alpha cutout inside the base image while maintaining aspect ratio and padding if necessary.
                - "cover_crop_center": Covers the base image with the alpha cutout while maintaining aspect ratio and cropping if necessary, with the alpha cutout centered.
                - "cover_crop": Covers the base image with the alpha cutout while maintaining aspect ratio and cropping if necessary, without centering the alpha cutout.
                - "fill": Covers the base image with the alpha cutout while distorting the aspect ratio if necessary.
                - "crop_larger_center": Crops the base image to match the size of the alpha cutout, with the alpha cutout centered.
                - "crop_larger_topleft": Crops the base image to match the size of the alpha cutout, without centering the alpha cutout.
                - "center_dont_resize": Pads the smaller image to match the size of the larger image.

        Returns:
            Tuple[torch.Tensor]: A tuple containing the resized base image and alpha cutout tensors.
        """
        return SizeMatcher().size_match_by_method_str(
            base_image, alpha_cutout, size_matching_method
        )

    def recombine_alpha(self, image: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Recombine the image and alpha channel into a single tensor.

        Args:
            image (torch.Tensor): The image tensor.
            alpha (torch.Tensor): The alpha channel tensor.

        Returns:
            torch.Tensor: The recombined image tensor.
        """
        if alpha.dim() == 2:
            alpha = alpha.unsqueeze(0)
        alpha = 1 - alpha
        return torch.cat((image, alpha), 0)
