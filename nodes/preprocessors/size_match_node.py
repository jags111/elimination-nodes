"""Method signatures automatically generated"""

import torch
from typing import Tuple, Union
from ...types_interfaces.image_tensor_types import ImageTensorTypes as itt
from ...utils.tensor_utils import TensorImgUtils
from ...equalize.equalize_size import SizeMatcher


class SizeMatchNode:
    def __init__(self):
        pass

    CATEGORY = "image"
    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
    )
    FUNCTION = "main"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "method": (
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
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                # "resize_which": (["two_images", "image_and_mask", "two_masks"],),
            },
            "optional": {
                "mask_1": ("MASK", {"default": None}),
                "mask_2": ("MASK", {"default": None}),
            },
        }

    def main(
        self,
        method: str,
        image_1: itt.B_H_W_RGB_Tensor,
        image_2: itt.B_H_W_RGB_Tensor_Optional = None,
        mask_1: Union[itt.H_W_Tensor_Optional, itt.H_W_A_Tensor_Optional] = None,
        mask_2: Union[itt.H_W_Tensor_Optional, itt.H_W_A_Tensor_Optional] = None,
    ) -> Tuple[torch.Tensor, ...]:

        # Handle when alpha is in shape [H, W]
        if mask_1 != None and mask_1.dim() == 2:
            mask_1 = mask_1.unsqueeze(0)
        if mask_2 != None and mask_2.dim() == 2:
            mask_2 = mask_2.unsqueeze(0)

        # Recurse so that batches are handled
        if image_1.dim() == 4 or image_2.dim() == 4:
            return (
                torch.cat(
                    tuple(
                        self.size_match_singletons(
                            method,
                            image_1[i] if image_1.dim() == 4 else image_1,
                            image_2[i] if image_2.dim() == 4 else image_2,
                        )[0]
                        for i in range(image_1.size(0))
                    ),
                    dim=0,  # Concat along batch dimension
                ),
                torch.cat(
                    tuple(
                        self.size_match_singletons(
                            method,
                            image_1[i] if image_1.dim() == 4 else image_1,
                            image_2[i] if image_2.dim() == 4 else image_2,
                        )[1]
                        for i in range(image_2.size(0))
                    ),
                    dim=0,  # Concat along batch dimension
                ),  # Include comma to force tuple type despite single element
            )

        # NOTE: comfy using [batch, height, width, channels], but we are recurring over batch
        image_1 = TensorImgUtils.to_chw_singleton(image_1)
        image_2 = TensorImgUtils.to_chw_singleton(image_2)
        # NOTE: masks don't have batch dimension either way

        # NOTE: comfy ImageLoader always takes rgb, gives alpha as separate output (inverted) (mask)
        # If no alpha channel, comfy makes default 64x64 alpha channel mask.
        # In that case, infer an alpha layer from the cutout image automatically.
        # if cutout_alpha.size(1) != cutout.size(1) or cutout_alpha.size(
        #     2
        # ) != cutout.size(2):
        #     chroma_key = ChromaKey()
        #     _, cutout_alpha, _ = chroma_key.infer_bg_and_remove(cutout)

        image_1, image_2 = SizeMatcher().size_match_by_method_str(
            image_1, image_2, method
        )

        image_1 = TensorImgUtils.to_hwc_singleton(image_1)
        image_2 = TensorImgUtils.to_hwc_singleton(image_2)

        image_1 = image_1.unsqueeze(0)
        image_2 = image_2.unsqueeze(0)

        return (
            image_1,
            image_2,
        )

    def size_match_singletons(
        self,
        method: str,
        image_1: itt.H_W_RGB_Tensor,
        image_2: itt.H_W_RGB_Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Resize the input images to match the size of the larger image.

        Args:
            image_1 (torch.Tensor): The first input image tensor.
            image_2 (torch.Tensor): The second input image tensor.
            method (str): The method to use for resizing the images.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The resized image tensors.
        """
        ret = SizeMatcher().size_match_by_method_str(image_1, image_2, method)
        print(f"ret: {ret[0].shape}")
        print(f"ret: {ret[1].shape}")
        return ret