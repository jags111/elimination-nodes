import torch
from typing import Tuple
from .utils.tensor_utils import TensorImgUtils
from .equalize.equalize_size import SizeMatcher


class CompositeAlphaToBase:
    def __init__(self):
        pass

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
            },
        }

    def main(
        self,
        base_image: torch.Tensor,  # [Batch_n, H, W, 3-channel]
        cutout: torch.Tensor,  # [Batch_n, H, W, 3-channel]
        cutout_alpha: torch.Tensor,  # [H, W, 1-channel]
        size_matching_method: str = "cover_crop_center",
    ) -> Tuple[torch.Tensor, ...]:

        # Recurse by batch dimension if present
        if base_image.dim() == 4 or cutout.dim() == 4 or cutout_alpha.dim() == 4:
            return tuple(
                self.main(
                    base_image[i] if base_image.dim() == 4 else base_image,
                    cutout[i] if cutout.dim() == 4 else cutout,
                    cutout_alpha[i] if cutout_alpha.dim() == 4 else cutout_alpha,
                    size_matching_method,
                )
                for i in range(base_image.size(0))
            )

        # NOTE: comfy using [batch, height, width, channels]
        base_image = self.__to_chw_singleton(base_image)
        cutout = self.__to_chw_singleton(cutout)

        alpha_cutout = self.recombine_alpha(cutout, cutout_alpha)
        base_image, alpha_cutout = self.match_size(
            base_image, alpha_cutout, size_matching_method
        )

        ret = self.composite(base_image, alpha_cutout)
        ret = self.__to_hwc_singleton(ret)

        # NOTE: return type
        return (ret,)

    def __to_chw_singleton(self, tensor: torch.Tensor) -> torch.Tensor:
        # If
        return TensorImgUtils.test_squeeze_batch(tensor).permute(2, 0, 1)

    def __to_hwc_singleton(self, tensor: torch.Tensor) -> torch.Tensor:
        return TensorImgUtils.test_squeeze_batch(tensor).permute(1, 2, 0)

    def __to_chw(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.permute(0, 3, 1, 2)

    def __to_hwc(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.permute(0, 2, 3, 1)

    def composite(
        self, base_image: torch.Tensor, alpha_cutout: torch.Tensor
    ) -> torch.Tensor:

        alpha_cutout = TensorImgUtils.test_squeeze_batch(alpha_cutout)
        base_image = TensorImgUtils.test_squeeze_batch(base_image)

        # Extract the alpha channel from the cutout
        alpha_only = alpha_cutout[3, :, :]

        # All pixels that are not transparent should be from the cutout
        composite = alpha_cutout[:3, :, :] * alpha_only + base_image * (1 - alpha_only)

        return composite

    def match_size(
        self,
        base_image: torch.Tensor,
        alpha_cutout: torch.Tensor,
        size_matching_method: str,
    ) -> Tuple[torch.Tensor]:
        base_image = base_image.unsqueeze(0)
        alpha_cutout = alpha_cutout.unsqueeze(0)

        size_matcher = SizeMatcher()
        if size_matching_method == "fit_center":
            base_image, alpha_cutout = size_matcher.fit_maintain_pad(
                base_image, alpha_cutout
            )
        elif size_matching_method == "cover_crop_center":
            base_image, alpha_cutout = size_matcher.cover_maintain(
                base_image, alpha_cutout, center=True
            )
        elif size_matching_method == "cover_crop":
            base_image, alpha_cutout = size_matcher.cover_maintain(
                base_image, alpha_cutout, center=False
            )
        elif size_matching_method == "fill":
            base_image, alpha_cutout = size_matcher.cover_distort(
                base_image, alpha_cutout
            )
        elif size_matching_method == "crop_larger_center":
            base_image, alpha_cutout = size_matcher.crop_larger_to_match(
                base_image, alpha_cutout, center=True
            )
        elif size_matching_method == "crop_larger_topleft":
            base_image, alpha_cutout = size_matcher.crop_larger_to_match(
                base_image, alpha_cutout, center=False
            )
        elif size_matching_method == "center_dont_resize":
            base_image, alpha_cutout = size_matcher.pad_smaller(
                base_image, alpha_cutout
            )

        return (base_image, alpha_cutout)

    def recombine_alpha(self, image: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Recombine the image and alpha channel into a single tensor.

        Args:
            image (torch.Tensor): The image tensor.
            alpha (torch.Tensor): The alpha channel tensor.

        Returns:
            torch.Tensor: The recombined image tensor.
        """
        # invert the alpha channel if it was a mask
        alpha = 1 - alpha
        return torch.cat((image, alpha), 0)
