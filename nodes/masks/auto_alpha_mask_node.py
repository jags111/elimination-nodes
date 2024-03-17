"""Method signatures automatically generated

pyenv local 3.10.6"""

import torch
from typing import Tuple
from ...utils.tensor_utils import TensorImgUtils
from ...segment.chromakey import ChromaKey


class AutoAlphaMaskNode:
    def __init__(self):
        pass

    CATEGORY = "image"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "main"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (
                    [
                        "remove_white_bg",
                        "remove_black_bg",
                        "remove_white_grey_bg",
                        "remove_neutrals_bg",
                        "remove_non_neutrals_bg",
                        "custom",
                    ],
                ),
                "invert": (
                    "BOOLEAN",
                    {"default": False, "label_on": "enabled", "label_off": "disabled"},
                ),
                "leniance": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 100,
                        "step": 2,
                        "display": "slider",
                    },
                ),
                "use_custom_color": (
                    "BOOLEAN",
                    {"default": False, "label_on": "enabled", "label_off": "disabled"},
                ),
            },
            "optional": {"custom_bg_rgb": ("STRING", {"default": "255, 255, 255"})},
        }

    def main(
        self,
        image: torch.Tensor,  # [Batch_n, H, W, 3-channel]
        method: str,
        invert: bool,
        leniance: int,
        use_custom_color: bool,
        custom_bg_rgb: str = "255, 255, 255",
    ) -> Tuple[torch.Tensor, ...]:

        image = TensorImgUtils.convert_to_type(image, "CHW")
        chroma_key = ChromaKey()
        error_margin = leniance / 100.0

        if method == "remove_white_bg":
            _, alpha, _ = chroma_key.remove_white(image, error_margin)
        elif method == "remove_white_grey_bg":
            _, alpha, _ = chroma_key.remove_specific_rgb(
                image, (128, 128, 128), error_margin * 2
            )
        elif method == "remove_black_bg":
            _, alpha, _ = chroma_key.remove_black(image, error_margin)
        elif method == "remove_non_neutrals_bg":
            _, alpha, _ = chroma_key.remove_non_neutrals(image, error_margin)
        elif method == "remove_neutrals_bg":
            _, alpha, _ = chroma_key.remove_neutrals(image, error_margin)
        elif method == "custom" or use_custom_color:
            try:
                custom_bg_rgb = custom_bg_rgb.split(",")
                custom_bg_rgb = [int(i) for i in custom_bg_rgb]
            except Exception:
                custom_bg_rgb = [235, 235, 235]
            _, alpha, _ = chroma_key.remove_specific_rgb(
                image, custom_bg_rgb, error_margin
            )

        if not invert:
            alpha = 1 - alpha

        # NOTE: masks don't have batch dimension
        return (alpha,)

