"""Method signatures automatically generated

pyenv local 3.10.6"""

import torch
import json
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


class LayerShifterNode:
    CATEGORY = "image"
    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )
    FUNCTION = "main"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "parallax_config": ("parallax_config",),
            },
            "optional": {
                "object_mask_1": ("MASK",),
                "object_mask_2": ("MASK",),
                "object_mask_3": ("MASK",),
                "object_mask_4": ("MASK",),
                "object_mask_5": ("MASK",),
                "object_mask_6": ("MASK",),
            },
        }

    def main(
        self,
        input_image: torch.Tensor,  # [Batch_n, H, W, 3-channel]
        parallax_config: str,  # [Batch_n, H, W, 3-channel]
    ) -> Tuple[torch.Tensor, ...]:

        self.TRANSPARENT = 0
        self.OPAQUE = 1

        # squeeze batch dimension
        input_image = TensorImgUtils.test_squeeze_batch(input_image)

        # parallax_config json string to dict
        parallax_config = json.loads(parallax_config)

        # Create mask Tensor.
        mask_tensor = torch.zeros((input_image.shape[0], input_image.shape[1]), dtype=torch.uint8)

        print(f"LayerShifterNode: input_image.shape: {input_image.shape}")
        print(f"LayerShifterNode: parallax_config: {parallax_config}")

        cur_height = 0
        max_height = input_image.shape[0]
        for layer in parallax_config:
            if cur_height >= max_height:
                break
            
            print(f"LayerShifterNode: layer: {layer}")
            if layer["height"] == 0 or layer["velocity"] == 0:
                cur_height += layer["height"]
            else:
                height = int(layer["height"])
                top = cur_height
                bottom = cur_height + height
                velocity = round(float(layer["velocity"]))
                if bottom > max_height:
                    bottom = max_height
                    height = bottom - top

                # Shift the layer
                input_image = self.shift_horizontal_slice(
                    input_image,
                    top,
                    bottom,
                    velocity,
                )
                # Make shifted region transparent in the mask
                mask_tensor = self.add_mask_to_shifted_region(
                    mask_tensor,
                    top,
                    bottom,
                    velocity,
                )
                cur_height += height


        input_image = TensorImgUtils.test_unsqueeze_batch(input_image)

        print(f"LayerShifterNode: input_image.shape: {input_image.shape}")
        print(f"LayerShifterNode: mask_tensor.shape: {mask_tensor.shape}")

        return (
            input_image,
            mask_tensor,
        )

    def add_mask_to_shifted_region(
        self, mask_tensor, start_row, end_row, shift_pixels
    ):  # [H, W, RGB]
        mask_tensor[start_row:end_row, mask_tensor.shape[1] - shift_pixels:] = self.OPAQUE
        return mask_tensor

    def shift_horizontal_slice(self, image_tensor, start_row, end_row, shift_pixels):
        # Extract the horizontal slice to be shifted
        slice_to_shift = image_tensor[start_row:end_row, :, :]

        # Apply the left shift to the slice
        shifted_slice = torch.roll(slice_to_shift, -shift_pixels, dims=1)

        # Concatenate the shifted slice with the rest of the image tensor
        shifted_image_tensor = torch.cat([shifted_slice, image_tensor[end_row:, :, :]], dim=0)
        shifted_image_tensor = torch.cat([image_tensor[:start_row, :, :], shifted_image_tensor], dim=0)

        return shifted_image_tensor