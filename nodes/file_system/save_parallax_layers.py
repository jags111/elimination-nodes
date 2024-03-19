"""Method signatures automatically generated

pyenv local 3.10.6"""

import torch
from torchvision import transforms
from PIL import Image
import os
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


class SaveParallaxLayersNode:
    CATEGORY = "parallax"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "main"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "parallax_config": ("parallax_config",),
            },
        }

    def main(
        self,
        input_image: torch.Tensor,  # [Batch_n, H, W, 3-channel]
        parallax_config: str,  # [Batch_n, H, W, 3-channel]
    ) -> Tuple[torch.Tensor, ...]:

        to_pil = transforms.ToPILImage()

        # squeeze batch dimension
        input_image = TensorImgUtils.test_squeeze_batch(input_image)

        # parallax_config json string to dict
        parallax_config = json.loads(parallax_config)

        max_height = input_image.shape[0]
        file_paths = []
        for layer_index, layer in enumerate(parallax_config):
            print(f"LayerSaveNode: layer_index: {layer_index}")
            if layer["height"] == 0 or layer["velocity"] == 0:
                continue

            layer_image = input_image[layer["top"]:layer["bottom"], :, :]
            layer_image = TensorImgUtils.convert_to_type(layer_image, "CHW")
            print(f"LayerSaveNode: layer_image.shape: {layer_image.shape}")
            pil_image = to_pil(layer_image)
            path = os.path.join("/home/c_byrne/tools/sd/sd-interfaces/ComfyUI/custom_nodes/elimination-nodes/nodes", f"layer_{layer_index}_{layer['height']}_{layer['velocity']}.png")
            print(f"LayerSaveNode: path: {path}")
            file_paths.append(path)
            pil_image.save(path)

            
            if layer["bottom"] > max_height:
                print(f"LayerSaveNode: layer['bottom'] > max_height: {layer['bottom']} > {max_height}")
                break

        
        return (json.dumps(file_paths),)