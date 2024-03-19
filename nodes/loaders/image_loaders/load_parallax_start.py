"""Method signatures automatically generated

pyenv local 3.10.6"""

import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import os
import json

from typing import Tuple


try:
    from ....utils.tensor_utils import TensorImgUtils
    from ....equalize.equalize_size import SizeMatcher
    from ....segment.chromakey import ChromaKey
    # from .... import folder_paths
    import folder_paths
except ImportError:
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from utils.tensor_utils import TensorImgUtils
    from equalize.equalize_size import SizeMatcher
    from segment.chromakey import ChromaKey
    import folder_paths

class LoadParallaxStartNode:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
        {
                "parallax_config": ("parallax_config",),
                    },
                    "optional": 
                    {"image_path": (sorted(files), {"image_upload": True}),
                    }
                }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("parallax_start_frame", "optional_mask")
    FUNCTION = "load_image"

    def load_image(
        self,
        parallax_config: str,  # json string
        image_path: str = None,
    ) -> Tuple[torch.Tensor, ...]:

        parallax_config = json.loads(parallax_config)

        # Check if there is a start image in the parallax project dir
        cur_image_path = self.try_get_start_img(parallax_config["unique_project_name"])
        if not cur_image_path:
            # If no start images found in the parallax project dir (it's the first step), use the input image
            cur_image_path = folder_paths.get_annotated_filepath(image_path)
            
        print(f"[LoadParallaxStart] image_path: {cur_image_path}")

        cur_image_path = folder_paths.get_annotated_filepath(cur_image_path)
        img = Image.open(cur_image_path)

        # If the image has exif data, rotate it to the correct orientation and remove the exif data
        img_raw = ImageOps.exif_transpose(img)

        # If in 32-bit mode, normalize the image appropriately
        if img_raw.mode == 'I':
            img_raw = img.point(lambda i: i * (1 / 255))

        # If image is rgba, create mask
        if 'A' in img_raw.getbands():
            mask = np.array(img_raw.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((img_raw.height, img_raw.width), dtype=torch.float32, device="cpu")
        mask = mask.unsqueeze(0) # Add a batch dimension to mask

        # Convert the image to RGB, TODO: should be able to handle rgba throughout the pipeline easily
        rgb_image = img_raw.convert("RGB")
        # Normalize the image's rgb values to {x | x ∈ float32, 0 <= x <= 1}
        rgb_image = np.array(rgb_image).astype(np.float32) / 255.0
        # Convert the image to a tensor (torch.from_numpy gives a tensor with the format of [H, W, C])
        rgb_image = torch.from_numpy(rgb_image)[None,] # Add a batch dimension, new format is [B, H, W, C]
        
        return (rgb_image, mask)

    def try_get_start_img(self, project_name):
        node_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"\n[LoadParallaxStart] node_dir: {node_dir}")
        output_path = os.path.join(node_dir, project_name)
        print(f"[LoadParallaxStart] output_path: {output_path}")
        cur_image_path = False
        if os.path.exists(output_path):
            start_images = [f for f in os.listdir(output_path) if "start" in f]
            print(f"[LoadParallaxStart] start_images: {start_images}")
            if len(start_images) > 0:
                start_images.sort()
                cur_image_path = os.path.join(output_path, start_images[-1])
                print(f"[LoadParallaxStart] cur_image_path: {cur_image_path}")
        return cur_image_path

    @classmethod
    def IS_CHANGED(s, image):
        return True
