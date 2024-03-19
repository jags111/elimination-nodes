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
    def __init__(self):
        self.start_frame_keyword = "start"

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        return {
            "required": {
                "parallax_config": ("parallax_config",),
            },
            "optional": {
                "image_path": (sorted(files), {"image_upload": True}),
            },
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

        self.__set_config(parallax_config)

        # Use most recent frame if it exists, otherwise use the input image (first iteration of the project)
        if self.get_project_frame_ct() == 0:
            cur_image_path = folder_paths.get_annotated_filepath(image_path)
            # Create the project directory, and save the loaded image as the start frame
            output_path = self.__get_parallax_proj_dirpath()
            os.makedirs(output_path, exist_ok=True)
            Image.open(cur_image_path).save(os.path.join(output_path, "original.png"))
        else:
            cur_image_path = self.try_get_start_img()

        img = Image.open(cur_image_path)

        # If the image has exif data, rotate it to the correct orientation and remove the exif data
        img_raw = ImageOps.exif_transpose(img)

        # If in 32-bit mode, normalize the image appropriately
        if img_raw.mode == "I":
            img_raw = img.point(lambda i: i * (1 / 255))

        # If image is rgba, create mask
        if "A" in img_raw.getbands():
            mask = np.array(img_raw.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros(
                (img_raw.height, img_raw.width), dtype=torch.float32, device="cpu"
            )
        mask = mask.unsqueeze(0)  # Add a batch dimension to mask

        # Convert the image to RGB, TODO: should be able to handle rgba throughout the pipeline easily
        rgb_image = img_raw.convert("RGB")
        # Normalize the image's rgb values to {x | x ∈ float32, 0 <= x <= 1}
        rgb_image = np.array(rgb_image).astype(np.float32) / 255.0
        # Convert the image to a tensor (torch.from_numpy gives a tensor with the format of [H, W, C])
        rgb_image = torch.from_numpy(rgb_image)[
            None,
        ]  # Add a batch dimension, new format is [B, H, W, C]

        return (rgb_image, mask)

    def get_project_frame_ct(self):
        if not self.__project_dir_exists():
            return 0
        return len(
            [f for f in os.listdir(self.__get_parallax_proj_dirpath()) if "start" in f]
        )

    def try_get_start_img(self):

        output_path = self.__get_parallax_proj_dirpath()
        cur_image_path = False
        if os.path.exists(output_path):
            start_images = [f for f in os.listdir(output_path) if "start" in f]
            print(f"[LoadParallaxStart] start_images: {start_images}")
            if len(start_images) > 0:
                start_images.sort()
                cur_image_path = os.path.join(output_path, start_images[-1])
                print(f"[LoadParallaxStart] cur_image_path: {cur_image_path}")
        return cur_image_path

    def __set_config(self, parallax_config: str) -> None:
        self.parallax_config = json.loads(parallax_config)

    def __get_proj_name(self):
        return self.parallax_config["unique_project_name"]

    def __project_dir_exists(self):
        return os.path.exists(self.__get_parallax_proj_dirpath())

    def __get_parallax_proj_dirpath(self):
        node_dir = os.path.dirname(os.path.abspath(__file__)).split(
            "elimination-nodes"
        )[0]
        node_dir = os.path.join(node_dir, "elimination-nodes", "nodes", "file_system")
        output_path = os.path.join(node_dir, self.__get_proj_name())
        return output_path

    @classmethod
    def IS_CHANGED(s, image):
        return LoadParallaxStartNode.get_project_frame_ct()
