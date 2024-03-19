"""Method signatures automatically generated

pyenv local 3.10.6"""

import os
import json
import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
from torchvision import transforms
from moviepy.editor import VideoClip, ImageClip, CompositeVideoClip

from typing import Tuple

try:
    from ...utils.tensor_utils import TensorImgUtils
    from ...equalize.equalize_size import SizeMatcher
    from ...segment.chromakey import ChromaKey

    # from ... import folder_paths
    import folder_paths
except ImportError:
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from utils.tensor_utils import TensorImgUtils
    from equalize.equalize_size import SizeMatcher
    from segment.chromakey import ChromaKey
    import folder_paths


class LayerFramesToParallaxVideoNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "parallax_config": ("parallax_config",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "main"
    OUTPUT_NODE = True

    def main(
        self,
        parallax_config: str,  # json string
    ) -> Tuple[str, ...]:

        self.__set_config(parallax_config)

        # If frame count matches the num_iterations, create the video, otherwise do nothing
        cur_frame_ct = self.get_project_frame_ct()
        iterations_needed = self.parallax_config["num_iterations"]
        if cur_frame_ct < iterations_needed:
            return f"Frames in Project: {cur_frame_ct}/{iterations_needed}, ({iterations_needed - cur_frame_ct} more frames before video will be created)."

        self.set_layer_frame_ct()
        self.set_original_dimensions()
        return (self.composite_layer_videoclips(),)

    def set_original_dimensions(self):
        # Get the original dimensions of the first frame
        cur_image_path = self.try_get_start_img()
        if not cur_image_path:
            return
        img = Image.open(cur_image_path)
        self.original_width, self.original_height = img.size

    def set_layer_frame_ct(self):
        # Use number of layer 0 frames as the frame count (bc all layers will have the same number of frames)
        self.layer_frame_ct = len(
            [f for f in os.listdir(self.__get_parallax_proj_dirpath()) if "l0" in f]
        )

    def composite_layer_videoclips(self):
        layer_video_clips = []
        for i, layer in enumerate(self.parallax_config["layers"]):
            layer_video_clips.append(self.create_layer_videoclip(layer, i))

        video_composite = CompositeVideoClip(
            layer_video_clips, size=(self.original_width, self.original_height)
        )
        output_path = self.__get_parallax_proj_dirpath()
        video_ct = len([f for f in os.listdir(output_path) if "parallax_video" in f])
        video_path = os.path.join(output_path, f"parallax_video_{video_ct}.mp4")

        video_composite.write_videofile(
            video_path,
            codec="libx264",
            fps=10,
            preset="medium",
            ffmpeg_params=(
                [
                    "-crf",
                    "18",
                    "-b:v",
                    "2M",
                    "-pix_fmt",
                    "yuv420p",
                    "-profile:v",
                    "high",
                    "-vf",
                    "scale=1920:1080",
                ]
            ),
            threads=12,
        )

        return video_path

    def create_layer_videoclip(self, layer_config, layer_index):
        # Get the layer height and velocity
        layer_velocity = layer_config["velocity"]

        # Get the parallax project directory
        output_path = self.__get_parallax_proj_dirpath()

        # Get the final width: number of steps * layer velocity
        added_width = int(self.layer_frame_ct * layer_velocity)
        final_width = added_width + self.original_width

        # Set the start frame slice by loading "original.png" and slicing by the layer's "top" and "bottom" values
        start_frame = Image.open(os.path.join(output_path, "original.png"))
        start_frame = start_frame.crop(
            (0, layer_config["top"], self.original_width, layer_config["bottom"])
        )

        stitched_image = Image.new("RGB", (final_width, self.original_height))
        stitched_image.paste(start_frame, (0, 0))

        # Stitch each layer frame horizontally, with velocity offset
        x_offset = self.original_width
        for i in range(self.layer_frame_ct):
            layer_frame_path = os.path.join(output_path, f"layer{layer_index}_{i}.png")
            layer_frame = Image.open(layer_frame_path)
            stitched_image.paste(layer_frame, (x_offset, layer_config["top"]))
            x_offset += layer_velocity

        # Set the duration of the videoclip
        duration = float(self.layer_frame_ct) * (
            1.0 / float(self.parallax_config["fps"])
        )

        # Create and return a videoclip from the stitched image
        image_clip = ImageClip(stitched_image)

        def make_frame(t):
            x = int(added_width * (t / duration))
            return image_clip.get_frame(t)[:, x : x + self.original_width]

        return VideoClip(make_frame, duration=duration)

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

    # @classmethod
    # def IS_CHANGED(s, image):
    #     return LoadParallaxStartNode.get_project_frame_ct()
