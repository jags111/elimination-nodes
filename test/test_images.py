import os
from PIL import Image
import numpy as np
import cv2
import random
import time
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
from moviepy.editor import VideoFileClip
import sys
from termcolor import colored

try:
    from ..constants import VIDEO_EXTENSION_LIST, PICTURE_EXTENSION_LIST
    from ..types_interfaces.image_tensor_types import ImageTensorTypes as itt
    from .results_webview import ComparisonGrid
except (ImportError, ModuleNotFoundError):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from constants import VIDEO_EXTENSION_LIST, PICTURE_EXTENSION_LIST
    from types_interfaces.image_tensor_types import ImageTensorTypes as itt
    from results_webview import ComparisonGrid


class TestImages:
    def __init__(self, max_width=224, max_height=224):
        self.repo_root = os.path.dirname(os.path.dirname(__file__))
        self.path = os.path.join(self.repo_root, "test")
        self.img_dir = os.path.join(self.path, "test-images")

        self.images = {}
        self.__map_images()

        self.max_width = max_width
        self.max_height = max_height

    def set_max_dimensions(self, width: int, height: int):
        self.max_width = int(width)
        self.max_height = int(height)

    def __resize_image(self, img: Image.Image) -> Image.Image:
        """Resizing image data before it is a tensor is considerably faster + higher fidelity."""
        ratio = min(self.max_width / img.width, self.max_height / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        return img.resize(new_size, Image.ANTIALIAS)

    def __path_to_chw_tensor(self, img_path: str) -> itt.C_H_W_Tensor:
        img = Image.open(img_path)
        img = transforms.ToTensor()(img)
        return img

    def get_media(
        self,
        count=1,
        tags=[],
        prioritize_real=True,
        is_video=False,
        is_picture=True,
        mb_limit=5,
    ):
        if not prioritize_real:
            return self.__generate_images(count)

        print(
            f"Getting {count} random images with tags {tags} and size limit {mb_limit}MB"
        )
        ret = []
        tags = set(tags)

        for key, img in self.images.items():
            if is_video and not img["is_video"]:
                continue
            if is_picture and not img["is_picture"]:
                continue
            if img["file_size_mb"] > mb_limit:
                continue
            if tags and not tags.issubset(set(img["tags"])):
                continue
            ret.append(img)
            if len(ret) == count:
                break

        if len(ret) != count:
            print(
                f"Only found {len(ret)} images with tags {tags} and size limit {mb_limit}MB",
                "\nGenerating random noise images to supplement.",
            )
        all_images = self.__generate_images(count - len(ret))
        to_tensor = transforms.ToTensor()
        print(
            f"Resizing {len(ret)} images to {self.max_width}x{self.max_height} while maintaining aspect ratio."
        )
        for img in ret:
            pil = Image.open(img["fullpath"])
            if (
                "rgba" in tags
                or "alpha" in tags
                or "alpha-layer" in tags
                or "alpha-layers" in tags
                or "cutout" in tags
                or "icon" in tags
                or "mask" in tags
            ):
                pil = pil.convert("RGBA")
                print(colored(f"Converting {img['file']} to RGBA", 'red'))
            else:
                pil = pil.convert("RGB")
                print(colored(f"Converting {img['file']} to RGB", 'green'))
            as_tensor = to_tensor(self.__resize_image(pil))
            print(f"Resized {img['file']} to {as_tensor.size()}")
            print(f"Shape: {as_tensor.shape}")
            all_images.append(to_tensor(self.__resize_image(pil)))

        return all_images

    def __generate_images(self, count):
        if count > 0:
            print(f"Generating {count} images with random noise.")
        return [torch.rand(3, self.max_height, self.max_width) for _ in range(count)]

    def __is_video(self, file_path):
        _, file_ext = os.path.splitext(file_path)
        return file_ext.lower() in VIDEO_EXTENSION_LIST

    def __is_picture(self, file_path):
        _, file_ext = os.path.splitext(file_path)
        return file_ext.lower() in PICTURE_EXTENSION_LIST

    def __map_images(self):
        print(f"Mapping images from test-images folder")
        for root, dirs, files in os.walk(self.img_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path) and (
                    self.__is_video(file_path) or self.__is_picture(file_path)
                ):
                    tags = set()
                    for path_path in os.path.relpath(root, self.img_dir).split(os.sep):
                        tags.add(path_path)
                    width, height = self.get_dimensions(file_path)
                    file_size = os.path.getsize(file_path)
                    self.images[file] = {
                        "fullpath": file_path,
                        "relpath": os.path.relpath(file_path, self.img_dir),
                        "ext": os.path.splitext(file_path)[1],
                        "tags": list(tags),
                        "file": file,
                        "basename": os.path.basename(file),
                        "is_video": self.__is_video(file_path),
                        "is_picture": self.__is_picture(file_path),
                        "width": width,
                        "height": height,
                        # in megabytes
                        "file_size_mb": round(file_size / (1024 * 1024), 2),
                    }

    def get_dimensions(self, file_path):
        _, file_ext = os.path.splitext(file_path)
        if file_ext.lower() in VIDEO_EXTENSION_LIST:
            with VideoFileClip(file_path) as video:
                width, height = video.size
        elif file_ext.lower() in PICTURE_EXTENSION_LIST:
            with Image.open(file_path) as img:
                width, height = img.size
        else:
            raise ValueError(f"File extension {file_ext} not supported")
        return width, height

    def clean_dir(self):
        for root, dirs, files in os.walk(self.img_dir):
            for index, file in enumerate(files):
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    try:
                        width, height = self.get_dimensions(file_path)
                        file_name, file_ext = os.path.splitext(file)
                        path_as_name = (
                            os.path.relpath(root, self.img_dir)
                            .replace("/", "-")
                            .replace("\\", "-")
                        )
                        new_name = f"{path_as_name}-H{height}px_W{width}px-{file_ext.replace('.','')}-{index}{file_ext}"
                        # print(f"\nRenaming {file} to {new_name}\n")
                        os.rename(file_path, os.path.join(root, new_name))
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
