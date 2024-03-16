import os
from PIL import Image
import random
import torch
import torchvision.transforms as transforms
from moviepy.editor import VideoFileClip
import sys
from termcolor import colored

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from constants import VIDEO_EXTENSION_LIST, PICTURE_EXTENSION_LIST
from test.test_tools_constants import *
from types_interfaces.image_tensor_types import ImageTensorTypes as itt
from utils.logger import _log


class TestImages:
    def __init__(self, max_width=224, max_height=224):
        self.repo_root = os.path.dirname(os.path.dirname(__file__))
        self.path = os.path.join(self.repo_root, "test")
        self.img_dir = os.path.join(self.path, "test-images")

        self.images = {}
        self.all_tags = set()
        self.__map_images()
        self.__log(
            "Found ",
            colored(f"{len(self.images)}", "green"),
            "images in test-images folder",
            "\nUse these tags to select images when using get_media:",
            colored(f"{self.all_tags}", "green"),
        )

        self.max_width = max_width
        self.max_height = max_height

        self.to_tensor = transforms.ToTensor()

    def __log(self, *args):
        if VERBOSE: 
            _log("Test Images", *args)

    def set_max_dimensions(self, width: int, height: int):
        self.max_width = int(width)
        self.max_height = int(height)

    def __resize_image(self, img: Image.Image) -> Image.Image:
        """Resizing image data before it is a tensor is considerably faster + higher fidelity."""
        ratio = min(self.max_width / img.width, self.max_height / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        return img.resize(new_size, Image.ANTIALIAS)

    def get_media(
        self,
        count,
        tags=[],
        prioritize_real=True,
        is_video=False,
        is_picture=True,
        mb_limit=5,
        as_pil=False,
    ):
        if not prioritize_real:
            return self.__generate_images(count)

        self.__log(
            "Getting",
            colored(f"{count}", "cyan"),
            "random images with tags",
            colored(f"{tags}", "light_cyan"),
            "and size limit",
            colored(f"{mb_limit}", "cyan"),
            "MB",
        )
        matches = []
        tags = set(tags)

        for img in self.images.values():
            if is_video and not img["is_video"]:
                continue
            if is_picture and not img["is_picture"]:
                continue
            if img["file_size_mb"] > mb_limit:
                continue
            if tags and not tags.issubset(set(img["tags"])):
                continue

            matches.append(img)

        random_selection = random.sample(matches, count)
        self.__log(
            f"Resizing {len(random_selection)} images to fit within scale/frame {self.max_width}x{self.max_height} while maintaining aspect ratio."
        )
        use_rgba_tags = [
            "rgba",
            "alpha",
            "alpha-layer",
            "alpha-layers",
            "cutout",
            "mask",
        ]
        ret = []
        for img in random_selection:
            pil = Image.open(img["fullpath"])
            if any(tag in img["tags"] for tag in use_rgba_tags):
                pil = pil.convert("RGBA")
                self.__log(f"Converting {img['file']} to RGBA")
            else:
                pil = pil.convert("RGB")

            if as_pil:
                ret.append(self.__resize_image(pil))
                continue

            as_tensor = self.to_tensor(self.__resize_image(pil))
            self.__log(
                "Shape of returned test image:", colored(f"{as_tensor.shape}", "yellow")
            )
            ret.append(as_tensor)

        if len(ret) < count:
            if as_pil:
                # duplicate images randomly to reach count
                ret.extend(random.sample(ret, count - len(ret)))
            else:
                self.__log(
                    colored(f"\nOnly found {len(ret)}/{count}", "red"),
                    f"{'images/videos' if is_picture and is_video else 'images' if is_picture else 'videos'} with tags:\n",
                    colored(f"{tags}", "light_cyan"),
                    "\nand size limit ",
                    colored(f"{mb_limit}mb", "light_red"),
                    "\nGenerating random noise images to supplement.",
                )
                ret.extend(self.__generate_images(count - len(ret)))

        return ret

    def __generate_images(self, count):
        if count > 0:
            self.__log(f"Generating {count} images with random noise.")
        return [torch.rand(3, self.max_height, self.max_width) for _ in range(count)]

    def __is_video(self, file_path):
        _, file_ext = os.path.splitext(file_path)
        return file_ext.lower() in VIDEO_EXTENSION_LIST

    def __is_picture(self, file_path):
        _, file_ext = os.path.splitext(file_path)
        return file_ext.lower() in PICTURE_EXTENSION_LIST

    def __map_images(self):
        for root, dirs, files in os.walk(self.img_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path) and (
                    self.__is_video(file_path) or self.__is_picture(file_path)
                ):
                    tags = set()
                    for path_path in os.path.relpath(root, self.img_dir).split(os.sep):
                        tags.add(path_path)
                    self.all_tags.update(tags)

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
