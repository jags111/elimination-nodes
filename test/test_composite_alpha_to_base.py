"""
pyenv local 3.10.6

"""

import unittest
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import os
import random
import sys
import matplotlib.pyplot as plt
import math
from typing import Tuple

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from composite_alpha_to_base import CompositeAlphaToBase


HOME = os.path.expanduser("~")
TEST_IMAGES = [
    # "test-image.jpg",
    # "2.jpg",
    # "3.jpg",
    # "star_wars.jpg",
    "star_wars-small.jpg",
    "star_wars-medium.jpg",
]


class TestCompositeAlphaToBase(unittest.TestCase):
    def test_composite_accepts_tensors(self):
        random_real_img = self.__random_real_img_tensors(1)[0]
        random_noise_img = self.__random_noise_img_tensors(1)[0]
        compositer = CompositeAlphaToBase()
        compositer.composite(random_real_img, random_noise_img)

        self.assertEqual(type(compositer.base_image), torch.Tensor)
        self.assertEqual(type(compositer.alpha_overlay), torch.Tensor)

    def test_composite_resizes_to_fit(self):
        compositer = CompositeAlphaToBase()
        permutations = self.__image_difference_permutations()
        results = []
        for index, (img1, img2) in enumerate(permutations):
            case_str = f"\n\n{'Case' if index < 7 else 'Edge Case' } {(index % 7) + 1}"
            pre_resize_str = (
                f"\n\nBefore Resize:\nimg1: {img1.shape}, img2: {img2.shape}"
            )
            resized = compositer.composite(img1, img2)
            post_resize_str = (
                f"\n\nAfter Resize:\n{resized[0].shape}, {resized[1].shape}"
            )
            self.assertEqual(
                resized[0].shape[1:],
                resized[1].shape[1:],
                f"{case_str}{pre_resize_str}{post_resize_str}",
            )
            results += [
                (f"{case_str} - Input Image 1 ({img1.shape})", img1),
                (f"{case_str} - Input Image 2 ({img2.shape})", img2),
                (f"{case_str} - Resized Image 1 ({resized[0].shape})", resized[0]),
                (f"{case_str} - Resized Image 2 ({resized[1].shape})", resized[1]),
            ]
        
        self.comparison_grid(results)

    def comparison_grid(self, images: list[Tuple[str, torch.Tensor]]):
        cmp = CompositeAlphaToBase()
        
        to_pil = transforms.ToPILImage()
        images = [(caption, to_pil(cmp.test_squeeze_batch(img))) for caption, img in images]
        
        temp_dirname = "temp"
        os.makedirs(temp_dirname, exist_ok=True)

        base_padding = 10 
        max_w = 0
        max_h = 0
        for i, (_, img) in enumerate(images):
            if img.width > max_w:
                max_w = img.width
            if img.height > max_h:
                max_h = img.height
            img.save(os.path.join(temp_dirname, f"img{i}.jpg"))

        max_w += base_padding
        max_h += base_padding


        rows, cols = self.best_square_grid(len(images))

        # Stitch to grid
        canvas = Image.new("RGB", (cols * max_w, rows * max_h))
        for i, (caption, img) in enumerate(images):
            row = i // cols
            col = i % cols

            # Draw caption
            ImageDraw.Draw(img).text((0, 0), caption, fill="white")

            # Center in cell
            padding_x = 0 if img.width >= max_w-2 else (max_w - img.width) // 2
            padding_y = 0 if img.height >= max_h-2 else (max_h - img.height) // 2
            canvas.paste(img, (col * max_w + padding_x, row * max_h + padding_y))

        canvas.save(os.path.join(temp_dirname, "comparison_grid.jpg"))

        os.system(
            f"{self.__get_xdg_equiv()} {os.path.join(temp_dirname, 'comparison_grid.jpg')}"
        )


    def __image_difference_permutations(self) -> list[list[torch.Tensor]]:
        """
        Case 1: img1_width > img2_width and img1_height > img2_height
        Case 2: img1_width > img2_width and img1_height < img2_height
        Case 3: img1_width < img2_width and img1_height > img2_height
        Case 4: img1_width < img2_width and img1_height < img2_height (redundant)
        Case 5: img1_width == img2_width and img1_height == img2_height
        Case 6: img1_width == img2_width and img1_height > img2_height
        Case 7: img1_width > img2_width and img1_height == img2_height
        Edge Case 1: img1_width = 1
        Edge Case 2: img1_height = 1
        Edge Case 3: img2_width = 1
        Edge Case 4: img2_height = 1
        Edge Case 5: img1_width = prime number
        Edge Case 6: img1_height = prime number
        """
        # Load 2 random real images
        img1, img2 = self.__random_real_img_tensors(2)

        # Get the upper/lower bounds (max/min x or y values of the 2 images)
        upper = max(img1.shape[1], img2.shape[1], img1.shape[2], img2.shape[2])
        lower = min(img1.shape[1], img2.shape[1], img1.shape[2], img2.shape[2])
        larger = lambda: random.randint(upper + 1, upper * 2)
        smaller = lambda: random.randint(1, lower - 1)

        # Create the permutations by slicing the images
        cases = [
            [
                img1[:, : larger(), : larger()],
                img2[:, : smaller(), : smaller()],
            ],  # Case 1
            [
                img1[:, : larger(), : smaller()],
                img2[:, : smaller(), : larger()],
            ],  # Case 2
            [
                img1[:, : smaller(), : larger()],
                img2[:, : larger(), : smaller()],
            ],  # Case 3
            [
                img1[:, : smaller(), : smaller()],
                img2[:, : larger(), : larger()],
            ],  # Case 4
            [img1, img2],  # Case 5
            [img1[:, : larger(), : larger()], img2],  # Case 6
            # [img1, img2[:, : larger(), : larger()]],  # Case 7
            # [img1[:, :1, : larger()], img2],  # Edge Case 1
            # [img1[:, : larger(), :1], img2],  # Edge Case 2
            # [img1, img2[:, :1, : larger()]],  # Edge Case 3
            # [img1, img2[:, : larger(), :1]],  # Edge Case 4
            # [img1[:, :71, : larger()], img2],  # Edge Case 5
            # [img1[:, : larger(), :71], img2],  # Edge Case 6
        ]
        return cases

    def __random_noise_img_tensors(self, n):
        return [torch.rand(3, 100, 100) for _ in range(n)]

    def __random_real_img_tensors(self, n):
        random_indices = random.sample(range(0, len(TEST_IMAGES)), n)
        fullpaths = [os.path.join(HOME, TEST_IMAGES[i]) for i in random_indices]
        return [self.__load_img_as_tensor(fullpaths[i]) for i in range(n)]

    def __load_img_as_tensor(self, img_fullpath):
        pil_image = Image.open(img_fullpath)
        tensor_image = transforms.ToTensor()(pil_image)
        return tensor_image


if __name__ == "__main__":
    unittest.main()
