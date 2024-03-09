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
from compare.make_grid import ComparisonGrid


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
    def test_composite_accepts_and_returns_tensors(self):
        random_real_img = self.__random_real_img_tensors(1)[0]
        random_noise_img = self.__random_noise_img_tensors(1)[0]
        compositer = CompositeAlphaToBase()
        res1, res2 = compositer.composite(random_real_img, random_noise_img)

        self.assertEqual(type(res1), torch.Tensor)
        self.assertEqual(type(res2), torch.Tensor)

    def test_composite_matches_input_sizes_all_modes(self):
        modes = [
            "cover_maintain_aspect_ratio_with_crop",
            "cover_perfect_by_distorting",
            "crop_larger_to_match",
            "fit_center_and_pad",
        ]
        for mode in modes:
            compositer = CompositeAlphaToBase()
            permutations = self.__image_difference_permutations()
            results = []
            for index, (img1, img2) in enumerate(permutations):
                case_str = (
                    f"\n\n{'Case' if index < 7 else 'Edge Case' } {(index % 7) + 1}"
                )
                pre_resize_str = (
                    f"\n\nBefore Resize:\nimg1: {img1.shape}, img2: {img2.shape}"
                )
                resized = compositer.composite(img1, img2, mode)
                post_resize_str = (
                    f"\n\nAfter Resize:\n{resized[0].shape}, {resized[1].shape}"
                )
                self.assertEqual(
                    resized[0].shape[1:],
                    resized[1].shape[1:],
                    f"\n\n{mode}\n{case_str}{pre_resize_str}{post_resize_str}",
                )
                results += [
                    (f"{case_str} - Input Image 1 ({img1.shape})", img1),
                    (f"{case_str} - Input Image 2 ({img2.shape})", img2),
                    (f"{case_str} - Resized Image 1 ({resized[0].shape})", resized[0]),
                    (f"{case_str} - Resized Image 2 ({resized[1].shape})", resized[1]),
                ]

            ComparisonGrid(results, mode)().show()

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
