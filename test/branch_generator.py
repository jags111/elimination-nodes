"""
pyenv local 3.10.6

"""

import unittest
import torch
from torchvision import transforms
from PIL import Image
import os
import random
from typing import Tuple
import sys
from itertools import permutations, product
from termcolor import colored
import random

# Symlink temp
try:
    from ..nodes.compositers.composite_alpha_to_base_node import (
        CompositeCutoutOnBaseNode,
    )
    from .results_webview import ComparisonGrid
    from ..utils.tensor_utils import TensorImgUtils
    from .test_images import TestImages
    from ..constants import ORDER_STRINGS, COLOR_ORDERS
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from nodes.compositers.composite_alpha_to_base_node import CompositeCutoutOnBaseNode
    from results_webview import ComparisonGrid
    from utils.tensor_utils import TensorImgUtils
    from test_images import TestImages
    from constants import ORDER_STRINGS, COLOR_ORDERS


VERBOSE = True
ALLOW_REPEAT_BRANCHES = True
root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
test_base_images = os.path.join(root, "test-images", "base-layers")
test_alpha_images = os.path.join(root, "test-images", "alpha-layers")


class BranchGenerator:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __generate_permutations(
        self, sequence_cardinality: int, items_repeatable: bool = False
    ):
        if items_repeatable:
            cases = product(
                range(1, sequence_cardinality + 1), repeat=sequence_cardinality
            )
        else:
            cases = permutations(
                range(1, sequence_cardinality + 1), sequence_cardinality
            )
        return list(cases)

    def __generate_dimension_permutations(self, sequence_cardinality: int):
        ordering_permutations = self.__generate_permutations(
            sequence_cardinality, items_repeatable=True
        )
        ret = []
        for height_ordering in ordering_permutations:
            for width_ordering in ordering_permutations:
                ret.append((width_ordering, height_ordering))
        return ret

    def __equalize_img_sizes_crop(self, images: list[Image.Image]):
        ret = []
        max_width = max([img.width for img in images])
        max_height = max([img.height for img in images])

        for img in images:
            # Try to resize and maintain aspect ratio
            aspect_maintained_height = int((max_height / img.height) * img.width)
            img = img.resize((max_width, aspect_maintained_height))
            # Crop extra height if height exceeds max
            if img.height > max_height:
                img = img.crop((0, 0, max_width, max_height))
            # If resizing by matching width didnt create enough height, enlarge further
            else:
                required_width = (max_height / img.height) * img.width
                img = img.resize((int(required_width), max_height))
                # Then crop extra width if width exceeds max
                if img.width > max_width:
                    img = img.crop((0, 0, max_width, max_height))

            ret.append(img)

        return ret, (max_height, max_width)

    def __get_subintervals(self, codomain_upper_bound: int, count: int):
        """Returns non-inclusive, non-overlapping n subintervals of a given range ∈ Z."""
        seperator = codomain_upper_bound // count
        breakpoints = [seperator * i for i in range(count + 1)]
        return [(breakpoints[i] + 1, breakpoints[i + 1] - 1) for i in range(count)]

    def preview_img_size_branches(
        self, branches: dict[str : list[dict]], height_intervals, width_intervals
    ):
        MAX_EXAMPLES = 16 // len(next(iter(branches.values())))

        print("\nPreview the Generated Image Size Branches:")
        for index in range(len(height_intervals)):
            print(
                colored(
                    f"{ORDER_STRINGS[index]} Height Range: {height_intervals[index]}, {ORDER_STRINGS[index]} Width Range: {width_intervals[index]}",
                    COLOR_ORDERS[index],
                )
            )

        for branch_index, branch in enumerate(branches.values()):
            print(colored(f"Branch {branch_index}:", "light_cyan"))
            for img_index, img_result in enumerate(branch):
                height_order = img_result["ordering"][0]
                width_order = img_result["ordering"][1]
                height_color = COLOR_ORDERS[height_order - 1]
                width_color = COLOR_ORDERS[width_order - 1]
                print(
                    f"Image {img_index + 1} - H x W: ",
                    colored(f"{height_order}", height_color),
                    " x ",
                    colored(f"{width_order}", width_color),
                    " Randomly Computed to ",
                    colored(f"{img_result['dimensions'][0]:>5}px", height_color),
                    " x ",
                    colored(f"{img_result['dimensions'][1]}px", width_color),
                )

            if branch_index > MAX_EXAMPLES - 1:
                print(
                    colored("And so on for", "light_cyan"),
                    f"{len(branches) - (MAX_EXAMPLES + 1)}",
                    colored("more branches...\n", "light_cyan"),
                )
                break

    def __img_size_branches(self, images: list[Image.Image]):
        ordering_permutations = self.__generate_dimension_permutations(len(images))
        equalized_images, standardized_size = self.__equalize_img_sizes_crop(images)

        height_intervals = self.__get_subintervals(standardized_size[0], len(images))
        width_intervals = self.__get_subintervals(standardized_size[1], len(images))
        height_intervals.reverse()
        width_intervals.reverse()

        def ordered_rand_permutation(height_order, width_order):
            height_lower_bound, height_upper_bound = height_intervals[height_order - 1]
            width_lower_bound, width_upper_bound = width_intervals[width_order - 1]
            return {
                "dimensions": (
                    random.randint(height_lower_bound, height_upper_bound),
                    random.randint(width_lower_bound, width_upper_bound),
                ),
                "ordering": (height_order, width_order),
            }

        branches = {}
        for index, ordering in enumerate(ordering_permutations):
            branch = []
            branch_descriptions = []
            repeat_order_map = {}
            for i in range(len(images)):
                height_order, width_order = ordering[0][i], ordering[1][i]
                # If orders repeat in a branch, assume equality is important for the test case and preserve it.
                rand_dims = ordered_rand_permutation(height_order, width_order)
                if f"height-{height_order}" in repeat_order_map:
                    rand_dims["dimensions"] = (
                        repeat_order_map[f"height-{height_order}"],
                        rand_dims["dimensions"][1],
                    )
                else:
                    repeat_order_map[f"height-{height_order}"] = rand_dims[
                        "dimensions"
                    ][0]
                if f"width-{width_order}" in repeat_order_map:
                    rand_dims["dimensions"] = (
                        rand_dims["dimensions"][0],
                        repeat_order_map[f"width-{width_order}"],
                    )
                else:
                    repeat_order_map[f"width-{width_order}"] = rand_dims["dimensions"][
                        1
                    ]

                branch_descriptions.append(
                    f"image{i + 1}: ({ORDER_STRINGS[height_order - 1]} H x {ORDER_STRINGS[width_order - 1]} W)"
                )
                branch.append(rand_dims)

            description = f"[BRANCH {index+1}] " + " | ".join(branch_descriptions)
            if description in branches:
                if not ALLOW_REPEAT_BRANCHES:
                    raise ValueError(f"Branches are not unique: {description}")
                print(f"Branches are not unique: {description}")

            branches[description] = branch

        if VERBOSE:
            self.preview_img_size_branches(branches, height_intervals, width_intervals)

        return equalized_images, branches

    def gen_branches_img_size(
        self,
        images: list[Image.Image],
        return_pil: bool = False,
        return_tensors: bool = True,
    ):
        resized_imgs, branches = self.__img_size_branches(images)
        for branch in branches:
            for img_index, img in enumerate(resized_imgs):
                # PIL uses (width, height), make sure to reverse the order
                img = img.resize(branches[branch][img_index]["dimensions"][::-1])
                # To preserve memory, release pil images if not needed
                if return_pil:
                    branches[branch][img_index]["pil_image"] = img
                if return_tensors:
                    branches[branch][img_index]["tensor_image"] = self.to_tensor(img)

        return branches
