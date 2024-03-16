"""
pyenv local 3.10.6

"""

import torch
from torchvision import transforms
from PIL import Image
import os
import random
import sys
from itertools import permutations, product
from termcolor import colored

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.tensor_utils import TensorImgUtils
from test.test_tools_constants import (
    ORDER_STRINGS,
    COLOR_ORDERS,
    EDGE_CASE_PIXELS,
    VERBOSE,
    ALLOW_REPEAT_BRANCHES,
)
from utils.logger import _log
from types_interfaces.image_tensor_types import ImageTensorTypes as itt


class BranchGenerator:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __log(self, *args):
        if VERBOSE:
            _log("Branch Generator", *args)

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
            elif img.height < max_height:
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

    def __preview_img_size_branches(
        self, branches: dict[str : list[dict]], height_intervals, width_intervals
    ):
        MAX_PREVIEWS = 12 // len(next(iter(branches.values())))

        self.__log("[IMG SIZES] Preview the Generated Image Size Branches:")
        for index in range(len(height_intervals)):
            print(
                colored(
                    f"{ORDER_STRINGS[index]} Height Range: {height_intervals[index]}, {ORDER_STRINGS[index]} Width Range: {width_intervals[index]}",
                    COLOR_ORDERS[index],
                )
            )

        for branch_index, branch in enumerate(branches.values()):
            print(colored(f"Branch {branch_index+1}:", "light_cyan"))
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

            if branch_index > MAX_PREVIEWS - 1:
                print(
                    colored("And so on for", "light_cyan"),
                    f"{len(branches) - (MAX_PREVIEWS + 1)}",
                    colored("more branches...\n", "light_cyan"),
                )
                break

    def __img_size_branches(
        self, images: list[Image.Image], max_branches=40, pad_with_edge_cases=False
    ):
        equalized_images, standardized_size = self.__equalize_img_sizes_crop(images)

        edge_case_index = len(images) + 1
        ordering_permutations = self.__generate_dimension_permutations(len(images) + 1)

        # If permutations is far larger than max allowed branches (and unrealistic to display), don't include edge case branches
        self.__log(
            f"[IMG SIZES] Total Permutations: {len(ordering_permutations)} (including edge cases)"
        )
        if len(ordering_permutations) > max_branches:
            temp = ordering_permutations
            self.__log(
                f"[IMG SIZES] Total branches exceeds max allowed branches ({max_branches}), removing edge cases from permutations before proceeding."
            )
            ordering_permutations = self.__generate_dimension_permutations(len(images))
            self.__log(
                f"[IMG SIZES] Filtered to core branches (still full coverage) — New branch count = {len(ordering_permutations)}"
            )

            if pad_with_edge_cases:
                self.__log(
                    "[IMG SIZES] Padding branches with edge-case branches until max-allowed # reached"
                )
                pad_n = max_branches - len(ordering_permutations)
                removed_branches = temp[: len(ordering_permutations)]
                if pad_n >= len(removed_branches):
                    ordering_permutations += removed_branches
                else:
                    pan_branches = random.sample(
                        temp[len(ordering_permutations) :], pad_n
                    )
                    ordering_permutations += pan_branches

        height_intervals = self.__get_subintervals(standardized_size[0], len(images))
        width_intervals = self.__get_subintervals(standardized_size[1], len(images))
        height_intervals.reverse()
        width_intervals.reverse()

        def ordered_rand_permutation(height_order, width_order):
            ret = {
                "ordering": (height_order, width_order),
            }
            dimensions = [None, None]
            if height_order == edge_case_index:
                dimensions[0] = EDGE_CASE_PIXELS
            else:
                height_lower_bound, height_upper_bound = height_intervals[
                    height_order - 1
                ]
                dimensions[0] = random.randint(
                    height_lower_bound,
                    height_upper_bound,
                )
            if width_order == edge_case_index:
                dimensions[1] = EDGE_CASE_PIXELS
            else:
                width_lower_bound, width_upper_bound = width_intervals[width_order - 1]
                dimensions[1] = random.randint(
                    width_lower_bound,
                    width_upper_bound,
                )
            ret["dimensions"] = dimensions
            return ret

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

                if height_order == edge_case_index:
                    height_descrip = "Edge Value"
                else:
                    height_descrip = f"{ORDER_STRINGS[height_order - 1]}"
                if width_order == edge_case_index:
                    width_descrip = "Edge Value"
                else:
                    width_descrip = f"{ORDER_STRINGS[width_order - 1]}"
                branch_descriptions.append(
                    f"image{i + 1}: ({height_descrip} H x {width_descrip} W)"
                )
                branch.append(rand_dims)

            description = f"[BRANCH {index+1}] " + " | ".join(branch_descriptions)
            if description in branches:
                if not ALLOW_REPEAT_BRANCHES:
                    raise ValueError(
                        f"[IMG SIZES] Branches are not unique: {description}"
                    )
                self.__log(f"[IMG SIZES] Branches are not unique: {description}")

            branches[description] = branch

        if VERBOSE:
            self.__preview_img_size_branches(
                branches, height_intervals, width_intervals
            )

        return equalized_images, branches

    def gen_branches_img_size(
        self,
        images: list[Image.Image],
        return_pil: bool = False,
        return_tensors: bool = True,
        max_branches=40,
    ):
        resized_imgs, branches = self.__img_size_branches(images, max_branches)
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

    def gen_branches_tensor_types(
        self,
        arg_types: list[str],
        batch_dimension=False,
    ):
        """


        Args:
            arg_types (list[str]): A tuple describing each argument type for the class being tested. Valid types are "image", and "mask". For example, if the class takes an image and a mask, the tuple would be ("image", "mask").
        """
        valid_arg_types = ["image", "mask"]
        if not all([arg in valid_arg_types for arg in arg_types]):
            raise ValueError(
                f"Invalid arg_types. Must be a list containing only items from: {valid_arg_types}."
            )

        b = 1
        height = 24
        width = 34
        rgb = 3
        rgba = 4
        a = 1
        image_types = [
            {
                "tensor_type": itt.H_W_RGB_Tensor,
                "tensor_image": torch.rand(height, width, rgb),
                "description": "(H, W, RGB) Tensor",
            },
            {
                "tensor_type": itt.H_W_RGBA_Tensor,
                "tensor_image": torch.rand(height, width, rgba),
                "description": "(H, W, RGBA) Tensor",
            },
            {
                "tensor_type": itt.RGB_H_W_Tensor,
                "tensor_image": torch.rand(rgb, height, width),
                "description": "(RGB, H, W) Tensor",
            },
            {
                "tensor_type": itt.RGBA_H_W_Tensor,
                "tensor_image": torch.rand(rgba, height, width),
                "description": "(RGBA, H, W) Tensor",
            },
        ]
        batch_types = [
            {
                "tensor_type": itt.B_H_W_RGB_Tensor,
                "tensor_image": torch.rand(b, height, width, rgb),
                "description": "(B, H, W, RGB) Tensor",
            },
            {
                "tensor_type": itt.B_H_W_RGBA_Tensor,
                "tensor_image": torch.rand(b, height, width, rgba),
                "description": "(B, H, W, RGBA) Tensor",
            },
            {
                "tensor_type": itt.B_RGB_H_W_Tensor,
                "tensor_image": torch.rand(b, rgb, height, width),
                "description": "(B, RGB, H, W) Tensor",
            },
            {
                "tensor_type": itt.B_RGBA_H_W_Tensor,
                "tensor_image": torch.rand(b, rgba, height, width),
                "description": "(B, RGBA, H, W) Tensor",
            },
        ]
        mask_types = [
            {
                "tensor_type": itt.H_W_Tensor,
                "tensor_image": torch.rand(height, width, a),
                "description": "(H, W) Tensor",
            },
            {
                "tensor_type": itt.A_H_W_Tensor,
                "tensor_image": torch.rand(a, height, width),
                "description": "(A, H, W) Tensor",
            },
            {
                "tensor_type": itt.H_W_A_Tensor,
                "tensor_image": torch.rand(b, height, width),
                "description": "(H, W, A) Tensor",
            },
        ]
        if batch_dimension:
            image_types += batch_types

        coord_ranges = [
            len(mask_types) if arg == "mask" else len(image_types) for arg in arg_types
        ]
        ranges = [range(i) for i in coord_ranges]
        permutations = list(product(*ranges))

        self.__log(f"[TENSOR TYPES] Total Permutations: {len(permutations)}")
        branches = {}
        for perm_index, perm in enumerate(permutations):
            tensor_descriptions = []
            branch = []
            for tensor_index in range(len(arg_types)):
                if arg_types[tensor_index] == "mask":
                    tensor = mask_types[perm[tensor_index]]
                elif arg_types[tensor_index] == "image":
                    tensor = image_types[perm[tensor_index]]

                branch.append(tensor)
                tensor_descriptions.append(
                    f"image{tensor_index + 1}: {tensor['description']}"
                )

            brannch_description = f"[BRANCH {perm_index+1}] " + " | ".join(
                tensor_descriptions
            )
            branches[brannch_description] = branch

        if VERBOSE:
            self.__preview_tensor_branches(branches)

        return branches

    def __preview_tensor_branches(self, branches):
        MAX_PREVIEWS = 9 // len(next(iter(branches.values())))
        self.__log("[TENSOR TYPES] Preview the Generated Tensor Type Branches:")
        for index, branch in enumerate(branches):
            print(colored(f"Branch {index}", "light_cyan"))
            for tensor_index, tensor in enumerate(branches[branch]):
                print(f"Image {tensor_index + 1} - {tensor['description']}")
            if index > MAX_PREVIEWS - 1:
                print(
                    colored("And so on for", "light_cyan"),
                    f"{len(branches) - (MAX_PREVIEWS + 1)}",
                    colored("more branches...\n", "light_cyan"),
                )
                break
