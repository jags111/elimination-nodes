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

# Symlink temp
try:
    from ..nodes.compositers.composite_alpha_to_base_node import (
        CompositeCutoutOnBaseNode,
    )
    from .results_webview import ComparisonGrid
    from ..utils.tensor_utils import TensorImgUtils
    from .test_images import TestImages
    from .branch_generator import BranchGenerator
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from nodes.compositers.composite_alpha_to_base_node import CompositeCutoutOnBaseNode
    from results_webview import ComparisonGrid
    from utils.tensor_utils import TensorImgUtils
    from test_images import TestImages
    from branch_generator import BranchGenerator


class TestCompositeAlphaToBase(unittest.TestCase):

    def setUp(self) -> None:
        self.test_images = TestImages()
        self.branch_gen = BranchGenerator()

    # def rtest_composite_pre_cut_alpha(self):

    #     base = self.__random_real_img_tensors(1)[0]
    #     alpha = self.__random_real_alpha_img_tensors(1)[0]

    #     compositer = CompositeCutoutOnBaseNode()
    #     composite = compositer.main(
    #         base, alpha, "cover_maintain_aspect_ratio_with_crop"
    #     )
    #     self.assertIn(composite.shape[1:], [base.shape[1:], alpha.shape[1:]])

    #     ComparisonGrid(
    #         [
    #             ("Base", base),
    #             ("Alpha", alpha),
    #             ("Composite", composite),
    #         ],
    #         "composite",
    #     )().show()

    # def rtest_main_resizes_and_composites_different_size_permutations(self):
    #     compositer = CompositeCutoutOnBaseNode()
    #     results = []
    #     for index, (img1, img2) in enumerate(self.__image_difference_permutations()):
    #         case_str = f"\n\n{'Case' if index < 7 else 'Edge Case' } {(index % 7) + 1}"
    #         pre_resize_str = (
    #             f"\n\nBefore Resize:\nimg1: {img1.shape}, img2: {img2.shape}"
    #         )
    #         composite = compositer.main(
    #             img1, img2, "cover_maintain_aspect_ratio_with_crop"
    #         )
    #         post_resize_str = f"\n\nAfter Resize:\n{composite.shape}"
    #         self.assertIn(
    #             composite.shape[1:],
    #             [img1.shape[1:], img2.shape[1:]],
    #             f"\n\n{case_str}{pre_resize_str}{post_resize_str}",
    #         )
    #         results += [
    #             (f"{case_str} - Input Image 1 ({img1.shape})", img1),
    #             (f"{case_str} - Input Image 2 ({img2.shape})", img2),
    #             (f"{case_str} - Composite Image ({composite.shape})", composite),
    #         ]

    #     ComparisonGrid(results, "composite")().show()

    # def rtest_composite_pastes_alpha_overlay_on_base_image_random_dimensions(
    #     self,
    # ) -> Tuple[torch.Tensor]:
    #     """
    #     Create a random alpha cutout from the given image tensor.
    #     Input img may not have an alpha channel at first, but the returned tensor will have one.
    #     The size of the alpha channel will be the same as the input image.
    #     """
    #     comp = CompositeCutoutOnBaseNode()
    #     random_images = self.__random_real_img_tensors(6)
    #     results = []
    #     i = 0
    #     while i < 4:
    #         base = random_images[i]
    #         overlay = random_images[i + 1]
    #         base, overlay = comp.match_size(
    #             base, overlay, "cover_maintain_aspect_ratio_with_crop"
    #         )
    #         base = TensorImgUtils.test_squeeze_batch(base)
    #         overlay = TensorImgUtils.test_squeeze_batch(overlay)
    #         lower_bound = min(
    #             base.shape[1], base.shape[2], overlay.shape[1], overlay.shape[2]
    #         )
    #         smaller = lambda: random.randint(1, (lower_bound // 2) - 4)
    #         x = smaller()
    #         y = smaller()
    #         width = smaller()
    #         height = smaller()

    #         # Create a transparent alpha channel
    #         alpha = torch.zeros(1, base.shape[1], base.shape[2])

    #         # Set the cutout dimensions to opaque
    #         alpha[:, x : x + width, y : y + height] = 1

    #         # Concat the alpha channel and alpha overlay along the 0th (rgb) dimension (assuming no batch dimension)
    #         overlay = torch.cat((overlay, alpha), 0)

    #         composite = comp.composite(base, overlay)

    #         self.assertEqual(composite.shape[-2:], base.shape[-2:])
    #         self.assertEqual(composite.shape[-2:], overlay.shape[-2:])

    #         results += [
    #             (f"Base Image {i + 1}", base),
    #             (f"Alpha Overlay {i + 1}", overlay),
    #             (f"Composite {i + 1}", composite),
    #         ]

    #         i += 2

    #     ComparisonGrid(results, "composite")().show()

    # def test_accepts_and_returns_tensors(self):
    #     random_real_img = self.__random_real_img_tensors(1)[0]
    #     random_noise_img = self.__random_noise_img_tensors(1)[0]
    #     compositer = CompositeCutoutOnBaseNode()
    #     res = compositer.composite(random_real_img, random_noise_img)
    #     self.assertEqual(type(res), torch.Tensor)

    def test_tensor_type_permutations(self):
        branches = self.branch_gen.gen_branches_tensor_types(["image", "image"])
        compositer = CompositeCutoutOnBaseNode()
        results = ComparisonGrid()
        for branch_descrip, branch in branches.items():
            img1 = branch[0]["tensor_image"]
            img2 = branch[1]["tensor_image"]
            try:
                img2_alpha = img2[:, :, 3]
                img2 = img2[:, :, :3]
            except:
                # Generate all opaque alpha layer
                img2_alpha = torch.ones_like(img2[:, :, 0])
                
            resized = compositer.main(
                img1, img2, img2_alpha, "cover_crop_center", invert_cutout=True
            )
            results.add(branch_descrip, "Input Image", img1)
            results.add(branch_descrip, "Input Alpha Layer", img2)
            results.add(branch_descrip, "Size Matched and Composited", resized)

    def rtest_size_permutations(self):
        compositer = CompositeCutoutOnBaseNode()
        results = ComparisonGrid()
        test_rgb_bg = self.test_images.get_media(1, tags=["people", "real"], as_pil=True)
        test_alpha_layer = self.test_images.get_media(1, tags=["alpha-layers"], as_pil=True)
        test_images = test_rgb_bg + test_alpha_layer
        branches = self.branch_gen.gen_branches_img_size(test_images)

        for branch_descrip, branch in branches.items():
            img1 = TensorImgUtils.to_hwc_singleton(branch[0]["tensor_image"])
            img2 = TensorImgUtils.to_hwc_singleton(branch[1]["tensor_image"])

            img2_alpha = img2[:, :, 3]
            img2 = img2[:, :, :3]

            img1.unsqueeze(0)
            img2.unsqueeze(0)

            resized = compositer.main(
                img1,
                img2,
                img2_alpha,
                "cover_crop_center",
                invert_cutout=True,  # Emulate the fact that alpha layers are auto-inverted in comfy
            )

            resized = resized.squeeze(0)
            self.assertIn(
                resized.shape[1],
                [img1.shape[1], img2.shape[1]],
                f"\n\n{branch_descrip}\nInputs: {img1.shape} and {img2.shape}\nOutputs: {resized.shape}\n",
            )

            img1 = TensorImgUtils.to_chw_singleton(img1)
            img2 = TensorImgUtils.to_chw_singleton(img2)
            resized = TensorImgUtils.to_chw_singleton(resized)
            results.add(branch_descrip, "Input Image", img1)
            results.add(branch_descrip, "Input Alpha Layer", img2)
            results.add(branch_descrip, "Size Matched and Composited", resized)

        results.show_webview()

    # def rtest_match_size_accepts_and_returns_tensors(self):
    #     random_real_img = self.__random_real_img_tensors(1)[0]
    #     random_noise_img = self.__random_noise_img_tensors(1)[0]
    #     compositer = CompositeCutoutOnBaseNode()
    #     res1, res2 = compositer.match_size(
    #         random_real_img, random_noise_img, "fit_center_and_pad"
    #     )

    #     self.assertEqual(type(res1), torch.Tensor)
    #     self.assertEqual(type(res2), torch.Tensor)

    # def rtest_match_size_matches_input_sizes_all_modes(self):
    #     modes = [
    #         "cover_maintain_aspect_ratio_with_crop",
    #         "cover_perfect_by_distorting",
    #         "crop_larger_to_match",
    #         "fit_center_and_pad",
    #     ]
    #     for mode in modes:
    #         compositer = CompositeCutoutOnBaseNode()
    #         permutations = self.__image_difference_permutations()
    #         results = []
    #         for index, (img1, img2) in enumerate(permutations):
    #             case_str = (
    #                 f"\n\n{'Case' if index < 7 else 'Edge Case' } {(index % 7) + 1}"
    #             )
    #             pre_resize_str = (
    #                 f"\n\nBefore Resize:\nimg1: {img1.shape}, img2: {img2.shape}"
    #             )
    #             resized = compositer.match_size(img1, img2, mode)
    #             post_resize_str = (
    #                 f"\n\nAfter Resize:\n{resized[0].shape}, {resized[1].shape}"
    #             )
    #             self.assertEqual(
    #                 resized[0].shape[1:],
    #                 resized[1].shape[1:],
    #                 f"\n\n{mode}\n{case_str}{pre_resize_str}{post_resize_str}",
    #             )
    #             results += [
    #                 (f"{case_str} - Input Image 1 ({img1.shape})", img1),
    #                 (f"{case_str} - Input Image 2 ({img2.shape})", img2),
    #                 (f"{case_str} - Resized Image 1 ({resized[0].shape})", resized[0]),
    #                 (f"{case_str} - Resized Image 2 ({resized[1].shape})", resized[1]),
    #             ]

    #         ComparisonGrid(results, mode)().show()




if __name__ == "__main__":
    unittest.main()
