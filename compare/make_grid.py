import os
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
import time
import math
from typing import List, Tuple
from utils.tensor_utils import TensorImgUtils

class ComparisonGrid:
    def __init__(
        self,
        images: list[Tuple[str, torch.Tensor]],
        favor_dimension = "height",
        filename_prefix: str = "",
        cell_padding: int = 10,
        temp_dirname: str = "temp",
    ):

        self.to_pil = transforms.ToPILImage()
        self.images = [
            (caption, self.to_pil(TensorImgUtils.test_squeeze_batch(img)))
            for caption, img in images
        ]

        self.favor = favor_dimension
        self.filename_prefix = filename_prefix
        self.cell_padding = int(cell_padding)
        self.temp_dirname = temp_dirname
        os.makedirs(temp_dirname, exist_ok=True)
        self.rows, self.cols = self.best_square_grid(len(images))
        self.__set_dimensions()

    def best_square_grid(self, n_items) -> Tuple[int, int]:
        """
        Given a number of items, return the best grid shape for a square grid
        that can contain all the items.
        """
        root = math.sqrt(n_items)
        if root.is_integer():
            return int(root), int(root)
        else:
            # distance to next integer
            dist = root - int(root)
            if dist < 0.5:
                if self.favor == "height":
                    return int(root), int(root) + 1
                else:
                    return int(root) + 1 , int(root)
            else:
                return int(root) + 1, int(root) + 1

    def __set_dimensions(self):
        self.cell_w = max([img.width for _, img in self.images]) + self.cell_padding
        self.cell_h = max([img.height for _, img in self.images]) + self.cell_padding

    def __call__(self):
        # Stitch to grid
        # high contrast light pink as rgb is (255, 182, 193)
        canvas = Image.new("RGB", (self.cols * self.cell_w, self.rows * self.cell_h), color=(255, 182, 193))
        for i, (caption, img) in enumerate(self.images):
            row = i // self.cols
            col = i % self.cols

            # Draw caption
            ImageDraw.Draw(img).text((0, 0), caption, fill="white")

            # Center in cell
            padding_x = (
                0 if img.width >= self.cell_w - 2 else (self.cell_w - img.width) // 2
            )
            padding_y = (
                0 if img.height >= self.cell_h - 2 else (self.cell_h - img.height) // 2
            )
            canvas.paste(
                img, (col * self.cell_w + padding_x, row * self.cell_h + padding_y)
            )

        canvas.save(
            os.path.join(
                self.temp_dirname, f"{self.filename_prefix}-comparison_grid{time.strftime('%I_%M%p')}.jpg"
            )
        )

        # Caller can use canvas.show() to display the grid with default app automatically, if they want
        return canvas
