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
        favor_dimension="width",
        filename_prefix: str = "",
        cell_padding: int = 8,
        border_padding: int = 18,
        temp_dirname: str = "test-results",
        bg_color=(234, 232, 255),
    ):
        self.favor = favor_dimension
        self.filename_prefix = filename_prefix
        self.cell_padding = int(cell_padding)
        self.border_padding = int(border_padding)
        self.temp_dirname = temp_dirname
        self.bg_color = bg_color
        self.TEXT_COLOR = (0, 0, 0)
        self.OFF_TEXT_COLOR = (255, 255, 255)

        # https://coolors.co/ff595e-ffca3a-8ac926-1982c4-6a4c93
        self.palette = [
            (255, 89, 94),
            (255, 202, 58),
            (138, 201, 38),
            (25, 130, 196),
            (106, 76, 147),
        ]
        # Alternate: https://coolors.co/247ba0-70c1b3-b2dbbf-f3ffbd-ff1654

        self.to_pil = transforms.ToPILImage()
        os.makedirs(temp_dirname, exist_ok=True)
        self.sections = []
        self.images = []

    def add_section(self, title: str):
        self.sections.append(title)

    def add_image(self, section: str, caption: str, img: torch.Tensor):
        self.images.append(
            (section, caption, self.to_pil(TensorImgUtils.test_squeeze_batch(img)))
        )

    def __assign_contrasting_colors(self):
        self.section_colors = {}
        cur_color = (255, 182, 193)
        for i, section in enumerate(self.sections):
            self.section_colors[section] = self.palette[i % len(self.palette)]

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
                if self.favor == "width":
                    return int(root), int(root) + 1
                else:
                    return int(root) + 1, int(root)
            else:
                return int(root) + 1, int(root) + 1

    def __set_dimensions(self):
        self.cell_w = max([img.width for _, _, img in self.images]) + self.cell_padding
        self.cell_h = max([img.height for _, _, img in self.images]) + self.cell_padding

    def __call__(self):
        self.__assign_contrasting_colors()
        self.rows, self.cols = self.best_square_grid(len(self.images))
        self.__set_dimensions()

        # if the sections dont fit neatly, add an extra border_padding to the right
        total_h = (
            self.rows * self.cell_h
            + self.rows * self.border_padding
            + self.border_padding
        )
        total_w = self.cols * self.cell_w + 2 * self.border_padding
        section_lengths = {}
        for img1 in self.images:
            section = img1[0]
            if section not in section_lengths:
                section_lengths[section] = 0
            section_lengths[section] += 1

        extra_border = False
        for section, length in section_lengths.items():
            if length % self.cols != 0:
                extra_border = True
                break

        if extra_border:
            total_w += self.border_padding

        canvas = Image.new(
            "RGB",
            (total_w, total_h),
            color=self.bg_color,
        )

        # sort by section
        self.images.sort(key=lambda x: x[0])

        cur_section = False
        for i, (section, caption, img) in enumerate(self.images):
            row = i // self.cols
            col = i % self.cols

            # Center in cell
            padding_x = (
                0 if img.width >= self.cell_w - 2 else (self.cell_w - img.width) // 2
            )
            padding_y = (
                0 if img.height >= self.cell_h - 2 else (self.cell_h - img.height) // 2
            )

            # Add border padding to top/bottom, left if first in section, right if last in section
            border_left = self.border_padding if col == 0 else 0
            border_right = self.border_padding if self.cols % (col + 1) == 0 else 0
            # border_top = self.border_padding if row == 0 else 0
            border_top = self.border_padding
            border_bottom = self.border_padding if self.rows % (row + 1) == 0 else 0

            if (i + 1) < len(self.images) and section != self.images[i + 1][0]:
                border_right = self.border_padding
            if (i - 1) >= 0 and section != self.images[i - 1][0]:
                border_left = self.border_padding

            # padding_x_b += self.border_padding if col == self.cols - 1 else 0
            # padding_y_b += self.border_padding if row == self.rows - 1 else 0

            # make background the sectinon color, except don't change the border
            bg_canvas = Image.new(
                "RGB",
                (self.cell_w, self.cell_h),
                color=self.section_colors[section],
            )
            bg_canvas.paste(img, (padding_x, padding_y))

            canvas.paste(
                bg_canvas,
                (
                    (col * self.cell_w) + border_left,
                    (row * self.cell_h) + ((row + 1) * border_top) + self.cell_padding,
                ),
            )

            # Draw section title if new section
            if section != cur_section:
                cur_section = section
                ImageDraw.Draw(canvas).text(
                    (
                        col * self.cell_w + self.cell_padding + self.border_padding + 2,
                        (
                            (
                                row
                                * (
                                    self.cell_h
                                    + (self.cell_padding // 2)
                                    + self.border_padding
                                )
                            )
                            + border_top
                            + self.border_padding
                            + 4
                        ),
                    ),
                    section,
                    fill=self.TEXT_COLOR,
                )
                # Draw section title again but smaller (for an outilne/text shadow effect)
                ImageDraw.Draw(canvas).text(
                    (
                        col * self.cell_w
                        + self.cell_padding
                        + self.border_padding
                        + 2
                        + 1,
                        (
                            (
                                row
                                * (
                                    self.cell_h
                                    + (self.cell_padding // 2)
                                    + self.border_padding
                                )
                            )
                            + border_top
                            + self.border_padding
                            + 4
                            + 1
                        ),
                    ),
                    section,
                    fill=self.OFF_TEXT_COLOR,
                )

            # Draw caption
            ImageDraw.Draw(canvas).text(
                (
                    col * self.cell_w + self.cell_padding + border_left + 2,
                    (row * self.cell_h) + self.cell_h - self.cell_padding // 2,
                ),
                caption,
                fill=self.TEXT_COLOR,
            )

            # Draw caption again but smaller (for an outilne/text shadow effect)
            ImageDraw.Draw(canvas).text(
                (
                    col * self.cell_w + self.cell_padding + border_left + 2 + 1,
                    (row * self.cell_h) + self.cell_h - self.cell_padding // 2 + 1,
                ),
                caption,
                fill=self.OFF_TEXT_COLOR,
            )

        canvas.save(
            os.path.join(
                self.temp_dirname,
                f"{self.filename_prefix}-comparison_grid{time.strftime('%I_%M%p')}.jpg",
            )
        )

        # Caller can use canvas.show() to display the grid with default app automatically, if they want
        return canvas
