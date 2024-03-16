import sys
import os
import subprocess
import shutil
from jinja2 import Environment, FileSystemLoader
from torchvision import transforms
from PIL import Image
from typing import List, Tuple
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
import time
import math
from typing import List, Tuple
from utils.tensor_utils import TensorImgUtils
from utils.os_utils import OS_Utils


class ComparisonGrid:
    def __init__(
        self,
        title: str = "",
        favor_dimension="width",
        cell_padding: int = 8,
        border_padding: int = 18,
        bg_color=(234, 232, 255),
    ):
        self.title = title
        self.sections = {}
        self.all_widths = []
        self.all_heights = []

        self.dir_abs_path = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(self.dir_abs_path, "test-results")
        self.templates_dir = os.path.join(self.dir_abs_path, "web-templates")
        self.__prep_dirs()
        self.to_pil = transforms.ToPILImage()

        # -----------------------------

        self.favor = favor_dimension
        self.cell_padding = int(cell_padding)
        self.border_padding = int(border_padding)
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

    def __prep_dirs(self):
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)
        os.makedirs(self.results_dir, exist_ok=True)

    def add(self, section_name: str, caption: str, img: torch.Tensor):
        if section_name not in self.sections:
            self.sections[section_name] = {
                "title": section_name,
                "images": [],
            }
            self.sections[section_name]["bg_color"] = self.palette[
                len(self.sections) % len(self.palette)
            ]

        self.sections[section_name]["caption"] = caption
        pil_img = self.to_pil(
            TensorImgUtils.convert_to_type(img, "CHW")
        )
        img_index = len(self.sections[section_name]["images"]) + 1
        filename = f"{section_name}-{img_index}.png"

        path = os.path.join(self.results_dir, filename)

        img_dict = {
            "title": caption,
            "pil_img": pil_img,
            "path": path,
            "filename": filename,
            "height": pil_img.height,
            "width": pil_img.width,
            "tensor_shape" : img.shape,
            "tensor_format" : TensorImgUtils.identify_type(img)[1],
            "metadata": str(pil_img.info) if pil_img.info else "",
        }
        self.sections[section_name]["images"].append(img_dict)
        self.all_heights.append(pil_img.height)
        self.all_widths.append(pil_img.width)

    def show_webview(self):
        print("\nShowing test results with option: webview")
        print("Normalizing images. Relative differences will still be visualized.\n")
        self.__save_normalized_images()
        env = Environment(loader=FileSystemLoader(self.templates_dir))
        template = env.get_template("test_results_template.html")
        html = template.render(sections=self.sections, title=self.title)
        with open(os.path.join(self.results_dir, "test_results.html"), "w") as f:
            f.write(html)

        subprocess.run(
            [
                OS_Utils.get_xdg_equiv(),
                os.path.join(self.results_dir, "test_results.html"),
            ]
        )

    # def show(self):
    #     path = os.path.join("test/test-results", "test_results.html")
    #     with open(path, "w") as f:
    #         f.write(self.rendered_html)
    #     os.system(f"xdg-open {path}")

    def __best_square_grid(self, n_items) -> Tuple[int, int]:
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

    def __save_normalized_images(self):
        self.cell_w = max(self.all_widths) + self.cell_padding
        self.cell_h = max(self.all_heights) + self.cell_padding

        for section, section_data in self.sections.items():
            bg_color = section_data["bg_color"]
            for img in section_data["images"]:
                # All images have same dimensions, the extent to which they fill the constant-size-canvas depending on their actual/initial size
                padding_x = (
                    0
                    if img["width"] >= self.cell_w - 2
                    else (self.cell_w - img["width"]) // 2
                )
                padding_y = (
                    0
                    if img["height"] >= self.cell_h - 2
                    else (self.cell_h - img["height"]) // 2
                )
                bg_canvas = Image.new(
                    "RGB",
                    (self.cell_w, self.cell_h),
                    bg_color,
                )
                bg_canvas.paste(img["pil_img"], (padding_x, padding_y))
                bg_canvas.save(img["path"])

    # def show_picture(self):
    #     self.rows, self.cols = self.__best_square_grid(len(self.images))
    #     self.__set_dimensions()

    #     # if the sections dont fit neatly, add an extra border_padding to the right
    #     total_h = (
    #         self.rows * self.cell_h
    #         + self.rows * self.border_padding
    #         + self.border_padding
    #     )
    #     total_w = self.cols * self.cell_w + 2 * self.border_padding
    #     section_lengths = {}
    #     for img1 in self.images:
    #         section = img1[0]
    #         if section not in section_lengths:
    #             section_lengths[section] = 0
    #         section_lengths[section] += 1

    #     extra_border = False
    #     for section, length in section_lengths.items():
    #         if length % self.cols != 0:
    #             extra_border = True
    #             break

    #     if extra_border:
    #         total_w += self.border_padding

    #     canvas = Image.new(
    #         "RGB",
    #         (total_w, total_h),
    #         color=self.bg_color,
    #     )

    #     # sort by section
    #     self.images.sort(key=lambda x: x[0])

    #     cur_section = False
    #     for i, (section, caption, img) in enumerate(self.images):
    #         row = i // self.cols
    #         col = i % self.cols

    #         # Center in cell
    #         padding_x = (
    #             0 if img.width >= self.cell_w - 2 else (self.cell_w - img.width) // 2
    #         )
    #         padding_y = (
    #             0 if img.height >= self.cell_h - 2 else (self.cell_h - img.height) // 2
    #         )

    #         # Add border padding to top/bottom, left if first in section, right if last in section
    #         border_left = self.border_padding if col == 0 else 0
    #         border_right = self.border_padding if self.cols % (col + 1) == 0 else 0
    #         # border_top = self.border_padding if row == 0 else 0
    #         border_top = self.border_padding
    #         border_bottom = self.border_padding if self.rows % (row + 1) == 0 else 0

    #         if (i + 1) < len(self.images) and section != self.images[i + 1][0]:
    #             border_right = self.border_padding
    #         if (i - 1) >= 0 and section != self.images[i - 1][0]:
    #             border_left = self.border_padding

    #         # padding_x_b += self.border_padding if col == self.cols - 1 else 0
    #         # padding_y_b += self.border_padding if row == self.rows - 1 else 0

    #         # make background the sectinon color, except don't change the border
    #         bg_canvas = Image.new(
    #             "RGB",
    #             (self.cell_w, self.cell_h),
    #             color=self.section_colors[section],
    #         )
    #         bg_canvas.paste(img, (padding_x, padding_y))

    #         canvas.paste(
    #             bg_canvas,
    #             (
    #                 (col * self.cell_w) + border_left,
    #                 (row * self.cell_h) + ((row + 1) * border_top) + self.cell_padding,
    #             ),
    #         )

    #         # Draw section title if new section
    #         if section != cur_section:
    #             cur_section = section
    #             ImageDraw.Draw(canvas).text(
    #                 (
    #                     col * self.cell_w + self.cell_padding + self.border_padding + 2,
    #                     (
    #                         (
    #                             row
    #                             * (
    #                                 self.cell_h
    #                                 + (self.cell_padding // 2)
    #                                 + self.border_padding
    #                             )
    #                         )
    #                         + border_top
    #                         + self.border_padding
    #                         + 4
    #                     ),
    #                 ),
    #                 section,
    #                 fill=self.TEXT_COLOR,
    #             )
    #             # Draw section title again but smaller (for an outilne/text shadow effect)
    #             ImageDraw.Draw(canvas).text(
    #                 (
    #                     col * self.cell_w
    #                     + self.cell_padding
    #                     + self.border_padding
    #                     + 2
    #                     + 1,
    #                     (
    #                         (
    #                             row
    #                             * (
    #                                 self.cell_h
    #                                 + (self.cell_padding // 2)
    #                                 + self.border_padding
    #                             )
    #                         )
    #                         + border_top
    #                         + self.border_padding
    #                         + 4
    #                         + 1
    #                     ),
    #                 ),
    #                 section,
    #                 fill=self.OFF_TEXT_COLOR,
    #             )

    #         # Draw caption
    #         ImageDraw.Draw(canvas).text(
    #             (
    #                 col * self.cell_w + self.cell_padding + border_left + 2,
    #                 (row * self.cell_h) + self.cell_h - self.cell_padding // 2,
    #             ),
    #             caption,
    #             fill=self.TEXT_COLOR,
    #         )

    #         # Draw caption again but smaller (for an outilne/text shadow effect)
    #         ImageDraw.Draw(canvas).text(
    #             (
    #                 col * self.cell_w + self.cell_padding + border_left + 2 + 1,
    #                 (row * self.cell_h) + self.cell_h - self.cell_padding // 2 + 1,
    #             ),
    #             caption,
    #             fill=self.OFF_TEXT_COLOR,
    #         )

    #     canvas.save(
    #         os.path.join(
    #             self.temp_dirname,
    #             f"{self.filename_prefix}-comparison_grid{time.strftime('%I_%M%p')}.jpg",
    #         )
    #     )

    #     # Caller can use canvas.show() to display the grid with default app automatically, if they want
    #     return canvas
