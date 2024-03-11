import os
from jinja2 import Environment, FileSystemLoader
from torchvision import transforms
from PIL import Image
from typing import List, Tuple


class ComparisonGridWebView:
    def __init__(self, sections):
        self.sections = sections
        self.save_image_tensors(self.sections)

        self.rendered_html = self.render_html()

    def save_image_tensors(self, sections):
        to_pil = transforms.ToPILImage()
        for section in sections:
            for index, image in enumerate(section["images"]):
                filename = f"{section['title']}-{index}.png"
                path = os.path.join("test/test-results", filename)
                to_pil(image["tensor"]).save(path)
                image["path"] = filename

    def render_html(self):
        env = Environment(
            loader=FileSystemLoader(
                "/home/c_byrne/tools/sd/sd-interfaces/ComfyUI/custom_nodes/elimination-nodes/test/web-templates"
            )
        )  # Assuming templates are stored in a directory named 'templates'
        template = env.get_template(
            "test_results_template.html"
        )  # Load template file named 'test_results_template.html'
        return template.render(sections=self.sections)

    def show(self):
        path = os.path.join("test/test-results", "test_results.html")
        with open(path, "w") as f:
            f.write(self.rendered_html)
        os.system(f"xdg-open {path}")

    def get_html(self):
        return self.rendered_html

    def get_sections(self):
        return self.sections

    def set_sections(self, sections):
        self.sections = sections
        self.rendered_html = self.render_html()
        return self.rendered_html
