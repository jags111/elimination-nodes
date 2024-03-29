"""Doc Strings automatically generated

pyenv local 3.10.6"""

import torch
from PIL import Image

from termcolor import colored
from typing import Tuple

from transformers import BlipProcessor, BlipForConditionalGeneration
from torchvision import transforms

try:
    from ...utils.tensor_utils import TensorImgUtils
except ImportError:
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.tensor_utils import TensorImgUtils


class Img2TxtBlipNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "conditional_image_captioning": (
                    "STRING",
                    {
                        "default": "a photograph of",
                        "multiline": True,
                    },
                ),
                "exclude_terms": (
                    "STRING",
                    {
                        "default": "",
                    },
                ),
                "skip_special_tokens": (
                    "BOOLEAN",
                    {"default": True, "label_on": "Skip", "label_off": "Keep"},
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
            "optional": {
                "output_text": (
                    "STRING",
                    {
                        "default": "",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "main"

    # INPUT_IS_LIST = True
    # OUTPUT_NODE = True
    # OUTPUT_IS_LIST = (True,)

    def main(
        self,
        input_image: torch.Tensor,  # [Batch_n, H, W, 3-channel]
        conditional_image_captioning: str,
        exclude_terms: str,
        skip_special_tokens: bool,
        output_text: str,
        unique_id=None,
        extra_pnginfo=None,
    ) -> Tuple[str, ...]:
        
        print(colored(f"kwards dict: {locals()}", "green"))
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to("cuda")

        raw_image = transforms.ToPILImage()(
            TensorImgUtils.convert_to_type(input_image, "CHW")
        ).convert("RGB")

        if conditional_image_captioning == "":
            # unconditional image captioning
            inputs = processor(raw_image, return_tensors="pt").to("cuda")
        else:
            # conditional image captioning
            inputs = processor(
                raw_image, conditional_image_captioning, return_tensors="pt"
            ).to("cuda")

        out = model.generate(**inputs)
        out_string = processor.decode(out[0], skip_special_tokens=skip_special_tokens)

        if exclude_terms != "":
            exclude_terms = [
                term.strip().lower() for term in exclude_terms.split(",") if term != ""
            ]
            for term in exclude_terms:
                out_string = out_string.replace(term, "")

        print(colored(f"{extra_pnginfo['workflow']}", "green"))
        print(colored(f"unique_id: {unique_id}", "green"))

        if unique_id and extra_pnginfo and "workflow" in extra_pnginfo:
            workflow = extra_pnginfo["workflow"]
            node = [x for x in workflow["nodes"] if str(x["id"]) == unique_id[0]][0]

            # node = next(
            #     (x for x in workflow["nodes"] if str(x["id"]) == unique_id[0]), None
            # )
            if node:
                print(colored(f"node: {node['widgets_values']}", "green"))
                node["widgets_values"] = [
                    out_string
                ]

        return {"ui": {"text": out_string}, "result": (out_string,)}
