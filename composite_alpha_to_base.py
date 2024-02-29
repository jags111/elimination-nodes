import torch
import os
import hashlib

from PIL import Image, ImageOps, ImageSequence
import numpy as np
import cv2
import pickle
from typing import List, Union
from skimage import img_as_float, img_as_ubyte


# sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import folder_paths

# class Example:
#     """
#     A example node

#     Class methods
#     -------------
#     INPUT_TYPES (dict):
#         Tell the main program input parameters of nodes.
#     IS_CHANGED:
#         optional method to control when the node is re executed.

#     Attributes
#     ----------
#     RETURN_TYPES (`tuple`):
#         The type of each element in the output tulple.
#     RETURN_NAMES (`tuple`):
#         Optional: The name of each output in the output tulple.
#     FUNCTION (`str`):
#         The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
#     OUTPUT_NODE ([`bool`]):
#         If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
#         The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
#         Assumed to be False if not present.
#     CATEGORY (`str`):
#         The category the node should appear in the UI.
#     execute(s) -> tuple || None:
#         The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
#         For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
#     """
#     def __init__(self):
#         pass

#     @classmethod
#     def INPUT_TYPES(s):
#         """
#             Return a dictionary which contains config for all input fields.
#             Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
#             Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
#             The type can be a list for selection.

#             Returns: `dict`:
#                 - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
#                 - Value input_fields (`dict`): Contains input fields config:
#                     * Key field_name (`string`): Name of a entry-point method's argument
#                     * Value field_config (`tuple`):
#                         + First value is a string indicate the type of field or a list for selection.
#                         + Secound value is a config for type "INT", "STRING" or "FLOAT".
#         """
#         return {
#             "required": {
#                 "image": ("IMAGE",),
#                 "int_field": ("INT", {
#                     "default": 0,
#                     "min": 0, #Minimum value
#                     "max": 4096, #Maximum value
#                     "step": 64, #Slider's step
#                     "display": "number" # Cosmetic only: display as "number" or "slider"
#                 }),
#                 "float_field": ("FLOAT", {
#                     "default": 1.0,
#                     "min": 0.0,
#                     "max": 10.0,
#                     "step": 0.01,
#                     "round": 0.001, #The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
#                     "display": "number"}),
#                 "print_to_screen": (["enable", "disable"],),
#                 "string_field": ("STRING", {
#                     "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
#                     "default": "Hello World!"
#                 }),
#             },
#         }

#     RETURN_TYPES = ("IMAGE",)
#     #RETURN_NAMES = ("image_output_name",)

#     FUNCTION = "test"

#     #OUTPUT_NODE = False

#     CATEGORY = "Example"

#     def test(self, image, string_field, int_field, float_field, print_to_screen):
#         if print_to_screen == "enable":
#             print(f"""Your input contains:
#                 string_field aka input text: {string_field}
#                 int_field: {int_field}
#                 float_field: {float_field}
#             """)
#         #do some processing on the image, in this example I just invert it
#         image = 1.0 - image
#         return (image,)

#     """
#         The node will always be re executed if any of the inputs change but
#         this method can be used to force the node to execute again even when the inputs don't change.
#         You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
#         executed, if it is different the node will be executed again.
#         This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
#         changes between executions the LoadImage node is executed again.
#     """
#     #@classmethod
#     #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
#     #    return ""

# # Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# # WEB_DIRECTORY = "./somejs"

# # A dictionary that contains all nodes you want to export with their names
# # NOTE: names should be globally unique
# NODE_CLASS_MAPPINGS = {
#     "Example": Example
# }

# # A dictionary that contains the friendly/humanly readable titles for the nodes
# NODE_DISPLAY_NAME_MAPPINGS = {
#     "Example": "Example Node"
# }


class CompositeAlphaToBase:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        # files = [
        #     f
        #     for f in os.listdir(input_dir)
        #     if os.path.isfile(os.path.join(input_dir, f))
        # ]
        return {
            "required": {
                # "base_image": (sorted(files), {"image_upload": True}),
                # "alpha_overlay": (sorted(files), {"image_upload": True}),
                "base_image": ("IMAGE",),
                "alpha_overlay": ("IMAGE",),
            },
        }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"

    """pickle"""

    def read_image(self, filename: str) -> Image:
        return Image.open(filename)

    def pickle_to_file(self, obj: object, file_path: str):
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)

    def load_pickle(self, file_name: str) -> object:
        with open(file_name, "rb") as f:
            obj = pickle.load(f)
        return obj

    def load_light_leak_images(self) -> list:
        file = os.path.join(folder_paths.models_dir, "layerstyle", "light_leak.pkl")
        return self.load_pickle(file)

    """Converter"""

    def cv22ski(self, cv2_image: np.ndarray) -> np.array:
        return img_as_float(cv2_image)

    def ski2cv2(self, ski: np.array) -> np.ndarray:
        return img_as_ubyte(ski)

    def cv22pil(self, cv2_img: np.ndarray) -> Image:
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv2_img)

    def pil2cv2(self, pil_img: Image) -> np.array:
        np_img_array = np.asarray(pil_img)
        return cv2.cvtColor(np_img_array, cv2.COLOR_RGB2BGR)

    def pil2tensor(self, image: Image) -> torch.Tensor:
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    def np2tensor(self, img_np: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
        if isinstance(img_np, list):
            return torch.cat([self.np2tensor(img) for img in img_np], dim=0)
        return torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)

    def tensor2np(self, tensor: torch.Tensor) -> List[np.ndarray]:
        if len(tensor.shape) == 3:  # Single image
            return np.clip(255.0 * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
        else:  # Batch of images
            return [
                np.clip(255.0 * t.cpu().numpy(), 0, 255).astype(np.uint8)
                for t in tensor
            ]

    def image2mask(self, image: Image) -> torch.Tensor:
        _image = image.convert("RGBA")
        alpha = _image.split()[0]
        bg = Image.new("L", _image.size)
        _image = Image.merge("RGBA", (bg, bg, bg, alpha))
        ret_mask = torch.tensor([self.pil2tensor(_image)[0, :, :, 3].tolist()])
        return ret_mask

    def mask2image(self, mask: torch.Tensor) -> Image:
        masks = self.tensor2np(mask)
        for m in masks:
            _mask = Image.fromarray(m).convert("L")
            _image = Image.new("RGBA", _mask.size, color="white")
            _image = Image.composite(
                _image, Image.new("RGBA", _mask.size, color="black"), _mask
            )
        return _image

    def tensor2pil(self, t_image: torch.Tensor) -> Image:
        """Converts a torch tensor to a PIL image with Alpha Channel intact"""
        # https://github.com/chflame163/ComfyUI_LayerStyle/blob/35a7f6e157391f6d3886985fad5279b9af12754d/py/imagefunc.py#L106
        return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    def resize_to_fit(self, image, target_size):
        # Resize the image to be larger than the target size
        aspect_ratio = image.shape[2] / image.shape[3]
        if aspect_ratio > target_size[0] / target_size[1]:
            new_size = (target_size[0], int(target_size[0] / aspect_ratio))
        else:
            new_size = (int(target_size[1] * aspect_ratio), target_size[1])
        image = torch.nn.functional.interpolate(
            image, size=new_size, mode="bilinear", align_corners=False
        )

        # Crop the image to the target size
        crop_top = (image.shape[2] - target_size[0]) // 2
        crop_left = (image.shape[3] - target_size[1]) // 2
        return image[
            :,
            :,
            crop_top : crop_top + target_size[0],
            crop_left : crop_left + target_size[1],
        ]

    def composite(self, base_image, alpha_overlay):
        # https://github.com/chflame163/ComfyUI_LayerStyle/blob/35a7f6e157391f6d3886985fad5279b9af12754d/py/image_blend.py
        # Convert tensors to PIL images
        base_pil = self.tensor2pil(base_image).convert('RGBA')
        overlay_pil = self.tensor2pil(alpha_overlay).convert('RGBA')

        # Resize overlay to match base image
        overlay_pil = overlay_pil.resize(base_pil.size)

        # Extract alpha channel from overlay
        alpha = overlay_pil.split()[-1]

        # Apply alpha channel as mask to overlay
        overlay_pil.putalpha(alpha)

        # Blend overlay onto base image
        blended = Image.alpha_composite(base_pil, overlay_pil)

        # Convert blended image back to tensor
        blended = self.pil2tensor(blended.convert('RGB'))

        return (blended,)