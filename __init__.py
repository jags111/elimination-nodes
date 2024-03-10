from .nodes.compositers.composite_alpha_to_base import *
from .nodes.masks.auto_alpha_mask import *

NODE_CLASS_MAPPINGS = {
    'Composite Alpha Layer | Elimination Nodes': CompositeCutoutOnBase,
    "Infer Alpha from Non-RGBA Cutout | Elimination Nodes": AutoAlphaMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'Composite Alpha Layer | Elimination Nodes': 'Paste Cutout on Base Image',
    "Infer Alpha from Non-RGBA Cutout | Elimination Nodes": "Infer Alpha from RGB Image"
}

# WEB_DIRECTORY = './js'