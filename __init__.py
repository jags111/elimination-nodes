"""
pyenv local 3.10.6
"""

from .nodes.compositers.composite_alpha_to_base_node import *
from .nodes.masks.auto_alpha_mask_node import *
from .nodes.preprocessors.size_match_node import *

import utils
from .equalize import *
from .segment import *

NODE_CLASS_MAPPINGS = {
    "Composite Alpha Layer | Elimination Nodes": CompositeCutoutOnBaseNode,
    "Infer Alpha from Non-RGBA Cutout | Elimination Nodes": AutoAlphaMaskNode,
    "Resize Images to Match Size | Elimination Nodes": SizeMatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Composite Alpha Layer | Elimination Nodes": "Paste Cutout on Base Image",
    "Infer Alpha from Non-RGBA Cutout | Elimination Nodes": "Infer Alpha from RGB Image",
    "Resize Images to Match Size | Elimination Nodes": "Size Match Images/Masks",
}

# WEB_DIRECTORY = './js'
