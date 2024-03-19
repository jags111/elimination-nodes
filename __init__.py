"""
pyenv local 3.10.6
"""

from .nodes.compositers.composite_alpha_to_base_node import *
from .nodes.masks.auto_alpha_mask_node import *
from .nodes.preprocessors.size_match_node import *
from .nodes.preprocessors.layer_shifter import *
from .nodes.config_dicts.parallax_config import *
from .nodes.file_system.save_parallax_layers import *

from .equalize import *
from .segment import *


CLASS_MAPPINGS = {
    "Infinite Parallax": [
        (
            "Layer Shifter for Parallax Outpainting",
            "Shift and Pad Slices for Parallax Outpainting | Infinite Parallax",
            LayerShifterNode,
        ),
        (
            "Parallax Config",
            "Parallax Config Dict | Infinite Parallax",
            ParallaxConfigDictNode,
        ),
        (
            "Save Parallax Layers",
            "Save Parallax Layers | Infinite Parallax",
            SaveParallaxLayersNode,
        ),
    ],
    "Compositers": [
        (
            "Paste Cutout on Base Image",
            "Composite Alpha Layer | Elimination Nodes",
            CompositeCutoutOnBaseNode,
        ),
        (
            "Infer Alpha from RGB Image",
            "Infer Alpha from Non-RGBA Cutout | Elimination Nodes",
            AutoAlphaMaskNode,
        ),
        (
            "Size Match Images/Masks",
            "Resize Images to Match Size | Elimination Nodes",
            SizeMatchNode,
        ),
    ],
    "Utils": [],
}

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for node_mapping in CLASS_MAPPINGS.values():
    for display_name, node_name, node_class in node_mapping:
        NODE_CLASS_MAPPINGS[node_name] = node_class
        NODE_DISPLAY_NAME_MAPPINGS[node_name] = display_name

# WEB_DIRECTORY = './js'
