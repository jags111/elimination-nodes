"""
pyenv local 3.10.6
"""

from .nodes.compositers.composite_alpha_to_base_node import *
from .nodes.masks.auto_alpha_mask_node import *
from .nodes.preprocessors.size_match_node import *
from .nodes.preprocessors.layer_shifter import *
from .nodes.config_dicts.parallax_config import *
from .nodes.file_system.save_parallax_step import *

from .equalize import *
from .segment import *

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _assign_class_mappings():
    """
    Assign the class mappings using the below class_mappings dictionary, whose
    items follow this format:
        "node category name" : [
            (
                "name displayed in the web ui",
                "name of the node in the backend" (also displayed in the subtext in the ui),
                reference to the node class
            )
        ]
    """
    node_name_suffix = "🔹Elimination Nodes🔹"
    class_mappings = {
        "🖼️✨🎬 Infinite Parallax": [
            (
                "Layer Shifter for Parallax Outpainting",
                "Wrap-Shift and Mask Slices for Parallax",
                LayerShifterNode,
            ),
            (
                "Parallax Config",
                "Infinite Parallax User Dict",
                ParallaxConfigDictNode,
            ),
            (
                "Save Parallax Iteration",
                "Save Infinite Parallax Step Components",
                SaveParallaxStepNode,
            ),
        ],
        "Compositers": [
            (
                "Paste Cutout on Base Image",
                "Composite Alpha Layer",
                CompositeCutoutOnBaseNode,
            ),
            (
                "Mask from RGB Image by Method",
                "Infer Alpha from Non-RGBA Cutout",
                AutoAlphaMaskNode,
            ),
            (
                "Size Match Images/Masks",
                "Resize Images to Match Size",
                SizeMatchNode,
            ),
        ],
        "Utils": [],
    }

    for node_category, node_mapping in class_mappings.items():
        for display_name, node_name, node_class in node_mapping:
            # Strip and add the suffix to the node name
            node_name = f"{node_name.strip()} {node_name_suffix.strip()}"
            # Assign node class
            NODE_CLASS_MAPPINGS[node_name] = node_class
            # Assign display name
            NODE_DISPLAY_NAME_MAPPINGS[node_name] = display_name.strip()
            # Update the node class's CATEGORY
            node_class.CATEGORY = node_category.strip()


_assign_class_mappings()

# WEB_DIRECTORY = './js'
