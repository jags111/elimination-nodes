# Some Custom ComfyUI Nodes

Some nodes I created / am creating 🤗

I wrote a ***[GUIDE](wiki/creating-custom-comfyui_nodes-guide.md)*** for creating custom nodes while I was learning how to make these, it covers the general process of creating a custom node and some of the errors I encountered and how to fix them.

# Nodes

The custom node highlighted is red in the screenshots

## Paste Cutout on Base Image

- Automatically matches size of two images with various size matching methods
- Invert cutout option
- Useful for creating logos and when doing things that require object segmentation/removal

![paste-cutout-on-base-image-demo_pic](wiki/wiki-pics/node-demos/paste-cutout-on-base-image-demo_pic.png)

Alongside auto segmentation

![paste-cutout-on-base-image-demo_pic-with_segmentation](wiki/wiki-pics/node-demos/paste-cutout-on-base-image-with_segmentation-demo_pic.png)


## Infer Alpha from RGB Image

- Chromakeying, remove white bg, remove black bg, remove neutrals, remove non-neutrals, remove by color
- Invert option
- Leniance/Tolerance/Threshold slider
- When you have an image that clearly has layers or is supposed to be a cutout but doesn't have an alpha channel, or you have lost the alpha channel at some point in the your workflow, and auto segmentation is not applicable

![infer-alpha-from-rgb-image-demo_pic](wiki/wiki-pics/node-demos/infer_alpha_from_rgb_image-demo.png)

---------------------

# Test Suite

![Test Suite](test/test_composite_alpha_to_base)

# TODO

- rename class
- match size node
- parallax node
- comparison grid