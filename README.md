
Some nodes I created / am creating 🤗

I wrote a ***[GUIDE](wiki/creating-custom-comfyui_nodes-guide.md)*** for creating custom nodes while I was learning how to make these, it covers the general process of creating a custom node and some of the errors I encountered and how to fix them.


*Table of Contents:*

- [Nodes](#nodes)
  - [*NODE* — Paste Cutout on Base Image](#node--paste-cutout-on-base-image)
  - [*NODE* — Infer Alpha from RGB Image](#node--infer-alpha-from-rgb-image)
  - [*NODE* — Size Match Images/Masks](#node--size-match-imagesmasks)
- [Test Suite](#test-suite)
- [To-do](#to-do)


&nbsp;

# Nodes

The custom node highlighted is red in the screenshots

## *NODE* — Paste Cutout on Base Image

- Automatically matches size of two images with various size matching methods
- If the cutout doesn't have an alpha channel (not really a cutout), the bg is automatically inferred and made transparent
- Invert option
- Useful for creating logos and when doing things that require object segmentation/removal

![paste-cutout-on-base-image-demo_pic](wiki/wiki-pics/node-demos/paste-cutout-on-base-image-demo_pic.png)


**When the cutout doesn't have an alpha channel (BG auto inferred)**

![paste-cutout-on-base-image-demo_pic-infer_bg](wiki/wiki-pics/node-demos/paste-cutout-on-base-image-inferred_bg-demo_pic.png)


**Alongside auto segmentation**

![paste-cutout-on-base-image-demo_pic-with_segmentation](wiki/wiki-pics/node-demos/paste-cutout-on-base-image-with_segmentation-demo_pic.png)


## *NODE* — Infer Alpha from RGB Image

- Chromakeying, remove white bg, remove black bg, remove neutrals, remove non-neutrals, remove by color
- Invert option
- Leniance/Tolerance/Threshold slider
- When you have an image that clearly has layers or is supposed to be a cutout but doesn't have an alpha channel, or you have lost the alpha channel at some point in the your workflow, and auto segmentation is not applicable

![infer-alpha-from-rgb-image-demo_pic](wiki/wiki-pics/node-demos/infer_alpha_from_rgb_image-demo.png)


## *NODE* — Size Match Images/Masks

- Automatically matches size of two images with various size matching methods


---------------------

# Test Suite

![Test Suite](test/test_composite_alpha_to_base)

# To-do

- match size node
- parallax node
- comparison grid