

----

*Table of Contents:*

- [Nodes](#nodes)
  - [Compositing Nodes](#compositing-nodes)
    - [*NODE* — Paste Cutout on Base Image](#node--paste-cutout-on-base-image)
    - [*NODE* — Infer Alpha from RGB Image](#node--infer-alpha-from-rgb-image)
  - [Infinite Parallax Nodes](#infinite-parallax-nodes)
    - [*NODE* — Layer Shifter for Parallax Outpainting](#node--layer-shifter-for-parallax-outpainting)
    - [*NODE* — Parallax Config](#node--parallax-config)
    - [*NODE* — Save Parallax Frame](#node--save-parallax-frame)
    - [*NODE* — Load Parallax Frame](#node--load-parallax-frame)
    - [*NODE* — Create Parallax Video](#node--create-parallax-video)
    - [*NODE* — Infinite Parallax - 3D Parallax](#node--infinite-parallax---3d-parallax)
  - [Utility Nodes](#utility-nodes)
    - [*NODE* — Size Match Images/Masks](#node--size-match-imagesmasks)
- [To-do](#to-do)


&nbsp;

Custom [ComfyUI](https://github.com/comfyanonymous/ComfyUI) Nodes 🤗. I also wrote a ***[GUIDE](wiki/creating-custom-comfyui_nodes-guide.md)*** for creating custom nodes.


# Nodes

The custom node highlighted is red in the screenshots

## Compositing Nodes

### *NODE* — Paste Cutout on Base Image

- Automatically matches size of two images with various size matching methods
- If the cutout doesn't have an alpha channel (not really a cutout), the bg is automatically inferred and made transparent
- Invert option
- Useful for creating logos and when doing things that require object segmentation/removal

![paste-cutout-on-base-image-demo_pic](wiki/wiki-pics/node-demos/paste-cutout-on-base-image-demo_pic.png)


**When the cutout doesn't have an alpha channel (BG auto inferred)**

![paste-cutout-on-base-image-demo_pic-infer_bg](wiki/wiki-pics/node-demos/paste-cutout-on-base-image-inferred_bg-demo_pic.png)


**Alongside auto segmentation**

![paste-cutout-on-base-image-demo_pic-with_segmentation](wiki/wiki-pics/node-demos/paste-cutout-on-base-image-with_segmentation-demo_pic.png)


### *NODE* — Infer Alpha from RGB Image

- Chromakeying, remove white bg, remove black bg, remove neutrals, remove non-neutrals, remove by color
- Invert option
- Leniance/Tolerance/Threshold slider
- When you have an image that clearly has layers or is supposed to be a cutout but doesn't have an alpha channel, or you have lost the alpha channel at some point in your workflow, and auto segmentation is not applicable

![infer-alpha-from-rgb-image-demo_pic](wiki/wiki-pics/node-demos/infer_alpha_from_rgb_image-demo.png)


## Infinite Parallax Nodes

### *NODE* — Layer Shifter for Parallax Outpainting

### *NODE* — Parallax Config

### *NODE* — Save Parallax Frame

### *NODE* — Load Parallax Frame

### *NODE* — Create Parallax Video


### *NODE* — Infinite Parallax - 3D Parallax

## Utility Nodes

### *NODE* — Size Match Images/Masks

- Automatically matches size of two images with various size matching methods

# To-do

- tests
  - match size node tests