
Some nodes I created / am creating 🤗. I also wrote a ***[GUIDE](wiki/creating-custom-comfyui_nodes-guide.md)*** for creating custom nodes while I was learning.

----

*Table of Contents:*

- [Nodes](#nodes)
  - [*NODE* — Paste Cutout on Base Image](#node--paste-cutout-on-base-image)
  - [*NODE* — Infer Alpha from RGB Image](#node--infer-alpha-from-rgb-image)
  - [*NODE* — Size Match Images/Masks](#node--size-match-imagesmasks)
- [Custom Node Testing Tools](#custom-node-testing-tools)
  - [Description](#description)
  - [Demo](#demo)
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

# Custom Node Testing Tools

## Description 

- Class that tries to create automatically generate test cases that test every branch of node code which manipulates image tensors
  - E.g., for a node that composites two images, it generates all permutations of image size comparisons (e.g., img1 width > img2 width, img1 height < img2 height, img1 width < img2 width and img1 height > img2 height)
  - In addition to branch coverage, it generates test cases for:
    - Edge cases
      - Prime numbers
      - Bounds (0 or 1)
      - Uncommon file formats
  - Different tensor formats (CHW, HWC, HW)
  - Tesnros with and without batch dimensions
- Organizes and displays test results in a nice webview that highlights all the important information about each test case
  - Also with an image grid option

## Demo

![test suite webview demo gif](wiki/wiki-pics/test-suite-demo/test-results-webview-gif.gif)

![test suite webview screenshot](wiki/wiki-pics/test-suite-demo/test-results-webview-picture.png)



# To-do

- match size node
- parallax node
- comparison grid