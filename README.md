
Some nodes I created / am creating 🤗. I also wrote a ***[GUIDE](wiki/creating-custom-comfyui_nodes-guide.md)*** for creating custom nodes while I was learning.

----

*Table of Contents:*

- [Nodes](#nodes)
  - [*NODE* — Paste Cutout on Base Image](#node--paste-cutout-on-base-image)
  - [*NODE* — Infer Alpha from RGB Image](#node--infer-alpha-from-rgb-image)
  - [*NODE* — Size Match Images/Masks](#node--size-match-imagesmasks)
- [Custom Node Testing Tools](#custom-node-testing-tools)
  - [Purposes](#purposes)
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
- When you have an image that clearly has layers or is supposed to be a cutout but doesn't have an alpha channel, or you have lost the alpha channel at some point in your workflow, and auto segmentation is not applicable

![infer-alpha-from-rgb-image-demo_pic](wiki/wiki-pics/node-demos/infer_alpha_from_rgb_image-demo.png)


## *NODE* — Size Match Images/Masks

- Automatically matches size of two images with various size matching methods


---------------------

# Custom Node Testing Tools

## Purposes 

- Allow for fast testing/debugging of custom nodes without requiring you to constantly relaunch comfy process or suffer from inconsistencies with the comfy webview's grid's state not updating the way you expect between tests
- Work with `unittest`
- Compare speed/efficiency of different methods
- Auto-generate permutations for full branch coverage of a custom node's processes
  - For IMAGE inputs, generate all permutations of image size comparisons (e.g., case 1: img1 width > img2 width and img1 height > img2 height, case 2: img1 width < img2 width and img1 height < img2 height,...)
  - For IMAGE inputs, generate all permutations of tensor formats (e.g., test with CHW, HWC, BCHW, HW, etc.)
  - For Number inputs, generate all permutations of number comparisons (e.g., case 1: num1 > num2, case 2: num1 < num2,...)
  - For Selection inputs, generate all permutations of selection choices
  - etc...
- Optionally, auto-generate Edge cases to fill out coverage until a threshold is hit
  - Prime numbers
  - Bounds (0 or 1)
  - Uncommon file formats
- Help identify issues with mismatched tensor shapes/sizes/formats more easily
- Organize and display test results, particularly results that involve generated/modified images
  - In a nice bootstrap webview that highlights all the important information about each test case
  - Or with a generated image grid that composites together all results

## Demo

![test suite webview demo gif](wiki/wiki-pics/test-suite-demo/test-results-webview-gif2.gif)

![test suite webview screenshot](wiki/wiki-pics/test-suite-demo/test-results-webview-picture.png)



# To-do

- match size node
- parallax node