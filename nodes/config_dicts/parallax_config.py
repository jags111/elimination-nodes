"""Method signatures automatically generated

pyenv local 3.10.6"""

import torch
import json
from typing import Tuple, Union


class ParallaxConfigDictNode:
    CATEGORY = "image"
    RETURN_TYPES = ("parallax_config",)
    FUNCTION = "main"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unique_project_name": (
                    "STRING",
                    {"default": "my-project", "multiline": False},
                ),
                "l1_height": (
                    "INT",
                    {
                        "min": 0,
                        "default": 100,
                        "step": 10,
                    },
                ),
                "l1_velocity": (
                    "FLOAT",
                    {"default": 150.0, "min": 0.0, "round": 0.01, "step": 5},
                ),
            },
            "optional": {
                "num_iterations": (
                    "INT",
                    {
                        "default": 10,
                        "min": 3,
                        "max": 50,
                        "step": 1,
                        "display": "slider",
                    },
                ),
                "fps": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": .001,
                        "max": 30.002,
                        "step": .0001,
                        "round" : .0001,
                        "display": "slider",
                    },
                ),
                "l2_height": (
                    "INT",
                    {
                        "min": 0,
                        "default": 80,
                        "step": 10,
                    },
                ),
                "l2_velocity": (
                    "FLOAT",
                    {"default": 60.0, "min": 0.0, "round": 0.01, "step": 5},
                ),
                "l3_height": (
                    "INT",
                    {
                        "min": 0,
                        "default": 200,
                        "step": 10,
                    },
                ),
                "l3_velocity": (
                    "FLOAT",
                    {"default": 190.0, "min": 0.0, "round": 0.01, "step": 5},
                ),
                "l4_height": (
                    "INT",
                    {
                        "min": 0,
                        "default": 600,
                        "step": 10,
                    },
                ),
                "l4_velocity": (
                    "FLOAT",
                    {"default": 240.0, "min": 0.0, "round": 0.01, "step": 5},
                ),
                "l5_height": (
                    "INT",
                    {
                        "min": 0,
                        "step": 10,
                    },
                ),
                "l5_velocity": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "round": 0.01, "step": 5},
                ),
                "l6_height": (
                    "INT",
                    {
                        "min": 0,
                        "step": 10,
                    },
                ),
                "l6_velocity": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "round": 0.01, "step": 5},
                ),
                "l7_height": (
                    "INT",
                    {
                        "min": 0,
                        "step": 10,
                    },
                ),
                "l7_velocity": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "round": 0.01, "step": 5},
                ),
            },
        }

    def main(
        self,
        unique_project_name: str,
        l1_height: int,
        l1_velocity: float,
        num_iterations: int = 10,
        fps: float = .020,
        l2_height: Union[int, None] = None,
        l2_velocity: Union[float, None] = None,
        l3_height: Union[int, None] = None,
        l3_velocity: Union[float, None] = None,
        l4_height: Union[int, None] = None,
        l4_velocity: Union[float, None] = None,
        l5_height: Union[int, None] = None,
        l5_velocity: Union[float, None] = None,
        l6_height: Union[int, None] = None,
        l6_velocity: Union[float, None] = None,
        l7_height: Union[int, None] = None,
        l7_velocity: Union[float, None] = None,
    ) -> Tuple[str, ...]:

        config = {
            # "unique_project_name": unique_project_name,
        }

        layers = [
            {
                "height": l1_height,
                "velocity": l1_velocity,
            },
            {
                "height": l2_height,
                "velocity": l2_velocity,
            },
            {
                "height": l3_height,
                "velocity": l3_velocity,
            },
            {
                "height": l4_height,
                "velocity": l4_velocity,
            },
            {
                "height": l5_height,
                "velocity": l5_velocity,
            },
            {
                "height": l6_height,
                "velocity": l6_velocity,
            },
            {
                "height": l7_height,
                "velocity": l7_velocity,
            },
        ]

        # Filter out None values
        layers = [
            x
            for x in layers
            if x["height"] is not None and x["height"] > 0 and x["velocity"] is not None
        ]

        # Create top, bottom for each layer
        cur_height = 0
        for layer in layers:
            height = int(layer["height"])
            top = cur_height
            layer["top"] = top
            bottom = cur_height + height
            layer["bottom"] = bottom
            cur_height += height

        config["layers"] = layers
        config["unique_project_name"] = unique_project_name
        config["num_iterations"] = int(num_iterations)
        config["fps"] = float(fps)
        config = json.dumps(config)

        return (config,)
