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
                "l1_height": (
                    "INT",
                    {
                        "min": 0,
                        "default": 100,
                    },
                ),
                "l1_velocity": (
                    "FLOAT",
                    {"default": 150.0, "min": 0.0, "round": 0.001},
                ),
            },
            "optional": {
                "l2_height": (
                    "INT",
                    {
                        "min": 0,
                        "default": 80,
                    },
                ),
                "l2_velocity": (
                    "FLOAT",
                    {"default": 60.0, "min": 0.0, "round": 0.001},
                ),
                "l3_height": (
                    "INT",
                    {
                        "min": 0,
                        "default": 200,
                    },
                ),
                "l3_velocity": (
                    "FLOAT",
                    {"default": 190.0, "min": 0.0, "round": 0.001},
                ),
                "l4_height": (
                    "INT",
                    {
                        "min": 0,
                        "default": 600,
                    },
                ),
                "l4_velocity": (
                    "FLOAT",
                    {"default": 240.0, "min": 0.0, "round": 0.001},
                ),
                "l5_height": (
                    "INT",
                    {
                        "min": 0,
                    },
                ),
                "l5_velocity": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "round": 0.001},
                ),
                "l6_height": (
                    "INT",
                    {
                        "min": 0,
                    },
                ),
                "l6_velocity": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "round": 0.001},
                ),
                "l7_height": (
                    "INT",
                    {
                        "min": 0,
                    },
                ),
                "l7_velocity": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "round": 0.001},
                ),
            },
        }

    def main(
        self,
        l1_height: int,
        l1_velocity: float,
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
    ) -> Tuple[torch.Tensor, ...]:

        config = [
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
        config = [
            x
            for x in config
            if x["height"] is not None and x["height"] > 0 and x["velocity"] is not None
        ]

        # To json string
        config = json.dumps(config)

        return (config,)
