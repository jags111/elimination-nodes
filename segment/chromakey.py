import torch
from typing import Tuple
from ..utils.tensor_utils import TensorImgUtils


class ChromaKey:
    """
    0 = full transparency
    1 = full opacity

    """

    def __init__(self):
        pass

    def __validate_threshold(self, threshold: float) -> float:
        """Validates threshold value, which must be between 0 and 1."""
        if threshold < 0:
            return 0.01
        if threshold > 1:
            return 0.99
        return threshold

    def __package_return(
        self, image: torch.Tensor, alpha: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle when no alpha generated
        if alpha.sum() == 0:
            alpha = torch.zeros_like(image[0, ...])

        rgba = torch.cat((image, alpha.unsqueeze(0)), dim=0)
        mask = 1 - alpha
        return rgba, alpha, mask

    def remove_specific_rgb(
        self, image: torch.Tensor, rgb: Tuple[int, int, int], leniance: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Removes a specific RGB color from the image."""
        leniance = self.__validate_threshold(leniance)
        custom_rgb_tensor = torch.tensor(rgb, dtype=torch.float32) / 255.0
        custom_rgb_tensor = custom_rgb_tensor.view(3, 1, 1).expand(
            3, image.shape[1], image.shape[2]
        )

        threshold = self.__validate_threshold(0.45 + leniance)

        alpha = torch.where(
            # Total differences across all channels > threshold
            (torch.abs(image - custom_rgb_tensor) > threshold).sum(dim=0) > 0,
            torch.tensor(1.0),
            torch.tensor(0.0),
        )
        return self.__package_return(image, alpha)

    def remove_neutrals(
        self, image: torch.Tensor, leniance: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.remove_by_diff(image, 0.14 + leniance, "less")

    def remove_non_neutrals(
        self, image: torch.Tensor, leniance: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """When the difference between the maximum and minimum value of the image is greater than the threshold, the pixel is assumed to not be a color, because it is too bright or too dark."""
        return self.remove_by_diff(image, 0.78 - leniance, "greater")

    def remove_white(
        self, image: torch.Tensor, leniance: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.remove_by_threshold(image, 0.96 - leniance, "greater")

    def remove_black(
        self, image: torch.Tensor, leniance: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.remove_by_threshold(image, 0.04 + leniance, "less")

    def remove_by_diff(
        self, image: torch.Tensor, threshold: float, comparison_type: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if comparison_type == "greater":
            alpha = torch.where(
                (image.max(dim=0).values - image.min(dim=0).values)
                > self.__validate_threshold(threshold),
                torch.tensor(0.0),
                torch.tensor(1.0),
            )
        elif comparison_type == "less":
            alpha = torch.where(
                (image.max(dim=0).values - image.min(dim=0).values)
                < self.__validate_threshold(threshold),
                torch.tensor(0.0),
                torch.tensor(1.0),
            )
        return self.__package_return(image, alpha)

    def remove_by_threshold(
        self, image: torch.Tensor, threshold: float, comparison_type: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        channels_mean = image.mean(dim=0)
        threshold = self.__validate_threshold(threshold)

        if comparison_type == "greater":
            alpha = torch.where(
                channels_mean > threshold,
                torch.tensor(0.0),
                torch.tensor(1.0),
            )
        elif comparison_type == "bounded":
            lower_bound = self.__validate_threshold(abs(threshold + 0.02))
            upper_bound = self.__validate_threshold(abs(1 - (threshold + 0.02)))
            alpha = torch.where(
                (channels_mean < lower_bound) & (channels_mean > upper_bound),
                torch.tensor(0.0),
                torch.tensor(1.0),
            )
        elif comparison_type == "less":
            alpha = torch.where(
                channels_mean < threshold,
                torch.tensor(0.0),
                torch.tensor(1.0),
            )

        return self.__package_return(image, alpha)
