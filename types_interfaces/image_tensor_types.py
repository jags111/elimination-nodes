from typing import Tuple, Union, Type
import torch


class ImageTensorTypes:
    """
    This class is a collection of type hints for image tensors.
    """

    @classmethod
    def is_valid_image_type(cls, variable) -> bool:
        """Validate if the variable matches one of the defined image types."""
        return (
            any(
                isinstance(variable, t)
                for t in cls.__dict__.values()
                if isinstance(t, tuple)
            )
        )

    @classmethod
    def is_type(cls, tensor: torch.Tensor, type: Type) -> bool:
        """Validate if the tensor is of the specified type."""
        return isinstance(tensor, type)

    BatchDim = int
    R_G_B_Channels = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    R_G_B_A_Channels = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    HeightDim = int
    WidthDim = int

    B_C_H_W_Tensor = Tuple[BatchDim, torch.Tensor, int, int]
    C_H_W_Tensor = Tuple[torch.Tensor, int, int]
    H_W_C_Tensor = Tuple[int, int, torch.Tensor]
    B_H_W_C_Tensor = Tuple[BatchDim, int, int, torch.Tensor]

    RGB_H_W_Tensor = Tuple[R_G_B_Channels, HeightDim, WidthDim]
    RGB_H_W_Tensor_Optional = Union[RGB_H_W_Tensor, None]
    RGBA_H_W_Tensor = Tuple[R_G_B_A_Channels, HeightDim, WidthDim]
    RGBA_H_W_Tensor_Optional = Union[RGBA_H_W_Tensor, None]

    B_RGB_H_W_Tensor = Tuple[BatchDim, R_G_B_Channels, HeightDim, WidthDim]
    B_RGB_H_W_Tensor_Optional = Union[B_RGB_H_W_Tensor, None]
    B_RGBA_H_W_Tensor = Tuple[BatchDim, R_G_B_A_Channels, HeightDim, WidthDim]
    B_RGBA_H_W_Tensor_Optional = Union[B_RGBA_H_W_Tensor, None]

    H_W_RGB_Tensor = Tuple[HeightDim, WidthDim, R_G_B_Channels]
    H_W_RGB_Tensor_Optional = Union[H_W_RGB_Tensor, None]
    H_W_RGBA_Tensor = Tuple[HeightDim, WidthDim, R_G_B_A_Channels]
    H_W_RGBA_Tensor_Optional = Union[H_W_RGBA_Tensor, None]

    B_H_W_RGB_Tensor = Tuple[BatchDim, HeightDim, WidthDim, R_G_B_Channels]
    B_H_W_RGB_Tensor_Optional = Union[B_H_W_RGB_Tensor, None]
    B_H_W_RGBA_Tensor = Tuple[BatchDim, HeightDim, WidthDim, R_G_B_A_Channels]
    B_H_W_RGBA_Tensor_Optional = Union[B_H_W_RGBA_Tensor, None]

    # Masks
    H_W_Tensor = Tuple[HeightDim, WidthDim]
    H_W_Tensor_Optional = Union[H_W_Tensor, None]

    H_W_A_Tensor = Tuple[HeightDim, WidthDim, torch.Tensor]
    H_W_A_Tensor_Optional = Union[H_W_A_Tensor, None]
    A_H_W_Tensor = Tuple[torch.Tensor, HeightDim, WidthDim]
    A_H_W_Tensor_Optional = Union[A_H_W_Tensor, None]
