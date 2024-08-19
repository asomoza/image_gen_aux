from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import PIL.Image
import torch


class Preprocessor(ABC):
    """
    This abstract base class defines the interface for image preprocessors.

    Subclasses should implement the abstract methods `from_pretrained` and
    `__call__` to provide specific loading and preprocessing logic for their
    respective models.

    Args:
        model (nn.Module): The torch model to be used.
    """

    def __init__(self, model):
        self.model = model

    def to(self, device):
        """
        Moves the underlying model to the specified device
        (e.g., CPU or GPU).

        Args:
            device (torch.device): The target device.

        Returns:
            Preprocessor: The preprocessor object itself (for method chaining).
        """
        self.model = self.model.to(device)
        return self

    @abstractmethod
    def from_pretrained(self):
        """
        This abstract method defines how the preprocessor loads pre-trained
        weights or configurations specific to the model it supports. Subclasses
        must implement this method to handle model-specific loading logic.

        This method might download pre-trained weights from a repository or
        load them from a local file depending on the model's requirements.
        """
        pass

    @abstractmethod
    def __call__(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        resolution_scale: float = 1.0,
        invert: bool = True,
        return_type: str = "pil",
    ):
        """
        Preprocesses an image for use with the underlying model.

        Args:
            image (Union[PIL.Image.Image, np.ndarray, torch.Tensor]): The input
                image in PIL Image format, NumPy array, or PyTorch tensor.
            resolution_scale (float, optional): A scaling factor to apply to
                the image resolution. Defaults to 1.0 (no scaling).
            invert (bool, optional): Whether to invert the image.
                Defaults to True.
            return_type (str, optional): The desired return type, either "pt" for PyTorch tensor,
                "np" for NumPy array, or "pil" for PIL image. Defaults to "pil" for PIL Image format.

        Returns:
            Union[PIL.Image.Image, torch.Tensor]: The preprocessed image in the
                specified format.
        """
        pass
