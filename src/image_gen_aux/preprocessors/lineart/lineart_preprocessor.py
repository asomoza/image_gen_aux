# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import Union

import numpy as np
import PIL.Image
import torch
from safetensors.torch import load_file

from ...image_processor import ImageMixin
from ...utils import SAFETENSORS_FILE_EXTENSION, get_model_path
from ..preprocessor import Preprocessor
from .model import Generator


class LineArtPreprocessor(Preprocessor, ImageMixin):
    """Preprocessor specifically designed for converting images to line art.

    This class inherits from both `Preprocessor` and `ImageMixin`. Please refer to each
    one to get more information.
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_or_path: Union[str, os.PathLike],
        filename: str = None,
        subfolder: str = None,
        weights_only: bool = True,
    ) -> Generator:
        model_path = get_model_path(pretrained_model_or_path, filename, subfolder)

        file_extension = os.path.basename(model_path).split(".")[-1]
        if file_extension == SAFETENSORS_FILE_EXTENSION:
            state_dict = load_file(model_path, device="cpu")
        else:
            state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=weights_only)

        model = Generator()
        model.load_state_dict(state_dict)

        return cls(model)

    @torch.inference_mode
    def __call__(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        resolution_scale: float = 1.0,
        invert: bool = True,
        return_type: str = "pil",
    ):
        """Preprocesses an image and generates line art using the pre-trained model.

        Args:
            image (Union[PIL.Image.Image, np.ndarray, torch.Tensor]): Input image in PIL Image,
                NumPy array, or PyTorch tensor format.
            resolution_scale (float, optional): Scale factor for image resolution during
                preprocessing and post-processing (defaults to 1.0 for no scaling).
            invert (bool, optional): Inverts the generated image if True (white or black background)
                Defaults to True.
            return_type (str, optional): The desired return type, either "pt" for PyTorch tensor, "np" for NumPy array,
                or "pil" for PIL image. Defaults to "pil" for PIL Image format.

        Returns:
            Union[PIL.Image.Image, np.ndarray, torch.Tensor]: The generated line art in the
                specified output format.
        """
        if not isinstance(image, torch.Tensor):
            image = self.convert_image_to_tensor(image)

        input_image = self.scale_image(image, resolution_scale) if resolution_scale != 1.0 else image

        input_image = input_image.to(self.model.device)
        lineart = self.model(input_image)

        if invert:
            lineart = 255 - lineart

        if resolution_scale != 1.0:
            lineart = self.scale_image(lineart, 1 / resolution_scale)

        image = self.post_process_image(lineart, return_type)

        return image
