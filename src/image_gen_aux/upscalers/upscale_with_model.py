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
from huggingface_hub import hf_hub_download, model_info
from spandrel import ImageModelDescriptor, ModelLoader

from ..image_processor import ImageMixin
from ..utils import tiled_upscale


class UpscaleWithModel(ImageMixin):
    r"""
    Upscaler class that uses a pytorch model.

    Args:
        model ([`ImageModelDescriptor`]):
            Upscaler model, must be supported by spandrel.
        scale (`int`, defaults to the scale of the model):
            The number of times to scale the image, it is recommended to use the model default scale which
            usually is what the model was trained for.
    """

    def __init__(self, model: ImageModelDescriptor, scale: int = None):
        super().__init__()
        self.model = model

    def to(self, device):
        self.model.to(device)
        return self

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, filename=None, subfolder=None):
        r"""
        Instantiate the Upscaler class from pretrained weights.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *repo id* (for example `OzzyGT/UltraSharp`) of a pretrained model
                      hosted on the Hub, must be saved in safetensors. If there's more than one checkpoint
                      in the repository and the filename wasn't specified, the first one found will be loaded.
                    - A path to a *directory* (for example `./upscaler_model/`) containing a pretrained
                      upscaler checkpoint.
            filename (`str`, *optional*):
                The name of the file in the repo.
            subfolder (`str`, *optional*):
                An optional value corresponding to a folder inside the model repo.
        """
        if filename is None:
            info = model_info(pretrained_model_or_path)
            filename = next(
                (
                    sibling.rfilename
                    for sibling in info.siblings
                    if os.path.splitext(sibling.rfilename)[1] == ".safetensors"
                ),
                None,
            )

            if filename is None:
                raise FileNotFoundError("No safetensors checkpoint found.")

        model_path = hf_hub_download(pretrained_model_or_path, filename, subfolder=subfolder)
        model = ModelLoader().load_from_file(model_path)

        # validate that it's the correct model
        assert isinstance(model, ImageModelDescriptor)

        return cls(model)

    @torch.inference_mode
    def __call__(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        tiling: bool = False,
        tile_width: int = 512,
        tile_height: int = 512,
        overlap: int = 8,
        return_type: str = "pil",
    ):
        r"""
        Function invoked when calling the upscaler.

        Args:
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`):
                The initial image will be upscaled
        """
        if not isinstance(image, torch.Tensor):
            image = self.convert_image_to_tensor(image)

        image = image.to(self.model.device)

        if tiling:
            upscaled_tensor = tiled_upscale(image, self.model, self.model.scale, tile_width, tile_height, overlap)
        else:
            upscaled_tensor = self.model(image)

        image = self.post_process_image(upscaled_tensor, return_type)[0]

        return image
