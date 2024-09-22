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


CONFIG_NAME = "config.json"
HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
SAFETENSORS_FILE_EXTENSION = "safetensors"
SAFETENSORS_WEIGHTS_NAME = "image_gen_aux_pytorch_model.safetensors"
SAFETENSORS_WEIGHTS_INDEX_NAME = "image_gen_aux_pytorch_model.safetensors.index.json"
WEIGHTS_NAME = "image_gen_aux_pytorch_model.bin"
WEIGHTS_INDEX_NAME = "image_gen_aux_pytorch_model.bin.index.json"
