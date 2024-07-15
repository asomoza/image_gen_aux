# UPSCALERS

## Upscale with model

Class to upscale images with a safetensor checkpoint. We use [spandrel](https://github.com/chaiNNer-org/spandrel) for loading, and you can see the list of supported models [here]([list](https://github.com/chaiNNer-org/spandrel?tab=readme-ov-file#model-architecture-support)).

Most of the super-resolution models are provided as `pickle` checkpoints, which are considered unsafe. We promote the use of safetensor checkpoints, and for convenient use, we recommend using the Hugging Face Hub. You can still use a locally downloaded model.

### How to use

```python
from image_gen_aux import UpscaleWithModel
from image_gen_aux.utils import load_image

original = load_image(
    "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/simple_upscale/hippowaffle_small.png"
)

upscaler = UpscaleWithModel.from_pretrained("OzzyGT/UltraSharp").to("cuda")
image = upscaler(original)
image.save("upscaled.png")

```

### List of safetensors checkpoints

This is the current list of safetensor checkpoints you can use from the hub.

|Model|Scale|Repository|Owner|
|---|---|---|---|
|UltraSharp|4X|OzzyGT/UltraSharp|[Kim2091](https://huggingface.co/Kim2091)|
|DAT|4X|OzzyGT/DAT_X4|[zhengchen1999](https://github.com/zhengchen1999)|

If you own the model and want us to change the repository to your name/organization please open an issue.
