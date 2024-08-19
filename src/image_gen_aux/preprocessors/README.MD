# PREPROCESSORS

## Supported preprocessors

This is the list of the currently supported preprocessors

* [LineArtPreprocessor](https://github.com/asomoza/image_gen_aux/blob/main/src/image_gen_aux/preprocesors/lineart/README.md)

### How to use

```python
from image_gen_aux import LineArtPreprocessor
from image_gen_aux.utils import load_image

input_image = load_image(
    "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/simple_upscale/hippowaffle.png"
)

lineart_preprocessor = LineArtPreprocessor.from_pretrained("OzzyGT/lineart").to("cuda")
image = lineart_preprocessor(input_image)
image.save("lineart.png")
```

To load a model you can use the HuggingFace Hub Repository ID, an url or a local path. You can also load `pickle` checkpoints but
its not recommended. If the checkpoint has custom imports (high risk!!!) you'll need to set the `weights_only` argument to `False`.

You can also specify a `filename` and a `subfolder` if needed.

Example:

```python
lineart_preprocessor = LineArtPreprocessor.from_pretrained("lllyasviel/Annotators", filename="sk_model.pth", weights_only=False).to("cuda")
```

### List of safetensors checkpoints

This is the current list of safetensor checkpoints you can use from the hub.

|Preprocessor|Repository|Author|
|---|---|---|
|LineArt|OzzyGT/lineart|[Caroline Chan](https://github.com/carolineec)|

If you own the model and want us to change the repository to your name/organization please open an issue.
