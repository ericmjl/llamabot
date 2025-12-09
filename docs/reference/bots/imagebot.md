# ImageBot API Reference

ImageBot is a specialized bot for generating images using DALL-E.

## Class Definition

```python
class ImageBot:
    """ImageBot class for generating images.

    :param model: The model to use. Defaults to "dall-e-3".
    :param size: The size of the image to generate. Defaults to "1024x1024".
    :param quality: The quality of the image to generate. Defaults to "standard".
    :param n: The number of images to generate. Defaults to 1.
    """
```

## Constructor

```python
def __init__(
    self,
    model: str = "dall-e-3",
    size: str = "1024x1024",
    quality: str = "hd",
    n: int = 1,
)
```

### Constructor Parameters

- **model** (`str`, default: `"dall-e-3"`): The DALL-E model to use.
  Currently supports `"dall-e-3"` and `"dall-e-2"`.

- **size** (`str`, default: `"1024x1024"`): The size of the image to generate.
  For DALL-E 3, valid sizes are `"1024x1024"`, `"1792x1024"`, and
  `"1024x1792"`. For DALL-E 2, valid sizes are `"256x256"`, `"512x512"`, and
  `"1024x1024"`.

- **quality** (`str`, default: `"hd"`): The quality of the image.
  For DALL-E 3, valid values are `"standard"` and `"hd"`.
  For DALL-E 2, this parameter is not used.

- **n** (`int`, default: `1`): The number of images to generate.
  For DALL-E 3, this must be 1. For DALL-E 2, can be 1-10.

## Methods

### `__call__`

```python
def __call__(
    self,
    prompt: str,
    return_url: bool = False,
    save_path: Optional[Path] = None,
) -> Union[str, Path]
```

Generate an image from a text prompt.

#### Parameters

- **prompt** (`str`): The text prompt describing the image to generate.

- **return_url** (`bool`, default: `False`): Whether to return the URL of the
  generated image. If `True`, overrides `save_path` parameter.
  Useful for Jupyter notebooks.

- **save_path** (`Optional[Path]`, default: `None`): The path to save the
  generated image to. If `None`, a filename will be generated from the prompt.

#### Returns

- **Union[str, Path]**:
  - If `return_url=True`: Returns the image URL as a string
    (useful for Jupyter notebooks)
  - Otherwise: Returns a `Path` object pointing to the saved image file

#### Raises

- **Exception**: If no image URL is found in the response.

#### Example

```python
import llamabot as lmb

bot = lmb.ImageBot()

# In a Jupyter notebook (returns URL)
url = bot("A painting of a sunset over mountains", return_url=True)

# In a Python script (saves to file)
image_path = bot("A painting of a sunset over mountains")
print(image_path)  # Path object
```

## Attributes

- **model** (`str`): The DALL-E model being used
- **size** (`str`): The image size
- **quality** (`str`): The image quality setting
- **n** (`int`): The number of images to generate

## Usage Examples

### Basic Image Generation

```python
import llamabot as lmb

bot = lmb.ImageBot()
image_path = bot("A futuristic cityscape at night")
```

### Custom Size and Quality

```python
import llamabot as lmb

bot = lmb.ImageBot(
    model="dall-e-3",
    size="1792x1024",
    quality="hd"
)

image_path = bot("A detailed landscape painting")
```

### In Jupyter Notebooks

```python
import llamabot as lmb

bot = lmb.ImageBot()

# Returns URL for display in notebook
url = bot("A beautiful sunset", return_url=True)
# Image automatically displays in notebook
```

### Save to Specific Path

```python
import llamabot as lmb
from pathlib import Path

bot = lmb.ImageBot()

image_path = bot(
    "A painting of a dog",
    save_path=Path("output/dog_painting.png")
)
```

### Using DALL-E 2

```python
import llamabot as lmb

bot = lmb.ImageBot(
    model="dall-e-2",
    size="512x512",
    n=3  # Can generate multiple images with DALL-E 2
)

image_paths = [bot("A cat wearing sunglasses") for _ in range(3)]
```

## Requirements

- OpenAI API key must be set in environment variable `OPENAI_API_KEY`
- For DALL-E 3, requires OpenAI API access
- For DALL-E 2, requires OpenAI API access

## Best Practices

1. **Be descriptive**: More detailed prompts produce better results
2. **Specify style**: Include artistic style, mood, or composition details
3. **Use DALL-E 3 for quality**: DALL-E 3 produces higher quality images
4. **Save important images**: Always save generated images if you need them later

## Related Classes

- **SimpleBot**: General-purpose chatbot (does not generate images)

## See Also

- [Which Bot Should I Use?](../getting-started/which-bot.md)
- [OpenAI DALL-E Documentation](https://platform.openai.com/docs/guides/images)
