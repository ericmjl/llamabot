# ImageBot API Reference

ImageBot is a specialized bot for generating images through LiteLLM's image
generation API. It can use any image generation provider supported by LiteLLM,
including OpenAI, Azure OpenAI, Vertex AI, Bedrock, OpenRouter, and Recraft.

## Class Definition

```python
class ImageBot:
    """ImageBot class for generating images.

    :param model: The model to use. Defaults to "dall-e-3".
    :param size: The size of the image to generate. Defaults to "1024x1024".
    :param quality: The quality of the image to generate. Defaults to "hd".
    :param n: The number of images to generate. Defaults to 1.
    :param response_format: The LiteLLM image response format to request.
        Defaults to "url".
    :param api_key: Optional API key to pass through to LiteLLM.
    :param api_base: Optional API base URL to pass through to LiteLLM.
    :param api_version: Optional API version to pass through to LiteLLM.
    :param timeout: Maximum time, in seconds, to wait for image generation.
    :param image_generation_kwargs: Additional provider-specific keyword arguments
        to pass through to ``litellm.image_generation``.
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
    response_format: str = "url",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    timeout: int = 600,
    **image_generation_kwargs,
)
```

### Constructor Parameters

- **model** (`str`, default: `"dall-e-3"`): The LiteLLM image generation model
  to use. Examples include `"dall-e-3"`, `"gpt-image-1"`,
  `"openrouter/google/gemini-2.5-flash-image"`, and provider-prefixed models
  such as `"azure/<deployment-name>"`.

- **size** (`str`, default: `"1024x1024"`): The size of the image to generate.
  For DALL-E 3, valid sizes are `"1024x1024"`, `"1792x1024"`, and
  `"1024x1792"`. For DALL-E 2, valid sizes are `"256x256"`, `"512x512"`, and
  `"1024x1024"`.

- **quality** (`str`, default: `"hd"`): The quality of the image.
  For DALL-E 3, valid values are `"standard"` and `"hd"`.
  For DALL-E 2, this parameter is not used.

- **n** (`int`, default: `1`): The number of images to generate.
  For DALL-E 3, this must be 1. For DALL-E 2, can be 1-10.

- **response_format** (`str`, default: `"url"`): The response format to request
  from LiteLLM. Use `"url"` for URL-backed images or `"b64_json"` for base64
  image data when supported by the provider.

- **api_key** (`Optional[str]`, default: `None`): API key to pass to LiteLLM.
  If omitted, LiteLLM uses the provider's environment variable conventions.

- **api_base** (`Optional[str]`, default: `None`): Custom API base URL for
  providers such as Azure OpenAI or OpenAI-compatible endpoints.

- **api_version** (`Optional[str]`, default: `None`): Provider API version,
  commonly used for Azure OpenAI deployments.

- **timeout** (`int`, default: `600`): Maximum generation time in seconds.

- **image_generation_kwargs**: Additional provider-specific keyword arguments
  passed directly to `litellm.image_generation`.

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

- **return_url** (`bool`, default: `False`): Whether to return a displayable
  image reference instead of saving the file. If `True`, overrides `save_path`.
  URL responses return the image URL; base64 responses return a data URI.

- **save_path** (`Optional[Path]`, default: `None`): The path to save the
  generated image to. If `None`, a filename will be generated from the prompt.

#### Returns

- **Union[str, Path]**:
  - If `return_url=True`: Returns the image URL or base64 data URI as a string
    (useful for notebooks)
  - Otherwise: Returns a `Path` object pointing to the saved image file

#### Raises

- **ValueError**: If no generated image data is found in the response.

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

### Provider-specific options

Any keyword arguments not listed in the constructor are passed through to
`litellm.image_generation`, which makes provider options available without
changing ImageBot's API.

```python
import llamabot as lmb

bot = lmb.ImageBot(
    model="openrouter/google/gemini-2.5-flash-image",
    quality="high",
    api_key="...",
    user="user-123",
)

image_path = bot("A watercolor red panda reading a book")
```

## Attributes

- **model** (`str`): The LiteLLM image generation model being used
- **size** (`str`): The image size
- **quality** (`str`): The image quality setting
- **n** (`int`): The number of images to generate
- **response_format** (`str`): The requested LiteLLM image response format
- **api_key** (`Optional[str]`): The optional API key passed to LiteLLM
- **api_base** (`Optional[str]`): The optional API base passed to LiteLLM
- **api_version** (`Optional[str]`): The optional API version passed to LiteLLM
- **timeout** (`int`): The image generation timeout
- **image_generation_kwargs** (`dict`): Additional LiteLLM kwargs

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

- Set the API key expected by the selected LiteLLM provider, or pass `api_key`
  to `ImageBot`.
- Provider-prefixed models may require additional LiteLLM parameters such as
  `api_base`, `api_version`, project IDs, or region settings.

## Best Practices

1. **Be descriptive**: More detailed prompts produce better results
2. **Specify style**: Include artistic style, mood, or composition details
3. **Pick provider settings deliberately**: Match `size`, `quality`, and
   `response_format` to the selected LiteLLM provider.
4. **Save important images**: Always save generated images if you need them later

## Related Classes

- **SimpleBot**: General-purpose chatbot (does not generate images)

## See Also

- [Which Bot Should I Use?](../../getting-started/which-bot.md)
- [LiteLLM Image Generation](https://docs.litellm.ai/docs/image_generation)
