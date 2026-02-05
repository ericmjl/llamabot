# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "llamabot[all]",
#     "marimo>=0.17.0",
#     "pydantic",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## How to Generate Blog Banner Images with ImageBot

    Learn how to use ImageBot and StructuredBot together to automatically generate
    beautiful banner images for blog posts. This guide shows you how to create a
    workflow that generates DALL-E prompts from blog content and then creates
    custom banner images.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Prerequisites

    Before you begin, ensure you have:

    - **OpenAI API key**: Set up your OpenAI API key (ImageBot uses DALL-E)
    - **Python 3.10+** with llamabot installed
    - **A blog post or text content** to generate a banner for

    **Note**: ImageBot requires an OpenAI API key with access to DALL-E models.
    Make sure your API key is configured in your environment.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Goal

    By the end of this guide, you'll have built a workflow that:

    - Takes blog post text as input
    - Uses StructuredBot to generate a detailed DALL-E prompt
    - Uses ImageBot to generate a banner image (16:4 aspect ratio)
    - Creates beautiful, watercolor-style banner images for your blog posts
    """
    )
    return


@app.cell
def _():
    from pydantic import BaseModel, Field

    import llamabot as lmb
    from llamabot.bot.imagebot import ImageBot
    from llamabot.bot.structuredbot import StructuredBot
    from llamabot.prompt_manager import prompt

    return BaseModel, Field, ImageBot, StructuredBot, prompt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Step 1: Define the DALL-E Prompt Schema

    First, define a Pydantic model to structure the DALL-E image generation prompt.
    This ensures we get well-formatted prompts that work well with DALL-E.
    """
    )
    return


@app.cell
def _(BaseModel, Field):
    class DallEImagePrompt(BaseModel):
        """Structured prompt for DALL-E image generation."""

        content: str = Field(
            ...,
            description="A detailed, descriptive prompt for generating a banner image for the blog post.",
        )

    return (DallEImagePrompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Step 2: Create the Prompt Generator Bot

    Create a StructuredBot that generates detailed DALL-E prompts from blog post text.
    This bot will translate the key concepts and themes from your blog post into
    a visual description suitable for image generation.
    """
    )
    return


@app.cell
def _(DallEImagePrompt, StructuredBot, prompt):
    @prompt("system")
    def bannerbot_dalle_prompter_sysprompt() -> str:
        """You are a prompt designer for DALL-E image generation.

        Your role is to create highly detailed and imaginative prompts for DALL-E,
        designed to generate banner images for blog posts in a watercolor style,
        with a 16:4 aspect ratio.

        You will be given a chunk of text or a summary that comes from the blog post.
        Your task is to translate the key concepts, ideas, and themes from the text
        into an image prompt.

        **Guidelines for creating the prompt:**
        - Use vivid and descriptive language to specify the image's mood, colors,
          composition, and style.
        - Vary your approach significantly between prompts - avoid repetitive patterns,
          elements, or compositions that could make images look similar.
        - Explore diverse watercolor techniques: washes, wet-on-wet, dry brush,
          salt effects, splattering, or layered glazes.
        - Consider different artistic styles within watercolor: impressionistic,
          expressionistic, minimalist, detailed botanical, atmospheric, or abstract.
        - Vary the color palettes: warm vs cool tones, monochromatic vs complementary,
          muted vs vibrant, or seasonal color schemes.
        - Mix different compositional approaches: centered focal points, rule of thirds,
          diagonal compositions, or asymmetrical balance.
        - Incorporate varied symbolic elements: natural objects, architectural forms,
          organic shapes, geometric patterns, or conceptual representations.
        - Focus on maximizing the use of imagery and symbols to represent ideas,
          avoiding any inclusion of text or character symbols in the image.
        - If the text is vague or lacks detail, make thoughtful and creative assumptions
          to create a compelling visual representation.

        The prompt should be suitable for a variety of blog topics,
        evoking an emotional or intellectual connection to the content.
        Ensure the description specifies the watercolor art style,
        the wide 16:4 banner aspect ratio,
        and your chosen artistic approach.

        Do **NOT** include any text or character symbols in the image description.
        """

    dalle_prompt_bot = StructuredBot(
        system_prompt=bannerbot_dalle_prompter_sysprompt(),
        pydantic_model=DallEImagePrompt,
        model_name="gpt-4o",
    )
    return (dalle_prompt_bot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Step 3: Create the ImageBot

    Create an ImageBot configured for banner images. We'll use a 16:4 aspect ratio
    (1792x1024 pixels) which is perfect for blog banners.
    """
    )
    return


@app.cell
def _(ImageBot):
    bannerbot = ImageBot(size="1792x1024", quality="hd")
    return (bannerbot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Step 4: Generate a Banner Image

    Now let's put it all together! We'll:
    1. Generate a DALL-E prompt from blog post text
    2. Use that prompt to generate a banner image
    3. Display the result
    """
    )
    return


@app.cell
def _(dalle_prompt_bot):
    # Example blog post text
    blog_post_text = """
    In this blog post, we explore how to build AI agents that can reason about
    complex problems. We'll dive into graph-based agent architectures, tool
    orchestration, and multi-step planning. Learn how to create agents that
    can break down complex tasks into manageable steps and execute them
    systematically.
    """

    # Generate the DALL-E prompt
    dalle_prompt = dalle_prompt_bot(blog_post_text)
    dalle_prompt.content
    return blog_post_text, dalle_prompt


@app.cell
def _(bannerbot, dalle_prompt):
    # Generate the banner image
    banner_url = bannerbot(dalle_prompt.content, return_url=True)
    banner_url
    return (banner_url,)


@app.cell
def _(banner_url, mo):
    # Display the generated banner image
    mo.image(banner_url)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Step 5: Save the Image Locally

    You can also save the image to a file instead of just getting the URL.
    ImageBot will automatically generate a filename from the prompt if you don't
    specify a save path.
    """
    )
    return


@app.cell
def _(bannerbot, dalle_prompt):
    from pathlib import Path

    # Save the image to a specific path
    image_path = bannerbot(
        dalle_prompt.content,
        save_path=Path("blog_banner.jpg"),
    )
    image_path
    return (Path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Step 6: Create a Complete Workflow Function

    Let's create a reusable function that takes blog text and generates a banner image.
    This makes it easy to integrate into your blog publishing workflow.
    """
    )
    return


@app.cell
def _(Path, bannerbot, dalle_prompt_bot):
    def generate_blog_banner(
        blog_text: str, save_path: Path | None = None
    ) -> str | Path:
        """Generate a banner image for a blog post.

        :param blog_text: The text content of the blog post
        :param save_path: Optional path to save the image. If None, returns URL.
        :return: URL if save_path is None, otherwise the path to saved image
        """
        # Step 1: Generate DALL-E prompt from blog text
        dalle_prompt = dalle_prompt_bot(blog_text)

        # Step 2: Generate the banner image
        if save_path:
            image_path = bannerbot(dalle_prompt.content, save_path=save_path)
            return image_path
        else:
            banner_url = bannerbot(dalle_prompt.content, return_url=True)
            return banner_url

    return (generate_blog_banner,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Step 7: Test with Different Blog Posts

    Try generating banners for different types of blog posts to see how the
    prompt generator adapts to different topics and styles.
    """
    )
    return


@app.cell
def _(generate_blog_banner, mo):
    # Example 1: Technical blog post
    technical_post = """
    Learn how to optimize your Python code for performance. We'll cover
    profiling techniques, optimization strategies, and best practices for
    writing efficient Python code.
    """

    technical_banner_url = generate_blog_banner(technical_post)
    mo.md(f"**Technical Post Banner:**")
    mo.image(technical_banner_url)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Customization Options

    You can customize the banner generation in several ways:

    - **Change the aspect ratio**: Modify the `size` parameter in ImageBot
      (e.g., "1024x1024" for square, "1024x1792" for portrait)
    - **Adjust the style**: Modify the system prompt to use different art styles
      (e.g., digital art, photography, illustration)
    - **Change image quality**: Use `quality="standard"` for faster/cheaper generation
      or `quality="hd"` for higher quality (default)
    - **Modify the prompt generator**: Adjust the system prompt to emphasize
      different visual elements or styles
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Summary

    You've built a complete workflow for generating blog banner images that:

    - Uses StructuredBot to create detailed, structured DALL-E prompts
    - Uses ImageBot to generate high-quality banner images
    - Supports both URL-based and file-based image generation
    - Can be easily integrated into blog publishing workflows

    **Key Takeaways:**

    - Combine StructuredBot and ImageBot for AI-powered image generation
    - Use detailed system prompts to guide the prompt generation
    - Configure ImageBot with appropriate aspect ratios for your use case
    - Create reusable functions to integrate into your workflows

    **Next Steps:**

    - Integrate this into your blog publishing pipeline
    - Experiment with different art styles and prompt variations
    - Consider caching generated prompts for similar blog posts
    - Add error handling and retry logic for production use
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Interactive Style Customization (Madlib UI)

    Now let's create an interactive interface that lets you experiment with different
    artistic styles for the same blog post. Use the controls below to customize the
    watercolor technique, artistic style, color palette, composition, and symbolic
    elements, then generate a new banner image with your selected styles.
    """
    )
    return


@app.cell
def _():
    watercolor_techniques = [
        "washes",
        "wet-on-wet",
        "dry brush",
        "salt effects",
        "splattering",
        "layered glazes",
    ]

    artistic_styles = [
        "impressionistic",
        "expressionistic",
        "minimalist",
        "detailed botanical",
        "atmospheric",
        "abstract",
    ]

    color_palettes = [
        "warm tones",
        "cool tones",
        "monochromatic",
        "complementary",
        "muted",
        "vibrant",
        "seasonal color scheme",
    ]

    compositional_approaches = [
        "centered focal points",
        "rule of thirds",
        "diagonal compositions",
        "asymmetrical balance",
    ]

    symbolic_elements = [
        "natural objects",
        "architectural forms",
        "organic shapes",
        "geometric patterns",
        "conceptual representations",
    ]
    return (
        artistic_styles,
        color_palettes,
        compositional_approaches,
        symbolic_elements,
        watercolor_techniques,
    )


@app.cell
def _(
    artistic_styles,
    color_palettes,
    compositional_approaches,
    mo,
    symbolic_elements,
    watercolor_techniques,
):
    mo.md("### Customize Your Banner Style")

    technique_selector = mo.ui.dropdown(
        options=watercolor_techniques,
        value="wet-on-wet",
        label="Watercolor Technique",
    )

    style_selector = mo.ui.dropdown(
        options=artistic_styles,
        value="atmospheric",
        label="Artistic Style",
    )

    palette_selector = mo.ui.dropdown(
        options=color_palettes,
        value="complementary",
        label="Color Palette",
    )

    composition_selector = mo.ui.dropdown(
        options=compositional_approaches,
        value="rule of thirds",
        label="Compositional Approach",
    )

    symbol_selector = mo.ui.dropdown(
        options=symbolic_elements,
        value="conceptual representations",
        label="Symbolic Elements",
    )

    mo.vstack(
        [
            technique_selector,
            style_selector,
            palette_selector,
            composition_selector,
            symbol_selector,
        ]
    )
    return (
        composition_selector,
        palette_selector,
        style_selector,
        symbol_selector,
        technique_selector,
    )


@app.cell
def _(
    DallEImagePrompt,
    StructuredBot,
    composition_selector,
    palette_selector,
    prompt,
    style_selector,
    symbol_selector,
    technique_selector,
):
    @prompt("system")
    def styled_bannerbot_dalle_prompter_sysprompt(
        technique: str,
        style: str,
        palette: str,
        composition: str,
        symbols: str,
    ) -> str:
        """You are a prompt designer for DALL-E image generation.

        Your role is to create highly detailed and imaginative prompts for DALL-E,
        designed to generate banner images for blog posts in a watercolor style,
        with a 16:4 aspect ratio.

        You will be given a chunk of text or a summary that comes from the blog post.
        Your task is to translate the key concepts, ideas, and themes from the text
        into an image prompt.

        **Style Requirements:**
        - Watercolor technique: {{ technique }}
        - Artistic style: {{ style }}
        - Color palette: {{ palette }}
        - Compositional approach: {{ composition }}
        - Symbolic elements: {{ symbols }}

        **Guidelines for creating the prompt:**
        - Use vivid and descriptive language to specify the image's mood, colors,
          composition, and style.
        - Incorporate the specified watercolor technique ({{ technique }}) prominently
          in your description.
        - Apply the {{ style }} artistic style throughout the image.
        - Use a {{ palette }} color palette as the foundation for the image.
        - Structure the composition using {{ composition }}.
        - Feature {{ symbols }} as the primary symbolic elements.
        - Focus on maximizing the use of imagery and symbols to represent ideas,
          avoiding any inclusion of text or character symbols in the image.
        - If the text is vague or lacks detail, make thoughtful and creative assumptions
          to create a compelling visual representation.

        The prompt should be suitable for a variety of blog topics,
        evoking an emotional or intellectual connection to the content.
        Ensure the description specifies the watercolor art style,
        the wide 16:4 banner aspect ratio,
        and incorporates all the specified style elements.

        Do **NOT** include any text or character symbols in the image description.
        """

    styled_dalle_prompt_bot = StructuredBot(
        system_prompt=styled_bannerbot_dalle_prompter_sysprompt(
            technique=technique_selector.value,
            style=style_selector.value,
            palette=palette_selector.value,
            composition=composition_selector.value,
            symbols=symbol_selector.value,
        ),
        pydantic_model=DallEImagePrompt,
        model_name="gpt-4o",
    )
    return (styled_dalle_prompt_bot,)


@app.cell
def _(blog_post_text, styled_dalle_prompt_bot):
    # Generate the DALL-E prompt with custom styles
    styled_dalle_prompt = styled_dalle_prompt_bot(blog_post_text)
    styled_dalle_prompt.content
    return (styled_dalle_prompt,)


@app.cell
def _(bannerbot, styled_dalle_prompt):
    # Generate the banner image with custom styles
    styled_banner_url = bannerbot(styled_dalle_prompt.content, return_url=True)
    styled_banner_url
    return (styled_banner_url,)


@app.cell
def _(mo, styled_banner_url):
    # Display the generated banner image with custom styles
    mo.image(styled_banner_url)
    return


if __name__ == "__main__":
    app.run()
