# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "marimo",
#     "llamabot",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../../", editable = true }
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def imports():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def title(mo):
    mo.md(r"""
    # Image Generation with Ollama

    This notebook explores image generation using **Ollama's local models** with two approaches:

    - the raw HTTP API via `httpx`,
    - `ImageBot` from `llamabot`.

    By the end you will know how to:

    - call Ollama's `/api/generate` endpoint directly,
    - use `ImageBot` with a `style` parameter for consistent visual direction,
    - compare **watercolor** and **photorealistic** styles on the same absurd subject.
    """)
    return


@app.cell(hide_code=True)
def prerequisites_md(mo):
    mo.md(r"""
    ## Prerequisites

    Ensure the following before running the notebook:

    - **Ollama** is running locally (`ollama serve` or a system service).
    - The image models are pulled:
      ```bash
      ollama pull x/flux2-klein:latest
      ollama pull x/z-image-turbo:latest
      ```
    """)
    return


@app.cell
def python_imports():
    import base64
    import json

    import httpx
    from pathlib import Path

    from llamabot.bot.imagebot import ImageBot, ImageReference

    return ImageBot, ImageReference, Path, httpx


@app.cell
def model_config():
    OLLAMA_API_BASE = "http://localhost:11434"
    FLUX2_MODEL = "x/flux2-klein:latest"
    TURBO_MODEL = "x/z-image-turbo:latest"
    IMAGE_SIZE = "1024x1024"

    WATERCOLOR_STYLE = (
        "vivid watercolor illustration, bold saturated colors, "
        "soft wet-on-wet edges, paint splatters, white paper showing through"
    )

    PHOTOREALISTIC_STYLE = (
        "photorealistic, high-resolution DSLR photo, natural lighting, "
        "sharp focus, detailed textures, National Geographic quality"
    )

    SUBJECT = (
        "a capybara sitting on top of an alligator, "
        "both eating berries together while watching TV"
    )
    return (
        FLUX2_MODEL,
        IMAGE_SIZE,
        OLLAMA_API_BASE,
        PHOTOREALISTIC_STYLE,
        SUBJECT,
        TURBO_MODEL,
        WATERCOLOR_STYLE,
    )


@app.cell(hide_code=True)
def concept_md(mo):
    mo.md(r"""
    ## Style as a system prompt

    `ImageBot` accepts a `style` parameter — a visual-language string
    prepended to every prompt automatically, like a system prompt for images.

    - **`style`** — visual language, lighting, mood, medium (set once on the bot).
    - **`prompt`** — the specific subject of each scene (set per call).

    We will apply two very different styles to the same delightfully absurd subject
    and see how each model interprets it.
    """)
    return


@app.cell(hide_code=True)
def raw_api_md(mo):
    mo.md(r"""
    ## Raw HTTP API with httpx — watercolor

    We call Ollama's `/api/generate` directly, manually combining the watercolor style and subject.
    """)
    return


@app.cell
def raw_api_generate(
    FLUX2_MODEL,
    ImageReference,
    OLLAMA_API_BASE,
    Path,
    SUBJECT,
    WATERCOLOR_STYLE,
    httpx,
):
    raw_prompt = f"{WATERCOLOR_STYLE}, {SUBJECT}"

    raw_payload = {
        "model": FLUX2_MODEL,
        "prompt": raw_prompt,
        "stream": False,
        "width": 1024,
        "height": 1024,
    }

    raw_response = httpx.post(
        f"{OLLAMA_API_BASE}/api/generate",
        json=raw_payload,
        timeout=120,
    )
    raw_response.raise_for_status()
    raw_json = raw_response.json()

    raw_b64 = raw_json.get("image")
    if not raw_b64:
        raise ValueError(
            "No image field in Ollama response. "
            f"Response keys: {sorted(raw_json.keys())}"
        )

    raw_image_ref = ImageReference(f"data:image/png;base64,{raw_b64}")
    raw_image_ref.save(Path("./generated_raw_watercolor.png"))
    raw_image_ref
    return


@app.cell(hide_code=True)
def imagebot_watercolor_md(mo):
    mo.md(r"""
    ## ImageBot — watercolor style

    Now we let `ImageBot` handle style + subject. The `style` is set once on the bot;
    each call only needs the subject.
    """)
    return


@app.cell
def imagebot_watercolor_flux2(
    FLUX2_MODEL,
    IMAGE_SIZE,
    ImageBot,
    OLLAMA_API_BASE,
    Path,
    SUBJECT,
    WATERCOLOR_STYLE,
):
    watercolor_bot = ImageBot(
        model=f"ollama/{FLUX2_MODEL}",
        size=IMAGE_SIZE,
        api_base=OLLAMA_API_BASE,
        style=WATERCOLOR_STYLE,
    )

    watercolor_flux2_ref = watercolor_bot(SUBJECT)
    watercolor_flux2_ref.save(Path("./generated_watercolor_flux2.png"))
    watercolor_flux2_ref
    return


@app.cell(hide_code=True)
def photorealistic_md(mo):
    mo.md(r"""
    ## Photorealistic style

    Same subject, completely different visual language. We swap in the photorealistic style
    and try it with both models.
    """)
    return


@app.cell
def imagebot_photorealistic_turbo(
    IMAGE_SIZE,
    ImageBot,
    OLLAMA_API_BASE,
    PHOTOREALISTIC_STYLE,
    Path,
    SUBJECT,
    TURBO_MODEL,
):
    photoreal_bot = ImageBot(
        model=f"ollama/{TURBO_MODEL}",
        size=IMAGE_SIZE,
        api_base=OLLAMA_API_BASE,
        style=PHOTOREALISTIC_STYLE,
    )

    photoreal_turbo_ref = photoreal_bot(SUBJECT)
    photoreal_turbo_ref.save(Path("./generated_photorealistic_turbo.png"))
    photoreal_turbo_ref
    return


@app.cell
def imagebot_photorealistic_flux2(
    FLUX2_MODEL,
    IMAGE_SIZE,
    ImageBot,
    OLLAMA_API_BASE,
    PHOTOREALISTIC_STYLE,
    Path,
    SUBJECT,
):
    photoreal_flux2_ref = ImageBot(
        model=f"ollama/{FLUX2_MODEL}",
        size=IMAGE_SIZE,
        api_base=OLLAMA_API_BASE,
        style=PHOTOREALISTIC_STYLE,
    )(SUBJECT)
    photoreal_flux2_ref.save(Path("./generated_photorealistic_flux2.png"))
    photoreal_flux2_ref
    return


@app.cell(hide_code=True)
def next_steps_md(mo):
    mo.md(r"""
    ## Next steps

    - Define your own styles (anime, oil painting, pixel art, pencil sketch) and re-run.
    - Swap in different subjects while keeping a style fixed.
    - Compare the two models on the same style + subject to evaluate quality and latency.
    """)
    return


if __name__ == "__main__":
    app.run()
