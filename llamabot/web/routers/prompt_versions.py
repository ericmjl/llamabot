"""Router for prompt version-related endpoints."""

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy import desc, func
from pathlib import Path
from difflib import unified_diff
from loguru import logger
from sqlalchemy.orm import Session

from llamabot.recorder import Prompt
from llamabot.web.database import DbSession, get_db

router = APIRouter(tags=["prompts"])
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


async def list_prompts(db: Session):
    """List all prompts with their version counts.

    :param db: Database session
    :return: List of dictionaries containing function names and their version counts
    """
    prompts = (
        db.query(Prompt.function_name, func.count(Prompt.function_name).label("count"))
        .group_by(Prompt.function_name)
        .all()
    )
    return [{"function_name": p.function_name, "count": p.count} for p in prompts]


@router.get("/history", response_class=HTMLResponse)
async def get_prompt_history(
    request: Request,
    db: DbSession,
    function_name: str,
):
    """Get the history of prompts for a given function name."""
    logger.debug(f"Getting prompt history for function: {function_name}")

    if not function_name:
        return templates.TemplateResponse(
            "prompt_history.html",
            {
                "request": request,
                "function_name": "",
                "prompt_history": [],
            },
        )

    prompts = (
        db.query(Prompt)
        .filter(Prompt.function_name == function_name)
        .order_by(desc(Prompt.id))
        .all()
    )

    logger.debug(f"Found {len(prompts)} prompts for function {function_name}")

    if not prompts:
        return templates.TemplateResponse(
            "prompt_history.html",
            {
                "request": request,
                "function_name": function_name,
                "prompt_history": [],
            },
        )

    prompt_history = []
    for i, prompt in enumerate(prompts):
        diff = ""
        if i < len(prompts) - 1:
            diff = "\n".join(
                unified_diff(
                    prompts[i + 1].template.splitlines(),
                    prompt.template.splitlines(),
                    fromfile=f"Version {prompts[i + 1].hash[:8]}",
                    tofile=f"Version {prompt.hash[:8]}",
                    lineterm="",
                )
            )

        prompt_history.append(
            {
                "hash": prompt.hash,
                "template": prompt.template,
                "diff": diff,
            }
        )

    return templates.TemplateResponse(
        "prompt_history.html",
        {
            "request": request,
            "function_name": function_name,
            "prompt_history": prompt_history,
        },
    )


@router.get("/functions", response_class=HTMLResponse)
async def get_prompt_functions(request: Request, db: Session = Depends(get_db)):
    """Get all unique prompt function names with their version counts."""
    prompts = await list_prompts(db)
    logger.debug(f"Found {len(prompts)} functions with counts: {prompts}")

    return templates.TemplateResponse(
        "prompt_dropdown.html",
        {
            "request": request,
            "prompts": prompts,
        },
    )


@router.get("/{prompt_hash}", response_class=HTMLResponse)
async def get_prompt(prompt_hash: str, request: Request, db: DbSession):
    """Get prompt details by hash."""
    prompt = db.query(Prompt).filter(Prompt.hash == prompt_hash).first()
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")

    return templates.TemplateResponse(
        "prompt_modal.html",
        {
            "request": request,
            "prompt": prompt,
        },
    )


@router.get("/", response_class=HTMLResponse)
async def prompts_index(request: Request, db: DbSession):
    """
    Render the main prompt comparison page.

    :param request: The FastAPI request object.
    :param db: Database session dependency.
    :return: TemplateResponse containing the rendered prompt comparison page.
    """
    prompts = await list_prompts(db)
    return templates.TemplateResponse(
        "prompts/index.html",
        {"request": request, "prompts": prompts},
    )


# Add other prompt-related endpoints here...
