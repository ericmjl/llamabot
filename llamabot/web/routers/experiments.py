"""Router for experiment-related endpoints.

This module provides FastAPI endpoints for viewing and managing ML experiment runs.
It includes functionality for:
- Listing experiments
- Viewing experiment details including metrics, message logs, and prompts
- Comparing runs within experiments
"""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from llamabot.web.database import get_db
from fastapi.templating import Jinja2Templates
from pathlib import Path
from llamabot.recorder import Prompt, Runs

router = APIRouter(prefix="/experiments")
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


@router.get("/details", response_class=HTMLResponse)
async def get_experiment_details(
    experiment_name: str,
    db: Session = Depends(get_db),
):
    """Get details for a specific experiment."""
    # Query all runs for this experiment
    runs = (
        db.query(Runs)
        .filter(Runs.experiment_name == experiment_name)
        .order_by(Runs.timestamp.desc())
        .all()
    )

    # Process runs to extract metrics and other data
    processed_runs = []
    metrics_set = set()

    for run in runs:
        run_data = run.run_data_dict
        metrics = run_data.get("metrics", {})
        metrics_set.update(metrics.keys())

        # Get prompt details for each prompt hash
        prompts_data = []
        for prompt_info in run_data.get("prompts", []):
            prompt_hash = prompt_info.get("hash")
            prompt = db.query(Prompt).filter(Prompt.hash == prompt_hash).first()
            prompts_data.append(
                {
                    "hash": prompt_hash,
                    "function_name": prompt.function_name if prompt else "Unknown",
                }
            )

        processed_runs.append(
            {
                "id": run.id,
                "metrics": metrics,
                "message_log_ids": run_data.get("message_log_ids", []),
                "prompts": prompts_data,
                "timestamp": run.timestamp,
            }
        )

    return templates.TemplateResponse(
        "experiment_details.html",
        {
            "request": {},
            "experiment_name": experiment_name,
            "runs": processed_runs,
            "metrics_columns": sorted(list(metrics_set)),
        },
    )


@router.get("/list")
async def list_experiments(db: Session = Depends(get_db)):
    """Get list of all experiments with their run counts."""
    experiments = (
        db.query(
            Runs.experiment_name.label("name"),
            func.count("*").label("count"),
        )
        .filter(Runs.experiment_name.isnot(None))
        .group_by(Runs.experiment_name)
        .all()
    )

    return [{"name": exp.name, "count": exp.count} for exp in experiments]


@router.get("/", response_class=HTMLResponse)
async def experiments_index(request: Request, db: Session = Depends(get_db)):
    """
    Render the main experiment view page with a list of experiments.

    :param request: The FastAPI request object.
    :param db: Database session dependency.
    :return: TemplateResponse containing the rendered experiment view page.
    """
    experiments = (
        db.query(
            Runs.experiment_name.label("name"),
            func.count("*").label("count"),
        )
        .filter(Runs.experiment_name.isnot(None))
        .group_by(Runs.experiment_name)
        .all()
    )
    experiments_list = [{"name": exp.name, "count": exp.count} for exp in experiments]
    return templates.TemplateResponse(
        "experiments/index.html",
        {"request": request, "experiments": experiments_list},
    )
