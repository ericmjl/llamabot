"""Router for experiment-related endpoints.

This module provides FastAPI endpoints for viewing and managing ML experiment runs.
It includes functionality for:
- Listing experiments
- Viewing experiment details including metrics, message logs, and prompts
- Comparing runs within experiments
"""

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, Table, MetaData, Column, Integer, String, JSON
import json

from llamabot.web.database import get_db
from fastapi.templating import Jinja2Templates
from pathlib import Path
from llamabot.recorder import Prompt

router = APIRouter(prefix="/experiments")
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

# Define the runs table
metadata = MetaData()
Runs = Table(
    "runs",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("experiment_name", String),
    Column("timestamp", String),
    Column("run_metadata", JSON),
    Column("run_data", JSON),
)


@router.get("/details", response_class=HTMLResponse)
async def get_experiment_details(
    experiment_name: str,
    db: Session = Depends(get_db),
):
    """Get details for a specific experiment."""
    # Query all runs for this experiment
    runs = (
        db.query(Runs)
        .filter(Runs.c.experiment_name == experiment_name)
        .order_by(Runs.c.timestamp.desc())
        .all()
    )

    # Process runs to extract metrics and other data
    processed_runs = []
    metrics_set = set()

    for run in runs:
        run_data = (
            run.run_data
            if isinstance(run.run_data, dict)
            else json.loads(run.run_data or "{}")
        )
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
            Runs.c.experiment_name.label("name"),
            func.count("*").label("count"),
        )
        .filter(Runs.c.experiment_name.isnot(None))
        .group_by(Runs.c.experiment_name)
        .all()
    )

    return [{"name": exp.name, "count": exp.count} for exp in experiments]
