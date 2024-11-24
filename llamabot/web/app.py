"""Create a FastAPI app to visualize and compare prompts and messages."""

from typing import Optional
from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from llamabot.recorder import Base, upgrade_database
from llamabot.web.routers import logs, prompt_versions, experiments
from llamabot.web.database import get_engine, init_sessionmaker, get_db
from llamabot.prompt_manager import find_or_set_db_path

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def create_app(db_path: Optional[Path] = None):
    """Create a FastAPI app to visualize and compare prompts and messages.

    :param db_path: The path to the database to use.
    """
    db_path = find_or_set_db_path(db_path)

    app = FastAPI()

    # Database setup
    engine = get_engine(db_path)
    init_sessionmaker(engine)  # Initialize the global SessionLocal

    # Ensure the database is initialized and upgraded
    Base.metadata.create_all(engine)
    upgrade_database(engine)

    # Static files
    app.mount(
        "/static",
        StaticFiles(directory=Path(__file__).parent / "static"),
        name="static",
    )

    # Include routers with proper prefixes
    app.include_router(logs.router, prefix="/logs")
    app.include_router(prompt_versions.router, prefix="/prompts")
    app.include_router(experiments.router)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request, db: Session = Depends(get_db)):
        """Render the index page with prompt and experiment data.

        :param request: The FastAPI request object
        :param db: Database session dependency
        :return: TemplateResponse containing the rendered index page
        """
        prompts = await prompt_versions.list_prompts(db)
        expts = await experiments.list_experiments(db)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "prompts": prompts, "experiments": expts},
        )

    return app
