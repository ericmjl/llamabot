"""Create a FastAPI app to visualize and compare prompts and messages."""

from typing import Optional
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from pyprojroot import here

from llamabot.recorder import Base, upgrade_database, Prompt
from llamabot.web.routers import logs, prompt_versions
from llamabot.web.database import get_engine, init_sessionmaker, DbSession

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def create_app(db_path: Optional[Path] = None):
    """Create a FastAPI app to visualize and compare prompts and messages.

    :param db_path: The path to the database to use.
    """
    if db_path is None:
        db_path = here() / "message_log.db"

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

    @app.get("/")
    async def root(request: Request, db: DbSession):
        """The root page."""
        prompts = (
            db.query(Prompt.function_name)
            .distinct()
            .order_by(Prompt.function_name)
            .all()
        )
        prompts = [{"function_name": p[0]} for p in prompts]

        return templates.TemplateResponse(
            "index.html", {"request": request, "prompts": prompts}
        )

    return app
