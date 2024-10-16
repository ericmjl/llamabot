"""Create a FastAPI app to visualize and compare prompts and messages."""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pathlib import Path
import json
from pyprojroot import here
import logging

from llamabot.recorder import MessageLog, Base, upgrade_database, Prompt

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="llamabot/web/templates")


def create_app(db_path: Path = here() / "message_log.db"):
    """Create a FastAPI app to visualize and compare prompts and messages.

    :param db_path: The path to the database to use.
    """
    app = FastAPI()

    engine = create_engine(f"sqlite:///{db_path}")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Ensure the database is initialized and upgraded
    Base.metadata.create_all(engine)
    upgrade_database(engine)

    app.mount("/static", StaticFiles(directory="llamabot/web/static"), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        """The root page.

        :param request: The FastAPI request object.
        :return: The root page.
        """
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/logs")
    async def get_logs():
        """Get all logs."""
        db = SessionLocal()
        try:
            logs = db.query(MessageLog).order_by(MessageLog.timestamp.desc()).all()
            logger.debug(f"Found {len(logs)} logs")

            log_data = []
            for log in logs:
                message_log = json.loads(log.message_log)
                prompt_hashes = set()
                for message in message_log:
                    if "prompt_hash" in message:
                        prompt_hashes.add(message["prompt_hash"])

                logger.debug(f"Log {log.id} has prompt hashes: {prompt_hashes}")

                prompts = db.query(Prompt).filter(Prompt.hash.in_(prompt_hashes)).all()
                logger.debug(f"Found prompts: {[p.function_name for p in prompts]}")

                prompt_names = [
                    f"- {prompt.function_name} ({prompt.hash[:6]})"
                    for prompt in prompts
                ]
                formatted_prompt_names = (
                    "\n".join(prompt_names) if prompt_names else "No prompts used"
                )
                logger.debug(
                    f"Prompt names for log {log.id}:\n{formatted_prompt_names}"
                )

                log_data.append(
                    {
                        "id": log.id,
                        "object_name": log.object_name,
                        "timestamp": log.timestamp,
                        "model_name": log.model_name,
                        "temperature": log.temperature,
                        "prompt_names": formatted_prompt_names,
                        "full_content": log.message_log,
                    }
                )

            return templates.TemplateResponse(
                "log_table.html",
                {
                    "request": {},
                    "logs": log_data,
                },
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            db.close()

    @app.get("/log/{log_id}")
    async def get_log(log_id: int):
        """Get a single log by ID."""
        db = SessionLocal()
        try:
            log = db.query(MessageLog).filter(MessageLog.id == log_id).first()
            if log is None:
                raise HTTPException(status_code=404, detail="Log not found")
            message_log = json.loads(log.message_log)

            # Fetch prompt names and templates for each message with a prompt_hash
            for message in message_log:
                if message.get("prompt_hash"):
                    prompt = (
                        db.query(Prompt)
                        .filter(Prompt.hash == message["prompt_hash"])
                        .first()
                    )
                    if prompt:
                        message["prompt_name"] = prompt.function_name
                        message["prompt_template"] = prompt.template

            return {
                "id": log.id,
                "object_name": log.object_name,
                "timestamp": log.timestamp,
                "message_log": message_log,
                "model_name": log.model_name,
                "temperature": log.temperature,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            db.close()

    return app
