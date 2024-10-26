"""Create a FastAPI app to visualize and compare prompts and messages."""

from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, desc, func, text, or_
from sqlalchemy.orm import sessionmaker
from pathlib import Path
import json
from pyprojroot import here
import logging
from difflib import unified_diff
import yaml
from tempfile import NamedTemporaryFile

from llamabot.recorder import MessageLog, Base, upgrade_database, Prompt

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def create_app(db_path: Optional[Path] = None):
    """Create a FastAPI app to visualize and compare prompts and messages.

    :param db_path: The path to the database to use.
    """
    if db_path is None:
        db_path = here() / "message_log.db"

    app = FastAPI()

    engine = create_engine(f"sqlite:///{db_path}")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Ensure the database is initialized and upgraded
    Base.metadata.create_all(engine)
    upgrade_database(engine)

    app.mount(
        "/static",
        StaticFiles(directory=Path(__file__).parent / "static"),
        name="static",
    )

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        """The root page.

        :param request: The FastAPI request object.
        :return: The root page.
        """
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/logs")
    async def get_logs(function_name: Optional[str] = None):
        """Get all logs, optionally filtered by function name."""
        logger.debug(f"Received request for logs with function_name: {function_name}")
        db = SessionLocal()
        try:
            query = db.query(MessageLog).order_by(MessageLog.timestamp.desc())

            if function_name:
                logger.debug(f"Filtering logs for function name: {function_name}")
                prompt_hashes = (
                    db.query(Prompt.hash)
                    .filter(Prompt.function_name == function_name)
                    .all()
                )
                prompt_hashes = [hash[0] for hash in prompt_hashes]

                logger.debug(
                    f"Found prompt hashes for function {function_name}: {prompt_hashes}"
                )

                if prompt_hashes:
                    conditions = [
                        MessageLog.message_log.like(f"%{hash}%")
                        for hash in prompt_hashes
                    ]
                    query = query.filter(or_(*conditions))
                else:
                    logger.warning(
                        f"No prompts found for function name: {function_name}"
                    )
                    return templates.TemplateResponse(
                        "log_table.html",
                        {
                            "request": {},
                            "logs": [],
                        },
                    )

            logs = query.all()
            logger.debug(f"Found {len(logs)} logs")

            log_data = []
            for log in logs:
                try:
                    message_log_content = (
                        json.loads(log.message_log) if log.message_log else []
                    )
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse message_log for log {log.id}")
                    message_log_content = []

                prompt_hashes = set()
                for message in message_log_content:
                    if isinstance(message, dict) and "prompt_hash" in message:
                        prompt_hash = message["prompt_hash"]
                        if prompt_hash is not None:
                            prompt_hashes.add(prompt_hash)

                prompts = db.query(Prompt).filter(Prompt.hash.in_(prompt_hashes)).all()
                prompt_names = [
                    f"- {prompt.function_name} ({prompt.hash[:6]})"
                    for prompt in prompts
                ]
                formatted_prompt_names = (
                    "\n".join(prompt_names) if prompt_names else "No prompts used"
                )

                log_data.append(
                    {
                        "id": log.id,
                        "object_name": log.object_name,
                        "timestamp": log.timestamp,
                        "model_name": log.model_name,
                        "temperature": log.temperature,
                        "prompt_names": formatted_prompt_names,
                        "prompt_hashes": ",".join(filter(None, prompt_hashes)),
                        "full_content": log.message_log,
                    }
                )

            logger.debug(f"Returning {len(log_data)} logs")
            return templates.TemplateResponse(
                "log_table.html",
                {
                    "request": {},
                    "logs": log_data,
                },
            )
        except Exception as e:
            logger.error(f"Error in get_logs: {str(e)}")
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
            message_log = json.loads(log.message_log if log.message_log else "[]")

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
            logger.error(f"Error in get_log: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            db.close()

    @app.get("/prompt_history/{function_name}")
    async def get_prompt_history(function_name: str):
        """Get the history of prompts for a given function name."""
        db = SessionLocal()
        try:
            prompts = (
                db.query(Prompt)
                .filter(Prompt.function_name == function_name)
                .order_by(desc(Prompt.id))
                .all()
            )

            if not prompts:
                raise HTTPException(
                    status_code=404, detail="No prompts found for this function name"
                )

            prompt_history = []
            for i, prompt in enumerate(prompts):
                diff = ""
                if i < len(prompts) - 1:
                    diff = "\n".join(
                        unified_diff(
                            prompts[i + 1].template.splitlines(),
                            prompt.template.splitlines(),
                            fromfile=f"Version {prompts[i+1].hash[:8]}",
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
                    "request": {},
                    "function_name": function_name,
                    "prompt_history": prompt_history,
                },
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            db.close()

    @app.get("/prompt_functions")
    async def get_prompt_functions():
        """Get all unique prompt function names with their version counts."""
        db = SessionLocal()
        try:
            function_counts = (
                db.query(
                    Prompt.function_name, func.count(Prompt.id).label("version_count")
                )
                .group_by(Prompt.function_name)
                .all()
            )
            return {
                "function_names": [
                    {"name": name, "count": count} for name, count in function_counts
                ]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            db.close()

    @app.get("/prompts")
    async def get_prompts():
        """Get all unique prompt function names."""
        db = SessionLocal()
        try:
            prompts = db.query(Prompt.function_name).distinct().all()
            return templates.TemplateResponse(
                "prompt_dropdown.html",
                {
                    "request": {},
                    "prompts": prompts,
                },
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            db.close()

    @app.get("/debug_logs")
    async def debug_logs():
        """Debug endpoint to view raw message logs."""
        db = SessionLocal()
        try:
            logs = db.query(MessageLog).order_by(text("timestamp DESC")).limit(10).all()
            return {
                "log_count": len(logs),
                "sample_logs": [
                    {
                        "id": log.id,
                        "object_name": log.object_name,
                        "timestamp": log.timestamp,
                        "message_log": log.message_log,
                    }
                    for log in logs
                ],
            }
        finally:
            db.close()

    @app.get("/export_logs")
    async def export_logs(function_name: Optional[str] = None):
        """Export logs as YAML file."""
        db = SessionLocal()
        try:
            query = db.query(MessageLog).order_by(MessageLog.timestamp.desc())

            if function_name:
                prompt_hashes = (
                    db.query(Prompt.hash)
                    .filter(Prompt.function_name == function_name)
                    .all()
                )
                prompt_hashes = [hash[0] for hash in prompt_hashes]
                if prompt_hashes:
                    conditions = [
                        MessageLog.message_log.like(f"%{hash}%")
                        for hash in prompt_hashes
                    ]
                    query = query.filter(or_(*conditions))

            logs = query.all()

            log_data = []
            for log in logs:
                log_entry = {
                    "id": log.id,
                    "object_name": log.object_name,
                    "timestamp": log.timestamp,
                    "model_name": log.model_name,
                    "temperature": log.temperature,
                    "message_log": (
                        json.loads(log.message_log) if log.message_log else []
                    ),
                }
                log_data.append(log_entry)

            with NamedTemporaryFile(
                mode="w", delete=False, suffix=".yaml"
            ) as temp_file:
                yaml.dump(log_data, temp_file, default_flow_style=False)
                temp_file_path = temp_file.name

            return FileResponse(
                temp_file_path,
                media_type="application/octet-stream",
                filename="exported_logs.yaml",
            )

        except Exception as e:
            logger.error(f"Error in export_logs: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            db.close()

    return app
