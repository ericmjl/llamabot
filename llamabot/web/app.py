"""Create a FastAPI app to visualize and compare prompts and messages."""

from typing import Optional
from fastapi import FastAPI, Request, HTTPException, Form, Query
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
from tempfile import NamedTemporaryFile
from enum import Enum

from llamabot.recorder import MessageLog, Base, upgrade_database, Prompt
from llamabot.components.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

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
        db = SessionLocal()
        try:
            # Get unique function names from prompts table
            prompts = (
                db.query(Prompt.function_name)
                .distinct()
                .order_by(Prompt.function_name)
                .all()
            )
            # Convert from list of tuples to list of strings
            prompts = [{"function_name": p[0]} for p in prompts]

            return templates.TemplateResponse(
                "index.html", {"request": request, "prompts": prompts}
            )
        finally:
            db.close()

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
    async def get_log(log_id: int, expanded: bool = False):
        """Get a single log by ID.

        :param log_id: The ID of the log to retrieve
        :param expanded: Whether to show expanded messages (default: False)
        """
        db = SessionLocal()
        try:
            log = db.query(MessageLog).filter(MessageLog.id == log_id).first()
            if log is None:
                raise HTTPException(status_code=404, detail="Log not found")

            message_log_str = (
                str(log.message_log) if log.message_log is not None else "[]"
            )
            message_log = json.loads(message_log_str)

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

            return templates.TemplateResponse(
                "log_details.html",
                {
                    "request": {},
                    "log": {
                        "id": log.id,
                        "object_name": log.object_name,
                        "timestamp": log.timestamp,
                        "message_log": message_log,
                        "model_name": log.model_name,
                        "temperature": log.temperature,
                        "rating": log.rating,
                    },
                    "expanded": expanded,
                    "log_id": log.id,
                    "rating": log.rating,
                },
            )
        finally:
            db.close()

    @app.get("/log/{log_id}/expand")
    async def expand_log(log_id: int, request: Request):
        """Get log details with all messages expanded."""
        db = SessionLocal()
        try:
            log = db.query(MessageLog).filter(MessageLog.id == log_id).first()
            if log is None:
                raise HTTPException(status_code=404, detail="Log not found")
            message_log = json.loads(str(log.message_log) if log.message_log else "[]")

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

            return templates.TemplateResponse(
                "message_log.html",
                {
                    "request": request,
                    "log": {
                        "id": log.id,
                        "message_log": message_log,
                    },
                    "expanded": True,
                },
            )
        finally:
            db.close()

    @app.get("/log/{log_id}/collapse")
    async def collapse_log(log_id: int, request: Request):
        """Get log details with all messages collapsed."""
        db = SessionLocal()
        try:
            log = db.query(MessageLog).filter(MessageLog.id == log_id).first()
            if log is None:
                raise HTTPException(status_code=404, detail="Log not found")
            message_log = json.loads(str(log.message_log) if log.message_log else "[]")

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

            return templates.TemplateResponse(
                "message_log.html",
                {
                    "request": request,
                    "log": {
                        "id": log.id,
                        "message_log": message_log,
                    },
                    "expanded": False,
                },
            )
        finally:
            db.close()

    @app.get("/prompt_history")
    async def get_prompt_history(request: Request, function_name: str):
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

        db = SessionLocal()
        try:
            # Get all prompts for this function, ordered by ID descending (newest first)
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
                            fromfile=f"Version {prompts[i+1].hash[:8]}",
                            tofile=f"Version {prompt.hash[:8]}",
                            lineterm="",
                        )
                    )
                    logger.debug(f"Generated diff between versions {i+1} and {i}")

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
        except Exception as e:
            logger.error(f"Error getting prompt history: {str(e)}")
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

    class ExportFormat(str, Enum):
        """Export formats supported by the API."""

        OPENAI = "openai"
        # Placeholders for future formats
        # LLAMA = "llama"
        # ANTHROPIC = "anthropic"

    @app.get("/export/{format}")
    async def export_logs(
        format: ExportFormat,
        text_filter: str = Query(default=""),
        function_name: str = Query(default=""),
        positive_only: bool = Query(default=False),
    ):
        """Export logs in various formats.

        :param format: Format to export in (openai, llama, anthropic)
        :param text_filter: Text to filter logs by
        :param function_name: Function name to filter logs by
        :param positive_only: Whether to export only positively rated logs
        """
        db = SessionLocal()
        try:
            # Start with base query
            query = db.query(MessageLog).order_by(MessageLog.timestamp.desc())

            # Apply filters to match what's visible in the table
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

            if text_filter:
                query = query.filter(
                    or_(
                        MessageLog.object_name.ilike(f"%{text_filter}%"),
                        MessageLog.message_log.ilike(f"%{text_filter}%"),
                        MessageLog.model_name.ilike(f"%{text_filter}%"),
                    )
                )

            # Apply positive rating filter if requested
            if positive_only:
                query = query.filter(MessageLog.rating == 1)

            logs = query.all()

            # Create temporary file for output
            suffix = ".jsonl" if format == ExportFormat.OPENAI else ".json"
            with NamedTemporaryFile(mode="w", delete=False, suffix=suffix) as temp_file:
                for log in logs:
                    try:
                        # Convert SQLAlchemy Column to string safely
                        message_log_str = (
                            str(log.message_log)
                            if log.message_log is not None
                            else "[]"
                        )
                        messages = []

                        for msg in json.loads(message_log_str):
                            role = msg.get("role", "")
                            content = msg.get("content", "")

                            # Create message objects based on role
                            message = None
                            if role == "system":
                                message = SystemMessage(content=content)
                            elif role == "user":
                                message = HumanMessage(content=content)
                            elif role == "assistant":
                                message = AIMessage(content=content)
                            elif role == "tool":
                                message = ToolMessage(content=content)

                            if message:
                                messages.append(message.dict(exclude={"prompt_hash"}))

                        if messages:
                            if format == ExportFormat.OPENAI:
                                # OpenAI format
                                json.dump({"messages": messages}, temp_file)
                                temp_file.write("\n")
                            elif format == ExportFormat.LLAMA:
                                # Placeholder for Llama format
                                # TODO: Implement Llama format
                                pass
                            elif format == ExportFormat.ANTHROPIC:
                                # Placeholder for Anthropic format
                                # TODO: Implement Anthropic format
                                pass

                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Failed to process log {log.id}: {str(e)}")
                        continue

                temp_file_path = temp_file.name

            return FileResponse(
                temp_file_path,
                media_type="application/octet-stream",
                filename=f"conversations{suffix}",
            )

        except Exception as e:
            logger.error(f"Error in export_logs: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            db.close()

    @app.get("/filtered_logs")
    async def get_filtered_logs(
        request: Request, text_filter: str = "", function_name: str = ""
    ):
        """Get filtered logs based on text search and function name."""
        db = SessionLocal()
        try:
            query = db.query(MessageLog).order_by(MessageLog.timestamp.desc())

            # Apply function name filter if provided
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

            # Apply text filter if provided
            if text_filter:
                text_filter = text_filter.lower()
                query = query.filter(
                    or_(
                        MessageLog.object_name.ilike(f"%{text_filter}%"),
                        MessageLog.message_log.ilike(f"%{text_filter}%"),
                        MessageLog.model_name.ilike(f"%{text_filter}%"),
                    )
                )

            logs = query.all()
            log_data = []

            for log in logs:
                try:
                    message_log_content = (
                        json.loads(log.message_log) if log.message_log else []
                    )
                except json.JSONDecodeError:
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

                # Ensure data is ordered according to table columns
                log_data.append(
                    {
                        "id": log.id,  # Column 1
                        "object_name": log.object_name,  # Column 2
                        "timestamp": log.timestamp,  # Column 3
                        "model_name": log.model_name,  # Column 4
                        "temperature": log.temperature,  # Column 5
                        "prompt_names": formatted_prompt_names,  # Column 6
                        "rating": log.rating,
                    }
                )

            return templates.TemplateResponse(
                "log_tbody.html",
                {
                    "request": request,
                    "logs": log_data,
                },
            )
        finally:
            db.close()

    @app.post("/log/{log_id}/rate")
    async def rate_log(
        log_id: int,
        rating: int = Form(...),
    ):
        """Rate a log entry as helpful (1) or not helpful (0).

        :param log_id: The ID of the log to rate
        :param rating: 1 for helpful, 0 for not helpful
        """
        db = SessionLocal()
        try:
            log = db.query(MessageLog).filter(MessageLog.id == log_id).first()
            if log is None:
                raise HTTPException(status_code=404, detail="Log not found")

            # Update the rating
            setattr(log, "rating", rating)  # Use setattr to avoid type checking issues
            db.commit()

            # Return just the updated rating buttons
            return templates.TemplateResponse(
                "rating_buttons.html",
                {
                    "request": {},
                    "log_id": log_id,
                    "rating": rating,
                },
            )
        except Exception as e:
            logger.error(f"Error in rate_log: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            db.close()

    @app.get("/prompt/{prompt_hash}")
    async def get_prompt(prompt_hash: str):
        """Get prompt details by hash."""
        db = SessionLocal()
        try:
            prompt = db.query(Prompt).filter(Prompt.hash == prompt_hash).first()
            if prompt is None:
                raise HTTPException(status_code=404, detail="Prompt not found")

            return templates.TemplateResponse(
                "prompt_modal.html",
                {
                    "request": {},
                    "prompt": prompt,
                },
            )
        finally:
            db.close()

    return app
