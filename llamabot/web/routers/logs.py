"""Router for log-related endpoints."""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Request, Form, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import text, or_, func
from pathlib import Path
import json
from enum import Enum
from tempfile import NamedTemporaryFile
from loguru import logger
from pydantic import BaseModel

from llamabot.recorder import MessageLog, Prompt
from llamabot.components.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from llamabot.web.database import DbSession

router = APIRouter(tags=["logs"])
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


class ExportFormat(str, Enum):
    """Export formats supported by the API."""

    OPENAI = "openai"


class LogResponse(BaseModel):
    """Response model for log endpoints."""

    logs: List[dict]


@router.get("/", response_class=HTMLResponse)
async def get_logs(
    request: Request, db: DbSession, function_name: Optional[str] = None
):
    """Get all logs, optionally filtered by function name."""
    logger.debug(f"Received request for logs with function_name: {function_name}")

    query = db.query(MessageLog).order_by(MessageLog.timestamp.desc())

    if function_name:
        logger.debug(f"Filtering logs for function name: {function_name}")
        prompt_hashes = (
            db.query(Prompt.hash).filter(Prompt.function_name == function_name).all()
        )
        prompt_hashes = [hash[0] for hash in prompt_hashes]

        if len(prompt_hashes) > 0:
            conditions = [
                MessageLog.message_log.like(f"%{hash}%") for hash in prompt_hashes
            ]
            query = query.filter(or_(*conditions))
        else:
            logger.warning(f"No prompts found for function name: {function_name}")
            return templates.TemplateResponse(
                "logs/index.html",
                {"request": request, "logs": []},
            )

    logs = query.all()
    log_data = []

    for log in logs:
        try:
            message_log_str = (
                str(log.message_log) if log.message_log is not None else "[]"
            )
            message_log_content = json.loads(message_log_str)
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
            f"- {prompt.function_name} ({prompt.hash[:6]})" for prompt in prompts
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

    # Query all prompt functions and their version counts
    prompt_functions = (
        db.query(Prompt.function_name, func.count(Prompt.hash).label("count"))
        .group_by(Prompt.function_name)
        .all()
    )
    prompts = [{"function_name": fn, "count": count} for fn, count in prompt_functions]

    return templates.TemplateResponse(
        "logs/index.html",
        {"request": request, "logs": log_data, "prompts": prompts},
    )


@router.get("/filtered_logs", response_class=HTMLResponse)
async def get_filtered_logs(
    request: Request,
    db: DbSession,
    text_filter: str = "",
    function_name: str = "",
):
    """Get filtered logs based on text search and function name."""
    query = db.query(MessageLog).order_by(MessageLog.timestamp.desc())

    # Apply function name filter if provided
    if function_name:
        prompt_hashes = (
            db.query(Prompt.hash).filter(Prompt.function_name == function_name).all()
        )
        prompt_hashes = [hash[0] for hash in prompt_hashes]
        if len(prompt_hashes) > 0:
            conditions = [
                MessageLog.message_log.like(f"%{hash}%") for hash in prompt_hashes
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
            message_log_str = str(log.message_log) if log.message_log else "[]"
            message_log_content = json.loads(message_log_str)
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
            f"- {prompt.function_name} ({prompt.hash[:6]})" for prompt in prompts
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
                "rating": log.rating,
            }
        )

    return templates.TemplateResponse(
        "logs/log_tbody.html",
        {"request": request, "logs": log_data},
    )


@router.get("/{log_id}", response_class=HTMLResponse)
async def get_log(
    log_id: int,
    request: Request,
    db: DbSession,
    expanded: bool = False,
):
    """Get a single log by ID."""
    log = db.query(MessageLog).filter(MessageLog.id == log_id).first()
    if log is None:
        raise HTTPException(status_code=404, detail="Log not found")

    message_log_str = str(log.message_log) if log.message_log is not None else "[]"
    message_log = json.loads(message_log_str)

    # Fetch prompt names and templates for each message with a prompt_hash
    for message in message_log:
        if message.get("prompt_hash"):
            prompt = (
                db.query(Prompt).filter(Prompt.hash == message["prompt_hash"]).first()
            )
            if prompt:
                message["prompt_name"] = prompt.function_name
                message["prompt_template"] = prompt.template

        # Ensure tool_calls is properly formatted
        if "tool_calls" in message:
            # If tool_calls is a string, try to parse it as JSON
            if isinstance(message["tool_calls"], str):
                try:
                    message["tool_calls"] = json.loads(message["tool_calls"])
                except json.JSONDecodeError:
                    message["tool_calls"] = []
            # If tool_calls is None, set it to an empty list
            elif message["tool_calls"] is None:
                message["tool_calls"] = []

    return templates.TemplateResponse(
        "logs/log_details.html",
        {
            "request": request,
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


@router.get("/{log_id}/expand", response_class=HTMLResponse)
async def expand_log(log_id: int, request: Request, db: DbSession):
    """Get log details with all messages expanded."""
    log = db.query(MessageLog).filter(MessageLog.id == log_id).first()
    if log is None:
        raise HTTPException(status_code=404, detail="Log not found")

    message_log = json.loads(str(log.message_log) if log.message_log else "[]")

    # Fetch prompt names and templates for each message with a prompt_hash
    for message in message_log:
        if message.get("prompt_hash"):
            prompt = (
                db.query(Prompt).filter(Prompt.hash == message["prompt_hash"]).first()
            )
            if prompt:
                message["prompt_name"] = prompt.function_name
                message["prompt_template"] = prompt.template

    return templates.TemplateResponse(
        "logs/message_log.html",
        {
            "request": request,
            "log": {
                "id": log.id,
                "message_log": message_log,
            },
            "expanded": True,
        },
    )


@router.get("/{log_id}/collapse", response_class=HTMLResponse)
async def collapse_log(log_id: int, request: Request, db: DbSession):
    """Get log details with all messages collapsed."""
    log = db.query(MessageLog).filter(MessageLog.id == log_id).first()
    if log is None:
        raise HTTPException(status_code=404, detail="Log not found")

    message_log = json.loads(str(log.message_log) if log.message_log else "[]")

    # Fetch prompt names and templates for each message with a prompt_hash
    for message in message_log:
        if message.get("prompt_hash"):
            prompt = (
                db.query(Prompt).filter(Prompt.hash == message["prompt_hash"]).first()
            )
            if prompt:
                message["prompt_name"] = prompt.function_name
                message["prompt_template"] = prompt.template

    return templates.TemplateResponse(
        "logs/message_log.html",
        {
            "request": request,
            "log": {
                "id": log.id,
                "message_log": message_log,
            },
            "expanded": False,
        },
    )


@router.post("/{log_id}/rate", response_class=HTMLResponse)
async def rate_log(
    request: Request,
    db: DbSession,
    log_id: int,
    rating: int = Form(...),
):
    """Rate a log entry as helpful (1) or not helpful (0).

    :param log_id: The ID of the log to rate
    :param rating: 1 for helpful, 0 for not helpful
    """
    log = db.query(MessageLog).filter(MessageLog.id == log_id).first()
    if log is None:
        raise HTTPException(status_code=404, detail="Log not found")

    # Update the rating
    setattr(log, "rating", rating)  # Use setattr to avoid type checking issues
    db.commit()

    # Return just the updated rating buttons
    return templates.TemplateResponse(
        "logs/rating_buttons.html",
        {
            "request": request,
            "log_id": log_id,
            "rating": rating,
        },
    )


@router.get("/export/{format}")
async def export_logs(
    request: Request,
    db: DbSession,
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
    # Start with base query
    query = db.query(MessageLog).order_by(MessageLog.timestamp.desc())

    # Apply filters to match what's visible in the table
    if function_name:
        prompt_hashes = (
            db.query(Prompt.hash).filter(Prompt.function_name == function_name).all()
        )
        prompt_hashes = [hash[0] for hash in prompt_hashes]
        if len(prompt_hashes) > 0:
            conditions = [
                MessageLog.message_log.like(f"%{hash}%") for hash in prompt_hashes
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
                    str(log.message_log) if log.message_log is not None else "[]"
                )
                messages = []

                for msg in json.loads(message_log_str):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    tool_calls = msg.get("tool_calls", [])

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
                        message_dict = message.dict(exclude={"prompt_hash"})
                        message_dict["tool_calls"] = tool_calls
                        messages.append(message_dict)

                if messages:
                    if format == ExportFormat.OPENAI:
                        # OpenAI format
                        json.dump({"messages": messages}, temp_file)
                        temp_file.write("\n")

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to process log {log.id}: {str(e)}")
                continue

        temp_file_path = temp_file.name

    return FileResponse(
        temp_file_path,
        media_type="application/octet-stream",
        filename=f"conversations{suffix}",
    )


@router.get("/debug", response_class=HTMLResponse)
async def debug_logs(db: DbSession):
    """Debug endpoint to view raw message logs."""
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


# Add other log-related endpoints here...
