"""Database setup for the web app."""

from typing import Generator, Annotated
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from pathlib import Path
from fastapi import Depends


def get_engine(db_path: Path):
    """Get SQLAlchemy engine.

    :param db_path: Path to the database file.
    """
    return create_engine(f"sqlite:///{db_path}")


def get_sessionmaker(engine):
    """Get SQLAlchemy sessionmaker.

    :param engine: SQLAlchemy engine.
    """
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Global SessionLocal variable
SessionLocal = None


def get_db() -> Generator[Session, None, None]:
    """Get database session.

    :yield: Database session.
    """
    if SessionLocal is None:
        raise RuntimeError("Database session maker not initialized")

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_sessionmaker(engine):
    """Initialize the global SessionLocal.

    :param engine: SQLAlchemy engine.
    """
    global SessionLocal
    SessionLocal = get_sessionmaker(engine)


# Type alias for dependency injection
DbSession = Annotated[Session, Depends(get_db)]
