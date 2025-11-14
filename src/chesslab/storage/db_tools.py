"""SQLAlchemy models for ChessLab database schema.

Defines the database structure for players, games, moves, evaluations, and batch requests.
Uses PostgreSQL with proper concurrent write support and JSON fields.
"""

import os

import structlog
from sqlalchemy import (
    Engine,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import (
    Session,
    sessionmaker,
)

logger = structlog.get_logger()
Base = declarative_base()


def get_default_database_url() -> str:
    """Get database URL from environment or use default."""
    return os.getenv(
        "DATABASE_URL", "postgresql://chesslab:chesslab_dev@localhost:5432/chesslab"
    )


def create_db_engine(
    database_url: str | None = None,
) -> Engine:
    """Create database engine with sensible defaults.

    Args:
        database_url: PostgreSQL connection string. If None, uses get_default_database_url()

    Returns:
        SQLAlchemy Engine instance
    """
    if database_url is None:
        database_url = get_default_database_url()

    logger.info("Creating database engine", database_url=database_url)
    engine = create_engine(
        database_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=False,
    )
    return engine


def create_session(engine: Engine) -> Session:
    """Create a new database session.

    Args:
        engine: SQLAlchemy Engine instance

    Returns:
        Session instance
    """
    Session = sessionmaker(bind=engine)
    return Session()


def get_session(database_url: str | None = None) -> Session:
    return create_session(create_db_engine(database_url))


def init_db(engine: Engine):
    """Initialize database schema by creating all tables.

    Args:
        engine: SQLAlchemy Engine instance
    """
    logger.info("Initializing database schema")
    Base.metadata.create_all(engine)
    logger.info("Database schema created successfully")
