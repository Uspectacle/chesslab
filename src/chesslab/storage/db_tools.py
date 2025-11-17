"""SQLAlchemy models for ChessLab database schema.

Defines the database structure for players, games, moves, evaluations, and batch requests.
Uses PostgreSQL with proper concurrent write support and JSON fields.
"""

from contextlib import contextmanager

import structlog
from sqlalchemy import (
    Engine,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Session,
    sessionmaker,
)

from chesslab.env import get_database_url

logger = structlog.get_logger()


class Base(DeclarativeBase):
    pass


def create_db_engine(
    database_url: str | None = None,
) -> Engine:
    """Create database engine with sensible defaults.

    Args:
        database_url: PostgreSQL connection string. If None, uses get_database_url()

    Returns:
        SQLAlchemy Engine instance
    """
    if database_url is None:
        database_url = get_database_url()

    logger.debug("Creating database engine", database_url=database_url)
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


@contextmanager
def get_session(database_url: str | None = None):
    """Context manager for database sessions with automatic cleanup."""
    session = create_session(create_db_engine(database_url))
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine: Engine):
    """Initialize database schema by creating all tables.

    Args:
        engine: SQLAlchemy Engine instance
    """
    logger.info("Initializing database schema")
    Base.metadata.create_all(engine)
    logger.info("Database schema created successfully")
