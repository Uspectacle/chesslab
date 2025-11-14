"""Database setup and migration script.

Initializes the PostgreSQL database schema and runs any migrations.
"""

import logging
import sys

import structlog

from chesslab.env import get_database_url
from chesslab.storage.db_tools import create_db_engine, init_db
from chesslab.storage.schema import Base

logger = structlog.get_logger()


def setup_database(database_url: str | None = None):
    """Initialize database schema.

    Args:
        database_url: PostgreSQL connection string (uses env var if not provided)
    """
    if database_url is None:
        database_url = get_database_url()

    logger.info("Setting up database", database_url=database_url)

    try:
        # Create engine
        engine = create_db_engine(database_url)

        # Test connection
        with engine.connect():
            logger.info("Database connection successful")

        # Create all tables
        init_db(engine)

        logger.info("Database setup completed successfully")

        # Print table summary
        print("\n=== Database Schema Created ===")
        for table_name in Base.metadata.tables.keys():
            print(f"  ✓ {table_name}")
        print()

        return True

    except Exception as e:
        logger.error("Database setup failed", error=str(e))
        print(f"\nERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Start the database: make db-up")
        print("2. Or run: docker compose up -d")
        print("3. Check DATABASE_URL in .env file")
        return False


def drop_all_tables(database_url: str | None = None):
    """Drop all tables (use with caution!).

    Args:
        database_url: PostgreSQL connection string
    """
    if database_url is None:
        database_url = get_database_url()

    logger.warning("Dropping all tables", database_url=database_url)

    try:
        engine = create_db_engine(database_url)
        Base.metadata.drop_all(engine)
        logger.info("All tables dropped")
        return True
    except Exception as e:
        logger.error("Failed to drop tables", error=str(e))
        return False


if __name__ == "__main__":
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )

    db_url = get_database_url()

    print("ChessLab Database Setup")
    print("=" * 50)
    print(f"Database: {db_url}")
    print()

    print("WARNING: Do you want to delete all existing data?")
    response = input("Type 'yes' to confirm: ")
    if response.lower() != "yes":
        print("Aborted")
    else:
        if not drop_all_tables(db_url):
            sys.exit(1)
    print()

    if setup_database(db_url):
        print("Setup completed successfully!")
