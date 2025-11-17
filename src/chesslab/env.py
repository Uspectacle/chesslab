"""Database setup and migration script.

Initializes the PostgreSQL database schema and runs any migrations.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List

import structlog
from dotenv import load_dotenv

CHESSLAB_DIR = Path(__file__).parent
SRC_DIR = CHESSLAB_DIR.parent
ROOT_DIR = SRC_DIR.parent

logger = structlog.get_logger()
sys.path.insert(0, str(ROOT_DIR))


def load_env():
    env_path = ROOT_DIR / ".env"

    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        logger.debug("Enviroment file loaded", env_path=env_path)
    else:
        logger.error("Enviroment file not found", env_path=env_path)


load_env()


def get_database_url() -> str:
    return os.getenv(
        "DATABASE_URL", "postgresql://chesslab:chesslab_dev@localhost:5432/chesslab"
    )


def get_stockfish_url() -> str | List[str]:
    return os.getenv("STOCKFISH_URL", f"{SRC_DIR}/third_party/stockfish/src/stockfish")


if __name__ == "__main__":
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )

    print(f"get_database_url: {get_database_url()}")
    print(f"get_stockfish_url: {get_stockfish_url()}")
