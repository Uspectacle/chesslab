import os
import sys
from pathlib import Path
from typing import Dict, List

from chesslab.engines.random_engine import RandomEngine

ENGINES_DIR = Path(__file__).parent
CHESSLAB_DIR = ENGINES_DIR.parent
SRC_DIR = CHESSLAB_DIR.parent

engine_commands: Dict[str, List[str] | str] = {
    "RandomEngine": [sys.executable, str(ENGINES_DIR / "random_engine.py")],
    "Stockfish": os.getenv(
        "STOCKFISH", f"{SRC_DIR}/third_party/stockfish/src/stockfish"
    ),
}

__all__ = ["RandomEngine", "engine_commands"]
