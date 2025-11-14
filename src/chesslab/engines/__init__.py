import sys
from pathlib import Path
from typing import Dict, List

from chesslab.env import get_stockfish_url

ENGINES_DIR = Path(__file__).parent

engine_commands: Dict[str, List[str] | str] = {
    "RandomEngine": [sys.executable, str(ENGINES_DIR / "random_engine.py")],
    "Stockfish": get_stockfish_url(),
}

__all__ = ["engine_commands"]
