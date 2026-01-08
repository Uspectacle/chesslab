"""Random move chess engine for ChessLab.

This engine selects moves randomly from all legal moves, with optional
seed support for reproducible games.
"""

import logging
import random
from typing import Optional

import structlog

from chesslab.engines.base_engine import BaseEngine
from chesslab.engines.options.options import OptionSpin

logger = structlog.get_logger(
    level=logging.DEBUG,
    format="%(levelname)s %(name)s: %(message)s",
)


class RandomEngine(BaseEngine):
    """Chess engine that selects random legal moves.

    This engine can be useful for:
    - Testing other engines against a baseline
    - Generating random game datasets
    - Educational purposes
    - Debugging tournament infrastructure

    Attributes:
        name: Engine name identifier
        author: Engine author
        options: List of configurable options (Seed)
    """

    name: str = "RandomEngine"
    author: str = "ChessLab"
    options = [OptionSpin(name="Seed", default=0, min=0, max=2147483647)]

    def __init__(self):
        """Initialize the random engine."""
        self._rng: Optional[random.Random] = None
        super().__init__()

    @property
    def seed(self) -> int:
        """Get the current random seed value."""
        option = self.get_option("Seed")
        if option is None:
            raise RuntimeError("Option Seed not found")

        return option.value

    def reset(self) -> None:
        """Reset the board and random number generator."""
        self._rng = random.Random(self.seed if self.seed else None)
        logger.debug("Random number generator initialized", seed=self.seed)
        super().reset()

    def bestmove(self) -> str:
        """Generate a random move from all legal moves.

        Returns:
            Random move in UCI string format (e.g., 'e2e4', 'e7e8q')

        Raises:
            RuntimeError: If RNG is not initialized
        """
        if self._rng is None:
            raise RuntimeError("Random number generator not initialized")

        if self._board is None:
            raise RuntimeError("Board not initialized")

        legal_moves = list(self._board.legal_moves)
        move = self._rng.choice(legal_moves)

        return move.uci()


if __name__ == "__main__":
    # Run in UCI mode for standalone execution
    # Usage: python random_engine.py
    # Or compile: pyinstaller --onefile random_engine.py
    #
    # Example UCI session:
    #   uci
    #   setoption name Seed value 42
    #   isready
    #   ucinewgame
    #   position startpos moves e2e4
    #   go
    #   quit

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
    )

    engine = RandomEngine()
    engine.loop()
