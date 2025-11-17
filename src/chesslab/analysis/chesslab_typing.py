"""Type definitions for collective chess."""

import logging
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

import chess

Move = str  # UCI format move string
Score = float  # Score/evaluation of a move
Elo = float  # Elo rating type
Ballot = Dict[Move, Score]  # Mapping of moves to their scores


@runtime_checkable
class Player(Protocol):
    """Protocol defining the interface for chess players."""

    elo: float = 0
    bot_id: str = "player"
    logger: logging.Logger

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize player.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(
            f"collective_chess.player.{self.bot_id}"
        )

    def evaluate(self, board: chess.Board) -> Ballot:
        """Evaluate the current position and return a ballot of move scores.

        Args:
            board: Current chess position

        Returns:
            A mapping of UCI format moves to their scores
        """
        raise NotImplementedError("Player undefined")

    def evaluate_multiple(self, boards: List[chess.Board]) -> List[Ballot]:
        """For each board, evaluate the position and return a ballot of move scores.

        Args:
            boards: List of chess position

        Returns:
            For each board, a mapping of UCI format moves to their scores
        """
        return [self.evaluate(board) for board in boards]

    def __str__(self) -> str:
        """Return string representation of player."""
        return f"{self.bot_id}_(ELO_{int(self.elo)})"


BallotMap = Dict[Player, Ballot]
Aggregator = Callable[[BallotMap], Ballot]
