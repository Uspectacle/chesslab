"""Stockfish-based chess evaluation utilities."""

from typing import Any

import chess
import chess.engine

from chesslab.env import get_stockfish_url


class Evaluator:
    """Reusable chess position evaluator using Stockfish."""

    def __init__(self, depth: int = 10, mate_score: int = 20000):
        """Initialize the evaluator with a persistent Stockfish engine.

        Args:
            stockfish_path: Path to Stockfish executable
            depth: Search depth for evaluation (default: 10)
            mate_score: Score to assign for mate positions (default: 20000)
        """
        self.engine = chess.engine.SimpleEngine.popen_uci(get_stockfish_url())
        self.depth = depth
        self.mate_score = mate_score
        self.limit = chess.engine.Limit(depth=depth)

    def get_centipawn(self, board: chess.Board) -> float:
        """Return Stockfish's centipawn evaluation of a position.

        Args:
            board: The chess position to evaluate

        Returns:
            Centipawn evaluation (positive for White, negative for Black)
        """
        info = self.engine.analyse(board, self.limit)
        score = info.get("score")
        assert score
        return score.white().score(mate_score=self.mate_score)

    def get_centipawn_loss(self, board: chess.Board, move: chess.Move) -> float:
        """Return the centipawn loss after making a move.

        Args:
            board (chess.Board): The position before the move.
            move (chess.Move): The move to evaluate.

        Returns:
            float: Absolute difference in evaluation before and after the move.
        """
        next_board = board.copy()
        next_board.push(move)

        return abs(self.get_centipawn(board) - self.get_centipawn(next_board))

    def close(self):
        """Close the Stockfish engine."""
        self.engine.quit()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        """Context manager exit - ensures engine is closed."""
        self.close()
