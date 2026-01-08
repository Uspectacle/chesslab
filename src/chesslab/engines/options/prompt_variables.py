"""LLM tools and utilities for ChessLab.

Provides model loading, prompt formatting, and move parsing functionality.
"""

from dataclasses import dataclass
from datetime import datetime

import chess
import chess.pgn
import structlog

logger = structlog.get_logger()


@dataclass
class PromptVariables:
    """Variables available for prompt template formatting.

    Attributes:
        elo: Expected ELO rating
        date: Current date/time
        fen: Current board position in FEN notation
        legal_moves: Comma-separated list of legal moves
        pgn: Game history in PGN format
        move_number: Current move number
        side_to_move: 'White' or 'Black'
    """

    elo: int
    date: str
    fen: str
    legal_moves: str
    pgn: str
    move_number: int
    side_to_move: str

    @classmethod
    def from_board(cls, board: chess.Board, elo: int = 1500) -> "PromptVariables":
        """Create PromptVariables from a chess board.

        Args:
            board: Current chess board state
            elo: Expected ELO rating

        Returns:
            PromptVariables instance
        """
        # Get legal moves
        legal_moves = [move.uci() for move in board.legal_moves]
        legal_moves_str = ", ".join(legal_moves)

        # Generate PGN
        game = chess.pgn.Game()
        node = game
        temp_board = chess.Board()
        for move in board.move_stack:
            node = node.add_variation(move)
            temp_board.push(move)
        pgn_str = str(game)

        return cls(
            elo=elo,
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            fen=board.fen(),
            legal_moves=legal_moves_str,
            pgn=pgn_str,
            move_number=board.fullmove_number,
            side_to_move="White" if board.turn else "Black",
        )

    def format_template(self, template: str) -> str:
        """Format a template string with these variables.

        Args:
            template: Template string with {variable} placeholders

        Returns:
            Formatted string
        """
        try:
            return template.format(
                elo=self.elo,
                date=self.date,
                fen=self.fen,
                legal_moves=self.legal_moves,
                pgn=self.pgn,
                move_number=self.move_number,
                side_to_move=self.side_to_move,
            )
        except KeyError as e:
            logger.error("Invalid template variable", variable=str(e))
            return template
