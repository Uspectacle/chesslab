"""Chess Game manager using Player instances."""

import logging
import math

import chess
import structlog

from chesslab.analysis.elo_tools import expected_score
from chesslab.analysis.evaluator import Evaluator
from chesslab.storage import Game, get_session
from chesslab.storage.game_tools import get_board, get_game

logger = structlog.get_logger()


class GameAnalysis:
    """Manages a single chess game between two Player instances."""

    def __init__(
        self,
        game: Game,
        evaluator: Evaluator,
    ) -> None:
        self.game = game
        self.evaluator = evaluator

    @property
    def board(self) -> chess.Board:
        return get_board(self.game)

    @property
    def white_score(self) -> float:
        """Return 1 if white wins, 0 if white loses, 0.5 if draw."""
        if self.board.result() == "1-0":
            return 1.0
        elif self.board.result() == "0-1":
            return 0.0
        return 0.5

    @property
    def black_score(self) -> float:
        """Return 1 if black wins, 0 if black loses, 0.5 if draw."""
        if self.board.result() == "0-1":
            return 1.0
        elif self.board.result() == "1-0":
            return 0.0
        return 0.5

    def get_score(self, player_id: int) -> float:
        if player_id is self.game.white_player_id:
            return self.white_score

        if player_id is self.game.black_player_id:
            return self.black_score

        raise ValueError(f"Player {player_id} is not in the game {self.game.id}")

    @property
    def number_of_white_move(self) -> int:
        """Number of time white made a move."""
        return math.ceil(len(self.board.move_stack) / 2)

    @property
    def number_of_black_move(self) -> int:
        """Number of time black made a move."""
        return math.floor(len(self.board.move_stack) / 2)

    def get_number_of_move(self, player_id: int) -> float:
        if player_id is self.game.white_player_id:
            return self.number_of_white_move

        if player_id is self.game.black_player_id:
            return self.number_of_black_move

        raise ValueError(f"Player {player_id} is not in the game {self.game.id}")

    @property
    def white_centipawn_loss(self) -> float:
        """Return the accumulated centipawn loss of white."""
        board = chess.Board()
        loss = 0.0

        for move in self.board.move_stack:
            if board.turn == chess.WHITE:
                loss += self.evaluator.get_centipawn_loss(board, move)
            board.push(move)

        return loss

    @property
    def black_centipawn_loss(self) -> float:
        """Return the accumulated centipawn loss of black."""
        board = chess.Board()
        loss = 0.0

        for move in self.board.move_stack:
            if board.turn == chess.BLACK:
                loss += self.evaluator.get_centipawn_loss(board, move)
            board.push(move)

        return loss

    def get_centipawn_loss(self, player_id: int) -> float:
        if player_id is self.game.white_player_id:
            return self.white_centipawn_loss

        if player_id is self.game.black_player_id:
            return self.black_centipawn_loss

        raise ValueError(f"Player {player_id} is not in the game {self.game.id}")

    @property
    def white_average_centipawn_loss(self) -> float:
        """Return the average centipawn loss per white move."""
        if not self.number_of_white_move:
            return float("nan")

        return self.white_centipawn_loss / self.number_of_white_move

    @property
    def black_average_centipawn_loss(self) -> float:
        """Return the average centipawn loss per black move."""
        if not self.number_of_black_move:
            return float("nan")

        return self.black_centipawn_loss / self.number_of_black_move

    def get_average_centipawn_loss(self, player_id: int) -> float:
        if player_id is self.game.white_player_id:
            return self.white_average_centipawn_loss

        if player_id is self.game.black_player_id:
            return self.black_average_centipawn_loss

        raise ValueError(f"Player {player_id} is not in the game {self.game.id}")

    @property
    def report(self) -> str:
        """Determine whether to use t-test or z-test based on number of games."""
        return (
            f"{'-' * 50}\n"
            f"Game: {self.game.white_player.engine_type} (ID: {self.game.white_player.id}) "
            f"vs {self.game.black_player.engine_type} (ID: {self.game.black_player.id})\n"
            f"{'-' * 50}\n"
            f"Result: {self.game.result}\n"
            f"Number of move: {len(self.board.move_stack)}\n"
            f"Average centipawn loss for white: {self.white_average_centipawn_loss}\n"
            f"Average centipawn loss for black: {self.black_average_centipawn_loss}\n"
            f"White expected ELO: {self.game.white_player.expected_elo}\n"
            f"Black expected ELO: {self.game.black_player.expected_elo}\n"
            f"Expected probability of white wining: {expected_score(self.game.white_player.expected_elo, self.game.black_player.expected_elo)}"
        )


if __name__ == "__main__":
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    )
    logger.info("Starting analyze game script")

    with get_session() as session:
        with Evaluator() as evaluator:
            logger.info("Database session created")

            game = get_game(session=session, game_id=100)
            assert game
            analysis = GameAnalysis(evaluator=evaluator, game=game)

            print(analysis.report)
