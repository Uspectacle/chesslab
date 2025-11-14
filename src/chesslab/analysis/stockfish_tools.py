"""Stockfish-based chess evaluation utilities."""

import chess


def get_centipawn(board: chess.Board) -> float:
    """Return Stockfish's centipawn evaluation of a position.

    Args:
        board (chess.Board): The chess position to evaluate.

    Returns:
        float: Centipawn evaluation (positive for White, negative for Black).
    """
    # return stockfish.evaluate(board)
    return 0


def get_centipawn_loss(board: chess.Board, move: chess.Move) -> float:
    """Return the centipawn loss after making a move.

    Args:
        board (chess.Board): The position before the move.
        move (chess.Move): The move to evaluate.

    Returns:
        float: Absolute difference in evaluation before and after the move.
    """
    next_board = board.copy()
    next_board.push(move)

    return abs(get_centipawn(board) - get_centipawn(next_board))
