"""LLM tools and utilities for ChessLab.

Provides model loading, prompt formatting, and move parsing functionality.
"""

import re
from typing import Callable, Dict, Optional

import chess
import structlog

logger = structlog.get_logger()


def parse_uci_only(response: str, board: chess.Board) -> Optional[str]:
    """Extract UCI move from response (e.g., e2e4, e7e8q).

    Args:
        response: LLM response text
        board: Current chess board

    Returns:
        UCI move string if found, None otherwise
    """
    legal_moves = {move.uci(): move for move in board.legal_moves}

    # UCI pattern: letter, digit, letter, digit, optional promotion
    uci_pattern = r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b"
    matches = re.findall(uci_pattern, response.lower())

    for match in matches:
        if match in legal_moves:
            logger.debug("Found valid UCI move", move=match, strategy="uci_only")
            return match

    return None


def parse_san_and_uci(response: str, board: chess.Board) -> Optional[str]:
    """Extract move from UCI or SAN notation (e.g., Nf3, e2e4).

    Args:
        response: LLM response text
        board: Current chess board

    Returns:
        UCI move string if found, None otherwise
    """
    # Try UCI first
    uci_move = parse_uci_only(response, board)
    if uci_move:
        return uci_move

    # Try SAN
    legal_moves = {move.uci(): move for move in board.legal_moves}
    for move_uci, move_obj in legal_moves.items():
        san = board.san(move_obj)
        # Check if SAN appears in response (case-insensitive)
        if san.lower() in response.lower():
            logger.debug(
                "Found valid SAN move", san=san, uci=move_uci, strategy="san_and_uci"
            )
            return move_uci

    return None


def parse_flexible(response: str, board: chess.Board) -> Optional[str]:
    """Extract move using multiple notation formats and patterns.

    Tries UCI, SAN, coordinate notation (e2-e4), and common variations.

    Args:
        response: LLM response text
        board: Current chess board

    Returns:
        UCI move string if found, None otherwise
    """
    # Try standard parsers first
    move = parse_san_and_uci(response, board)
    if move:
        return move

    legal_moves = {move.uci(): move for move in board.legal_moves}

    # Try coordinate notation with dash (e2-e4)
    coord_pattern = r"\b([a-h][1-8])-([a-h][1-8])([qrbn]?)\b"
    matches = re.findall(coord_pattern, response.lower())
    for from_sq, to_sq, promotion in matches:
        uci = from_sq + to_sq + promotion
        if uci in legal_moves:
            logger.debug("Found valid coordinate move", move=uci, strategy="flexible")
            return uci

    # Try to find piece + destination (e.g., "knight to f3")
    piece_map = {
        "knight": "n",
        "bishop": "b",
        "rook": "r",
        "queen": "q",
        "king": "k",
        "pawn": "",
    }
    for piece_name, piece_letter in piece_map.items():
        pattern = rf"{piece_name}\s+(?:to\s+)?([a-h][1-8])"
        matches = re.findall(pattern, response.lower())
        for dest in matches:
            # Try all moves that end at this square with this piece
            for move_uci, move_obj in legal_moves.items():
                if move_uci[2:4] == dest:
                    piece = board.piece_at(move_obj.from_square)
                    if piece and piece.symbol().lower() == (
                        piece_letter if piece_letter else "p"
                    ):
                        logger.debug(
                            "Found valid descriptive move",
                            move=move_uci,
                            strategy="flexible",
                        )
                        return move_uci

    return None


PARSERS: Dict[str, Callable[[str, chess.Board], Optional[str]]] = {
    "uci_only": parse_uci_only,
    "san_and_uci": parse_san_and_uci,
    "flexible": parse_flexible,
}
