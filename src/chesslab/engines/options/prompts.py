"""LLM tools and utilities for ChessLab.

Provides model loading, prompt formatting, and move parsing functionality.
"""

import structlog

logger = structlog.get_logger()

minimal_system_prompt = "You are a chess engine. Respond with only the best move in UCI format (e.g., e2e4)."

standard_system_prompt = (
    "You are a chess engine with an ELO rating of {elo}. Analyze the position carefully and respond with your best move in UCI format (e.g., e2e4 for pawn to e4, e7e8q for pawn promotion to queen).\n"
    "\n"
    "Only output the move, nothing else."
)

detailed_system_prompt = (
    "You are a strong chess engine with an ELO rating of {elo}. You are playing as {side_to_move} on move {move_number}.\n"
    "\n"
    "Analyze the position and select your best move. Consider:\n"
    "- Material balance\n"
    "- King safety\n"
    "- Piece activity\n"
    "- Pawn structure\n"
    "\n"
    "Respond with only the move in UCI format (e.g., e2e4)."
)

SYSTEM_PROMPTS = {
    "minimal": minimal_system_prompt,
    "standard": standard_system_prompt,
    "detailed": detailed_system_prompt,
}

minimal_user_prompt = "Position: {fen}\nLegal moves: {legal_moves}\n\nYour move:"

standard_user_prompt = (
    "Current position (FEN): {fen}\n"
    "Side to move: {side_to_move}\n"
    "Move number: {move_number}\n"
    "\n"
    "Legal moves: {legal_moves}\n"
    "\n"
    "What is your move?"
)

detailed_user_prompt = (
    "Game position:\n"
    "{pgn}\n"
    "\n"
    "Current FEN: {fen}\n"
    "You are playing: {side_to_move}\n"
    "Move number: {move_number}\n"
    "\n"
    "Available legal moves: {legal_moves}\n"
    "\n"
    "Analyze the position and provide your best move in UCI format:"
)

USER_PROMPTS = {
    "minimal": minimal_user_prompt,
    "standard": standard_user_prompt,
    "detailed": detailed_user_prompt,
}
