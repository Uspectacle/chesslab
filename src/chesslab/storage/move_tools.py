"""Database access layer with CRUD operations and query helpers.

Provides high-level functions for interacting with the database,
abstracting away SQLAlchemy details for common operations.
"""

from datetime import datetime
from typing import List, Optional

import structlog
from sqlalchemy.orm import Session

from chesslab.storage.schema import Game, Move

logger = structlog.get_logger()


def create_move(
    session: Session,
    game_id: int,
    ply_index: int,
    move_number: int,
    is_white: bool,
    fen_before: str,
    uci_move: Optional[str] = None,
) -> Move:
    logger.info(
        "Creating move",
        game_id=game_id,
        ply_index=ply_index,
        move_number=move_number,
        is_white=is_white,
        uci_move=uci_move,
    )

    move = Move(
        game_id=game_id,
        ply_index=ply_index,
        move_number=move_number,
        is_white=is_white,
        fen_before=fen_before,
        uci_move=uci_move,
        created_at=datetime.now(),
        played_at=datetime.now() if uci_move is not None else None,
    )
    session.add(move)
    session.commit()
    session.refresh(move)

    logger.info(
        "Move created successfully",
        move_id=move.id,
        game_id=game_id,
        ply_index=ply_index,
    )

    return move


def get_game_moves(session: Session, game_id: int) -> List[Move]:
    """Get all moves for a game in order.

    Args:
        session: Database session
        game_id: Game ID

    Returns:
        List of move instances ordered by ply
    """
    logger.debug("Fetching game moves", game_id=game_id)
    moves = (
        session.query(Move)
        .filter(Move.game_id == game_id)
        .order_by(Move.ply_index)
        .all()
    )
    logger.info(
        "Retrieved game moves",
        game_id=game_id,
        move_count=len(moves),
    )
    return moves


def delete_moves_not_played(session: Session, game: Game) -> None:
    logger.debug("Deleting moves not played", game_id=game.id)
    moves = (
        session.query(Move)
        .filter((Move.game_id == game.id) & (not Move.uci_move))
        .all()
    )

    if not len(moves):
        return

    for move in moves:
        session.delete(move)

    session.commit()

    logger.info(
        "Deleted moves successfully",
        game_id=game.id,
        deleted_count=len(moves),
    )
