"""Database access layer with CRUD operations and query helpers.

Provides high-level functions for interacting with the database,
abstracting away SQLAlchemy details for common operations.
"""

from datetime import datetime
from typing import List, Optional

import chess
import chess.engine
import structlog
from sqlalchemy.orm import Session

from chesslab.storage.async_tools import run_async_background
from chesslab.storage.db_tools import get_session
from chesslab.storage.player_tools import get_engine
from chesslab.storage.schema import Move

logger = structlog.get_logger()


async def play_move(move: Move):
    if move.is_white:
        player = move.game.white_player
    else:
        player = move.game.black_player

    engine = get_engine(player)
    board = chess.Board(move.fen_before)

    result = engine.play(board=board, limit=chess.engine.Limit())  # can be very long
    assert result.move

    session = get_session()
    updated_move = session.get(Move, move.id)
    assert updated_move

    if updated_move.is_white:
        player = updated_move.game.white_player
    else:
        player = updated_move.game.black_player

    engine = get_engine(player)
    board = chess.Board(updated_move.fen_before)

    updated_move.played_at = datetime.now()
    updated_move.uci_move = result.move.uci()

    session.commit()
    session.refresh(updated_move)


def create_move(
    session: Session,
    game_id: int,
    ply_index: int,
    move_number: int,
    is_white: bool,
    fen_before: str,
    uci_move: Optional[str] = None,
) -> Move:
    move = Move(
        game_id=game_id,
        ply_index=ply_index,
        move_number=move_number,
        is_white=is_white,
        fen_before=fen_before,
        uci_move=uci_move,
        created_at=datetime.now(),
    )
    session.add(move)
    session.commit()
    session.refresh(move)

    run_async_background(play_move(move=move))

    return move


def get_game_moves(session: Session, game_id: int) -> List[Move]:
    """Get all moves for a game in order.

    Args:
        session: Database session
        game_id: Game ID

    Returns:
        List of move instances ordered by ply
    """
    return (
        session.query(Move)
        .filter(Move.game_id == game_id)
        .order_by(Move.ply_index)
        .all()
    )
