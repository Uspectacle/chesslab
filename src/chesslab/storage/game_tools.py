"""Database access layer with CRUD operations and query helpers.

Provides high-level functions for interacting with the database,
abstracting away SQLAlchemy details for common operations.
"""

from datetime import datetime
from typing import Dict, List, Optional

import structlog
from sqlalchemy import desc
from sqlalchemy.orm import Session

from chesslab.storage.schema import Game, Move

logger = structlog.get_logger()


def create_game(
    session: Session,
    white_player_id: int,
    black_player_id: int,
    opening_fen: Optional[str] = None,
) -> Game:
    game = Game(
        white_player_id=white_player_id,
        black_player_id=black_player_id,
        started_at=datetime.now(),
        game_metadata={"opening_fen": opening_fen} if opening_fen else {},
    )
    session.add(game)
    session.commit()
    logger.info("Game created", game_id=game.id)
    return game


def get_game(session: Session, game_id: int) -> Optional[Game]:
    return session.query(Game).filter(Game.id == game_id).first()


def get_player_games(session: Session, player_id: int, limit: int = 100) -> List[Game]:
    return (
        session.query(Game)
        .filter(
            (Game.white_player_id == player_id) | (Game.black_player_id == player_id)
        )
        .order_by(desc(Game.started_at))
        .limit(limit)
        .all()
    )


def get_head_to_head_games(
    session: Session, player1_id: int, player2_id: int
) -> List[Game]:
    return (
        session.query(Game)
        .filter(
            (
                (Game.white_player_id == player1_id)
                & (Game.black_player_id == player2_id)
            )
            | (
                (Game.white_player_id == player2_id)
                & (Game.black_player_id == player1_id)
            )
        )
        .all()
    )


def get_games_by_players(
    session: Session, white_player_id: int, black_player_id: int
) -> List[Game]:
    return (
        session.query(Game)
        .filter(
            (
                (Game.white_player_id == white_player_id)
                & (Game.black_player_id == black_player_id)
            )
        )
        .all()
    )


def get_or_create_games(
    session: Session,
    white_player_id: int,
    black_player_id: int,
    num_games: int = 1,
    remove_exiting: bool = True,
    get_exiting: bool = True,
) -> List[Game]:
    if remove_exiting:
        delete_games_by_players(
            session=session,
            white_player_id=white_player_id,
            black_player_id=black_player_id,
        )

    games: List[Game] = []

    if get_exiting:
        games = get_games_by_players(
            session=session,
            white_player_id=white_player_id,
            black_player_id=black_player_id,
        )

    while len(games) < num_games:
        games.append(create_game(session, white_player_id, black_player_id))

    return games


def delete_games_by_players(
    session: Session, white_player_id: int, black_player_id: int
) -> None:
    session.query(Game).filter(
        (Game.white_player_id == white_player_id)
        & (Game.black_player_id == black_player_id)
    ).delete(synchronize_session=False)

    session.commit()
    logger.info(
        "Deleted games",
        white_player_id=white_player_id,
        black_player_id=black_player_id,
    )


def get_move_dict(moves: List[Move]) -> Dict[int, Move]:
    move_dict: Dict[int, Move] = {}

    for move in moves:
        move_dict[move.ply_index] = move

    return move_dict
