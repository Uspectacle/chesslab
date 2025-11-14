"""Database access layer with CRUD operations and query helpers.

Provides high-level functions for interacting with the database,
abstracting away SQLAlchemy details for common operations.
"""

import json
from typing import Any, Dict, List, Optional

import chess
import chess.engine
import structlog
from sqlalchemy import String, cast
from sqlalchemy.orm import Session

from chesslab.engines import engine_commands
from chesslab.storage.schema import Player

logger = structlog.get_logger()


def get_player_by_id(session: Session, player_id: int) -> Optional[Player]:
    return session.query(Player).filter(Player.id == player_id).first()


def get_player_by_attributes(
    session: Session,
    engine_type: str,
    expected_elo: int,
    options: Optional[Dict[str, Any]] = None,
) -> Optional[Player]:
    player = Player(
        engine_type=engine_type, expected_elo=expected_elo, options=options or {}
    )
    return (
        session.query(Player)
        .filter(
            Player.engine_type == player.engine_type,
            Player.expected_elo == player.expected_elo,
            cast(Player.options, String)
            == cast(json.dumps(options, sort_keys=True), String),
        )
        .first()
    )


def create_player(
    session: Session,
    engine_type: str,
    expected_elo: int,
    options: Optional[Dict[str, Any]] = None,
) -> Player:
    player = Player(
        engine_type=engine_type, expected_elo=expected_elo, options=options or {}
    )
    session.add(player)
    session.commit()
    session.refresh(player)

    return player


def get_or_create_player(
    session: Session,
    engine_type: str,
    expected_elo: int,
    options: Optional[Dict[str, Any]] = None,
) -> Player:
    player = get_player_by_attributes(
        session=session,
        engine_type=engine_type,
        expected_elo=expected_elo,
        options=options,
    )

    return player or create_player(session, engine_type, expected_elo, options)


def get_engine(player: Player) -> chess.engine.SimpleEngine:
    command = engine_commands.get(player.engine_type)

    if not command:
        raise RuntimeError(f"Unkown engine type: {player.engine_type}")

    engine = chess.engine.SimpleEngine.popen_uci(command)

    engine.configure(player.options)

    return engine


def list_players(session: Session, engine_type: Optional[str] = None) -> List[Player]:
    """List all players, optionally filtered by engine type.

    Args:
        session: Database session
        engine_type: Optional filter by engine type

    Returns:
        List of player instances
    """
    query = session.query(Player)

    if engine_type:
        query = query.filter(Player.engine_type == engine_type)

    return query.all()
