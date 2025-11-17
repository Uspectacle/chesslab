"""Database access layer with CRUD operations and query helpers.

Provides high-level functions for interacting with the database,
abstracting away SQLAlchemy details for common operations.
"""

from dataclasses import asdict
from typing import Any, Dict, List, Optional

import chess
import chess.engine
import structlog
from sqlalchemy import ColumnElement
from sqlalchemy.orm import Session

from chesslab.storage.schema import Player

logger = structlog.get_logger()


def get_player_by_id(session: Session, player_id: int) -> Optional[Player]:
    logger.debug("Fetching player by ID", player_id=player_id)
    player = session.query(Player).filter(Player.id == player_id).first()
    if not player:
        logger.warning("Player not found", player_id=player_id)
    else:
        logger.debug(
            "Player retrieved",
            player_id=player_id,
            engine_type=player.engine_type,
            expected_elo=player.expected_elo,
        )
    return player


def get_player_by_attributes(
    session: Session,
    engine_type: str,
    expected_elo: int,
    options: Optional[Dict[str, Any]] = None,
    limit: Optional[chess.engine.Limit] = None,
    create_not_raise: bool = True,
) -> Player:
    logger.debug(
        "Fetching player by attributes",
        engine_type=engine_type,
        expected_elo=expected_elo,
        has_options=options is not None,
    )
    player = Player(
        engine_type=engine_type,
        expected_elo=expected_elo,
        options=options or {},
        limit=asdict(limit or chess.engine.Limit()),
    )
    candidates = session.query(Player).filter(
        Player.engine_type == player.engine_type,
        Player.expected_elo == player.expected_elo,
    )

    for candidate in candidates:
        if candidate.options == player.options and candidate.limit == player.limit:
            logger.debug(
                "Player found by attributes",
                player_id=candidate.id,
                engine_type=engine_type,
            )
            return candidate

    if not create_not_raise:
        raise ValueError(
            f"No player found with type={engine_type} and elo={expected_elo}",
        )

    session.add(player)
    session.commit()
    session.refresh(player)

    logger.info(
        "Player created successfully",
        player_id=player.id,
        engine_type=engine_type,
        expected_elo=expected_elo,
    )
    return player


def list_players(
    session: Session,
    engine_type: Optional[str] = None,
    min_elo: Optional[int] = None,
    max_elo: Optional[int] = None,
) -> List[Player]:
    """List all players, optionally filtered by engine type.

    Args:
        session: Database session
        engine_type: Optional filter by engine type

    Returns:
        List of player instances
    """
    logger.debug(
        "Listing players",
        engine_type_filter=engine_type,
    )
    filters: List[ColumnElement[bool]] = []

    if engine_type:
        filters.append((Player.engine_type == engine_type))

    if min_elo is not None:
        filters.append((Player.expected_elo >= min_elo))

    if max_elo is not None:
        filters.append((Player.expected_elo <= max_elo))

    players = session.query(Player).filter(*filters).all()
    logger.info(
        "Retrieved players list",
        player_count=len(players),
        engine_type_filter=engine_type,
    )
    return players
