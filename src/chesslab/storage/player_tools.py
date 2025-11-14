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
) -> Optional[Player]:
    logger.debug(
        "Fetching player by attributes",
        engine_type=engine_type,
        expected_elo=expected_elo,
        has_options=options is not None,
    )
    player = Player(
        engine_type=engine_type, expected_elo=expected_elo, options=options or {}
    )
    result = (
        session.query(Player)
        .filter(
            Player.engine_type == player.engine_type,
            Player.expected_elo == player.expected_elo,
            cast(Player.options, String)
            == cast(json.dumps(options, sort_keys=True), String),
        )
        .first()
    )

    if result:
        logger.debug(
            "Player found by attributes",
            player_id=result.id,
            engine_type=engine_type,
        )
    else:
        logger.debug(
            "No player found with attributes",
            engine_type=engine_type,
            expected_elo=expected_elo,
        )

    return result


def create_player(
    session: Session,
    engine_type: str,
    expected_elo: int,
    options: Optional[Dict[str, Any]] = None,
) -> Player:
    logger.info(
        "Creating player",
        engine_type=engine_type,
        expected_elo=expected_elo,
        has_options=options is not None,
    )
    player = Player(
        engine_type=engine_type, expected_elo=expected_elo, options=options or {}
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


def get_or_create_player(
    session: Session,
    engine_type: str,
    expected_elo: int,
    options: Optional[Dict[str, Any]] = None,
) -> Player:
    logger.info(
        "Getting or creating player",
        engine_type=engine_type,
        expected_elo=expected_elo,
    )
    player = get_player_by_attributes(
        session=session,
        engine_type=engine_type,
        expected_elo=expected_elo,
        options=options,
    )

    if player:
        logger.info(
            "Player already exists",
            player_id=player.id,
            engine_type=engine_type,
        )
        return player

    logger.info(
        "Creating new player",
        engine_type=engine_type,
        expected_elo=expected_elo,
    )
    return create_player(session, engine_type, expected_elo, options)


def get_engine(player: Player) -> chess.engine.SimpleEngine:
    logger.info(
        "Getting engine for player",
        player_id=player.id,
        engine_type=player.engine_type,
    )
    command = engine_commands.get(player.engine_type)

    if not command:
        logger.critical(
            "Engine type not found",
            player_id=player.id,
            engine_type=player.engine_type,
        )
        raise RuntimeError(f"Unkown engine type: {player.engine_type}")

    logger.debug(
        "Starting engine process",
        player_id=player.id,
        engine_type=player.engine_type,
        command=command,
    )

    engine = chess.engine.SimpleEngine.popen_uci(command)

    if player.options:
        logger.debug(
            "Configuring engine options",
            player_id=player.id,
            option_count=len(player.options),
        )

    engine.configure(player.options)

    logger.info(
        "Engine ready",
        player_id=player.id,
        engine_type=player.engine_type,
    )

    return engine


def list_players(session: Session, engine_type: Optional[str] = None) -> List[Player]:
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
    query = session.query(Player)

    if engine_type:
        logger.debug("Applying engine type filter", engine_type=engine_type)
        query = query.filter(Player.engine_type == engine_type)

    players = query.all()
    logger.info(
        "Retrieved players list",
        player_count=len(players),
        engine_type_filter=engine_type,
    )
    return players
