import logging
from typing import Any, Dict, Optional

import chess.engine
import structlog
from sqlalchemy.orm import Session

from chesslab.storage import Player, get_session
from chesslab.storage.player_tools import get_player_by_attributes

logger = structlog.get_logger()


def stockfish_elo(depth: int) -> int:
    elo = 66 * depth + 1570

    return int(round(elo))


def get_stockfish_player(
    session: Session,
    elo: Optional[int | float] = None,
    depth: int = 10,
    create_not_raise: bool = True,
) -> Player:
    if bool(elo):
        options: Dict[str, Any] = {
            "UCI_LimitStrength": True,
            "UCI_Elo": int(elo),
        }
        logger.debug("Using ELO limit strength", elo=elo)
    else:
        options: Dict[str, Any] = {
            "UCI_LimitStrength": False,
        }
        calculated_elo = stockfish_elo(depth)
        logger.debug(
            "No ELO specified, using unlimited strength",
            calculated_elo=calculated_elo,
        )

    player = get_player_by_attributes(
        session=session,
        engine_type="Stockfish",
        expected_elo=int(elo) if elo else stockfish_elo(depth),
        options=options,
        limit=chess.engine.Limit(depth=depth),
        create_not_raise=create_not_raise,
    )

    logger.info(
        "Stockfish player ready",
        player_id=player.id,
        expected_elo=player.expected_elo,
    )

    return player


def get_random_player(
    session: Session,
    seed: Optional[int] = None,
    create_not_raise: bool = True,
) -> Player:
    logger.debug("Getting or creating random player", seed=seed)

    player = get_player_by_attributes(
        session=session,
        engine_type="RandomEngine",
        expected_elo=300,
        options={"Seed": seed} if seed else {},
        create_not_raise=create_not_raise,
    )
    logger.info(
        "Random player ready",
        player_id=player.id,
        seed=seed,
    )
    return player


if __name__ == "__main__":
    logger.info("Initializing engines script")
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
    )

    with get_session() as session:
        logger.info("Database session created")

        logger.info("Creating Stockfish player with ELO 1320")
        white_player = get_stockfish_player(
            session=session,
            elo=1320,
        )
        logger.info("White player created", player_id=white_player.id)

        logger.info("Creating random player with seed 2")
        black_player = get_random_player(
            session=session,
            seed=2,
        )
        logger.info("Black player created", player_id=black_player.id)

        print(white_player)
        print(black_player)
