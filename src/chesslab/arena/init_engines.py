import logging
import random
from typing import Any, Dict, Optional

import structlog
from sqlalchemy.orm import Session

from chesslab.storage import Player, get_or_create_player, get_session

logger = structlog.get_logger()


def stockfish_elo(depth: int) -> int:
    logger.debug("Calculating Stockfish ELO", depth=depth)
    elo = 3500 / (1 + 30 * (2.71828 ** (-0.25 * depth)))
    calculated_elo = int(round(elo))
    logger.debug("Calculated Stockfish ELO", depth=depth, elo=calculated_elo)
    return calculated_elo


def get_or_create_stockfish_player(
    session: Session, elo: Optional[int], depth: int = 10
) -> Player:
    logger.info(
        "Getting or creating Stockfish player",
        elo=elo,
        depth=depth,
        use_limit_strength=bool(elo),
    )

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

    player = get_or_create_player(
        session=session,
        engine_type="Stockfish",
        expected_elo=elo or stockfish_elo(depth),
        options=options,
    )
    logger.info(
        "Stockfish player ready",
        player_id=player.id,
        expected_elo=player.expected_elo,
    )
    return player


def get_or_create_random_player(
    session: Session,
    seed: Optional[int],
) -> Player:
    logger.info("Getting or creating random player", seed=seed)

    if seed is None:
        seed = random.randint(0, 2147483647)
        logger.debug("Generated random seed", seed=seed)
    else:
        logger.debug("Using provided seed", seed=seed)

    player = get_or_create_player(
        session=session,
        engine_type="RandomEngine",
        expected_elo=300,
        options={"Seed": seed},
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
        white_player = get_or_create_stockfish_player(
            session=session,
            elo=1320,
        )
        logger.info("White player created", player_id=white_player.id)

        logger.info("Creating random player with seed 2")
        black_player = get_or_create_random_player(
            session=session,
            seed=2,
        )
        logger.info("Black player created", player_id=black_player.id)

        print(white_player)
        print(black_player)
