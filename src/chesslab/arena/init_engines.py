import logging
import random
from typing import Any, Dict, Optional

import structlog
from sqlalchemy.orm import Session

from chesslab.storage.db_tools import get_session
from chesslab.storage.player_tools import get_or_create_player
from chesslab.storage.schema import Player


def stockfish_elo(depth: int) -> int:
    elo = 3500 / (1 + 30 * (2.71828 ** (-0.25 * depth)))

    return int(round(elo))


def get_or_create_stockfish_player(
    session: Session, elo: Optional[int], depth: int = 10
) -> Player:
    if bool(elo):
        options: Dict[str, Any] = {
            "UCI_LimitStrength": True,
            "UCI_Elo": int(elo),
        }
    else:
        options: Dict[str, Any] = {
            "UCI_LimitStrength": False,
        }

    return get_or_create_player(
        session=session,
        engine_type="Stockfish",
        expected_elo=elo or stockfish_elo(depth),
        options=options,
    )


def get_or_create_random_player(
    session: Session,
    seed: Optional[int],
) -> Player:
    if seed is None:
        seed = random.randint(0, 2147483647)

    return get_or_create_player(
        session=session,
        engine_type="RandomEngine",
        expected_elo=300,
        options={"Seed": seed},
    )


if __name__ == "__main__":
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
    )
    session = get_session()

    white_player = get_or_create_stockfish_player(
        session=session,
        elo=1320,
    )
    black_player = get_or_create_random_player(
        session=session,
        seed=2,
    )
