import asyncio
import logging
from typing import List

import numpy as np
import structlog
from sqlalchemy.orm import Session

from chesslab.arena.init_engines import (
    get_random_player,
    get_stockfish_player,
)
from chesslab.arena.run_match import get_or_create_match, run_multiple_games
from chesslab.storage import Game, Player, get_session

logger = structlog.get_logger()


def run_stockfish_range(
    session: Session,
    player: Player,
    min_elo: int = 1320,
    max_elo: int = 2200,
    num_step: int = 3,
    num_games: int = 10,
    remove_existing: bool = True,
    get_existing: bool = True,
    alternate_color: bool = True,
    max_concurrent: int = 10,
):
    logger.info("Setting up all matchs")
    games: List[Game] = []
    for elo in np.linspace(min_elo, max_elo, num_step):
        stockfish = get_stockfish_player(
            session=session,
            elo=elo,
        )
        games += get_or_create_match(
            session=session,
            white_player_id=player.id,
            black_player_id=stockfish.id,
            num_games=num_games,
            remove_existing=remove_existing,
            get_existing=get_existing,
            alternate_color=alternate_color,
        )
    logger.info("Running games", game_count=len(games))
    asyncio.run(
        run_multiple_games(session=session, games=games, max_concurrent=max_concurrent)
    )
    logger.info("Games finished", game_count=len(games))


if __name__ == "__main__":
    logger.info("Starting match runner script")
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )

    with get_session() as session:
        player = get_random_player(session=session)
        run_stockfish_range(session=session, player=player)
