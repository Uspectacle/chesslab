"""Tournament runner and game coordinator with async support.

Manages parallel game execution, player registration, and result persistence.
Supports concurrent games with configurable limits to avoid resource exhaustion.
"""

import asyncio
import logging
import random
from typing import List, Optional

import structlog
from sqlalchemy.orm import Session

from chesslab.arena.run_game import run_game
from chesslab.engines.init_engines import (
    Player,
    get_random_player,
    get_stockfish_range,
)
from chesslab.storage import (
    Game,
    get_or_create_games,
    get_session,
)

logger = structlog.get_logger()


async def run_multiple_games(
    session: Session, games: List[Game], max_concurrent: int = 8
):
    """Run games asynchronously with controlled concurrency."""
    logger.debug("Starting async game runner", total_games=len(games))

    games_to_complete = [game for game in games if not game.result]
    logger.info("Games to complete", count=len(games_to_complete))
    random.shuffle(games_to_complete)

    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_limit(game: Game):
        async with semaphore:
            try:
                return await run_game(session=session, game=game)
            except Exception as e:
                logger.error("Error running game", game_id=game.id, error=str(e))
                return game

    await asyncio.gather(*[run_with_limit(game) for game in games_to_complete])

    logger.info("All games completed", total_games=len(games))


def get_or_create_match(
    session: Session,
    white_player_id: int,
    black_player_id: int,
    num_games: int = 100,
    remove_existing: bool = True,
    get_existing: bool = True,
    alternate_color: bool = True,
) -> List[Game]:
    """Create or retrieve games for a match between two players.

    Args:
        session: Database session
        white_player_id: ID of the white player
        black_player_id: ID of the black player
        num_games: Total number of games to create/retrieve
        remove_existing: Whether to delete existing games before creating new ones
        get_existing: Whether to retrieve existing games
        alternate_color: Whether to alternate colors between games

    Returns:
        List of Game objects
    """
    games: List[Game] = []

    if alternate_color and white_player_id != black_player_id:
        logger.debug(
            "Alternating colors",
            first_color_games=(num_games // 2) + (num_games % 2),
            second_color_games=num_games // 2,
        )
        games = get_or_create_games(
            session=session,
            white_player_id=white_player_id,
            black_player_id=black_player_id,
            num_games=(num_games // 2) + (num_games % 2),
            remove_existing=remove_existing,
            get_existing=get_existing,
        ) + get_or_create_games(
            session=session,
            white_player_id=black_player_id,
            black_player_id=white_player_id,
            num_games=num_games // 2,
            remove_existing=remove_existing,
            get_existing=get_existing,
        )
    else:
        logger.debug("Not alternating colors")
        games = get_or_create_games(
            session=session,
            white_player_id=white_player_id,
            black_player_id=black_player_id,
            num_games=num_games,
            remove_existing=remove_existing,
            get_existing=get_existing,
        )

    logger.info(
        "Match setup complete",
        white_player_id=white_player_id,
        black_player_id=black_player_id,
        total_games=len(games),
        alternate_colors=alternate_color,
        remove_existing=remove_existing,
        get_existing=get_existing,
    )
    return games


def run_range(
    session: Session,
    players: List[Player],
    opponents: Optional[List[Player]] = None,
    num_games: int = 10,
    remove_existing: bool = True,
    get_existing: bool = True,
    alternate_color: bool = True,
    max_concurrent: int = 8,
) -> List[Game]:
    games: List[Game] = []

    if not opponents:
        opponents = get_stockfish_range(session)

    for opponent in opponents:
        for player in players:
            games += get_or_create_match(
                session=session,
                white_player_id=player.id,
                black_player_id=opponent.id,
                num_games=num_games,
                remove_existing=remove_existing,
                get_existing=get_existing,
                alternate_color=alternate_color,
            )

    asyncio.run(
        run_multiple_games(session=session, games=games, max_concurrent=max_concurrent)
    )

    return games


if __name__ == "__main__":
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )
    logger.info("Starting match runner script")

    with get_session() as session:
        games = run_range(
            session=session,
            players=[get_random_player(session=session)],
        )

        for game in games:
            print(game.result)
