"""Tournament runner and game coordinator with async support.

Manages parallel game execution, player registration, and result persistence.
Supports concurrent games with configurable limits to avoid resource exhaustion.
"""

import asyncio
import logging
from typing import List

import structlog
from sqlalchemy.orm import Session

from chesslab.arena.init_engines import (
    get_random_player,
    get_stockfish_player,
)
from chesslab.arena.run_game import run_game
from chesslab.storage import (
    Game,
    delete_moves_not_played,
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

    for game in games_to_complete:
        delete_moves_not_played(session=session, game=game)

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
    logger.info(
        "Setting up match",
        white_player_id=white_player_id,
        black_player_id=black_player_id,
        num_games=num_games,
        alternate_colors=alternate_color,
        remove_existing=remove_existing,
        get_existing=get_existing,
    )

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
    )
    return games


if __name__ == "__main__":
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    )
    logger.info("Starting match runner script")

    with get_session() as session:
        # white_player = get_random_player(session=session)
        white_player = get_stockfish_player(session=session)
        black_player = get_random_player(session=session)

        games = get_or_create_match(
            session=session,
            white_player_id=white_player.id,
            black_player_id=black_player.id,
            num_games=10,
        )

        asyncio.run(run_multiple_games(session=session, games=games))
        for game in games:
            print(game.result)
