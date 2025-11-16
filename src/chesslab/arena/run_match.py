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
    get_or_create_random_player,
    get_or_create_stockfish_player,
)
from chesslab.arena.run_game import run_game
from chesslab.storage import (
    Game,
    get_or_create_games,
    get_session,
)
from chesslab.storage.move_tools import delete_moves_not_played

logger = structlog.get_logger()


async def run_multiple_games(
    session: Session, games: List[Game], max_concurrent: int = 10
):
    """Run games asynchronously with controlled concurrency.

    Args:
        session: Database session
        games: List of games to run
        max_concurrent: Maximum number of concurrent game updates
    """
    logger.info("Starting async game runner", total_games=len(games))
    games_to_complete = [game for game in games if not game.result]

    logger.info(
        "Games to complete",
        count=len(games_to_complete),
        finished_count=len(games) - len(games_to_complete),
    )

    for game in games_to_complete:
        delete_moves_not_played(session=session, game=game)

    iteration = 0
    while len(games_to_complete):
        iteration += 1
        logger.debug(
            "Running game iteration",
            iteration=iteration,
            games_pending=len(games_to_complete),
        )

        # Create tasks for all games, but limit concurrency with a semaphore
        semaphore = asyncio.Semaphore(max_concurrent)

        async def update_with_semaphore(game: Game):
            async with semaphore:
                try:
                    return await run_game(session=session, game=game)

                except Exception as e:
                    logger.error(
                        "Error updating game board",
                        game_id=game.id,
                        iteration=iteration,
                        error=str(e),
                        exc_info=True,
                    )
                    return None

        # Run all game updates concurrently with controlled parallelism
        await asyncio.gather(
            *[update_with_semaphore(game) for game in games_to_complete]
        )

        previous_count = len(games_to_complete)
        games_to_complete = [game for game in games_to_complete if not game.result]
        completed_in_iteration = previous_count - len(games_to_complete)

        logger.info(
            "Iteration completed",
            iteration=iteration,
            completed_in_iteration=completed_in_iteration,
            remaining=len(games_to_complete),
        )

    logger.info(
        "All games completed",
        total_iterations=iteration,
        total_games=len(games),
    )


def get_or_create_match(
    session: Session,
    white_player_id: int,
    black_player_id: int,
    num_games: int = 1,
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

    if alternate_color:
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
    logger.info("Starting match runner script")
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    )

    with get_session() as session:
        logger.info("Database session created")

        logger.info("Creating Stockfish player")
        white_player = get_or_create_stockfish_player(
            session=session,
            elo=1320,
        )
        logger.info("White player ready", player_id=white_player.id)

        logger.info("Creating random player")
        black_player = get_or_create_random_player(
            session=session,
            seed=2,
        )
        logger.info("Black player ready", player_id=black_player.id)

        logger.info("Setting up match")
        games = get_or_create_match(
            session=session,
            white_player_id=white_player.id,
            black_player_id=black_player.id,
            num_games=10,
        )
        logger.info("Match setup complete", game_count=len(games))

        logger.info("Running games")
        asyncio.run(run_multiple_games(session=session, games=games))
        logger.info("Games finished", game_count=len(games))
