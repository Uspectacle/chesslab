"""Database access layer with CRUD operations and query helpers.

Provides high-level functions for interacting with the database,
abstracting away SQLAlchemy details for common operations.
"""

from datetime import datetime
from typing import Dict, List, Optional

import chess
import structlog
from sqlalchemy import desc
from sqlalchemy.orm import Session

from chesslab.storage.schema import Game, Move

logger = structlog.get_logger()


def create_game(
    session: Session,
    white_player_id: int,
    black_player_id: int,
    opening_fen: Optional[str] = None,
) -> Game:
    """Create a new chess game in the database.

    Args:
        session: Database session
        white_player_id: ID of the white player
        black_player_id: ID of the black player
        opening_fen: Optional FEN string for custom starting position

    Returns:
        Created Game object

    Raises:
        ValueError: If players are the same or don't exist
    """
    logger.info(
        "Creating game",
        white_player_id=white_player_id,
        black_player_id=black_player_id,
        has_opening_fen=opening_fen is not None,
    )

    if opening_fen:
        try:
            chess.Board(opening_fen)
        except ValueError as e:
            raise ValueError(f"Invalid opening FEN: {e}")

    game = Game(
        white_player_id=white_player_id,
        black_player_id=black_player_id,
        started_at=datetime.now(),
        game_metadata={"opening_fen": opening_fen} if opening_fen else {},
    )
    session.add(game)
    session.commit()
    session.refresh(game)

    logger.info("Game created successfully", game_id=game.id)
    return game


def get_game(session: Session, game_id: int) -> Optional[Game]:
    logger.debug("Fetching game", game_id=game_id)
    game = session.query(Game).filter(Game.id == game_id).first()
    if not game:
        logger.warning("Game not found", game_id=game_id)
    else:
        logger.debug("Game retrieved", game_id=game_id, result=game.result)
    return game


def get_player_games(session: Session, player_id: int, limit: int = 100) -> List[Game]:
    logger.debug("Fetching player games", player_id=player_id, limit=limit)
    games = (
        session.query(Game)
        .filter(
            (Game.white_player_id == player_id) | (Game.black_player_id == player_id)
        )
        .order_by(desc(Game.started_at))
        .limit(limit)
        .all()
    )
    logger.info("Retrieved player games", player_id=player_id, game_count=len(games))
    return games


def get_head_to_head_games(
    session: Session, player1_id: int, player2_id: int
) -> List[Game]:
    logger.debug(
        "Fetching head-to-head games", player1_id=player1_id, player2_id=player2_id
    )
    games = (
        session.query(Game)
        .filter(
            (
                (Game.white_player_id == player1_id)
                & (Game.black_player_id == player2_id)
            )
            | (
                (Game.white_player_id == player2_id)
                & (Game.black_player_id == player1_id)
            )
        )
        .all()
    )
    logger.info(
        "Retrieved head-to-head games",
        player1_id=player1_id,
        player2_id=player2_id,
        game_count=len(games),
    )
    return games


def get_games_by_players(
    session: Session, white_player_id: int, black_player_id: int
) -> List[Game]:
    logger.debug(
        "Fetching games by players",
        white_player_id=white_player_id,
        black_player_id=black_player_id,
    )
    games = (
        session.query(Game)
        .filter(
            (
                (Game.white_player_id == white_player_id)
                & (Game.black_player_id == black_player_id)
            )
        )
        .all()
    )
    logger.info(
        "Retrieved games by players",
        white_player_id=white_player_id,
        black_player_id=black_player_id,
        game_count=len(games),
    )
    return games


def get_or_create_games(
    session: Session,
    white_player_id: int,
    black_player_id: int,
    num_games: int = 1,
    remove_existing: bool = True,
    get_existing: bool = True,
) -> List[Game]:
    """Get existing games or create new ones for a match.

    Args:
        session: Database session
        white_player_id: ID of the white player
        black_player_id: ID of the black player
        num_games: Number of games to ensure exist
        remove_existing: Delete existing games before creating
        get_existing: Retrieve existing games if available

    Returns:
        List of Game objects
    """
    logger.info(
        "Getting or creating games",
        white_player_id=white_player_id,
        black_player_id=black_player_id,
        num_games=num_games,
        remove_existing=remove_existing,
        get_existing=get_existing,
    )

    if remove_existing:
        logger.debug(
            "Removing existing games",
            white_player_id=white_player_id,
            black_player_id=black_player_id,
        )
        delete_games_by_players(
            session=session,
            white_player_id=white_player_id,
            black_player_id=black_player_id,
        )

    games: List[Game] = []

    if get_existing:
        logger.debug(
            "Retrieving existing games",
            white_player_id=white_player_id,
            black_player_id=black_player_id,
        )
        games = get_games_by_players(
            session=session,
            white_player_id=white_player_id,
            black_player_id=black_player_id,
        )

    games_needed = num_games - len(games)
    if games_needed > 0:
        logger.info(
            "Creating additional games",
            games_needed=games_needed,
            current_count=len(games),
        )

    while len(games) < num_games:
        game = create_game(session, white_player_id, black_player_id)
        games.append(game)
        logger.debug(
            "Game added", game_id=game.id, current_total=len(games), target=num_games
        )

    logger.info(
        "Get or create games completed",
        white_player_id=white_player_id,
        black_player_id=black_player_id,
        final_count=len(games),
    )
    return games


def delete_games_by_players(
    session: Session, white_player_id: int, black_player_id: int
) -> None:
    logger.info(
        "Deleting games",
        white_player_id=white_player_id,
        black_player_id=black_player_id,
    )
    deleted_count = (
        session.query(Game)
        .filter(
            (Game.white_player_id == white_player_id)
            & (Game.black_player_id == black_player_id)
        )
        .delete(synchronize_session=False)
    )

    session.commit()
    logger.info(
        "Deleted games successfully",
        white_player_id=white_player_id,
        black_player_id=black_player_id,
        deleted_count=deleted_count,
    )


def get_move_dict(moves: List[Move]) -> Dict[int, Move]:
    logger.debug("Building move dictionary", move_count=len(moves))
    move_dict: Dict[int, Move] = {}

    for move in moves:
        move_dict[move.ply_index] = move

    logger.debug("Move dictionary built", dictionary_size=len(move_dict))
    return move_dict
