"""Database access layer with CRUD operations and query helpers.

Provides high-level functions for interacting with the database,
abstracting away SQLAlchemy details for common operations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

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
    game_round: Optional[int] = None,
    guess_round: bool = True,
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

    game_metadata: Dict[str, Any] = {}

    if opening_fen:
        try:
            chess.Board(opening_fen)
        except ValueError as e:
            raise ValueError(f"Invalid opening FEN: {e}")

        game_metadata["opening_fen"] = opening_fen

    if game_round:
        game_metadata["round"] = game_round
    elif guess_round:
        played_rounds = get_head_to_head_games(
            session=session,
            player1_id=white_player_id,
            player2_id=black_player_id,
        )
        game_metadata["round"] = len(played_rounds) + 1

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
    logger.debug(
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
    logger.debug(
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

    while len(games) < num_games:
        game = create_game(session, white_player_id, black_player_id)
        games.append(game)
        logger.debug(
            "Game added", game_id=game.id, current_total=len(games), target=num_games
        )

    logger.debug(
        "Get or create games completed",
        white_player_id=white_player_id,
        black_player_id=black_player_id,
        final_count=len(games),
    )
    return games


def delete_games_by_players(
    session: Session, white_player_id: int, black_player_id: int
) -> None:
    logger.debug(
        "Deleting games",
        white_player_id=white_player_id,
        black_player_id=black_player_id,
    )

    games = (
        session.query(Game)
        .filter(
            (Game.white_player_id == white_player_id)
            & (Game.black_player_id == black_player_id)
        )
        .all()
    )

    if not len(games):
        return

    for game in games:
        session.delete(game)

    session.commit()

    logger.info(
        "Deleted games successfully",
        white_player_id=white_player_id,
        black_player_id=black_player_id,
        deleted_count=len(games),
    )


def get_move_dict(moves: List[Move]) -> Dict[int, Move]:
    move_dict: Dict[int, Move] = {}

    for move in moves:
        move_dict[move.ply_index] = move

    return move_dict


def get_board(game: Game) -> chess.Board:
    opening_fen = game.game_metadata.get("opening_fen")
    board = chess.Board(opening_fen) if opening_fen else chess.Board()

    move_dict = get_move_dict(game.moves)

    while not board.is_game_over():
        ply_index = board.ply() + 1
        move = move_dict.get(ply_index)

        if not move or not move.uci_move:
            return board

        board.push(chess.Move.from_uci(move.uci_move))

    logger.debug(
        "Board extracted",
        game_id=game.id,
        ply=board.ply(),
        fen=board.fen(),
    )
    return board
