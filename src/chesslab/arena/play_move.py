from datetime import datetime

import chess
import chess.engine
import structlog
from sqlalchemy.orm import Session

from chesslab.arena.async_tools import run_async_background
from chesslab.storage import (
    Game,
    Move,
    create_move,
    get_engine,
    get_move_dict,
    get_session,
)

logger = structlog.get_logger()


async def play_move(move: Move):
    """Calculate and execute a chess move using the appropriate engine.

    Args:
        move: Move object to be calculated and played
    """
    logger.info(
        "Playing move",
        move_id=move.id,
        game_id=move.game_id,
        ply_index=move.ply_index,
        is_white=move.is_white,
    )
    engine = None

    try:
        if move.is_white:
            player = move.game.white_player
        else:
            player = move.game.black_player

        logger.debug(
            "Retrieved player for move",
            move_id=move.id,
            player_id=player.id,
            engine_type=player.engine_type,
        )

        engine = get_engine(player)
        board = chess.Board(move.fen_before)

        logger.debug(
            "Starting engine calculation",
            move_id=move.id,
            fen=move.fen_before,
        )

        result = engine.play(
            board=board, limit=chess.engine.Limit()
        )  # can be very long

        if not result.move:
            logger.error(
                "Engine failed to find move",
                move_id=move.id,
                game_id=move.game_id,
                fen=move.fen_before,
            )
            return

        logger.debug(
            "Engine move calculated",
            move_id=move.id,
            uci_move=result.move.uci(),
        )

        with get_session() as session:
            updated_move = session.get(Move, move.id)
            if not updated_move:
                logger.error("Move not found in database", move_id=move.id)
                return

            # Check if move was already played by another process
            if updated_move.uci_move:
                logger.warning(
                    "Move already played by another process",
                    move_id=move.id,
                    existing_move=updated_move.uci_move,
                )
                return

            updated_move.played_at = datetime.now()
            updated_move.uci_move = result.move.uci()

            session.commit()
            session.refresh(updated_move)

            logger.info(
                "Move played successfully",
                move_id=move.id,
                game_id=move.game_id,
                uci_move=result.move.uci(),
            )
    except Exception as e:
        logger.error(
            "Error playing move",
            move_id=move.id,
            game_id=move.game_id,
            error=str(e),
            exc_info=True,
        )
    finally:
        try:
            if engine:
                engine.quit()
        except Exception as e:
            logger.error(
                "Error during engine quit",
                engine=engine,
                move_id=move.id,
                error=str(e),
                exc_info=True,
            )


def update_and_get_board(session: Session, game: Game) -> chess.Board:
    logger.debug("Updating board for game", game_id=game.id)

    try:
        board = (
            chess.Board(game.game_metadata.get("opening_fen"))
            if game.game_metadata
            else chess.Board()
        )
        logger.debug(
            "Board initialized",
            game_id=game.id,
            has_opening_fen=bool(
                game.game_metadata and game.game_metadata.get("opening_fen")
            ),
        )
    except Exception as e:
        logger.error(
            "Failed to initialize board",
            game_id=game.id,
            error=str(e),
            exc_info=True,
        )
        raise

    if board.is_game_over():
        logger.info("Game already over", game_id=game.id, result=board.result())
        return board

    move_dict = get_move_dict(game.moves)

    while not board.is_game_over():
        ply_index = board.ply() + 1
        move = move_dict.get(board.ply() + 1)

        if not move:
            logger.info(
                "Creating new move",
                game_id=game.id,
                ply_index=ply_index,
                move_number=board.fullmove_number,
                is_white=(board.turn == chess.WHITE),
            )
            move = create_move(
                session=session,
                game_id=game.id,
                ply_index=ply_index,
                move_number=board.fullmove_number,
                is_white=(board.turn == chess.WHITE),
                fen_before=board.fen(),
            )

            logger.debug("Scheduling async move playback", move_id=move.id)
            run_async_background(play_move(move=move))
            return board

        if not move.uci_move:
            logger.info(
                "Move pending engine calculation",
                game_id=game.id,
                move_id=move.id,
                ply_index=ply_index,
            )
            return board

        try:
            board.push(chess.Move.from_uci(move.uci_move))
            logger.debug(
                "Move applied to board",
                game_id=game.id,
                uci_move=move.uci_move,
                ply_index=ply_index,
            )
        except ValueError as e:
            logger.error(
                "Invalid move",
                game_id=game.id,
                move_id=move.id,
                uci_move=move.uci_move,
                error=str(e),
            )
            raise e

    game.result = board.result()
    game.finished_at = datetime.now()
    session.commit()
    session.refresh(game)

    logger.info(
        "Game finished",
        game_id=game.id,
        result=game.result,
        move_count=len(game.moves),
    )

    return board
