import asyncio
import logging
from datetime import datetime
from typing import Optional

import chess
import chess.engine
import structlog
from sqlalchemy.orm import Session

from chesslab.arena.init_engines import (
    get_random_player,
    get_stockfish_player,
)
from chesslab.arena.play_move import get_protocol, play_move
from chesslab.storage import (
    Game,
    create_move,
    get_move_dict,
    get_session,
)
from chesslab.storage.game_tools import create_game, get_board

logger = structlog.get_logger()


async def run_game(
    session: Session,
    game: Game,
    white_protocol: Optional[chess.engine.Protocol] = None,
    black_protocol: Optional[chess.engine.Protocol] = None,
) -> Game:
    """Creates moves and schedules them asynchronously."""
    logger.info("Playing game", game_id=game.id)

    board = get_board(game)

    if board.is_game_over():
        logger.info("Game already over", game_id=game.id, result=board.result())
        return game

    move_dict = get_move_dict(game.moves)
    close_white_protocol = False
    close_black_protocol = False

    if not white_protocol:
        white_protocol = await get_protocol(game.white_player)
        close_white_protocol = True
    if not black_protocol:
        black_protocol = await get_protocol(game.black_player)
        close_black_protocol = True

    try:
        while not board.is_game_over():
            ply_index = board.ply() + 1
            move = move_dict.get(ply_index)

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

                # Get the correct protocol
                if move.is_white:
                    protocol = white_protocol
                else:
                    protocol = black_protocol

                logger.debug("Scheduling async move playback", move_id=move.id)
                move = await play_move(session=session, move=move, protocol=protocol)

            if not move.uci_move:
                raise ValueError(f"Error while computing move {move.id}")

            board.push(chess.Move.from_uci(move.uci_move))

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
    except Exception as e:
        logger.error(
            "Error playing game",
            move_id=game.id,
            error=str(e),
            exc_info=True,
        )
        raise
    finally:
        if close_white_protocol:
            try:
                await white_protocol.quit()
            except Exception as e:
                logger.warning("Error closing white engine", error=str(e))
        if close_black_protocol:
            try:
                await black_protocol.quit()
            except Exception as e:
                logger.warning("Error closing black engine", error=str(e))
    return game


if __name__ == "__main__":
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    )

    with get_session() as session:
        white_player = get_stockfish_player(
            session=session,
            elo=1320,
        )

        black_player = get_random_player(session=session)

        game = create_game(
            session=session,
            white_player_id=white_player.id,
            black_player_id=black_player.id,
        )
        asyncio.run(run_game(session=session, game=game))
