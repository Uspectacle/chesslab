from datetime import datetime
from typing import Optional

import chess
import chess.engine
import structlog
from sqlalchemy.orm import Session

from chesslab.engines import engine_commands
from chesslab.storage import Move, Player

logger = structlog.get_logger()


async def get_protocol(player: Player) -> chess.engine.Protocol:
    """Get an async chess engine for the given player.

    Args:
        player: Player object with engine configuration

    Returns:
        An async chess engine instance
    """
    logger.info(
        "Getting engine for player",
        player_id=player.id,
        engine_type=player.engine_type,
    )
    command = engine_commands.get(player.engine_type)

    if not command:
        logger.critical(
            "Engine type not found",
            player_id=player.id,
            engine_type=player.engine_type,
        )
        raise RuntimeError(f"Unknown engine type: {player.engine_type}")

    logger.debug(
        "Starting engine process",
        player_id=player.id,
        engine_type=player.engine_type,
        command=command,
    )

    _transport, protocol = await chess.engine.popen_uci(command)

    if player.options:
        logger.debug(
            "Configuring engine options",
            player_id=player.id,
            option_count=len(player.options),
        )
        await protocol.configure(player.options)

    logger.info(
        "Engine ready",
        player_id=player.id,
        engine_type=player.engine_type,
    )

    return protocol


async def play_move(
    session: Session, move: Move, protocol: Optional[chess.engine.Protocol] = None
) -> Move:
    """Play a move by ID asynchronously, creating its own database session.

    Args:
        session: Database session
        move: Move object to play

    Returns:
        Updated move object with uci_move set
    """
    # Check if already played
    if move.uci_move:
        logger.warning("Move already played", move_id=move.id, uci_move=move.uci_move)
        return move

    # Get the correct player
    if move.is_white:
        player = move.game.white_player
    else:
        player = move.game.black_player

    logger.info("Playing move", move_id=move.id)
    board = chess.Board(move.fen_before)

    close_protocol = False

    if not protocol:
        protocol = await get_protocol(player)
        close_protocol = True
    try:
        logger.debug(
            "Starting engine calculation", move_id=move.id, fen=move.fen_before
        )

        result = await protocol.play(board, limit=chess.engine.Limit(**player.limit))

        if not result.move:
            raise ValueError("Engine failed to find move")

        logger.debug(
            "Engine move calculated",
            move_id=move.id,
            uci_move=result.move.uci(),
        )

        move.played_at = datetime.now()
        move.uci_move = result.move.uci()
        session.commit()

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
            error=str(e),
            exc_info=True,
        )
        raise
    finally:
        if close_protocol:
            try:
                await protocol.quit()
            except Exception as e:
                logger.warning("Error closing engine", error=str(e))

    return move
