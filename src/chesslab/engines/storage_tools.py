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
    logger.debug(
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


async def get_uci_move(
    board: chess.Board, player: Player, protocol: Optional[chess.engine.Protocol] = None
) -> str:
    """Play a move by ID asynchronously, creating its own database session.

    Args:
        session: Database session
        move: Move object to play

    Returns:
        Updated move object with uci_move set
    """
    close_protocol = False

    if not protocol:
        protocol = await get_protocol(player)
        close_protocol = True
    try:
        logger.debug(
            "Starting engine calculation", player_id=player.id, board=board.fen()
        )

        result = await protocol.play(board, limit=chess.engine.Limit(**player.limit))

        if not result.move:
            raise ValueError("Engine failed to find move")

        uci_move = result.move.uci()

        logger.debug(
            "Engine move calculated",
            player_id=player.id,
            board=board.fen(),
            uci_move=uci_move,
        )

    except Exception as e:
        logger.error(
            "Error playing move",
            player_id=player.id,
            board=board.fen(),
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

    return uci_move


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

    board = chess.Board(move.fen_before)

    uci_move = await get_uci_move(board=board, player=player, protocol=protocol)

    if move.uci_move:
        logger.warning("Move already played", move_id=move.id, uci_move=move.uci_move)
        return move

    if not board.is_legal(chess.Move.from_uci(uci_move)):
        logger.error("Move illegal", move_id=move.id, uci_move=uci_move)
        return move

    move.uci_move = uci_move
    move.played_at = datetime.now()
    session.commit()

    logger.info(
        "Move played",
        move_id=move.id,
        game_id=move.game_id,
        uci_move=move.uci_move,
    )

    return move
