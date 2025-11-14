"""Tournament runner and game coordinator.

Manages parallel game execution, player registration, and result persistence.
Supports concurrent games with configurable limits to avoid resource exhaustion.
"""

import logging
from datetime import datetime
from typing import List

import chess
import structlog
from sqlalchemy.orm import Session

from chesslab.arena.init_engines import (
    get_or_create_random_player,
    get_or_create_stockfish_player,
)
from chesslab.storage import (
    Game,
    create_move,
    get_move_dict,
    get_or_create_games,
    get_session,
)

logger = structlog.get_logger()


def update_and_get_board(session: Session, game: Game) -> chess.Board:
    board = (
        chess.Board(game.game_metadata.get("opening_fen"))
        if game.game_metadata
        else chess.Board()
    )

    if board.is_game_over():
        return board

    move_dict = get_move_dict(game.moves)

    while not board.is_game_over():
        ply_index = board.ply() + 1
        move = move_dict.get(board.ply() + 1)

        if not move:
            create_move(
                session=session,
                game_id=game.id,
                ply_index=ply_index,
                move_number=board.fullmove_number,
                is_white=(board.turn == chess.WHITE),
                fen_before=board.fen(),
            )
            return board

        if not move.uci_move:
            return board

        board.push(chess.Move.from_uci(move.uci_move))

    game.result = board.result()
    game.finished_at = datetime.now()
    session.commit()
    session.refresh(game)

    return board


def run_games(session: Session, games: List[Game]):
    game_to_complete = [game for game in games if not game.result]

    while len(game_to_complete):
        for game in game_to_complete:
            update_and_get_board(session=session, game=game)

        game_to_complete = [game for game in game_to_complete if not game.result]


def get_or_create_match(
    session: Session,
    white_player_id: int,
    black_player_id: int,
    num_games: int = 1,
    remove_exiting: bool = True,
    get_exiting: bool = True,
    alernate_color: bool = True,
) -> List[Game]:
    if alernate_color:
        games = get_or_create_games(
            session=session,
            white_player_id=white_player_id,
            black_player_id=black_player_id,
            num_games=(num_games // 2) + (num_games % 2),
            remove_exiting=remove_exiting,
            get_exiting=get_exiting,
        ) + get_or_create_games(
            session=session,
            white_player_id=black_player_id,
            black_player_id=white_player_id,
            num_games=num_games // 2,
            remove_exiting=remove_exiting,
            get_exiting=get_exiting,
        )
    else:
        games = get_or_create_games(
            session=session,
            white_player_id=white_player_id,
            black_player_id=black_player_id,
            num_games=num_games,
            remove_exiting=remove_exiting,
            get_exiting=get_exiting,
        )

    return games


if __name__ == "__main__":
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
    )
    session = get_session()

    white_player = get_or_create_stockfish_player(
        session=session,
        elo=1320,
    )
    black_player = get_or_create_random_player(
        session=session,
        seed=2,
    )

    games = get_or_create_match(
        session=session,
        white_player_id=white_player.id,
        black_player_id=black_player.id,
        num_games=10,
    )

    run_games(session=session, games=games)

    print(games)
