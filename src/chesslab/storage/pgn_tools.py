"""PGN import and export utilities.

Provides functions to import games from PGN files and export games
from the database to PGN format for analysis in other chess software.
"""

from datetime import datetime
from typing import List, Optional

import chess
import chess.pgn
import structlog
from sqlalchemy.orm import Session

from chesslab.storage.db_tools import create_db_engine, create_session
from chesslab.storage.schema import Game, Move, Player

logger = structlog.get_logger()


def export_game_to_pgn(
    session: Session, game_id: int, include_evaluations: bool = False
) -> Optional[chess.pgn.Game]:
    """Export a game from database to PGN format.

    Args:
        session: Database session
        game_id: Game ID to export
        include_evaluations: If True, add evaluation comments (future feature)

    Returns:
        chess.pgn.Game object or None if game not found
    """
    game = Game.get_game(session, game_id)
    if not game:
        logger.error("Game not found", game_id=game_id)
        return None

    moves = Move.get_game_moves(session, game_id)
    white_player = Player.get_player(session, game.white_player_id)
    black_player = Player.get_player(session, game.black_player_id)

    # Create PGN game
    pgn_game = chess.pgn.Game()

    # Set headers
    pgn_game.headers["Event"] = game.metadata.get("event", "ChessLab Game")
    pgn_game.headers["Site"] = game.metadata.get("site", "ChessLab")
    pgn_game.headers["Date"] = game.started_at.strftime("%Y.%m.%d")
    pgn_game.headers["Round"] = str(game.metadata.get("round", "-"))
    pgn_game.headers["White"] = white_player.engine_type if white_player else "Unknown"
    pgn_game.headers["Black"] = black_player.engine_type if black_player else "Unknown"
    pgn_game.headers["Result"] = game.result or "*"

    # Add custom headers
    pgn_game.headers["WhitePlayerID"] = str(game.white_player_id)
    pgn_game.headers["BlackPlayerID"] = str(game.black_player_id)
    pgn_game.headers["GameID"] = str(game_id)

    if game.finished_at:
        duration = (game.finished_at - game.started_at).total_seconds()
        pgn_game.headers["Duration"] = f"{int(duration)}s"

    # Add opening FEN if not standard
    if game.metadata.get("opening_fen"):
        pgn_game.headers["FEN"] = game.metadata["opening_fen"]
        pgn_game.headers["SetUp"] = "1"

    # Build move list
    board = (
        chess.Board(game.metadata.get("opening_fen"))
        if game.metadata.get("opening_fen")
        else chess.Board()
    )
    node = pgn_game

    for move in moves:
        if not move.uci_move:
            logger.warning("Move without UCI string", move_id=move.id, game_id=game_id)
            continue

        try:
            chess_move = chess.Move.from_uci(move.uci_move)
            node = node.add_variation(chess_move)
            board.push(chess_move)

            # Add evaluation comment if requested (placeholder for future feature)
            if include_evaluations:
                # Future: Add evaluation data from Evaluation table
                pass

        except Exception as e:
            logger.error(
                "Failed to add move to PGN",
                move_id=move.id,
                uci_move=move.uci_move,
                error=str(e),
            )

    logger.info("Game exported to PGN", game_id=game_id, moves=len(moves))
    return pgn_game


def export_games_to_file(
    session: Session,
    output_file: str,
    game_ids: Optional[List[int]] = None,
    player_id: Optional[int] = None,
    limit: Optional[int] = None,
) -> int:
    """Export multiple games to a PGN file.

    Args:
        session: Database session
        output_file: Path to output PGN file
        game_ids: Specific game IDs to export (None = export all)
        player_id: Filter by player ID (None = all players)
        limit: Maximum number of games to export

    Returns:
        Number of games exported
    """

    # Build query
    query = session.query(Game)

    if game_ids:
        query = query.filter(Game.id.in_(game_ids))

    if player_id:
        query = query.filter(
            (Game.white_player_id == player_id) | (Game.black_player_id == player_id)
        )

    if limit:
        query = query.limit(limit)

    games = query.all()

    logger.info(
        "Exporting games to PGN", output_file=output_file, game_count=len(games)
    )

    exported_count = 0

    with open(output_file, "w") as f:
        for game in games:
            pgn_game = export_game_to_pgn(session, game.id)
            if pgn_game:
                print(pgn_game, file=f)
                print(file=f)  # Empty line between games
                exported_count += 1

    logger.info(
        "PGN export completed", output_file=output_file, exported=exported_count
    )

    return exported_count


def import_pgn_file(
    session: Session,
    input_file: str,
    white_player_id: Optional[int] = None,
    black_player_id: Optional[int] = None,
    create_players: bool = True,
) -> int:
    """Import games from a PGN file into the database.

    Args:
        session: Database session
        input_file: Path to PGN file
        white_player_id: Default white player ID (None = create from PGN headers)
        black_player_id: Default black player ID (None = create from PGN headers)
        create_players: If True, create player records from PGN headers

    Returns:
        Number of games imported
    """

    logger.info("Importing PGN file", input_file=input_file)

    imported_count = 0

    with open(input_file, "r") as f:
        while True:
            pgn_game = chess.pgn.read_game(f)
            if pgn_game is None:
                break

            try:
                # Get or create players
                if white_player_id is None:
                    white_name = pgn_game.headers.get("White", "Unknown")
                    white = (
                        Player.create_player(
                            session,
                            engine_type=f"pgn_import:{white_name}",
                            options={"name": white_name},
                        )
                        if create_players
                        else None
                    )
                    w_id = white.id if white else 1  # Fallback to ID 1
                else:
                    w_id = white_player_id

                if black_player_id is None:
                    black_name = pgn_game.headers.get("Black", "Unknown")
                    black = (
                        Player.create_player(
                            session,
                            engine_type=f"pgn_import:{black_name}",
                            options={"name": black_name},
                        )
                        if create_players
                        else None
                    )
                    b_id = black.id if black else 2  # Fallback to ID 2
                else:
                    b_id = black_player_id

                # Create game record
                metadata = {
                    "event": pgn_game.headers.get("Event"),
                    "site": pgn_game.headers.get("Site"),
                    "date": pgn_game.headers.get("Date"),
                    "round": pgn_game.headers.get("Round"),
                }

                # Handle opening FEN
                if pgn_game.headers.get("SetUp") == "1" and "FEN" in pgn_game.headers:
                    metadata["opening_fen"] = pgn_game.headers["FEN"]

                game = Game.create_game(session, w_id, b_id, metadata=metadata)

                # Import moves
                board = pgn_game.board()
                ply_index = 1

                for node in pgn_game.mainline():
                    move_number = board.fullmove_number
                    is_white = board.turn == chess.WHITE
                    fen_before = board.fen()

                    Move.create_move(
                        session,
                        game_id=game.id,
                        ply_index=ply_index,
                        move_number=move_number,
                        is_white=is_white,
                        fen_before=fen_before,
                        uci_move=node.move.uci(),
                    )

                    board.push(node.move)
                    ply_index += 1

                # Set game result
                result = pgn_game.headers.get("Result", "*")
                if result != "*":
                    game.result = result
                    game.finished_at = datetime.now()
                    session.commit()

                imported_count += 1
                logger.debug("Game imported", game_id=game.id, moves=ply_index - 1)

            except Exception as e:
                logger.error("Failed to import game", error=str(e))
                session.rollback()

    logger.info("PGN import completed", input_file=input_file, imported=imported_count)

    return imported_count


# CLI interface for PGN tools


def main():
    """Command-line interface for PGN import/export."""
    import argparse

    parser = argparse.ArgumentParser(description="ChessLab PGN Tools")
    parser.add_argument(
        "--db", default="postgresql://localhost/chesslab", help="Database URL"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export games to PGN")
    export_parser.add_argument("output", help="Output PGN file")
    export_parser.add_argument("--player", type=int, help="Filter by player ID")
    export_parser.add_argument("--limit", type=int, help="Maximum games to export")
    export_parser.add_argument("--games", nargs="+", type=int, help="Specific game IDs")

    # Import command
    import_parser = subparsers.add_parser("import", help="Import games from PGN")
    import_parser.add_argument("input", help="Input PGN file")
    import_parser.add_argument("--white", type=int, help="White player ID")
    import_parser.add_argument("--black", type=int, help="Black player ID")

    args = parser.parse_args()

    # Setup logging
    import logging

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )

    # Connect to database
    engine = create_db_engine(args.db)
    session = create_session(engine)

    try:
        if args.command == "export":
            count = export_games_to_file(
                session,
                args.output,
                game_ids=args.games,
                player_id=args.player,
                limit=args.limit,
            )
            print(f"Exported {count} games to {args.output}")

        elif args.command == "import":
            count = import_pgn_file(
                session,
                args.input,
                white_player_id=args.white,
                black_player_id=args.black,
            )
            print(f"Imported {count} games from {args.input}")

    finally:
        session.close()


if __name__ == "__main__":
    main()
