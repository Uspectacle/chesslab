"""Chess Game manager using Player instances."""

import _csv
import csv
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO

import chess
import chess.pgn

from chesslab.analysis.stockfish_tools import get_centipawn_loss
from chesslab.analysis.typing import Ballot, Player


class Game:
    """Manages a single chess game between two Player instances."""

    player_1: Player
    player_2: Player
    player_1_is_white: bool
    player_white: Player
    player_black: Player

    chess_round: int
    max_moves: int

    result_folder: Path
    export_csv_log: bool
    export_pgn: bool
    verbose: bool

    board: chess.Board
    game: chess.pgn.Game
    node: chess.pgn.GameNode

    result: str
    moves: int
    duration: float
    fen: str
    termination: str

    csv_file: Optional[TextIO]
    csv_writer: Optional[_csv._writer]  # pyright: ignore[reportPrivateUsage]

    def __init__(
        self,
        player_1: Player,
        player_2: Player,
        fix_player_1_to_white: bool = False,
        chess_round: int = 1,
        max_moves: int = 200,
        result_folder: str | Path = "results",
        export_csv_log: bool = False,
        export_pgn: bool = False,
        verbose: bool = False,
    ) -> None:
        # Assign players to sides
        self.player_1_is_white = fix_player_1_to_white or random.choice([True, False])

        if self.player_1_is_white:
            self.player_white, self.player_black = player_1, player_2
        else:
            self.player_white, self.player_black = player_2, player_1

        self.chess_round = chess_round
        self.max_moves = max_moves
        self.result_folder = Path(result_folder)
        self.export_csv_log = export_csv_log
        self.export_pgn = export_pgn
        self.verbose = verbose

        # Game state
        self.board = chess.Board()
        self.game = chess.pgn.Game()
        self.node = self.game

        # Result data
        self.result = "*"
        self.moves = 0
        self.duration = 0.0
        self.fen = ""
        self.termination = "Unknown"

        # CSV logging
        self.csv_file = None
        self.csv_writer = None

        # Run the game and get results
        self.play_game()

    def play_game(self):
        """Main game loop."""
        self.init_game()
        start_time = time.time()

        while not self.is_game_over():
            self.next_move()

        self.duration = time.time() - start_time
        self.close_game()

    def init_game(self):
        """Initialize the PGN headers and prepare CSV if needed."""
        self.game.headers["Event"] = "Collective Chess"
        self.game.headers["Site"] = "https://github.com/Uspectacle/collective-chess"
        self.game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        self.game.headers["Round"] = str(self.chess_round)
        self.game.headers["White"] = str(self.player_white)
        self.game.headers["Black"] = str(self.player_black)
        self.game.headers["Result"] = "*"

        if self.export_csv_log:
            self.result_folder.mkdir(parents=True, exist_ok=True)
            csv_path = self.result_folder / f"game_{self.chess_round}_log.csv"
            self.csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(
                ["move_number", "fen", "side", "chosen_move", "aggregated_probs"]
            )

    def close_game(self):
        """Export the results and close files if needed."""

        outcome = self.board.outcome(claim_draw=True)
        self.result = self.board.result(claim_draw=True)
        self.fen = self.board.fen()
        self.termination = outcome.termination.name if outcome else "Unknown"
        self.game.headers["Result"] = self.result

        if self.export_pgn:
            self.export_pgn_file()

        if self.csv_file:
            self.csv_file.close()

        if self.verbose:
            print("\nGame Over!")
            print("-" * 50)
            if outcome:
                winner_str = (
                    str(self.player_white)
                    if outcome.winner
                    else str(self.player_black)
                    if outcome.winner is False
                    else "Draw"
                )
                print(f"Winner: {winner_str}")
                print(f"Termination: {self.termination}")
            print(f"Final position: {self.fen}")

    def is_game_over(self):
        """Test if the game must terminate"""

        if self.moves >= self.max_moves:
            return True

        return self.board.is_game_over(claim_draw=True)

    def next_move(self):
        """Play a single move."""
        player = (
            self.player_white if self.board.turn == chess.WHITE else self.player_black
        )
        ballot = player.evaluate(self.board)
        chosen_move = max(ballot.items(), key=lambda x: x[1])[0]

        self.board.push(chess.Move.from_uci(chosen_move))
        self.moves += 1

        self.export_csv(self.moves, chosen_move, ballot)
        self.node = self.node.add_variation(chess.Move.from_uci(chosen_move))

        if self.verbose:
            print(f"\nMove {self.moves}: {player} plays {chosen_move}")
            print(self.board)
            if chosen_move in ballot:
                print(f"Score: {ballot[chosen_move]:.3f}")

    def export_pgn_file(self):
        """Export the game to PGN if enabled."""
        self.result_folder.mkdir(parents=True, exist_ok=True)
        pgn_path = self.result_folder / f"game_{self.chess_round}.pgn"
        with open(pgn_path, "w", encoding="utf-8") as f:
            f.write(str(self.game))

    def export_csv(self, move_count: int, chosen_move: str, ballot: Ballot):
        """Log the move to CSV."""
        if not self.csv_writer:
            return

        side = "white" if self.board.turn == chess.WHITE else "black"
        self.csv_writer.writerow(
            [move_count, self.board.fen(), side, chosen_move, json.dumps(ballot)]
        )

    @property
    def white_score(self) -> float:
        """Return 1 if white wins, 0 if white loses, 0.5 if draw."""
        if self.result == "1-0":
            return 1.0
        elif self.result == "0-1":
            return 0.0
        return 0.5

    @property
    def black_score(self) -> float:
        """Return 1 if black wins, 0 if black loses, 0.5 if draw."""
        if self.result == "0-1":
            return 1.0
        elif self.result == "1-0":
            return 0.0
        return 0.5

    @property
    def player_1_score(self) -> float:
        """Return 1 if player 1 wins, 0 if loses, 0.5 if draw."""
        return self.white_score if self.player_1_is_white else self.black_score

    @property
    def player_2_score(self) -> float:
        """Return 1 if player 2 wins, 0 if loses, 0.5 if draw."""
        return self.black_score if self.player_1_is_white else self.white_score

    @property
    def number_of_white_move(self) -> int:
        """Number of time white made a move."""
        return math.ceil(len(self.board.move_stack) / 2)

    @property
    def number_of_black_move(self) -> int:
        """Number of time black made a move."""
        return math.floor(len(self.board.move_stack) / 2)

    @property
    def white_centipawn_loss(self) -> float:
        """Return the accumulated centipawn loss of white."""
        board = chess.Board()
        loss = 0.0

        for move in self.board.move_stack:
            if board.turn == chess.WHITE:
                loss += get_centipawn_loss(board, move)
            board.push(move)

        return loss

    @property
    def black_centipawn_loss(self) -> float:
        """Return the accumulated centipawn loss of black."""
        board = chess.Board()
        loss = 0.0

        for move in self.board.move_stack:
            if board.turn == chess.BLACK:
                loss += get_centipawn_loss(board, move)
            board.push(move)

        return loss

    @property
    def white_average_centipawn_loss(self) -> float:
        """Return the average centipawn loss per white move."""
        return self.white_centipawn_loss / self.number_of_white_move

    @property
    def black_average_centipawn_loss(self) -> float:
        """Return the average centipawn loss per black move."""
        return self.black_centipawn_loss / self.number_of_black_move

    @property
    def number_of_player_1_move(self) -> float:
        """Number of time player 1 made a move."""
        if self.player_1_is_white:
            return self.number_of_white_move

        return self.number_of_black_move

    @property
    def number_of_player_2_move(self) -> float:
        """Number of time player 2 made a move."""
        if self.player_1_is_white:
            return self.number_of_black_move

        return self.number_of_white_move

    @property
    def player_1_centipawn_loss(self) -> float:
        """Return the accumulated centipawn loss of player 1."""
        if self.player_1_is_white:
            return self.white_centipawn_loss

        return self.black_centipawn_loss

    @property
    def player_2_centipawn_loss(self) -> float:
        """Return the accumulated centipawn loss of player 2."""
        if self.player_1_is_white:
            return self.black_centipawn_loss

        return self.white_centipawn_loss

    @property
    def player_1_average_centipawn_loss(self) -> float:
        """Return the average centipawn loss per move of player 1."""
        if self.player_1_is_white:
            return self.white_average_centipawn_loss

        return self.black_average_centipawn_loss

    @property
    def player_2_average_centipawn_loss(self) -> float:
        """Return the average centipawn loss per move of player 2."""
        if self.player_1_is_white:
            return self.black_average_centipawn_loss

        return self.white_average_centipawn_loss
