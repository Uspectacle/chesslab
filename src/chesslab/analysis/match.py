"""Chess Game manager using Player instances."""

from pathlib import Path
from typing import List

import numpy as np

from chesslab.analysis.elo_tools import elo_from_mean_score, expected_score
from chesslab.analysis.game import Game
from chesslab.analysis.stat_tools import compute_p_value, standard_error_of_proportion
from chesslab.analysis.typing import Player


class Match:
    """Manages a series of chess game between two Player instances."""

    player_1: Player
    player_2: Player
    num_round: int
    max_moves: int
    result_folder: Path
    export_csv_log: bool
    export_pgn: bool
    verbose: bool

    games: List[Game]

    def __init__(
        self,
        player_1: Player,
        player_2: Player,
        fix_player_1_to_white: bool = False,
        num_round: int = 1,
        max_moves: int = 200,
        result_folder: str | Path = "results",
        export_csv_log: bool = False,
        export_pgn: bool = False,
        verbose: bool = False,
    ) -> None:
        self.player_1 = player_1
        self.player_2 = player_2

        self.num_round = num_round
        self.max_moves = max_moves
        self.result_folder = Path(result_folder)
        self.export_csv_log = export_csv_log
        self.export_pgn = export_pgn
        self.verbose = verbose

        # Run the games and get results
        self.games = [
            Game(
                player_1=player_1,
                player_2=player_2,
                fix_player_1_to_white=fix_player_1_to_white,
                chess_round=chess_round + 1,
                max_moves=max_moves,
                result_folder=result_folder,
                export_csv_log=export_csv_log,
                export_pgn=export_pgn,
                verbose=verbose,
            )
            for chess_round in range(num_round)
        ]

    @property
    def player_1_scores(self) -> List[float]:
        """Return list of player 1 score for each game."""
        return [game.player_1_score for game in self.games]

    @property
    def player_2_scores(self) -> List[float]:
        """Return list of player 2 score for each game."""
        return [game.player_2_score for game in self.games]

    @property
    def player_1_score(self) -> float:
        """Return the total score of player 1 score."""
        return np.sum(self.player_1_scores)

    @property
    def player_2_score(self) -> float:
        """Return the total score of player 2 score."""
        return np.sum(self.player_2_scores)

    @property
    def player_1_mean(self) -> float:
        """Average score of player 1."""
        return self.player_1_score / self.num_round

    @property
    def player_2_mean(self) -> float:
        """Average score of player 2."""
        return self.player_2_score / self.num_round

    @property
    def number_of_player_1_win(self) -> int:
        """Number of game won by player 1."""
        return np.sum([game.player_1_score == 1 for game in self.games])

    @property
    def number_of_player_2_win(self) -> int:
        """Number of game won by player 2."""
        return np.sum([game.player_2_score == 1 for game in self.games])

    @property
    def number_of_draw(self) -> int:
        """Number of game resulting in a draw."""
        return np.sum([game.player_1_score == 0.5 for game in self.games])

    @property
    def player_1_win_ratio(self) -> float:
        """Ratio of game won by player 1."""
        return self.number_of_player_1_win / self.num_round

    @property
    def player_2_win_ratio(self) -> float:
        """Ratio of game won by player 2."""
        return self.number_of_player_2_win / self.num_round

    @property
    def estimated_player_1_elo(self) -> float:
        """Estimation of player 1 elo based on the elo of player 2."""
        return elo_from_mean_score(
            mean_score=self.player_1_mean, opponent_elo=self.player_2.elo
        )

    @property
    def estimated_player_2_elo(self) -> float:
        """Estimation of player 2 elo based on the elo of player 1."""
        return elo_from_mean_score(
            mean_score=self.player_2_mean, opponent_elo=self.player_1.elo
        )

    @property
    def expected_player_1_mean(self) -> float:
        """Expected score of player 1."""
        return expected_score(self.player_1.elo, self.player_2.elo)

    @property
    def expected_player_2_mean(self) -> float:
        """Expected score of player 2."""
        return expected_score(self.player_2.elo, self.player_1.elo)

    @property
    def estimated_player_1_mean(self) -> float:
        """Expected score of player 1 based on estimation."""
        return expected_score(self.estimated_player_1_elo, self.player_2.elo)

    @property
    def estimated_player_2_mean(self) -> float:
        """Expected score of player 2 based on estimation."""
        return expected_score(self.estimated_player_2_elo, self.player_1.elo)

    @property
    def standard_error(self) -> float:
        """Standard error based on observation."""
        return standard_error_of_proportion(
            proportion=self.player_1_mean, num_round=self.num_round
        )

    @property
    def expected_standard_error(self) -> float:
        """Standard error based on expectation."""
        return standard_error_of_proportion(
            proportion=self.expected_player_1_mean, num_round=self.num_round
        )

    @property
    def estimated_standard_error_from_player_1(self) -> float:
        """Standard error based on estimation of player 1 elo."""
        return standard_error_of_proportion(
            proportion=self.estimated_player_1_mean, num_round=self.num_round
        )

    @property
    def estimated_standard_error_from_player_2(self) -> float:
        """Standard error based on estimation of player 2 elo."""
        return standard_error_of_proportion(
            proportion=self.estimated_player_2_mean, num_round=self.num_round
        )

    @property
    def max_standard_error_of_player_1(self) -> float:
        """Maximum standard error assuming player 2 expected elo."""
        return max(
            self.standard_error,
            self.expected_standard_error,
            self.estimated_standard_error_from_player_1,
        )

    @property
    def p_value_of_expectation(self) -> float:
        """Probability of getting score average based on expectation."""
        return compute_p_value(
            observed_value=self.player_1_mean,
            expected_value=self.expected_player_1_mean,
            num_round=self.num_round,
        )

    @property
    def p_value_of_estimation_from_player_1(self) -> float:
        """Probability of getting score average based on expectation."""
        return compute_p_value(
            observed_value=self.player_1_mean,
            expected_value=self.estimated_player_1_mean,
            num_round=self.num_round,
        )

    @property
    def p_value_of_estimation_from_player_2(self) -> float:
        """Probability of getting score average based on expectation."""
        return compute_p_value(
            observed_value=self.player_2_mean,
            expected_value=self.estimated_player_2_mean,
            num_round=self.num_round,
        )

    @property
    def number_of_player_1_move(self) -> float:
        """Number of time player 1 made a move."""
        return np.sum([game.number_of_player_1_move for game in self.games])

    @property
    def number_of_player_2_move(self) -> float:
        """Number of time player 2 made a move."""
        return np.sum([game.number_of_player_2_move for game in self.games])

    @property
    def player_1_centipawn_loss(self) -> float:
        """Return the accumulated centipawn loss of player 1."""
        return np.sum([game.player_1_centipawn_loss for game in self.games])

    @property
    def player_2_centipawn_loss(self) -> float:
        """Return the accumulated centipawn loss of player 2."""
        return np.sum([game.player_2_centipawn_loss for game in self.games])

    @property
    def player_1_average_centipawn_loss(self) -> float:
        """Return the average centipawn loss per move of player 1."""
        return self.player_1_centipawn_loss / self.number_of_player_1_move

    @property
    def player_2_average_centipawn_loss(self) -> float:
        """Return the average centipawn loss per move of player 2."""
        return self.player_2_centipawn_loss / self.number_of_player_2_move

    @property
    def report(self) -> str:
        """Determine whether to use t-test or z-test based on number of games."""
        return (
            f"{'-' * 50}\n"
            f"Match: {self.player_1} vs {self.player_2}\n"
            f"{'-' * 50}\n"
            f"Games played: {self.num_round}\n"
            f"-> For Player 1 {self.player_1}:\n"
            f"Score: {self.number_of_player_1_win:.3f} win - "
            f"{self.number_of_draw:.3f} draw - {self.number_of_player_2_win:.3f} loose\n"
            f"Average centipawn loss: {self.player_1_average_centipawn_loss}\n"
            f"Mean observed score: {self.player_1_mean:.3f}\n"
            f"Expected score: {self.expected_player_1_mean:.3f}\n"
            f"P-value of expectation: {self.p_value_of_expectation:.5f}\n"
            f"Estimated ELO: {int(self.estimated_player_1_elo)}\n"
            f"Estimated score: {self.estimated_player_1_mean:.3f}\n"
            f"P-value of estimation: {self.p_value_of_estimation_from_player_1:.5f}\n"
        )
