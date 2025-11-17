"""Chess Game manager using Player instances."""

import logging
from typing import List

import numpy as np
import structlog
from sqlalchemy.orm import Session

from chesslab.analysis.analyze_game import GameAnalysis
from chesslab.analysis.elo_tools import elo_from_mean_score, expected_score
from chesslab.analysis.evaluator import Evaluator
from chesslab.analysis.stat_tools import compute_p_value, standard_error_of_proportion
from chesslab.arena.init_engines import (
    get_random_player,
    get_stockfish_player,
)
from chesslab.storage import Player, get_head_to_head_games, get_session
from chesslab.storage.game_tools import get_games_by_players

logger = structlog.get_logger()


class MatchAnalysis:
    """Manages a series of chess game between two Player instances."""

    def __init__(
        self,
        session: Session,
        evaluator: Evaluator,
        player_1: Player,
        player_2: Player,
        player_1_only_white: bool = False,
    ) -> None:
        self.player_1 = player_1
        self.player_2 = player_2
        self.evaluator = evaluator

        if player_1_only_white:
            games = get_games_by_players(
                session=session,
                white_player_id=player_1.id,
                black_player_id=player_2.id,
            )
        else:
            games = get_head_to_head_games(
                session=session, player1_id=player_1.id, player2_id=player_2.id
            )

        # Run the games and get results
        self.games_analysis = [
            GameAnalysis(
                evaluator=evaluator,
                game=game,
            )
            for game in games
            if game.result
        ]

    @property
    def player_1_scores(self) -> List[float]:
        """Return list of player 1 score for each game."""
        return [
            game_analysis.get_score(self.player_1.id)
            for game_analysis in self.games_analysis
        ]

    @property
    def player_2_scores(self) -> List[float]:
        """Return list of player 2 score for each game."""
        return [
            game_analysis.get_score(self.player_2.id)
            for game_analysis in self.games_analysis
        ]

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
        return self.player_1_score / len(self.games_analysis)

    @property
    def player_2_mean(self) -> float:
        """Average score of player 2."""
        return self.player_2_score / len(self.games_analysis)

    @property
    def number_of_player_1_win(self) -> int:
        """Number of game won by player 1."""
        return np.sum(
            [
                game_analysis.get_score(self.player_1.id) == 1.0
                for game_analysis in self.games_analysis
            ]
        )

    @property
    def number_of_player_2_win(self) -> int:
        """Number of game won by player 2."""
        return np.sum(
            [
                game_analysis.get_score(self.player_2.id) == 1.0
                for game_analysis in self.games_analysis
            ]
        )

    @property
    def number_of_draw(self) -> int:
        """Number of game resulting in a draw."""
        return np.sum(
            [
                game_analysis.get_score(self.player_1.id) == 0.5
                for game_analysis in self.games_analysis
            ]
        )

    @property
    def player_1_win_ratio(self) -> float:
        """Ratio of game won by player 1."""
        return self.number_of_player_1_win / len(self.games_analysis)

    @property
    def player_2_win_ratio(self) -> float:
        """Ratio of game won by player 2."""
        return self.number_of_player_2_win / len(self.games_analysis)

    @property
    def estimated_player_1_elo(self) -> float:
        """Estimation of player 1 elo based on the elo of player 2."""
        return elo_from_mean_score(
            mean_score=self.player_1_mean, opponent_elo=self.player_2.expected_elo
        )

    @property
    def estimated_player_2_elo(self) -> float:
        """Estimation of player 2 elo based on the elo of player 1."""
        return elo_from_mean_score(
            mean_score=self.player_2_mean, opponent_elo=self.player_1.expected_elo
        )

    @property
    def expected_player_1_mean(self) -> float:
        """Expected score of player 1."""
        return expected_score(self.player_1.expected_elo, self.player_2.expected_elo)

    @property
    def expected_player_2_mean(self) -> float:
        """Expected score of player 2."""
        return expected_score(self.player_2.expected_elo, self.player_1.expected_elo)

    @property
    def estimated_player_1_mean(self) -> float:
        """Expected score of player 1 based on estimation."""
        return expected_score(self.estimated_player_1_elo, self.player_2.expected_elo)

    @property
    def estimated_player_2_mean(self) -> float:
        """Expected score of player 2 based on estimation."""
        return expected_score(self.estimated_player_2_elo, self.player_1.expected_elo)

    @property
    def standard_error(self) -> float:
        """Standard error based on observation."""
        return standard_error_of_proportion(
            proportion=self.player_1_mean, num_round=len(self.games_analysis)
        )

    @property
    def expected_standard_error(self) -> float:
        """Standard error based on expectation."""
        return standard_error_of_proportion(
            proportion=self.expected_player_1_mean, num_round=len(self.games_analysis)
        )

    @property
    def estimated_standard_error_from_player_1(self) -> float:
        """Standard error based on estimation of player 1 elo."""
        return standard_error_of_proportion(
            proportion=self.estimated_player_1_mean, num_round=len(self.games_analysis)
        )

    @property
    def estimated_standard_error_from_player_2(self) -> float:
        """Standard error based on estimation of player 2 elo."""
        return standard_error_of_proportion(
            proportion=self.estimated_player_2_mean, num_round=len(self.games_analysis)
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
            num_round=len(self.games_analysis),
        )

    @property
    def p_value_of_estimation_from_player_1(self) -> float:
        """Probability of getting score average based on expectation."""
        return compute_p_value(
            observed_value=self.player_1_mean,
            expected_value=self.estimated_player_1_mean,
            num_round=len(self.games_analysis),
        )

    @property
    def p_value_of_estimation_from_player_2(self) -> float:
        """Probability of getting score average based on expectation."""
        return compute_p_value(
            observed_value=self.player_2_mean,
            expected_value=self.estimated_player_2_mean,
            num_round=len(self.games_analysis),
        )

    @property
    def number_of_player_1_move(self) -> float:
        """Number of time player 1 made a move."""
        return np.sum(
            [
                game_analysis.get_number_of_move(self.player_1.id)
                for game_analysis in self.games_analysis
            ]
        )

    @property
    def number_of_player_2_move(self) -> float:
        """Number of time player 2 made a move."""
        return np.sum(
            [
                game_analysis.get_number_of_move(self.player_2.id)
                for game_analysis in self.games_analysis
            ]
        )

    @property
    def player_1_centipawn_loss(self) -> float:
        """Return the accumulated centipawn loss of player 1."""
        return np.sum(
            [
                game_analysis.get_centipawn_loss(self.player_1.id)
                for game_analysis in self.games_analysis
            ]
        )

    @property
    def player_2_centipawn_loss(self) -> float:
        """Return the accumulated centipawn loss of player 2."""
        return np.sum(
            [
                game_analysis.get_centipawn_loss(self.player_2.id)
                for game_analysis in self.games_analysis
            ]
        )

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
            f"Match: {self.player_1.engine_type} (ID: {self.player_1.id}) "
            f"vs {self.player_2.engine_type} (ID: {self.player_2.id})\n"
            f"{'-' * 50}\n"
            f"Games played: {len(self.games_analysis)}\n"
            f"Score: {self.number_of_player_1_win:.3f} win - "
            f"{self.number_of_draw:.3f} draw - {self.number_of_player_2_win:.3f} loose\n"
            f"\n"
            f"For Player 1: \n"
            f"Average centipawn loss: {self.player_1_average_centipawn_loss}\n"
            f"Mean observed score: {self.player_1_mean:.3f}\n"
            f"Expected ELO: {self.player_1.expected_elo}\n"
            f"Expected score: {self.expected_player_1_mean:.3f}\n"
            f"P-value of expectation: {self.p_value_of_expectation:.5f}\n"
            f"Estimated ELO: {int(self.estimated_player_1_elo)}\n"
            f"Estimated score: {self.estimated_player_1_mean:.3f}\n"
            f"P-value of estimation: {self.p_value_of_estimation_from_player_1:.5f}\n"
            f"\n"
            f"For Player 2: \n"
            f"Average centipawn loss: {self.player_2_average_centipawn_loss}\n"
            f"Mean observed score: {self.player_2_mean:.3f}\n"
            f"Expected ELO: {self.player_2.expected_elo}\n"
            f"Expected score: {self.expected_player_2_mean:.3f}\n"
            f"P-value of expectation: {self.p_value_of_expectation:.5f}\n"
            f"Estimated ELO: {int(self.estimated_player_2_elo)}\n"
            f"Estimated score: {self.estimated_player_2_mean:.3f}\n"
            f"P-value of estimation: {self.p_value_of_estimation_from_player_2:.5f}\n"
        )


if __name__ == "__main__":
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )
    logger.info("Starting match runner script")

    with get_session() as session:
        with Evaluator() as evaluator:
            stockfish = get_stockfish_player(session=session, create_not_raise=False)

            logger.info("Creating random player")
            random_player = get_random_player(session=session, create_not_raise=False)

            logger.info("analyze_match")
            analysis = MatchAnalysis(
                session=session,
                evaluator=evaluator,
                player_1=random_player,
                player_2=stockfish,
            )

            print(analysis.report)
            for game in analysis.games_analysis:
                print(game.report)
