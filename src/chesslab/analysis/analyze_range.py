"""Chess Game manager using Player instances."""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib

from chesslab.storage.player_tools import list_players

matplotlib.use("Agg")  # To avoid interacting with stockfish
import matplotlib.pyplot as plt
import numpy as np
import structlog
from matplotlib.axes import Axes
from sqlalchemy.orm import Session

from chesslab.analysis.analyze_match import MatchAnalysis
from chesslab.analysis.elo_tools import ensemble_elo_from_scores, expected_score
from chesslab.analysis.evaluator import Evaluator
from chesslab.analysis.stat_tools import (
    compute_ensemble_p_value,
    ensemble_mean_score,
    ensemble_standard_error,
)
from chesslab.engines.init_engines import get_random_player
from chesslab.storage import Player, get_session

logger = structlog.get_logger()


class RangeAnalysis:
    """Manages a series of match between a player and a series of opponents."""

    def __init__(
        self,
        session: Session,
        evaluator: Evaluator,
        player: Player,
        opponents: List[Player],
        num_games: Optional[int] = None,
    ) -> None:
        self.player = player
        self.opponents = opponents
        self.evaluator = evaluator

        # Run the games and get results
        self.matches_analysis = [
            MatchAnalysis(
                session=session,
                evaluator=evaluator,
                player_1=player,
                player_2=opponent,
                num_games=num_games,
            )
            for opponent in self.opponents
        ]

    @property
    def opponent_elos(self) -> List[float]:
        """The elo for each opponent."""
        return [opponent.expected_elo for opponent in self.opponents]

    @property
    def win_ratios(self) -> List[float]:
        """Ratio of game won by the player for each match."""
        return [
            match_analysis.player_1_win_ratio
            for match_analysis in self.matches_analysis
        ]

    @property
    def loss_ratios(self) -> List[float]:
        """Ratio of game lost by the player for each match."""
        return [
            match_analysis.player_2_win_ratio
            for match_analysis in self.matches_analysis
        ]

    @property
    def max_standard_errors(self) -> List[float]:
        """Maximum standard error of the player score for each match."""
        return [
            match_analysis.max_standard_error_of_player_1
            for match_analysis in self.matches_analysis
        ]

    @property
    def observed_scores(self) -> List[float]:
        """Observed mean scores from each match."""
        return [
            match_analysis.player_1_mean for match_analysis in self.matches_analysis
        ]

    @property
    def num_rounds_per_match(self) -> List[int]:
        """Number of rounds in each match."""
        return [
            len(match_analysis.games_analysis)
            for match_analysis in self.matches_analysis
        ]

    @property
    def expected_scores(self) -> List[float]:
        """Expected scores based on player's declared Elo vs each opponent."""
        return [
            expected_score(self.player.expected_elo, opponent.expected_elo)
            for opponent in self.opponents
        ]

    @property
    def ensemble_observed_mean(self) -> float:
        """Weighted mean of observed scores across all matches."""
        return ensemble_mean_score(self.observed_scores, self.num_rounds_per_match)

    @property
    def ensemble_expected_mean(self) -> float:
        """Weighted mean of expected scores across all matches."""
        return ensemble_mean_score(self.expected_scores, self.num_rounds_per_match)

    @property
    def estimated_player_elo(self) -> float:
        """Estimate player's Elo based on results against all opponents.

        Uses optimization to find the Elo that best fits all match results.
        """
        return ensemble_elo_from_scores(
            observed_scores=self.observed_scores,
            opponent_elos=self.opponent_elos,
            num_rounds_per_match=self.num_rounds_per_match,
        )

    @property
    def estimated_scores(self) -> List[float]:
        """Expected scores based on estimated player Elo vs each opponent."""
        return [
            expected_score(self.estimated_player_elo, opponent.expected_elo)
            for opponent in self.opponents
        ]

    @property
    def ensemble_estimated_mean(self) -> float:
        """Weighted mean of estimated scores across all matches."""
        return ensemble_mean_score(self.estimated_scores, self.num_rounds_per_match)

    @property
    def ensemble_expected_standard_error(self) -> float:
        """Standard error of ensemble based on expected scores."""
        return ensemble_standard_error(self.expected_scores, self.num_rounds_per_match)

    @property
    def ensemble_estimated_standard_error(self) -> float:
        """Standard error of ensemble based on estimated scores."""
        return ensemble_standard_error(self.estimated_scores, self.num_rounds_per_match)

    @property
    def ensemble_p_value_of_expectation(self) -> float:
        """P-value comparing observed results to expected scores (based on declared Elo)."""
        return compute_ensemble_p_value(
            observed_scores=self.observed_scores,
            expected_scores=self.expected_scores,
            num_rounds_per_match=self.num_rounds_per_match,
        )

    @property
    def ensemble_p_value_of_estimation(self) -> float:
        """P-value comparing observed results to estimated scores (based on fitted Elo)."""
        return compute_ensemble_p_value(
            observed_scores=self.observed_scores,
            expected_scores=self.estimated_scores,
            num_rounds_per_match=self.num_rounds_per_match,
        )

    @property
    def number_of_move(self) -> float:
        """Number of time the player made a move."""
        return np.sum(
            [
                match_analysis.number_of_player_1_move
                for match_analysis in self.matches_analysis
            ]
        )

    @property
    def centipawn_loss(self) -> float:
        """Return the accumulated centipawn loss by the player."""
        return np.sum(
            [
                match_analysis.player_1_centipawn_loss
                for match_analysis in self.matches_analysis
            ]
        )

    @property
    def average_centipawn_loss(self) -> float:
        """Return the average centipawn loss per move by the player."""
        return self.centipawn_loss / self.number_of_move

    @property
    def report(self) -> str:
        """Textual report of all matches and ensemble statistics."""
        report_lines: List[str] = []

        # Individual match reports
        for match_analysis in self.matches_analysis:
            report_lines.append(match_analysis.report)

        # Ensemble statistics
        report_lines.append(f"{'=' * 50}\n")
        report_lines.append(
            f"ENSEMBLE STATISTICS FOR {self.player.engine_type} (ID: {self.player.id})\n"
        )
        report_lines.append(f"{'=' * 50}\n")
        report_lines.append(f"Total matches: {len(self.matches_analysis)}\n")
        report_lines.append(f"Total games: {sum(self.num_rounds_per_match)}\n")
        report_lines.append("\n")
        report_lines.append(f"Observed mean score: {self.ensemble_observed_mean:.3f}\n")
        report_lines.append(f"Average centipawn loss: {self.average_centipawn_loss}\n")
        report_lines.append(f"Expected mean score: {self.ensemble_expected_mean:.3f}\n")
        report_lines.append(
            f"Expected standard error: {self.ensemble_expected_standard_error:.5f}\n"
        )
        report_lines.append(
            f"P-value of expectation: {self.ensemble_p_value_of_expectation:.5f}\n"
        )
        report_lines.append("\n")
        report_lines.append(f"Estimated player Elo: {int(self.estimated_player_elo)}\n")
        report_lines.append(f"Declared player Elo: {int(self.player.expected_elo)}\n")
        report_lines.append(
            f"Elo difference: {int(self.estimated_player_elo - self.player.expected_elo):+d}\n"
        )
        report_lines.append(
            f"Estimated mean score: {self.ensemble_estimated_mean:.3f}\n"
        )
        report_lines.append(
            f"Estimated standard error: {self.ensemble_estimated_standard_error:.5f}\n"
        )
        report_lines.append(
            f"P-value of estimation: {self.ensemble_p_value_of_estimation:.5f}\n"
        )

        return "".join(report_lines)

    def plot_score_on_ax(self, ax: Axes) -> None:
        """Plot statistics for the player."""
        # ----- WIN/LOSS RATIO BARS -----
        width = 30
        # Win ratio
        ax.bar(  # pyright: ignore[reportUnknownMemberType]
            self.opponent_elos,
            self.win_ratios,
            color="#BDE7BD",
            width=width,
            align="center",
            bottom=0.0,
            edgecolor="none",
            zorder=1,
        )

        # Loss ratio
        ax.bar(  # pyright: ignore[reportUnknownMemberType]
            self.opponent_elos,
            -np.array(self.loss_ratios),
            color="#FFB6B3",
            width=width,
            align="center",
            bottom=1.0,
            edgecolor="none",
            zorder=1,
        )

        # ----- MEAN ± SE POINT -----
        ax.errorbar(  # pyright: ignore[reportUnknownMemberType]
            self.opponent_elos,
            self.observed_scores,
            yerr=self.max_standard_errors,
            fmt="o",
            color="black",
            capsize=5,
            label="Mean ± SE",
            zorder=2,
        )

        # ----- EXPECTED CURVES -----
        expected_x = np.linspace(min(self.opponent_elos), max(self.opponent_elos), 100)

        # Plot expected curve based on declared Elo
        expected_y = np.array(
            [
                expected_score(self.player.expected_elo, opponent_elo)
                for opponent_elo in expected_x
            ]
        )
        ax.plot(  # pyright: ignore[reportUnknownMemberType]
            expected_x,
            expected_y,
            "-",
            color="#ff0084",
            zorder=3,
            label=(
                f"Expected (Elo={int(self.player.expected_elo)}, "
                f"p={self.ensemble_p_value_of_expectation:.3f})"
            ),
        )

        # Plot estimated curve based on fitted Elo
        estimated_y = np.array(
            [
                expected_score(self.estimated_player_elo, opponent_elo)
                for opponent_elo in expected_x
            ]
        )
        ax.plot(  # pyright: ignore[reportUnknownMemberType]
            expected_x,
            estimated_y,
            "--",
            color="#0063dc",
            zorder=3,
            label=(
                f"Estimated (Elo={int(self.estimated_player_elo)}, "
                f"p={self.ensemble_p_value_of_estimation:.3f})"
            ),
        )

        # ----- STYLE -----
        ax.set_title(f"{self.player.engine_type} (ID: {self.player.id})")  # pyright: ignore[reportUnknownMemberType]
        ax.set_ylabel("Mean score")  # pyright: ignore[reportUnknownMemberType]
        ax.set_ylim(0, 1)
        ax.set_xlim(min(self.opponent_elos) - 50, max(self.opponent_elos) + 50)
        ax.grid(alpha=0.3, zorder=0)  # pyright: ignore[reportUnknownMemberType]
        ax.legend()  # pyright: ignore[reportUnknownMemberType]

    def plot_scores(self, path_folder: Path | str) -> None:
        """Plot statistics for one player."""
        _fig, axes = plt.subplots(1, 1, figsize=(10, 4), sharex=True)  # pyright: ignore[reportUnknownMemberType]

        self.plot_score_on_ax(axes)

        axes.set_xlabel("Opponent Elo")  # pyright: ignore[reportUnknownMemberType]
        plt.tight_layout()
        plot_path = Path(path_folder) / f"campaign_player_{self.player.id}.png"
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)  # pyright: ignore[reportUnknownMemberType]
        plt.close()


if __name__ == "__main__":
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )
    logger.info("Starting Range Analysis script")

    with get_session() as session:
        with Evaluator() as evaluator:
            logger.info("Getting players")
            random_player = get_random_player(session=session, create_not_raise=False)
            opponents = list_players(
                session=session,
                engine_type="Stockfish",
                min_elo=1320,
                max_elo=2200,
            )

            analysis = RangeAnalysis(
                session=session,
                evaluator=evaluator,
                player=random_player,
                opponents=opponents,
            )

            analysis.plot_scores("temp")

            print(analysis.report)
