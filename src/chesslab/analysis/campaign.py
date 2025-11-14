"""Chess Game manager using Player instances."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from chesslab.analysis.elo_tools import ensemble_elo_from_scores, expected_score
from chesslab.analysis.match import Match
from chesslab.analysis.stat_tools import (
    compute_ensemble_p_value,
    ensemble_mean_score,
    ensemble_standard_error,
)
from chesslab.analysis.typing import Player


class Campaign:
    """Manages a series of match between a player and a series of opponents."""

    player: Player
    opponents: List[Player]
    num_round: int
    max_moves: int
    result_folder: Path
    export_csv_log: bool
    export_pgn: bool
    verbose: bool

    matches: List[Match]

    def __init__(
        self,
        player: Player,
        opponents: List[Player],
        fix_player_to_white: bool = False,
        num_round: int = 1,
        max_moves: int = 200,
        result_folder: str | Path = "results",
        export_csv_log: bool = False,
        export_pgn: bool = False,
        verbose: bool = False,
        export_report: bool = False,
        export_plot_scores: bool = False,
    ) -> None:
        self.player = player
        self.opponents = opponents

        self.num_round = num_round
        self.max_moves = max_moves
        self.result_folder = Path(result_folder)
        self.export_csv_log = export_csv_log
        self.export_pgn = export_pgn
        self.verbose = verbose

        # Run the games and get results
        self.matches = [
            Match(
                player_1=player,
                player_2=opponent,
                fix_player_1_to_white=fix_player_to_white,
                num_round=num_round,
                max_moves=max_moves,
                result_folder=f"{result_folder}/{player}_vs_{opponent}",
                export_csv_log=export_csv_log,
                export_pgn=export_pgn,
                verbose=verbose,
            )
            for opponent in self.opponents
        ]

        if export_report:
            self.write_report()

        if export_plot_scores:
            self.plot_scores()

    @property
    def opponent_elos(self) -> List[float]:
        """The elo for each opponent."""
        return [opponent.elo for opponent in self.opponents]

    @property
    def win_ratios(self) -> List[float]:
        """Ratio of game won by the player for each match."""
        return [match.player_1_win_ratio for match in self.matches]

    @property
    def loss_ratios(self) -> List[float]:
        """Ratio of game lost by the player for each match."""
        return [match.player_2_win_ratio for match in self.matches]

    @property
    def max_standard_errors(self) -> List[float]:
        """Maximum standard error of the player score for each match."""
        return [match.max_standard_error_of_player_1 for match in self.matches]

    @property
    def observed_scores(self) -> List[float]:
        """Observed mean scores from each match."""
        return [match.player_1_mean for match in self.matches]

    @property
    def num_rounds_per_match(self) -> List[int]:
        """Number of rounds in each match."""
        return [match.num_round for match in self.matches]

    @property
    def expected_scores(self) -> List[float]:
        """Expected scores based on player's declared Elo vs each opponent."""
        return [
            expected_score(self.player.elo, opponent.elo) for opponent in self.opponents
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
            expected_score(self.estimated_player_elo, opponent.elo)
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
        return np.sum([match.number_of_player_1_move for match in self.matches])

    @property
    def centipawn_loss(self) -> float:
        """Return the accumulated centipawn loss by the player."""
        return np.sum([match.player_1_centipawn_loss for match in self.matches])

    @property
    def average_centipawn_loss(self) -> float:
        """Return the average centipawn loss per move by the player."""
        return self.centipawn_loss / self.number_of_move

    @property
    def report(self) -> str:
        """Textual report of all matches and ensemble statistics."""
        report_lines: List[str] = []

        # Individual match reports
        for match in self.matches:
            report_lines.append(match.report)

        # Ensemble statistics
        report_lines.append(f"{'=' * 50}\n")
        report_lines.append(f"ENSEMBLE STATISTICS FOR {self.player}\n")
        report_lines.append(f"{'=' * 50}\n")
        report_lines.append(f"Total matches: {len(self.matches)}\n")
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
        report_lines.append(f"Declared player Elo: {int(self.player.elo)}\n")
        report_lines.append(
            f"Elo difference: {int(self.estimated_player_elo - self.player.elo):+d}\n"
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

    def write_report(self) -> None:
        """Write a campaign textual report of all matches."""
        report_path = f"{self.result_folder}/campaign_report.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"=== {self.player} Campaign Report ===\n\n")
            f.write(self.report)

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
                expected_score(self.player.elo, opponent_elo)
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
                f"Expected (Elo={int(self.player.elo)}, "
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
        ax.set_title(str(self.player))  # pyright: ignore[reportUnknownMemberType]
        ax.set_ylabel("Mean score")  # pyright: ignore[reportUnknownMemberType]
        ax.set_ylim(0, 1)
        ax.set_xlim(min(self.opponent_elos) - 50, max(self.opponent_elos) + 50)
        ax.grid(alpha=0.3, zorder=0)  # pyright: ignore[reportUnknownMemberType]
        ax.legend()  # pyright: ignore[reportUnknownMemberType]

    def plot_scores(self) -> None:
        """Plot statistics for one player."""
        _fig, axes = plt.subplots(1, 1, figsize=(10, 4), sharex=True)  # pyright: ignore[reportUnknownMemberType]

        self.plot_score_on_ax(axes)

        axes.set_xlabel("Opponent Elo")  # pyright: ignore[reportUnknownMemberType]
        plt.tight_layout()
        png_path = f"{self.result_folder}/campaign_scores.png"
        plt.savefig(png_path)  # pyright: ignore[reportUnknownMemberType]
        plt.close()
