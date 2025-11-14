"""Database-driven analysis tools for ChessLab.

Retrieves game data from the database and performs statistical analysis,
ELO estimation, and visualization without running new games.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy.orm import Session

from chesslab.analysis.analyze_match import MatchAnalysis
from chesslab.analysis.elo_tools import expected_score
from chesslab.arena.init_engines import get_or_create_random_player
from chesslab.storage import Game, Player, get_session


class CampaignAnalysis:
    """Analysis of one player against multiple opponents."""

    def __init__(
        self,
        session: Session,
        player: Player,
        opponents: List[Player],
    ):
        self.session = session
        self.player = player
        self.opponents = opponents

        # Analyze each matchup
        self.matches = [
            MatchAnalysis(
                session=session,
                player=player,
                opponent=opponent,
            )
            for opponent in opponents
        ]

    @property
    def opponent_elos(self) -> List[float]:
        """ELOs of all opponents."""
        return [match.opponent.elo for match in self.matches]

    @property
    def observed_scores(self) -> List[float]:
        """Observed mean scores against each opponent."""
        return [match.player_mean_score for match in self.matches]

    @property
    def expected_scores(self) -> List[float]:
        """Expected scores based on declared ELOs."""
        return [match.expected_player_score for match in self.matches]

    @property
    def num_games_per_opponent(self) -> List[int]:
        """Number of games against each opponent."""
        return [match.num_games for match in self.matches]

    @property
    def total_games(self) -> int:
        """Total number of games."""
        return sum(self.num_games_per_opponent)

    @property
    def ensemble_observed_mean(self) -> float:
        """Weighted mean of observed scores."""
        if self.total_games == 0:
            return 0.0
        weighted_sum = sum(
            score * n_games
            for score, n_games in zip(self.observed_scores, self.num_games_per_opponent)
        )
        return weighted_sum / self.total_games

    @property
    def ensemble_expected_mean(self) -> float:
        """Weighted mean of expected scores."""
        if self.total_games == 0:
            return 0.0
        weighted_sum = sum(
            score * n_games
            for score, n_games in zip(self.expected_scores, self.num_games_per_opponent)
        )
        return weighted_sum / self.total_games

    @property
    def report(self) -> str:
        """Generate text report."""
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append(
            f"Campaign Analysis: Player {self.player.id} ({self.player.engine_type})"
        )
        lines.append("=" * 60)
        lines.append(f"Total opponents: {len(self.opponents)}")
        lines.append(f"Total games: {self.total_games}")
        lines.append("")

        for match in self.matches:
            lines.append(f"vs Player {match.opponent.id}: {match.num_games} games")
            lines.append(f"  Score: {match.player_score:.1f}/{match.num_games}")
            lines.append(f"  Mean: {match.player_mean_score:.3f}")

        lines.append("")
        lines.append(f"Ensemble Observed Mean: {self.ensemble_observed_mean:.3f}")
        lines.append(f"Ensemble Expected Mean: {self.ensemble_expected_mean:.3f}")
        lines.append("")

        return "\n".join(lines)

    def plot_scores(
        self, output_path: Optional[str] = None, show: bool = False
    ) -> None:
        """Plot campaign results.

        Args:
            output_path: Path to save the plot
            show: Whether to display the plot
        """
        _fig, ax = plt.subplots(figsize=(10, 6))  # pyright: ignore[reportUnknownMemberType]

        # Win/loss ratios
        win_ratios: List[float] = []
        loss_ratios: List[float] = []

        for match in self.matches:
            wins = sum(
                1
                for _game in self.session.query(Game)
                .filter(
                    (
                        (Game.white_player_id == self.player.id)
                        & (Game.black_player_id == match.opponent.id)
                        & (Game.result == "1-0")
                    )
                    | (
                        (Game.black_player_id == self.player.id)
                        & (Game.white_player_id == match.opponent.id)
                        & (Game.result == "0-1")
                    )
                )
                .all()
            )
            losses = sum(
                1
                for _game in self.session.query(Game)
                .filter(
                    (
                        (Game.white_player_id == self.player.id)
                        & (Game.black_player_id == match.opponent.id)
                        & (Game.result == "0-1")
                    )
                    | (
                        (Game.black_player_id == self.player.id)
                        & (Game.white_player_id == match.opponent.id)
                        & (Game.result == "1-0")
                    )
                )
                .all()
            )

            win_ratio = wins / match.num_games if match.num_games > 0 else 0.0
            loss_ratio = losses / match.num_games if match.num_games > 0 else 0.0

            win_ratios.append(win_ratio)
            loss_ratios.append(loss_ratio)

        # Plot bars
        width = 30
        ax.bar(  # pyright: ignore[reportUnknownMemberType]
            self.opponent_elos,
            win_ratios,
            color="#BDE7BD",
            width=width,
            label="Win Ratio",
        )
        ax.bar(  # pyright: ignore[reportUnknownMemberType]
            self.opponent_elos,
            [-r for r in loss_ratios],
            color="#FFB6B3",
            width=width,
            bottom=1.0,
            label="Loss Ratio",
        )

        # Plot observed scores
        ax.plot(  # pyright: ignore[reportUnknownMemberType]
            self.opponent_elos,
            self.observed_scores,
            "o",
            color="black",
            markersize=8,
            label="Observed Score",
        )

        # Plot expected curve
        x_range = np.linspace(
            min(self.opponent_elos) - 50, max(self.opponent_elos) + 50, 100
        )
        expected_curve = [
            expected_score(self.player.elo, opp_elo) for opp_elo in x_range
        ]

        ax.plot(  # pyright: ignore[reportUnknownMemberType]
            x_range,
            expected_curve,
            "-",
            color="#ff0084",
            label=f"Expected (ELO={int(self.player.elo)})",
        )

        ax.set_xlabel("Opponent ELO")  # pyright: ignore[reportUnknownMemberType]
        ax.set_ylabel("Score")  # pyright: ignore[reportUnknownMemberType]
        ax.set_title(f"Campaign: {self.player.engine_type} (Player {self.player.id})")  # pyright: ignore[reportUnknownMemberType]
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)  # pyright: ignore[reportUnknownMemberType]
        ax.legend()  # pyright: ignore[reportUnknownMemberType]

        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150)  # pyright: ignore[reportUnknownMemberType]

        if show:
            plt.show()  # pyright: ignore[reportUnknownMemberType]
        else:
            plt.close()


def analyze_campaign(
    session: Session,
    player: Player,
    opponents: List[Player],
    output_dir: Optional[str] = None,
    plot: bool = True,
) -> CampaignAnalysis:
    """Analyze campaign of one player against multiple opponents.

    Args:
        player_id: Player ID
        opponent_ids: List of opponent IDs
        database_url: Database connection string
        output_dir: Directory to save report and plot
        plot: Whether to generate plot

    Returns:
        CampaignAnalysis object
    """
    try:
        analysis = CampaignAnalysis(
            session=session,
            player=player,
            opponents=opponents,
        )

        print(analysis.report)

        if output_dir:
            # Save report
            report_path = Path(output_dir) / f"campaign_player_{player.id}.txt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as f:
                f.write(analysis.report)

            # Save plot
            if plot:
                plot_path = Path(output_dir) / f"campaign_player_{player.id}.png"
                analysis.plot_scores(str(plot_path))

        return analysis

    finally:
        session.close()


if __name__ == "__main__":
    with get_session() as session:
        player = get_or_create_random_player(
            session=session,
            seed=1,
        )
        opponents = [
            get_or_create_random_player(
                session=session,
                seed=2,
            ),
            get_or_create_random_player(
                session=session,
                seed=3,
            ),
        ]
        analyze_campaign(
            session=session,
            player=player,
            opponents=opponents,
            output_dir="analysis_results",
        )
