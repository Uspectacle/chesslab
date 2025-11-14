"""Chess Game manager using Player instances."""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from chesslab.analysis.campaign import Campaign
from chesslab.analysis.typing import Player


class Tournament:
    """Manages a series of match between teams of players."""

    players: List[Player]
    opponents: List[Player]
    num_round: int
    max_moves: int
    result_folder: Path
    export_csv_log: bool
    export_pgn: bool
    verbose: bool

    campaigns: List[Campaign]

    def __init__(
        self,
        players: List[Player],
        opponents: Optional[List[Player]] = None,
        fix_players_to_white: bool = False,
        num_round: int = 1,
        max_moves: int = 200,
        result_folder: str | Path = "results",
        export_csv_log: bool = False,
        export_pgn: bool = False,
        verbose: bool = False,
        export_report: bool = False,
        export_plot_scores: bool = False,
        export_campaign_report: bool = False,
        export_campaign_plot_scores: bool = False,
    ) -> None:
        self.player = players
        self.opponents = opponents if opponents else players

        self.num_round = num_round
        self.max_moves = max_moves
        self.result_folder = Path(result_folder)
        self.export_csv_log = export_csv_log
        self.export_pgn = export_pgn
        self.verbose = verbose

        # Run the games and get results
        self.campaigns = [
            Campaign(
                player=player,
                opponents=self.opponents,
                fix_player_to_white=fix_players_to_white,
                num_round=num_round,
                max_moves=max_moves,
                result_folder=result_folder,
                export_csv_log=export_csv_log,
                export_pgn=export_pgn,
                export_plot_scores=export_campaign_plot_scores,
                export_report=export_campaign_report,
                verbose=verbose,
            )
            for player in self.player
        ]

        if export_report:
            self.write_report()

        if export_plot_scores:
            self.plot_scores()

    @property
    def report(self) -> str:
        """Textual report of all matches."""
        report_lines: List[str] = []

        for campaign in self.campaigns:
            report_lines.append(campaign.report)
            report_lines.append(f"\n{'.' * 50}\n\n")

        return "".join(report_lines)

    def write_report(self) -> None:
        """Write a textual report of all matches."""
        self.result_folder.mkdir(parents=True, exist_ok=True)
        report_path = f"{self.result_folder}/tournament_report.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== Tournament Report ===\n")
            f.write(self.report)

    @property
    def opponent_elos(self) -> List[float]:
        """The elo for each opponent."""
        return [opponent.elo for opponent in self.opponents]

    def plot_scores(self) -> None:
        """Plot statistics for all white ELO levels."""
        num_subplot = len(self.campaigns)

        _fig: Figure
        axes: Axes | List[Axes]

        _fig, axes = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
            num_subplot, 1, figsize=(10, 4 * num_subplot), sharex=True
        )

        ax_list: List[Axes] = axes if num_subplot > 1 else [axes]  # pyright: ignore[reportAssignmentType]

        for ax, campaign in zip(ax_list, self.campaigns):
            campaign.plot_score_on_ax(ax)

        ax_list[-1].set_xlabel("Opponent Elo")  # pyright: ignore[reportUnknownMemberType]
        plt.tight_layout()
        self.result_folder.mkdir(parents=True, exist_ok=True)
        png_path = f"{self.result_folder}/tournament_scores.png"
        plt.savefig(png_path)  # pyright: ignore[reportUnknownMemberType]
        plt.close()
