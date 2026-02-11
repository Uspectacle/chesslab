import logging
from pathlib import Path

import structlog
from matplotlib import pyplot as plt

from chesslab.analysis.analyze_range import RangeAnalysis
from chesslab.analysis.evaluator import Evaluator
from chesslab.arena.run_match import run_range
from chesslab.engines.init_engines import (
    get_madchess_range,
    get_voting_player,
)
from chesslab.storage import get_session

logger = structlog.get_logger()


if __name__ == "__main__":
    logger.info("Aggregation variation script")
    folder = Path(__file__).parent / "results/aggregation_variation"
    folder.mkdir(parents=True, exist_ok=True)

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )

    num_games = 10
    aggregator = "majority"
    crowd_kind = "MadChess gaussian"
    crowd_mean_elo = 1500
    crowd_std_dev = 200

    with get_session() as session:
        crowd_path = folder / "crowd.txt"

        options = {
            session: session,
            aggregator: aggregator,
            crowd_kind: crowd_kind,
            crowd_std_dev: crowd_std_dev,
            crowd_mean_elo: crowd_mean_elo,
        }
        players = [
            get_voting_player(crowd_size=1, **options),
            get_voting_player(crowd_size=2, **options),
            get_voting_player(crowd_size=4, **options),
            get_voting_player(crowd_size=8, **options),
            get_voting_player(crowd_size=16, **options),
        ]
        names = [
            f"Crowd size = {player.options.get('Crowd_size')}" for player in players
        ]
        opponents = get_madchess_range(
            session=session, min_elo=800, max_elo=2000, num_step=7
        )

        run_range(
            session=session,
            players=players,
            opponents=opponents,
            num_games=num_games,
            remove_existing=False,
            get_existing=True,
            alternate_color=True,
        )

        with Evaluator() as evaluator:
            logger.info("Start analyse")
            ranges_analysis = [
                RangeAnalysis(
                    session=session,
                    evaluator=evaluator,
                    player=player,
                    opponents=opponents,
                    num_games=num_games,
                )
                for player in players
            ]

            num_subplot = len(players)

            _fig, axes = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
                num_subplot, 1, figsize=(7, 3 * num_subplot), sharex=True
            )

            ax_list = axes if num_subplot > 1 else [axes]  # pyright: ignore[reportAssignmentType]

            for ax, range_analysis, name in zip(ax_list, ranges_analysis, names):
                range_analysis.plot_score_on_ax(ax, ignore_declaration=True)
                ax.set_title(name)
                logger.info(f"{name} plotted")

            ax_list[-1].set_xlabel("Opponent MadChess Elo")  # pyright: ignore[reportUnknownMemberType]
            plt.tight_layout()

            plot_path = folder / "plot.png"
            plt.savefig(plot_path)  # pyright: ignore[reportUnknownMemberType]
            plt.close()
            logger.info(f"Plot saved at {plot_path}")
