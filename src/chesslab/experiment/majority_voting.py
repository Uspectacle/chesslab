import logging
from pathlib import Path

import structlog
from matplotlib import pyplot as plt

from chesslab.analysis.analyze_range import RangeAnalysis
from chesslab.analysis.evaluator import Evaluator
from chesslab.analysis.stat_tools import estimate_gaussian_std
from chesslab.arena.run_match import run_range
from chesslab.engines.init_engines import (
    get_stockfish_gaussian,
    get_stockfish_range,
    get_voting_player,
)
from chesslab.storage import get_session

logger = structlog.get_logger()


if __name__ == "__main__":
    logger.info("Starting majority voting script")
    folder = Path(__file__).parent / "results/majority_voting"
    folder.mkdir(parents=True, exist_ok=True)

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    )

    num_games = 20
    mean = 1600
    std_dev = 200
    seed = 49
    num_samples = 10

    with get_session() as session:
        crowd = get_stockfish_gaussian(
            session=session,
            mean=mean,
            std_dev=std_dev,
            num_samples=num_samples,
            seed=seed,
        )

        crowd_path = folder / "crowd.txt"
        expected_elos = [player.expected_elo for player in crowd]
        expected_elos.sort()
        true_mean = int(sum(expected_elos) / len(expected_elos))
        true_std = int(estimate_gaussian_std(expected_elos))
        report = f"[seed={seed}] Stockfish Gaussian {true_mean} (+/- {true_std}) ELO x {num_samples}\n\n"
        report += ", ".join([str(expected_elo) for expected_elo in expected_elos])
        with open(crowd_path, "w", encoding="utf-8") as f:
            f.write(report)

            logger.info(f"Crowd explanation at {crowd_path}")

        players = [
            get_voting_player(session=session, players=crowd, aggregator="randomized"),
            get_voting_player(
                session=session, players=crowd, aggregator="top_elo_dictator"
            ),
            get_voting_player(session=session, players=crowd, aggregator="elo_weight"),
            get_voting_player(session=session, players=crowd, aggregator="majority"),
        ]
        opponents = get_stockfish_range(
            session=session, min_elo=1320, max_elo=2200, num_step=3
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

            for range_analysis in ranges_analysis:
                range_analysis.report
                report_path = folder / f"player_{range_analysis.player.id}.txt"

                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(range_analysis.report)

                logger.info(f"Repport created at {report_path}")

            num_subplot = len(players)

            _fig, axes = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
                num_subplot, 1, figsize=(10, 4 * num_subplot), sharex=True
            )

            ax_list = axes if num_subplot > 1 else [axes]  # pyright: ignore[reportAssignmentType]

            for ax, range_analysis in zip(ax_list, ranges_analysis):
                range_analysis.plot_score_on_ax(ax)
                ax.set_title(
                    f"{range_analysis.player.options.get('Aggregator')} (ID: {range_analysis.player.id})"
                )

            ax_list[-1].set_xlabel("Opponent Elo")  # pyright: ignore[reportUnknownMemberType]
            plt.tight_layout()

            plot_path = folder / "plot.png"
            plt.savefig(plot_path)  # pyright: ignore[reportUnknownMemberType]
            plt.close()
            logger.info(f"Plot saved at {plot_path}")
