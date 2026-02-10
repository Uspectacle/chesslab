import logging
from pathlib import Path

import structlog
from matplotlib import pyplot as plt

from chesslab.analysis.analyze_range import RangeAnalysis
from chesslab.analysis.evaluator import Evaluator
from chesslab.arena.run_match import run_range
from chesslab.engines.init_engines import get_stockfish_range
from chesslab.storage import get_session

logger = structlog.get_logger()

num_games = 100

if __name__ == "__main__":
    logger.info("Starting coherence stockfish script")
    folder = Path(__file__).parent / "results/coherence_stockfish"
    folder.mkdir(parents=True, exist_ok=True)

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )

    with get_session() as session:
        players = get_stockfish_range(session=session, num_step=5)

        opponents = get_stockfish_range(session=session)

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

            # for range_analysis in ranges_analysis:
            #     report_path = folder / f"player_{range_analysis.player.id}.txt"

            #     with open(report_path, "w", encoding="utf-8") as f:
            #         f.write(range_analysis.report)

            #     logger.info(f"Report created at {report_path}")

            num_subplot = len(players)

            _fig, axes = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
                num_subplot, 1, figsize=(7, 3 * num_subplot), sharex=True
            )

            ax_list = axes if num_subplot > 1 else [axes]  # pyright: ignore[reportAssignmentType]

            for ax, range_analysis in zip(ax_list, ranges_analysis):
                range_analysis.plot_score_on_ax(ax)
                ax.set_title(f"Stockfish {range_analysis.player.expected_elo} ELO")
                logger.info(f"Stockfish {range_analysis.player.expected_elo} plotted")

            ax_list[-1].set_xlabel("Opponent Stockfish Elo")  # pyright: ignore[reportUnknownMemberType]
            plt.tight_layout()

            plot_path = folder / "plot.png"
            plt.savefig(plot_path)  # pyright: ignore[reportUnknownMemberType]
            plt.close()
            logger.info(f"Plot saved at {plot_path}")
