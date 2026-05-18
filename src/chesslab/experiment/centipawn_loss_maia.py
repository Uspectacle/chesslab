"""Experiment analyzing centipawn loss distribution for a Maia player."""

import logging
from pathlib import Path

import structlog
from matplotlib import pyplot as plt

from chesslab.analysis.centipawn_analysis import CentipawnAnalysis, plot_distribution
from chesslab.analysis.evaluator import Evaluator
from chesslab.engines.init_engines import get_maia_player
from chesslab.storage import get_session

logger = structlog.get_logger()


if __name__ == "__main__":
    logger.info("Starting centipawn loss analysis for Maia 1100")
    folder = Path(__file__).parent / "results/centipawn_loss_maia"
    folder.mkdir(parents=True, exist_ok=True)

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )

    with get_session() as session:
        # Get Maia player at 1100 ELO
        player = get_maia_player(session=session, elo=1100)

        num_games = 1000  # None

        logger.info(
            "Loaded player",
            player_id=player.id,
            engine_type=player.engine_type,
            expected_elo=player.expected_elo,
        )

        with Evaluator() as evaluator:
            # Create centipawn analysis
            logger.info("Starting centipawn loss analysis")
            cp_analysis = CentipawnAnalysis(
                session=session, evaluator=evaluator, player=player, num_games=num_games
            )

            logger.info(
                "Centipawn analysis completed",
                num_games=len(cp_analysis.games_analysis),
            )

            # Analyze per-move losses (5 centipawn bins, log-scale)
            logger.info("Analyzing per-move centipawn losses")
            per_move_result = cp_analysis.analyze_distribution(
                cp_analysis.all_moves_losses, log_scale=True
            )
            logger.info(
                "Per-move analysis complete",
                mean=per_move_result["mean"],
                std_dev=per_move_result["std_dev"],
                r_squared=per_move_result["r_squared"],
            )

            # Analyze per-game losses (10 centipawn bins)
            logger.info("Analyzing per-game centipawn losses")
            per_game_result = cp_analysis.analyze_distribution(
                cp_analysis.per_game_losses
            )
            logger.info(
                "Per-game analysis complete",
                mean=per_game_result["mean"],
                std_dev=per_game_result["std_dev"],
                r_squared=per_game_result["r_squared"],
            )

            # Analyze per-opponent losses (10 centipawn bins)
            logger.info("Analyzing per-opponent centipawn losses")
            opponent_elos, opponent_cpls = cp_analysis.per_opponent_losses
            per_opponent_result = cp_analysis.analyze_distribution(opponent_cpls)
            logger.info(
                "Per-opponent analysis complete",
                mean=per_opponent_result["mean"],
                std_dev=per_opponent_result["std_dev"],
                r_squared=per_opponent_result["r_squared"],
            )

            # Create figure with 3 subplots
            _fig, axes = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
                3, 1, figsize=(10, 12)
            )

            # Plot 1: Per-move losses
            plot_distribution(
                axes[0],  # pyright: ignore[reportUnknownMemberType]
                per_move_result,
                "Centipawn Loss per Move",
                "Centipawn Loss",
                log_scale=True,
            )

            # Plot 2: Per-game losses
            plot_distribution(
                axes[1],  # pyright: ignore[reportUnknownMemberType]
                per_game_result,
                "Average Centipawn Loss per Game",
                "Average Centipawn Loss",
            )

            # Plot 3: Per-opponent losses
            plot_distribution(
                axes[2],  # pyright: ignore[reportUnknownMemberType]
                per_opponent_result,
                "Average Centipawn Loss per Opponent",
                "Average Centipawn Loss",
            )

            plt.tight_layout()

            plot_path = folder / "centipawn_loss_analysis_maia1100.png"
            plt.savefig(plot_path)  # pyright: ignore[reportUnknownMemberType]
            plt.close()
            logger.info(f"Plot saved at {plot_path}")

            # Save detailed report
            report_path = folder / "centipawn_loss_report_maia1100.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("=" * 70 + "\n")
                f.write(
                    f"Centipawn Loss Analysis Report: {player.engine_type} {player.expected_elo} ELO\n"
                )
                f.write("=" * 70 + "\n\n")

                f.write(f"Player ID: {player.id}\n")
                f.write(
                    f"Number of games analyzed: {len(cp_analysis.games_analysis)}\n"
                )
                f.write(f"Total moves analyzed: {per_move_result['num_samples']}\n\n")

                f.write("-" * 70 + "\n")
                f.write("PER-MOVE CENTIPAWN LOSS (5 centipawn bins)\n")
                f.write("-" * 70 + "\n")
                f.write(f"Mean: {per_move_result['mean']:.2f}\n")
                f.write(f"Std Dev: {per_move_result['std_dev']:.2f}\n")
                f.write(f"R²: {per_move_result['r_squared']:.6f}\n")
                f.write(f"Sample size: {per_move_result['num_samples']}\n\n")

                f.write("-" * 70 + "\n")
                f.write("PER-GAME AVERAGE CENTIPAWN LOSS (10 centipawn bins)\n")
                f.write("-" * 70 + "\n")
                f.write(f"Mean: {per_game_result['mean']:.2f}\n")
                f.write(f"Std Dev: {per_game_result['std_dev']:.2f}\n")
                f.write(f"R²: {per_game_result['r_squared']:.6f}\n")
                f.write(f"Sample size: {per_game_result['num_samples']}\n\n")

                f.write("-" * 70 + "\n")
                f.write("PER-OPPONENT AVERAGE CENTIPAWN LOSS (10 centipawn bins)\n")
                f.write("-" * 70 + "\n")
                f.write(f"Mean: {per_opponent_result['mean']:.2f}\n")
                f.write(f"Std Dev: {per_opponent_result['std_dev']:.2f}\n")
                f.write(f"R²: {per_opponent_result['r_squared']:.6f}\n")
                f.write(f"Sample size: {per_opponent_result['num_samples']}\n")
                f.write(f"Opponent Elos analyzed: {sorted(opponent_elos)}\n")

            logger.info(f"Report saved at {report_path}")
