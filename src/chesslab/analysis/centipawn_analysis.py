"""Analysis of centipawn loss distributions across games."""

import random
from typing import List, Optional, Tuple

import numpy as np
import structlog
from scipy import stats
from sqlalchemy.orm import Session

from chesslab.analysis.analyze_game import GameAnalysis
from chesslab.analysis.evaluator import Evaluator
from chesslab.analysis.stat_tools import coefficient_of_determination
from chesslab.storage import Player
from chesslab.storage.game_tools import get_player_games

logger = structlog.get_logger()


def bin_data(
    data: List[float], bin_size: float, log_scale: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin continuous data and create histogram.

    Args:
        data: List of values to bin
        bin_size: Size of each bin

    Returns:
        Tuple of (bin_edges, bin_centers, bin_counts)
    """
    data_array = np.array(data)
    # Filter out or clip zeros for log calculations
    min_val = max(0.1, np.min(data_array))
    max_val = np.max(data_array)

    if log_scale:
        # Create bins that are equal width in LOG space
        # We calculate how many bins we need based on the 'bin_size' logic
        num_bins = int((np.log10(max_val) - np.log10(min_val)) / 0.1)  # 0.1 is log-step
        bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), num=num_bins)
    else:
        # Standard linear bins
        min_bin = np.floor(np.min(data_array) / bin_size) * bin_size
        max_bin = np.ceil(max_val / bin_size) * bin_size
        bin_edges = np.arange(min_bin, max_bin + bin_size, bin_size)

    bin_counts, _ = np.histogram(data_array, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_edges, bin_centers, bin_counts


def fit_gaussian(
    data: List[float], bin_size: float, log_scale: bool = False
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Fit a Gaussian distribution to binned data.

    Args:
        data: List of values to fit
        bin_size: Size of each bin (in linear or log space)
        log_scale: If True, create logarithmically-spaced bins

    Returns:
        Tuple of (mean, std_dev, bin_centers, fitted_values)
    """
    bin_edges, bin_centers, bin_counts = bin_data(data, bin_size, log_scale=log_scale)

    # Fit Gaussian using MLE on the raw data
    mean_est, std_est = stats.norm.fit(data)

    # Generate fitted distribution at bin centers
    fitted_values = stats.norm.pdf(bin_centers, mean_est, std_est)

    # Scale to match histogram (normalize histogram first)
    total_count = np.sum(bin_counts)
    if total_count > 0:
        normalized_counts = bin_counts / total_count
        # Find scaling factor to match the histogram
        max_hist = np.max(normalized_counts)
        max_fit = np.max(fitted_values)
        if max_fit > 0:
            fitted_values = fitted_values / max_fit * max_hist

    return mean_est, std_est, bin_centers, fitted_values


def compute_gaussian_stats(
    observed_counts: np.ndarray, predicted_values: np.ndarray
) -> float:
    """Compute R² for Gaussian fit.

    Args:
        observed_counts: Observed histogram counts
        predicted_values: Predicted values from Gaussian fit

    Returns:
        R² value (coefficient of determination)
    """
    if len(observed_counts) != len(predicted_values):
        raise ValueError("observed_counts and predicted_values must have same length")

    # Normalize for comparison
    observed_norm = observed_counts / (np.sum(observed_counts) + 1e-10)
    predicted_norm = predicted_values / (np.sum(predicted_values) + 1e-10)

    # Compute R²
    r_squared = coefficient_of_determination(list(observed_norm), list(predicted_norm))

    return r_squared


def plot_distribution(ax, analysis_result, title, xlabel, log_scale=None):
    """Plot a centipawn loss distribution with Gaussian fit.

    Args:
        ax: Matplotlib axis
        analysis_result: Dictionary from analyze_distribution()
        title: Plot title
        xlabel: X-axis label
        log_scale: If True, use logarithmic scale for x-axis and bins.
                  If None, use the log_scale flag from analysis_result if available.
    """
    # Use log_scale from result if not explicitly provided
    if log_scale is None:
        log_scale = analysis_result.get("log_scale", False)

    if not analysis_result["bin_counts"].size:
        return

    bin_edges = analysis_result["bin_edges"]
    bin_centers = analysis_result["bin_centers"]
    bin_counts = analysis_result["bin_counts"]
    fitted_gaussian = analysis_result["fitted_gaussian"]

    # Calculate widths for each bar individually
    widths = np.diff(bin_edges)

    # Plot histogram
    ax.bar(
        bin_centers,
        bin_counts,
        width=widths,  # Use calculated widths
        alpha=0.7,
        label="Observed",
        edgecolor="black",
        align="center",
    )

    # Plot fitted Gaussian
    ax.plot(bin_centers, fitted_gaussian, "r-", linewidth=2, label="Gaussian fit")

    # Add statistics to plot
    mean = analysis_result["mean"]
    std_dev = analysis_result["std_dev"]
    r_squared = analysis_result["r_squared"]
    num_samples = analysis_result["num_samples"]

    stats_text = (
        f"μ = {mean:.1f}\nσ = {std_dev:.1f}\nR² = {r_squared:.4f}\nn = {num_samples}"
    )
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=9,
    )

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    if log_scale:
        ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)


class CentipawnAnalysis:
    """Analyze centipawn loss distribution across all games of a player."""

    def __init__(
        self,
        session: Session,
        evaluator: Evaluator,
        player: Player,
        num_games: Optional[int] = None,
    ) -> None:
        """Initialize with a player and load their games.

        Args:
            session: Database session
            evaluator: Evaluator instance for computing CPL
            player: The player to analyze
            num_games: Optional limit on number of games to analyze
        """
        self.player = player
        self.evaluator = evaluator
        self.session = session

        # Get all games for the player
        all_games = get_player_games(session=session, player_id=player.id)
        if num_games:
            all_games = random.sample(all_games, min(num_games, len(all_games)))

        # Analyze each game
        self.games_analysis = [
            GameAnalysis(game=game, evaluator=evaluator)
            for game in all_games
            if game.result  # Only completed games
        ]

        logger.info(
            "CentipawnAnalysis initialized",
            player_id=player.id,
            num_games=len(self.games_analysis),
        )

    @property
    def all_moves_losses(self) -> List[float]:
        """Get all centipawn losses for each move by the player."""
        all_losses = []
        for game_analysis in self.games_analysis:
            losses = game_analysis.get_player_moves_centipawn_losses(self.player.id)
            all_losses.extend(losses)
        return all_losses

    @property
    def per_game_losses(self) -> List[float]:
        """Get average centipawn loss per game."""
        game_losses = []
        for game_analysis in self.games_analysis:
            average_cpl = game_analysis.get_average_centipawn_loss(self.player.id)
            if not np.isnan(average_cpl):
                game_losses.append(average_cpl)
        return game_losses

    @property
    def per_opponent_losses(self) -> Tuple[List[int], List[float]]:
        """Get average centipawn loss per opponent.

        Returns:
            Tuple of (opponent_elos, average_cpl_per_opponent)
        """
        opponent_stats = {}

        for game_analysis in self.games_analysis:
            # Determine opponent
            if game_analysis.game.white_player_id == self.player.id:
                opponent = game_analysis.game.black_player
            else:
                opponent = game_analysis.game.white_player

            opponent_elo = opponent.expected_elo
            average_cpl = game_analysis.get_average_centipawn_loss(self.player.id)

            if not np.isnan(average_cpl):
                if opponent_elo not in opponent_stats:
                    opponent_stats[opponent_elo] = []
                opponent_stats[opponent_elo].append(average_cpl)

        # Compute average CPL per opponent Elo
        opponent_elos = sorted(opponent_stats.keys())
        average_cpls = [np.mean(opponent_stats[elo]) for elo in opponent_elos]

        return opponent_elos, average_cpls

    def analyze_distribution(
        self, data: List[float], bin_size: float = 5, log_scale: bool = False
    ) -> dict:
        """Analyze a centipawn loss distribution.

        Args:
            data: List of centipawn loss values
            bin_size: Size of bins for histogram
            log_scale: If True, create logarithmically-spaced bins

        Returns:
            Dictionary with analysis results
        """
        if not data:
            logger.warning("No data provided for distribution analysis")
            return {
                "mean": np.nan,
                "std_dev": np.nan,
                "r_squared": np.nan,
                "bin_centers": [],
                "bin_counts": [],
                "fitted_gaussian": [],
                "log_scale": log_scale,
            }

        # Bin the data
        bin_edges, bin_centers, bin_counts = bin_data(
            data, bin_size, log_scale=log_scale
        )

        # Fit Gaussian
        mean_est, std_est, _, fitted_values = fit_gaussian(
            data, bin_size, log_scale=log_scale
        )

        # Normalize bin counts to probability density for comparison with Gaussian PDF
        total_count = np.sum(bin_counts)
        normalized_counts = bin_counts / total_count if total_count > 0 else bin_counts

        # Compute R² using normalized values
        r_squared = compute_gaussian_stats(normalized_counts, fitted_values)

        return {
            "mean": mean_est,
            "std_dev": std_est,
            "r_squared": r_squared,
            "data": data,
            "bin_edges": bin_edges,
            "bin_centers": bin_centers,
            "bin_counts": normalized_counts,
            "fitted_gaussian": fitted_values,
            "num_samples": len(data),
            "log_scale": log_scale,
        }
