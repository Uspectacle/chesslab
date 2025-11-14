"""Experience: Calibrate Stockfish ELO levels (per-game statistical evaluation)."""

from typing import List

import numpy as np
from scipy.optimize import minimize_scalar


def expected_score(elo_a: float, elo_b: float) -> float:
    """Compute expected score for player A against B using the Elo formula.

    Args:
        elo_a: Elo of player A.
        elo_b: Elo of player B.

    Returns:
        Expected score (probability of winning) for player A.
    """
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def elo_from_mean_score(mean_score: float, opponent_elo: float) -> float:
    """Estimate Elo from mean score against a single opponent.

    Args:
        mean_score: Observed mean score (0 to 1)
        opponent_elo: Known Elo of the opponent

    Returns:
        Estimated Elo rating
    """
    # Avoid division by zero / log of 0
    eps = 1e-9
    mean_score = np.clip(mean_score, eps, 1 - eps)

    elo = opponent_elo + 400 * np.log10(mean_score / (1 - mean_score))

    return np.clip(elo, 250, 3000)


def ensemble_elo_from_scores(
    observed_scores: List[float],
    opponent_elos: List[float],
    num_rounds_per_match: List[int],
) -> float:
    """Estimate player Elo from results against multiple opponents.

    This finds the Elo rating that minimizes the squared difference between
    observed and expected scores, weighted by the number of rounds in each match.

    Args:
        observed_scores: List of observed mean scores from each match
        opponent_elos: List of opponent Elo ratings
        num_rounds_per_match: List of number of rounds in each match

    Returns:
        Estimated Elo rating that best fits the ensemble of results
    """
    if len(observed_scores) != len(opponent_elos) or len(observed_scores) != len(
        num_rounds_per_match
    ):
        raise ValueError("All input lists must have the same length")

    if len(observed_scores) == 0:
        return float("nan")

    # If only one opponent, use the simple formula
    if len(observed_scores) == 1:
        return elo_from_mean_score(observed_scores[0], opponent_elos[0])

    def objective(player_elo: float) -> float:
        """Weighted sum of squared errors between observed and expected scores."""
        total_error = 0.0
        total_weight = sum(num_rounds_per_match)

        for obs_score, opp_elo, n_rounds in zip(
            observed_scores, opponent_elos, num_rounds_per_match
        ):
            expected = expected_score(player_elo, opp_elo)
            weight = n_rounds / total_weight
            total_error += weight * (obs_score - expected) ** 2

        return total_error

    # Initial guess: weighted average of individual Elo estimates
    # initial_elos = [
    #     elo_from_mean_score(score, opp_elo)
    #     for score, opp_elo in zip(observed_scores, opponent_elos)
    # ]
    # total_rounds = sum(num_rounds_per_match)
    # initial_guess = sum(
    #     elo * n_rounds / total_rounds
    #     for elo, n_rounds in zip(initial_elos, num_rounds_per_match)
    # )

    # Minimize the objective function
    result = minimize_scalar(
        objective,
        bounds=(250, 3000),
        method="bounded",
        options={"xatol": 0.1},  # Elo precision to 0.1
    )

    return float(np.clip(result.x, 250, 3000))
