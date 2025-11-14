"""Helpers to compute statistics."""

import math
from typing import List

from scipy import stats


def standard_error_of_proportion(
    proportion: float,
    num_round: int,
) -> float:
    """Compute the standard error of a sample proportion.

    Args:
        proportion: Observed or expected proportion value (between 0 and 1).
        num_round: Number of rounds or trials used to estimate the proportion.

    Returns:
        Standard error of the proportion. Returns NaN if num_round is zero.
    """
    if num_round == 0:
        return float("nan")

    return math.sqrt(proportion * (1 - proportion) / num_round)


def compute_test_statistic(
    observed_value: float,
    expected_value: float,
    standard_error: float,
) -> float:
    """Compute the test statistic for a hypothesis test on proportions.

    The test statistic is computed as the absolute difference between the
    observed and expected values divided by the standard error.

    Args:
        observed_value: Observed sample proportion or value.
        expected_value: Expected proportion or theoretical mean under H₀.
        standard_error: Standard error of the expected statistic.

    Returns:
        Absolute test statistic value. Returns NaN if standard_error is zero or NaN.
    """
    if math.isnan(standard_error) or standard_error == 0:
        return float("nan")

    return abs(observed_value - expected_value) / standard_error


def compute_p_value(
    observed_value: float,
    expected_value: float,
    num_round: int,
    test: str | None = None,
) -> float:
    """Compute a two-tailed p-value for a proportion test.

    Automatically selects a z-test or t-test depending on sample size
    and expected frequency criteria unless specified.

    Args:
        observed_value: Observed proportion from data.
        expected_value: Expected proportion under the null hypothesis.
        num_round: Number of trials or rounds in the experiment.
        test: Optional; 'z' for z-test, 't' for t-test, or None for automatic choice.

    Returns:
        Two-tailed p-value. Returns NaN if inputs are invalid.
    """
    standard_error = standard_error_of_proportion(
        proportion=expected_value, num_round=num_round
    )
    test_statistic = compute_test_statistic(
        observed_value=observed_value,
        expected_value=expected_value,
        standard_error=standard_error,
    )

    if not test:
        if expected_value * num_round >= 5 and (1 - expected_value) * num_round >= 5:
            test = "z"
        else:
            test = "t"

    if test == "z":
        cdf = stats.norm.cdf(test_statistic)
    elif test == "t":
        cdf = stats.t.cdf(test_statistic, df=num_round - 1)
    else:
        return float("nan")

    return float(2 * (1 - cdf))


def ensemble_mean_score(
    observed_scores: List[float],
    num_rounds_per_match: List[int],
) -> float:
    """Compute the weighted mean score across multiple matches.

    Args:
        observed_scores: List of mean scores from each match
        num_rounds_per_match: List of number of rounds in each match

    Returns:
        Weighted mean score across all matches
    """
    total_rounds = sum(num_rounds_per_match)

    if total_rounds == 0:
        return float("nan")

    weighted_sum = sum(
        score * n_rounds
        for score, n_rounds in zip(observed_scores, num_rounds_per_match)
    )

    return weighted_sum / total_rounds


def ensemble_standard_error(
    expected_scores: List[float],
    num_rounds_per_match: List[int],
) -> float:
    """Compute the standard error for ensemble of matches with different expected scores.

    For multiple independent matches with different expected probabilities,
    the variance of the pooled proportion is:
    Var(p_pooled) = sum(n_i * p_i * (1 - p_i)) / N^2
    where n_i is the number of rounds in match i, p_i is the expected score, and N is total rounds.

    Args:
        expected_scores: List of expected mean scores for each match
        num_rounds_per_match: List of number of rounds in each match

    Returns:
        Standard error of the ensemble mean score
    """
    total_rounds = sum(num_rounds_per_match)

    if math.isnan(total_rounds) or total_rounds == 0:
        return float("nan")

    variance_sum = sum(
        n_rounds * expected_score * (1 - expected_score)
        for expected_score, n_rounds in zip(expected_scores, num_rounds_per_match)
    )

    variance = variance_sum / (total_rounds**2)

    return math.sqrt(variance)


def compute_ensemble_p_value(
    observed_scores: List[float],
    expected_scores: List[float],
    num_rounds_per_match: List[int],
    test: str | None = None,
) -> float:
    """Compute p-value for ensemble of matches with potentially different expected scores.

    Args:
        observed_scores: List of observed mean scores from each match
        expected_scores: List of expected mean scores for each match
        num_rounds_per_match: List of number of rounds in each match
        test: 'z' for z-test, 't' for t-test, None for automatic selection

    Returns:
        Two-tailed p-value
    """
    observed_mean = ensemble_mean_score(observed_scores, num_rounds_per_match)
    expected_mean = ensemble_mean_score(expected_scores, num_rounds_per_match)
    standard_error = ensemble_standard_error(expected_scores, num_rounds_per_match)

    if math.isnan(standard_error) or standard_error == 0:
        return float("nan")

    test_statistic = abs(observed_mean - expected_mean) / standard_error

    total_rounds = sum(num_rounds_per_match)

    if not test:
        # Check if normal approximation is valid for the pooled data
        # Use the minimum n*p and n*(1-p) across all matches as a conservative check
        min_np = min(n * p for n, p in zip(num_rounds_per_match, expected_scores))
        min_nq = min(n * (1 - p) for n, p in zip(num_rounds_per_match, expected_scores))

        if min_np >= 5 and min_nq >= 5:
            test = "z"
        else:
            test = "t"

    if test == "z":
        cdf = stats.norm.cdf(test_statistic)
    elif test == "t":
        # Use total rounds - number of matches as degrees of freedom
        df = total_rounds - len(observed_scores)
        cdf = stats.t.cdf(test_statistic, df=max(1, df))
    else:
        return float("nan")

    return float(2 * (1 - cdf))
