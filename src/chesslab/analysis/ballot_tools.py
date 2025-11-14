"""Tools for ballots"""

import random

from chesslab.analysis.typing import Ballot


def normalize_ballot(ballot: Ballot) -> Ballot:
    """
    Normalize the scores in a ballot each score is positive,
    the total sum equals 1, then shuffle the resulting dictionary order.

    Args:
        ballot (Ballot): A dictionary mapping moves (or options) to their scores.

    Returns:
        Ballot: A new dictionary with normalized and shuffled scores,
        where the sum of all values is 1.0.
    """
    minimum = min(ballot.values())
    positive_ballot = {move: score - minimum for move, score in ballot.items()}

    total = sum(positive_ballot.values())

    if total == 0:
        normalized = {move: 1 / len(positive_ballot) for move in positive_ballot.keys()}
    else:
        normalized = {move: score / total for move, score in positive_ballot.items()}

    items = list(normalized.items())
    random.shuffle(items)

    return dict(items)
