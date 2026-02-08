import random
from collections import defaultdict
from typing import Callable, Dict, List

import chess


def majority(votes: List[str], weights: List[float], board: chess.Board) -> str:
    """Selects the move with the highest total weighted sum of votes."""
    score_map = defaultdict(float)
    for move, weight in zip(votes, weights):
        score_map[move] += weight
    max_score = max(score_map.values())
    winners = [move for move, score in score_map.items() if score == max_score]
    return random.choice(winners)


def minority(votes: List[str], weights: List[float], board: chess.Board) -> str:
    """Picks a legal move that NO ONE suggested or selects the move with the lowest total weighted sum of votes."""
    legal_moves = [m.uci() for m in board.legal_moves]
    unpopular_moves = [m for m in legal_moves if m not in votes]

    if unpopular_moves:
        return random.choice(unpopular_moves)

    score_map = defaultdict(float)
    for move, weight in zip(votes, weights):
        score_map[move] += weight
    min_score = min(score_map.values())
    winners = [move for move, score in score_map.items() if score == min_score]
    return random.choice(winners)


def randomized(votes: List[str], weights: List[float], board: chess.Board) -> str:
    """Probability-based selection: higher weight moves have a better chance."""
    return random.choices(votes, weights=weights)[0]


def rotating(votes: List[str], weights: List[float], board: chess.Board) -> str:
    """Cycles through players move-by-move, weighted by their influence."""
    scaled_indices = []
    for idx, weight in enumerate(weights):
        scaled_indices.extend([idx] * round(weight))

    current_index = scaled_indices[board.ply() % len(scaled_indices)]

    return votes[current_index]


Aggregator = Callable[[List[str], List[float], chess.Board], str]

AGGREGATORS: Dict[str, Aggregator] = {
    "majority": majority,
    "minority": minority,
    "randomized": randomized,
    "rotating": rotating,
}
