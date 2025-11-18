import random
from collections import Counter
from typing import Callable, Dict, List

import chess

from chesslab.storage import Player


def majority(votes: List[str], players: List[Player], board: chess.Board) -> str:
    vote_counts = Counter(votes)

    max_votes = max(vote_counts.values())
    winners = [move for move, count in vote_counts.items() if count == max_votes]

    return random.choice(winners)


def minority(votes: List[str], players: List[Player], board: chess.Board) -> str:
    vote_counts = Counter(votes)

    max_votes = min(vote_counts.values())
    winners = [move for move, count in vote_counts.items() if count == max_votes]

    return random.choice(winners)


def randomized(votes: List[str], players: List[Player], board: chess.Board) -> str:
    return random.choice(votes)


def top_elo_dictator(
    votes: List[str], players: List[Player], board: chess.Board
) -> str:
    vote, _dictator = max(
        zip(votes, players), key=lambda vote_player: vote_player[1].expected_elo
    )

    return vote


def rotating_dictator(
    votes: List[str], players: List[Player], board: chess.Board
) -> str:
    return votes[board.ply() % len(votes)]


def elo_weight(votes: List[str], players: List[Player], board: chess.Board) -> str:
    weight_counter: Counter[str] = Counter()

    for vote, player in zip(votes, players):
        weight_counter[vote] += player.expected_elo

    moves, weights = zip(*weight_counter.items())
    probs = [weight / sum(weights) for weight in weights]

    return random.choices(moves, weights=probs, k=1)[0]


def contrarian(votes: List[str], players: List[Player], board: chess.Board) -> str:
    moves = [move.uci() for move in board.legal_moves]
    no_votes = [move for move in moves if move not in votes]

    if len(no_votes):
        return random.choice(no_votes)

    return minority(votes=votes, players=players, board=board)


Aggregator = Callable[[List[str], List[Player], chess.Board], str]

AGGREGATORS: Dict[str, Aggregator] = {
    "majority": majority,
    "minority": minority,
    "randomized": randomized,
    "top_elo_dictator": top_elo_dictator,
    "rotating_dictator": rotating_dictator,
    "elo_weight": elo_weight,
    "contrarian": contrarian,
}
