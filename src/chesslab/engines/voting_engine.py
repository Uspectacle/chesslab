import asyncio
import logging
from typing import List, Optional

import chess
import numpy as np
import structlog

from chesslab.engines.base_engine import BaseEngine
from chesslab.engines.init_engines import (
    get_madchess_player,
    get_maia_player,
    get_stockfish_player,
)
from chesslab.engines.options.aggregators import AGGREGATORS, Aggregator
from chesslab.engines.options.options import OptionCombo, OptionSpin, OptionString
from chesslab.engines.storage_tools import get_uci_move
from chesslab.storage import Player, get_session
from chesslab.storage.player_tools import get_player_by_id

logger = structlog.get_logger()

CROWD_KINDS = ["Stockfish gaussian", "MadChess gaussian", "Maia gaussian", "Explicit"]


async def get_votes(
    board: chess.Board, players: List[Player], max_concurrent: int
) -> List[str]:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def get_vote(
        semaphore: asyncio.Semaphore, board: chess.Board, player: Player
    ) -> str:
        async with semaphore:
            return await get_uci_move(board=board, player=player)

    return await asyncio.gather(
        *[get_vote(semaphore, board, player) for player in players]
    )


class VotingEngine(BaseEngine):
    """Chess engine that selects moves by majority vote from multiple engines.

    This engine manages a committee of chess engines and selects the move
    that receives the most votes. In case of a tie, a random move is chosen
    from the tied candidates.

    Attributes:
        name: Engine name identifier
        author: Engine author
        options: List of configurable options (Crowd_ids, Max_concurrent)
    """

    name = "VotingEngine"
    author = "ChessLab"
    options = [
        OptionSpin(name="Crowd_size", default=10, min=1, max=100),
        OptionSpin(name="Crowd_mean_elo", default=1500, min=300, max=2600),
        OptionSpin(name="Crowd_std_dev", default=200, min=0, max=2000),
        OptionSpin(name="Seed", default=0, min=0, max=2147483647),
        OptionCombo(name="Crowd_kind", default="Explicit", vars=CROWD_KINDS),
        OptionString(name="Crowd_ids", default="None"),
        OptionString(name="Weights", default="None"),
        OptionCombo(
            name="Aggregator", default="majority", vars=list(AGGREGATORS.keys())
        ),
        OptionSpin(name="Max_concurrent", default=1, min=1, max=8),
    ]

    def __init__(self) -> None:
        """Initialize the voting engine."""
        self._players: List[Player] = []
        self._session = None
        super().__init__()

    @property
    def seed(self) -> int:
        """Get the current random seed value."""
        option = self.get_option("Seed")
        if option is None:
            raise RuntimeError("Option Seed not found")

        return option.value

    @property
    def aggregator(self) -> Aggregator:
        """Get the maximum number of concurrent engine evaluations."""
        option = self.get_option("Aggregator")
        if option is None:
            raise RuntimeError("Option Aggregator not found")

        aggregator = AGGREGATORS.get(option.value)

        if not aggregator:
            raise ValueError(f"Aggregator {option.value} not defined")

        return aggregator

    @property
    def crowd_kind(self) -> str:
        """Get the crowd kind."""
        option = self.get_option("Crowd_kind")
        if option is None:
            raise RuntimeError("Option Crowd_kind not found")

        return option.value

    @property
    def crowd_size(self) -> int:
        if self.crowd_ids:
            return len(self.crowd_ids)

        option = self.get_option("Crowd_size")
        if option is None:
            raise RuntimeError("Option Crowd_size not found")

        return option.value

    @property
    def crowd_mean_elo(self) -> int:
        option = self.get_option("Crowd_mean_elo")
        if option is None:
            raise RuntimeError("Option Crowd_mean_elo not found")

        return option.value

    @property
    def crowd_std_dev(self) -> int:
        option = self.get_option("Crowd_std_dev")
        if option is None:
            raise RuntimeError("Option Crowd_std_dev not found")

        return option.value

    @property
    def max_concurrent(self) -> int:
        """Get the maximum number of concurrent engine evaluations."""
        option = self.get_option("Max_concurrent")
        if option is None:
            raise RuntimeError("Option Max_concurrent not found")

        return option.value

    @property
    def weights(self) -> List[float]:
        """Get the weights for each player.

        If no weights are given, they are set to 1 for each player.

        Returns:
            List of weights (one per player)

        Raises:
            ValueError: If weights list length doesn't match players length or if weights are invalid
        """
        if not len(self._players):
            raise RuntimeError("Players not initialized before accessing weights")

        option = self.get_option("Weights")

        if (
            option is None
            or not str(option.value).strip()
            or str(option.value).strip().lower() == "none"
        ):
            return [1.0 for _ in self._players]

        weights_str = str(option.value).strip()

        if weights_str.lower() == "elo":
            return [float(player.expected_elo) for player in self._players]

        if weights_str.startswith("[") and weights_str.endswith("]"):
            weights_str = weights_str[1:-1].strip()

        try:
            weights = [float(w.strip()) for w in weights_str.split(",")]
        except ValueError as e:
            raise ValueError(f"Weights must be comma-separated numbers: {e}")

        if len(weights) != len(self._players):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of players ({len(self._players)})"
            )

        return weights

    @property
    def crowd_ids(self) -> Optional[List[int]]:
        """Get the list of player IDs from the Crowd_ids option.

        Returns:
            List of player IDs to use for voting, or None if should generate from Gaussian
        """
        option = self.get_option("Crowd_ids")

        if (
            option is None
            or not str(option.value).strip()
            or str(option.value).strip().lower() == "none"
        ):
            # If no explicit IDs, return None - players will be generated from Gaussian
            return None

        player_ids_str = str(option.value).strip()

        if player_ids_str.startswith("[") and player_ids_str.endswith("]"):
            player_ids_str = player_ids_str[1:-1].strip()

        return [int(player_id.strip()) for player_id in player_ids_str.split(",")]

    def reset(self) -> None:
        """Reset the board, session, and players."""
        self.reset_session()
        self.reset_players()
        super().reset()

    def reset_players(self) -> None:
        """Load or generate all players.

        If Crowd_ids is specified, load players from database by ID.
        Otherwise, generate players using Gaussian distribution based on Crowd_kind.

        Raises:
            ValueError: If any player ID is not found in the database or if Crowd_kind is invalid
        """
        assert self._session, "Session must be initialized before loading players"

        self.quit_players()

        crowd_ids = self.crowd_ids

        # If explicit IDs are provided, load from database
        if crowd_ids is not None:
            players: List[Player] = []

            for player_id in crowd_ids:
                player = get_player_by_id(self._session, player_id)

                if player is None:
                    raise ValueError(
                        f"Player with ID {player_id} not found in database"
                    )

                players.append(player)

                logger.debug(
                    "Player loaded",
                    player_id=player_id,
                    engine_type=player.engine_type,
                    expected_elo=player.expected_elo,
                )

            self._players = players
        else:
            # Generate players from Gaussian distribution
            crowd_kind = self.crowd_kind
            rng = np.random.default_rng(self.seed) if self.seed else np.random
            sampled_elos = rng.normal(
                loc=self.crowd_mean_elo, scale=self.crowd_std_dev, size=self.crowd_size
            )

            if crowd_kind == "Stockfish gaussian":
                self._players = [
                    get_stockfish_player(session=self._session, elo=elo)
                    for elo in sampled_elos
                ]
            elif crowd_kind == "MadChess gaussian":
                self._players = [
                    get_madchess_player(session=self._session, elo=elo)
                    for elo in sampled_elos
                ]
            elif crowd_kind == "Maia gaussian":
                self._players = [
                    get_maia_player(session=self._session, elo=elo)
                    for elo in sampled_elos
                ]
            else:
                raise ValueError(
                    f"Invalid Crowd_kind: {crowd_kind}. "
                    f"Must be one of: {CROWD_KINDS} or set Crowd_ids explicitly"
                )

    def reset_session(self) -> None:
        """Reset the database session."""
        self.quit_session()
        self._session = get_session().__enter__()
        logger.debug("Database session opened")

    def quit_players(self) -> None:
        """Clear all loaded players."""
        self._players = []

    def quit_session(self) -> None:
        """Close the database session."""
        if self._session is not None:
            try:
                self._session.close()
                logger.debug("Database session closed")
            except Exception as e:
                logger.error("Error closing database session", error=str(e))
            finally:
                self._session = None

    def quit(self) -> None:
        """Clean up engine resources and close database session."""
        logger.info("Stopping VotingEngine")
        self.quit_players()
        self.quit_session()
        super().quit()

    def bestmove(self) -> str:
        """Generate a move by voting among all player engines.

        Each engine votes for their best move, and the move with the most
        votes is selected. In case of a tie, a random move is chosen from
        the tied candidates.

        Returns:
            The winning move in UCI string format (e.g., 'e2e4', 'e7e8q')

        Raises:
            RuntimeError: If board is not initialized
        """
        assert self._board, (
            f"Engine {self.name} board not initialized - call position() first"
        )
        board = self._board
        votes = asyncio.run(
            get_votes(
                board=board,
                players=self._players,
                max_concurrent=self.max_concurrent,
            )
        )

        return self.aggregator(votes, self.weights, board)


if __name__ == "__main__":
    # Run in UCI mode for standalone execution
    # Usage: python voting_engine.py
    #
    # Example UCI session:
    #   uci
    #   setoption name Crowd_ids value 1,2,3
    #   setoption name Max_concurrent value 2
    #   isready
    #   ucinewgame
    #   position startpos moves e2e4
    #   go
    #   quit
    #
    # Or with Gaussian generation:
    #   uci
    #   setoption name Crowd_kind value Stockfish gaussian
    #   setoption name Crowd_size value 20
    #   setoption name Crowd_mean_elo value 1800
    #   setoption name Crowd_std_dev value 150
    #   setoption name Seed value 42
    #   isready
    #   ucinewgame
    #   position startpos moves e2e4
    #   go
    #   quit

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
    )

    engine = VotingEngine()
    engine.loop()
