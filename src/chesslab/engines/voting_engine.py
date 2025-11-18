import asyncio
import logging
from typing import Dict, List

import chess
import structlog

from chesslab.engines.aggregators import AGGREGATORS, Aggregator
from chesslab.engines.base_engine import BaseEngine
from chesslab.engines.options import OptionCombo, OptionSpin, OptionString
from chesslab.engines.storage_tools import get_uci_move
from chesslab.storage import Player, get_session
from chesslab.storage.player_tools import get_player_by_id

logger = structlog.get_logger()


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
        options: List of configurable options (Player_ids, Max_concurrent)
    """

    name = "VotingEngine"
    author = "ChessLab"
    options = [
        OptionString(name="Player_ids"),
        OptionCombo(
            name="Aggregator", default="majority", vars=list(AGGREGATORS.keys())
        ),
        OptionSpin(name="Max_concurrent", default=1, min=1, max=8),
    ]

    def __init__(self) -> None:
        """Initialize the voting engine."""
        self._players: Dict[int, Player] = {}
        self._session = None
        super().__init__()

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
    def max_concurrent(self) -> int:
        """Get the maximum number of concurrent engine evaluations."""
        option = self.get_option("Max_concurrent")
        if option is None:
            raise RuntimeError("Option Max_concurrent not found")

        return option.value

    @property
    def player_ids(self) -> List[int]:
        """Get the list of player IDs from the Player_ids option.

        Returns:
            List of player IDs to use for voting

        Raises:
            RuntimeError: If Player_ids option is not found or empty
        """
        option = self.get_option("Player_ids")

        if option is None:
            raise RuntimeError("Option Player_ids not found")

        player_ids_str = str(option.value).strip()

        if not player_ids_str:
            raise RuntimeError("Option Player_ids is empty")

        return [int(player_id.strip()) for player_id in player_ids_str.split(",")]

    def reset(self) -> None:
        """Reset the board, session, and players."""
        self.reset_session()
        self.reset_players()
        super().reset()

    def reset_players(self) -> None:
        """Load all players from the database.

        Raises:
            ValueError: If any player ID is not found in the database
        """
        assert self._session, "Session must be initialized before loading players"

        self.quit_players()

        for player_id in self.player_ids:
            player = get_player_by_id(self._session, player_id)

            if player is None:
                raise ValueError(f"Player with ID {player_id} not found in database")

            self._players[player_id] = player

            logger.debug(
                "Player loaded",
                player_id=player_id,
                engine_type=player.engine_type,
                expected_elo=player.expected_elo,
            )

    def reset_session(self) -> None:
        """Reset the database session."""
        self.quit_session()
        self._session = get_session().__enter__()
        logger.debug("Database session opened")

    def quit_players(self) -> None:
        """Clear all loaded players."""
        self._players = {}

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
        players = list(self._players.values())
        votes = asyncio.run(
            get_votes(
                board=board,
                players=players,
                max_concurrent=self.max_concurrent,
            )
        )

        return self.aggregator(votes, players, board)


if __name__ == "__main__":
    # Run in UCI mode for standalone execution
    # Usage: python voting_engine.py
    #
    # Example UCI session:
    #   uci
    #   setoption name Player_ids value 1,2,3
    #   setoption name Max_concurrent value 2
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
