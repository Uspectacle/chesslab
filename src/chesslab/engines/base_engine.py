"""Base engine interface for ChessLab.

All engines must implement this protocol to be compatible with the arena runner.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import chess
import structlog

from chesslab.engines.options import Option

logger = structlog.get_logger()


class BaseEngine(ABC):
    """Abstract base class defining the interface for chess engines.

    All engine implementations must provide these methods and attributes
    to work with the arena runner and tournament coordinator.
    """

    name: str = "BaseEngine"
    author: str = "ChessLab"
    options: List[Option] = []

    def __init__(self):
        """Initialize the engine."""
        self._started = False
        self._board: Optional[chess.Board] = None
        logger.info("Engine initialized", name=self.name)

    def get_option(self, name: str) -> Optional[Option]:
        """Get an option by name (case-insensitive).

        Args:
            name: The option name to search for

        Returns:
            The matching Option or None if not found
        """
        for option in self.options:
            if option.name.lower() == name.lower():
                return option
        return None

    def reset(self) -> None:
        """Reset the board to the starting position."""
        self._board = chess.Board()
        logger.debug("Board reset", name=self.name)

    def start(self) -> None:
        """Start the engine and prepare it for move generation.

        This may involve starting subprocesses, loading models, or
        establishing API connections.
        """
        if not self._started:
            self.reset()
            self._started = True
            logger.debug("Engine started", name=self.name)

    def quit(self) -> None:
        """Clean up engine resources and terminate any subprocesses.

        Should be called when the engine is no longer needed.
        """
        if self._started:
            self._started = False
            logger.debug("Engine closed", name=self.name)

    @abstractmethod
    def bestmove(self) -> str:
        """Generate a move for the current board position.

        Returns:
            Move in UCI string format (e.g., 'e2e4', 'e7e8q')

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement bestmove()")

    def uci(self) -> None:
        """UCI: Identify the engine.

        Outputs engine name, author information, and available options.
        """
        print(f"id name {self.name}")
        print(f"id author {self.author}")

        for option in self.options:
            print(option)

        print("uciok")

    def isready(self) -> None:
        """UCI: Check if engine is ready.

        Starts the engine if not already started and confirms readiness.
        """
        self.start()
        print("readyok")

    def ucinewgame(self) -> None:
        """UCI: Prepare for a new game.

        Resets the board to the starting position.
        """
        self.reset()
        logger.debug("New game initialized", name=self.name)

    def position(self, args: List[str]) -> None:
        """UCI: Set up the board position.

        Args:
            args: Position command arguments
                  Format: ['startpos'] or ['fen', '<fen_string>']
                  Optionally followed by ['moves', '<move1>', '<move2>', ...]

        Examples:
            position(['startpos'])
            position(['startpos', 'moves', 'e2e4', 'e7e5'])
            position(['fen', 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR', 'w', 'KQkq', '-', '0', '1'])
        """
        if not args:
            logger.error("No position specified", args=args)
            return

        # Separate initial position from moves
        moves_idx = args.index("moves") if "moves" in args else len(args)
        initial_position = args[:moves_idx]
        moves = args[moves_idx + 1 :] if moves_idx < len(args) else []

        # Set up initial position
        if initial_position[0] == "startpos":
            self._board = chess.Board()
            logger.debug("Position set to startpos")

        elif initial_position[0] == "fen":
            if len(initial_position) < 2:
                logger.error("FEN string missing", args=args)
                return

            fen_string = " ".join(initial_position[1:])
            try:
                self._board = chess.Board(fen_string)
                logger.debug("Position set from FEN", fen=fen_string)
            except ValueError as e:
                logger.error("Invalid FEN string", fen=fen_string, error=str(e))
                return

        else:
            logger.error("Invalid initial position", position=initial_position[0])
            return

        # Apply moves
        for move_str in moves:
            try:
                self._board.push_uci(move_str)
            except ValueError as e:
                logger.error(
                    "Invalid move",
                    move=move_str,
                    fen=self._board.fen(),
                    error=str(e),
                )
                return

        logger.debug(
            "Position updated",
            fen=self._board.fen(),
            move_count=len(moves),
        )

    def go(self, args: Optional[List[str]] = None) -> None:
        """UCI: Start calculating the best move.

        Args:
            args: Optional go command arguments (e.g., time controls)
                  Currently ignored by base implementation
        """
        try:
            if not self._started:
                raise RuntimeError(
                    f"Engine {self.name} not started - call start() first"
                )

            if self._board is None:
                raise RuntimeError(
                    f"Engine {self.name} board not initialized - call position() first"
                )

            legal_moves = list(self._board.legal_moves)

            if not legal_moves:
                logger.error("No legal moves available", fen=self._board.fen())
                raise ValueError("No legal moves available - game is over")

            move = self.bestmove()

            logger.debug(
                "Move selected",
                move=move,
                legal_count=len(legal_moves),
                fen=self._board.fen(),
            )
            print(f"bestmove {move}")
        except (ValueError, RuntimeError) as e:
            logger.error("Failed to generate move", error=str(e))
            print("bestmove (none)")

    def setoption(self, args: List[str]) -> None:
        """UCI: Set engine options.

        Args:
            args: Option command arguments
                  Format: ['name', '<option_name>', 'value', '<option_value>']

        Note: Override this method in subclasses to handle specific options.
        Base implementation logs but handles standard options.
        """
        if not args:
            logger.error("No option specified", args=args)
            return

        if len(args) < 4 or args[0] != "name" or args[2] != "value":
            logger.error("Invalid setoption format", args=args)
            return

        name = args[1]
        value = " ".join(args[3:])

        option = self.get_option(name)
        if not option:
            logger.warning("Option not found", option=name)
            return

        option.set_value(value)

    def loop(self) -> None:
        """Run UCI protocol loop for standalone execution.

        This allows the engine to be used like any standard UCI engine.
        Compile with: pyinstaller --onefile your_engine.py

        Supports the following UCI commands:
        - uci: Identify engine
        - isready: Check readiness
        - ucinewgame: Start new game
        - position: Set board position
        - go: Calculate best move
        - setoption: Configure engine
        - quit: Exit engine
        """
        print(f"{self.name} v1.0 by {self.author}")

        while True:
            try:
                line = input().strip()
            except EOFError:
                break

            if not line:
                continue

            tokens = line.split()
            command = tokens[0]
            args = tokens[1:] if len(tokens) > 1 else []

            try:
                if command == "uci":
                    self.uci()

                elif command == "isready":
                    self.isready()

                elif command == "ucinewgame":
                    self.ucinewgame()

                elif command == "position":
                    self.position(args)

                elif command == "go":
                    self.go(args)

                elif command == "setoption":
                    self.setoption(args)

                elif command == "quit":
                    self.quit()
                    break

                else:
                    logger.warning("Unknown UCI command", command=command)

            except Exception as e:
                logger.error(
                    "Error processing command",
                    command=command,
                    error=str(e),
                    exc_info=True,
                )


if __name__ == "__main__":
    # This base class cannot be run directly - use a concrete implementation
    print(
        "BaseEngine is an abstract class. Please use a concrete engine implementation."
    )
