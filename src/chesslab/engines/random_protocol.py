import asyncio
import random
from typing import Iterable, MutableMapping, Optional

import chess
from chess.engine import (
    AnalysisResult,
    ConfigMapping,
    Cp,
    Info,
    InfoDict,
    Limit,
    Opponent,
    Option,
    PlayResult,
    PovScore,
    Protocol,
)


class RandomProtocol(Protocol):
    options: MutableMapping[str, Option] = {
        "Seed": Option(name="Seed", type="spin", default=0, min=0, max=999999, var=None)
    }
    id: dict[str, str] = {"name": "RandomProtocol", "author": "Uspectacle"}

    _seed: Optional[int] = None
    _random: Optional[random.Random] = None

    async def initialize(self) -> None:
        """Initializes the engine."""
        if self._seed is not None:
            self._random = random.Random(self._seed)
        else:
            self._random = random.Random()
        self.initialized = True

    async def ping(self) -> None:
        return

    async def configure(self, options: ConfigMapping) -> None:
        seed = options.get("Seed")
        if seed is not None:
            self._seed = int(seed)
            self._random = random.Random(self._seed)

    async def send_opponent_information(
        self,
        *,
        opponent: Optional[Opponent] = None,
        engine_rating: Optional[int] = None,
    ) -> None:
        return

    async def play(
        self,
        board: chess.Board,
        limit: Limit,
        *,
        game: object = None,
        info: Info = Info.NONE,
        ponder: bool = False,
        draw_offered: bool = False,
        root_moves: Optional[Iterable[chess.Move]] = None,
        options: ConfigMapping = {},
        opponent: Optional[Opponent] = None,
    ) -> PlayResult:
        legal_moves = list(root_moves or board.legal_moves)
        if not legal_moves:
            return PlayResult(move=None, ponder=None)

        seed = options.get("Seed")
        if seed is not None:
            _random = random.Random(int(seed))
        else:
            _random = self._random

        assert _random, "Engine not initialized. Call await engine.initialize()"
        move = _random.choice(legal_moves)

        return PlayResult(move=move, ponder=None)

    async def analysis(
        self,
        board: chess.Board,
        limit: Optional[Limit] = None,
        *,
        multipv: Optional[int] = None,
        game: object = None,
        info: Info = Info.ALL,
        root_moves: Optional[Iterable[chess.Move]] = None,
        options: ConfigMapping = {},
    ) -> AnalysisResult:
        legal_moves = list(root_moves or board.legal_moves)
        if not legal_moves:
            return AnalysisResult()

        seed = options.get("Seed")
        if seed is not None:
            _random = random.Random(int(seed))
        else:
            _random = self._random
        assert _random, "Engine not initialized. Call await engine.initialize()"

        results = AnalysisResult()

        for move in legal_moves:
            results.post(
                InfoDict(
                    pv=[move],
                    # Random score between -1.0 and +1.0 pawn units
                    score=PovScore(
                        Cp(int(_random.uniform(-1.0, 1.0) * 100)), board.turn
                    ),
                )
            )

        return results

    async def send_game_result(
        self,
        board: chess.Board,
        winner: Optional[chess.Color] = None,
        game_ending: Optional[str] = None,
        game_complete: bool = True,
    ) -> None:
        pass

    async def quit(self) -> None:
        self._random = None
        self.initialized = False


async def main():
    RandomProtocol()


if __name__ == "__main__":
    asyncio.run(main())
