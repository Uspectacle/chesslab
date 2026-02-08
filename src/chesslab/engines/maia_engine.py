"""Maia2 chess engine for ChessLab.

This engine uses the Maia2 neural network models to mimic human play at specific
Elo ratings.
"""

import logging
from typing import Literal, Optional

import chess
import structlog
from maia2 import inference, model
from maia2.main import MAIA2Model

from chesslab.engines.base_engine import BaseEngine
from chesslab.engines.options.options import OptionCombo, OptionSpin
from chesslab.env import get_maia_url

logger = structlog.get_logger(
    level=logging.DEBUG,
    format="%(levelname)s %(name)s: %(message)s",
)


class Maia2Engine(BaseEngine):
    """Chess engine that uses Maia2 to play like a human.

    Maia2 is a unified model that can simulate human play across a wide range
    of skill levels.

    Attributes:
        name: Engine name identifier
        author: Engine author (CSSLab / Wrapper)
        options: List of configurable options (Elo, ModelType, Device)
    """

    name: str = "Maia2Engine"
    author: str = "ChessLab (Maia2 by CSSLab)"

    # Define UCI options
    options = [
        OptionSpin(name="UCI_Elo", default=1500, min=1100, max=1900),
        OptionSpin(name="OpponentElo", default=1500, min=1100, max=1900),
        OptionCombo(name="ModelType", default="rapid", vars=["rapid", "blitz"]),
        OptionCombo(name="Device", default="cpu", vars=["cpu", "gpu"]),
    ]

    def __init__(self):
        """Initialize the Maia2 engine wrapper."""
        self._model: Optional[MAIA2Model] = None
        self._prepared_inference = None
        self._current_model_type = None
        self._current_device = None
        super().__init__()

    @property
    def elo(self) -> int:
        """Get the current target Elo for the engine (Self)."""
        opt = self.get_option("UCI_Elo")
        return int(opt.value) if opt else 1500

    @property
    def opponent_elo(self) -> int:
        """Get the opponent's estimated Elo."""
        opt = self.get_option("OpponentElo")
        return int(opt.value) if opt else 1500

    @property
    def model_type(self) -> Literal["blitz", "rapid"]:
        """Get the model type (rapid or blitz)."""
        opt = self.get_option("ModelType")
        val = opt.value.lower() if opt else "rapid"
        return val if val in ["rapid", "blitz"] else "rapid"

    @property
    def device(self) -> Literal["cpu", "gpu"]:
        """Get the inference device."""
        opt = self.get_option("Device")
        return opt.value.lower() if opt else "cpu"

    def start(self) -> None:
        """Start the engine and load the Maia2 model.

        This method checks if the configuration has changed (e.g. switching
        from cpu to gpu or rapid to blitz) and reloads the model if necessary.
        """

        # Check if we need to (re)load the model
        if (
            self._model is None
            or self._current_model_type != self.model_type
            or self._current_device != self.device
        ):
            logger.info(
                "Loading Maia2 model...",
                type=self.model_type,
                device=self.device,
            )
            try:
                self._model = model.from_pretrained(
                    self.model_type, device=self.device, save_root=get_maia_url()
                )
                self._prepared_inference = inference.prepare()

                self._current_model_type = self.model_type
                self._current_device = self.device
                self._started = True

                logger.info("Maia2 model loaded successfully")
            except Exception as e:
                logger.error("Failed to load Maia2 model", error=str(e))
                raise RuntimeError(f"Failed to load Maia2 model: {e}")
        else:
            self._started = True
            logger.debug("Maia2 model already loaded")

        super().reset()

    def bestmove(self) -> str:
        """Generate a move using the Maia2 model.

        Returns:
            Best move in UCI string format.

        Raises:
            RuntimeError: If model or board is not initialized.
        """
        if self._model is None or self._prepared_inference is None:
            self.start()

        if self._model is None:
            raise RuntimeError("Model not initialized")

        if self._prepared_inference is None:
            raise RuntimeError("Model is not prepared")

        if self._board is None:
            raise RuntimeError("Board not initialized")

        # Get FEN from current board
        fen = self._board.fen()

        # Get Elo ratings from options
        elo_self = self.elo
        elo_oppo = self.opponent_elo

        logger.debug("Running inference", fen=fen, elo_self=elo_self, elo_oppo=elo_oppo)

        # Run inference
        move_probs, win_prob = inference.inference_each(
            model=self._model,
            prepared=self._prepared_inference,
            fen=fen,
            elo_self=elo_self,
            elo_oppo=elo_oppo,
        )

        best_move_str = max(move_probs, key=lambda move: move_probs[move])
        move = chess.Move.from_uci(best_move_str)

        logger.debug(
            "Maia2 selected move",
            move=move.uci(),
            win_prob=f"{win_prob:.2f}",
            confidence=f"{move_probs[best_move_str]:.2f}",
        )

        return move.uci()


if __name__ == "__main__":
    # Configure logging for standalone execution
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )

    engine = Maia2Engine()
    engine.loop()
