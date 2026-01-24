"""LLM-based chess engine for ChessLab.

This engine uses language models to select chess moves based on configurable
prompts and conversation history.
"""

import logging
import random
from typing import Callable, List, Optional

import chess
import structlog
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from chesslab.engines.base_engine import BaseEngine
from chesslab.engines.llm_tools import generate_response, load_model
from chesslab.engines.options.options import (
    OptionCheck,
    OptionCombo,
    OptionSpin,
    OptionString,
)
from chesslab.engines.options.parsers import PARSERS
from chesslab.engines.options.prompt_variables import PromptVariables
from chesslab.engines.options.prompts import SYSTEM_PROMPTS, USER_PROMPTS

logger = structlog.get_logger()


class LlmEngine(BaseEngine):
    """Chess engine that uses language models to select moves.

    This engine can be configured with:
    - Different language models from HuggingFace
    - Custom or preset system and user prompts with template variables
    - Conversation history (single-shot or continuous dialogue)
    - Move parsing strategies
    - Quantization for performance
    - Chain-of-thought reasoning

    Template Variables Available:
    - {elo}: Expected Elo rating
    - {date}: Current date/time
    - {fen}: Current board position in FEN notation
    - {legal_moves}: Comma-separated list of legal moves
    - {pgn}: Game history in PGN format
    - {move_number}: Current move number
    - {side_to_move}: 'White' or 'Black'

    Attributes:
        name: Engine name identifier
        author: Engine author
        options: List of configurable options
    """

    name: str = "LlmEngine"
    author: str = "ChessLab"
    options = [
        OptionString(
            name="Model",
            default="meta-llama/Llama-3.2-1B-Instruct",
        ),
        OptionString(
            name="System_Prompt",
            default="standard",
        ),
        OptionString(
            name="User_Prompt",
            default="standard",
        ),
        OptionCombo(
            name="Parser",
            default="flexible",
            vars=list(PARSERS.keys()),
        ),
        OptionCombo(
            name="Quantization",
            default="none",
            vars=["none", "4bit", "8bit"],
        ),
        OptionSpin(name="Max_Tokens", default=100, min=10, max=500),
        OptionSpin(name="Temperature_x100", default=70, min=1, max=200),
        OptionSpin(name="Max_Retries", default=4, min=1, max=10),
        OptionCheck(name="Continuous_Conversation", default=False),
    ]

    def __init__(self) -> None:
        """Initialize the LLM engine."""
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizer | PreTrainedTokenizerFast] = None
        self._device: str = "cpu"
        self._conversation_history: List[dict[str, str]] = []
        self._current_model_name: Optional[str] = None
        self._current_quantization: Optional[str] = None
        super().__init__()

    @property
    def model_name(self) -> str:
        """Get the model name."""
        option = self.get_option("Model")

        if option is None:
            raise RuntimeError("Option Model not found")

        return option.value

    @property
    def system_prompt(self) -> str:
        """Get the system prompt (custom or preset)."""
        option = self.get_option("System_Prompt")

        if option is None:
            raise RuntimeError("Option System_Prompt not found")

        return SYSTEM_PROMPTS.get(option.value, str(option.value))

    @property
    def user_prompt(self) -> str:
        """Get the user prompt (custom or preset)."""
        option = self.get_option("User_Prompt")

        if option is None:
            raise RuntimeError("Option User_Prompt not found")

        return USER_PROMPTS.get(option.value, str(option.value))

    @property
    def parser(self) -> Callable[[str, chess.Board], str | None]:
        """Get the move parser function."""
        option = self.get_option("Parser")
        if option is None:
            raise RuntimeError("Option Parser not found")

        parser = PARSERS.get(option.value)
        if not parser:
            raise ValueError(f"Parser {option.value} not defined")

        return parser

    @property
    def quantization(self) -> Optional[str]:
        """Get the quantization mode."""
        option = self.get_option("Quantization")
        if option is None:
            raise RuntimeError("Option Quantization not found")

        quant = option.value
        return None if quant == "none" else quant

    @property
    def max_tokens(self) -> int:
        """Get the maximum tokens to generate."""
        option = self.get_option("Max_Tokens")
        if option is None:
            raise RuntimeError("Option Max_Tokens not found")
        return option.value

    @property
    def temperature(self) -> float:
        """Get the sampling temperature."""
        option = self.get_option("Temperature_x100")
        if option is None:
            raise RuntimeError("Option Temperature_x100 not found")
        return option.value / 100.0

    @property
    def max_retries(self) -> int:
        """Get the maximum number of retries."""
        option = self.get_option("Max_Retries")
        if option is None:
            raise RuntimeError("Option Max_Retries not found")
        return option.value

    @property
    def continuous_conversation(self) -> bool:
        """Get whether to use continuous conversation."""
        option = self.get_option("Continuous_Conversation")
        if option is None:
            raise RuntimeError("Option Continuous_Conversation not found")
        return option.value

    def _load_model(self) -> None:
        """Load the language model and tokenizer."""
        model_name = self.model_name
        quantization = self.quantization

        # Check if model is already loaded with correct settings
        if (
            self._model is not None
            and self._current_model_name == model_name
            and self._current_quantization == quantization
        ):
            logger.debug("Model already loaded", model=model_name)
            return

        # Clean up previous model
        if self._model is not None:
            logger.debug("Unloading previous model", model=self._current_model_name)
            del self._model
            del self._tokenizer
            if self._device == "cuda":
                torch.cuda.empty_cache()

        # Load new model
        self._model, self._tokenizer, self._device = load_model(
            model_name=model_name,
            quantization=quantization,
        )

        self._current_model_name = model_name
        self._current_quantization = quantization

    def reset(self) -> None:
        """Reset the board and conversation history."""
        self._conversation_history = []
        super().reset()

    def quit(self) -> None:
        """Clean up model resources."""
        if self._model is not None:
            logger.info("Unloading model")
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._current_model_name = None
            self._current_quantization = None
            if self._device == "cuda":
                torch.cuda.empty_cache()
        super().quit()

    def _generate_move_with_retries(self) -> str:
        """Generate a move with retry logic.

        Returns:
            UCI move string

        Raises:
            RuntimeError: If board or model not initialized
        """
        if self._board is None:
            raise RuntimeError("Board not initialized")

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not initialized")

        # Prepare prompt variables
        variables = PromptVariables.from_board(self._board)

        # Format prompts
        system_prompt = variables.format_template(self.system_prompt)
        user_prompt = variables.format_template(self.user_prompt)

        # Try to generate valid move
        for attempt in range(self.max_retries):
            logger.debug(
                "Attempting move generation",
                attempt=attempt + 1,
                max_retries=self.max_retries,
            )

            # Build conversation
            if self.continuous_conversation and self._conversation_history:
                messages = self._conversation_history.copy()
                messages.append({"role": "user", "content": user_prompt})
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

            # Generate response
            response = generate_response(
                model=self._model,
                tokenizer=self._tokenizer,
                device=self._device,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            logger.debug("LLM response", response=response[:200], attempt=attempt + 1)

            # Try to parse move
            move = self.parser(response, self._board)

            if move:
                # Update conversation history if continuous
                if self.continuous_conversation:
                    self._conversation_history = messages
                    self._conversation_history.append(
                        {"role": "assistant", "content": response}
                    )

                logger.debug("Valid move found", move=move, attempt=attempt + 1)
                return move

            logger.warning(
                "No valid move found in response",
                attempt=attempt + 1,
                response=response[:200],
            )

        # All retries exhausted - return random legal move
        logger.error(
            "Failed to generate valid move after retries, using random fallback",
            max_retries=self.max_retries,
        )
        legal_moves = list(self._board.legal_moves)
        fallback_move = random.choice(legal_moves).uci()
        logger.info("Random fallback move selected", move=fallback_move)
        return fallback_move

    def bestmove(self) -> str:
        """Generate a move using the language model.

        Returns:
            Move in UCI string format (e.g., 'e2e4', 'e7e8q')

        Raises:
            RuntimeError: If board is not initialized or model fails
        """
        if self._board is None:
            raise RuntimeError("Board not initialized")

        # Load model if needed
        self._load_model()

        # Generate move with retries
        return self._generate_move_with_retries()


if __name__ == "__main__":
    # Run in UCI mode for standalone execution
    # Usage: python llm_engine.py
    #
    # Example UCI session:
    #   uci
    #   setoption name Model value meta-llama/Llama-3.2-1B-Instruct
    #   setoption name System_Prompt_Preset value detailed
    #   setoption name User_Prompt_Preset value detailed
    #   setoption name Quantization value 4bit
    #   isready
    #   ucinewgame
    #   position startpos moves e2e4
    #   go
    #   quit

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )

    engine = LlmEngine()
    engine.loop()
