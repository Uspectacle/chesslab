"""LLM tools and utilities for ChessLab.

Provides model loading, prompt formatting, and move parsing functionality.
"""

from typing import Optional, cast

import structlog
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from chesslab.env import get_device

logger = structlog.get_logger()


def load_model(
    model_name: str,
    quantization: Optional[str] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer | PreTrainedTokenizerFast, str]:
    """Load a language model and tokenizer.

    Args:
        model_name: HuggingFace model identifier
        quantization: Quantization mode ('4bit', '8bit', or None)

    Returns:
        Tuple of (model, tokenizer, device)
    """
    logger.info("Loading LLM", model=model_name, quantization=quantization)

    # Determine device
    device = get_device()
    if device == "cuda":
        logger.info("Using GPU", device=torch.cuda.get_device_name(0))
    else:
        logger.info("Using CPU")

    # Load tokenizer
    logger.debug("Loading tokenizer")
    tokenizer = cast(
        PreTrainedTokenizer | PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained(model_name),  # pyright: ignore[reportUnknownMemberType]
    )

    # Configure quantization
    quantization_config = None
    if quantization and device == "cuda":
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization")
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Using 8-bit quantization")

    # Load model
    logger.debug("Loading model")

    if quantization_config:
        dtype = None
        device_map = "auto"
    elif device == "cuda":
        dtype = torch.float16
        device_map = "cuda:0"
    else:
        dtype = torch.float32
        device_map = "cpu"

    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
            model_name,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
            dtype=dtype,
            device_map=device_map,
        ),
    )

    if device == "cpu" and not quantization_config:
        model = model.to(device)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]

    logger.info("Model loaded successfully", model=model_name)
    return model, tokenizer, device


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    device: str,
    messages: list[dict[str, str]],
    max_tokens: int = 100,
    temperature: float = 0.7,
) -> str:
    """Generate a response from the language model.

    Args:
        model: Loaded language model
        tokenizer: Loaded tokenizer
        device: Device to use ('cuda' or 'cpu')
        messages: List of message dictionaries with 'role' and 'content'
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated response text
    """
    logger.debug("Generating response", message_count=len(messages))

    # Prepare input
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:  # pyright: ignore[reportUnknownMemberType]
        input_text = cast(
            str,
            tokenizer.apply_chat_template(  # pyright: ignore[reportUnknownMemberType]
                messages, tokenize=False, add_generation_prompt=True
            ),
        )
    else:
        # Fallback: concatenate messages
        input_text = "\n\n".join([f"{m['role']}: {m['content']}" for m in messages])

    inputs = tokenizer(input_text, return_tensors="pt").to(device)  # pyright: ignore[reportUnknownMemberType]

    # Generate
    with torch.no_grad():
        outputs = cast(
            list[int],
            model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,  # pyright: ignore[reportUnknownMemberType, reportCallIssue]
            ),
        )

    # Decode
    response: str = tokenizer.decode(outputs[0], skip_special_tokens=True)  # pyright: ignore[reportAssignmentType, reportUnknownMemberType]

    # Remove input from response if present
    if input_text in response:
        response = response.replace(input_text, "").strip()

    return response
