"""
Script to test multiple language models with a specific prompt.
Supports local model paths via environment variables or automatic download from HuggingFace.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


def get_model_path(model_name: str) -> str:
    """
    Get the model path from environment variable or download it.

    Args:
        model_name: The HuggingFace model identifier (e.g., 'google/gemma-3-1b-it')

    Returns:
        Path to the model directory
    """
    # Create a safe environment variable name from model name
    env_var_name = model_name.replace("/", "_").replace("-", "_").upper() + "_PATH"

    # Check if path is in environment variable
    model_path = os.getenv(env_var_name, "")

    if model_path and Path(model_path).exists():
        print(f"✓ Using model from: {model_path}")
        return model_path

    return model_name


def run_inference(
    model_name: str, prompt: str, use_gpu: bool = True, max_new_tokens: int = 256
) -> str:
    """
    Run inference on a language model.

    Args:
        model_name: The HuggingFace model identifier
        prompt: The prompt to send to the model
        use_gpu: Whether to use GPU (if available)
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        The model's response
    """
    start_time = time.time()

    print(f"\n{'=' * 80}")
    print(f"Testing model: {model_name}")
    print(f"{'=' * 80}")

    model_path = get_model_path(model_name)

    device = "cpu"
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("✓ Using CPU")

    print("⏳ Loading tokenizer...")
    tokenizer = cast(
        PreTrainedTokenizer | PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained(model_path),  # pyright: ignore[reportUnknownMemberType]
    )

    print("⏳ Loading model...")
    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
            model_path,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="cuda:0" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        ),
    )

    if device == "cpu":
        model = model.to(device)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]

    # Prepare input
    messages: list[Dict[str, str]] = [{"role": "user", "content": prompt + "\n"}]

    # Apply chat template if available
    input_text: str
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:  # pyright: ignore[reportUnknownMemberType]
        input_text = cast(
            str,
            tokenizer.apply_chat_template(  # pyright: ignore[reportUnknownMemberType]
                messages, tokenize=False, add_generation_prompt=True
            ),
        )
    else:
        input_text = prompt

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    print("💭 Generating response...")
    with torch.no_grad():
        outputs = cast(
            List[int],
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,  # pyright: ignore[reportUnknownMemberType, reportCallIssue]
            ),
        )

    # Decode
    response: str = tokenizer.decode(outputs[0], skip_special_tokens=True)  # pyright: ignore[reportUnknownMemberType]

    # Try to extract just the response (remove the prompt)
    if input_text in response:
        response = response.replace(input_text, "").strip()

    print("\n📝 Response:")
    print(f"{'-' * 80}")
    print(response)
    print(f"{'-' * 80}\n")

    elapsed = time.time() - start_time
    print(f"⏱️ Time taken: {elapsed:.2f} seconds\n")

    del model
    del tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    return response


def main() -> None:
    """Main function to test all models."""
    # Define models to test
    models: list[str] = [
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "google/gemma-3-1b-it",
    ]

    # Define the prompt
    prompt: str = "What is your chess ELO?"

    # GPU usage setting
    use_gpu: bool = True  # Set to False to force CPU usage

    print("🚀 Starting model testing...")
    print(f"📋 Prompt: '{prompt}'")
    print(f"🎮 GPU Usage: {'Enabled' if use_gpu else 'Disabled'}")

    # Test each model
    results: Dict[str, str] = {}
    for model_name in models:
        try:
            response = run_inference(
                model_name=model_name,
                prompt=prompt,
                use_gpu=use_gpu,
                max_new_tokens=200,
            )
            results[model_name] = response
        except Exception as e:
            print(f"❌ Error with model {model_name}: {e}")
            results[model_name] = f"Error: {e}"

    # Print summary
    print(f"\n{'=' * 80}")
    print("📊 SUMMARY")
    print(f"{'=' * 80}")
    for model_name, response in results.items():
        print(f"\n🤖 {model_name}:")
        print(f"   {response[:200]}...")


if __name__ == "__main__":
    main()
