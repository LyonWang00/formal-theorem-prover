"""Tokenizer construction shared by SFT and GRPO."""

from transformers import AutoTokenizer


def load_tokenizer(model_name_or_path: str, *, padding_side: str):
    """Load a tokenizer and configure its padding behavior."""

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer

