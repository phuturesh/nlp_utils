"""To avoid reimplemening simple NLP wheels"""

__version__ = "0.0.1dev5"

from typing import Optional

from .deepseek_calc_tokens import (
    num_tokens_from_messages as num_tokens_from_messages_deepseek,
    num_tokens_from_text as num_tokens_from_text_deepseek,
)
from .openai_calc_tokens import (
    num_tokens_from_messages as num_tokens_from_messages_openai,
    num_tokens_from_text as num_tokens_from_text_openai,
)
from .transformers_calc_tokens import (
    num_tokens_from_messages as num_tokens_from_messages_transformers,
    num_tokens_from_text as num_tokens_from_text_transformers,
)
from .who_is_unk import check_unk_after_bpe

__all__ = [
    "num_tokens_from_messages",
    "check_unk_after_bpe",
]


def num_tokens_from_messages(
    messages: list[dict],
    model: str,
    transformers_token: str | bool = True,
    tokenizer: Optional[str] = None,
    deepseek_version: Optional[str] = "v3",
) -> int:
    if tokenizer is not None:
        if tokenizer.lower() in ["openai", "tiktoken", "gpt"]:
            return num_tokens_from_messages_openai(messages, model)
        elif tokenizer.lower() in ["deepseek", "deepseek-chat", "deepseek-reasoner"]:
            return num_tokens_from_messages_deepseek(messages, version=deepseek_version)
        elif tokenizer.lower() == "transformers":
            return num_tokens_from_messages_transformers(
                messages, model, transformers_token
            )
        else:
            raise ValueError(f"Unknown tokenizer: {tokenizer}")

    if model.startswith("gpt-"):
        return num_tokens_from_messages_openai(messages, model)
    elif model.startswith("deepseek"):
        return num_tokens_from_messages_deepseek(messages, version=deepseek_version)
    else:
        return num_tokens_from_messages_transformers(
            messages, model, transformers_token
        )


def num_tokens_from_text(
    text: str,
    model: str,
    transformers_token: str | bool = True,
    tokenizer: Optional[str] = None,
    version: Optional[str] = "v3",
) -> int:
    if tokenizer is not None:
        if tokenizer.lower() in ["openai", "tiktoken", "gpt"]:
            return num_tokens_from_text_openai(text, model)
        elif tokenizer.lower() in ["deepseek", "deepseek-chat", "deepseek-reasoner"]:
            return num_tokens_from_text_deepseek(text, model, version=version)
        elif tokenizer.lower() == "transformers":
            return num_tokens_from_text_transformers(text, model, transformers_token)
        else:
            raise ValueError(f"Unknown tokenizer: {tokenizer}")

    if model.startswith("gpt-"):
        return num_tokens_from_text_openai(text, model)
    elif model.startswith("deepseek"):
        return num_tokens_from_text_deepseek(text, model, version=version)
    else:
        return num_tokens_from_text_transformers(text, model, transformers_token)
