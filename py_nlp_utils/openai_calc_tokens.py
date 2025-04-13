"""
This script references https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
"""

import json
from typing import List

import tiktoken
from tqdm import tqdm


def num_tokens_from_text(text: str, model="gpt-4o-mini-2024-07-18"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    }:
        pass
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125."
        )
        return num_tokens_from_text(text, model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        print(
            "Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18."
        )
        return num_tokens_from_text(text, model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        print(
            "Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06."
        )
        return num_tokens_from_text(text, model="gpt-4o-2024-08-06")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return num_tokens_from_text(text, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_text() is not implemented for model {model}."""
        )
    tokens = len(encoding.encode(text))
    return tokens


def num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    }:
        tokens_per_message = 3  # role, rolename, content
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        print(
            "Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18."
        )
        return num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        print(
            "Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06."
        )
        return num_tokens_from_messages(messages, model="gpt-4o-2024-08-06")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def num_tokens_from_batch_input(batch_input):
    """
    Return the number of tokens used by a batch input.
    Require batch_input to be a path to a JSONL file.
    We assume batch_input already contains legal JSONL lines.
    """
    with open(batch_input, "r") as f:
        tokens = 0
        lines = list(f.readlines())
        for line in tqdm(lines):
            jsonl = json.loads(line)
            body = jsonl["body"]
            model = body["model"]
            messages = body["messages"]
            tokens += num_tokens_from_messages(messages, model)
    return tokens


def num_tokens_from_raw_text(text_input: str, model: str) -> int:
    """Process raw text file as a single message."""
    with open(text_input, "r") as f:
        text = str(f.readlines())
    messages = [{"role": "user", "content": text}]
    return num_tokens_from_messages(messages, model)


def num_tokens_from_message_file(messages_str: str, model: str) -> int:
    """Process a JSON file containing messages."""
    try:
        with open(messages_str, "r") as f:
            text = f.readlines()
        messages = json.loads(text)
        return num_tokens_from_messages(messages, model)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format for messages")


if __name__ == "__main__":

    class NumTokens:
        def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
            self.model = model

        def batch(self, *batch_inputs: List[str]):
            for bi in batch_inputs:
                print(bi, f"{num_tokens_from_batch_input(bi):,}")

        def raw(self, *text_inputs: List[str]):
            for ti in text_inputs:
                print(ti, f"{num_tokens_from_raw_text(ti, self.model):,}")

        def message(self, *messages_strs: List[str]):
            for ms in messages_strs:
                print(ms, f"{num_tokens_from_message_file(ms, self.model):,}")

        def b(self, *batch_inputs: List[str]):
            self.batch(*batch_inputs)

        def r(self, *text_inputs: List[str]):
            self.raw(*text_inputs)

        def m(self, *messages_strs: List[str]):
            self.message(*messages_strs)

    from fire import Fire

    Fire(NumTokens)
