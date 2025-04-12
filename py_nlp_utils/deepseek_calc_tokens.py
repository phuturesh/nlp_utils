import os
import importlib.resources
import importlib.metadata

import transformers



def num_tokens_from_messages(messages, model="deepseek-chat", version="v3"):
    # model actually doesn't matter
    assert model in ["deepseek-chat", "deepseek-reasoner"]
    if version == 'v3':
        chat_tokenizer_dir = importlib.resources.files("py_nlp_utils").joinpath("deepseek_tokenizer/v3")
    elif version == 'v2':
        chat_tokenizer_dir = importlib.resources.files("py_nlp_utils").joinpath("deepseek_tokenizer/v2")
    else:
        raise ValueError(f"Invalid version {version}, only v2 and v3 are supported")
    assert isinstance(chat_tokenizer_dir, os.PathLike)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        chat_tokenizer_dir, trust_remote_code=True
    )
    tokens = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    return len(tokens)


if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am doing well, thank you for asking."},
        {"role": "user", "content": "What is the weather like today in Hangzhou?"},
        {"role": "assistant", "content": "It is sunny with a high of 75F."},
    ]
    print(num_tokens_from_messages(messages))
