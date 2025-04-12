import os

def check_unk_after_bpe(dict_file: str | os.PathLike, bpe_file: str | os.PathLike) -> set:
    """
    Checks which tokens in the `bpe_file` (already BPE-processed) are not in the `dict_file`.

    Args:
        dict_file (str | os.PathLike): Path to the fairseq dictionary file.
        bpe_file (str | os.PathLike): Path to the file containing BPE-processed tokens.

    Returns:
        set: A set of unknown tokens from `bpe_file`.
    """

    # Load dictionary into a set for efficient lookup
    vocab = set()
    with open(dict_file, 'r', encoding='utf-8') as f:
        for line in f:
            word, _ = line.strip().split()
            vocab.add(word)

    unk_tokens = set()
    with open(bpe_file, 'r', encoding='utf-8') as f:
        for line in f:
            for token in line.strip().split():  # Split the line into tokens
                if token not in vocab:
                    unk_tokens.add(token)

    return unk_tokens

if __name__ == '__main__':
    def main(dict_file, bpe_file):
        unknown_tokens = check_unk_after_bpe(dict_file, bpe_file)
        print("Unknown tokens:", unknown_tokens)
    import sys
    if '-h' in sys.argv or '--help' in sys.argv:
        print("Checks which tokens in the BPE-processed file are not in the dictionary file.")
        print("Usage: python who_is_unk.py <dict_file> <bpe_file>")
        sys.exit(0)
    if len(sys.argv) != 3:
        print("Usage: python who_is_unk.py <dict_file> <bpe_file>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])