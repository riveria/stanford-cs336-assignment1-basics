from __future__ import annotations

from collections import defaultdict
import os
import regex as re

pre_tokenizer_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[int, int]] = []
    for i in range(len(special_tokens)):
        vocab[len(vocab)] = special_tokens[i].encode('utf-8')

    # chunk by special_tokens
    special_tokens_escaped = [re.escape(t) for t in special_tokens]
    split = '|'.join(special_tokens_escaped)
    with open(input_path, encoding='utf-8') as f:
        lines = re.split(split, f.read())

    # regex-based pre-tokenizer used by GPT-2
    pre_tokens_and_counts: dict[bytes, tuple[int, list[int]]] = {}
    for line in lines:
        for t in re.finditer(pre_tokenizer_pattern, line):
            token = t.group().encode('utf-8')
            if token not in pre_tokens_and_counts:
                pre_tokens_and_counts[token] = (1, list(token))
            else:
                c, l = pre_tokens_and_counts[token]
                pre_tokens_and_counts[token] = (c + 1, l)
    # print(pre_tokens_and_counts)

    while len(vocab) < vocab_size:
        # find the most popular bi-gram
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for pre_token, (c, token_list) in pre_tokens_and_counts.items():
            for i in range(1, len(token_list)):
                counts[(token_list[i - 1], token_list[i])] += c
        new_v = max(counts, key=lambda k: (counts[k], vocab[k[0]], vocab[k[1]]))
        # new_v = max(counts, key=counts.get)
        new_v_inx = len(vocab)
        vocab[new_v_inx] = vocab[new_v[0]] + vocab[new_v[1]]
        # print(new_v_inx, new_v, vocab[new_v[0]].encode("utf-8"), vocab[new_v[1]].encode("utf-8"), counts[new_v])
        merges.append(new_v)
        # merge with the new vocab
        for pre_token, (c, token_list) in pre_tokens_and_counts.items():
            i = 0
            new_token_list: list[int] = []
            while i < len(token_list):
                if token_list[i] == new_v[0] and i + 1 < len(token_list) and token_list[i + 1] == new_v[1]:
                    new_token_list.append(new_v_inx)
                    i += 2
                else:
                    new_token_list.append(token_list[i])
                    i += 1
            pre_tokens_and_counts[pre_token] = (c, new_token_list)
        # if len(vocab) >= vocab_size:
        # print(counts)

    return vocab, [(vocab[a], vocab[b]) for a, b in merges]
