from __future__ import annotations

import json
from typing import Iterable, Iterator
import regex as re

import cs336_basics.bpe_trainer


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens. This
        function should accept the following parameters:
            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
            special_tokens: list[str] | None = None
        """
        self.vocab = vocab
        self.vocab_index = {v:k for k, v in self.vocab.items()}
        for t in special_tokens or []:
            if t.encode('utf-8') not in self.vocab_index:
                self.vocab_index[t] = len(self.vocab)
                self.vocab[self.vocab_index[t]] = t
        self.merges = merges
        self.merges_rank = {self.merges[i]:i for i in range(len(self.merges))}
        self.special_tokens = special_tokens

    @staticmethod
    def from_files(vocab_filepath, merges_filepath, special_tokens=None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges (in the same
        format that your BPE training code output) and (optionally) a list of special tokens. This method should accept the
        following additional parameters:
            vocab_filepath: str
            merges_filepath: str
            special_tokens: list[str] | None = None
        """
        with open(vocab_filepath) as f:
            vocab = json.load(f)
        vocab = { v:k for k, v in vocab.items() }
        merges = []
        with open(merges_filepath) as f:
            for line in f:
                tokens = line.rstrip().split(" ")
                if len(tokens) == 2:
                    merges.append(tuple(tokens))

        yield from Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        texts = [text]
        tokens_endoftext = []
        #print(len(self.vocab) - 1, self.vocab[len(self.vocab) - 1])
        if self.special_tokens:
            special_tokens_escaped = sorted([re.escape(t) for t in self.special_tokens], key=len, reverse=True)
            split = '|'.join(special_tokens_escaped)
            texts = re.split(split, text)
            for t in re.finditer(split, text):
                tokens_endoftext.append(self.vocab_index[t.group().encode('utf-8')])
        #print(texts, tokens_endoftext)
        #print(self.vocab_index)

        tokens = []
        #print(texts)
        for text in texts:
            if text:
                for t in re.finditer(cs336_basics.bpe_trainer.pre_tokenizer_pattern, text):
                    words = [bytes([c]) for c in t.group().encode('utf-8')]

                    while True:
                        low_rank, to_merge = -1, None
                        for i in range(1, len(words)):
                            if (words[i-1], words[i]) in self.merges_rank:
                                if low_rank < 0 or self.merges_rank[words[i-1], words[i]] < low_rank:
                                    low_rank = self.merges_rank[words[i-1], words[i]]
                                    to_merge = i
                        if low_rank < 0:
                            break
                        #print(words, low_rank, words[to_merge-1], words[to_merge])
                        words = words[:to_merge-1] + [b''.join([words[to_merge-1], words[to_merge]])] + words[to_merge+1:]
                        #print(words)
                    #print(words)
                    tokens += [self.vocab_index[w] for w in words]
            if tokens_endoftext:
                tokens.append(tokens_endoftext[0])
                tokens_endoftext.pop(0)
        #print(tokens)
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This
        is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text"""
        all_bytes = b''.join(self.vocab[id] for id in ids)
        #print(all_bytes)
        return all_bytes.decode('utf-8', errors='replace')
