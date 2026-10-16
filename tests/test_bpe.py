"""Uji tokenizer BPE (Bab 8.2) terhadap contoh korpus buku.

Toleransi desimal ke-3 (ADR-0002); di sini kebanyakan uji berupa kecocokan eksak.
"""

import pytest

from llmbepees.np.bpe import WORD_END, encode, split_symbols, train_bpe

CORPUS = {"rendang": 5, "rendah": 3, "renang": 2, "rendam": 4}

EXPECTED_MERGES = [
    ("r", "e"),
    ("re", "n"),
    ("ren", "d"),
    ("rend", "a"),
    ("n", "g"),
    ("ng", WORD_END),
]


def test_split_symbols() -> None:
    assert split_symbols("rendang") == ("r", "e", "n", "d", "a", "n", "g", "</w>")


def test_split_symbols_rejects_empty() -> None:
    with pytest.raises(ValueError):
        split_symbols("")


def test_train_bpe_merge_order() -> None:
    assert train_bpe(CORPUS, 6) == EXPECTED_MERGES


def test_train_bpe_single_merge_picks_earliest_tie() -> None:
    assert train_bpe(CORPUS, 1) == [("r", "e")]


def test_encode_final_tokenization() -> None:
    merges = train_bpe(CORPUS, 6)
    assert encode("rendang", merges) == ["renda", "ng</w>"]
    assert encode("rendah", merges) == ["renda", "h", "</w>"]
    assert encode("renang", merges) == ["ren", "a", "ng</w>"]
    assert encode("rendam", merges) == ["renda", "m", "</w>"]


def test_encode_token_counts() -> None:
    merges = train_bpe(CORPUS, 6)
    counts = {word: len(encode(word, merges)) for word in CORPUS}
    assert counts == {"rendang": 2, "rendah": 3, "renang": 3, "rendam": 3}


def test_frequent_word_gets_shortest_representation() -> None:
    merges = train_bpe(CORPUS, 6)
    assert len(encode("rendang", merges)) == min(len(encode(w, merges)) for w in CORPUS)


def test_train_bpe_stops_when_no_pairs() -> None:
    assert train_bpe({"a": 1}, 5) == [("a", WORD_END)]
