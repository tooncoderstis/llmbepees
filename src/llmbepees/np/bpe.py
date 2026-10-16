"""Byte Pair Encoding minimal (Bab 8.2) — rujukan kebenaran NumPy murni.

BPE berbasis karakter dengan penanda akhir kata ``</w>`` (bentuk paling mudah
dihitung, seperti contoh buku). Menggabungkan pasangan simbol yang paling sering
muncul, berulang, sampai sejumlah penggabungan tercapai.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

WORD_END = "</w>"

Pair = tuple[str, str]
Symbols = tuple[str, ...]


def split_symbols(word: str) -> Symbols:
    """Pecah kata menjadi simbol karakter, dengan ``</w>`` di akhir."""
    if not word:
        raise ValueError("kata tidak boleh kosong")
    return (*tuple(word), WORD_END)


def _pair_counts(symbol_freqs: Mapping[Symbols, int]) -> dict[Pair, int]:
    """Hitung frekuensi pasangan bersebelahan; urutan = kemunculan pertama."""
    counts: dict[Pair, int] = {}
    for symbols, freq in symbol_freqs.items():
        for pair in zip(symbols, symbols[1:], strict=False):
            counts[pair] = counts.get(pair, 0) + freq
    return counts


def _merge_symbols(symbols: Symbols, pair: Pair) -> Symbols:
    """Gabung ``pair`` menjadi satu simbol di seluruh deretan simbol."""
    a, b = pair
    merged = a + b
    out: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def _to_symbol_freqs(word_freqs: Mapping[str, int]) -> dict[Symbols, int]:
    freqs: dict[Symbols, int] = {}
    for word, freq in word_freqs.items():
        if freq <= 0:
            raise ValueError(f"frekuensi kata {word!r} harus positif")
        symbols = split_symbols(word)
        freqs[symbols] = freqs.get(symbols, 0) + freq
    return freqs


def train_bpe(word_freqs: Mapping[str, int], num_merges: int) -> list[Pair]:
    """Latih daftar merge BPE (Bab 8.2).

    Seri (frekuensi sama) diputus dengan memilih pasangan yang muncul lebih dulu.
    """
    if num_merges < 0:
        raise ValueError("num_merges tidak boleh negatif")
    symbol_freqs = _to_symbol_freqs(word_freqs)
    merges: list[Pair] = []
    for _ in range(num_merges):
        counts = _pair_counts(symbol_freqs)
        if not counts:
            break
        best: Pair | None = None
        best_count = -1
        for pair, count in counts.items():
            if count > best_count:
                best_count = count
                best = pair
        assert best is not None
        merges.append(best)
        merged_freqs: dict[Symbols, int] = {}
        for symbols, freq in symbol_freqs.items():
            new_symbols = _merge_symbols(symbols, best)
            merged_freqs[new_symbols] = merged_freqs.get(new_symbols, 0) + freq
        symbol_freqs = merged_freqs
    return merges


def encode(word: str, merges: Iterable[Pair]) -> list[str]:
    """Terapkan merge secara berurutan untuk memecah kata menjadi token."""
    symbols = split_symbols(word)
    for pair in merges:
        symbols = _merge_symbols(symbols, pair)
    return list(symbols)
