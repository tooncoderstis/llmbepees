"""Latih mini-LLM 188 parameter dari nol & periksa titik pemeriksaan buku (Bab 21).

Jalankan::

    .venv\\Scripts\\python scripts\\train_mini_llm.py

Cetak kurva loss dan pemeriksaan angka buku (loss `2,4480 -> 0,0955`, dll.).
"""

from __future__ import annotations

import sys

import numpy as np


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    from llmbepees.np.mini_llm import (
        SENTENCE_MEJA,
        TRAIN_SEQUENCES,
        init_params,
        next_token_probs,
        params_count,
        sequence_loss,
        train,
    )

    params = init_params()
    print(f"parameter   : {params_count(params)} (184 dilatih, P[6] tak tersentuh)")
    print(f"loss awal   : {sequence_loss(params, TRAIN_SEQUENCES):.4f}")
    print("\nmelatih 400 langkah (gradien beda-hingga terpusat, Adam lr=0,05)...")

    trained, history = train()
    print("\nlangkah   loss    perplexity")
    for step in sorted(history):
        loss = history[step]
        print(f"{step:>6}   {loss:.4f}  {np.exp(loss):.2f}")

    probs = next_token_probs(trained, SENTENCE_MEJA[:5])
    meja, tikar = trained["E"][5], trained["E"][6]
    cosine = float(meja @ tikar / (np.linalg.norm(meja) * np.linalg.norm(tikar)))
    final = sequence_loss(trained, TRAIN_SEQUENCES)

    print("\npemeriksaan titik buku:")
    print(f"  loss akhir        {final:.4f}   (buku 0,0955)")
    print(f"  P(meja)           {probs[5]:.4f}   (buku 0,7492)")
    print(f"  P(tikar)          {probs[6]:.4f}   (buku 0,2494)")
    print(f"  cos(meja, tikar)  {cosine:.4f}   (buku 0,9960)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
