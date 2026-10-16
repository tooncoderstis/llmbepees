# Lampiran — Rincian Eksperimen Bab 21

> Catatan belajar dari Lampiran buku *"Memahami Matematika di Balik LLM"*. Ringkasan keputusan yang menentukan hasil [Mini-LLM 188 parameter](bab-21-mini-llm-dari-nol.md).
>
> **Batas klaim:** ditujukan untuk **replikasi pedagogis**, bukan reproduksi bit demi bit. Angka rinci dapat berbeda sedikit; **tiga pemeriksaan** yang harus bertahan: arah/urutan penurunan loss, `P(meja)≈0,75`/`P(tikar)≈0,25`, dan routing attention akhir hampir one-hot.

## A.1 Model
Arsitektur Transformer decoder-only kausal, satu blok; `V=8`, `d=4`, `L=1`, head 1, `d_ff=8` (=`2d`), panjang maksimum 7; posisi terlatih; LayerNorm **Post-LN** (`ε=10⁻⁵`); aktivasi ReLU; **tanpa bias**; weight tying `W_out = Eᵀ`.

| Matriks | Ukuran | Parameter |
|---|---|---|
| `E` | 8 × 4 | 32 |
| `P` | 7 × 4 | 28 |
| `W_Q`/`W_K`/`W_V`/`W_O` | 4 × 4 | 16 masing-masing |
| `W1` | 4 × 8 | 32 |
| `W2` | 8 × 4 | 32 |
| **Total** | | **188** (184 dilatih) |

## A.2 Data & tokenisasi
Tokenisasi trivial: pecah pada spasi, setiap potongan ada di kosakata. Kosakata (0–7): `<bos> kucing duduk di atas meja tikar <eos>`. Empat kalimat, **urutannya berpengaruh** (diproses sebagai satu batch penuh): meja, tikar, meja, meja. Total **24 pasangan** (4 × 6).

## A.3 Inisialisasi
Normal `N(0; 0,5²)`, seluruh 188 parameter dari **satu penarikan acak**, **seed 42**, urutan pengisian `E → P → W_Q → W_K → W_V → W_O → W1 → W2`. (Implementasi proyek: `numpy.random.default_rng(42)`, `E` digambar pertama.) Simpangan baku 0,5 besar untuk model sekecil ini → loss awal 2,4480 (di atas `ln 8 = 2,0794`).

## A.4 Optimizer & jadwal
**Adam** (bukan AdamW), `η = 0,05` tetap, `β1 = 0,9`, `β2 = 0,999`, `ε = 10⁻⁸`, koreksi bias dipakai; **tanpa** weight decay, warmup, decay, atau clipping.

## A.5 Batch & jumlah pembaruan
Batch penuh (keempat kalimat sekaligus); tanpa gradient accumulation; **400** pembaruan; tanpa pengacakan (tiap langkah memakai seluruh korpus) → kurva loss mulus.

## A.6 Loss
Cross-entropy rata-rata per token target: `L(θ) = −(1/24) ∑_{24 pasangan} ln P_θ(target | konteks)`, dengan pengaman `10⁻¹²` di dalam logaritma.

## A.7 Cara gradien dihitung
**Bukan backpropagation**, melainkan **beda hingga terpusat** `∂L/∂θ_i ≈ (L(θ+h e_i) − L(θ−h e_i))/(2h)`, `h = 10⁻⁵`. Setiap langkah memerlukan **2 × 188 = 376** evaluasi loss. Alasan: hasil tak bergantung implementasi backprop dan bisa diperiksa langsung terhadap definisi turunan (Bab 7). Konsekuensi: galat pemotongan/pembulatan sekitar `10⁻⁶`.

## A.8 Presisi
**float64** — supaya angka tercetak stabil sampai empat desimal.

## A.9 Pseudocode
1. Siapkan kosakata 8 token & 4 kalimat.
2. Ubah tiap kalimat menjadi deret nomor token.
3. Tarik 188 angka `N(0; 0,5)` seed 42.
4. Isikan ke `E, P, W_Q, W_K, W_V, W_O, W1, W2` sesuai urutan.
5. Siapkan `m, v` Adam (nol).
6. Ulangi 400 langkah: hitung loss seluruh 24 pasangan; gradien numerik tiap parameter; `m ← 0,9m + 0,1g`; `v ← 0,999v + 0,001g²`; koreksi bias; `θ ← θ − 0,05·m̂/(√v̂ + 10⁻⁸)`.
7. Laporkan loss akhir, prediksi akhir, bobot attention akhir.

## A.10 Titik pemeriksaan

| Besaran | Awal | Akhir |
|---|---|---|
| Loss | 2,4480 | 0,0955 |
| Perplexity | 11,57 | 1,10 |
| `P(meja)` setelah *di atas* | mendekati merata | 0,7492 |
| `P(tikar)` setelah *di atas* | mendekati merata | 0,2494 |
| cos(meja, tikar) | 0,7557 | 0,9960 |

Tiga pemeriksaan terpenting (harus bertahan pada implementasi mana pun):
1. Loss akhir mendekati **0,0937 dari atas** (batas bawah dari perbandingan 3:1).
2. `P(meja) → 0,75`, `P(tikar) → 0,25` (meniru proporsi data, bukan satu pemenang).
3. Routing attention akhir **hampir one-hot**.

## Implementasi proyek
Diimplementasikan di [`src/llmbepees/np/mini_llm.py`](../../src/llmbepees/np/mini_llm.py) + [`scripts/train_mini_llm.py`](../../scripts/train_mini_llm.py), diuji [`tests/test_mini_llm.py`](../../tests/test_mini_llm.py). Reproduksi proyek **cocok persis** dengan tabel A.10 (loss `2,4480 → 0,0955`, `P(meja)=0,7492`, `P(tikar)=0,2494`, cos `0,9960`).
