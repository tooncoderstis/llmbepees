# Bab 17 — Lapisan Keluaran dan Perjalanan Penuh Satu Token

> Catatan belajar, ditulis ulang dengan bahasa sendiri dari buku *"Memahami Matematika di Balik LLM"* (Bab 17).
> Bagian dari **Fase 9 — Blok Transformer Lengkap**. Kode: [`src/llmbepees/np/transformer.py`](../../src/llmbepees/np/transformer.py) · Uji: [`tests/test_transformer.py`](../../tests/test_transformer.py).
>
> Bab ini tidak memperkenalkan gagasan baru. Yang baru adalah melihat semuanya sebagai **satu rangkaian utuh**.

## Masalah

Setelah blok terakhir, kita punya `H(L)` berukuran `n×d`. Yang dibutuhkan adalah `n×V` — satu skor untuk setiap token kosakata di setiap posisi. Ukurannya tidak cocok, jadi perlu satu langkah lagi: **lapisan keluaran**.

## Gagasan

### Perjalanan token *atas*

Masukan: `"<bos> kucing duduk di atas"`. Alur: token → embedding → posisi → attention → FFN → logit → probabilitas → prediksi.

| Tahap | Diwakili apa | Jumlah angka | Informasi baru | Belum dilakukan |
|---|---|---|---|---|
| 1. token → nomor | nomor urut di kosakata | 1 | tidak ada | nomor tidak punya makna |
| 2. nomor → embedding | baris ke-`i` matriks `E` | `d`=4 | identitas token | vektor sama di mana pun muncul |
| 3. + posisi | `E[x] + P` | 4 | posisi | masih hanya tentang token sendiri |
| 4. attention | rangkuman berbobot | 4 | **konteks** | belum digabung dengan representasi asli |
| 5. residual + LN | `LN(H(0)+MHA)` | 4 | (reguler) | — |
| 6. FFN | olah non-linear | 4 (tengah 8) | pengolahan di dalam posisi | — |
| 7. residual + LN | keadaan akhir | 4 | — | belum jadi jawaban |

Angka: `"atas" → 4`; embedding `[0,0 0,0 1,0 1,0]`; + posisi 4 (`[0,5 0,5 0,0 0,0]`) → `[0,5 0,5 1,0 1,0]`. Dua head attention memberi baris *atas*: head 1 `[0,0200, 0,2375, 0,2375, 0,1668, 0,3382]`, head 2 `[0,0211, 0,0427, 0,3563, 0,0726, 0,5074]`; setelah concat·`W_O` → `[0,7204, 0,6665, 0,6393, 0,7476]`.

Residual + LN: `[0,5 0,5 1,0 1,0] + [0,7204 0,6665 0,6393 0,7476] = [1,2204 1,1665 1,6393 1,7476]` → LN → `[−0,8793 −1,0919 0,7721 1,1991]`. FFN (naik ke 8, ReLU, turun ke 4) → `[−0,2067 −0,2861 0,2330 0,5596]`. Setelah residual + LN kedua, **keadaan akhir** token *atas*:

```
h_akhir = [−0,8678, −1,0861, 0,6953, 1,2585]
```

Empat angka ini memuat segala sesuatu yang diketahui model tentang posisi 4 setelah membaca seluruh konteks sebelumnya.

### Dari vektor terakhir ke logit

```
Z = H(L)·W_out        dengan weight tying:  W_out = Eᵀ
Z = H(L)·Eᵀ,          Z_ij = h_i · e_j
```

Logit untuk token `j` adalah **dot product** antara keadaan akhir model di posisi `i` dan vektor embedding token `j`. **Hati-hati:** `h·e = ‖h‖‖e‖cos θ`, jadi embedding bernorma besar bisa menang walau arahnya kurang cocok.

Contoh dengan `h_akhir` di atas:

| Token | e | Dot product | Logit |
|---|---|---|---|
| `<bos>` | [0,0,0,0] | 0 | 0,0000 |
| kucing | [1,0,1,0] | −0,8678 + 0,6953 | −0,1725 |
| duduk | [0,1,0,1] | −1,0861 + 1,2585 | 0,1725 |
| di | [1,1,0,0] | −0,8678 − 1,0861 | −1,9539 |
| atas | [0,0,1,1] | 0,6953 + 1,2585 | 1,9539 |
| meja | [1,0,0,1] | −0,8678 + 1,2585 | 0,3907 |
| tikar | [0,1,1,0] | −1,0861 + 0,6953 | −0,3907 |
| `<eos>` | [1,1,1,1] | −0,8678 − 1,0861 + 0,6953 + 1,2585 | 0,0000 |

### Dari logit menjadi probabilitas

Softmax (Bab 10) mengubah logit menjadi probabilitas:

| Token | Logit | `e^{z−max}` | Probabilitas |
|---|---|---|---|
| `<bos>` | 0,0000 | 0,1417 | 0,0747 |
| kucing | −0,1725 | 0,1193 | 0,0629 |
| duduk | 0,1725 | 0,1684 | 0,0888 |
| di | −1,9539 | 0,0201 | 0,0106 |
| atas | 1,9539 | 1,0000 | 0,5273 |
| meja | 0,3907 | 0,2095 | 0,1105 |
| tikar | −0,3907 | 0,0959 | 0,0506 |
| `<eos>` | 0,0000 | 0,1417 | 0,0747 |

Prediksi teratas = *atas* (52,7%); jawaban benar *meja* hanya 11,1%. Loss `L = −ln(0,1105) = 2,2032` nat — **lebih buruk** dari tebak-asal `ln 8 = 2,0794`. Wajar: seluruh bobot dipilih sembarangan untuk contoh, bukan hasil pelatihan (Bab 21 melatihnya sungguhan). Gradien `p − y = [0,0747, 0,0629, 0,0888, 0,0106, 0,5273, −0,8895, 0,0506, 0,0747]`.

### Semua posisi diprediksi sekaligus

Matriks logit berukuran **`n×V`**, bukan `1×V` — model menghasilkan prediksi untuk **setiap posisi** dalam satu kali jalan. Satu urutan `n` token memberi **`n − 1`** contoh pelatihan sekaligus (token terakhir tidak punya target di dalam urutan). Untuk urutan 2048 token: **2047 sinyal loss** dari satu kali jalan. Ini yang membuat pretraining LLM efisien secara ekstrem, dan ia bekerja hanya karena **causal mask** (Bab 13) menjamin tiap baris tidak melihat masa depan. Saat menghasilkan teks, baris 1…`n−1` dibuang; hanya baris terakhir dipakai.

### Ringkasan ukuran matriks

| Tahap | Ukuran | Mini-LLM | GPT-3 (n=2048) |
|---|---|---|---|
| Token ID | `n` | 5 | 2.048 |
| Embedding `X` | `n×d` | 5×4 | 2048×12288 |
| `Q,K,V` | `n×d` | 5×4 | 2048×12288 |
| Skor `QKᵀ` | `n×n` per head | 5×5 | 2048×2048×96 |
| Keluaran MHA | `n×d` | 5×4 | 2048×12288 |
| FFN tengah | `n×d_ff` | 5×8 | 2048×49152 |
| Keluaran blok | `n×d` | 5×4 | 2048×12288 |
| Logit | `n×V` | 5×8 | 2048×50257 |

`n×d` tidak pernah berubah sepanjang tumpukan; yang membesar hanya di dua tempat: matriks skor (`n×n`) dan lapisan tengah FFN (`n×d_ff`) — keduanya menyusut lagi sebelum keluar.

### Berapa banyak operasi

| Operasi | Ukuran | Perkalian |
|---|---|---|
| `HW_Q` | (5×4)(4×4) | 80 |
| `HW_K` / `HW_V` | | 80 / 80 |
| `QKᵀ` (2 head) | 2×(5×2)(2×5) | 100 |
| `AV` (2 head) | 2×(5×5)(5×2) | 100 |
| `Concat·W_O` | (5×4)(4×4) | 80 |
| FFN naik / turun | (5×4)(4×8) / (5×8)(8×4) | 160 / 160 |
| Logit | (5×4)(4×8) | 160 |
| **Total** | | **1.000** |

Seribu perkalian untuk lima token. Bandingkan GPT-3: `2·N·n = 2 × 175×10⁹ × 2048 = 7,17 × 10¹⁴` FLOP — sekitar **360 miliar kali** lebih banyak. (Catatan: satuannya harus konsisten; `2Nn` adalah hampiran lapisan padat dan tidak mencacah suku `n²` attention, normalisasi, atau softmax secara terpisah.)

### Rumus lengkap satu model

```
H(0) = E[x] + P
A(ℓ) = H(ℓ−1) + MHA(LN(H(ℓ−1)))
H(ℓ) = A(ℓ) + FFN(LN(A(ℓ)))
Z    = LN(H(L))·Eᵀ
p_t  = softmax(z_t)
```

Lima baris. Itulah seluruh jalur maju sebuah Transformer decoder-only kausal dalam bentuk paling lazim. (Rumus di atas ditulis dalam bentuk **Pre-LN**; penelusuran angka di catatan ini memakai **Post-LN** mengikuti mini-LLM. Letak LayerNorm berbeda → angkanya memang tidak sama; untuk satu lapisan, alasan stabilitas Pre-LN belum muncul.)

## Implementasi

- [`src/llmbepees/np/transformer.py`](../../src/llmbepees/np/transformer.py): `output_logits` (`Z = H·Eᵀ`), `decoder_forward` (tumpuk blok → logit), `approx_model_parameters`/`model_parameters`, `forward_macs`, `forward_flops`, `matrix_sizes`, `predictions_per_sequence`.
- Uji memverifikasi logit `[0, −0,1725, 0,1725, −1,9539, 1,9539, 0,3907, −0,3907, 0]`, probabilitas, parameter GPT-2/GPT-3, 1.000 perkalian mini-LLM, dan `2Nn`.

## Inti bab

Jalur maju sebuah decoder-only Transformer adalah **lima baris rumus yang dijalankan berulang pada matriks berukuran tetap**, dan logit di ujungnya hanyalah **dot product** antara keadaan akhir dan embedding setiap token — tidak ada mekanisme khusus. Dua hal yang layak diingat: ukuran matriks tidak pernah berubah sepanjang tumpukan (selalu `n×d`); dan matriks logit berukuran `n×V`, sehingga model memprediksi **semua posisi sekaligus** — saat dilatih semuanya dipakai, saat menghasilkan teks hanya baris terakhir.

## Cek pemahaman

1. Pada tahap mana informasi berpindah antar-token, dan pada tahap mana tidak?
2. Mengapa satu kalimat bisa menghasilkan banyak contoh pelatihan sekaligus?
3. Mengapa keliru mengatakan token dengan embedding paling searah pasti menang?

## Rujukan

- Buku *"Memahami Matematika di Balik LLM"*, Bab 17.
- Radford dkk. (2019), Brown dkk. (2020) — parameter & arsitektur GPT.
