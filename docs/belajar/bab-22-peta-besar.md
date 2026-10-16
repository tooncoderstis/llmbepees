# Bab 22 — Peta Besar

> Catatan belajar, ditulis ulang dengan bahasa sendiri dari buku *"Memahami Matematika di Balik LLM"* (Bab 22).
> Bagian dari **Fase 12 — Layanan Inference & Demo Deploy**. Tidak ada kode khusus; ini rangkuman semua bab.

## 22.1 Seluruh LLM dalam satu halaman

Masukan:
```
teks →(BPE)→ [x_1, …, x_n],  x_t ∈ {1, …, V}
```

Embedding & posisi (Bab 8, 15):
```
H(0) = E[x_{1:n}] + P_{1:n} ∈ ℝ^{n×d}
```

Untuk setiap lapisan `ℓ = 1, …, L` (Bab 13–16):
```
H̃ = LN(H(ℓ−1))
Q_i = H̃W_Qᵢ,  K_i = H̃W_Kᵢ,  V_i = H̃W_Vᵢ
A_i = softmax(Q_iK_iᵀ/√d_k + M),  M_jk = { 0  k ≤ j ; −∞  k > j }
U = Concat(A_1V_1, …, A_hV_h)W_O
G = H(ℓ−1) + U
H(ℓ) = G + act(LN(G)W1)W2
```

Keluaran (Bab 10, 17):
```
Z = LN(H(L))Eᵀ ∈ ℝ^{n×V}
P(x_{t+1}=j | x_≤t) = e^{Z_tj} / ∑_k e^{Z_tk}
```

Pelatihan (Bab 6, 11, 12):
```
L(θ) = −(1/(n−1)) ∑_{t=1}^{n−1} ln P(x_{t+1} | x_≤t)
θ ← θ − η·m̂/(√v̂ + ε)
```

Dua belas baris itu — seluruh jalur maju dan pelatihan Transformer decoder-only kausal yang padat. **Batasnya:** ini **tidak** mencakup RoPE, GQA/MQA, SwiGLU, MoE, encoder-decoder, multimodal, atau model non-Transformer.

## 22.2 Tabel besaran, dari kecil ke raksasa

| Besaran | Mini-LLM | GPT-2 small | Llama 2 7B | GPT-3 175B |
|---|---|---|---|---|
| `V` | 8 | 50.257 | 32.000 | 50.257 |
| `d_model` | 4 | 768 | 4.096 | 12.288 |
| `L` | 1 | 12 | 32 | 96 |
| `h` | 1–2 | 12 | 32 | 96 |
| `d_k` | 4 atau 2 | 64 | 128 | 128 |
| `d_ff` | 8 | 3.072 | 11.008 | 49.152 |
| Konteks | 6 | 1.024 | 4.096 | 2.048 |
| Parameter | 188 | 124 juta | 6,7 miliar | 175 miliar |
| Token latih | 24 | ~10 miliar | 2 triliun | 300 miliar |
| Norm | LayerNorm | LayerNorm | RMSNorm | LayerNorm |
| Posisi | terlatih | terlatih | RoPE | terlatih |
| Aktivasi | ReLU | GELU | SwiGLU | GELU |

Kalimat yang benar bukan "semua model sama, hanya beda ukuran", melainkan: **semua model ini adalah varian dari satu kerangka yang sama**, dan sebagian besar matematika buku ini berlaku untuk semuanya.

## 22.3 Ke mana setiap angka pergi

Untuk `d_ff = 4d`, sebaran parameter per blok: **attention `4d²` (33,3%)**, **FFN `8d²` (66,7%)** — dua per tiga ada di FFN. Sebaran operasi hitung saat inference berbeda, karena attention punya suku `n²` tanpa parameter: untuk `d = 4096`, porsi suku `n²` ≈ 11% (`n=1024`), 33% (`n=4096`), 67% (`n=16384`).

## 22.4 Delapan hal yang, jika kamu ingat, cukup

1. LLM = fungsi deret token → distribusi probabilitas atas kosakata. Tidak lebih.
2. Komputasinya didominasi **perkalian matriks**; fungsi non-linear FFN **wajib** (tanpa itu dua proyeksi FFN runtuh jadi satu).
3. Loss = `−ln P(token benar)`; gradiennya terhadap logit = `p − y`.
4. **Attention** = rata-rata berbobot value, bobot dari softmax atas dot product query–key; `√d_k` menjaga varians skor 1.
5. **Attention buta urutan** — posisi harus disuntikkan (embedding posisi, RoPE, ALiBi).
6. **Residual** memberi gradien jalur tanpa perkalian → model dalam bisa dilatih.
7. Dua per tiga parameter tiap blok di FFN (GPT-3: 66%; GPT-2 small: 46% karena embedding besar).
8. Loss pada data nyata **tidak menuju nol** — batasnya entropi intrinsik bahasa.

## 22.5 Apa yang tidak dibahas buku ini

MoE, kuantisasi (INT8/INT4/GPTQ/AWQ), paralelisme (tensor/pipeline/sequence), FlashAttention rinci, state-space model (Mamba), multimodalitas, interpretability/transformer circuits, evaluasi & tolok ukur. Dengan fondasi buku ini, semua itu bisa dibaca langsung dari makalah aslinya.

## 22.6 Cara memverifikasi pemahamanmu

Sepuluh pertanyaan (jawaban ada di Bab 10, 10, 13, 13, 4, 16, 10, 11, 19, 18):
1. Kenapa keluaran model harus berjumlah 1, dan mekanisme apa yang memaksanya?
2. Kenapa softmax memakai `eˣ` dan bukan fungsi lain untuk membuat nilai positif?
3. Apa yang terjadi jika `√d_k` dihapus dari rumus attention?
4. Kenapa causal mask memakai `−∞` dan bukan 0?
5. Kenapa dua transformasi affine berurutan tanpa aktivasi tidak menambah daya ekspresif?
6. Berapa parameter model dengan `d=1024, L=24, V=32000`?
7. Kenapa gradien loss terhadap logit berbentuk `p − y`?
8. Kenapa residual connection memungkinkan model yang dalam?
9. Kenapa KV cache mempercepat inference, dan berapa memorinya?
10. Kenapa loss pelatihan mendatar di atas nol?

**Nomor 6:** `32000×1024 + 24×12×1024² = 32.768.000 + 301.989.888 = 334.757.888` ≈ **335 juta** parameter.

## 22.7 Penutup

Jika ada satu hal yang diharapkan tersampaikan: **tidak ada bagian dari LLM yang misterius.** Setiap angka yang keluar dari model bisa dilacak mundur sampai ke perkalian dan penjumlahan yang menghasilkannya. Model 175 miliar parameter dan model 188 parameter berbagi kerangka yang sama (Transformer decoder autoregresif), meskipun rincian formula, komponen, dan konfigurasinya tidak harus sama.

Skala adalah salah satu pembeda terbesar, tapi bukan satu-satunya: pilihan normalisasi, aktivasi, metode posisi, tokenizer, data, dan tahap pasca-pelatihan semuanya ikut menentukan. Dan pertanyaan tentang **kenapa** skala menghasilkan kemampuan tetap terbuka — itulah sebabnya bidang ini masih menarik.

## Rujukan

- Buku *"Memahami Matematika di Balik LLM"*, Bab 22.
