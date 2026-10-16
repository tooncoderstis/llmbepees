# Panduan — Cara Memahami Proyek Ini

> Panduan membaca & memahami proyek `llmbepees`. Bukan ringkasan materi, melainkan **cara menavigasi** catatan, kode, dan uji agar Anda benar-benar paham alurnya.
> Terkait: [`README.md`](README.md) (indeks bab) · [`STATUS.md`](../../STATUS.md) (progres) · [`PRD.md`](../../PRD.md) (tujuan).

## 1. Gambaran besar

Proyek ini punya tiga lapis yang saling menopang. Setiap bab buku menghasilkan ketiganya sekaligus:

```
Catatan belajar (kenapa & bagaimana)   →  docs/belajar/bab-*.md
Kode NumPy    (rumus yang bisa jalan)  →  src/llmbepees/np/*.py
Uji           (bukti angkanya benar)   →  tests/test_*.py
```

Aturannya: **satu bab = satu catatan + satu modul + satu file uji**. Jadi Anda bisa masuk lewat mana saja — baca catatan, lihat kodenya, lalu jalankan ujinya untuk mengonfirmasi.

## 2. Peta repo

```
llmbepees/
├─ docs/
│  ├─ belajar/            ← catatan belajar (bahasa Indonesia)
│  │  ├─ bab-01..bab-22.md + lampiran-eksperimen-bab-21.md
│  │  ├─ cheatsheet.md    ← ringkasan rumus per bab
│  │  ├─ glosarium.md     ← arti istilah kunci
│  │  └─ PANDUAN.md       ← file ini
│  ├─ adr/                ← keputusan desain (kenapa dibuat begini)
│  ├─ referensi/          ← makalah yang dirujuk buku
│  └─ runbooks/           ← cara operasional (deploy dll)
├─ data/
│  └─ corpus.txt          ← korpus char-level untuk model kecil (Fase 11)
├─ src/llmbepees/
│  ├─ np/                 ← implementasi NumPy murni (rujukan kebenaran)
│  ├─ torch/              ← port PyTorch (paritas ADR-0001) + model char-level kecil
│  └─ serve/              ← layanan FastAPI demo (engine + app)
├─ deploy/                ← Dockerfile & skrip build image (EasyPanel)
├─ scripts/               ← latih mini-LLM & model char-level
├─ tests/                 ← uji pytest (verifikasi angka buku + paritas + layanan)
├─ PRD.md · STATUS.md · CHANGELOG.md
└─ pyproject.toml
```

Status terkini ada di `STATUS.md`; riwayat perubahan di `CHANGELOG.md`; **kenapa** sebuah keputusan diambil ada di `docs/adr/`.

## 3. Apa yang sudah dibuat (Fase 1–12, Bab 1–22 + Lampiran)

**Catatan** (`docs/belajar/`) — 22 bab + Lampiran berformat **Masalah → Gagasan → Inti bab**, tiap rumus disertai contoh angka yang bisa dihitung ulang, ditutup "Cek pemahaman".

**Kode NumPy** (`src/llmbepees/np/`) — 23 modul, masing-masing menyalin rumus buku apa adanya:

| Bab | Modul | Fungsi inti |
|---|---|---|
| 3 | `vectors.py` | `dot`, `norm_l2`, `cosine_similarity`, `add`, `scale`, `hadamard` |
| 4 | `matrices.py` | `matmul`, `matvec`, `transpose`, `matadd`, `identity`, `flops_matmul` |
| 5 | `probability.py` | `validate_distribution`, `conditional_probability`, `expected_value`, `variance`, `sample_categorical`, `normal_pdf` |
| 6 | `information.py` | `entropy`, `cross_entropy`, `kl_divergence`, `perplexity`, `mean_negative_log_likelihood` |
| 7 | `calculus.py` | `numerical_gradient`, `gradient_descent_step`, `sigmoid`/`relu` + turunan, `linear_backward` |
| 8 | `bpe.py` | `split_symbols`, `train_bpe`, `encode` (tokenizer BPE) |
| 8 | `embedding.py` | `one_hot`, `embedding_lookup`, `embed_one_hot`, `tied_logits`, `embedding_parameters` |
| 9 | `activation.py` | `gelu`, `gelu_approx`, `silu`, `tanh`, turunan; registri `ACTIVATIONS` (ReLU/sigmoid dipakai ulang dari `calculus`) |
| 9 | `neuron.py` | `neuron`, `linear`/`dense`, `layer_parameters`, `collapse_linear_layers`, `ffn`, `ffn_parameters`, `swiglu`, `ffn_swiglu` |
| 10 | `softmax.py` | `softmax`, `log_softmax`, `log_sum_exp` (semua stabil secara numerik) |
| 10 | `loss.py` | `softmax_cross_entropy` (`LSE(z) − z_c`), `softmax_cross_entropy_gradient` (`p − y`), `label_smoothing`, pembanding MSE |
| 11 | `backprop.py` | aturan mundur (`matmul_backward`, `relu_backward`, `residual_*`), `two_layer_mlp_backward`, `training_flops` (`6ND`), `activation_memory`, `factor_power` |
| 12 | `optimizer.py` | `sgd_step`, `momentum_step`, `adam_step`/`adamw_step` (+koreksi bias), `lr_warmup`/`lr_cosine`, `gradient_clip`, `adam_memory_bytes` (`16N`) |
| 13 | `attention.py` | `project_qkv`, `attention_scores`/`attention_weights`, `causal_mask`, `scaled_dot_product_attention`, `score_variance`, `attention_flops` |
| 14 | `multihead.py` | `split_heads`/`concat_heads`, `multi_head_attention`, `kv_cache_bytes`, `kv_cache_ratio` |
| 15 | `position.py` | `add_position`, `sinusoidal_encoding`, `rope_theta`/`rope_rotate`/`rope_apply`, `alibi_bias`, `position_interpolation` |
| 16 | `layernorm.py` | `layernorm`, `rmsnorm` |
| 16 | `block.py` | `pre_ln_block`, `post_ln_block`, `stack_blocks`, `block_parameters` (dll.) |
| 17 | `transformer.py` | `output_logits` (`Z=H·Eᵀ`), `decoder_forward`, `approx_model_parameters`, `forward_macs`/`forward_flops`, `matrix_sizes` |
| 18 | `training.py` | `shift_pairs`, `sequence_cross_entropy`, `tokens_per_step`, `gradient_accumulation`, `scaling_law`, `chinchilla_tokens` |
| 19 | `inference.py` | `greedy`, `temperature_softmax`, `top_k`/`top_p`, `sample`, `decode_next`, `beam_step`, KV-cache helpers |
| 21 | `mini_llm.py` | `init_params`, `forward`, `sequence_loss`, `finite_difference_gradients`, `train`, `next_token_probs`, `generate` |
| 20 | `alignment.py` | `sft_loss`, `bradley_terry`, `reward_model_loss`, `rlhf_objective`, `dpo_loss`, `lora_delta`/`lora_parameters` |

**Kode PyTorch** (`src/llmbepees/torch/`, paritas ADR-0001):

| Modul | Isi |
|---|---|
| `functional.py` | port operasi inti (softmax, layernorm, gelu/silu, attention, multi-head, cross-entropy) — setara versi NumPy |
| `model.py` | GPT char-level kecil (Pre-LN, weight tying) + `CharTokenizer`; untuk demo (ADR-0006) |

**Kode layanan** (`src/llmbepees/serve/`, FastAPI — ADR-0003):

| Modul | Isi |
|---|---|
| `engine.py` | `Engine` (bungkus model char-level; `generate` & `explain` per-langkah) |
| `app.py` | FastAPI: `GET /health`, `POST /generate`, `POST /explain`, halaman chat `GET /` + rate-limit sederhana |

**Uji** (`tests/`) — 27 file, **314 uji lolos**, memverifikasi angka buku (toleransi desimal ke-3), **paritas NumPy↔PyTorch**, dan endpoint layanan. Uji mini-LLM melatih 400 langkah sekali (~25 dtk); uji PyTorch/FastAPI dilewati otomatis bila paketnya tidak ada.

Semua lapis terisi: catatan, kode NumPy (rujukan), port PyTorch (paritas), dan layanan demo FastAPI. Demo dijalankan lokal via `uvicorn llmbepees.serve.app:app` (lihat §7).

## 4. Cara memahami tiap lapis

### a. Membaca catatan
Baca berurutan Bab 1 → 22 (judul bab: `README.md`). Tiap bab menjawab satu pertanyaan dan menutup dengan **Inti bab** + **Cek pemahaman**. Kalau macet pada simbol, buka `bab-02-notasi.md` atau `cheatsheet.md`.

### b. Membaca kode
Setiap fungsi = satu rumus. Docstring menyebut bab asalnya, jadi kode dibaca sebagai "rumus yang bisa dijalankan", bukan kotak hitam.

```python
def dot(a, b):
    """Dot product: a · b = sum_i a_i * b_i — dua vektor jadi satu skalar (Bab 3.4)."""
    ...
```

### c. Membaca uji sebagai kunci jawaban
Uji memuat angka persis dari buku. Kalau ragu memahami sebuah rumus, baca ujinya — di situlah angkanya diverifikasi.

```python
def test_dot_first_example():
    assert dot([2.0, 3.0], [4.0, -1.0]) == pytest.approx(5.0, abs=TOL)  # 2·4 + 3·(−1) = 5
```

### d. Rujukan cepat
Butuh rumus → `cheatsheet.md`. Lupa istilah → `glosarium.md`. Ingin tahu *kenapa* diputuskan begitu → `docs/adr/`.

## 5. Benang merah konsep (Bab 1–22 → menuju LLM)

Bab-bab yang sudah dikerjakan membentuk satu alur logis:

1. **Bab 1** — LLM = fungsi yang memetakan deret token → distribusi probabilitas token berikutnya.
2. **Bab 2** — cara membaca rumus (`∑`, `∏`, `eˣ`, `ln`) agar sisanya bisa dibaca.
3. **Bab 3** — isi LLM berbentuk **vektor**; **dot product** = ukuran kecocokan (dasar attention).
4. **Bab 4** — **matriks** = banyak dot product sekaligus; dua lapisan linear = satu lapisan → fungsi **non-linear wajib**.
5. **Bab 5** — keluaran = **distribusi probabilitas**; **aturan rantai** membenarkan "menebak satu token" = memodelkan bahasa.
6. **Bab 6** — mengubah distribusi menjadi satu angka: **loss = −log p(token benar)**; **perplexity = e^loss**.
7. **Bab 7** — **turunan/gradien** menunjukkan ke arah mana tiap parameter digeser agar loss turun; **aturan rantai** = dasar backpropagation.
8. **Bab 8** — gerbang masuk: **tokenisasi** memecah teks jadi sub-kata, **embedding** menukarnya menjadi vektor yang **dipelajari** (bukan ditulis manusia).
9. **Bab 9** — **neuron, lapisan, FFN**: komponen yang memuat ~2/3 parameter tiap blok; **fungsi non-linear** (ReLU/GELU/SiLU) membuat kedalaman model berarti — tanpa itu dua lapisan linear runtuh jadi satu.
10. **Bab 10** — **softmax** mengubah logit menjadi probabilitas; loss `LSE(z) − z_c`; dan gradiennya yang legendaris **`p − y`** — besarnya otomatis sebanding dengan kesalahan.
11. **Bab 11** — **backpropagation**: membawa sinyal `p − y` dari ujung ke seluruh model lewat aturan rantai mundur (biaya ≈ 2× forward); **residual connection** `y = x + F(x)` menyediakan jalur identitas agar gradien tidak meluruh.
12. **Bab 12** — **optimizer**: setelah gradien (arah), tentukan ukuran langkah. **Adam** melepas besar langkah dari skala gradien (`m̂/√v̂`), ditambah warmup, cosine decay, dan gradient clipping.
13. **Bab 13** — **attention**: satu-satunya jembatan antarposisi. Setiap token membuat **query/key/value**; kecocokan `QKᵀ` di-softmax jadi bobot, lalu mencampur `V`. `√d_k` menjaga skala, **causal mask** mencegah mencontek.
14. **Bab 14** — **multi-head**: beberapa attention di subruang `d/h` memberi beberapa cara pandang terhadap konteks **tanpa menambah parameter**; hasilnya digabung lewat `W_O`.
15. **Bab 15** — **informasi posisi**: attention **buta urutan**, jadi urutan harus disuntikkan — learned `X+P`, sinusoidal, **RoPE** (memutar Q/K tiap lapisan sehingga dot product hanya bergantung selisih posisi), atau ALiBi (penalti jarak).
16. **Bab 16** — **blok Transformer**: attention + FFN dibungkus **residual** & **normalisasi** (LayerNorm/RMSNorm); masuk `n×d`, keluar `n×d` sehingga bisa ditumpuk. FFN memuat ~⅔ parameter.
17. **Bab 17** — **lapisan keluaran**: keadaan akhir diubah menjadi logit lewat dot product dengan embedding (`Z=H·Eᵀ`), lalu softmax; model memprediksi **semua posisi** sekaligus.
18. **Bab 18** — **pretraining**: menebak token berikutnya dengan label teks itu sendiri (self-supervised); biaya `C ≈ 6ND`, loss awal `ln V`, dan scaling law `L ∝ N^{−0,076}`, Chinchilla `D ≈ 20N`.
19. **Bab 19** — **inference**: pemilihan token (greedy/sampling/temperature/top-k/top-p/beam) dan **KV cache** (attention `O(n²)→O(n)`; decode dibatasi bandwidth memori).
20. **Bab 20** — **penyelarasan**: SFT (loss hanya di jawaban), reward model (Bradley–Terry), RLHF (`reward − β·KL`, PPO), **DPO** (langsung dari preferensi), dan **LoRA** (fine-tuning hemat parameter).
21. **Bab 21 (MILESTONE)** — **latih mini-LLM 188 parameter dari nol**: loss `2,4480 → 0,0955`, `P(meja)=0,7492`, cos(meja,tikar) `0,9960`, attention one-hot — struktur muncul dari optimasi, bukan diprogram.

Ringkasnya: Bab 1 memotret tujuan; Bab 2–4 perkakas angka; Bab 5–6 ukuran benar/salah; Bab 7 cara memperbaiki; Bab 8 teks→angka; Bab 9 neuron & FFN; Bab 10 jalur maju & `p−y`; Bab 11 backpropagation; Bab 12 optimizer; Bab 13–17 attention → blok → model penuh; Bab 18 cara melatihnya; Bab 19 cara memakainya; Bab 20 menyelaraskannya jadi asisten; Bab 21 **membuktikan semuanya bekerja** pada model 188 parameter. Fase 11 mem-port ke PyTorch (paritas) + model char-level; Fase 12 demo web.

## 6. Peta baca rinci

Urutan yang disarankan. Kolom kanan = perintah untuk melihat sendiri.

| # | Baca catatan | Lihat kode | Buktikan | Estimasi |
|---|---|---|---|---|
| 1 | `bab-01-apa-yang-dihitung-llm.md` | — | — | 20 mnt |
| 2 | `bab-02-notasi.md` | — | — | 20 mnt |
| 3 | `bab-03-vektor.md` | `np/vectors.py` | `pytest tests/test_vectors.py` | 30 mnt |
| 4 | `bab-04-matriks.md` | `np/matrices.py` | `pytest tests/test_matrices.py` | 40 mnt |
| 5 | `bab-05-probabilitas.md` | `np/probability.py` | `pytest tests/test_probability.py` | 40 mnt |
| 6 | `bab-06-informasi.md` | `np/information.py` | `pytest tests/test_information.py` | 40 mnt |
| 7 | `bab-07-turunan.md` | `np/calculus.py` | `pytest tests/test_calculus.py` | 50 mnt |
| 8 | `bab-08-tokenisasi-embedding.md` | `np/bpe.py`, `np/embedding.py` | `pytest tests/test_bpe.py tests/test_embedding.py` | 50 mnt |
| 9 | `bab-09-neuron-lapisan-aktivasi.md` | `np/activation.py`, `np/neuron.py` | `pytest tests/test_activation.py tests/test_neuron.py` | 50 mnt |
| 10 | `bab-10-softmax-loss.md` | `np/softmax.py`, `np/loss.py` | `pytest tests/test_softmax.py tests/test_loss.py` | 50 mnt |
| 11 | `bab-11-backpropagation.md` | `np/backprop.py` | `pytest tests/test_backprop.py` | 50 mnt |
| 12 | `bab-12-optimizer.md` | `np/optimizer.py` | `pytest tests/test_optimizer.py` | 50 mnt |
| 13 | `bab-13-attention.md` | `np/attention.py` | `pytest tests/test_attention.py` | 60 mnt |
| 14 | `bab-14-multi-head-attention.md` | `np/multihead.py` | `pytest tests/test_multihead.py` | 50 mnt |
| 15 | `bab-15-informasi-posisi.md` | `np/position.py` | `pytest tests/test_position.py` | 50 mnt |
| 16 | `bab-16-blok-transformer.md` | `np/layernorm.py`, `np/block.py` | `pytest tests/test_layernorm.py tests/test_block.py` | 60 mnt |
| 17 | `bab-17-lapisan-keluaran.md` | `np/transformer.py` | `pytest tests/test_transformer.py` | 50 mnt |
| 18 | `bab-18-pretraining.md` | `np/training.py` | `pytest tests/test_training.py` | 50 mnt |
| 19 | `bab-19-inference.md` | `np/inference.py` | `pytest tests/test_inference.py` | 50 mnt |
| 20 | `bab-20-sft-rlhf-dpo-lora.md` | `np/alignment.py` | `pytest tests/test_alignment.py` | 50 mnt |
| 21 | `bab-21-mini-llm-dari-nol.md` | `np/mini_llm.py` | `pytest tests/test_mini_llm.py` (+ `python scripts/train_mini_llm.py`) | 60 mnt |
| 11a | *(port PyTorch)* | `torch/functional.py`, `torch/model.py` | `pytest tests/test_torch_parity.py tests/test_torch_model.py` (+ `python scripts/train_char_model.py`) | 60 mnt |
| 22 | `bab-22-peta-besar.md` | — (rangkuman) | — | 30 mnt |
| — | `lampiran-eksperimen-bab-21.md` | `np/mini_llm.py` | `pytest tests/test_mini_llm.py` | 20 mnt |
| 12 | *(layanan demo)* | `serve/engine.py`, `serve/app.py` | `pytest tests/test_serve.py` (+ `uvicorn llmbepees.serve.app:app`) | 40 mnt |

Cara pakai tiap baris: baca catatannya, buka modulnya untuk melihat rumus dalam kode, lalu jalankan ujinya. Jika ingin mencoba sendiri:

```sh
.venv\Scripts\python -c "from llmbepees.np import dot, cosine_similarity; print(dot([2,3],[4,-1]))"
```

> Bab 2 hampir tanpa kode (kamus notasi); Bab 1 murni konsep. Bab 3–22 punya kode + uji; Bab 22 murni rangkuman.

### Bab berikutnya (belum ditulis)

- Tidak ada — **seluruh 22 bab + Lampiran tuntas.** Lanjutan opsional (BACKLOG): fine-tuning lanjutan, kuantisasi, MoE, dll. (lihat `PRD.md`).

Saat menulis bab baru, teks PDF-nya diekstrak dulu (lihat `scripts/extract_bab.py` dan bagian "Menjalankan" di bawah).

## 7. Menjalankan & menguji

```sh
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # untuk uji paritas
.venv\Scripts\ruff check .
.venv\Scripts\pytest
```

**Menjalankan demo (layanan inference):**

```sh
.venv\Scripts\python -m pip install -e ".[serve]"
.venv\Scripts\python -m uvicorn llmbepees.serve.app:app --port 8000
# buka http://127.0.0.1:8000/  (halaman chat); API: /health, /generate, /explain
```

Jalankan demo secara lokal seperti di atas. (Versi tayang menyusul pada fase deploy; lihat `docs/runbooks/deploy.md`.)

`pytest` hijau berarti semua angka buku cocok (toleransi desimal ke-3, lihat [ADR-0002](../adr/0002-strategi-verifikasi.md)).

**Membaca teks bab dari PDF** (untuk menulis catatan bab baru). PDF buku tidak di-commit (hak cipta) dan `pypdf` ada di extra `dev`:

```sh
.venv\Scripts\python scripts\extract_bab.py "per bab\bab 20.pdf"
```

## 8. Status & berikutnya

> Bagian catatan belajar ini disalin dari `aasallm` (mirror starter); seluruh 22 bab + Lampiran **sudah lengkap** dan menjadi rujukan. Progres proyek `llmbepees` sendiri ada di [`STATUS.md`](../../STATUS.md).
>
> Catatan belajar (Fase 0–12, Bab 1–22 + Lampiran): **selesai** ✅ — lihat riwayat di `aasallm`.
>
> **Berikutnya (llmbepees)**: Fase 1 (Tokenizer BPE & dataset domain) → Fase 2 (model sedang + pretraining) → Fase 3 (SFT) → Fase 4 (grounding/RAG) → Fase 5 (evaluasi faktual & anti-halusinasi) → Fase 6 (deploy CPU).

Detail dan daftar centang ada di [`STATUS.md`](../../STATUS.md) dan [`README.md`](README.md).

## 9. Dokumen lain yang berguna

- Tujuan & lingkup proyek → [`PRD.md`](../../PRD.md)
- Progres terkini → [`STATUS.md`](../../STATUS.md)
- Riwayat rilis → [`CHANGELOG.md`](../../CHANGELOG.md)
- Keputusan teknis → [`docs/adr/`](../adr/)
- Makalah rujukan → [`docs/referensi/`](../referensi/)
