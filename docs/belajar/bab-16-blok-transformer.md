# Bab 16 — Blok Transformer Lengkap

> Catatan belajar, ditulis ulang dengan bahasa sendiri dari buku *"Memahami Matematika di Balik LLM"* (Bab 16).
> Bagian dari **Fase 9 — Blok Transformer Lengkap**. Kode: [`src/llmbepees/np/layernorm.py`](../../src/llmbepees/np/layernorm.py), [`src/llmbepees/np/block.py`](../../src/llmbepees/np/block.py) · Uji: [`tests/test_layernorm.py`](../../tests/test_layernorm.py), [`tests/test_block.py`](../../tests/test_block.py).

## Masalah

Potongannya sudah ada: **multi-head attention** (Bab 13–14) memindahkan informasi antarposisi, **FFN** (Bab 9) mengolah informasi dalam setiap posisi, **informasi posisi** (Bab 15) memberi tahu urutan. Yang belum ada adalah **susunannya**. Tiga pertanyaan: (1) urutan menjalankannya? (2) apa yang menjaga angka tidak meledak/menyusut saat diulang puluhan kali? (3) apa yang menjaga gradien sampai ke lapisan bawah?

Jawaban (2) adalah **normalisasi**; jawaban (3) adalah **residual connection**. Keduanya terutama membantu optimasi & stabilitas — meski keduanya tetap mengubah fungsi yang dihitung model (LayerNorm adalah operasi non-linear berparameter, residual mengubah kelas fungsi yang mudah direpresentasikan).

## Gagasan

### Membangun blok satu langkah pada satu waktu

Bentuk yang dibangun disebut **Pre-LN** (dipakai hampir semua model modern). `H` adalah representasi seluruh token `n×d`.

| Panah | Operasi | Ukuran | Informasi baru |
|---|---|---|---|
| 1 | `H → LayerNorm` | `n×d` | tidak ada (hanya seragamkan skala) |
| 2 | `LayerNorm → Attention` | `n×d` | **konteks** (satu-satunya tempat info antarposisi) |
| 3 | `Attention → Residual` | `n×d` | — |
| 4 | `Residual → LayerNorm` | `n×d` | tidak ada |
| 5 | `LayerNorm → FFN` | `n×d` | olah non-linear di dalam posisi |
| 6 | `FFN → Residual` | `n×d` | — |

Karena normalisasi ada **di depan** sub-blok, bentuknya disebut Pre-LN. Dua rumus inilah seluruh isi satu blok:

```
H'  = H + MHA(LN(H))
H'' = H' + FFN(LN(H'))
```

**Mengapa residual penting.** Saat model dipakai: attention menghasilkan rangkuman konteks, tetapi informasi asli token tetap berharga — penjumlahan membuat keduanya tersedia. Saat dilatih (yang menentukan): turunan dari penjumlahan mengandung **suku identitas** yang meneruskan gradien tanpa dikalikan apa pun. Tanpa itu, dengan 96 lapisan dan faktor 0,8 per lapisan, gradien yang sampai ke lapisan pertama tinggal `5,0 × 10⁻¹⁰` — praktis nol.

**Perhatikan:** masuk `n×d`, keluar `n×d`. Karena ukurannya tidak berubah, blok bisa ditumpuk sebanyak apa pun tanpa mengubah satu baris rumus.

### LayerNorm, dilihat lebih dekat

Tanpa normalisasi, jika tiap lapisan mengalikan besaran vektor dengan 1,1, setelah 96 lapisan besarnya `1,1⁹⁶ = 9,4 × 10³` kali; faktor 0,9 → `0,9⁹⁶ = 4,0 × 10⁻⁵`.

```
μ  = (1/d) ∑ h_i
σ² = (1/d) ∑ (h_i − μ)²
LN(h)_i = γ_i · (h_i − μ)/√(σ² + ε) + β_i
```

Normalisasi dilakukan **sepanjang dimensi fitur, untuk setiap token terpisah** — token posisi 1 dan 500 dinormalisasi dengan rata-rata & variansnya masing-masing. Tidak ada informasi yang berpindah antarposisi.

**Contoh** `h = [2, 4, 4, 6]`, `γ=1, β=0`:

| Langkah | Perhitungan | Hasil |
|---|---|---|
| `μ` | (2+4+4+6)/4 | 4,0000 |
| `h_i − μ` | | [−2, 0, 0, 2] |
| `(h_i − μ)²` | | [4, 0, 0, 4] |
| `σ²` | (4+0+0+4)/4 | 2,0000 |
| `σ` | √2 | 1,4142 |
| hasil | `(h_i−μ)/σ` | [−1,4142, 0, 0, 1,4142] |

Rata-rata hasilnya 0, variansnya 1.

**Dua syarat yang sering terlewat.** (1) Pada Pre-LN, yang dinormalisasi adalah **masukan sub-blok**, bukan aliran residualnya — `H + F(LN(H))` sendiri tidak otomatis ternormalisasi dan besarannya memang cenderung tumbuh. (2) `γ` dan `β` **dilatih**, sehingga keluaran akhirnya boleh punya rata-rata/varians selain 0/1; yang dijamin hanya bagian sebelum `γ`/`β`.

### RMSNorm

Model modern (Llama, Mistral, Gemma) memakai versi lebih murah:

```
RMSNorm(h)_i = γ_i · h_i / √( (1/d) ∑ h_j² + ε )
```

`ε` berada **di dalam** akar. Dua perbedaan dari LayerNorm: **tidak ada pengurangan rata-rata** dan **tidak ada `β`**.

**Contoh** `h = [2, 4, 4, 6]`: RMS `= √((4+16+16+36)/4) = √18 = 4,2426`, hasilnya `[0,4714, 0,9428, 0,9428, 1,4142]`. Bandingkan LayerNorm `[−1,4142, 0, 0, 1,4142]`: RMSNorm **mempertahankan tanda dan proporsi asli**, LayerNorm memusatkan di nol.

| | LayerNorm | RMSNorm |
|---|---|---|
| Hitung rata-rata | ya | tidak |
| Hitung sebaran | varians | momen kedua mentah (akar rata-rata kuadrat) |
| Kurangi rata-rata | ya | tidak |
| Parameter | `2d` (γ, β) | `d` (γ) |

### Perhitungan lengkap mini-LLM (opsional)

Mini-LLM memakai bentuk **Post-LN** (normalisasi **setelah** penjumlahan residual), sesuai makalah Transformer asli. Setelah residual pertama, `H(0) + MHA` baris *atas* = `[1,2204, 1,1665, 1,6393, 1,7476]`; LayerNorm-nya → `[−0,8793, −1,0919, 0,7721, 1,1991]`. FFN (`d=4`, `d_ff=8`, ReLU, `W2 = 0,5·W1ᵀ`) baris *atas* → `[−0,2067, −0,2861, 0,2330, 0,5596]`. Setelah residual + LayerNorm kedua, keadaan akhir *atas* = `[−0,8678, −1,0861, 0,6953, 1,2585]`.

### Menumpuk blok

```
H(0) = X + P
H(ℓ) = Block_ℓ(H(ℓ−1)),  ℓ = 1, …, L
logit = LN_final(H(L))·W_out
```

Setiap blok punya parameter sendiri. Ukuran matriks **tidak pernah berubah** sepanjang tumpukan — selalu `n×d`. Yang berubah hanya isinya.

### Menghitung jumlah parameter

| Komponen per blok | Parameter |
|---|---|
| `W_Q, W_K, W_V, W_O` | `4d²` |
| `W1` (`d×d_ff`) | `d·d_ff` |
| `W2` (`d_ff×d`) | `d_ff·d` |
| 2×LayerNorm | `4d` |
| **Total** | **`4d² + 2·d·d_ff + 4d`** |

Dengan `d_ff = 4d` dan mengabaikan `4d`: `≈ 12d²`. Total model:

```
N ≈ V·d + n_max·d + L·12d²
```

**GPT-2 small** (`V=50257, d=768, L=12, n_max=1024`): embedding `38.597.376` + posisi `786.432` + 12 blok `84.934.656` = **124.318.464** (sebenarnya 124.439.808; rumus meleset <0,1%). **GPT-3 175B** (`d=12288, L=96`): `617.558.016 + 25.165.824 + 173.946.175.488 = 174.588.899.328` ≈ 175 miliar.

**Sebaran parameter dalam satu blok** (untuk `d_ff=4d`): attention `4d²` = **33,3%**, FFN `8d²` = **66,7%**. Dua pertiga parameter setiap blok ada di FFN, bukan di attention — sering mengejutkan karena attention yang paling banyak dibahas.

### Varian lain: Post-LN

```
Post-LN: H' = LN(H + F(H))
Pre-LN:  H' = H + F(LN(H))
```

Pada Post-LN, gradien harus melewati LayerNorm setiap kali melewati satu blok, sehingga jalur residualnya tidak murni lagi. Pada Pre-LN ada jalur langsung dari `H` ke `H'`, sehingga `H(L) = H(0) + ∑ F_ℓ(LN(H(ℓ−1)))` dan gradiennya punya suku identitas.

| Faktor per lapisan | 12 lapisan | 96 lapisan |
|---|---|---|
| 0,8 (tanpa jalur murni) | 6,9 × 10⁻² | 5,0 × 10⁻¹⁰ |
| 1,0 (Pre-LN) | 1,000 | 1,000 |

Post-LN (Xiong dkk. 2020) butuh warmup hati-hati agar konvergen; Pre-LN bisa dilatih tanpa warmup pada banyak kasus. Praktik hari ini: hampir semua LLM besar memakai **Pre-LN**.

## Implementasi

- [`src/llmbepees/np/layernorm.py`](../../src/llmbepees/np/layernorm.py): `layernorm` (per token, sepanjang dimensi fitur), `rmsnorm`.
- [`src/llmbepees/np/block.py`](../../src/llmbepees/np/block.py): `pre_ln_block`, `post_ln_block`, `stack_blocks`, dan perhitungan parameter (`block_parameters`, `attention_block_parameters`, `ffn_block_parameters`).
- Uji memverifikasi LayerNorm/RMSNorm contoh buku, keadaan akhir mini-LLM `[−0,8678, −1,0861, 0,6953, 1,2585]`, rumus parameter, dan sebaran ⅔ FFN.

## Inti bab

Satu blok Transformer isinya hanya dua sub-blok, dengan pembagian tugas bersih: **attention** memindahkan informasi antarposisi, **FFN** memproses informasi di dalam setiap posisi. Dua komponen pendamping membuat tumpukan bisa dilatih: **residual** menyediakan jalur tempat nilai (dan gradien) lewat tanpa diubah, **normalisasi** mengendalikan skala representasi pada titik ia dipasang. Kuncinya: masuk `n×d`, keluar `n×d` — justru sifat itulah yang membuat blok bisa ditumpuk sebanyak apa pun.

## Cek pemahaman

1. Mengapa satu blok membutuhkan attention **dan** FFN, bukan salah satunya saja?
2. Apa yang terjadi pada pelatihan model 96 lapisan jika residual connection dihapus?
3. Mengapa ukuran keluaran blok harus sama dengan ukuran masukannya?

## Rujukan

- Buku *"Memahami Matematika di Balik LLM"*, Bab 16.
- Ba dkk. (2016) — LayerNorm; Zhang & Sennrich (2019) — RMSNorm; Xiong dkk. (2020) — analisis Pre-LN vs Post-LN.
