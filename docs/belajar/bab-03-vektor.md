# Bab 3 — Vektor

> Catatan belajar, ditulis ulang dengan bahasa sendiri dari buku *"Memahami Matematika di Balik LLM"* (Bab 3).
> Bagian dari **Fase 1 — Fondasi Matematika**. Kode: [`src/llmbepees/np/vectors.py`](../../src/llmbepees/np/vectors.py) · Uji: [`tests/test_vectors.py`](../../tests/test_vectors.py).

## Masalah

Setiap token di dalam LLM diwakili sebuah vektor. Setiap posisi juga vektor. Keluaran setiap lapisan adalah kumpulan vektor. Jika ada satu objek matematika yang wajib dikuasai sebelum lanjut, inilah objeknya.

## Gagasan

### Definisi

Sebuah vektor berdimensi `n` adalah **deretan terurut** berisi `n` bilangan real:

```
x = [x_1, x_2, …, x_n]
```

Kata "terurut" penting: `[1, 2, 3]` dan `[3, 2, 1]` adalah dua vektor berbeda — posisi angka punya arti. Notasi `x ∈ ℝ⁴` dibaca "x anggota R pangkat 4", artinya vektor berisi 4 bilangan real.

Contoh mini-LLM: token *kucing* diwakili `e_kucing = [1,0, 0,0, 1,0, 0,0] ∈ ℝ⁴`. Di GPT-3 vektor yang sama punya 12.288 angka, tetapi operasinya identik.

**Vektor baris vs kolom.** Vektor yang sama bisa ditulis mendatar (baris) atau menurun (kolom); ini bukan perbedaan konsep, tapi menentukan urutan penulisan saat perkalian matriks (Bab 4). Buku memakai konvensi umum implementasi LLM: vektor adalah **baris**, dan perkalian ditulis `xW` (vektor di kiri).

### Penjumlahan vektor

```
a + b = [a_1 + b_1, a_2 + b_2, …, a_n + b_n]
```

Vektor berdimensi berbeda **tidak bisa** dijumlahkan. Contoh nyata (dipakai Bab 15, informasi posisi):

```
[1,0, 0,0, 1,0, 0,0]   (embedding "kucing")
+ [0,0, 0,5, 0,0, 0,0] (embedding posisi 1)
= [1,0, 0,5, 1,0, 0,0]
```

Vektor hasil ini yang masuk ke lapisan Transformer pertama.

### Perkalian dengan skalar

```
c · x = [c · x_1, c · x_2, …, c · x_n]
```

Contoh: `0,5 × [2,0, −4,0, 1,0, 6,0] = [1,0, −2,0, 0,5, 3,0]`. Operasi ini muncul di dua tempat penting: pembagian dengan `√d_k` dalam attention (Bab 13) dan pembagian logit dengan *temperature* saat sampling (Bab 19) — keduanya perkalian skalar.

### Dot product: operasi terpenting

```
a · b = ∑_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + … + a_n b_n
```

Masukan dua vektor, keluaran **satu skalar** (bukan vektor). Contoh `a = [2, 3]`, `b = [4, −1]`:

| i | a_i | b_i | a_i b_i |
|---|---|---|---|
| 1 | 2 | 4 | 8 |
| 2 | 3 | −1 | −3 |
| | | | **5** |

Contoh dari mini-LLM (dipakai Bab 13): `q = [2,0, 1,0, 1,0, 2,5]`, `k = [1,5, 1,0, 1,5, 1,0]`:

```
q · k = 3,0 + 1,0 + 1,5 + 2,5 = 8,0
```

Angka `8,0` ini adalah skor mentah seberapa besar token *atas* memperhatikan *duduk*. Itulah seluruh mekanisme attention pada level paling dasar: sebuah dot product.

Sifat dot product: (1) simetris `a·b = b·a`; (2) distributif `a·(b+c) = a·b + a·c`; (3) elemen yang sama-sama besar bertanda sama → hasil besar; (4) tanda berlawanan → hasil bisa negatif.

### Norma: panjang vektor

```
‖x‖ = √(∑_{i=1}^{n} x_i²) = √(x_1² + x_2² + … + x_n²)
```

Hasilnya selalu ≥ 0 (tanda hilang karena dikuadratkan). Contoh: `‖[2, 3]‖ = √13 = 3,6056`; `‖[4, −1]‖ = √17 = 4,1231`. Hubungan dengan dot product: `‖x‖² = x · x`.

Gagasan mengukur besaran inilah yang dipakai **RMSNorm** dan **LayerNorm** (Bab 16) untuk menjaga skala angka tetap terkendali — meski rumus keduanya bukan norma L2 persis.

### Cosine similarity: kemiripan arah

Dot product menyesatkan: nilainya membesar hanya karena vektornya panjang, bukan karena arahnya mirip. Solusinya bagi dengan panjang kedua vektor:

```
cos(θ) = (a · b) / (‖a‖ ‖b‖)
```

Nilainya selalu di rentang `[−1, +1]`: `+1` arah persis sama; `0` tegak lurus (tidak searah, bukan bukti tak ada hubungan); `−1` arah berlawanan.

Contoh `a = [2, 3]`, `b = [4, −1]`: `cos(θ) = 5 / (3,6056 × 4,1231) = 5/14,8661 = 0,3363` (sudut ≈ 70,35°).

Contoh relevan LLM (angka ilustrasi, bukan hasil model nyata):

| Kata | Vektor |
|---|---|
| raja | `[0,9, 0,1, 0,4]` |
| ratu | `[0,8, 0,2, 0,5]` |
| traktor | `[−0,7, 0,6, −0,2]` |

`cos(raja, ratu) = 0,94 / (0,9899 × 0,9644) = 0,9846`; `cos(raja, traktor) = −0,65 / 0,9339 = −0,6960`. Inilah yang dimaksud "embedding menangkap makna": kata dalam konteks serupa berakhir dengan cosine similarity tinggi (Bab 8).

### Dimensi tinggi dan mengapa penting

`d_model` di LLM bukan 3, melainkan ribuan: Transformer asli 512, GPT-2 small 768, Llama 2 7B 4096, GPT-3 12288. Kenapa tinggi? Karena jumlah "arah" yang bisa dibedakan tumbuh cepat: di `ℝ²` hanya ada 2 vektor saling tegak lurus, tetapi di `ℝ⁷⁶⁸` ada 768 — dan jauh lebih banyak yang *hampir* tegak lurus.

Perilaku mengejutkan di dimensi tinggi: dua vektor acak (komponen normal) hampir selalu punya cosine similarity mendekati 0, makin dekat ke 0 seiring bertambahnya `n`. Konsekuensinya banyak "ruang kosong" untuk menempatkan konsep berbeda tanpa bertabrakan.

Ini juga sumber masalah teknis Bab 13: jika `q` dan `k` berdimensi `d_k` dengan komponen normal standar, maka `Var(q · k) = d_k` (diperiksa numerik: `d_k=4→4,00`; `64→64,49`; `512→508,16`). Makin besar dimensi, makin liar sebaran dot product-nya — itulah kenapa hasilnya dibagi `√d_k`.

### Ringkasan operasi vektor di LLM

| Operasi | Rumus | Masukan → Keluaran | Muncul di |
|---|---|---|---|
| Penjumlahan | `a + b` | 2 vektor → 1 vektor | Residual (16), posisi (15) |
| Kali skalar | `c x` | skalar + vektor → vektor | Scaling (13), temperature (19) |
| Dot product | `a · b` | 2 vektor → 1 skalar | Skor attention (13), logit (17) |
| Norma | `‖x‖` | 1 vektor → 1 skalar | RMSNorm (16) |
| Cosine | `(a·b)/(‖a‖‖b‖)` | 2 vektor → 1 skalar | Analisis embedding (8) |
| Elemen-wise | `a ⊙ b` | 2 vektor → 1 vektor | Gating di SwiGLU (16) |

Operasi terakhir `⊙` disebut **Hadamard product**: `a ⊙ b = [a_1 b_1, a_2 b_2, …]`. Hasilnya vektor, bukan skalar. Contoh: `[2, 3] ⊙ [4, −1] = [8, −3]`, sedangkan `[2, 3] · [4, −1] = 5`.

## Implementasi

- [`src/llmbepees/np/vectors.py`](../../src/llmbepees/np/vectors.py): `add`, `scale`, `dot`, `norm_l2`, `cosine_similarity`, `hadamard` (plus `as_vector`).
- [`tests/test_vectors.py`](../../tests/test_vectors.py): memverifikasi semua contoh angka di atas (toleransi desimal ke-3).

## Inti bab

Semua yang ada di dalam LLM berbentuk vektor, dan dari sekian operasinya satu jauh lebih penting: **dot product** menerima dua vektor dan mengembalikan satu angka "kecocokan". Dua operasi pendamping melengkapinya: **norma** (panjang) dan **cosine similarity** (dot product yang sudah dibagi panjang, sehingga tersisa kemiripan arah).

## Cek pemahaman

1. Mengapa dot product bisa dipakai sebagai ukuran kecocokan?
2. Kapan dot product menyesatkan, dan mengapa cosine similarity memperbaikinya?
3. Untuk apa norma dipakai di dalam LLM?

## Rujukan

- Buku *"Memahami Matematika di Balik LLM"*, Bab 3.
- Vaswani dkk. (2017); Radford dkk. (2019); Touvron dkk. (2023b); Brown dkk. (2020) — ukuran `d_model`.
