# Bab 4 — Matriks dan Perkalian Matriks

> Catatan belajar, ditulis ulang dengan bahasa sendiri dari buku *"Memahami Matematika di Balik LLM"* (Bab 4).
> Bagian dari **Fase 1 — Fondasi Matematika**. Kode: [`src/llmbepees/np/matrices.py`](../../src/llmbepees/np/matrices.py) · Uji: [`tests/test_matrices.py`](../../tests/test_matrices.py).

## Masalah

Lebih dari 99% operasi aritmetika sebuah LLM adalah satu operasi yang sama: **perkalian matriks**. Jika paham bab ini, kita paham apa yang sebenarnya dikerjakan GPU berjam-jam saat model dilatih.

> Bedakan sejak awal: "99%" itu jumlah **operasi hitung**, bukan waktu. Saat model menghasilkan teks satu token demi satu token, penghambatnya adalah kecepatan membaca bobot dari memori, bukan kecepatan mengalikannya (Bab 19).

## Gagasan

### Definisi

Matriks berukuran `m × n` adalah tabel bilangan real dengan `m` baris dan `n` kolom. `A_ij` = elemen baris ke-`i`, kolom ke-`j` — **baris dulu, baru kolom**. Notasi `A ∈ ℝ^{m×n}`.

Contoh mini-LLM: kalimat `<bos> kucing duduk di atas` punya 5 token, tiap token vektor `ℝ⁴`, digabung menjadi matriks `5 × 4` (**X**). Konvensi seluruh buku: **satu baris = satu token, satu kolom = satu dimensi fitur**. Jadi `X_23 = 1,0` berarti token kedua (*kucing*), dimensi ketiga.

### Transpos

```
(Aᵀ)_ij = A_ji
```

Baris dan kolom ditukar; jika **A** `m × n` maka **Aᵀ** `n × m`. Contoh:

```
A = [1 2]        Aᵀ = [1 3 5]
    [3 4]   →         [2 4 6]
    [5 6]
```

Transpos muncul mencolok di rumus attention `QKᵀ` (alasannya di bagian berikutnya).

### Penjumlahan matriks

Elemen per elemen; ukuran harus **identik**: `(A + B)_ij = A_ij + B_ij`. Contoh nyata dari Bab 15 (menambah embedding posisi **P** ke embedding token **X**):

```
X + P = H⁽⁰⁾
```

**H⁽⁰⁾** inilah masukan resmi ke blok Transformer pertama, dan akan dipakai berulang sampai Bab 17.

### Perkalian matriks

Jika **A** berukuran `m × k` dan **B** berukuran `k × n`, maka `C = AB` berukuran `m × n` dengan:

```
C_ij = ∑_{p=1}^{k} A_ip B_pj
```

Dalam kalimat: elemen baris `i` kolom `j` dari hasil adalah **dot product** antara baris ke-`i` dari **A** dan kolom ke-`j` dari **B**.

**Aturan ukuran:** dimensi tengah harus cocok; ukuran hasil diambil dari dimensi luar:

```
A (m × k) · B (k × n) = C (m × n)
        └──── sama ────┘
```

Jika tidak cocok, perkalian tidak terdefinisi — sumber 90% error dalam implementasi deep learning.

Contoh penuh `A(3×2) · B(2×3)`:

```
      [1 2]         [ 7  8  9]        [ 27  30  33]
A =   [3 4]   B =   [10 11 12]  → C = [ 61  68  75]
      [5 6]                            [ 95 106 117]
```

Misalnya `C_21` = baris 2 dari **A** `[3, 4]` · kolom 1 dari **B** `[7, 10]` = `3(7) + 4(10) = 61`.

Sifat: (1) **tidak komutatif** `AB ≠ BA` (bahkan `BA` sering tidak terdefinisi); (2) asosiatif; (3) distributif; (4) `(AB)ᵀ = BᵀAᵀ` — urutan terbalik.

### Kenapa QKᵀ?

Misalkan **Q** dan **K** masing-masing `n × d_k` (satu baris per token). Belum tentu **Q** dan **K** punya baris yang sama. Tetapi tujuan kita adalah menghitung dot product antara **setiap baris Q** dengan **setiap baris K**, semua pasangan. Jika langsung `QK` (tanpa transpos), dimensi tengahnya `d_k ≠ n` — tidak terdefinisi. Dengan **Kᵀ** berukuran `d_k × n`:

```
QKᵀ : (n × d_k) · (d_k × n) = (n × n) ✓
```

dan elemennya:

```
(QKᵀ)_ij = ∑_{p=1}^{d_k} Q_ip K_jp = q_i · k_j
```

Persis yang kita mau. Transpos di rumus attention **bukan hiasan** — itu satu-satunya cara membuat ukuran cocok sambil menghasilkan semua pasangan dot product sekaligus.

### Vektor dikali matriks: transformasi linear

Kasus paling sering: satu vektor baris **x** `1 × d` dikali matriks **W** `d × d'`:

```
y = xW,   y_j = ∑_{i=1}^{d} x_i W_ij
```

Hasilnya vektor `1 × d'`. Contoh mini-LLM: vektor token *atas* setelah ditambah posisi `h = [0,5, 0,5, 1,0, 1,0]`, dan matriks bobot query:

```
W_Q = [1 1 1 1]
      [1 1 1 0]
      [0 0 0 1]
      [1 0 0 1]
```

Menghitung `q = hW_Q` kolom demi kolom:

| Kolom | Kolom ke-j dari W_Q | Perhitungan | q_j |
|---|---|---|---|
| 1 | `[1,1,0,1]` | `0,5(1)+0,5(1)+1,0(0)+1,0(1)` | 2,0 |
| 2 | `[1,1,0,0]` | `0,5(1)+0,5(1)+0+0` | 1,0 |
| 3 | `[1,1,0,0]` | `0,5(1)+0,5(1)+0+0` | 1,0 |
| 4 | `[1,0,1,1]` | `0,5(1)+0,5(0)+1,0(1)+1,0(1)` | 2,5 |

`q_atas = [2,0, 1,0, 1,0, 2,5]` — angka yang sama dipakai di Bab 3 saat menghitung dot product. Sekarang jelas asalnya.

Operasi `x ↦ xW` disebut **transformasi linear** karena `(x₁+x₂)W = x₁W + x₂W` dan `(cx)W = c(xW)`. Konsekuensinya besar: menumpuk dua transformasi linear **tidak menghasilkan apa pun yang baru**:

```
z = (xW₁)W₂ = x(W₁W₂) = xW₃
```

Dua lapisan setara satu lapisan; seratus lapisan yang semuanya linear juga setara satu lapisan. Inilah alasan matematis jaringan saraf **harus** punya fungsi non-linear di antara lapisannya (Bab 9).

### Matriks batch: banyak token sekaligus

Untuk `n` token, kita tidak menghitung `hW` satu per satu, melainkan menumpuk semua `h` menjadi matriks **H** `n × d`, lalu:

```
Q = HW_Q   ;   (n × d) · (d × d_k) = (n × d_k)
```

Satu operasi menghasilkan query untuk semua token. Dengan **H⁽⁰⁾** dari sebelumnya dan **W_Q**:

```
Q = H⁽⁰⁾ W_Q = [0,5 0,5 0,5 0,5]   ← <bos>
               [1,5 1,5 1,5 2,0]   ← kucing
               [2,0 1,0 1,0 1,5]   ← duduk
               [2,5 2,0 2,0 1,5]   ← di
               [2,0 1,0 1,0 2,5]   ← atas
```

Baris terakhir cocok dengan hitungan manual `q_atas`. Kemampuan memproses semua token dalam satu perkalian matriks inilah keunggulan Transformer atas RNN yang harus memproses token satu per satu secara berurutan.

### Biaya komputasi

FLOP = satu operasi aritmetika pada bilangan desimal. Mengalikan `m × k` dengan `k × n` menghasilkan `m × n` elemen, tiap elemen butuh `k` perkalian + `k−1` penjumlahan:

```
FLOP ≈ 2 · m · n · k
```

Contoh: satu proyeksi query GPT-3 untuk konteks 2048 token, `m = 2048`, `k = n = 12288`:

```
2 × 2048 × 12288 × 12288 = 618.475.290.624 ≈ 6,2 × 10^11 FLOP
```

Itu baru **satu** dari empat proyeksi (`W_Q, W_K, W_V, W_O`), di **satu** dari 96 lapisan, sekali forward pass. Bagian attention berskala `n²` terhadap panjang konteks:

| n | n² |
|---|---|
| 512 | 262.144 |
| 2.048 | 4.194.304 |
| 8.192 | 67.108.864 |
| 32.768 | 1.073.741.824 |

Melipatgandakan konteks 4× membuat biaya bagian ini naik 16× — alasan konteks panjang mahal (Bab 19).

## Implementasi

- [`src/llmbepees/np/matrices.py`](../../src/llmbepees/np/matrices.py): `transpose`, `matadd`, `matmul`, `matvec`, `identity`, `flops_matmul` (+ `as_matrix`).
- [`tests/test_matrices.py`](../../tests/test_matrices.py): memverifikasi semua contoh angka di atas (toleransi desimal ke-3).

## Inti bab

Matriks bukan objek baru — hanya tumpukan vektor, satu baris per token. Perkalian matriks bukan operasi baru — hanya cara menghitung banyak dot product sekaligus dalam satu perintah yang bisa diparalelkan di GPU. Dan karena dua transformasi linear bertumpuk selalu setara satu transformasi linear, fungsi non-linear di Bab 9 adalah **keharusan**, bukan pilihan gaya.

## Cek pemahaman

1. Mengapa dimensi tengah pada perkalian matriks harus cocok?
2. Mengapa menumpuk seratus lapisan linear tidak menambah kemampuan apa pun?
3. Mengapa hampir seluruh waktu komputasi LLM habis di perkalian matriks?

## Rujukan

- Buku *"Memahami Matematika di Balik LLM"*, Bab 4.
- Vaswani dkk. (2017) — keunggulan paralel Transformer atas RNN.
