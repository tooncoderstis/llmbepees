# Bab 11 — Backpropagation

> Catatan belajar, ditulis ulang dengan bahasa sendiri dari buku *"Memahami Matematika di Balik LLM"* (Bab 11).
> Bagian dari **Fase 5 — Backpropagation**. Kode: [`src/llmbepees/np/backprop.py`](../../src/llmbepees/np/backprop.py) · Uji: [`tests/test_backprop.py`](../../tests/test_backprop.py).

## Masalah

Bab 10 memberi kita gradien di ujung jaringan: `p − y`. Tapi parameter yang harus diperbarui tersebar di **semua** lapisan, termasuk yang paling awal. Bagaimana satu angka loss di ujung bisa menghasilkan gradien untuk 174 miliar parameter yang tersebar di seluruh model?

**Struktur masalahnya** (dikecilkan):

```
x →[W1]→ h1 →[W2]→ h2 →[W3]→ z →[softmax]→ p →[CE]→ L
```

Kita ingin `∂L/∂W1` — parameter yang berada 4 langkah dari `L`. Cara naif: hitung gradien numerik (Bab 7). Untuk `N` parameter butuh `2N` *forward pass*. Untuk GPT-3 itu **350 miliar forward pass untuk satu langkah pelatihan** — jika satu forward pass 0,1 detik, itu 1.100 tahun per langkah. Backpropagation menyelesaikan semuanya dengan biaya satu *backward pass* ≈ **dua kali** forward pass — total, **bukan per parameter**.

## Gagasan

### Prinsipnya: aturan rantai yang disusun ulang

Dari aturan rantai (Bab 7):

```
∂L/∂W1 = (∂L/∂p)·(∂p/∂z)·(∂z/∂h2)·(∂h2/∂h1)·(∂h1/∂W1)
```

Kuncinya **urutan pengalian**. Kalikan dari kiri ke kanan (dari loss ke parameter): setiap hasil antara adalah vektor kecil. Kalikan dari kanan ke kiri: hasil antaranya matriks besar. Karena `L` skalar, `∂L/∂p` hanyalah sebuah vektor, dan semua hasil antara tetap berukuran vektor.

**Definisi.** Besaran `δ^(ℓ) = ∂L/∂z^(ℓ)` disebut **error signal** atau **delta** pada lapisan `ℓ` — inilah yang dirambatkan mundur.

### Contoh angka: jaringan mungil dari awal sampai akhir

```
z1 = w1·x + b1 → h = ReLU(z1) → y = w2·h + b2 → L = (y − t)²
```

| Simbol | Nilai | Keterangan |
|---|---|---|
| `x` | 1,0 | masukan |
| `w1` | 0,5 | parameter |
| `b1` | 0,0 | parameter |
| `w2` | 2,0 | parameter |
| `b2` | 1,0 | parameter |
| `t` | 3,0 | target |

**Forward pass**

| Langkah | Perhitungan | Hasil |
|---|---|---|
| `z1` | 0,5 × 1,0 + 0,0 | 0,5 |
| `h` | max(0, 0,5) | 0,5 |
| `y` | 2,0 × 0,5 + 1,0 | 2,0 |
| `L` | (2,0 − 3,0)² | 1,0 |

**Backward pass** (setiap langkah memakai hasil langkah sebelumnya):

1. `∂L/∂y = 2(y − t) = 2(2,0 − 3,0) = −2,0` — menaikkan `y` akan menurunkan loss.
2. Lapisan kedua `y = w2·h + b2`: `∂L/∂w2 = (−2,0)(0,5) = −1,0`; `∂L/∂b2 = −2,0`.
3. Teruskan mundur ke `h`: `∂L/∂h = (−2,0)(2,0) = −4,0`.
4. Lewati ReLU (`z1 = 0,5 > 0`, turunannya 1): `∂L/∂z1 = −4,0`.
5. Lapisan pertama: `∂L/∂w1 = (−4,0)(1,0) = −4,0`; `∂L/∂b1 = −4,0`.

**Pembaruan parameter** dengan `η = 0,1`:

| Parameter | Lama | Gradien | Baru |
|---|---|---|---|
| `w1` | 0,5 | −4,0 | 0,5 − 0,1(−4,0) = **0,9** |
| `b1` | 0,0 | −4,0 | **0,4** |
| `w2` | 2,0 | −1,0 | **2,1** |
| `b2` | 1,0 | −2,0 | **1,2** |

**Verifikasi.** Forward pass ulang: `z1 = 0,9(1,0) + 0,4 = 1,3`, `h = 1,3`, `y = 2,1(1,3) + 1,2 = 3,93`, `L = (3,93 − 3,0)² = 0,8649`. Loss turun dari `1,0000` ke `0,8649` — backpropagation bekerja. (`y` sempat melewati target 3,0 karena `η = 0,1` relatif besar; iterasi berikutnya menariknya kembali.)

### Aturan mundur untuk setiap jenis operasi

Notasi: `gout` = gradien yang datang dari atas, `gin` = gradien yang diteruskan ke bawah.

| Operasi maju | Gradien ke masukan | Gradien ke parameter |
|---|---|---|
| `y = x + b` | `gout` | `gout` |
| `y = xW` | `gout Wᵀ` | `xᵀ gout` |
| `y = ReLU(x)` | `gout ⊙ 1[x > 0]` | tidak ada |
| `y = a ⊙ b` | `gout ⊙ b` (ke `a`) | tidak ada |
| `y = x1 + x2` (residual) | `gout` ke keduanya | tidak ada |
| `L = CE(softmax(z))` | `p − y` | tidak ada |

**Contoh baris `xW`.** `x = [2, 3]`, `W` 2×2, `gout = [0,5, −1,0]`:

```
∂L/∂W = xᵀ gout = [2 3]ᵀ [0,5  −1,0] = [1,0  −2,0]
                                        [1,5  −3,0]
```

Jika `W = I`, gradien ke masukan `∂L/∂x = gout Wᵀ = [0,5, −1,0]`.

### Residual connection: jalur langsung untuk gradien

**Definisi.** Residual connection (skip connection) adalah struktur `y = x + F(x)` dengan `F` suatu sub-jaringan. Dipakai di setiap blok Transformer. Gradiennya:

```
∂L/∂x = gout · (1 + ∂F/∂x) = gout + gout·(∂F/∂x)
```

Suku pertama adalah gradien yang lewat **tanpa dikalikan apa pun**.

**Kenapa ini menentukan.** Tanpa residual, gradien di lapisan ke-`ℓ` dari bawah adalah hasil kali `ℓ` faktor. Jika tiap faktor 0,8 dan ada 96 lapisan (seperti GPT-3), `0,8⁹⁶ ≈ 5 × 10⁻¹⁰` — gradien di lapisan pertama praktis nol, lapisan itu tidak pernah belajar. Jika tiap faktor 1,2, `1,2⁹⁶ ≈ 4 × 10⁷` — gradien meledak dan pelatihan menjadi `NaN`.

| Faktor per lapisan | 12 lapisan | 48 lapisan | 96 lapisan |
|---|---|---|---|
| 0,80 | 6,9 × 10⁻² | 2,2 × 10⁻⁵ | 5,0 × 10⁻¹⁰ |
| 0,95 | 0,540 | 0,085 | 0,0073 |
| **1,00 (residual)** | **1,000** | **1,000** | **1,000** |
| 1,05 | 1,80 | 10,4 | 108,2 |
| 1,20 | 8,9 | 6.320 | 4,0 × 10⁷ |

Dengan residual, bentuk matriksnya `∇_x L = (I + J_F(x))ᵀ ∇_y L`, dan suku `I` itulah jalur yang meneruskan gradien tanpa dikalikan apa pun.

**Yang dijamin dan yang tidak.** Yang **dijamin**: selalu ada kontribusi identitas yang meneruskan gradien dari lapisan terakhir ke lapisan pertama tanpa dikalikan apa pun — gradien tidak meluruh semata-mata karena kedalaman. Yang **tidak dijamin**: gradien totalnya persis 1. Suku `J_F` tetap bisa memperbesar, mengecilkan, atau meniadakan sebagian gradien; pelatihan model dalam tetap butuh normalisasi dan jadwal learning rate yang hati-hati. Residual bukan jaminan matematis, melainkan perubahan struktur yang membuat kegagalan itu tidak lagi kelaziman.

### Backpropagation untuk banyak token sekaligus

Dalam LLM, satu forward pass memproses matriks `n × d`, bukan satu vektor. Aturannya sama, hanya ukurannya berubah:

```
Maju:  Y = XW     (n×d)·(d×m) = n×m
Mundur ke W:  Xᵀ G  (d×n)·(n×m) = d×m ✓
```

Perhatikan: ukuran gradien `W` tetap `d × m`, sama dengan `W` sendiri, **berapapun `n`**. Artinya kontribusi dari semua `n` token **dijumlahkan** ke satu gradien — konsisten dengan aturan percabangan Bab 7 (satu parameter dipakai di banyak tempat → gradiennya dijumlahkan). Praktik standar membaginya dengan jumlah token agar skala tidak bergantung ukuran batch; itu sebabnya loss selalu dilaporkan sebagai rata-rata per token.

### Biaya komputasi backpropagation

Aturan praktis yang dipakai luas: **backward pass ≈ 2× forward pass**, sehingga satu langkah pelatihan penuh (maju + mundur) ≈ **3× satu forward pass**. Untuk setiap perkalian matriks `Y = XW` arah maju, arah mundur butuh dua perkalian matriks berukuran serupa (`XᵀG` dan `GWᵀ`). Ini menghasilkan rumus biaya pelatihan yang dipakai di literatur scaling law:

```
C ≈ 6·N·D        (C = total FLOP, N = jumlah parameter, D = jumlah token)
```

Angka 6 berasal dari 2 FLOP per parameter per token untuk forward, dikali 3 (1 maju + 2 mundur). Melatih model 7 miliar parameter pada 2 triliun token: `C ≈ 6 × 7×10⁹ × 2×10¹² = 8,4 × 10²²` FLOP. Jika satu GPU efektif `1,5 × 10¹⁴` FLOP/detik, itu `5,6 × 10⁸` detik = **17,7 tahun-GPU** → dengan 1000 GPU ≈ **6,5 hari**. Melatih LLM adalah proyek infrastruktur, bukan laptop.

### Kebutuhan memori

Backpropagation mensyaratkan: **semua nilai antara dari forward pass harus disimpan** sampai backward pass selesai. Misalnya `∂L/∂W = XᵀG` butuh `X` dari saat forward. Untuk satu urutan panjang `n`, `L` lapisan, dimensi `d`:

```
aktivasi ≈ k·L·n·d        (k ≈ 10–20 untuk Transformer standar)
```

Model 7B (`L = 32`, `d = 4096`), `n = 2048`, `k = 12`, presisi 16-bit: `12 × 32 × 2048 × 4096 × 2 byte = 6,4 GB` **per urutan**. Dengan batch 8 → **51 GB**, lebih besar dari memori satu GPU kelas atas. Inilah kenapa teknik seperti *gradient checkpointing* (menyimpan lebih sedikit, menghitung ulang saat backward) diperlukan.

## Implementasi

- [`src/llmbepees/np/backprop.py`](../../src/llmbepees/np/backprop.py): aturan mundur (`add_backward`, `relu_backward`, `hadamard_backward`, `matmul_backward`, `residual_backward`, `residual_gradient`), contoh jaringan mungil (`forward_mlp`, `two_layer_mlp_backward`), serta biaya (`training_flops`, `training_time_seconds`, `activation_memory`, `factor_power`).
- `matmul_backward` menerima `x` vektor maupun batch dan menjumlahkan kontribusi tiap token ke satu `grad_W` — sesuai Bab 11.6.
- Uji memverifikasi angka 11.3 (forward, backward, pembaruan parameter), tabel aturan 11.4, tabel residual 11.5, `residual_gradient` lewat `numerical_gradient`, dan rumus biaya 11.7–11.8.

## Inti bab

Bab 10 memberi kita gradien di satu titik (ujung jaringan); backpropagation adalah cara membawa sinyal perbaikan itu dari ujung ke **seluruh** model. Prinsipnya cuma aturan rantai Bab 7 yang dijalankan mundur dan disusun ulang supaya efisien. Yang membuatnya layak dipakai adalah biayanya: menghitung seluruh gradien untuk miliaran parameter hanya memakan sekitar dua kali satu forward pass — berbeda jauh dari cara naif yang butuh dua kali jumlah parameter. Selisih inilah yang membuat deep learning mungkin secara praktis.

Bab ini juga menjelaskan satu komponen yang kelihatannya sepele tapi menentukan segalanya: **residual connection** (`y = x + F(x)`). Karena gradien merambat dengan cara dikalikan, faktor sedikit di bawah satu meluruh menjadi nol setelah puluhan lapisan, dan sedikit di atas satu meledak. Residual menyediakan jalur di mana gradien lewat tanpa dikalikan apa pun — tanpa itu, model dengan sembilan puluh enam lapisan tidak akan bisa dilatih.

## Cek pemahaman

1. Mengapa gradien dihitung mundur, bukan maju?
2. Mengapa gradien bisa menghilang pada model yang sangat dalam?
3. Mengapa residual connection memperbaiki masalah itu?

## Rujukan

- Buku *"Memahami Matematika di Balik LLM"*, Bab 11.
- He dkk. (2016) — residual connection; Kaplan dkk. (2020) — rumus biaya pelatihan `C ≈ 6ND`.
