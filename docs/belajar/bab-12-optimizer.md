# Bab 12 — Optimizer: SGD, Momentum, dan Adam

> Catatan belajar, ditulis ulang dengan bahasa sendiri dari buku *"Memahami Matematika di Balik LLM"* (Bab 12).
> Bagian dari **Fase 6 — Optimizer**. Kode: [`src/llmbepees/np/optimizer.py`](../../src/llmbepees/np/optimizer.py) · Uji: [`tests/test_optimizer.py`](../../tests/test_optimizer.py).

## Masalah

Gradien memberi tahu **arah**. Optimizer memutuskan **seberapa jauh melangkah**. Kedengarannya sepele, tetapi keputusan inilah yang menentukan apakah pelatihan selesai dalam seminggu atau tidak pernah konvergen sama sekali. Di mini-LLM Bab 21, gradien pada embedding bernilai `0,3420` sementara gradien pada `W_Q` bernilai `0,0055` — beda **60×**. Dengan satu learning rate untuk semuanya, tidak ada nilai yang pas.

## Gagasan

### Gradient descent

Aturan pembaruan paling dasar, dengan `η` = learning rate (sudah muncul di Bab 7):

```
θ_{t+1} = θ_t − η·∇L(θ_t)
```

Contoh `f(w) = (w − 3)²` dengan `η = 0,1`:

| t | w | f(w) | ∇ |
|---|---|---|---|
| 0 | 0,0000 | 9,0000 | −6,0000 |
| 1 | 0,6000 | 5,7600 | −4,8000 |
| 2 | 1,0800 | 3,6864 | −3,8400 |
| 3 | 1,4640 | 2,3593 | −3,0720 |
| 4 | 1,7712 | 1,5099 | −2,4576 |
| 5 | 2,0170 | 0,9664 | −1,9661 |

Polanya: langkah makin lama makin kecil, karena gradien makin kecil saat mendekati minimum. Ini melambat justru saat kita ingin cepat sampai.

### Pengaruh learning rate

| η | Perilaku | Hasil |
|---|---|---|
| Terlalu kecil (0,001) | Bergerak, tapi sangat lambat | Butuh jutaan iterasi |
| Pas (0,1) | Konvergen mulus | ✓ |
| Terlalu besar (1,1) | Melompati minimum, makin jauh | Loss meledak, `NaN` |

Untuk `η = 1,1` pada `f(w) = (w − 3)²`:

| t | w | f(w) |
|---|---|---|
| 0 | 0,0000 | 9,0000 |
| 1 | 6,6000 | 12,9600 |
| 2 | −1,3200 | 18,6624 |
| 3 | 8,1840 | 26,8739 |
| 4 | −3,2208 | 38,6984 |

Batas teoretis untuk fungsi kuadrat `f(w) = a(w − c)²` adalah `η < 1/a` (gradiennya `2a(w − c)`, jadi faktornya `1 − 2aη`). Di sini `a = 1`, jadi `η` harus `< 1`; nilai `1,1` melewatinya dan hasilnya persis seperti prediksi teori. Untuk jaringan nyata tidak ada rumus batas seperti itu — `η` ditentukan secara empiris.

### Stochastic gradient descent (SGD)

Menghitung gradien pada seluruh dataset (miliaran token) untuk **satu** langkah tidak mungkin. Solusinya: hitung pada sebagian kecil data.

```
∇_θ L ≈ (1/B) ∑_{i=1}^{B} ∇_θ L_i
```

| Jenis | B | Sifat |
|---|---|---|
| Full-batch | seluruh data | Gradien tepat, tapi tidak praktis |
| Mini-batch | 32–jutaan token | Standar |
| Stochastic murni | 1 | Sangat berisik |

Jika contoh diambil acak seragam dan saling bebas, gradien mini-batch adalah **penduga tak bias** dari gradien penuh: rata-ratanya benar, tetapi tiap sampel punya derau, dan standar deviasi deraunya berkurang sebanding `1/√B`. Contoh: derau `0,5` pada `B = 100` menjadi `0,5 × √(100/10.000) = 0,05` pada `B = 10.000` — batch 100× lebih besar hanya mengurangi derau 10× (hukum imbal-hasil menurun). GPT-3 memakai sekitar 3,2 juta token per batch pada tahap akhir.

### Momentum

Masalah SGD: pada lembah sempit yang panjang, gradien menunjuk ke dinding lembah, bukan ke arah panjangnya → gerakan zigzag.

```
v_t = β·v_{t−1} + ∇L(θ_t)
θ_{t+1} = θ_t − η·v_t         (β ≈ 0,9)
```

`v` adalah rata-rata bergerak gradien sebelumnya: komponen yang arahnya konsisten menumpuk, yang berganti tanda saling meniadakan.

Contoh gradien 4 langkah. Parameter **berayun** `[1,0, −0,8, 0,9, −0,7]`:

| t | grad | v_t |
|---|---|---|
| 1 | 1,00 | 1,000 |
| 2 | −0,80 | 0,100 |
| 3 | 0,90 | 0,990 |
| 4 | −0,70 | 0,191 |

Parameter **konsisten** `[0,3, 0,3, 0,3, 0,3]`:

| t | grad | v_t |
|---|---|---|
| 1 | 0,30 | 0,300 |
| 2 | 0,30 | 0,570 |
| 3 | 0,30 | 0,813 |
| 4 | 0,30 | 1,032 |

Parameter yang gradiennya konsisten berakhir dengan `v` **5× lebih besar** (1,032 vs 0,191), padahal gradien mentahnya lebih kecil (0,3 vs 0,7–1,0). Nilai batas untuk gradien konstan `g` adalah `v∞ = g/(1 − β) = 10g` — momentum dengan `β = 0,9` mempercepat hingga 10×.

### Adam

Optimizer yang dipakai hampir semua LLM adalah **Adam** (Kingma & Ba, 2015), atau variannya **AdamW**. Ide tambahannya di luar momentum: **setiap parameter mendapat learning rate sendiri**, disesuaikan dengan seberapa besar gradiennya biasanya.

```
m_t = β1·m_{t−1} + (1 − β1)·g_t          (momen pertama: rata-rata)
v_t = β2·v_{t−1} + (1 − β2)·g_t²         (momen kedua mentah: rata-rata kuadrat)
m̂_t = m_t / (1 − β1^t)                    (koreksi bias)
v̂_t = v_t / (1 − β2^t)
θ_{t+1} = θ_t − η · m̂_t / (√v̂_t + ε)
```

`g_t²` = kuadrat **elemen-per-elemen**, bukan dot product. Nilai bawaan `β1 = 0,9`, `β2 = 0,999`, `ε = 10⁻⁸`.

Membacanya dengan kalimat biasa: baris 1 mengingat ke arah mana gradien biasanya menunjuk; baris 2 mengingat seberapa besar; baris 3 memperbaiki kedua ingatan itu (yang pada langkah awal masih terlalu kecil karena mulai dari nol); baris 4 mengambil langkah ke arah ingatan pertama, dengan besar langkah dibagi akar ingatan kedua.

**Kenapa ada koreksi bias.** `m_0` diinisialisasi nol. Pada langkah pertama `m_1 = 0,1·g_1` — hanya 10% dari gradien sebenarnya, terlalu kecil. Membagi dengan `(1 − β1^t)` memperbaikinya: `m̂_1 = 0,1g_1 / 0,1 = g_1` — tepat. Pengaruh koreksi mengecil seiring `t` membesar (`1 − 0,9^100 = 0,99997`).

**Contoh lengkap.** Minimalkan `f(w) = (w − 3)²` dari `w = 0`, `η = 0,1`:

| t | g_t | m_t | v_t | m̂_t | v̂_t | m̂/(√v̂+ε) | w_{t+1} |
|---|---|---|---|---|---|---|---|
| 1 | −6,0000 | −0,6000 | 0,0360 | −6,0000 | 36,0000 | −1,0000 | 0,1000 |
| 2 | −5,8000 | −1,1200 | 0,0696 | −5,8947 | 34,8194 | −0,9991 | 0,1999 |
| 3 | −5,6002 | −1,5680 | 0,1009 | −5,7861 | 33,6659 | −0,9973 | 0,2996 |
| 4 | −5,4008 | −1,9513 | 0,1300 | −5,6740 | 32,5398 | −0,9946 | 0,3991 |
| 5 | −5,2018 | −2,2763 | 0,1569 | −5,5587 | 31,4414 | −0,9913 | 0,4982 |

Kolom kedua dari kanan selalu ≈ `−1`. Karena itu tiap langkah menggeser `w` sekitar `+0,1` — persis sebesar `η`, tidak peduli gradiennya `−6,0` atau `−5,2`. Itulah sifat terpenting Adam: **ukuran langkah kira-kira `η`**, terlepas dari besar gradien. "Kira-kira" penting: kesetaraan itu persis hanya jika arah gradien konsisten dan `ε` dapat diabaikan; saat gradien berayun, `m̂` mengecil sementara `√v̂` tidak. Karena pembilang dan penyebut sama-sama berskala dengan gradien, keduanya saling meniadakan: `m̂/√v̂ ≈ g/|g| = ±1`.

**Perbandingan SGD vs Adam** (masalah sama, `η = 0,1`):

| t | SGD: w | Adam: w |
|---|---|---|
| 1 | 0,6000 | 0,1000 |
| 2 | 1,0800 | 0,1999 |
| 3 | 1,4640 | 0,2996 |
| 4 | 1,7712 | 0,3991 |
| 5 | 2,0170 | 0,4982 |

Di masalah sesederhana ini SGD justru lebih cepat karena gradien awalnya besar. Keunggulan Adam muncul ketika skala gradien antarparameter sangat berbeda — kondisi di jaringan nyata.

**Biaya memori.** Adam menyimpan `m` dan `v` untuk tiap parameter. Ditambah parameter dan gradiennya:

| Yang disimpan | Ukuran (fp32) untuk `N` parameter |
|---|---|
| Parameter | `4N` byte |
| Gradien | `4N` byte |
| `m` | `4N` byte |
| `v` | `4N` byte |
| **Total** | **`16N` byte** |

Model 7 miliar parameter: `16 × 7×10⁹ = 112 GB` — lebih besar dari kapasitas satu H100 (80 GB). Padahal untuk *menjalankan* model yang sama (inference) hanya butuh `2N = 14 GB` dalam presisi 16-bit. Inilah kenapa melatih model 7B saja butuh beberapa GPU sekaligus.

### AdamW dan weight decay

**Weight decay** adalah regularisasi: dorong parameter mendekati nol sedikit demi sedikit. Bias dan parameter normalisasi biasanya dikecualikan.

Versi lama menambahkannya ke dalam loss (`L' = L + (λ/2)‖θ‖²`, gradiennya bertambah `λθ`). Masalahnya, di Adam suku itu ikut dibagi `√v̂`, sehingga kekuatan peluruhannya berbeda-beda tiap parameter. AdamW memisahkannya:

```
θ_{t+1} = θ_t − η·( m̂_t/(√v̂_t + ε) + λ·θ_t )
```

Nilai umum `λ = 0,1` (dipakai GPT-3).

### Learning rate schedule

Learning rate tidak dibuat konstan. Skema standar LLM punya dua fase:

**Fase 1 — warmup (linear naik)**, selama `T_w` langkah pertama:

```
η_t = η_max · t / T_w
```

**Fase 2 — cosine decay**, setelah warmup:

```
η_t = η_min + ½(η_max − η_min)(1 + cos(π·(t − T_w)/(T − T_w)))
```

Contoh `η_max = 6×10⁻⁴`, `η_min = 6×10⁻⁵`, `T_w = 2000`, `T = 100.000`:

| Langkah t | Fase | η_t |
|---|---|---|
| 0 | warmup | 0 |
| 500 | warmup | 1,5 × 10⁻⁴ |
| 1.000 | warmup | 3,0 × 10⁻⁴ |
| 2.000 | akhir warmup | 6,0 × 10⁻⁴ |
| 26.500 | cosine | 5,2 × 10⁻⁴ |
| 51.000 | cosine (tengah) | 3,3 × 10⁻⁴ |
| 75.500 | cosine | 1,4 × 10⁻⁴ |
| 100.000 | akhir | 6,0 × 10⁻⁵ |

**Kenapa warmup.** Pada langkah awal, estimasi `√v̂` di Adam masih sangat kasar (baru berbasis satu-dua sampel). Langkah penuh di fase itu bisa membawa parameter ke wilayah buruk yang sulit dipulihkan. Warmup membuat langkah awal kecil sampai statistik gradien stabil.

**Kenapa decay.** Di akhir pelatihan, parameter sudah dekat nilai baiknya. Langkah besar hanya membuatnya berayun di sekitar minimum tanpa perbaikan tetap.

### Gradient clipping

Jika satu batch berisi data aneh, gradiennya bisa sangat besar dan merusak model dalam satu langkah. Pengamannya:

```
jika ‖g‖ > c:  g ← g · c/‖g‖
```

Arah dipertahankan, panjangnya dipotong menjadi tepat `c`. Nilai umum `c = 1,0`. Contoh `g = [3, 4]`, `c = 1,0`: normanya `√(9+16) = 5`, jadi `g_baru = [3,4] × 1/5 = [0,6, 0,8]` (norma baru `= 1,0` ✓).

### Hiperparameter model nyata

Nilai untuk GPT-3 175B:

| Hiperparameter | Nilai |
|---|---|
| Optimizer | Adam |
| `β1` | 0,9 |
| `β2` | 0,95 |
| `ε` | 10⁻⁸ |
| Weight decay | 0,1 |
| Gradient clipping | 1,0 |
| Learning rate maksimum | 0,6 × 10⁻⁴ |
| Batch size (akhir) | 3,2 juta token |
| Schedule | warmup + cosine decay ke 10% |

Perhatikan `β2 = 0,95`, bukan nilai bawaan `0,999`. Nilai lebih rendah membuat estimasi momen kedua lebih responsif terhadap perubahan terkini — pilihan lazim untuk model bahasa besar.

## Implementasi

- [`src/llmbepees/np/optimizer.py`](../../src/llmbepees/np/optimizer.py): `sgd_step`, `mini_batch_gradient`, `noise_std_after_batch`, `momentum_step`, `momentum_limit`, `adam_step`, `adamw_step`, `gradient_clip`, `lr_warmup`, `lr_cosine`, `adam_memory_bytes`.
- Uji memverifikasi semua tabel di atas: urutan SGD (konvergen & divergen), momentum berayun/konsisten, langkah Adam ≈ `η`, jadwal LR, clipping, dan memori `16N`.
- Berbeda dari `calculus.gradient_descent_step` yang beroperasi satu langkah tanpa state, optimizer di sini membawa **state** (`v`, `m`, `t`) — persis seperti optimizer nyata.

## Inti bab

Gradien memberi **arah**, optimizer memberi **ukuran langkah**. Gradient descent dasar punya kelemahan yang terlihat jelas di data nyata: gradien antarparameter bisa berbeda skala puluhan kali, dan satu learning rate tidak bisa melayani semuanya. Adam menyelesaikannya dengan membagi setiap gradien dengan akar rata-rata kuadratnya sendiri, sehingga pembilang dan penyebut sama-sama berskala dengan besar gradien dan saling meniadakan. Sifat yang benar-benar dijamin adalah **kekebalan terhadap penskalaan ulang gradien per parameter**: mengalikan gradien sebuah parameter dengan konstanta tidak mengubah langkahnya. Harganya adalah memori: `16N` byte, sehingga model 7 miliar butuh ~112 GB hanya untuk state optimizer.

Tambahan wajib yang menyertai Adam: **warmup**, **cosine decay**, dan **gradient clipping** — tanpa itu pelatihan model besar sering gagal di langkah-langkah pertama.

## Cek pemahaman

1. Mengapa gradien dihitung pada sebagian data saja, bukan seluruh data?
2. Apa yang ditambahkan momentum dibanding SGD biasa?
3. Mengapa Adam membutuhkan memori tambahan, dan berapa banyak?

## Rujukan

- Buku *"Memahami Matematika di Balik LLM"*, Bab 12.
- Kingma & Ba (2015) — Adam; Loshchilov & Hutter (2019) — AdamW; Brown dkk. (2020) — hiperparameter GPT-3.
