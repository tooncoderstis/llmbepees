# Bab 10 — Softmax dan Fungsi Kerugian

> Catatan belajar, ditulis ulang dengan bahasa sendiri dari buku *"Memahami Matematika di Balik LLM"* (Bab 10).
> Bagian dari **Fase 4 — Softmax & Loss**. Kode: [`src/llmbepees/np/softmax.py`](../../src/llmbepees/np/softmax.py), [`src/llmbepees/np/loss.py`](../../src/llmbepees/np/loss.py) · Uji: [`tests/test_softmax.py`](../../tests/test_softmax.py), [`tests/test_loss.py`](../../tests/test_loss.py).

## Masalah

Keluaran terakhir jaringan adalah sekumpulan angka sembarang: bisa negatif, bisa 47, bisa −3,2. Bab 5 sudah menetapkan bahwa keluaran LLM harus berupa **distribusi probabilitas**. Bab ini adalah jembatan di antara keduanya — dan bagian yang paling berharga bukan softmax-nya, melainkan **gradiennya**.

## Gagasan

### Logit

**Logit** adalah keluaran mentah lapisan terakhir jaringan, sebelum diubah menjadi probabilitas. Tidak ada batasan nilainya: bisa negatif, besar, atau nol.

| Token | Logit |
|---|---|
| `<bos>` | 0,0000 |
| kucing | −0,1725 |
| duduk | 0,1725 |
| di | −1,9539 |
| atas | 1,9539 |
| meja | 0,3907 |
| tikar | −0,3907 |
| `<eos>` | 0,0000 |

Jumlahnya 0, bukan 1. Ada yang negatif. Belum bisa disebut probabilitas.

### Softmax: definisi

```
softmax(z)_i = e^{z_i} / ∑_j e^{z_j}
```

Dibaca: "eksponensial dari logit ke-`i`, dibagi jumlah eksponensial semua logit." Dua langkah: (1) `e^{z}` membuat semua nilai **positif**; (2) bagi dengan totalnya agar jumlahnya **1**. Kedua syarat probabilitas (positif & berjumlah 1) terpenuhi otomatis, apa pun nilai logitnya.

**Contoh lengkap.** `z = [2,0, 1,0, 0,1]`:

| i | z_i | e^{z_i} | e^{z_i}/Σ |
|---|---|---|---|
| 1 | 2,0 | 7,3891 | 0,6590 |
| 2 | 1,0 | 2,7183 | 0,2424 |
| 3 | 0,1 | 1,1052 | 0,0986 |
| | | Σ = 11,2125 | Σ = 1,0000 ✓ |

Selisih logit 2,0 vs 1,0 hanya 1,0, tetapi probabilitasnya 0,659 vs 0,242 — hampir tiga kali lipat. Softmax **memperbesar** selisih, karena `e¹ = 2,718`.

**Contoh mini-LLM.** Dengan logit di atas:

| Token | Logit | e^z | Probabilitas |
|---|---|---|---|
| `<bos>` | 0,0000 | 1,0000 | 0,0747 |
| kucing | −0,1725 | 0,8415 | 0,0629 |
| duduk | 0,1725 | 1,1883 | 0,0888 |
| di | −1,9539 | 0,1418 | 0,0106 |
| atas | 1,9539 | 7,0563 | 0,5273 |
| meja | 0,3907 | 1,4780 | 0,1105 |
| tikar | −0,3907 | 0,6766 | 0,0506 |
| `<eos>` | 0,0000 | 1,0000 | 0,0747 |
| | | Σ = 13,3825 | Σ = 1,0000 |

### Sifat softmax

**Sifat 1 — invarian terhadap pergeseran.** Menambahkan konstanta `c` ke semua logit tidak mengubah hasilnya: `softmax(z + c) = softmax(z)` (faktor `e^c` muncul di pembilang dan penyebut lalu saling meniadakan). Ini penting secara praktis: `e^{1002}` meluap pada float32 (`float32` mentok ≈ `3,4 × 10³⁸`, sedangkan `e^{1002} ≈ 10^{435}`) → hasilnya `inf`, lalu `inf/inf = NaN`, dan seluruh pelatihan rusak. Solusinya: kurangi dulu semua logit dengan yang terbesar.

```
softmax(z)_i = e^{z_i − max_j z_j} / ∑_k e^{z_k − max_j z_j}
```

Setelah pengurangan, logit terbesar menjadi 0 sehingga `e⁰ = 1` — tidak mungkin meluap. Semua implementasi softmax nyata memakai trik ini.

**Sifat 2 — mempertahankan urutan.** Jika `z_i > z_j` maka `p_i > p_j`. Karena itu `argmax` pada logit dan pada probabilitas menghasilkan token yang sama — kalau hanya butuh token terbaik, softmax tidak perlu dihitung.

**Sifat 3 — sensitivitas terhadap skala.** Jika semua logit dikali konstanta besar, distribusinya menjadi runcing. Contoh `z = [3,0, 1,0, 0,5, −1,0]`:

| Skala | Softmax |
|---|---|
| ×1 | [0,8092, 0,1095, 0,0664, 0,0148] |
| ×8 | [1,0000, 0,0000, 0,0000, 0,0000] |

Dengan skala ×8, distribusinya praktis menjadi one-hot. Fenomena ini akan menjadi masalah di Bab 13 (diselesaikan dengan pembagian `√d_k`) dan menjadi alat kendali di Bab 19 (*temperature*).

### Menyatukan softmax dengan cross-entropy

Dari Bab 6, loss untuk satu token adalah `L = −ln p_c` (`c` = indeks token benar). Substitusi definisi softmax:

```
L = −ln( e^{z_c} / ∑_j e^{z_j} ) = −z_c + ln ∑_j e^{z_j}
```

`ln ∑_j e^{z_j}` disebut **log-sum-exp (LSE)**, jadi:

```
L = LSE(z) − z_c
```

**Contoh angka.** Logit mini-LLM dengan token benar *meja* (`z_c = 0,3907`): `LSE = ln(13,3825) = 2,5939`, sehingga `L = 2,5939 − 0,3907 = 2,2032`. Cocok dengan `−ln(0,1105) = 2,2032` dari Bab 6. Bentuk LSE lebih baik secara numerik: tidak butuh pembagian, dan LSE punya versi stabil sendiri (`LSE(z) = max_j z_j + ln ∑_j e^{z_j − max_j z_j}`). Karena itu PyTorch `cross_entropy` menerima **logit**, bukan probabilitas.

### Turunan softmax + cross-entropy — penurunan terpenting

Kita ingin `∂L/∂z_i`. Kasus `i = c`: turunan `−z_c` adalah `−1`; turunan `ln ∑ e^{z_j}` lewat aturan rantai adalah `e^{z_c}/∑ e^{z_j} = p_c`. Jadi `∂L/∂z_c = −1 + p_c = p_c − 1`. Kasus `i ≠ c`: suku pertama 0, suku kedua `p_i`. Menggabungkan keduanya dengan `y` = vektor one-hot label:

```
∂L/∂z = p − y
```

Ini salah satu rumus paling elegan di deep learning: **gradien = selisih antara yang diprediksi dan yang seharusnya.**

**Verifikasi numerik.** `z = [2,0, 1,0, 0,1]`, label benar kelas 1 (`y = [1, 0, 0]`), jadi `p = [0,6590, 0,2424, 0,0986]`. Rumus memprediksi `p − y = [−0,3410, 0,2424, 0,0986]`; hitungan numerik (central difference, `ε = 10⁻⁶`) memberi `[−0,340999, 0,242433, 0,098566]`. Cocok.

**Cara membaca gradien.** Besarnya gradien otomatis sebanding dengan besarnya kesalahan:

| Kondisi | p_i − y_i | Efek pembaruan |
|---|---|---|
| Token benar, prediksi rendah (`p_c = 0,1`) | −0,9 | logit dinaikkan kuat |
| Token benar, prediksi tinggi (`p_c = 0,9`) | −0,1 | logit dinaikkan sedikit |
| Token salah, prediksi tinggi (`p_i = 0,7`) | +0,7 | logit diturunkan kuat |
| Token salah, prediksi rendah (`p_i = 0,01`) | +0,01 | hampir tidak diubah |

**Contoh mini-LLM** (target *meja*): gradien terbesar kedua ada pada *atas* (`+0,5273`) — model terlalu yakin pada jawaban salah sehingga logit-nya ditekan paling keras. Jumlah seluruh gradien **selalu nol** (karena `∑p = 1` dan `∑y = 1`): menaikkan satu logit efektif berarti menurunkan yang lain.

| Token | p | y | p − y |
|---|---|---|---|
| `<bos>` | 0,0747 | 0 | +0,0747 |
| kucing | 0,0629 | 0 | +0,0629 |
| duduk | 0,0888 | 0 | +0,0888 |
| di | 0,0106 | 0 | +0,0106 |
| atas | 0,5273 | 0 | +0,5273 |
| meja | 0,1105 | 1 | −0,8895 |
| tikar | 0,0506 | 0 | +0,0506 |
| `<eos>` | 0,0747 | 0 | +0,0747 |
| | | | Σ ≈ 0 |

### Kenapa bukan mean squared error

Tanpa faktor `p_j`, gradien MSE lewat softmax berbentuk:

```
∂L_MSE/∂z_j = 2 p_j [(p_j − y_j) − ∑_i (p_i − y_i) p_i]
```

Faktor `p_j` di depan seluruh kurung membuat gradien **lenyap** saat model sangat salah (`p_c → 0`): sinyal perbaikan menghilang justru ketika model paling butuh dikoreksi. Cross-entropy tidak punya masalah itu karena gradiennya `p_c − 1` tanpa faktor pengecil. Contoh (kosakata 5 token, gradien terhadap logit token yang benar):

| p_c | Cross-entropy | MSE |
|---|---|---|
| 0,001 | −0,999 | −0,0025 |
| 0,500 | −0,500 | −0,3125 |
| 0,999 | −0,001 | −0,0000025 |

Saat model hampir benar, keduanya sama-sama kecil (memang diinginkan). Tetapi saat hampir sepenuhnya salah, cross-entropy memberi gradien hampir maksimum sementara MSE ~400 kali lebih kecil. Inilah alasan matematis cross-entropy menjadi pilihan baku untuk klasifikasi probabilistik. (Bukan berarti satu-satunya: hinge loss, focal loss, contrastive loss, ranking loss punya tujuannya masing-masing.)

### Label smoothing

Beberapa model mengganti label one-hot dengan versi "dilunakkan". Bentuk yang dipakai di sini menyebar `ε` merata ke seluruh `V` kelas (termasuk kelas target):

```
y_smooth_i = (1 − ε)·y_i + ε/V
```

Contoh `V = 8`, `ε = 0,1`, target *meja*: nilai target `0,9 + 0,0125 = 0,9125`, kelas lain `0,0125`; jumlah `0,9125 + 7 × 0,0125 = 1,0000` ✓. Efeknya: model tidak pernah didorong menuju probabilitas `1,0` mutlak, sehingga logitnya tidak tumbuh tanpa batas — sebuah bentuk **regularisasi**.

## Implementasi

- [`src/llmbepees/np/softmax.py`](../../src/llmbepees/np/softmax.py): `softmax`, `log_softmax`, `log_sum_exp` (semua stabil secara numerik).
- [`src/llmbepees/np/loss.py`](../../src/llmbepees/np/loss.py): `softmax_cross_entropy` (`LSE(z) − z_c`), `softmax_cross_entropy_gradient` (`p − y`), `label_smoothing`, serta `softmax_mse_loss`/`softmax_mse_gradient` (pembanding 10.6).
- Uji memverifikasi paritas dengan `information.cross_entropy` (Bab 6) dan `numerical_gradient` (Bab 7).

## Inti bab

Bagian paling berharga dari bab ini bukan softmax-nya, melainkan **gradiennya**. Ketika softmax dan cross-entropy diturunkan bersama, semua suku rumit saling menghapus dan yang tersisa hanya `p − y`. Hasil sesederhana itu punya sifat sangat berguna: **ketika model salah besar, gradiennya besar; ketika model sudah hampir benar, gradiennya mengecil sendiri.** Tidak ada yang perlu diatur manual.

Sifat kedua yang perlu dibawa: softmax sangat sensitif terhadap skala. Mengalikan semua masukan dengan angka besar membuat distribusinya runtuh menjadi satu pemenang mutlak — masalah nyata di Bab 13, dan pembagian dengan `√d_k` ada persis untuk mengatasinya.

## Cek pemahaman

1. Mengapa softmax memangkatkan `e` lebih dulu, bukan langsung membagi setiap angka dengan totalnya?
2. Mengapa mengurangi nilai maksimum sebelum softmax tidak mengubah hasilnya?
3. Apa arti praktis dari fakta bahwa gradiennya adalah `p − y`?

## Rujukan

- Buku *"Memahami Matematika di Balik LLM"*, Bab 10.
- Vaswani dkk. (2017) — label smoothing `ε = 0,1` di Transformer asli; Szegedy dkk. (2016) — label smoothing.
