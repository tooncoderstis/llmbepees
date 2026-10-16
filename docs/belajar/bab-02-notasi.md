# Bab 2 — Kamus Notasi: Cara Membaca Rumus

> Catatan belajar, ditulis ulang dengan bahasa sendiri dari buku *"Memahami Matematika di Balik LLM"* (Bab 2).
> Bagian dari **Fase 1 — Fondasi Matematika**. Tanpa kode: bab ini adalah kamus notasi, bukan konsep baru.

## Masalah

Rumus di buku ini penuh simbol: `∑`, `∏`, `eˣ`, `∂`, huruf tebal. Tanpa kamus, notasi matematika terasa seperti bahasa rahasia. Padahal ia cuma **sistem singkatan**: setiap simbol berdiri untuk satu instruksi yang, jika ditulis dengan kalimat biasa, akan memakan tiga baris.

## Gagasan

### Variabel, indeks, dan konvensi huruf

| Bentuk | Arti | Contoh |
|---|---|---|
| `x` (huruf kecil biasa) | satu angka tunggal (**skalar**) | `x = 3,5` |
| **x** (huruf kecil tebal) | **vektor**: deretan angka | `x = [1, 0, 2, 5]` |
| **X** (huruf besar tebal) | **matriks**: tabel angka | tabel 5 baris × 4 kolom |
| `x_i` | elemen ke-`i` dari vektor **x** | `x_3` = elemen ketiga |
| `X_ij` | elemen baris ke-`i`, kolom ke-`j` | `X_24` = baris 2 kolom 4 |

Angka kecil di bawah = **indeks** (nomor urut), bukan pangkat. Angka kecil di atas biasanya pangkat (`x² = x·x`), **tetapi** angka di atas dalam kurung berarti nomor langkah/lapisan: `h⁽³⁾` = keluaran lapisan ke-3, bukan pangkat tiga.

Contoh angka, `x = [0,5, 1,2, −0,3, 4,0]`: `x_1 = 0,5`, `x_4 = 4,0`, `x_2² = (−0,3)² = 0,09`.

> Buku menomori indeks **mulai 1** di rumus matematika, sedangkan kode komputer **mulai 0**. Setiap kali penomoran dari 0 dipakai, dinyatakan eksplisit.

### Σ — penjumlahan

`∑` (sigma kapital) berarti *jumlahkan*. Anatominya: batas bawah (mulai), batas atas (berhenti), ekspresi yang dijumlahkan.

```
∑_{i=1}^{n} a_i = a_1 + a_2 + … + a_n
```

Terjemahan perintah: *mulai `i=1`, ambil `a_i`, tambahkan; naikkan `i` satu; ulangi sampai `i=n`.*

Contoh, `a = [2, 5, 1, 7]`:

```
∑_{i=1}^{4} a_i = 2 + 5 + 1 + 7 = 15
∑_{i=1}^{4} a_i² = 2² + 5² + 1² + 7² = 4 + 25 + 1 + 49 = 79
```

Sigma ganda = sigma bersarang. Untuk setiap `i`, jalankan semua `j`. Dengan **M** = `[[1,2,3],[4,5,6]]`:

```
∑_{i=1}^{2} ∑_{j=1}^{3} M_ij = (1+2+3) + (4+5+6) = 6 + 15 = 21
```

### ∏ — perkalian

`∏` (pi kapital) sama seperti sigma, tetapi **mengalikan**:

```
∏_{i=1}^{n} a_i = a_1 × a_2 × … × a_n
```

Dengan `a = [2, 5, 1, 7]`: `∏_{i=1}^{4} a_i = 2 × 5 × 1 × 7 = 70`. Simbol ini muncul di Bab 1 saat menghitung probabilitas kalimat utuh.

### Fungsi

`f(x)` dibaca "f dari x": ada aturan bernama `f`, masukkan `x`, keluar satu nilai.

- `f(x) = 2x + 3` → `f(5) = 13`.
- Bisa banyak masukan: `f(x,y) = x² + y` → `f(3,4) = 9 + 4 = 13`.
- Bisa menerima & mengeluarkan **vektor** (paling sering di LLM). Contoh `ReLU`:
  `ReLU(x) = [max(0, x_1), …, max(0, x_n)]`; untuk `x = [2, −1, 0,5, −3]` → `[2, 0, 0,5, 0]`.
- Notasi tipe: `f : ℝⁿ → ℝᵐ` dibaca "f memetakan dari daftar `n` angka ke daftar `m` angka". `ℝ` = himpunan semua bilangan real.

### Eksponensial dan logaritma

`e ≈ 2,718281828` — konstanta (bukan variabel). `exp(x)` dan `eˣ` penulisan yang sama.

Sifat penting `eˣ`:

1. **Selalu positif**, berapa pun `x` (bahkan `e^{−100} ≈ 3,7 × 10^{-44}`). Sifat inilah yang dipakai softmax agar probabilitas tak pernah negatif.
2. `e⁰ = 1`.
3. `e^{a+b} = e^a × e^b`.
4. Naik sangat cepat: dari `x=1` (`2,7`) ke `x=5` (`148,4`).

`ln(x)` adalah **kebalikan** `eˣ`: jika `e^a = b` maka `ln(b) = a`. Sifat penting:

1. `ln(1) = 0`.
2. `ln(x) < 0` untuk `0 < x < 1`. Inilah alasan fungsi loss LLM ditulis `−ln(p)`: karena `p` adalah probabilitas (0–1), `ln(p)` negatif, dan tanda minus di depannya membuatnya positif.
3. Tidak terdefinisi untuk `x ≤ 0`.
4. `ln(a × b) = ln(a) + ln(b)` — **mengubah perkalian menjadi penjumlahan**.

Contoh angka (menyambung Bab 1), `0,50 × 0,25 × 0,40 × 0,10 = 0,005`:

```
ln(0,005) = ln(0,5) + ln(0,25) + ln(0,4) + ln(0,1)
          = (−0,6931) + (−1,3863) + (−0,9163) + (−2,3026) = −5,2983
```

Cek: `e^{−5,2983} = 0,00500`. Cocok. Keuntungannya terasa untuk kalimat panjang: perkalian 1000 probabilitas menghasilkan sekitar `10^{−1000}` (jadi 0, informasi hilang), sedangkan penjumlahan 1000 logaritma menghasilkan sekitar `−2300` (aman).

**Basis logaritma:** `ln` ber-basis `e`; ada juga `log₂` (basis 2) dan `log₁₀`. `log₂(x) = ln(x) / ln(2) = ln(x) / 0,6931`. Dalam teori informasi (Bab 6), `log₂` untuk satuan **bit** dan `ln` untuk **nat**; pelatihan LLM hampir selalu memakai `ln`.

### Simbol lain yang akan muncul

| Simbol | Dibaca | Arti |
|---|---|---|
| `≈` | kira-kira sama dengan | nilai hampiran |
| `∝` | sebanding dengan | sama sampai faktor pengali konstan |
| `∈` | anggota dari | `x ∈ ℝ`: `x` bilangan real |
| `ℝⁿ` | R pangkat n | himpunan vektor berisi `n` bilangan real |
| `argmax_i` | argumen yang memaksimalkan | **indeks** `i` yang memberi nilai terbesar |
| `max` | maksimum | **nilai** terbesar |
| `Aᵀ` | A transpos | baris ↔ kolom (Bab 4) |
| `‖x‖` | norma x | panjang vektor (Bab 3) |
| `a ⋅ b` | a dot b | dot product (Bab 3) |
| `∂L/∂w` | del L per del w | turunan parsial (Bab 7) |
| `∇` | nabla / gradien | kumpulan semua turunan parsial (Bab 7) |
| `𝔼[·]` | ekspektasi | nilai rata-rata (Bab 5) |
| `∼` | berdistribusi sebagai | `x ∼ P`: `x` diambil acak dari `P` |

**max vs argmax** (sering tertukar). Untuk `p = [0,1, 0,7, 0,2]` (indeks mulai 1): `max_i p_i = 0,7` (nilainya), `argmax_i p_i = 2` (posisinya). Di LLM, `argmax` dipakai memilih token: kita butuh **token mana** (posisi), bukan nilai probabilitasnya.

### Latihan membaca rumus lengkap

Rumus asli dari *Attention Is All You Need*:

```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) V
```

| Potongan | Cara baca |
|---|---|
| `Attention(Q,K,V)` | fungsi bernama Attention, tiga masukan (semuanya matriks) |
| `QKᵀ` | matriks **Q** dikalikan matriks **K** yang ditranspos |
| `√d_k` | akar kuadrat dari bilangan bernama `d_k` |
| `⋅ / ⋅` | bagi seluruh matriks di atas dengan angka di bawah |
| `softmax(⋅)` | terapkan fungsi softmax pada hasilnya |
| `(⋅)V` | kalikan hasil softmax dengan matriks **V** |

(Maksud rumusnya baru dibahas di Bab 13.) Jika tabel ini bisa dibaca, 90% notasi di buku ini sudah bisa dibaca.

## Inti bab

Bab ini tidak menambah pengetahuan tentang LLM, hanya kemampuan **membaca**. Empat simbol yang paling sering muncul sampai akhir buku:

1. `∑` — jumlahkan semuanya.
2. `∏` — kalikan semuanya.
3. `eˣ` — selalu positif; dipakai softmax agar probabilitas tak negatif.
4. `ln` — kebalikannya; mengubah perkalian yang beranak-pinak menjadi penjumlahan yang jinak.

> Jika hanya mengingat satu hal: **setiap simbol matematika adalah singkatan dari sebuah instruksi, bukan mantra.** Bisa menerjemahkan simbol menjadi kalimat = bisa membaca rumus apa pun di buku ini.

## Cek pemahaman

1. Apa beda `max` dan `argmax`?
2. Mengapa `eˣ` berguna untuk membuat probabilitas?
3. Apa maksud huruf tebal kapital seperti **W** dibanding huruf biasa `w`?

## Rujukan

- Buku *"Memahami Matematika di Balik LLM"*, Bab 2.
- Vaswani dkk. (2017), *Attention Is All You Need* — arXiv:1706.03762 (contoh rumus Attention).
