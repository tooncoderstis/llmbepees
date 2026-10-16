# Bab 5 — Probabilitas

> Catatan belajar, ditulis ulang dengan bahasa sendiri dari buku *"Memahami Matematika di Balik LLM"* (Bab 5).
> Bagian dari **Fase 1 — Fondasi Matematika**. Kode: [`src/llmbepees/np/probability.py`](../../src/llmbepees/np/probability.py) · Uji: [`tests/test_probability.py`](../../tests/test_probability.py).

## Masalah

Keluaran LLM adalah **distribusi probabilitas**. Fungsi kerugian saat pelatihan berasal dari teori probabilitas, begitu pula metode sampling saat generasi. Bab ini membangun semuanya dari definisi.

## Gagasan

### Ruang sampel dan probabilitas

**Ruang sampel** `Ω` adalah himpunan semua hasil yang mungkin. Untuk LLM, ruang sampelnya adalah **kosakata**:

```
Ω = { <bos>, kucing, duduk, di, atas, meja, tikar, <eos> },   |Ω| = 8
```

**Distribusi probabilitas** `P` memberi setiap anggota `ω ∈ Ω` sebuah angka `P(ω)` dengan dua syarat mutlak:

- **(A1)** `0 ≤ P(ω) ≤ 1` untuk semua `ω`.
- **(A2)** `∑_{ω∈Ω} P(ω) = 1`.

Kedua syarat inilah yang harus dipenuhi keluaran LLM; mekanisme yang memaksanya (softmax) dibahas di Bab 10. Contoh distribusi terlatih setelah konteks `<bos> kucing duduk di atas`: `P(meja) = 0,7492`, `P(tikar) = 0,2494`, sisanya kecil, total (dengan pembulatan) `1,0001` → nilai sebenarnya tepat 1.

**Distribusi seragam**: semua anggota punya probabilitas sama `1/|Ω|`; untuk `|Ω| = 8` masing-masing `0,125`. Ini keadaan model sebelum belajar apa pun — menebak asal.

### Probabilitas bersyarat

```
P(A | B) = P(A ∩ B) / P(B),   P(B) > 0
```

`P(A ∩ B)` dibaca "probabilitas A dan B terjadi bersama". Inilah inti seluruh LLM: **model bahasa adalah kalkulator probabilitas bersyarat** — berapa peluang token berikutnya, diberikan konteks yang sudah ada.

Contoh: korpus 100 kalimat, 40 memuat "di atas", 30 di antaranya "di atas meja", 8 "di atas tikar":

```
P(meja | di atas) = (30/100) / (40/100) = 30/40 = 0,75
P(tikar | di atas) = 8/40 = 0,20
```

Cek A2: `0,75 + 0,20 + 2/40 = 1,00` ✓. Pendekatan menghitung frekuensi seperti ini disebut **model n-gram** — pendahulu LLM. Masalahnya: untuk konteks 20 kata, jumlah kombinasi mencapai `V²⁰`; dengan `V = 50.000` itu `10^94` kombinasi, tak ada korpus cukup besar. Inilah masalah yang diselesaikan jaringan saraf: alih-alih menyimpan tabel, ia menyimpan **fungsi** yang menghitung probabilitas dari parameter.

### Aturan rantai

Dari definisi bersyarat, `P(A ∩ B) = P(B) · P(A | B)`, diperluas ke banyak kejadian:

```
P(x_1, x_2, …, x_n) = ∏_{t=1}^{n} P(x_t | x_1, …, x_{t-1})
```

Untuk `t=1` tidak ada konteks, jadi sukunya `P(x_1)`. Contoh 3 token:

```
P(x_1, x_2, x_3) = P(kucing) × P(duduk | kucing) × P(di | kucing duduk)
                = 0,5 × 0,25 × 0,4 = 0,05
```

Yang penting: aturan rantai berlaku **persis, bukan hampiran** — tidak ada asumsi yang dilanggar. Karena itu, model yang hanya bisa memprediksi satu token berikutnya secara prinsip cukup untuk memodelkan teks sepanjang apa pun. Ini justifikasi matematis dari seluruh pendekatan LLM.

### Log-probabilitas

Karena perkalian probabilitas cepat menuju nol, praktik standar bekerja dengan logaritmanya:

```
log P(x_1, …, x_n) = ∑_{t=1}^{n} log P(x_t | x_{<t})    (x_{<t} = x_1, …, x_{t-1})
```

Contoh kalimat dengan probabilitas `[0,5, 0,25, 0,4, 0,1]`:

| t | P | ln P |
|---|---|---|
| 1 | 0,50 | −0,6931 |
| 2 | 0,25 | −1,3863 |
| 3 | 0,40 | −0,9163 |
| 4 | 0,10 | −2,3026 |
| | | **−5,2983** |

Rata-ratanya `−5,2983/4 = −1,3246`. Angka ini muncul lagi di Bab 6 sebagai **loss**, dan `e^{1,3246} = 3,7606` sebagai **perplexity**.

Kenapa harus log? Tiga alasan: (1) **rentang bilangan** — probabilitas 100 token dengan rata-rata 0,1 adalah `10^{−100}` yang dibulatkan jadi 0 pada float, sedangkan bentuk log-nya `−230,3` aman; (2) penjumlahan lebih jinak untuk rentang bilangan; (3) **turunan lebih sederhana**, sehingga backpropagation jauh lebih ringan (Bab 10).

### Nilai harapan (ekspektasi)

```
E_{x∼P}[f(x)] = ∑_{x∈Ω} P(x) · f(x)
```

Rata-rata berbobot dengan probabilitas sebagai bobotnya. Contoh dadu 4 sisi tak seragam: `E[x] = 1(0,1) + 2(0,2) + 3(0,3) + 4(0,4) = 3,0`. Nilai harapan **tidak harus** salah satu nilai yang mungkin: untuk `P = [0,5, 0,5, 0, 0]`, `E[x] = 1,5` padahal dadu tak punya sisi 1,5.

Nilai harapan muncul di semua definisi loss LLM: `L = E_{x∼data}[−log P_θ(x)]`, dengan `θ` lambang semua parameter model; ekspektasi dihampiri rata-rata pada batch data (Bab 18).

### Varians dan simpangan baku

```
Var(x) = E[(x − E[x])²] = E[x²] − (E[x])²,   σ = √Var(x)
```

Contoh `h = [2, 4, 4, 6]` (bobot sama `1/4`): rata-rata `μ = 4`, selisih `[−2, 0, 0, 2]`, kuadrat `[4, 0, 0, 4]`, varians `2`, simpangan baku `√2 = 1,4142`.

Normalisasi persis inilah yang dilakukan **LayerNorm** pada setiap vektor token di setiap lapisan:

```
ĥ = (h − μ)/σ = [−2, 0, 0, 2]/1,4142 = [−1,4142, 0, 0, 1,4142]
```

Cek: rata-rata hasilnya 0, variansnya 1 — persis seperti yang diinginkan (Bab 16).

### Distribusi normal

```
p(x) = 1/(σ√(2π)) · exp(−(x − μ)² / (2σ²)),   ditulis x ∼ N(μ, σ²)
```

Sekitar 68% nilai jatuh dalam satu simpangan baku dari rata-rata, 95% dalam dua. Muncul di dua tempat: (1) **inisialisasi parameter** — bobot diisi acak dari `N(0, σ²)` dengan `σ` kecil (GPT-2 memakai `σ = 0,02`); (2) **analisis skala** — asumsi komponen `q` dan `k` berdistribusi `N(0,1)` adalah dasar argumen `√d_k` di Bab 13.

### Sampling dari distribusi diskrit

Saat menghasilkan teks, model harus memilih satu token dari distribusi. Prosedurnya (**inverse transform sampling**):

1. Hitung probabilitas kumulatif `C_k = ∑_{i=1}^{k} p_i`.
2. Ambil `u` seragam di `[0, 1)`.
3. Pilih token pertama yang `C_k > u`.

Contoh `p = [0,5, 0,3, 0,15, 0,05]` untuk token A, B, C, D:

| Token | p | Kumulatif C | Rentang u |
|---|---|---|---|
| A | 0,50 | 0,50 | [0,00, 0,50) |
| B | 0,30 | 0,80 | [0,50, 0,80) |
| C | 0,15 | 0,95 | [0,80, 0,95) |
| D | 0,05 | 1,00 | [0,95, 1,00) |

Untuk `u = 0,62` → token **B**. Panjang tiap segmen tepat sama dengan probabilitasnya, sehingga peluang `u` jatuh di segmen B tepat 0,30. Bab 19 memodifikasi prosedur ini dengan temperature, top-k, dan top-p — semuanya mengubah `p` **sebelum** langkah 1.

## Implementasi

- [`src/llmbepees/np/probability.py`](../../src/llmbepees/np/probability.py): `validate_distribution`, `uniform`, `conditional_probability`, `expected_value`, `variance`, `std`, `cumulative_sum`, `sample_categorical`, `normal_pdf`.
- [`tests/test_probability.py`](../../tests/test_probability.py): semua contoh angka di atas + uji statistik sampling (20000 undian).

## Inti bab

Keluaran LLM bukan kata, melainkan **distribusi probabilitas**: satu angka per token, semua positif, totalnya tepat satu. Bagian terpenting adalah **aturan rantai probabilitas** — probabilitas kalimat utuh = hasil kali probabilitas tiap token, dan ia berlaku **persis**. Karena itu, model yang hanya sanggup menebak satu token berikutnya sudah cukup untuk memodelkan teks sepanjang apa pun.

## Cek pemahaman

1. Mengapa model bahasa disebut kalkulator probabilitas bersyarat?
2. Mengapa bekerja dengan logaritma probabilitas, bukan probabilitas langsung?
3. Apa arti "aturan rantai berlaku persis" bagi cara kerja LLM?

## Rujukan

- Buku *"Memahami Matematika di Balik LLM"*, Bab 5.
- Bengio dkk. (2003) — model bahasa saraf; Radford dkk. (2019) — `σ = 0,02`.
