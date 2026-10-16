# Bab 6 — Teori Informasi: Entropi, Cross-Entropy, Perplexity

> Catatan belajar, ditulis ulang dengan bahasa sendiri dari buku *"Memahami Matematika di Balik LLM"* (Bab 6).
> Bagian dari **Fase 1 — Fondasi Matematika**. Kode: [`src/llmbepees/np/information.py`](../../src/llmbepees/np/information.py) · Uji: [`tests/test_information.py`](../../tests/test_information.py).

## Masalah

Fungsi kerugian (loss) yang dipakai untuk melatih setiap LLM di dunia berasal dari satu bidang: teori informasi, yang dirumuskan Shannon (1948). Bab ini menurunkan fungsi itu dari awal.

## Gagasan

### Mengukur kejutan

**Kandungan informasi** (self-information) sebuah kejadian berprobabilitas `p`:

```
I(p) = −log p
```

Basis `2` → satuan **bit**; basis `e` (yaitu `ln`) → satuan **nat**. Dalam pelatihan LLM hampir selalu `ln`. Tiga syarat yang membenarkan bentuk ini: kejadian pasti (`p=1`) tidak mengejutkan (`−log 1 = 0`); makin jarang makin mengejutkan; kejutan dua kejadian bebas dijumlahkan (`−log(p₁p₂) = −log p₁ − log p₂`). Fungsi `−log` adalah satu-satunya bentuk (sampai faktor konstan) yang memenuhi ketiganya.

| p | −log₂ p (bit) | −ln p (nat) |
|---|---|---|
| 1,000 | 0,000 | 0,000 |
| 0,500 | 1,000 | 0,693 |
| 0,250 | 2,000 | 1,386 |
| 0,100 | 3,322 | 2,303 |
| 0,001 | 9,966 | 6,908 |

Jika model memberi probabilitas `0,001` pada token yang ternyata benar, ia "terkejut" sebesar `6,908` nat — itulah loss untuk token tersebut.

### Entropi

Entropi = rata-rata kejutan sebuah distribusi:

```
H(P) = −∑_x P(x) log P(x) = E_{x∼P}[−log P(x)]
```

Contoh `P = [0,5, 0,25, 0,125, 0,125]` (log basis 2): `H(P) = 1,750` bit. Distribusi seragam `Q = [0,25, 0,25, 0,25, 0,25]`: `H(Q) = 2,000` bit.

Untuk `|Ω| = n`, entropi **maksimum** `log n` (tercapai saat seragam); **minimum** 0 (saat satu kejadian berprobabilitas 1). Untuk kosakata mini-LLM (`n=8`): maksimum `log₂ 8 = 3` bit = `ln 8 = 2,0794` nat. Angka `2,0794` ini patokan berguna: jika model menebak seragam, loss-nya sekitar `ln V`. Loss **di atas** `ln V` berarti model lebih buruk daripada menebak seragam.

### Cross-entropy: fungsi loss LLM

```
H(P, Q) = −∑_x P(x) log Q(x)
```

Dibaca: *ambil probabilitas sebenarnya sebagai bobot, tapi hitung kejutannya menurut model.* Contoh `P = [0,5, 0,25, 0,125, 0,125]`, `Q = [0,25]×4`:

| x | P(x) | Q(x) | −log₂ Q(x) | P(x)·(−log₂ Q(x)) |
|---|---|---|---|---|
| 1 | 0,500 | 0,25 | 2 | 1,00 |
| 2 | 0,250 | 0,25 | 2 | 0,50 |
| 3 | 0,125 | 0,25 | 2 | 0,25 |
| 4 | 0,125 | 0,25 | 2 | 0,25 |
| | | | | **2,00** |

Cross-entropy selalu ≥ entropi: `H(P,Q) ≥ H(P)`, dengan kesamaan hanya jika `P = Q`. Selisih `2,00 − 1,75 = 0,25` bit adalah harga yang dibayar karena model salah.

### KL divergence

```
D_KL(P‖Q) = ∑_x P(x) log(P(x)/Q(x)) = H(P,Q) − H(P)
```

Untuk contoh di atas `D_KL = 0,25` bit — cocok dengan `H(P,Q) − H(P)`. Dua sifat wajib: (1) `D_KL ≥ 0` selalu, `= 0` hanya jika `P = Q`; (2) **tidak simetris** `D_KL(P‖Q) ≠ D_KL(Q‖P)` pada umumnya. Muncul sebagai suku penalti di RLHF (Bab 20) agar model hasil fine-tuning tidak menyimpang jauh dari model awal.

Karena `H(P)` tidak bergantung pada parameter model, **meminimalkan cross-entropy = meminimalkan KL divergence**:

```
argmin_θ H(P, Q_θ) = argmin_θ D_KL(P ‖ Q_θ)
```

Jadi "melatih LLM = meminimalkan cross-entropy" dan "membuat distribusi model semirip mungkin dengan data" adalah hal yang identik.

### Bentuk khusus untuk LLM: target one-hot

Saat pelatihan, kita **tahu persis** token mana yang muncul, jadi distribusi target `P` bersifat **one-hot**: `1` pada token yang teramati, `0` di sisanya. (Ini target satu pengamatan, **bukan** klaim bahwa bahasa deterministik.) Jika token benar berindeks `c`, semua suku selain token benar menjadi nol:

```
L = −log P_θ(x_benar | konteks)
```

Rumusnya sesederhana itu: **negatif logaritma dari probabilitas yang diberikan model untuk token yang seharusnya keluar.**

Contoh dari mini-LLM: model demonstrasi memberi `P(meja) = 0,1105` → `L = −ln(0,1105) = 2,2032` nat, **lebih besar** dari patokan tebak-asal `2,0794` (bobot sembarangan itu aktif menyesatkan). Model terlatih (Bab 21) memberi `P(meja) = 0,7492` → `L = −ln(0,7492) = 0,2887` nat.

### Loss rata-rata untuk banyak token

```
L = −(1/N) ∑_{t=1}^{N} log P_θ(x_t | x_{<t})
```

Contoh empat token dengan probabilitas `[0,5, 0,25, 0,4, 0,1]`:

```
L = −(1/4)[ln 0,5 + ln 0,25 + ln 0,4 + ln 0,1] = −(−5,2983)/4 = 1,3246
```

### Perplexity

Angka loss dalam nat sulit dibayangkan, jadi orang melaporkan **perplexity**:

```
PPL = exp(L) = exp(−(1/N) ∑ ln P_θ(x_t | x_{<t}))
```

Untuk `L = 1,3246`: `PPL = e^{1,3246} = 3,7606`. Cara membacanya: **perplexity `k` = model sama bingungnya dengan orang yang harus memilih acak seragam dari `k` pilihan tiap langkah.** Bukti: jika model seragam `1/k`, maka `L = −ln(1/k) = ln k`, sehingga `PPL = e^{ln k} = k`.

| L (nat) | PPL | Tafsiran |
|---|---|---|
| 0,00 | 1,00 | sempurna |
| 0,69 | 2,00 | seakan memilih dari 2 pilihan |
| 1,32 | 3,76 | |
| 2,08 | 8,00 | setara menebak seragam dari 8 token |
| 10,82 | 50.000 | menebak asal dari kosakata penuh |

Kemajuan perplexity mini-LLM selama pelatihan (Bab 21): `2,4480` (PPL 11,57) di langkah 0 → `0,0955` (PPL 1,10) di langkah 400. Kurvanya turun cepat di awal lalu mendatar — pola yang muncul di hampir semua pelatihan jaringan saraf (Bab 18).

### Bit per karakter (dan byte)

Perplexity bergantung pada tokenizer: tokenizer yang memecah teks jadi potongan lebih kecil → lebih banyak token per kalimat → tiap token lebih mudah ditebak → perplexity terlihat lebih rendah padahal kualitas sama. Karena itu dilaporkan **bits per byte (BPB)** / **bits per character (BPC)**:

```
BPB = L_total (nat) / (ln 2 × jumlah byte teks)
```

Contoh: teks 1000 byte → 250 token, loss rata-rata 1,5 nat/token. Loss total `250 × 1,5 = 375` nat → `375/0,6931 = 541,1` bit → `541,1/1000 = 0,541` bit per byte.

## Implementasi

- [`src/llmbepees/np/information.py`](../../src/llmbepees/np/information.py): `information`, `entropy`, `cross_entropy`, `kl_divergence`, `mean_negative_log_likelihood`, `perplexity`, `bits_per_byte`.
- [`tests/test_information.py`](../../tests/test_information.py): semua contoh angka di atas (nat & bit).

## Inti bab

Setelah model mengeluarkan daftar probabilitas, kita menilai jawabannya dengan **satu angka**: loss = negatif logaritma dari probabilitas yang diberikan model untuk token yang benar. Jika model memberi peluang besar pada jawaban benar, loss kecil; jika kecil, loss membengkak. Tidak ada komponen lain. **Perplexity** `= e^L` adalah cara lain membaca angka yang sama, biasanya lebih mudah dibayangkan.

## Cek pemahaman

1. Mengapa loss memakai logaritma, bukan langsung selisih probabilitas?
2. Mengapa loss sebuah LLM tidak pernah turun sampai nol, sekalipun modelnya sangat besar?
3. Apa arti praktis perplexity 20?

## Rujukan

- Buku *"Memahami Matematika di Balik LLM"*, Bab 6.
- Shannon (1948) — teori informasi.
