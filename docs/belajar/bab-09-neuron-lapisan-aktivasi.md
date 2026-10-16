# Bab 9 — Neuron, Lapisan, dan Fungsi Aktivasi

> Catatan belajar, ditulis ulang dengan bahasa sendiri dari buku *"Memahami Matematika di Balik LLM"* (Bab 9).
> Bagian dari **Fase 3 — Neuron & Aktivasi**. Kode: [`src/llmbepees/np/activation.py`](../../src/llmbepees/np/activation.py), [`src/llmbepees/np/neuron.py`](../../src/llmbepees/np/neuron.py) · Uji: [`tests/test_activation.py`](../../tests/test_activation.py), [`tests/test_neuron.py`](../../tests/test_neuron.py).

## Masalah

Di dalam setiap blok Transformer ada komponen bernama **feed-forward network (FFN)** yang menyimpan sekitar **dua pertiga** dari seluruh parameter model — tetapi jarang dibahas orang. Bab ini membongkarnya dari satuan terkecilnya: satu neuron → satu lapisan → FFN dua lapis. Pertanyaan yang dijawab: kenapa harus ada fungsi **non-linear**?

## Gagasan

### Neuron

Satu neuron dengan `n` masukan adalah fungsi:

```
y = f( Σ_i w_i x_i + b ) = f(x · w + b)
```

| Simbol | Nama | Jenis |
|---|---|---|
| `x_i` | masukan | diberikan dari luar |
| `w_i` | bobot (weight) | parameter, dilatih |
| `b` | bias | parameter, dilatih |
| `f` | fungsi aktivasi | dipilih perancang, tidak dilatih |
| `y` | keluaran | satu angka |

Bagian `x · w + b` disebut **pre-aktivasi**, ditulis `z`.

**Contoh angka.** `x = [2,0, −1,0, 0,5]`, `w = [0,5, 1,0, −2,0]`, `b = 0,3`, aktivasi ReLU:

```
z = (2,0)(0,5) + (−1,0)(1,0) + (0,5)(−2,0) + 0,3 = 1,0 − 1,0 − 1,0 + 0,3 = −0,7
y = ReLU(−0,7) = max(0, −0,7) = 0
```

Neuron ini **"mati"** untuk masukan tersebut: keluarannya nol.

### Lapisan: banyak neuron sekaligus

Sebuah **lapisan linear** (disebut juga *dense* atau *fully connected*) dengan `n` masukan dan `m` keluaran:

```
y = f(xW + b),   W ∈ ℝ^{n×m},   b ∈ ℝ^m
```

Setiap **kolom** `W` adalah bobot satu neuron. Menulisnya sebagai matriks berarti menghitung `m` neuron sekaligus dengan **satu** perkalian matriks. Jumlah parameter: `n × m` (bobot) `+ m` (bias).

**Contoh dari mini-LLM.** Lapisan pertama FFN punya `n = 4, m = 8`, dengan `W1 = 0,5 ×` matriks tanda:

```
        ⎡  1  0  1 −1  0  1 −1  0 ⎤
W1 = 0,5⎢  0  1 −1  1  1  0  0 −1 ⎥
        ⎢  1  1  0  0 −1  1  0  1 ⎥
        ⎣ −1  0  1  1  0 −1  1  0 ⎦
```

Masukan (keluaran LayerNorm untuk token *atas*, dari Bab 16): `h = [−0,8793, −1,0919, 0,7721, 1,1991]`. Hitung `z = hW1` (bias nol). Kolom pertama `W1` adalah `0,5 × [1, 0, 1, −1] = [0,5, 0, 0,5, −0,5]`:

```
z₁ = (−0,8793)(0,5) + (−1,0919)(0) + (0,7721)(0,5) + (1,1991)(−0,5)
   = −0,4397 + 0 + 0,3861 − 0,5996 = −0,6531
```

Hasil lengkap kedelapan komponennya:

```
z      = [−0,6531, −0,1599, 0,7058, 0,4933, −0,9320, −0,6531, 1,0392, 0,9320]
ReLU(z)= [ 0,      0,      0,7058, 0,4933,  0,      0,      1,0392, 0,9320]
```

Empat dari delapan neuron **mati** — ini normal: pada jaringan ReLU terlatih, sebagian besar neuron tidak aktif untuk sebagian besar masukan.

### Kenapa harus ada fungsi non-linear

Tanpa aktivasi, dua lapisan linear murni:

```
y = (xW1)W2
```

Karena perkalian matriks asosiatif, `W1W2` bisa diringkas menjadi satu matriks `W3`:

```
W1 = ⎡1 2⎤   W2 = ⎡5 6⎤   →   W1W2 = ⎡1·5+2·7  1·6+2·8⎤ = ⎡19 22⎤ = W3
     ⎣3 4⎦        ⎣7 8⎦              ⎣3·5+4·7  3·6+4·8⎦   ⎣43 50⎦
```

Cek dengan `x = [1, 1]`:

- **Dua lapisan:** `xW1 = [4, 6]`, lalu `[4, 6]W2 = [4·5+6·7, 4·6+6·8] = [62, 72]`
- **Satu lapisan:** `xW3 = [19+43, 22+50] = [62, 72]` ✓

Identik. **Menumpuk 96 lapisan linear = 1 lapisan linear.** Kedalaman tidak memberi apa-apa.

Sekarang tambahkan ReLU di tengah, `y = ReLU(xW1)W2`, dengan `x = [1, −1]`:

```
xW1 = [1−3, 2−4] = [−2, −2]  →  ReLU → [0, 0]  →  hasil akhir [0, 0]
Tanpa ReLU: [−2, −2]W2 = [−24, −28]
```

Kedua hasil berbeda dan **tidak ada matriks tunggal `W3`** yang bisa mereproduksi perilaku versi ReLU untuk semua masukan. Inilah satu-satunya alasan fungsi aktivasi ada.

### Katalog fungsi aktivasi

**ReLU**

```
ReLU(x) = max(0, x)          ReLU′(x) = { 1  x > 0
                                        { 0  x < 0
```

Di `x = 0` turunannya tak terdefinisi secara matematis; implementasi biasanya memakai 0. *Kelebihan:* murah, turunannya 1 untuk `x > 0` sehingga gradien tidak mengecil di lapisan dalam. *Kekurangan:* untuk `z < 0` neuron tidak menerima gradien sama sekali dari contoh itu; bila kondisinya menetap untuk hampir semua data, neuron berhenti belajar (*dying ReLU*).

**Sigmoid**

```
σ(x) = 1 / (1 + e^{−x})      σ′(x) = σ(x)(1 − σ(x))
```

| x | σ(x) | σ′(x) |
|---|---|---|
| −2 | 0,1192 | 0,1050 |
| −1 | 0,2689 | 0,1966 |
| 0 | 0,5000 | 0,2500 |
| 1 | 0,7311 | 0,1966 |
| 2 | 0,8808 | 0,1050 |
| 5 | 0,9933 | 0,0066 |

Turunan maksimumnya hanya 0,25: setelah 10 lapisan, gradien dikalikan paling banyak `0,25¹⁰ = 9,5 × 10⁻⁷`. Karena itu sigmoid **bukan** aktivasi utama lapisan dalam. Ia tetap hadir di dalam SiLU (`x·σ(x)`) sebagai gerbang, bukan sebagai aktivasi seluruh lapisan.

**Tanh**

```
tanh(x) = (e^x − e^{−x}) / (e^x + e^{−x})     tanh′(x) = 1 − tanh²(x)
```

Keluaran di `(−1, 1)`, turunan maksimum 1 (di `x = 0`). Lebih baik dari sigmoid, tetapi tetap jenuh di ujung.

**GELU** (Gaussian Error Linear Unit, Hendrycks & Gimpel 2016):

```
GELU(x) = x · Φ(x)
```

dengan `Φ(x)` = fungsi distribusi kumulatif normal standar. Hampiran lazim yang dipakai:

```
GELU(x) ≈ 0,5x (1 + tanh[ √(2/π) (x + 0,044715x³) ])
```

Ini aktivasi GPT-2, GPT-3, dan BERT.

**SiLU/Swish**

```
SiLU(x) = x · σ(x) = x / (1 + e^{−x})
```

Bentuk sangat mirip GELU; dipakai sebagai komponen SwiGLU di Llama.

**Perbandingan angka semua aktivasi**

| x | ReLU | GELU (hampiran) | GELU (eksak) | Sigmoid | Tanh | SiLU |
|---|---|---|---|---|---|---|
| −2,0 | 0,0000 | −0,0454 | −0,0455 | 0,1192 | −0,9640 | −0,2384 |
| −1,0 | 0,0000 | −0,1588 | −0,1587 | 0,2689 | −0,7616 | −0,2689 |
| −0,5 | 0,0000 | −0,1543 | −0,1543 | 0,3775 | −0,4621 | −0,1888 |
| 0,0 | 0,0000 | 0,0000 | 0,0000 | 0,5000 | 0,0000 | 0,0000 |
| 0,5 | 0,5000 | 0,3457 | 0,3457 | 0,6225 | 0,4621 | 0,3112 |
| 1,0 | 1,0000 | 0,8412 | 0,8413 | 0,7311 | 0,7616 | 0,7311 |
| 2,0 | 2,0000 | 1,9546 | 1,9545 | 0,8808 | 0,9640 | 1,7616 |

Dua hal penting: (1) hampiran tanh untuk GELU akurat sampai ~4 angka desimal, jadi jauh lebih murah dari `Φ` eksak; (2) GELU **negatif** untuk `x` negatif kecil — berbeda dari ReLU yang langsung nol — sehingga neuron yang sedikit negatif masih punya gradien bukan-nol dan tetap bisa belajar.

### SwiGLU: aktivasi dengan gating

Model modern (Llama, PaLM) memakai varian berstruktur berbeda (Shazeer 2020):

```
SwiGLU(x) = SiLU(xW_gate) ⊙ (xW_up)
```

`⊙` = perkalian elemen-per-elemen. Perbedaan strukturalnya: ada **dua** matriks bobot di jalur naik, bukan satu. Satu diaktivasi dan berperan sebagai **gerbang** (gate), lalu dikalikan dengan yang lain.

**Contoh angka.** `xW_gate = [2,0, −1,0, 0,5]`, `xW_up = [1,0, 3,0, −2,0]`:

| Komponen | SiLU(gate) | up | Hasil ⊙ |
|---|---|---|---|
| 1 | 2,0 × 0,8808 = 1,7616 | 1,0 | 1,7616 |
| 2 | −1,0 × 0,2689 = −0,2689 | 3,0 | −0,8067 |
| 3 | 0,5 × 0,6225 = 0,3112 | −2,0 | −0,6224 |

```
SwiGLU(x) = [1,7616, −0,8067, −0,6224]
```

**Konsekuensi parameter.** FFN biasa punya 2 matriks; SwiGLU punya 3 (`W_gate, W_up, W_down`). Agar total parameter setara, `d_ff` dikurangi: Llama memakai `d_ff = (2/3) × 4d` lalu dibulatkan; untuk Llama 2 7B (`d = 4096`) → `d_ff = 11008`.

```
FFN standar : 2 × 4096 × 16384 = 134,2 juta
SwiGLU      : 3 × 4096 × 11008 = 135,3 juta   (hampir sama)
```

### Feed-forward network di Transformer

FFN di Transformer asli (Vaswani dkk. 2017):

```
FFN(x) = ReLU(xW1 + b1)W2 + b2,   W1 ∈ ℝ^{d×d_ff},   W2 ∈ ℝ^{d_ff×d}
```

Strukturnya: **naik** ke dimensi lebih besar → aktivasi → **turun** kembali ke dimensi semula. Konvensi standarnya `d_ff = 4d` (Transformer asli: `d = 512, d_ff = 2048`; GPT-3: `d = 12288, d_ff = 49152`).

```
Parameter FFN = d·d_ff + d_ff + d_ff·d + d = 2·d·d_ff + d_ff + d
```

Untuk GPT-3 per lapisan: `2 × 12288 × 49152 ≈ 1,208 miliar` parameter — dua pertiga dari total ~1,812 miliar per lapisan; attention hanya sepertiga.

**Sifat penting:** FFN bekerja pada **setiap posisi token secara terpisah dan identik**. Token di posisi 1 dan posisi 500 melewati matriks yang sama dan tidak saling bertukar informasi di dalam FFN. Semua pertukaran antarposisi terjadi di attention (Bab 13).

| Komponen | Fungsi |
|---|---|
| Attention | memindahkan informasi antarposisi |
| FFN | memproses informasi di dalam satu posisi |

## Implementasi

- [`src/llmbepees/np/activation.py`](../../src/llmbepees/np/activation.py): `relu`/`sigmoid`/`tanh` (dipakai ulang dari `calculus.py` Bab 7), ditambah `gelu`, `gelu_approx`, `silu`, turunannya, dan registri `ACTIVATIONS`/`get_activation`.
- [`src/llmbepees/np/neuron.py`](../../src/llmbepees/np/neuron.py): `pre_activation`, `neuron`, `linear`, `dense`, `layer_parameters`, `collapse_linear_layers`, `ffn`, `ffn_parameters`, `swiglu`, `ffn_swiglu`.
- [`tests/test_activation.py`](../../tests/test_activation.py) & [`tests/test_neuron.py`](../../tests/test_neuron.py): semua contoh angka di atas.

## Inti bab

FFN adalah komponen yang paling jarang dibicarakan padahal memuat sekitar dua pertiga parameter LLM. Satu neuron menjumlahkan masukan berbobot lalu melewatkannya ke fungsi non-linear; satu lapisan = banyak neuron dihitung sekaligus lewat satu perkalian matriks; FFN = dua lapisan seperti itu (naik lalu turun).

Alasan keberadaan seluruh bab ini: **tanpa aktivasi di tengah, kedua proyeksi FFN (`W1` lalu `W2`) runtuh menjadi satu proyeksi tunggal `W1W2`, sehingga lebar `d_ff` menjadi sia-sia**. Catatan jujur: yang runtuh persis adalah FFN-nya — attention, softmax, dan LayerNorm memang sudah non-linear. Aktivasilah yang membuat FFN benar-benar menambah kemampuan.

## Cek pemahaman

1. Mengapa ReLU, yang hanya mengubah angka negatif menjadi nol, sudah cukup sebagai fungsi non-linear?
2. Mengapa FFN melebarkan representasi ke `d_ff` lalu menyempitkannya kembali ke `d`?
3. Apa yang tidak bisa dilakukan FFN, dan komponen mana yang mengerjakannya?

## Rujukan

- Buku *"Memahami Matematika di Balik LLM"*, Bab 9.
- Hendrycks & Gimpel (2016) — GELU; Shazeer (2020) — GLU/SwiGLU; Vaswani dkk. (2017) — FFN Transformer; Touvron dkk. (2023b) — `d_ff` Llama 2.
