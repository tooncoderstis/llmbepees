# Bab 7 — Turunan, Gradien, dan Aturan Rantai

> Catatan belajar, ditulis ulang dengan bahasa sendiri dari buku *"Memahami Matematika di Balik LLM"* (Bab 7).
> Bagian dari **Fase 1 — Fondasi Matematika**. Kode: [`src/llmbepees/np/calculus.py`](../../src/llmbepees/np/calculus.py) · Uji: [`tests/test_calculus.py`](../../tests/test_calculus.py).

## Masalah

Bab 6 menghasilkan sebuah angka: **loss**. Bab ini menjawab pertanyaan berikutnya: bagaimana caranya mengubah 174 miliar parameter supaya angka itu turun? Jawabannya satu kata: **turunan**.

## Gagasan

### Turunan: definisi

```
f'(x) = df/dx = lim_{h→0} [f(x + h) − f(x)] / h
```

Turunan mengukur: jika `x` dinaikkan sedikit, `f(x)` naik berapa kali lipat dari kenaikan itu. Contoh numerik `f(x) = x²` di `x = 3`, dengan `h` makin kecil:

| h | f(3+h) | selisih | selisih/h |
|---|---|---|---|
| 1,000 | 16,0000 | 7,0000 | 7,0000 |
| 0,100 | 9,6100 | 0,6100 | 6,1000 |
| 0,010 | 9,0601 | 0,0601 | 6,0100 |
| 0,001 | 9,0060 | 0,0060 | 6,0010 |

Nilainya menuju **6**. Secara aljabar `f'(x) = 2x`, dan `2 × 3 = 6` — cocok. Tafsirannya: jika `x` naik `0,001`, `f(x)` naik sekitar `6 × 0,001 = 0,006`. Turunan adalah **faktor pengali** antara perubahan masukan dan perubahan keluaran.

### Aturan turunan yang dipakai di LLM

| No | Fungsi | Turunan |
|---|---|---|
| 1 | `f(x) = c` | `0` |
| 2 | `f(x) = xⁿ` | `n x^{n−1}` |
| 3 | `f(x) = eˣ` | `eˣ` (turunannya dirinya sendiri) |
| 4 | `f(x) = ln x` | `1/x` |
| 5 | `f = g + h` | `g' + h'` |
| 6 | `f = c·g` | `c·g'` |

Contoh gabungan `f(x) = 3x² − 4x + 7` → `f'(x) = 6x − 4`; di `x = 2`: `f'(2) = 8`.

### Kenapa turunan bisa mencari nilai minimum

Fakta: `f'(x) > 0` → fungsi naik; `f'(x) < 0` → turun; `f'(x) = 0` → datar (bisa minimum/maksimum/belok). Untuk menurunkan `f`, geser `x` **berlawanan** tanda turunannya:

```
x_baru = x_lama − η · f'(x_lama)
```

`η` (eta) = **learning rate**. Tanda minus bekerja di kedua kasus: fungsi naik (`f'` positif) → geser ke kiri; fungsi turun (`f'` negatif) → geser ke kanan.

Contoh meminimalkan `f(w) = (w − 3)²`, mulai `w = 0`, `η = 0,1`: `w` merambat `0 → 0,6 → 1,08 → 1,464 → …` menuju 3, dan `f(w)` terus turun. Prosedur ini **gradient descent** (Bab 12 membahasnya penuh).

**Bahaya learning rate terlalu besar:** dengan `η = 1,1` pada fungsi sama, `w` melompat `0 → 6,6 → −1,32 → 8,184 → …` dan loss **meledak**. Ini fenomena nyata di pelatihan LLM (kurva loss melonjak lalu NaN).

### Turunan parsial

Fungsi di LLM punya miliaran variabel, jadi butuh turunan parsial: turunan terhadap satu variabel dengan **variabel lain dianggap konstanta** (`∂`, dibaca "del"/"parsial").

Contoh `f(x,y) = x²y + 3y`:

```
∂f/∂x = 2xy
∂f/∂y = x² + 3
```

Di titik `(x,y) = (2,5)`: `∂f/∂x = 20`, `∂f/∂y = 7`. Tafsiran: menaikkan `x` sebesar `0,01` menaikkan `f` sekitar `0,20`; `x` tiga kali lebih berpengaruh daripada `y` di titik ini.

### Gradien

Gradien = vektor semua turunan parsial:

```
∇f = [∂f/∂x₁, ∂f/∂x₂, …, ∂f/∂xₙ]
```

Untuk contoh di atas, `∇f(2,5) = [20, 7]`. **Sifat kunci:** `∇f` menunjuk arah `f` naik paling cepat, dan `−∇f` arah turun paling cepat. Sifat ini berlaku **lokal** — dijamin menurunkan `f` hanya jika langkahnya cukup kecil. Dasar aturan pembaruan:

```
θ_baru = θ_lama − η ∇_θ L
```

Di LLM, `θ` = seluruh parameter; untuk GPT-3, `∇_θ L` punya 175 miliar komponen, masing-masing menjawab "jika parameter ini dinaikkan sedikit, loss naik atau turun, seberapa kuat?"

Contoh mini-LLM: 188 parameter → gradien vektor 188 angka. Statistik gradien langkah pertama (loss awal `2,4480`): norma `1,0442`, komponen terbesar `0,4703`. Pola penting: gradien pada embedding jauh lebih besar daripada pada `W_Q` (`0,3420` vs `0,0055`, ~60×). Artinya mengubah embedding jauh lebih efektif menurunkan loss di awal — ketimpangan skala seperti inilah yang membuat optimizer modern (Adam, Bab 12) menyesuaikan langkah per-parameter.

### Aturan rantai: mesin di balik backpropagation

**Satu variabel**: jika `z = f(y)` dan `y = g(x)`:

```
dz/dx = (dz/dy) · (dy/dx)
```

Turunan menembus lapisan dengan cara **dikalikan**. Contoh `y = 3x + 1`, `z = y²`, cari `dz/dx` di `x = 2`: `dy/dx = 3`, `dz/dy = 2y = 14`, jadi `dz/dx = 14 × 3 = 42`. Verifikasi langsung: `z = (3x+1)² = 9x² + 6x + 1` → `dz/dx = 18x + 6 = 42` ✓. Verifikasi numerik (`h = 0,0001`) juga memberi `42,0009`.

**Banyak lapisan**: untuk rantai `x → y₁ → … → y_k → L`:

```
∂L/∂x = (∂L/∂y_k) · (∂y_k/∂y_{k−1}) · … · (∂y₂/∂y₁) · (∂y₁/∂x)
```

Nilai mengalir **maju** (forward), gradien merambat **mundur** (backward) sambil mengalikan turunan lokal di setiap simpul — inilah **backpropagation** (Rumelhart dkk., 1986). Catatan notasi: di jaringan sungguhan, `y_k` vektor dan "turunan lokal" berupa matriks **Jacobian**; aturan rantainya tetap, hanya menjadi perkalian matriks-vektor (Bab 11 memberi bentuk matriksnya).

**Percabangan**: jika `x` memengaruhi `L` lewat beberapa jalur, gradiennya **dijumlahkan**: `∂L/∂x = ∑_j (∂L/∂y_j)(∂y_j/∂x)`. Ini muncul di residual connection (Bab 16) — sebuah nilai dipakai dua kali (jalur utama dan jalur pintas).

### Turunan fungsi yang dipakai LLM

| Fungsi | Rumus | Turunan |
|---|---|---|
| ReLU | `max(0, x)` | `1` jika `x > 0`, `0` jika `x < 0` |
| Sigmoid | `σ(x) = 1/(1+e^{−x})` | `σ(x)(1 − σ(x))` |
| Tanh | `tanh(x)` | `1 − tanh²(x)` |
| Kuadrat | `x²` | `2x` |
| Log | `ln x` | `1/x` |
| Eksponen | `eˣ` | `eˣ` |

Contoh sigmoid: `σ(0) = 0,5` → turunan `0,25` (maksimum). `σ(5) = 0,9933` → turunan `0,0067`. Jika 10 lapisan sigmoid ditumpuk, faktor aktivasi saja sudah `0,0067¹⁰ ≈ 1,8 × 10^{−22}` — praktis nol. Gejala ini **vanishing gradient** — alasan matematis LLM tidak memakai sigmoid di lapisan dalam (Bab 9) dan mengapa residual connection diperlukan (Bab 16).

### Turunan terhadap vektor dan matriks

Jika `y = xW`:

```
∂L/∂W = xᵀ (∂L/∂y),     ∂L/∂x = (∂L/∂y) Wᵀ
```

Contoh `x = [2, 3]`, `∂L/∂y = [0,5, −1,0]` → `∂L/∂W = [[1,0, −2,0], [1,5, −3,0]]`. Cek manual: `y₁ = x₁W₁₁ + x₂W₂₁`, jadi `∂y₁/∂W₁₁ = x₁ = 2`, sehingga `∂L/∂W₁₁ = 0,5 × 2 = 1,0` ✓. **Ukuran matriks gradien selalu sama dengan ukuran objek yang diturunkan** — cara cepat mengecek rumus gradien.

### Gradien numerik: cara memverifikasi

```
∂L/∂θ_i ≈ [L(θ_i + ε) − L(θ_i − ε)] / (2ε)     (ε kecil, mis. 10⁻⁵)
```

Ini disebut **central difference**. Contoh: gradien loss terhadap logit seharusnya `p − y`. Dengan `z = [2,0, 1,0, 0,1]`, label kelas 1, softmax `p = [0,6590, 0,2424, 0,0986]`, dan `y = [1,0,0]`: rumus `p − y = [−0,3410, 0,2424, 0,0986]` cocok dengan hitungan numerik sampai galat pembulatan.

Metode ini terlalu lambat untuk model sungguhan: untuk `N` parameter butuh `2N` evaluasi model (GPT-3: 350 miliar forward pass untuk satu langkah). Backpropagation menghasilkan seluruh gradien dengan biaya sekitar **dua** forward pass. Untuk mini-LLM 188 parameter, gradien numerik masih layak dan menjadi alat verifikasi utama (Bab 21).

## Implementasi

- [`src/llmbepees/np/calculus.py`](../../src/llmbepees/np/calculus.py): `central_difference`, `numerical_derivative`, `numerical_gradient`, `gradient_descent_step`, `relu`, `relu_derivative`, `sigmoid`, `sigmoid_derivative`, `tanh_derivative`, `linear_backward`.
- [`tests/test_calculus.py`](../../tests/test_calculus.py): semua contoh angka di atas + verifikasi `∂L/∂z = p − y` secara numerik.

## Inti bab

1. Turunan menjawab satu pertanyaan spesifik: jika parameter ini dinaikkan sedikit, loss naik atau turun, dan seberapa kuat.
2. Gradien adalah kumpulan jawaban itu untuk semua parameter sekaligus, dan ia menunjuk arah loss **naik** paling cepat — jadi kita bergerak ke arah berlawanan.
3. Aturan rantai memungkinkan jawaban dihitung menembus lapisan berapa pun; itulah yang membuat pelatihan model dalam mungkin.

## Cek pemahaman

1. Mengapa aturan pembaruan parameter memakai tanda minus?
2. Apa yang dilakukan aturan rantai, dan mengapa ia penting untuk model berlapis?
3. Apa yang terjadi pada gradien di titik percabangan, misalnya residual connection?

## Rujukan

- Buku *"Memahami Matematika di Balik LLM"*, Bab 7.
- Rumelhart dkk. (1986) — backpropagation.
