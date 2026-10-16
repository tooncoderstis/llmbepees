# Bab 15 — Informasi Posisi: Positional Encoding dan RoPE

> Catatan belajar, ditulis ulang dengan bahasa sendiri dari buku *"Memahami Matematika di Balik LLM"* (Bab 15).
> Bagian dari **Fase 8 — Informasi Posisi**. Kode: [`src/llmbepees/np/position.py`](../../src/llmbepees/np/position.py) · Uji: [`tests/test_position.py`](../../tests/test_position.py).

## Masalah

Skor kecocokan attention (Bab 13) adalah `S_ij = q_i · k_j`, dan keluarannya `o_i = ∑_j A_ij v_j` — **rata-rata berbobot dari value**. Tidak ada satu suku pun yang mengandung `i` atau `j` sebagai **angka posisi**. Akibatnya, attention hanya membandingkan **isi**, bukan **urutan**.

**Klaim.** Attention tanpa causal mask bersifat **equivarian terhadap permutasi**: jika urutan baris `H` diacak, baris keluaran ikut teracak dengan cara yang sama, tetapi isinya tidak berubah. Akibatnya, tanpa perbaikan, kedua kalimat ini diproses **identik**:

```
kucing menggigit anjing
anjing menggigit kucing
```

**Bukti angka.** `h_A = [1, 0]`, `h_B = [0, 1]`, matriks identitas, tanpa causal mask. Dengan `√d_k = √2 = 1,4142`, skor baris A menjadi `0,7071` dan `0`, softmax-nya `[0,6697, 0,3303]`:

```
o_A = 0,6697[1, 0] + 0,3303[0, 1] = [0,6697, 0,3303]
```

Sekarang tukar urutannya (B, A). Bobotnya sama, hanya urutan kolom bertukar, dan:

```
o_A = 0,3303[0, 1] + 0,6697[1, 0] = [0,6697, 0,3303]
```

**Identik.** Padahal posisi A berubah dari pertama ke kedua.

**Bagaimana dengan causal mask?** Causal mask (Bab 13) memang menyalurkan sebagian sinyal posisi secara tidak langsung (jumlah posisi yang boleh dilihat tiap baris berbeda). Tetapi ia **tidak menyatakan jarak**: dalam satu baris, "3 token lalu" dan "50 token lalu" diperlakukan sama persis.

## Gagasan

### Tiga hal yang mudah tertukar

| Informasi | Pertanyaan | Sumber |
|---|---|---|
| **Identitas token** | "kata apa ini?" | tabel embedding (Bab 8) — tetap, di mana pun muncul |
| **Posisi token** | "ini kata keberapa?" | tidak bergantung pada kata yang ada di sana |
| **Representasi gabungan** | keduanya | inilah yang benar-benar masuk ke attention |

Contoh: identitas `[1,0, 0,0, 1,0, 0,0]` + posisi `[0,0, 0,5, 0,0, 0,0]` = representasi `[1,0, 0,5, 1,0, 0,0]`. Seluruh bab ini membahas satu pertanyaan: **bagaimana baris kedua dimasukkan ke baris ketiga**. Ada empat jawaban.

### Jawaban 1 — positional embedding yang dipelajari

Buat tabel embedding kedua, satu baris per posisi, lalu jumlahkan:

```
H(0) = X + P
```

`X` = embedding token (`n×d`), `P` = embedding posisi (`n_max×d`), `P` berisi **parameter yang dilatih**. Contoh mini-LLM:

| Posisi | p1 | p2 | p3 | p4 |
|---|---|---|---|---|
| 0 | 0,5 | 0,0 | 0,0 | 0,0 |
| 1 | 0,0 | 0,5 | 0,0 | 0,0 |
| 2 | 0,0 | 0,0 | 0,5 | 0,0 |
| 3 | 0,0 | 0,0 | 0,0 | 0,5 |
| 4 | 0,5 | 0,5 | 0,0 | 0,0 |

Token *kucing* `[1,0, 0,0, 1,0, 0,0]` di posisi 1 menjadi `[1,0, 0,5, 1,0, 0,0]`; di posisi 3 menjadi `[1,0, 0,0, 1,0, 0,5]` — vektor berbeda, sehingga attention bisa membedakannya. Dipakai BERT, GPT-2, GPT-3.

**Keterbatasan:** tabel `P` punya jumlah baris **tetap** (`n_max×d` parameter; GPT-3: `2048 × 12288 = 25,2 juta`). Untuk token ke-2049 tidak ada baris yang bisa diambil, sehingga model tidak bisa langsung dipakai pada konteks lebih panjang daripada saat dilatih.

### Jawaban 2 — sinusoidal positional encoding

Hitung posisi dengan rumus, bukan simpan dalam tabel. Untuk posisi `pos` dan indeks pasangan `i`:

```
PE(pos, 2i)   = sin( pos / 10000^{2i/d} )
PE(pos, 2i+1) = cos( pos / 10000^{2i/d} )
```

Dimensi genap pakai sinus, ganjil pakai kosinus, pasangan `(2i, 2i+1)` memakai frekuensi sama. Contoh `d = 4` (pembagi `i=0`: 1; `i=1`: 100):

| pos | sin(pos/1) | cos(pos/1) | sin(pos/100) | cos(pos/100) |
|---|---|---|---|---|
| 0 | 0,0000 | 1,0000 | 0,0000 | 1,0000 |
| 1 | 0,8415 | 0,5403 | 0,0100 | 1,0000 |
| 2 | 0,9093 | −0,4161 | 0,0200 | 0,9998 |
| 3 | 0,1411 | −0,9900 | 0,0300 | 0,9996 |
| 4 | −0,7568 | −0,6536 | 0,0400 | 0,9992 |

Dua kolom pertama berubah cepat (periode `2π ≈ 6,28` posisi), dua kolom terakhir lambat (periode `200π ≈ 628`). Untuk `d` besar, panjang gelombangnya membentuk deret geometri dari `2π` sampai `10000·2π`:

| Pasangan `i` | Pembagi (`d = 512`) | Panjang gelombang |
|---|---|---|
| 0 | 1,0 | 6,3 posisi |
| 64 | 10,0 | 62,8 posisi |
| 128 | 100,0 | 628 posisi |
| 192 | 1000,0 | 6.283 posisi |
| 255 | ≈9.647 | ≈60.600 posisi |

Dimensi berfrekuensi tinggi menangkap posisi tepat; frekuensi rendah menangkap posisi kasar dalam dokumen panjang. **Kelebihan:** tidak ada batas pada perhitungannya — `PE` untuk posisi 100.000 bisa dihitung dengan rumus yang sama, tanpa parameter tambahan. (Catatan jujur: dapat dihitung **tidak sama dengan** bekerja baik; model yang dilatih pada konteks pendek tetap menghadapi sebaran masukan yang asing.)

Ada identitas trigonometri `sin(a+b) = sin a cos b + cos a sin b` (dan serupa untuk cos), sehingga `PE_{pos+k}` adalah kombinasi linear `PE_pos` dengan koefisien yang hanya bergantung pada `k` — hipotesis mengapa model bisa belajar memperhatikan posisi **relatif**.

### Jawaban 3 — RoPE (Rotary Position Embedding)

Dua jawaban sebelumnya **menambahkan** posisi ke embedding **sekali** di awal, dengan dua akibat: informasi posisi bisa memudar setelah banyak lapisan, dan yang dikodekan adalah posisi **absolut** — padahal yang penting bagi bahasa adalah **jarak**. RoPE (Su dkk., 2021) menjawab keduanya: posisi tidak ditambahkan, melainkan **q dan k diputar** dengan sudut sebanding posisi. Dipakai Llama, Mistral, Qwen, dan sebagian besar LLM open-weight modern.

Untuk pasangan komponen `(q_{2i}, q_{2i+1})` dan posisi `m`:

```
[q'_{2i}  ]   [ cos(mθ_i)  −sin(mθ_i) ] [ q_{2i}   ]
[q'_{2i+1}] = [ sin(mθ_i)   cos(mθ_i) ] [ q_{2i+1} ]
```

dengan `θ_i = 10000^{−2i/d}`. Matriks `2×2` itu adalah **matriks rotasi**, yang memutar vektor tanpa mengubah panjangnya.

**Contoh** (`d = 4`, `q = [1, 0, 1, 0]`; `θ_0 = 1`, `θ_1 = 0,01`):

| Posisi | Pasangan 1 | Pasangan 2 |
|---|---|---|
| 0 | [1,0000, 0,0000] | [1,0000, 0,0000] |
| 1 | [0,5403, 0,8415] | [1,0000, 0,0100] |
| 2 | [−0,4161, 0,9093] | [0,9998, 0,0200] |

Cek `m = 1` pasangan 1: `q'_1 = 1(0,5403) − 0(0,8415) = 0,5403`; `q'_2 = 1(0,8415) + 0(0,5403) = 0,8415`. Panjang sebelum `√(1+0) = 1`, sesudah `√(0,5403² + 0,8415²) = 1` — **tidak berubah**, sesuai sifat rotasi.

**Sifat kunci.** Dot product antara query posisi `m` dan key posisi `n` hanya bergantung pada **selisih** `m − n`:

| m | n | m − n | q'_m · k'_n |
|---|---|---|---|
| 0 | 0 | 0 | 2,0000 |
| 1 | 0 | 1 | 1,5403 |
| 2 | 0 | 2 | 0,5837 |
| 3 | 1 | 2 | 0,5837 |
| 5 | 3 | 2 | 0,5837 |

Tiga baris terakhir punya `m` dan `n` sangat berbeda tetapi selisih sama, dan hasilnya **identik**. **Alasan matematisnya:** rotasi `mθ` diikuti rotasi balik `nθ` menghasilkan rotasi bersih `(n − m)θ`:

```
R(mθ)ᵀ R(nθ) = R((n − m)θ)
```

Karena `q'_m · k'_n = qᵀ R(mθ)ᵀ R(nθ) k = qᵀ R((n−m)θ) k`, hasilnya hanya bergantung `n − m`.

| Sifat | Positional terlatih | Sinusoidal | RoPE |
|---|---|---|---|
| Parameter tambahan | `n_max × d` | 0 | 0 |
| Ada batas panjang pada rumusnya | ya | tidak | tidak |
| Posisi relatif eksplisit | tidak | tidak langsung | ya |
| Diterapkan di | embedding awal | embedding awal | setiap Q, K di tiap lapisan |

Baris terakhir penting: RoPE diterapkan pada `q`/`k` **setiap lapisan**, bukan sekali di awal. (Jujur: ini alasan perancangan yang masuk akal, bukan jaminan matematis bahwa informasi posisi tidak memudar.) Untuk `d = 128`, `θ` berkisar dari `θ_0 = 1` sampai `θ_63 ≈ 1,2 × 10⁻⁴`. **Memperpanjang konteks:** *position interpolation* menskalakan posisi sehingga posisi 8192 diperlakukan seperti 2048 (biasanya tetap butuh sedikit fine-tuning).

### Jawaban 4 — ALiBi

Jangan sentuh embedding sama sekali; tambahkan penalti langsung ke skor attention:

```
S_ij = q_i·k_j / √d_k − m_h·(i − j)
```

Penalti ditambahkan **setelah** pembagian `√d_k`, dan penaltinya sendiri tidak ikut dibagi — ALiBi menumpang di atas penskalaan Bab 13, bukan menggantikannya. `m_h` adalah konstanta tetap per head (tidak dilatih). Contoh head `m_h = 0,25`, query di posisi 10:

| Key di posisi | Jarak `i−j` | Penalti | Skor mentah | Skor akhir |
|---|---|---|---|---|
| 10 | 0 | 0,00 | 3,0 | 3,00 |
| 9 | 1 | 0,25 | 3,0 | 2,75 |
| 6 | 4 | 1,00 | 3,0 | 2,00 |
| 0 | 10 | 2,50 | 3,0 | 0,50 |

Makin jauh, makin dihukum; karena penaltinya linear tanpa batas, model bisa dipakai pada konteks lebih panjang. Head yang berbeda diberi `m_h` berbeda (deret geometri seperti 1/2, 1/4, 1/8, …), sehingga sebagian head fokus jarak dekat dan sebagian bisa melihat jauh.

### Perbandingan keempat jawaban

| | Learned PE | Sinusoidal | RoPE | ALiBi |
|---|---|---|---|---|
| Ditambahkan ke | embedding | embedding | rotasi Q,K | skor attention |
| Parameter | `n_max·d` | 0 | 0 | 0 |
| Ekstrapolasi | tidak secara bawaan | terbatas | baik (dengan penyesuaian) | baik |
| Posisi relatif | implisit | implisit | eksplisit | eksplisit |
| Dipakai | GPT-2/3, BERT | Transformer asli | Llama, Mistral, Qwen | BLOOM, MPT |

## Implementasi

- [`src/llmbepees/np/position.py`](../../src/llmbepees/np/position.py): `add_position` (learned), `sinusoidal_encoding`, `rope_theta`/`rope_rotate`/`rope_apply`, `alibi_bias`, `position_interpolation`.
- Uji memverifikasi tabel sinusoidal `d=4`, contoh rotasi RoPE (termasuk panjang yang tetap), sifat "hanya bergantung `m−n`", tabel ALiBi, dan `θ_63` untuk `d=128`.

## Inti bab

Bab ini dibuka dengan bukti yang mengganggu: **attention sama sekali tidak melihat urutan**. Ini bukan kelalaian implementasi, melainkan sifat matematis yang bisa dibuktikan — skor hanya memakai isi vektor, dan penjumlahan di akhir tidak peduli urutan suku. Tanpa perbaikan, *kucing menggigit anjing* dan *anjing menggigit kucing* diproses identik.

Empat metode menjawab pertanyaan yang sama di **tempat berbeda**: dua yang pertama mengubah embedding (learned `X+P`, sinusoidal), RoPE mengubah query/key di **setiap** lapisan, dan ALiBi mengubah skor. RoPE — yang paling luas dipakai model open-weight modern — memutar query dan key, sehingga hasil dot product hanya bergantung pada **selisih posisi**, bukan posisi absolutnya.

## Cek pemahaman

1. Mengapa causal mask saja tidak cukup untuk memberi model informasi posisi?
2. Apa keuntungan menghitung informasi posisi dengan rumus dibanding menyimpannya dalam tabel?
3. Mengapa RoPE diterapkan di setiap lapisan, sedangkan positional embedding hanya sekali di awal?

## Rujukan

- Buku *"Memahami Matematika di Balik LLM"*, Bab 15.
- Vaswani dkk. (2017) — sinusoidal encoding; Su dkk. (2021) — RoPE; Press dkk. (2022) — ALiBi; Chen dkk. (2023) — position interpolation.
