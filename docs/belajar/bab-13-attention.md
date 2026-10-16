# Bab 13 — Attention: Query, Key, Value

> Catatan belajar, ditulis ulang dengan bahasa sendiri dari buku *"Memahami Matematika di Balik LLM"* (Bab 13).
> Bagian dari **Fase 7 — Attention**. Kode: [`src/llmbepees/np/attention.py`](../../src/llmbepees/np/attention.py) · Uji: [`tests/test_attention.py`](../../tests/test_attention.py).
>
> **Bab inti buku.** Kalau ada satu bagian yang layak dibaca dua kali, ini bagiannya.

## Masalah

Sampai Bab 12, tiap token sudah menjadi deretan angka, tetapi tidak ada satu pun mekanisme yang memindahkan informasi **antarposisi**. FFN (Bab 9) mengolah setiap posisi terpisah: token di posisi 5 diproses tanpa melihat apa pun di posisi 2. Padahal perhatikan kalimat:

> Kucing itu duduk di atas meja karena **ia** lelah.

Kata *ia* merujuk pada *kucing*, tujuh token sebelumnya. Untuk memprosesnya, model harus mengambil informasi dari posisi 1.

## Gagasan

Mekanisme yang dibutuhkan punya tiga syarat: (1) pemilihannya bergantung pada **isi** token, bukan jarak tetap; (2) **bisa diturunkan** (diperbaiki lewat gradient descent); (3) bisa dihitung untuk semua posisi **sekaligus** dalam bentuk perkalian matriks. Mekanisme itu bernama **attention** (Bahdanau dkk., 2015).

### Tiga peran: apa yang dicari, ditawarkan, diambil

| Vektor | Lambang | Peran dalam perhitungan |
|---|---|---|
| Query | `q` | Dipakai token ini untuk **mencari** |
| Key | `k` | Dipakai token lain untuk **dicocokkan** |
| Value | `v` | Isi yang **diambil** jika cocok |

Ketiganya dibuat dari vektor yang sama (representasi token) hanya dengan tiga matriks parameter berbeda:

```
Q = H·W_Q,   K = H·W_K,   V = H·W_V
```

`H` berukuran `n × d` (satu baris per token). Nama "query/key/value" hanyalah **label**, bukan sifat matematis; yang membedakan peran mereka adalah **posisinya dalam rumus**: `Q` dan `K` masuk ke perhitungan skor, `V` dikalikan dengan skor yang sudah menjadi bobot.

**Contoh dua token** (`d = 2`, `W_Q = W_K = I`): `h1 = [1,0, 0,0]`, `h2 = [0,0, 1,0]`, maka `q1 = k1 = [1,0, 0,0]` dan `q2 = k2 = [0,0, 1,0]`. Dengan matriks identitas, query dan key sama persis dengan masukannya — sengaja, supaya terlihat bahwa perkalian ini **tidak** memindahkan apa pun antar token. Yang berubah baru cara setiap token menyatakan kebutuhannya.

### Skor kecocokan: dot product

Kecocokan setiap pasangan query–key dihitung dengan **dot product** (Bab 3), dan semuanya sekaligus lewat satu perkalian matriks:

```
S = Q·Kᵀ
```

`S` berukuran `n × n`: baris `i` kolom `j` = dot product query token `i` dengan key token `j`. Transpos `K` diperlukan agar dimensi tengah cocok (`n×d_k` dikali `d_k×n`).

### Mengubah skor menjadi bobot: softmax

Softmax (Bab 10) diterapkan **per baris**, karena setiap token membagi perhatiannya secara terpisah. Sebelum softmax ada dua penyesuaian:

**1. Dibagi `√d_k`.** Semakin besar `d_k`, semakin liar sebaran nilai dot product. Untuk komponen acak bebas (rata-rata 0, varians 1), variansnya tumbuh sebanding `d_k`:

| `d_k` | Varians terukur | Prediksi teori |
|---|---|---|
| 4 | 4,00 | 4 |
| 64 | 64,49 | 64 |
| 512 | 508,16 | 512 |

Softmax sangat sensitif terhadap skala (lihat Bab 10): logit `[3,0, 1,0, 0,5, −1,0]` ×8 menjadi praktis one-hot, sehingga model kehilangan kemampuan menggabungkan beberapa posisi dan gradiennya mati. Membagi dengan `√d_k` mengembalikan varians skor ke 1:

```
Var( q·k / √d_k ) = d_k / d_k = 1
```

Asumsi kebebasan itu masuk akal saat inisialisasi (bobot masih acak); ini alasan perancangan, bukan teorema tentang model terlatih.

**2. Causal mask.** Model bahasa menebak token berikutnya; jika saat memproses token ke-2 ia boleh melihat token ke-5, ia mencontek jawaban. Semua skor posisi masa depan diganti `−∞`:

```
S''_ij = { S'_ij  jika j ≤ i
         { −∞     jika j > i
```

Memakai `−∞` (bukan 0) karena bobot ditentukan oleh `e^S`: `e^{−∞} = 0` membuat bobot benar-benar nol, sedangkan `e⁰ = 1` justru memberi bobot besar. Implementasi nyata memakai angka sangat negatif seperti `−10⁹`. Hasilnya matriks bobot berbentuk **segitiga bawah** — tanda pengenal decoder-only Transformer (GPT dan hampir semua LLM generatif).

### Value: informasi yang benar-benar dipindahkan

```
O = A·V
```

Keluaran attention untuk sebuah posisi adalah **rata-rata berbobot** dari seluruh vektor value, dengan bobot `A`. **Kenapa key dan value dipisah?** Karena yang membuat sebuah token *pantas diperhatikan* belum tentu sama dengan *informasi yang berguna diambil* darinya; memisahkannya memberi model kebebasan mengatur keduanya.

### Rumus lengkapnya

```
Attention(Q, K, V) = softmax( QKᵀ/√d_k + M )·V
```

| Langkah | Operasi | Ukuran hasil |
|---|---|---|
| 1 | `S = QKᵀ` | `n × n` |
| 2 | `S' = S/√d_k` | `n × n` |
| 3 | `S'' = S' + M` (causal mask) | `n × n` |
| 4 | `A = softmax(S'')` per baris | `n × n` |
| 5 | `O = AV` | `n × d_v` |

### Perhitungan lengkap mini-LLM (opsional)

Lima token `<bos> kucing duduk di atas`, `d_k = d_v = 4`. `H`:

| Token | h1 | h2 | h3 | h4 |
|---|---|---|---|---|
| `<bos>` | 0,5 | 0,0 | 0,0 | 0,0 |
| kucing | 1,0 | 0,5 | 1,0 | 0,0 |
| duduk | 0,0 | 1,0 | 0,5 | 1,0 |
| di | 1,0 | 1,0 | 0,0 | 0,5 |
| atas | 0,5 | 0,5 | 1,0 | 1,0 |

Untuk query *atas* (`q = [2,0, 1,0, 1,0, 2,5]`), skor mentahnya adalah `[0,50, 5,00, 8,00, 5,25, 9,00]`; dibagi `√4 = 2` menjadi `[0,25, 2,50, 4,00, 2,625, 4,50]`. Setelah mask + softmax, bobotnya:

| Key | bobot |
|---|---|
| `<bos>` | 0,0075 |
| kucing | 0,0709 |
| duduk | 0,3176 |
| di | 0,0803 |
| atas | 0,5237 |

Keluaran *atas* `O = AV = [0,5851, 0,7243, 0,6491, 0,7884]`.

### Membaca hasilnya

Tiga hal dari heatmap bobot:
1. **Setiap baris adalah daftar perhatian** yang berjumlah 1 — baris *atas*: ~52% ke dirinya, ~32% ke *duduk*, sisanya tersebar.
2. **Segitiga atas selalu kosong** — causal mask bekerja.
3. **Baris pertama selalu 1,00** — token `<bos>` hanya punya satu kolom tak tertutup (dirinya sendiri); softmax dari satu elemen selalu 1.

### Kenapa attention bisa dilatih

Tidak ada aturan bawaan semacam "jika ada kata ganti, lihat kata benda sebelumnya". Yang ada hanya `W_Q, W_K, W_V` yang diinisialisasi acak. Setiap langkah punya turunan, sehingga gradien mengalir ke ketiganya:

```
∂L/∂W_Q = Hᵀ · ∂L/∂Q
```

Jika memperhatikan posisi tertentu mengurangi loss, gradien menyesuaikan `W_Q`/`W_K` sehingga dot product pasangan itu membesar. Tidak ada yang memprogramnya — ia muncul dari optimasi.

**Attention vs pencarian tabel:** pencarian keras membandingkan sama/tidak dan mengambil satu hasil (tidak bisa diturunkan); attention memakai dot product kontinu dan mengambil **semua** dengan bobot (bisa diturunkan). Sifat "bisa diturunkan" itulah yang membuat pemilihan di attention bisa **dilatih**.

## Implementasi

- [`src/llmbepees/np/attention.py`](../../src/llmbepees/np/attention.py): `project_qkv`, `attention_scores`, `attention_weights`, `causal_mask`, `scaled_dot_product_attention`, `single_head_attention`, `score_variance`, `attention_flops`.
- Uji memverifikasi seluruh angka 13.7 (Q/K/V, S, S/√d_k, A, O), sifat mask, varians skor ≈ `d_k`, dan biaya `n²` vs `n`.

## Inti bab

Attention adalah **rata-rata berbobot dari vektor value**, dengan bobot yang ditentukan oleh kecocokan antara query dan key. Dua penyesuaian melengkapinya: pembagian `√d_k` menjaga varians skor tetap 1 supaya softmax tidak runtuh, dan **causal mask** memaksa tiap posisi hanya melihat ke belakang sehingga model tidak mencontek jawaban yang harusnya ditebak — inilah yang memungkinkan satu forward pass menghasilkan banyak sinyal pelatihan sekaligus.

## Cek pemahaman

1. Mengapa sebuah token membutuhkan attention, padahal ia sudah punya embedding dan informasi posisi?
2. Apa perbedaan peran antara key dan value, padahal keduanya sama-sama dibuat dari `H`?
3. Mengapa causal mask memakai `−∞` dan bukan 0?

## Rujukan

- Buku *"Memahami Matematika di Balik LLM"*, Bab 13.
- Bahdanau dkk. (2015) — attention untuk penerjemahan mesin; Vaswani dkk. (2017) — scaled dot-product attention; Dao dkk. (2022) — FlashAttention.
