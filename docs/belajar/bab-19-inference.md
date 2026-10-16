# Bab 19 — Inference: Sampling, Temperature, dan KV Cache

> Catatan belajar, ditulis ulang dengan bahasa sendiri dari buku *"Memahami Matematika di Balik LLM"* (Bab 19).
> Bagian dari **Fase 10 — Latih Mini-LLM dari Nol**. Kode: [`src/llmbepees/np/inference.py`](../../src/llmbepees/np/inference.py) · Uji: [`tests/test_inference.py`](../../tests/test_inference.py).

## Masalah

Model sudah dilatih dan parameternya dibekukan. Bab ini membahas apa yang terjadi setiap kali kamu mengetik sesuatu dan menekan enter — **seluruhnya hanya jalur maju**.

## Gagasan

### Loop generasi

```
prompt = [t1, t2, ..., tk]
ULANGI:
  1. Jalankan model pada seluruh urutan
  2. Ambil baris logit TERAKHIR saja
  3. Ubah menjadi probabilitas (softmax)
  4. Pilih satu token dari distribusi itu
  5. Tempel token itu ke akhir urutan
SAMPAI <eos> keluar atau batas panjang tercapai
```

Langkah 4 — "pilih satu token" — adalah tempat semua keputusan desain berada.

### Greedy decoding

```
x_{t+1} = argmax_i p_i
```

Distribusi mini-LLM terlatih setelah `<bos> kucing duduk di atas`: `meja 0,7492`, `tikar 0,2494`, sisanya <0,001. Greedy memilih `meja`. Sifatnya deterministik, bagus untuk jawaban tepat, **buruk untuk teks kreatif** (cenderung berulang) — fenomena *neural text degeneration* (Holtzman dkk., 2020): frasa yang muncul membuat dirinya makin mungkin muncul lagi.

### Sampling murni & masalah ekor panjang

Ambil token acak sesuai probabilitasnya. Untuk kosakata 50.257, misalkan 50.000 token masing-masing `10⁻⁵`: massa ekornya `50.000 × 10⁻⁵ = 0,5`. **Setengah massa ada di token yang secara individual sangat tidak mungkin** — sampling murni menarik dari ekor ini cukup sering dan menghasilkan teks aneh.

### Temperature

```
p_i = e^{z_i/T} / ∑_j e^{z_j/T},   T > 0
```

Logit demo `[0,0, −0,1725, 0,1725, −1,9539, 1,9539, 0,3907, −0,3907, 0,0]`:

| Token | T=0,5 | T=0,8 | T=1,0 | T=1,5 | T=2,0 |
|---|---|---|---|---|---|
| `<bos>` | 0,0177 | 0,0559 | 0,0747 | 0,0997 | 0,1102 |
| kucing | 0,0125 | 0,0451 | 0,0629 | 0,0889 | 0,1010 |
| duduk | 0,0250 | 0,0694 | 0,0888 | 0,1118 | 0,1201 |
| di | 0,0004 | 0,0049 | 0,0106 | 0,0271 | 0,0415 |
| atas | 0,8801 | 0,6433 | 0,5273 | 0,3667 | 0,2926 |
| meja | 0,0386 | 0,0912 | 0,1105 | 0,1293 | 0,1339 |
| tikar | 0,0081 | 0,0343 | 0,0506 | 0,0768 | 0,0906 |
| `<eos>` | 0,0177 | 0,0559 | 0,0747 | 0,0997 | 0,1102 |

| `T` | Efek | Kegunaan |
|---|---|---|
| → 0 | → one-hot | setara greedy |
| < 1 | lebih runcing | jawaban faktual, kode |
| = 1 | distribusi asli | netral |
| > 1 | lebih rata | teks kreatif |
| → ∞ | → seragam | acak total |

**Kenapa `T → 0` setara greedy:** rasio `e^{z_i/T} / e^{z_max/T} = e^{(z_i − z_max)/T}`; karena `z_i − z_max < 0` dan `T → 0`, rasio → 0. Hanya logit terbesar bertahan. Verifikasi untuk *atas* (1,9539) vs *meja* (0,3907), selisih `1,5632`:

| `T` | `e^{−1,5632/T}` | Rasio meja:atas |
|---|---|---|
| 2,0 | 0,4577 | 1 : 2,18 |
| 1,0 | 0,2094 | 1 : 4,78 |
| 0,5 | 0,0439 | 1 : 22,8 |
| 0,1 | 1,6 × 10⁻⁷ | 1 : 6,3 juta |

### Top-k sampling

Ambil `k` token teratas, buang sisanya, normalisasi ulang. Contoh `k = 3`:

| Peringkat | Token | p | Kumulatif |
|---|---|---|---|
| 1 | atas | 0,5273 | 0,5273 |
| 2 | meja | 0,1105 | 0,6377 |
| 3 | duduk | 0,0888 | 0,7265 |

Bobot baru: `atas 0,5273/0,7266 = 0,7257`, `meja 0,1520`, `duduk 0,1222`. **Kelemahan:** `k` tetap, padahal bentuk distribusi berubah-ubah.

### Top-p (nucleus) sampling

Ambil token teratas sampai kumulatifnya mencapai `p`, lalu normalisasi. Dengan `p = 0,8` pada distribusi demo: `atas (0,5273)`, `meja (0,6377)`, `duduk (0,7265)`, `<bos> (0,8012)` — **4 token**, dibagi `0,8012`: `0,6581 / 0,1379 / 0,1108 / 0,0932`. Pada distribusi **runcing** model terlatih (`meja 0,7492`, `tikar 0,2494`): hanya **2 token**. Jumlah kandidat menyesuaikan kepercayaan diri model — inilah yang tidak bisa top-k.

### Menggabungkan ketiganya

```
logit → bagi T → softmax → top-k → top-p → sampling
```

Konfigurasi netral: `T = 1,0`, `p = 1,0`, `k` nonaktif. Teks kreatif: `T ≈ 0,7–1,0`, `p ≈ 0,9–0,95`; jawaban faktual/kode: `T → 0`.

### KV cache

Loop generasi menjalankan model dari awal tiap langkah. Untuk 100 token dari prompt 5 token, total token diproses `∑_{i=5}^{104} i = 5450`. Padahal vektor `k`/`v` token lama tidak berubah (causal mask!). **Simpan semua `K` dan `V`:** pada langkah baru hanya hitung `q,k,v` token baru, tempel `k,v` ke cache, lalu `q` baru ber-attention dengan seluruh `K` cache. Biaya per langkah turun dari `O(n²d)` ke `O(nd)`.

| | Tanpa cache | Dengan cache |
|---|---|---|
| Token diproses per langkah | `n` | 1 |
| Total diproses (100 token) | 5450 | 105 |
| Penghematan | | ~52× |

**Yang dihemat dan tidak:** cache menghapus **perhitungan ulang** representasi token lama, bukan seluruh biaya — bagian attention per langkah tetap `O(nd)`, dan proyeksi `Q,K,V,W_O`/FFN tetap dijalankan untuk token baru.

**Harga: memori.**
```
ukuran cache = 2 × L × n × d_kv × byte
```
Model 7B (`L=32, n=2048, d=4096`, fp16): `2 × 32 × 2048 × 4096 × 2 = 1,073 × 10⁹ byte = 1,00 GiB` **per urutan**; 40 pengguna ≈ 40 GB.

| Konteks `n` | Cache 7B/urutan |
|---|---|
| 2.048 | 1,0 GB |
| 8.192 | 4,0 GB |
| 32.768 | 16,0 GB |
| 131.072 | 64,0 GB |

### Prefill dan decode

**Prefill** (proses seluruh prompt sekaligus) dibatasi FLOP; **decode** (satu token per langkah) dibatasi **bandwidth memori**: untuk satu token, GPU harus membaca seluruh bobot (14 GB untuk 7B fp16) tapi hanya mengalikan 1 token. Batas atas: `2000 GB/s ÷ 14 GB = 143 token/detik`. Melampauinya dengan **batching** (satu pembacaan bobot melayani banyak token).

### Beam search

Lacak `b` urutan terbaik berdasarkan log-probabilitas kumulatif. Contoh `b = 2`, langkah 1: `kucing(−0,693)`, `anjing(−1,204)`. Langkah 2 total: `kucing duduk (−1,050)`, `kucing lari (−2,302)`, `anjing duduk (−1,427)`, `anjing lari (−2,408)`; simpan 2 teratas. Greedy akan terkunci pada *kucing* sejak awal; beam menjaga *anjing* tetap hidup. **Jarang dipakai untuk LLM percakapan**: urutan berprobabilitas tertinggi cenderung membosankan & berulang; teks manusia punya variasi lebih tinggi.

## Implementasi

- [`src/llmbepees/np/inference.py`](../../src/llmbepees/np/inference.py): `greedy`, `temperature_softmax`, `top_k`, `top_p`, `sample`, `decode_next`, `beam_step`, `prefill_work`, `decode_speed_upper_bound` (dan `kv_cache_bytes` dari Bab 14).
- Uji memverifikasi tabel temperature, top-k `0,7257/0,1520/0,1222`, top-p `0,6581/0,1379/0,1108/0,0932` (adaptif: 2 token pada distribusi runcing), dua langkah beam `−1,050`/`−1,427`, dan `5450 → 105`.

## Inti bab

Model hanya menghasilkan distribusi; **pemilihan token adalah langkah terpisah**. Greedy stabil tapi berulang; sampling beragam tapi sesekali menarik dari ekor. Temperature, top-k, dan top-p mengatur keseimbangan itu, semuanya dengan mengubah distribusi **sebelum** sampling. KV cache menukar memori dengan waktu: ia menghapus hampir semua perhitungan berulang, tetapi ukurannya tumbuh sebanding panjang konteks — itulah alasan sesungguhnya konteks panjang mahal saat inference.

## Cek pemahaman

1. Mengapa greedy decoding cenderung menghasilkan teks berulang?
2. Apa beda top-k dan top-p, dan mengapa top-p sering lebih disukai?
3. Mengapa konteks panjang mahal saat model dipakai?

## Rujukan

- Buku *"Memahami Matematika di Balik LLM"*, Bab 19.
- Fan dkk. (2018) — top-k; Holtzman dkk. (2020) — top-p/nucleus & neural text degeneration.
