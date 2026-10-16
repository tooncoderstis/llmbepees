# Bab 14 — Multi-Head Attention

> Catatan belajar, ditulis ulang dengan bahasa sendiri dari buku *"Memahami Matematika di Balik LLM"* (Bab 14).
> Bagian dari **Fase 7 — Attention**. Kode: [`src/llmbepees/np/multihead.py`](../../src/llmbepees/np/multihead.py) · Uji: [`tests/test_multihead.py`](../../tests/test_multihead.py).

## Masalah

Satu head attention punya keterbatasan yang tidak bisa diakali dengan mengatur bobotnya:

> Kucing yang **lapar** itu duduk di atas **meja**.

Untuk memproses *duduk*, model butuh beberapa jenis informasi sekaligus: siapa pelakunya (*kucing*), sifatnya (*lapar*), dan tempatnya (*meja*). Satu head hanya punya **satu** pasang `W_Q`/`W_K`, jadi hanya ada **satu** ukuran kecocokan; jika ukuran untuk mencari pelaku berbeda dari ukuran untuk mencari tempat, satu head terpaksa berkompromi. Lebih lanjut, keluarannya hanya satu rata-rata berbobot, sehingga informasi pelaku dan tempat tercampur di satu tempat yang sulit dipisahkan lapisan berikutnya.

Perlu diluruskan: satu head **bisa** menyebarkan perhatian ke banyak token sekaligus. Keterbatasannya lebih halus — hanya satu *cara pandang*, dan hasilnya tercampur jadi satu.

## Gagasan

### Potong ruang, jalankan paralel

Jika setiap token diwakili `d` angka, kita **tidak** menambah kapasitas baru — kita membagi ruang yang ada. Dengan `h` head, setiap head bekerja pada subruang `d/h`, punya `W_Q, W_K, W_V` sendiri, dan menjalankan seluruh langkah Bab 13 secara terpisah.

Karena setiap head dibuat lebih kecil dengan perbandingan tepat, jumlah parameter **tidak bertambah**:

| Konfigurasi | Ukuran per head | Total kolom `W_Q` |
|---|---|---|
| 1 head, `d_k = 512` | 512 × 512 | 512 |
| 8 head, `d_k = 64` | 512 × 64 | 8 × 64 = 512 |

Delapan head kecil memakai parameter persis sama dengan satu head besar. Yang bertambah hanya **jumlah pola** yang bisa direpresentasikan sekaligus.

Cara praktis membentuk matriks per head adalah **memotong kolom** dari matriks besar: head 1 memakai kolom 1–2, head 2 kolom 3–4 (untuk mini-LLM `d = 4`, `h = 2`, `d_k = 2`).

### Head pertama dan kedua (mini-LLM)

Dari `H` yang **sama persis**, kedua head menghasilkan pola berbeda. Bandingkan baris *di*:

| Head | `<bos>` | kucing | duduk | di |
|---|---|---|---|---|
| Head 1 | 0,0121 | 0,3481 | 0,3481 | 0,2917 |
| Head 2 | 0,0306 | 0,1260 | 0,7379 | 0,1056 |

Head 1 menyebarkan perhatian hampir merata antara *kucing*, *duduk*, dan *di*; head 2 memusatkan hampir tiga perempat perhatiannya pada *duduk* saja. Dua pola berbeda, dari masukan yang sama, tanpa satu pun parameter tambahan.

Keluaran head untuk *atas*: head 1 → `[0,6389, 0,6933]`, head 2 → `[0,6397, 0,8019]`.

### Menggabungkan hasil kedua head

Hasilnya dua matriks `5 × 2`; lapisan berikutnya butuh satu matriks `5 × 4` seperti masukannya. Dua langkah:

**1. Concat** (sambung mendatar): `Concat(O1, O2)` berukuran `5 × 4`, kolom 1–2 dari head 1, kolom 3–4 dari head 2. Ukurannya benar, tetapi **kedua head masih terpisah rapi** di blok kolomnya.

**2. Campurkan lewat `W_O`**, matriks proyeksi keluaran:

```
MultiHead(H) = Concat(head_1, …, head_h)·W_O
```

Hitung baris *atas* kolom 1: kolom 1 `W_O` = `0,5×[1,0,0,1] = [0,5,0,0,0,5]`, sehingga `0,6389(0,5) + 0,8019(0,5) = 0,3195 + 0,4010 = 0,7204`. Angka `0,3195` datang dari head 1 dan `0,4010` dari head 2 — **di sinilah kedua head akhirnya bertemu**. Secara umum:

```
m_j = ∑_{p=1}^{h·d_v} concat_p · W_O[p,j]
```

Penjumlahannya melintasi semua kolom sambungan, artinya melintasi **semua head**. `W_O` adalah **satu-satunya** tempat informasi antarhead digabungkan di dalam blok attention.

Keluaran akhir untuk *atas*: `[0,7204, 0,6665, 0,6393, 0,7476]`.

### Rumus lengkapnya

```
head_i      = Attention(H·W_Qᵢ, H·W_Kᵢ, H·W_Vᵢ)
MultiHead(H) = Concat(head_1, …, head_h)·W_O
```

`W_O` berukuran `h·d_v × d`; biasanya `d_k = d_v = d/h`.

### Konfigurasi head pada model nyata

| Model | `d` | `h` | `d_k = d/h` |
|---|---|---|---|
| Transformer base | 512 | 8 | 64 |
| GPT-2 small | 768 | 12 | 64 |
| Llama 2 7B | 4096 | 32 | 128 |
| GPT-3 175B | 12288 | 96 | 128 |

`d_k` hampir selalu 64 atau 128 dan **tidak bertambah besar** seiring model membesar; yang bertambah adalah jumlah head.

### Multi-Query dan Grouped-Query Attention

Saat menghasilkan token satu-per-satu (Bab 19), semua `K` dan `V` token sebelumnya harus disimpan di memori GPU (*KV cache*):

```
ukuran KV cache = 2 · L · h · n · d_k · byte
```

(Faktor 2 karena `K` dan `V` sama-sama disimpan; `L` = jumlah lapisan, `h` = jumlah head, `n` = panjang konteks.)

Contoh model 7B (`L = 32`, `h = 32`, `d_k = 128` → `h·d_k = 4096`), konteks `n = 2048`, presisi 16-bit: `2 × 32 × 2048 × 4096 × 2 = 1,07 × 10⁹ byte = 1,00 GiB` **per urutan**. Melayani 50 pengguna sekaligus ≈ 50 GiB hanya untuk cache. Solusinya berbagi `K`/`V`:

| Metode | Jumlah K/V | Ukuran cache (relatif) |
|---|---|---|
| MHA | 32 | 1,000 |
| GQA (g = 8) | 8 | 0,250 |
| GQA (g = 4) | 4 | 0,125 |
| MQA | 1 | 0,031 |

**MQA** (Shazeer, 2019): semua head berbagi satu `K` dan satu `V`, tapi tetap punya `Q` masing-masing. **GQA** (Ainslie dkk., 2023): jalan tengah — head dibagi `g` grup, tiap grup berbagi satu pasang `K`/`V`. Llama 2 versi 70B memakai GQA. Yang dikurangi adalah keragaman `K`/`V`, bukan jumlah pola attention yang bisa dipelajari.

### Apa yang sebenarnya dipelajari tiap head

Analisis empiris menemukan head mengembangkan spesialisasi, dua yang paling luas dilaporkan: head yang memberi bobot besar pada token tepat sebelumnya (*previous-token head*), dan head yang mencari kemunculan sebelumnya dari token saat ini lalu menyalin token yang mengikutinya (*induction head*). **Batas jujurnya:** ini temuan empiris pada model tertentu, bukan sifat yang dijamin arsitektur. Yang dijamin matematikanya hanya: setiap head **dapat** mempelajari pola berbeda karena punya parameter berbeda.

## Implementasi

- [`src/llmbepees/np/multihead.py`](../../src/llmbepees/np/multihead.py): `split_heads`, `concat_heads`, `multi_head_attention`, `kv_cache_bytes`, `kv_cache_ratio`.
- Uji memverifikasi angka 14.3–14.5 (bobot & keluaran kedua head, concat, `W_O`, keluaran akhir `[0,7204, 0,6665, 0,6393, 0,7476]`) serta ukuran KV cache (1,00 GiB; GQA/MQA mengecilkan).

## Inti bab

Satu head punya satu ukuran kecocokan dan menghasilkan satu campuran value, jadi model menjalankan banyak head paralel di subruang berbeda, lalu menggabungkannya lewat `W_O`. Karena tiap head dibuat lebih kecil dengan perbandingan yang sama (`d_k = d/h`), **jumlah parameter tidak bertambah sedikit pun**. Yang bertambah adalah banyaknya pola hubungan yang bisa diwakili sekaligus. `W_O` adalah satu-satunya tempat hasil antarhead bertemu.

## Cek pemahaman

1. Jika satu head sudah bisa menyebarkan perhatian ke banyak token, apa sebenarnya yang tidak bisa ia lakukan?
2. Mengapa menambah head tidak menambah jumlah parameter?
3. Apa yang akan terjadi jika langkah perkalian dengan `W_O` dihapus?

## Rujukan

- Buku *"Memahami Matematika di Balik LLM"*, Bab 14.
- Vaswani dkk. (2017) — multi-head attention; Shazeer (2019) — MQA; Ainslie dkk. (2023) — GQA; Elhage dkk. (2021) / Olsson dkk. (2022) — induction head.
