# Bab 8 — Dari Teks ke Angka: Tokenisasi dan Embedding

> Catatan belajar, ditulis ulang dengan bahasa sendiri dari buku *"Memahami Matematika di Balik LLM"* (Bab 8).
> Bagian dari **Fase 2 — Tokenisasi & Embedding**. Kode: [`src/llmbepees/np/bpe.py`](../../src/llmbepees/np/bpe.py), [`src/llmbepees/np/embedding.py`](../../src/llmbepees/np/embedding.py) · Uji: [`tests/test_bpe.py`](../../tests/test_bpe.py), [`tests/test_embedding.py`](../../tests/test_embedding.py).

## Masalah

Model tidak bisa mengalikan huruf. Sebelum ada satu pun operasi matematika, teks harus diubah menjadi angka. Ada dua tahap: **memecah teks menjadi token**, lalu **mengubah token menjadi vektor**.

## Gagasan

### Kenapa bukan per kata, dan kenapa bukan per huruf

| Pendekatan | Ukuran kosakata | Masalah |
|---|---|---|
| Per karakter | ~100 | urutan jadi sangat panjang; biaya attention naik `n²` |
| Per kata | ratusan ribu–jutaan | kata di luar kosakata tak terwakili (jadi `<unk>`) |
| Per sub-kata | ~30.000–100.000 | kompromi; dipakai hampir semua LLM modern |

Contoh masalah "per kata": jika kosakata dibatasi 50.000 kata dan model bertemu *mengejawantahkan*, satu-satunya pilihan adalah `<unk>` — seluruh informasi kata itu hilang. Untuk bahasa Indonesia yang produktif imbuhan (*me-, -kan, ber-, -nya, di-*), masalah ini muncul terus-menerus. Pemecahan per sub-kata menyelesaikannya: kata jarang dipecah menjadi potongan yang sering.

### Byte Pair Encoding (BPE)

Algoritma yang dipakai GPT-2/GPT-3. Langkah pelatihannya:

1. Pecah semua kata dalam korpus menjadi karakter tunggal.
2. Hitung frekuensi setiap **pasangan simbol bersebelahan**.
3. Gabungkan pasangan yang paling sering muncul menjadi satu simbol baru.
4. Ulangi langkah 2–3 sampai jumlah simbol mencapai target (mis. 50.000).

Contoh korpus mini (dengan penanda akhir kata `</w>`):

| Kata | Frekuensi |
|---|---|
| rendang | 5 |
| rendah | 3 |
| renang | 2 |
| rendam | 4 |

Kosakata awal: `</w>, a, d, e, g, h, m, n, r` (9 simbol). Urutan penggabungan (seri diputus "yang muncul lebih dulu"):

| Langkah | Pasangan | Hitungan | Simbol baru |
|---|---|---|---|
| 1 | `r+e` | 14 | `re` |
| 2 | `re+n` | 14 | `ren` |
| 3 | `ren+d` | 12 (seri dgn `d+a`) | `rend` |
| 4 | `rend+a` | 12 | `renda` |
| 5 | `n+g` | 7 (seri dgn `g+</w>`) | `ng` |
| 6 | `ng+</w>` | 7 | `ng</w>` |

Hasil akhir setelah 6 penggabungan:

| Kata | Tokenisasi | Jumlah token |
|---|---|---|
| rendang | `renda ng</w>` | 2 |
| rendah | `renda h </w>` | 3 |
| renang | `ren a ng</w>` | 3 |
| rendam | `renda m </w>` | 3 |

Perhatikan: kata yang **paling sering** (rendang, 5×) berakhir dengan **token paling sedikit** — algoritmanya memang mengoptimalkan itu. Makin sering pola muncul, makin efisien representasinya.

Ukuran kosakata di model nyata: Transformer asli 37.000 · GPT-2/GPT-3 50.257 · Llama 2 32.000. (SentencePiece dengan model unigram adalah varian umum lain; keluarannya tetap deretan ID integer.)

**Konsekuensi praktis tokenisasi:** (1) model sulit mengeja/menghitung huruf karena tidak "melihat" huruf secara terpisah; (2) aritmetika sulit — `1234` bisa terpecah jadi `12`+`34`; (3) bahasa non-Inggris lebih mahal karena tokenizer yang dilatih dominan Inggris memecah teks Indonesia jadi lebih banyak token per kalimat.

### Dari token ID ke vektor: one-hot

Setelah tokenisasi, teks menjadi deretan integer: `<bos> kucing duduk di atas → [0, 1, 2, 3, 4]`. Integer ini **tidak boleh** langsung dipakai sebagai angka — ID hanyalah label, tidak ada arti dalam "meja lima kali kucing".

**One-hot** dari token ID `i` (kosakata `V`) adalah vektor panjang `V`, bernilai 1 di posisi `i`, 0 di tempat lain. Untuk `V = 8`:

| Token | ID | Vektor one-hot |
|---|---|---|
| `<bos>` | 0 | `[1, 0, 0, 0, 0, 0, 0, 0]` |
| kucing | 1 | `[0, 1, 0, 0, 0, 0, 0, 0]` |
| meja | 5 | `[0, 0, 0, 0, 0, 1, 0, 0]` |

Sifat: dot product dua one-hot berbeda selalu 0; dengan dirinya sendiri selalu 1. Artinya semua token berjarak sama — belum ada informasi makna sama sekali di tahap ini. Masalahnya: untuk `V = 50.257`, kita menyimpan 50.257 angka yang 50.256 di antaranya nol.

### Embedding: tabel pencarian yang bisa dilatih

**Matriks embedding** `E ∈ ℝ^{V×d}`: `V` baris (satu per token), `d` kolom (dimensi model). Baris ke-`i` adalah vektor embedding token ID `i`. Mengambilnya secara matematis:

```
e_i = o_i E     (o_i = one-hot token i)
```

Contoh mini-LLM (`V = 8`, `d = 4`): `o_5 E` hanya menyisakan baris ke-6 (karena one-hot bernilai 1 di sana), memberi `e_meja = [1, 0, 0, 1]`.

**Optimasi implementasi:** karena hasilnya selalu persis satu baris, tidak ada implementasi nyata yang benar-benar mengalikan `o_i E`; semuanya langsung **mengambil baris ke-`i`** dari memori. Biaya turun dari `O(V·d)` menjadi `O(d)`. Rumus `o_i E` tetap berguna untuk menurunkan gradiennya (Bab 11).

Ukuran matriks embedding: GPT-2 small `50257 × 768 = 38.597.376` (≈31% dari total parameter), Llama 2 7B `32000 × 4096 = 131.072.000`, GPT-3 `50257 × 12288 = 617.558.016`.

### Embedding bukan diprogram, melainkan dipelajari

Matriks `E` **tidak diisi manusia**. Nilainya diinisialisasi acak lalu diperbarui gradient descent seperti parameter lain. Contoh mini-LLM (inisialisasi `N(0, 0,5²)`, dilatih 400 langkah pada korpus 4 kalimat):

| Kata | Embedding awal (acak) | Embedding setelah dilatih |
|---|---|---|
| meja | `[-0,0924, -0,3405, 0,6113, -0,0773]` | `[-0,3744, -1,7296, 4,0288, -1,5429]` |
| tikar | `[-0,2142, -0,1761, 0,2662, 0,1827]` | `[-0,6374, -1,5958, 3,4239, -1,5035]` |

Hitung kemiripan arah antara *meja* dan *tikar* (keduanya selalu muncul setelah "di atas"):

```
cos(meja, tikar):  sebelum dilatih 0,7557  →  setelah dilatih 0,9960
```

Kedua vektor bergerak menjadi hampir searah. Tidak ada baris kode yang menyuruh mereka mirip; itu muncul murni dari gradient descent karena peran distribusionalnya identik dalam data.

> Tahan dulu sebelum menyebut "model memahami makna". Pada model sekecil ini, yang terjadi lebih tepat: dua token yang selalu muncul di konteks sama dan diikuti token sama berakhir di arah sama karena itulah satu-satunya cara menurunkan loss. Itu keteraturan statistik korpus, bukan bukti pemahaman semantik. Di model besar struktur lebih kaya (word2vec: `v(king) − v(man) + v(woman)` ≈ `v(queen)`), tetapi pada LLM modern representasi bersifat **kontekstual** (berubah tergantung kalimat).

### Weight tying: berbagi matriks embedding

Di akhir model (Bab 17), kita mengubah vektor `d` dimensi kembali menjadi `V` skor — butuh matriks `d × V`. Banyak model memakai **matriks embedding yang sama, ditranspos**:

```
logit = h Eᵀ
```

Keuntungan: (1) menghemat `V × d` parameter (GPT-2 small: 38,6 juta tidak disimpan dua kali); (2) memaksa konsistensi — token yang mirip saat masuk juga mirip saat keluar.

Contoh: vektor keluaran akhir posisi terakhir `h = [-0,8678, -1,0861, 0,6953, 1,2585]`.

```
logit_meja = h · e_meja = (-0,8678)(1) + (-1,0861)(0) + (0,6953)(0) + (1,2585)(1) = 0,3907
logit_di   = h · e_di   = -0,8678 - 1,0861 = -1,9539
```

Logit adalah dot product antara keadaan akhir model dan embedding tiap token. Hati-hati: `h · e = ‖h‖‖e‖cos θ`, jadi embedding bernorma besar bisa menang walaupun arahnya kurang cocok.

### Alur lengkap tahap ini

```
"kucing duduk di atas"
  → ["kucing", "duduk", "di", "atas"]      (tokenizer BPE)
  → [1, 2, 3, 4]                            (ID token)
  → ambil baris E → matriks 4 × d           (embedding)
  → masuk ke blok Transformer
```

Setelah tahap ini **tidak ada lagi teks di dalam model** — hanya matriks bilangan real. Semua operasi berikutnya adalah aljabar pada matriks itu. Rumusnya:

```
X = OE,   O ∈ ℝ^{n×V},   E ∈ ℝ^{V×d},   X ∈ ℝ^{n×d}
```

## Implementasi

- [`src/llmbepees/np/bpe.py`](../../src/llmbepees/np/bpe.py): `split_symbols`, `train_bpe` (daftar merge; seri diputus "muncul lebih dulu"), `encode`. Mereproduksi persis urutan merge contoh buku.
- [`src/llmbepees/np/embedding.py`](../../src/llmbepees/np/embedding.py): `one_hot`, `one_hot_matrix`, `embedding_lookup`, `embed_one_hot`, `tied_logits`, `embedding_parameters`.
- [`tests/test_bpe.py`](../../tests/test_bpe.py) & [`tests/test_embedding.py`](../../tests/test_embedding.py): semua contoh angka di atas.

## Inti bab

Bab ini adalah gerbang masuk: di sinilah teks berhenti menjadi teks dan berubah menjadi angka, dan setelah tahap ini tidak ada satu pun huruf tersisa di dalam model. Prosesnya dua langkah: **tokenisasi** memecah teks menjadi sub-kata, lalu **embedding** menukar tiap potongan dengan sebuah vektor lewat tabel. Yang paling penting: **isi tabel embedding tidak ditulis manusia** — diinisialisasi acak lalu diperbaiki oleh pelatihan.

## Cek pemahaman

1. Mengapa tokenizer memecah kata menjadi potongan, bukan memakai kata utuh?
2. Perkalian one-hot dengan matriks embedding sebenarnya setara dengan operasi apa?
3. Apa arti weight tying, dan mengapa ia menghemat parameter?

## Rujukan

- Buku *"Memahami Matematika di Balik LLM"*, Bab 8.
- Sennrich dkk. (2016) — BPE; Kudo & Richardson (2018) — SentencePiece; Radford dkk. (2019), Brown dkk. (2020), Touvron dkk. (2023b) — ukuran kosakata/embedding; Mikolov dkk. (2013) — word2vec.
