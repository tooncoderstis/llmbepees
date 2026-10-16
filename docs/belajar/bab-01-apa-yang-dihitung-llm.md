# Bab 1 — Apa Sebenarnya yang Dihitung LLM?

> Catatan belajar, ditulis ulang dengan bahasa sendiri dari buku *"Memahami Matematika di Balik LLM"* (Bab 1).
> Bagian dari **Fase 1 — Fondasi Matematika**. Tanpa kode (bab konsep murni).

## Masalah

Ketika kita mengetik sesuatu dan model menjawab, apa yang **sebenarnya** dihitung di dalam? Banyak penjelasan memakai perumpamaan ("attention itu seperti lampu sorot") yang memberi rasa paham tetapi tidak bisa dihitung. Bab ini memasang definisi tegas sejak awal: tidak ada database jawaban, tidak ada aturan tata bahasa buatan manusia, tidak ada mesin pencari tersembunyi — hanya angka yang dikalikan dalam jumlah besar.

## Gagasan

Sebuah LLM adalah **fungsi matematika** yang menerima deretan token dan mengeluarkan deretan angka berupa **probabilitas tiap kemungkinan token berikutnya**.

Secara formal, untuk token ke-`t`:

```
P(x_{t+1} | x_1, x_2, …, x_t)
```

Dibaca: *probabilitas token berikutnya, diberikan semua token sebelumnya.*

### Contoh angka: satu langkah prediksi

Mini-LLM contoh (dipakai terus-menerus sepanjang buku) punya kosakata 8 token:

| Indeks | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Token | `<bos>` | kucing | duduk | di | atas | meja | tikar | `<eos>` |

`<bos>` = *beginning of sequence*, `<eos>` = *end of sequence*; keduanya token buatan yang diperlakukan sama seperti token lain.

Diberi konteks `<bos> kucing duduk di atas`, keluaran model **bukan** satu kata, melainkan delapan angka:

| Token | `<bos>` | kucing | duduk | di | atas | meja | tikar | `<eos>` |
|---|---|---|---|---|---|---|---|---|
| Probabilitas | 0,0001 | 0,0000 | 0,0003 | 0,0000 | 0,0005 | **0,7492** | 0,2494 | 0,0006 |

Dua sifat yang **selalu** berlaku untuk keluaran LLM:

1. Setiap angka di antara 0 dan 1.
2. Jumlah semua angka tepat 1.

Cek: `0,0001 + 0,0000 + 0,0003 + 0,0000 + 0,0005 + 0,7492 + 0,2494 + 0,0006 = 1,0001`. Selisih `0,0001` murni pembulatan ke empat angka desimal; nilai sebenarnya tepat 1. Mekanisme yang memaksa hal ini bernama **softmax** (Bab 10).

### Dari satu token menjadi kalimat: autoregressive generation

Model hanya bisa memprediksi **satu** token. Kalimat panjang dihasilkan dengan **mengulang** fungsi yang sama:

```
Langkah 1: input = [<bos>]                         → pilih "kucing"
Langkah 2: input = [<bos>, kucing]                 → pilih "duduk"
Langkah 3: input = [<bos>, kucing, duduk]          → pilih "di"
Langkah 4: input = [<bos>, kucing, duduk, di]      → pilih "atas"
Langkah 5: input = [<bos>, …, atas]                → pilih "meja"
Langkah 6: input = [<bos>, …, meja]                → pilih "<eos>" → BERHENTI
```

Token baru ditempelkan ke ujung masukan, lalu seluruh model dihitung ulang. Inilah sebab teks muncul potongan demi potongan di layar. Sistem nyata menyimpan hasil vektor token lama lewat **KV cache** (Bab 19) agar tiap langkah hanya menghitung token baru — hasilnya sama, hanya lebih cepat.

### Probabilitas seluruh kalimat

Untuk menilai satu kalimat utuh, kita kalikan probabilitas tiap langkah memakai **aturan rantai** (Bab 5):

```
P(x_1, x_2, …, x_n) = ∏_{t=1}^{n} P(x_t | x_1, …, x_{t-1})
```

Contoh: misalkan probabilitas langkah demi langkah adalah 0,50 · 0,25 · 0,40 · 0,10, maka

```
P(kalimat) = 0,50 × 0,25 × 0,40 × 0,10 = 0,005
```

Perhatikan: angkanya **cepat mengecil**. Untuk 100 token dengan rata-rata probabilitas 0,1 per token, hasilnya `10^-100` — terlalu kecil untuk presisi normal. Solusinya (Bab 6): jangan kalikan probabilitas, **jumlahkan logaritmanya**.

### Berapa banyak angka yang terlibat?

Model punya **parameter** — angka yang ditentukan lewat pelatihan lalu dibekukan. Jumlah parameter sebuah Transformer decoder (diturunkan penuh di Bab 16):

```
N ≈ V · d + L · (4d² + 2 · d · d_ff)
```

dengan `V` ukuran kosakata, `d` dimensi model, `L` jumlah lapisan, `d_ff` dimensi feed-forward (umumnya `4d`; bila begitu, isi kurung menjadi `12d²`).

- **GPT-2 small** (`V=50.257`, `d=768`, `L=12`): rumus memberi **124.318.464**; angka sebenarnya **124.439.808** — selisih 121.344 (< 0,1%) dari suku-suku kecil (bias, LayerNorm).
- **GPT-3** (`V=50.257`, `d=12.288`, `L=96`): rumus memberi **≈174,6 miliar** — inilah asal nama "175B".

## Inti bab

1. **LLM = fungsi** yang memetakan deretan token ke distribusi probabilitas atas kosakata.
2. **Kalimat panjang** dihasilkan dengan mengulang fungsi itu, satu token per pemanggilan (*autoregressive*).
3. Fungsi itu dibangun dari **perkalian matriks** dan beberapa fungsi non-linear — tidak ada komponen lain.

Saat model dipakai, tidak ada yang "terjadi" pada parameter: nilainya dibekukan setelah pelatihan dan hanya dibaca.

## Cek pemahaman

1. Jika model hanya bisa menebak satu token berikutnya, bagaimana ia menghasilkan paragraf panjang?
2. Mengapa keluaran model berupa daftar probabilitas, bukan satu kata tunggal?
3. Apa yang terjadi pada 174 miliar parameter itu ketika kita memakai model, bukan melatihnya?

## Rujukan

- Buku *"Memahami Matematika di Balik LLM"*, Bab 1.
- Radford dkk. (2019) — GPT-2; Brown dkk. (2020) — GPT-3 (lihat `docs/referensi/README.md`).
