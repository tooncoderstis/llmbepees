# Bab 21 — Mengamati Mini-LLM Belajar dari Nol

> Catatan belajar, ditulis ulang dengan bahasa sendiri dari buku *"Memahami Matematika di Balik LLM"* (Bab 21).
> Bagian dari **Fase 10 — Latih Mini-LLM dari Nol (MILESTONE)**. Kode: [`src/llmbepees/np/mini_llm.py`](../../src/llmbepees/np/mini_llm.py) · Uji: [`tests/test_mini_llm.py`](../../tests/test_mini_llm.py) · Skrip: [`scripts/train_mini_llm.py`](../../scripts/train_mini_llm.py).
>
> Ini **pembuktian**, bukan penjelasan baru: sebuah Transformer 188 parameter dilatih dari angka acak dan setiap angka di bawah adalah hasil perhitungan nyata yang **direproduksi persis** dari buku.

## Masalah

Sepanjang buku, bobot dipilih manual agar mudah ditelusuri. Bab ini beralih ke pelatihan konkret: parameter dimulai acak, diperbarui lewat gradient descent, dan perubahannya diamati bertahap. Tujuannya menunjukkan bahwa konsep dari embedding sampai gradien **benar-benar bekerja ketika digabungkan** — dan struktur yang muncul bukan aturan yang ditulis manusia, melainkan hasil optimasi.

## Model

Transformer terkecil yang masih utuh — semua jenis komponen ada, tiap ukuran sekecil mungkin.

| Komponen | Nilai |
|---|---|
| Kosakata `V` | 8 |
| `d_model` | 4 |
| Jumlah lapisan `L` | 1 |
| Jumlah head | 1 |
| `d_ff` | 8 |
| Panjang maksimum | 7 |
| Positional embedding | terlatih |
| Normalisasi | LayerNorm (**Post-LN**, **tanpa** γ/β) |
| Aktivasi | ReLU |
| Weight tying | ya (`W_out = Eᵀ`) |

**Jumlah parameter:**

| Matriks | Ukuran | Parameter |
|---|---|---|
| `E` | 8 × 4 | 32 |
| `P` | 7 × 4 | 28 |
| `W_Q` | 4 × 4 | 16 |
| `W_K` | 4 × 4 | 16 |
| `W_V` | 4 × 4 | 16 |
| `W_O` | 4 × 4 | 16 |
| `W1` | 4 × 8 | 32 |
| `W2` | 8 × 4 | 32 |
| **Total** | | **188** |

Keempat dari 188 angka itu **tidak pernah menerima gradien**: masukan terpanjang hanya 6 token, jadi hanya `P[0]`…`P[5]` tersentuh. Model **menyimpan 188 parameter, tapi data hanya melatih 184**.

## Data

Dua pola kalimat, empat kalimat total:

| Kalimat | Jumlah |
|---|---|
| `<bos> kucing duduk di atas meja <eos>` | 3 |
| `<bos> kucing duduk di atas tikar <eos>` | 1 |

Tiap kalimat 7 token → 6 pasangan → **4 × 6 = 24 pasangan**. Konteks `di atas` muncul 4× dengan kelanjutan tidak selalu sama (3× *meja*, 1× *tikar*) — membuat data **tidak deterministik**, dan model akan mempelajari **perbandingan** itu.

## Kondisi awal

Semua 188 angka diundi dari `N(0, 0,5²)`, seed 42. Loss langkah 0: **2,4480** — lebih buruk dari tebak-asal `ln 8 = 2,0794` (bobot acak membuat model percaya diri ke arah salah; cross-entropy menghukum keyakinan keliru).

Distribusi keluaran untuk `<bos> kucing duduk di atas`:

| Token | p |
|---|---|
| `<bos>` | 0,2373 |
| kucing | 0,0975 |
| duduk | 0,1858 |
| di | 0,0182 |
| atas | 0,1736 |
| meja | 0,1298 |
| tikar | 0,0862 |
| `<eos>` | 0,0715 |

Sebaran masih di sekitar `1/8 = 0,125`, tapi sudah miring acak. Pola attention awal hampir seragam (baris terakhir tiap bobot ≈ 0,2).

## Proses pelatihan

1. 188 parameter acak → 2. forward seluruh kalimat → 3. bandingkan dengan token benar → 4. loss → 5. gradien → 6. perbarui (Adam) → 7. ulangi 400 langkah.

**Gradien** tiap parameter dihitung dengan **beda-hingga terpusat** (Bab 7), float64. Pada langkah pertama, norma vektor gradien `1,0442`, komponen terbesar `0,4703`. Ketimpangan penting: gradien terbesar pada embedding (`0,3420`) **~62×** lebih besar daripada pada `W_Q` (`0,0055`) — kondisi yang membuat Adam (Bab 12) lebih mudah dipakai daripada SGD.

**Hyperparameter:** Adam, `lr = 0,05` tetap, `β1 = 0,9`, `β2 = 0,999`, `ε = 10⁻⁸`, 400 langkah, batch penuh (24 pasangan).

### Perubahan loss

| Langkah | Loss | Perplexity |
|---|---|---|
| 0 | 2,4480 | 11,57 |
| 1 | 2,1266 | 8,39 |
| 2 | 1,9002 | 6,69 |
| 5 | 1,4479 | 4,25 |
| 10 | 1,0009 | 2,72 |
| 20 | 0,6009 | 1,82 |
| 50 | 0,1770 | 1,19 |
| 100 | 0,1100 | 1,12 |
| 200 | 0,0984 | 1,10 |
| 300 | 0,0964 | 1,10 |
| 400 | **0,0955** | 1,10 |

**Batas bawah teoretis** `0,0937` (lihat Bab 18); loss akhir hanya **1,9%** di atasnya. Loss berhenti di atas nol karena data latih sendiri memuat `di atas` dengan dua kelanjutan berbeda.

### Perubahan prediksi

Setelah 400 langkah, untuk `<bos> kucing duduk di atas`:

| Token | Logit | Probabilitas |
|---|---|---|
| `<bos>` | 0,4788 | 0,0001 |
| kucing | −1,1845 | 0,0000 |
| duduk | 1,3170 | 0,0003 |
| di | −6,5175 | 0,0000 |
| atas | 1,9015 | 0,0005 |
| meja | 9,2650 | **0,7492** |
| tikar | 8,1651 | **0,2494** |
| `<eos>` | 2,1264 | 0,0006 |

| | Data latih | Model |
|---|---|---|
| meja | 0,7500 | 0,7492 |
| tikar | 0,2500 | 0,2494 |

Selisih <0,001. **Model tidak menghafal satu jawaban — ia mempelajari distribusinya**, persis seperti yang diminta cross-entropy.

Prediksi di setiap posisi: `kucing` 0,998, `duduk` 0,999, `di` 0,997, `atas` 0,998, lalu `meja` 0,749 — singkatnya: hampir pasti di posisi deterministik, dan menyisakan ketidakpastian persis di tempat data tidak pasti. Diminta menulis dari awal, model menghasilkan `<bos> kucing duduk di atas meja <eos>` lengkap dan berhenti sendiri.

### Perubahan embedding

Dua hal berubah. **Pertama, besaran tumbuh:** norma rata-rata naik `0,80 → 3,87` (~5×), karena logit adalah dot product dan memperbesar norma embedding adalah cara langsung memperbesar selisih logit (keadaan akhir sudah dinormalisasi LayerNorm). **Kedua, *meja* dan *tikar* menjadi hampir sejajar:**

| | cos(meja, tikar) |
|---|---|
| Sebelum | 0,7557 |
| Sesudah | **0,9960** |

Keduanya punya peran identik dalam data (selalu setelah *di atas*, diikuti `<eos>`). Tidak ada satu baris instruksi pun yang menyuruhnya.

### Perubahan routing attention

| Baris `atas` | `<bos>` | kucing | duduk | di | atas |
|---|---|---|---|---|---|
| Sebelum | 0,1913 | 0,1836 | 0,1894 | 0,2278 | 0,2079 |
| Sesudah | 0,0000 | 0,0000 | 0,0000 | 0,0002 | **0,9998** |

Distribusi yang hampir rata menjadi **hampir one-hot**. Empat dari lima baris memusat pada satu posisi (tiga di antaranya diri sendiri); satu-satunya yang menoleh ke posisi lain adalah *di*, menyerahkan seluruh bobotnya ke *kucing*.

### Apa yang boleh dan tidak boleh disimpulkan

**Boleh:** attention mempelajari routing apa pun yang paling menurunkan loss; jika data tidak menuntut pertukaran antarposisi, routing mengerut sampai hampir tidak memindahkan apa-apa. **Yang mengerut hanya routing, bukan seluruh sub-blok** — keluaran sub-blok tetap `A·V·W_O`, jadi `W_V` dan `W_O` tetap mengubah representasi.

**Tidak boleh:** dengan 4 kalimat dan 1 lapisan, tidak bisa disimpulkan apa pun tentang attention di model besar. Pola kaya seperti *previous-token head*/*induction head* menuntut data lebih beragam dan lebih dari satu lapisan.

### Mini-LLM vs LLM besar

| | Mini-LLM | GPT-3 |
|---|---|---|
| Parameter | 188 | 175.000.000.000 |
| Kosakata | 8 | 50.257 |
| Lapisan | 1 | 96 |
| Token latih | 24 | 300.000.000.000 |
| Waktu latih | ~1 menit | puluhan–ratusan tahun-GPU |

Rasio parameter ~9 × 10⁸ kali. **Yang sama adalah kerangka perhitungannya, bukan resep arsitekturnya** (GPT-3 memakai Pre-LN & GELU; mini-LLM Post-LN & ReLU). Kenapa perbedaan skala saja menghasilkan perbedaan kemampuan sebesar itu masih menjadi riset aktif.

## Implementasi

- [`src/llmbepees/np/mini_llm.py`](../../src/llmbepees/np/mini_llm.py): `init_params` (seed 42, `N(0,0,5²)`), `forward` (batched, Post-LN tanpa γ/β, ReLU, weight tying), `sequence_loss`, `finite_difference_gradients`, `train` (Adam lr 0,05, 400 langkah), `next_token_probs`, `generate`.
- [`scripts/train_mini_llm.py`](../../scripts/train_mini_llm.py): melatih dan mencetak kurva loss + pemeriksaan titik buku.
- Uji mereproduksi **seluruh** titik: loss `2,4480 → 0,0955`, perplexity `11,57 → 1,10`, `P(meja)=0,7492`, `P(tikar)=0,2494`, `cos(meja,tikar)=0,9960`, attention `0,9998`, 188 parameter (184 dilatih).

## Inti bab

Bab ini pembuktian: sebuah Transformer 188 parameter dimulai acak, dilatih 400 langkah, dan setiap angka yang dilaporkan adalah hasil perhitungan nyata. Loss turun `2,4480 → 0,0955` (hanya 1,9% di atas batas teoretis), distribusinya menyamai proporsi data (`0,7492` vs `0,7500`), embedding *meja* & *tikar* bergerak dari cos `0,7557` ke `0,9960`, dan routing attention mengerut menjadi hampir one-hot. **Tidak ada satu instruksi pun yang menyuruh semua itu terjadi** — semuanya muncul dari menggeser angka ke arah yang menurunkan loss, yaitu 184 angka, karena 4 sisanya tidak pernah tersentuh data.

## Cek pemahaman

1. Mengapa loss model ini berhenti di 0,0955 dan tidak bisa mencapai nol?
2. Mengapa embedding *meja* dan *tikar* menjadi hampir sejajar, padahal tidak ada instruksi?
3. Mengapa pola attention yang mengerut menjadi hampir identitas tidak berarti sub-blok attention berhenti bekerja?

## Rujukan

- Buku *"Memahami Matematika di Balik LLM"*, Bab 21 + Lampiran (rincian eksperimen).
- Lihat juga [`docs/adr/0002-strategi-verifikasi.md`](../adr/0002-strategi-verifikasi.md).
