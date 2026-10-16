# Bab 18 — Pretraining: Objektif, Batch, dan Scaling Law

> Catatan belajar, ditulis ulang dengan bahasa sendiri dari buku *"Memahami Matematika di Balik LLM"* (Bab 18).
> Bagian dari **Fase 10 — Latih Mini-LLM dari Nol**. Kode: [`src/llmbepees/np/training.py`](../../src/llmbepees/np/training.py) · Uji: [`tests/test_training.py`](../../tests/test_training.py).

## Masalah

Bagian III menjelaskan **bentuk** model. Yang belum dijawab: dari mana 175 miliar angka di dalamnya? Tidak ada manusia yang menulisnya. Semuanya ditentukan oleh proses otomatis bernama **pretraining**. Tiga hal yang dibutuhkan: satu angka yang menyatakan seberapa buruk model (loss), jawaban benar untuk setiap prediksi, dan data dalam jumlah besar.

Yang paling mengejutkan: **jawaban yang benar tidak perlu dibuat siapa pun**.

## Gagasan

### Labelnya adalah teks itu sendiri

Ambil kalimat biasa `kucing duduk di atas meja`. Kalimat ini sudah memuat lima soal beserta kuncinya:

| Masukan | Target |
|---|---|
| `<bos>` | kucing |
| `<bos> kucing` | duduk |
| `<bos> kucing duduk` | di |
| `<bos> kucing duduk di` | atas |
| `<bos> kucing duduk di atas` | meja |

Labelnya adalah teks itu sendiri — **self-supervised**. Berkat causal mask (Bab 13), kelima soal dihitung dalam **satu kali jalan**, bukan lima. Urutan `n` token menghasilkan `n − 1` sinyal loss.

### Rumus loss, empat tingkat

```
L(θ) = −(1/∑|x|) ∑_{x∈D} ∑_{t=1}^{|x|} ln P_θ(x_t | x_<t)
```

Tingkat 1 (satu prediksi): `L_t = −ln P_θ(x_t | x_<t)`. Tingkat 2 (satu dokumen): jumlahkan posisi. Tingkat 3 (korpus): jumlahkan dokumen. Tingkat 4: **bagi dengan jumlah token**, bukan jumlah dokumen — kalau dibagi jumlah dokumen, dokumen panjang dapat bobot lebih besar. Yang dipakai praktik: **rata-rata per token**.

**Pasangan input–target** dibentuk dengan menggeser: `input = tokens[:-1]`, `target = tokens[1:]`.

### Skala data & batch

| Model | Token pelatihan |
|---|---|
| GPT-2 | ~10 miliar |
| GPT-3 | ~300 miliar |
| Chinchilla 70B | 1,4 triliun |
| Llama 2 | 2 triliun |

Satu langkah pelatihan: ambil batch `B` urutan sepanjang `n` → forward → loss → backward → optimizer. Token per langkah = `B × n`. GPT-3 (batch ~3,2 juta token, konteks 2048): `B = 3.200.000/2048 ≈ 1562` urutan; `300×10⁹ / 3,2×10⁶ ≈ 93.750` langkah. **Gradient accumulation** memecah batch besar menjadi `k` sub-batch: hitung gradien tiap sub-batch, jumlahkan, baru perbarui sekali — kebutuhan memori aktivasi menjadi `1/k`.

### Biaya komputasi

```
C ≈ 6·N·D
```

GPT-3: `6 × 175×10⁹ × 300×10⁹ = 3,15 × 10²³` FLOP. Pada `1,5 × 10¹⁴` FLOP/s per GPU ≈ `2,1 × 10⁹` detik-GPU = **66,6 tahun-GPU** → dengan 1000 GPU ≈ 24 hari.

### Scaling law

Loss turun mengikuti **hukum pangkat** (power law) terhadap parameter `N`, token `D`, atau komputasi `C`:

```
L(N) ≈ (N_c / N)^{α_N},   α_N ≈ 0,076
L(D) ≈ (D_c / D)^{α_D},   α_D ≈ 0,095
L(C) ≈ (C_c / C)^{α_C},   α_C ≈ 0,050
```

Karena `α` sangat kecil, **melipatgandakan `N` 10× hanya mengurangi loss ~16%** (`10^{−0,076} = 0,84`). Perbaikan menuntut lompatan skala makin besar.

**Syarat yang sering hilang:** bentuk di atas tidak punya batas bawah. Parameterisasi lebih lengkap menambah suku tetap:

```
L(N, D) = E + A/N^α + B/D^β
```

`E` adalah loss yang **tidak bisa direduksi** (entropi teks alami untuk data/tokenizer/konteks tertentu). Berapa pun `N` dan `D`, loss tidak akan turun di bawah `E`.

### Chinchilla: alokasi optimal

Dengan komputasi tetap, ukuran model dan jumlah token sebaiknya tumbuh berimbang:

```
D_optimal ≈ 20 × N
```

| `N` | `D_optimal` |
|---|---|
| 1 × 10⁹ | 2 × 10¹⁰ |
| 7 × 10⁹ | 1,4 × 10¹¹ |
| 70 × 10⁹ | 1,4 × 10¹² |
| 175 × 10⁹ | 3,5 × 10¹² |

**Konsekuensi untuk GPT-3:** dilatih pada 300 miliar token untuk 175 miliar parameter — rasio hanya 1,7 token/parameter, jauh di bawah 20, jadi **kurang dilatih**. Chinchilla (70B) yang dilatih pada 1,4 triliun token mengungguli Gopher (280B).

**Catatan praktis:** kalau yang penting adalah biaya **inference** (model dipanggil miliaran kali), model kecil yang dilatih lebih lama lebih menguntungkan — Llama 2 7B dilatih pada 2 triliun token (`D/N = 286`).

### Kurva loss

Bentuk umumnya: turun cepat lalu mendatar. Loss awal ≈ `ln V` (untuk `V = 50257`: 10,82) — jika jauh dari itu, curigai bug (inisialisasi/target), tapi bukan bukti. Mini-LLM Bab 21 mulai dari `2,4480` padahal `ln 8 = 2,0794` — bukan bug, melainkan akibat bobot acak yang percaya diri ke arah salah.

**Batas bawah teoretis mini-LLM.** Dari 4 kalimat × 6 prediksi = 24 prediksi: 20 deterministik, 4 setelah *di atas* (3× *meja*, 1× *tikar*). Model terbaik memberi `P(meja)=0,75`, `P(tikar)=0,25`:

```
L_min = (3(−ln 0,75) + 1(−ln 0,25)) / 24 = (3(0,2877) + 1,3863)/24 = 2,2494/24 = 0,0937
```

Loss akhir `0,0955` hanya **1,9%** di atas batas ini.

### Data: kualitas & pengulangan

**Deduplikasi** (hapus duplikat) meningkatkan efisiensi & mengurangi hafalan. **Pengulangan** data sampai ~4 epoch hampir sebaik data baru, setelah itu manfaatnya turun cepat. **Kontaminasi** (data uji masuk data latih) membuat evaluasi tidak bermakna.

## Implementasi

- [`src/llmbepees/np/training.py`](../../src/llmbepees/np/training.py): `shift_pairs`, `sequence_cross_entropy` (rata-rata `−ln P`), `tokens_per_step`, `gradient_accumulation`, `random_guess_loss` (`ln V`), `scaling_law`, `chinchilla_tokens`.
- Uji memverifikasi `ln 8 = 2,0794`, `ln 50257 = 10,82`, faktor `10^{−0,076} = 0,84`, dan `D ≈ 20N`.

## Inti bab

Pretraining tidak memerlukan satu pun label buatan manusia — targetnya adalah token berikutnya di dalam teks itu sendiri. Berkat causal mask, satu urutan `n` token menghasilkan `n−1` sinyal loss sekaligus. Memperbaiki model menuntut lompatan skala yang makin besar (loss turun mengikuti hukum pangkat dengan eksponen sangat kecil), dan **loss pada distribusi data berhenti di atas nol** — batas bawahnya adalah ketidakpastian bersyarat yang melekat pada data.

## Cek pemahaman

1. Mengapa pretraining disebut self-supervised, bukan unsupervised?
2. Mengapa loss awal sebuah model biasanya mendekati `ln V`?
3. Apa isi pesan utama Chinchilla tentang alokasi anggaran komputasi?

## Rujukan

- Buku *"Memahami Matematika di Balik LLM"*, Bab 18.
- Kaplan dkk. (2020) — scaling law; Hoffmann dkk. (2022) — Chinchilla; Muennighoff dkk. (2023) — pengulangan data.
