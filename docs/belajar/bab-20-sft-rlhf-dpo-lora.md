# Bab 20 — Setelah Pretraining: SFT, RLHF, DPO, LoRA

> Catatan belajar, ditulis ulang dengan bahasa sendiri dari buku *"Memahami Matematika di Balik LLM"* (Bab 20).
> Bagian dari **Fase 11 — Port PyTorch + Model Kecil**. Kode: [`src/llmbepees/np/alignment.py`](../../src/llmbepees/np/alignment.py) · Uji: [`tests/test_alignment.py`](../../tests/test_alignment.py).

## Masalah

Model hasil pretraining dioptimasi untuk **memprediksi token berikutnya pada distribusi teks internet**, bukan untuk menjawab. Diberi `Apa ibu kota Prancis?`, kelanjutan yang paling mungkin di internet bisa jadi justru daftar pertanyaan serupa. **Objektif ≠ tujuan.**

Perlu dinyatakan tepat: kemampuan menjawab **bisa muncul** selama pretraining dan bisa dipancing lewat *prompting* (temuan few-shot GPT-3). Yang kurang pada base model bukan kemampuannya, melainkan **keandalannya** — belum konsisten mengikuti instruksi. SFT dan penyelarasan preferensi membuat perilakunya lebih konsisten; yang mereka tambahkan terutama **keandalan**, bukan penciptaan seluruh kemampuan dari nol.

## Gagasan

### SFT (Supervised Fine-Tuning)

Pretraining yang dilanjutkan pada dataset pasangan instruksi–jawaban, dengan loss **hanya dihitung pada bagian jawaban**:

```
L_SFT = −(1/N_jawaban) ∑_{t∈jawaban} ln P_θ(x_t | x_<t)
```

Bentuknya **persis sama** dengan pretraining (Bab 18); yang berbeda hanya **posisi mana yang ikut dijumlahkan**. Token pertanyaan diproses (masuk konteks) tapi tidak menyumbang loss. Contoh urutan 20 token (8 pertanyaan, 12 jawaban): `L = −(1/12) ∑_{t=9}^{20} ln P_θ(x_t | x_<t)`.

Skala: InstructGPT memakai ~13.000 contoh SFT — beberapa orde di bawah data pretraining (`~10⁷` token vs `~10¹¹`), tetapi mengubah perilaku secara besar. Ini menunjukkan kemampuannya sudah ada sejak pretraining.

### Reward model

Untuk banyak pertanyaan tidak ada jawaban "benar" tunggal, tetapi manusia bisa membandingkan dua jawaban. Reward model `r_φ(x, y)` mengubah perbandingan itu menjadi satu angka (biasanya salinan model + lapisan keluaran skalar).

**Model Bradley–Terry:**

```
P(y_w ≻ y_l | x) = σ( r_φ(x, y_w) − r_φ(x, y_l) )
L_RM = −ln σ( r_φ(x, y_w) − r_φ(x, y_l) )
```

Contoh `r(y_w)=2,0`, `r(y_l)=0,5`: selisih `1,5`, `σ(1,5)=0,8176`, `L=0,2014`. Jika terbalik (`0,5; 2,0`): `σ(−1,5)=0,1824`, `L=1,7014`.

| r_w − r_l | σ(·) | Loss |
|---|---|---|
| −3,0 | 0,0474 | 3,0486 |
| −1,0 | 0,2689 | 1,3133 |
| 0,0 | 0,5000 | 0,6931 |
| 1,0 | 0,7311 | 0,3133 |
| 3,0 | 0,9526 | 0,0486 |
| 5,0 | 0,9933 | 0,0067 |

**Sifat penting:** pergeseran reward (menambah konstanta ke semua) **tidak berarti** — σ hanya melihat selisih. Tetapi **skala tidak bebas**: mengali semua reward dengan 10 mengubah selisih → mengubah gradien.

### RLHF dengan PPO

```
max_θ  𝔼_{x∼D, y∼π_θ}[ r_φ(x, y) ] − β·D_KL( π_θ(·|x) ‖ π_ref(·|x) )
```

Dua suku berlawanan arah: `𝔼[r]` mendorong jawaban yang disukai; `−β·D_KL` menahan model agar tidak menyimpang jauh dari model SFT (`π_ref`, dibekukan). **Kenapa KL wajib:** tanpa itu, optimasi mencari celah pada reward model yang tidak sempurna (*reward hacking*).

Contoh efek `β` (reward +2,0, KL 5,0):

| β | 2,0 − β·5,0 | Diterima? |
|---|---|---|
| 0,01 | 1,95 | ya |
| 0,10 | 1,50 | ya |
| 0,40 | 0,00 | seimbang |
| 1,00 | −3,00 | tidak |

Algoritmanya **PPO** (membatasi seberapa besar kebijakan berubah per langkah dengan *clipping* rasio). Susunan lazim butuh 4 peran model: policy, reference, reward, value — pilihan implementasi, bukan syarat matematis.

### DPO (Direct Preference Optimization)

Apakah semua kerumitan itu perlu? DPO menunjukkan bahwa untuk objektif RLHF berpenalti KL, preferensi bisa **langsung** dipakai melatih model, tanpa reward model terpisah dan tanpa RL.

Kuncinya **log-rasio**: bukan seberapa besar probabilitas sebuah jawaban, melainkan seberapa besar **dibanding model referensi**. Memakai logaritma karena (1) probabilitas urutan menyusut sangat cepat (underflow), dan (2) rasio tidak simetris (`ln 2 = +0,69` vs `ln 0,5 = −0,69`).

```
Δ_w = ln π_θ(y_w|x) − ln π_ref(y_w|x)
Δ_l = ln π_θ(y_l|x) − ln π_ref(y_l|x)
L_DPO = −ln σ( β·(Δ_w − Δ_l) )
```

Contoh: `ln π_θ(y_w)=−10,0`, `ln π_ref(y_w)=−10,5` → `Δ_w=+0,5`; `ln π_θ(y_l)=−12,0`, `ln π_ref(y_l)=−11,5` → `Δ_l=−0,5`; `β=0,1` → argumen `0,1(0,5) − 0,1(−0,5) = 0,10`; `σ(0,10)=0,5250`; `L=0,6444` (dekat netral `0,6931`). Bentuknya sama dengan loss reward model: `−ln σ(selisih)`.

| RLHF/PPO | DPO |
|---|---|
| 4 peran model | 2 |
| Butuh sampling saat latih | tidak |
| Algoritma RL | gradient descent biasa |
| Preferensi + prompt sampling online | pasangan preferensi saja |

### LoRA (Low-Rank Adaptation)

Fine-tuning penuh 7B butuh ~112 GB state optimizer (Bab 12). LoRA menambahkan suku berperingkat rendah:

```
W' = W + ΔW = W + (α/r)·B·A
```

`W` dibekukan; hanya `B` (`d×r`) dan `A` (`r×k`) dilatih. Untuk `W_Q` 4096×4096 dengan `r=8`: `4096×8 + 8×4096 = 65.536` vs `16.777.216` penuh → **0,39%** (penghematan 256×).

**Inisialisasi:** `A` acak kecil, `B` nol → `BA = 0` di awal, sehingga `W' = W` persis. Setelah selesai, `ΔW` bisa dijumlahkan permanen → tanpa biaya tambahan saat inference.

**Batasan (jujur):** LoRA membatasi perubahan ke subruang berdimensi `r`; cukup untuk gaya/format/domain, tetapi untuk kemampuan benar-benar baru fine-tuning penuh kadang lebih unggul — temuan empiris yang bergantung tugas.

### Alur lengkap

```
1. PRETRAINING     (triliunan token; loss prediksi token)
2. SFT             (puluhan ribu instruksi; loss hanya di jawaban)
3. RLHF atau DPO   (preferensi manusia; model jadi selaras)
```

| Tahap | Loss | Yang dilatih |
|---|---|---|
| Pretraining | `−ln P_θ(x_t|x_<t)` | semua parameter |
| SFT | sama, hanya di token jawaban | semua parameter |
| Reward model | `−ln σ(r_w − r_l)` | model reward terpisah |
| RLHF | `E[r] − β·D_KL` | policy (lewat PPO) |
| DPO | `−ln σ(β·Δln-rasio)` | policy (langsung) |
| LoRA | loss apa pun di atas | hanya `A`, `B` |

## Implementasi

- [`src/llmbepees/np/alignment.py`](../../src/llmbepees/np/alignment.py): `sft_loss` (mask jawaban), `bradley_terry`, `reward_model_loss`, `rlhf_objective`, `dpo_loss`, `lora_delta`, `lora_parameters`.
- Uji memverifikasi contoh 20.2–20.6: SFT `L=1,0`, reward `0,2014/1,7014`, tabel sensitivitas, objektif RLHF `β=0,4 → 0`, DPO `0,6444`, dan `ΔW = BA` peringkat-1.

## Inti bab

Ketiga tahap memakai **tulang punggung Transformer yang sama**; yang berubah terutama **data** dan **bentuk loss**. Reward head, value head, model referensi, atau adapter LoRA hanyalah tambahan di sekelilingnya. Objektif model mentah adalah melanjutkan teks, sehingga ia bisa menjawab tetapi belum andal mengikuti instruksi; ia lalu diajari bentuk jawaban lewat contoh (SFT), kemudian dirapikan lewat perbandingan pilihan manusia (RLHF/DPO). Yang terlihat sebagai "kepribadian" dan "kepatuhan" asisten dibentuk oleh puluhan ribu contoh, di atas fondasi ratusan miliar token.

## Cek pemahaman

1. Mengapa model hasil pretraining perlu tahap tambahan sebelum andal dipakai sebagai asisten?
2. Apa yang dipelajari sebuah reward model, dan dari data seperti apa?
3. Apa yang disederhanakan DPO dibanding RLHF?

## Rujukan

- Buku *"Memahami Matematika di Balik LLM"*, Bab 20.
- Christiano dkk. (2017) — RLHF; Ouyang dkk. (2022) — InstructGPT; Schulman dkk. (2017) — PPO; Rafailov dkk. (2023) — DPO; Hu dkk. (2022) — LoRA.
