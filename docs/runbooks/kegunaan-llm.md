# Runbook — Apa gunanya LLM & bagaimana cara kerjanya

> Ringkas edukatif untuk pengguna demo. Menjawab dua pertanyaan: **apa gunanya** dan **bagaimana cara kerjanya**. Detail rumus ada di `docs/belajar/`.

## Apa gunanya
Sebuah LLM adalah fungsi yang memetakan **deretan token** menjadi **distribusi probabilitas token berikutnya** (Bab 1). Dari satu tugas sederhana itu muncul kemampuan menulis, menerjemahkan, meringkas, dan memprogram — karena untuk menebak kelanjutan teks dengan baik, model harus menangkap pola bahasa, fakta, dan gaya.

## Bagaimana cara kerjanya (alur satu token)
```
teks → tokenisasi (BPE) → embedding + posisi
     → L × [ attention → FFN ]  (residual + normalisasi)
     → lapisan keluaran (logit = keadaan akhir · embedding)
     → softmax → distribusi probabilitas
     → pilih satu token (greedy / sampling / temperature / top-k / top-p)
     → ulangi (autoregresif)
```
- **Attention** memindahkan informasi antarposisi (satu-satunya jembatan; Bab 13–14).
- **FFN** mengolah informasi di dalam tiap posisi (Bab 9) — memuat ~⅔ parameter tiap blok.
- **Posisi** disuntikkan karena attention buta urutan (Bab 15).
- **Normalisasi + residual** membuat tumpukan bisa dilatih (Bab 16).
- Pelatihan: `loss = −ln P(token benar)`, gradien `p − y`, optimizer Adam (Bab 6, 10–12).

## Yang bisa dilihat di demo ini
- `/generate` — menghasilkan teks dari prompt.
- `/explain` — menampilkan **logit/probabilitas tiap langkah sampling**, sehingga terlihat model sebenarnya "hanya memilih dari distribusi".
- Halaman `/` — antarmuka chat sederhana + tabel top-k probabilitas.

## Batas jujur
- Model demo adalah GPT **char-level** kecil yang dilatih dari korpus mini — kualitas teksnya terbatas. Tujuannya **memperlihatkan mekanisme**, bukan menjadi asisten.
- Mini-LLM 188 parameter (Bab 21) dipakai untuk **verifikasi matematika**, bukan untuk demo (lihat [ADR-0006](../adr/0006-model-demo-dan-data.md)).
- Model hanya melakukan **jalur maju** saat dipakai; parameter sudah dibekukan setelah pelatihan (Bab 19).
