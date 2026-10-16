# Catatan Belajar — Memahami Matematika di Balik LLM

Ringkasan tiap bab ditulis ulang dengan bahasa sendiri (Indonesia), disertai kode pendamping di `src/llmbepees/` dan uji di `tests/`.

> Baru mulai? Baca **[PANDUAN.md](PANDUAN.md)** — cara menavigasi catatan, kode, dan uji, plus peta baca rinci.

> Sumber acuan: buku *"Memahami Matematika di Balik LLM"* (PDF di root: `LLM.pdf`, `per bab/`). Catatan ini **bukan** salinan; fokus pada konsep, rumus kunci, dan contoh angka.

## Bagian I — Fondasi (Fase 1)
- [x] [Bab 1 — Apa Sebenarnya yang Dihitung LLM?](bab-01-apa-yang-dihitung-llm.md)
- [x] [Bab 2 — Kamus Notasi: Cara Membaca Rumus](bab-02-notasi.md)
- [x] [Bab 3 — Vektor](bab-03-vektor.md)
- [x] [Bab 4 — Matriks dan Perkalian Matriks](bab-04-matriks.md)
- [x] [Bab 5 — Probabilitas](bab-05-probabilitas.md)
- [x] [Bab 6 — Teori Informasi: Entropi, Cross-Entropy, Perplexity](bab-06-informasi.md)
- [x] [Bab 7 — Turunan, Gradien, dan Aturan Rantai](bab-07-turunan.md)

## Bagian II — Komponen Neural (Fase 2–6)
- [x] [Bab 8 — Dari Teks ke Angka: Tokenisasi dan Embedding](bab-08-tokenisasi-embedding.md)
- [x] [Bab 9 — Neuron, Lapisan, dan Fungsi Aktivasi](bab-09-neuron-lapisan-aktivasi.md)
- [x] [Bab 10 — Softmax dan Fungsi Kerugian](bab-10-softmax-loss.md)
- [x] [Bab 11 — Backpropagation](bab-11-backpropagation.md)
- [x] [Bab 12 — Optimizer: SGD, Momentum, dan Adam](bab-12-optimizer.md)

## Bagian III — Transformer (Fase 7–9)
- [x] [Bab 13 — Attention: Query, Key, Value](bab-13-attention.md)
- [x] [Bab 14 — Multi-Head Attention](bab-14-multi-head-attention.md)
- [x] [Bab 15 — Informasi Posisi: Positional Encoding dan RoPE](bab-15-informasi-posisi.md)
- [x] [Bab 16 — Blok Transformer Lengkap](bab-16-blok-transformer.md)
- [x] [Bab 17 — Lapisan Keluaran dan Perjalanan Penuh Satu Token](bab-17-lapisan-keluaran.md)

## Bagian IV — Pelatihan & Praktik (Fase 10–11)
- [x] [Bab 18 — Pretraining: Objektif, Batch, dan Scaling Law](bab-18-pretraining.md)
- [x] [Bab 19 — Inference: Sampling, Temperature, dan KV Cache](bab-19-inference.md)
- [x] [Bab 20 — Setelah Pretraining: SFT, RLHF, DPO, LoRA](bab-20-sft-rlhf-dpo-lora.md)
- [x] [Bab 21 — Mengamati Mini-LLM Belajar dari Nol](bab-21-mini-llm-dari-nol.md)
- [x] [Bab 22 — Peta Besar](bab-22-peta-besar.md)
- [x] [Lampiran — Rincian Eksperimen Bab 21](lampiran-eksperimen-bab-21.md)

## Berkas pendukung
- [x] `cheatsheet.md` — ringkasan rumus & simbol
- [x] `glosarium.md` — istilah kunci

## Konvensi penulisan
- Mulai dengan "Masalah" lalu "Gagasan", diakhiri "Inti bab".
- Setiap rumus disertai satu contoh angka yang bisa dihitung ulang.
- Tautkan ke kode (`src/llmbepees/...`) dan uji (`tests/...`) yang relevan.
