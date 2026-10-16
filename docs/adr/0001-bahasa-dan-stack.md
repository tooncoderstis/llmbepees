# ADR-0001: Bahasa & stack — Python, NumPy-first lalu PyTorch

- **Status**: Diterima
- **Tanggal**: 2026-10-11

## Konteks
Proyek ini bertujuan **memahami matematika LLM** sekaligus membangunnya. Buku acuan menekankan perhitungan eksplisit (dot product, softmax, LayerNorm, cross-entropy, gradien) dan secara eksplisit *tidak* membahas pemrograman PyTorch. Framework otomatis (autograd) menyembunyikan justru bagian yang ingin dipahami.

## Keputusan
Implementasi inti ditulis dengan **NumPy murni** (`src/llmbepees/np/`) sebagai rujukan kebenaran yang transparan. Setelah tervalidasi, port ke **PyTorch** (`src/llmbepees/torch/`) untuk pelatihan & skala. Catatan belajar ditulis dalam **bahasa Indonesia**. Runtime: **Python 3.11**, paket bernama `llmbepees`.

## Alasan
- NumPy memaksa setiap langkah matematis ditulis manual → pemahaman nyata.
- PyTorch dibutuhkan untuk performa pada model kecil (Fase 11) tanpa mengorbankan versi transparan.
- Catatan bahasa Indonesia mengikuti konteks pengguna & buku.

## Konsekuensi
- **Positif**: kode pendidikan yang dapat diaudit baris-per-baris; paritas NumPy↔PyTorch menjadi uji kuat.
- **Negatif**: implementasi ganda (NumPy + PyTorch) menambah pekerjaan.
- **Lain-lain**: `torch` sebagai extra opt-in (unduhan besar) agar tidak membebani kontributor awal.

## Alternatif yang dipertimbangkan
- **Langsung PyTorch saja** — lebih cepat sampai model jalan, tetapi menyembunyikan matematika.
- **NumPy murni saja** — paling edukatif, tetapi lambat/tidak praktis untuk model kecil & demo.
