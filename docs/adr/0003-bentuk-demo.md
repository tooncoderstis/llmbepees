# ADR-0003: Bentuk demo — FastAPI (API inference + halaman chat)

- **Status**: Diterima
- **Tanggal**: 2026-10-11

## Konteks
Tujuan pengguna adalah memahami **apa gunanya LLM dan bagaimana cara kerjanya**, bukan sekadar melatih model. Perlu wajah yang dapat dijalankan agar kegunaan dan mekanisme terlihat langsung.

## Keputusan
Membangun layanan **FastAPI** di `src/llmbepees/serve/` dengan:
- `GET /health` — status layanan.
- `POST /generate` — menghasilkan token dari model (greedy/sampling).
- `POST /explain` — mengembalikan logit/probabilitas tiap langkah + langkah sampling (menunjukkan "cara kerjanya").
- Halaman chat web sederhana (static) yang menampilkan top-k probabilitas tiap langkah.

Model yang disajikan adalah **checkpoint hasil latihan sendiri** (edukatif, char-level kecil), bukan model siap pakai.

## Alasan
- `/explain` menjadikan mekanisme internal (distribusi probabilitas, sampling) terlihat — inti pesan edukatif.
- Satu layanan ringan (FastAPI + uvicorn) mudah dikontainerkan & di-deploy ke EasyPanel.
- Menyajikan model latihan sendiri menjaga konsistensi narasi "dari nol".

## Konsekuensi
- **Positif**: demo menjelaskan konsep tanpa perlu membaca kode; deploy sederhana (satu image).
- **Negatif**: kualitas teks model latihan sendiri terbatas (dibatasi jujur di UI/runbook).
- **Lain-lain**: API inference di CPU; perlu rate-limit sederhana agar aman publik.

## Alternatif yang dipertimbangkan
- **API saja** — lebih ringkas, tetapi tidak memvisualkan probabilitas/sampling.
- **Serve model HuggingFace siap pakai** — benar-benar bisa ngobrol, tetapi keluar dari narasi "dari nol".
