# ADR-0006: Model untuk demo & data latihnya

- **Status**: Diterima
- **Tanggal**: 2026-10-11

## Konteks
Ada dua "model" berbeda dalam rencana dan keduanya sering tertukar:
1. **Mini-LLM Bab 21** — toy model untuk memverifikasi matematika. Dilatih hanya dari **4 kalimat** (korpus "kucing duduk di atas meja/tikar", 7 token/kalimat → 24 pasangan input–target), vocab 8 token, 188 parameter, batch penuh, 400 langkah. Ia hanya meniru proporsi data (`P(meja)≈0,75`, `P(tikar)≈0,25`) dan tidak bisa generalisasi/mengobrol.
2. **Model Fase 11** — model kecil char-level yang dilatih pada korpus teks nyata (Indonesia), dengan tokenizer & sampling sungguhan.

Demo web (Fase 12, ADR-0003) berencana "menyajikan checkpoint hasil latihan sendiri". Jika yang disajikan mini-LLM Bab 21, demo akan tampak rusak/ tak berguna.

## Keputusan
- **Mini-LLM Bab 21 tetap ada**, tetapi perannya murni **alat verifikasi** (acceptance gate Fase 10, ADR-0002) — bukan model yang didemo.
- **Yang didemo adalah model Fase 11** (char-level, dilatih pada korpus nyata kecil), lengkap dengan tokenizer + sampling.
- Batas kemampuan model demo **dinyatakan jujur** di halaman chat & runbook (bukan model berbahasa umum).

## Alasan
- Mini-LLM 4 kalimat memang dirancang buku untuk melacak angka, bukan untuk bahasa.
- Demo edukatif tetap bermakna bila model dapat menghasilkan teks yang masuk akal dari korpus nyata (walau kecil).
- Memisahkan "verifikasi" dan "demo" menghindari ekspektasi keliru.

## Konsekuensi
- **Positif**: peran tiap model jelas; demo bermakna; uji numerik tetap tajam.
- **Negatif**: Fase 11 harus menyertakan alur data (korpus + tokenizer), menambah pekerjaan.
- **Lain-lain**: ukuran model demo dibatasi agar dapat dilatih/di-serve di CPU.

## Alternatif yang dipertimbangkan
- **Demo = mini-LLM Bab 21** — ditolak; output tak berguna untuk chat.
- **Demo = model siap pakai (HuggingFace)** — ditolak; keluar dari narasi "dari nol" (ADR-0003).
