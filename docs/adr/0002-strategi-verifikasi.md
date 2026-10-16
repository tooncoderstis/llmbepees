# ADR-0002: Verifikasi — replikasi eksperimen Bab 21 sebagai acceptance gate

- **Status**: Diterima
- **Tanggal**: 2026-10-11

## Konteks
Buku menyediakan satu eksperimen konkret (mini-LLM 188 parameter dilatih dari nol) dengan **titik pemeriksaan** terukur di Lampiran: loss `2,4480 → 0,0955`, `P(meja) ≈ 0,7492`, `P(tikar) ≈ 0,2494`, perplexity `11,57 → 1,10`, `cos(meja, tikar) 0,7557 → 0,9960`, dan routing attention akhir ≈ one-hot. Resepnya lengkap: seed 42, N(0, 0,5), Adam lr 0,05 tetap, 400 langkah, gradien beda-hingga terpusat, float64.

## Keputusan
Menjadikan **replikasi eksperimen Bab 21 sebagai acceptance gate** (Fase 10). Hasilnya di-assert di `tests/` dengan toleransi desimal ke-3. Implementasi yang tidak mereproduksi tiga pemeriksaan inti (loss mendekati 0,0937 dari atas; P(meja)≈0,75 & P(tikar)≈0,25; attention ≈ one-hot) dianggap belum benar.

## Alasan
- Memberi kriteria "selesai" yang objektif dan dapat diuji, bukan klaim subjektif.
- Membuktikan pemahaman: bisa membangun ulang dari definisi, bukan sekadar membaca.
- Buku memberi toleransi eksplisit pada desimal ketiga → realistis untuk implementasi berbeda.

## Konsekuensi
- **Positif**: CI menjadi bukti pemahaman; regresi langsung terdeteksi.
- **Negatif**: butuh disiplin tinggi meniru resep (urutan pengisian bobot, presisi, skema gradien).
- **Lain-lain**: angka rinci dapat berbeda antar-platform; penekanan pada pola, bukan bit-per-bit.

## Alternatif yang dipertimbangkan
- **Tanpa gate numerik** — hanya test unit masing-masing operasi; lebih longgar tetapi tidak membuktikan integrasi.
- **Reproduksi bit-per-bit** — terlalu rapuh terhadap perbedaan floating-point & platform.
