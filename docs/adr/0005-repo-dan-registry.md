# ADR-0005: Repo GitHub publik + image GHCR publik

- **Status**: Diterima
- **Tanggal**: 2026-10-11

## Konteks
Proyek bersifat edukatif (buku + kode belajar) dan demo publik. Image harus dapat ditarik EasyPanel tanpa menanamkan kredensial registry.

## Keputusan
Repo GitHub **publik** bernama `llmbepees`; image **GHCR publik** (`ghcr.io/<owner>/llmbepees:<tag>`). Tidak ada data pribadi atau rahasia di dalam repo maupun image.

## Alasan
- Repo publik sesuai semangat belajar terbuka.
- Image GHCR publik → EasyPanel pull tanpa kredensial registry (menyederhanakan provisi).
- Menghindari pengelolaan service account/pat untuk registry privat.

## Konsekuensi
- **Positif**: distribusi mudah; demo dapat diakses siapa pun.
- **Negatif**: wajib disiplin agar tidak ada rahasia ter-commit (mitigasi: `.env` di-ignore + gitleaks di CI).
- **Lain-lain**: lisensi & atribusi buku acuan perlu dicantumkan.

## Alternatif yang dipertimbangkan
- **Repo privat + GHCR privat** — lebih tertutup, tetapi menambah kerumitan kredensial registry di EasyPanel.
