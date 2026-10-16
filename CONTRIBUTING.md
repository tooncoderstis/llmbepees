# Kontribusi

## Alur
1. Buat branch dari `main` (`feat/...`, `fix/...`).
2. Kerjakan perubahan + test.
3. Jalankan formatter/linter dan test lokal hingga hijau.
4. Buka Pull Request (isi template). CI harus lolos.
5. Squash merge dengan pesan Conventional Commits.

## Definisi Selesai (DoD)
`kode + test → formatter → entri CHANGELOG → update STATUS.md → ADR bila ada keputusan → npm run hub:report`.

## Konvensi commit
[Conventional Commits](https://www.conventionalcommits.org): `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.

## Rahasia
Jangan pernah commit `.env` atau kredensial. Gunakan `.env.example`.

## Verifikasi numerik
Mengikuti ADR-0002: hasil harus cocok dengan titik pemeriksaan buku pada toleransi desimal ke-3, diuji di `tests/`.
