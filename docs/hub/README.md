# Integrasi Hub Portofolio

Proyek ini melapor progres ke hub portofolio. Setiap perubahan status fase di `STATUS.md` (🔄 mulai / ✅ selesai) memicu notifikasi (inbox + Web Push).

- Skrip: [`scripts/hub-report.mjs`](../../scripts/hub-report.mjs)
- Perintah: `npm run hub:report`
- Env (di `.env`, jangan di-commit): `HUB_INGEST_URL`, `HUB_INGEST_SECRET` (opsional `HUB_PROJECT_SLUG`, `HUB_PROJECT_NAME`).
- Konvensi lengkap: `docs/hub/project-convention.md` di repo **portofolio**.

## Format fase di STATUS.md
Baris tabel `| id | judul | status |`, emoji: `✅` selesai · `🔄` dikerjakan · `⬜` belum.

## Alur
1. Mulai fase → tandai 🔄 → `npm run hub:report`.
2. Fase selesai → tandai ✅ → `npm run hub:report`.
3. Laporan pertama hanya menyinkronkan (tanpa notifikasi).
