# ADR-0004: Deploy EasyPanel — source Docker image via GHCR, programatik MCP

- **Status**: Diterima
- **Tanggal**: 2026-10-11

## Konteks
Standar Aasa: deploy **programatik** ke EasyPanel, bukan klik UI. EasyPanel hanya menarik (pull) image — tidak membangun di server. Dibutuhkan registry untuk image dan cara provisi otomatis.

## Keputusan
- **Build & publish** image di CI (GitHub Actions) ke **GHCR** saat push ke `main`; image publik.
- EasyPanel menggunakan **Source: Docker Image**.
- Provisi & deploy dijalankan **programatik via MCP** (`search_procedures` → `listProjectsAndServices` → `createProject` `llmbepees` → `createAppService` `web` (image) → `createDomain` `llmbepees-web.<wildcard>`, https → `deployAppService`) lalu verifikasi `GET /health = 200`.
- Secret (`EASYPANEL_API_KEY`) di `.env` lokal / env panel; **tidak** di-commit.
- Berkas dukungan: `deploy/easypanel-docker/` (Dockerfile, nginx, supervisord, build-push) & `deploy/easypanel/` (`deploy.mjs`, runbook).

## Alasan
- Sesuai standar Aasa (programatik, image-based, pull-only di server).
- Image publik di GHCR menghindari kebutuhan kredensial registry di EasyPanel.
- MCP menghindari langkah manual & dapat direproduksi.

## Konsekuensi
- **Positif**: deploy satu perintah, dapat diulang, audit trail jelas.
- **Negatif**: bergantung pada CI (build harus hijau) & akses panel/MCP.
- **Lain-lain**: perubahan env perlu restart service; image besar memperlambat pull.

## Alternatif yang dipertimbangkan
- **Build di server (Nixpacks/buildpack)** — ditolak; lambat & tidak sesuai standar.
- **CLI `easypanel` sebagai utama** — dipertahankan hanya sebagai fallback tanpa MCP.
