# ADR-0007: Target deploy aktual — project `m`, service `llmbepeesweb` (batas project EasyPanel)

- **Status**: Diterima
- **Tanggal**: 2026-10-17

## Konteks
[ADR-0004](0004-deploy-easypanel.md) merencanakan project EasyPanel `llmbepees` dengan service `web` dan image via **GHCR**. Saat provisi programatik via MCP dijalankan, panel menolak `createProject llmbepees` dengan **`BAD_REQUEST` — "You are not allowed to create more projects"** (batas paket akun). Selain itu, kredensial Docker yang tersedia ter-login ke **Docker Hub** (`index.docker.io`), bukan GHCR; akun memakai namespace `aasatech/*`.

## Keputusan
- Image dibangun & di-push ke **Docker Hub**: `aasatech/llmbepees-web:latest`.
- Deploy ke **project yang sudah ada `m`**, service **`llmbepeesweb`** (source: Docker image).
- Domain: **`llmbepees.hgteop.easypanel.host`** (HTTPS), port kontainer **8000**.
- Mekanisme tetap programatik via MCP (sesuai semangat ADR-0004): `createAppService` → `deployAppService` → verifikasi `/health`.

## Alasan
- Membuat project baru tidak diizinkan paket; memakai project lama adalah satu-satunya jalur tanpa mengubah paket.
- Docker Hub sudah terautentikasi; GHCR butuh kredensial tambahan yang belum tersedia.
- Wildcard domain `*.hgteop.easypanel.host` sama, sehingga host `llmbepees-web.<wildcard>` tetap sesuai rencana.

## Konsekuensi
- **Positif**: demo hidup publik, terverifikasi (`GET /health = 200`, `/generate`, `/explain`, halaman chat).
- **Negatif**: service berbagi project `m` dengan layanan lain; image di Docker Hub (bukan GHCR).
- **Lain-lain**: bila paket ditingkatkan / kredensial GHCR tersedia, migrasi ke project `llmbepees` + GHCR dapat dilakukan tanpa mengubah image aplikasi.

## Alternatif yang dipertimbangkan
- **Batalkan deploy** — artefak siap tetapi demo tidak tayang; ditolak (user memilih deploy).
- **Build di server (upload/nixpacks)** — melanggar ADR-0004 (pull-only) dan tidak perlu.
