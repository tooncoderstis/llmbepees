# ADR-0008: Starter `llmbepees` disalin dari `aasallm` + arah asisten domain (grounding)

- **Status**: Diterima
- **Tanggal**: 2026-10-17

## Konteks
Pengguna ingin proyek **terpisah** bernama `llmbepees` yang berfokus pada **asisten domain yang faktual & anti-halusinasi**, tetapi **memakai starter yang sama persis** dengan `aasallm` (struktur paket NumPy→PyTorch→FastAPI, uji, CI, deploy, hub, catatan belajar 22 bab). Komputasi terbatas **CPU** (EasyPanel), tanpa GPU.

## Keputusan
- **Salin penuh (mirror)** file ter-track `aasallm` → `llmbepees` (tanpa riwayat `.git`, tanpa `.env`/`.venv`/PDF), lalu **rename**: paket `src/aasallm` → `src/llmbepees`, seluruh referensi teks `aasallm` → `llmbepees`, domain `llmbepees.hgteop.easypanel.host`.
- **Reset dokumen keadaan** (`STATUS.md`, `PRD.md`, `CHANGELOG.md`, `README.md`, `AGENTS.md`) untuk proyek baru; catatan belajar & ADR dipertahankan sebagai rujukan.
- **Arah teknis**: faktualitas dicapai dengan **grounding (RAG)** — model menjawab hanya dari konteks dokumen yang diambil (retriever + embedding sendiri), dan **abstain** bila tidak ada. Model dilatih sendiri: **BPE** + GPT sedang (CPU) + **SFT**.
- **Deploy**: mengikuti [ADR-0007](0007-target-deploy-aktual.md) — project baru EasyPanel ditolak (batas paket) → project `m`, service `llmbepeesweb`, domain `llmbepees.hgteop.easypanel.host`, image Docker Hub `aasatech/llmbepees-web`.

## Alasan
- Menyalin starter menjamin konvensi (DoD, CI, hub, deploy programatik) konsisten dan menghemat waktu.
- Grounding adalah satu-satunya cara realistis mencapai "faktual & anti-halusinasi" pada model kecil di CPU.
- Memakai Kembali mekanisme deploy yang sudah terbukti (ADR-0007) menghindari blokir batas paket.

## Konsekuensi
- **Positif**: mulai cepat; konvensi seragam; arah akurasi jelas & terukur.
- **Negatif**: warisan docstring/tautan berkaitan buku perlu dirapikan bertahap; dua repo berbagi nenek moyang kode.
- **Lain-lain**: perubahan perbaikan di satu repo **tidak** otomatis tersinkron ke yang lain (sengaja — proyek terpisah).

## Alternatif yang dipertimbangkan
- **Scaffold baru dari nol (aasaprojectkit)** tanpa menyalin isi — ditolak; pengguna ingin starter persis.
- **Monorepo** (paket baru dalam `aasallm`) — ditolak; pengguna ingin proyek terpisah.
