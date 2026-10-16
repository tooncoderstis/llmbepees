# AGENTS — llmbepees

> Proyek: **asisten domain faktual & anti-halusinasi** yang dibangun dari nol — tokenizer BPE, model GPT PyTorch, dan **grounding (RAG)** atas dokumen sendiri (abstain bila tak ada dasar).
> Starter (struktur, kode, uji, CI, deploy, hub, catatan 22 bab) disalin dari `aasallm` lalu di-rename; lihat `docs/adr/0008-starter-dari-aasallm.md`.

## Project Bootstrap (WAJIB DIBACA DULU)

> Hemat token & jaga konteks antar sesi. Urutan baca:
> 1. `STATUS.md` — kondisi & pekerjaan terkini (satu-satunya sumber kebenaran progres).
> 2. `PRD.md` — **jangan baca penuh**; hanya buka section yang relevan dengan tugas.
> 3. `docs/adr/` — 1–2 ADR yang terkait perubahan.
> 4. Ekor `CHANGELOG.md` bila perlu konteks rilis.

Aturan kerja:
- Ikuti DoD: `kode → test → formatter → entri CHANGELOG → update STATUS → ADR bila keputusan → npm run hub:report`.
- **Mulai fase**: tandai baris fase di `STATUS.md` menjadi 🔄 lalu jalankan `npm run hub:report` (status hub jadi *dikerjakan*). **Fase selesai**: tandai ✅ lalu `npm run hub:report` lagi (status hub jadi *selesai*). Tiap perubahan status memicu notifikasi (inbox + Web Push) di aplikasi portofolio.
- Conventional Commits; jangan commit rahasia (`.env` di-ignore).
- Setiap route/endpoint ber-mutasi harus punya test.
- Deploy ke EasyPanel dilakukan **programatik** (MCP/CLI), bukan klik UI manual — lihat runbook deploy.
- Proyek ini **melapor ke hub portofolio** (lihat `docs/hub/README.md`).
- Untuk proyek baru/scaffold, gunakan skill `aasaprojectkit`.

## Konvensi kode
- **NumPy-first, PyTorch-second**: `src/llmbepees/np/` adalah rujukan kebenaran; `src/llmbepees/torch/` harus setara (uji paritas).
- Catatan belajar & dokumen proyek: **bahasa Indonesia**.
- Verifikasi mengikuti titik pemeriksaan buku (toleransi desimal ke-3); lihat ADR-0002.
- Formatter/linter: `ruff` (`ruff check .`, `ruff format`). Test: `pytest`.

## Perintah cepat
```sh
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\ruff check .
.venv\Scripts\pytest
npm run hub:report
```

## Sumber acuan
- `LLM.pdf` (buku lengkap), `per bab/bab NN.pdf` (per bab). PDF **tidak** didaur ulang mentah; ditulis ulang bahasa sendiri.
- Tidak ada angka yang diklaim tanpa rujukan/verifikasi.
