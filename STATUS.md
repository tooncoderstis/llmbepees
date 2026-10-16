# STATUS — llmbepees

> **Sumber kebenaran status & progres.** Baca file ini dulu sebelum melanjutkan pekerjaan.
> Riwayat rilis: [`CHANGELOG.md`](CHANGELOG.md) · Keputusan: [`docs/adr/`](docs/adr/) · Kebutuhan: [`PRD.md`](PRD.md).
> **Jangan simpan kredensial di file ini.**

- **Terakhir diperbarui**: 2026-10-17
- **Versi produksi aktif**: — (belum dideploy)
- **Platform**: Python 3.11 (NumPy → PyTorch + FastAPI); deploy EasyPanel (Docker image)

## TL;DR (konteks 30 detik)
`llmbepees` adalah proyek **asisten domain yang faktual & anti-halusinasi**, dibangun dari nol. Starter (struktur, kode, uji, CI, deploy, hub) **disalin dari `aasallm`** lalu diganti nama (lihat [ADR-0008](docs/adr/0008-starter-dari-aasallm.md)); catatan belajar 22 bab ikut sebagai rujukan. Arah: tokenizer **BPE**, model GPT sendiri (sedang, CPU), **SFT**, dan **grounding (RAG)** atas dokumen sendiri sehingga jawaban mengutip konteks (abstain bila tak ada). **Fase 0 (Setup & Fondasi) dikerjakan** 🔄.

## Checklist Tahap
| Fase | Judul | Status |
|---|---|---|
| 0 | Setup & Fondasi (starter dari aasallm) | 🔄 |
| 1 | Tokenizer BPE & dataset domain | ⬜ |
| 2 | Model sedang + pretraining domain | ⬜ |
| 3 | SFT (instruksi/tanya-jawab) | ⬜ |
| 4 | Grounding (retrieval/RAG) + abstain | ⬜ |
| 5 | Evaluasi faktual & anti-halusinasi | ⬜ |
| 6 | Integrasi layanan + deploy CPU | ⬜ |

> Fase dikerjakan: tandai 🔄 saat mulai dan ✅ saat selesai, lalu jalankan `npm run hub:report` di tiap perubahan — hub portofolio mengirim notifikasi (inbox + Web Push).

## Cara melanjutkan (resume) — baca ini saat membuka proyek lagi
1. **Baca `STATUS.md` ini** — kondisi & langkah berikutnya ada di sini (satu-satunya sumber kebenaran progres).
2. **Pasang lingkungan**:
   `python -m venv .venv` lalu `.venv\Scripts\python -m pip install -e ".[dev]"`; `torch` (CPU) untuk model/uji paritas; `fastapi`/`uvicorn`/`httpx` bila menjalankan layanan.
3. **Verifikasi cepat**: `.venv\Scripts\ruff check .` dan `.venv\Scripts\pytest`.
4. **Mulai Fase 1**: tandai baris Fase 1 menjadi 🔄 di tabel di atas → `npm run hub:report`; lalu ikuti **DoD** di `AGENTS.md` (kode → test → formatter → CHANGELOG → STATUS → ADR bila perlu → hub:report).
5. **Panduan membaca** proyek & catatan: `docs/belajar/PANDUAN.md`.

Kondisi ringkas: **starter selesai disalin & di-rename**; siap mulai Fase 1 (BPE & dataset).

## Kondisi saat ini
- **Fase 0 (Setup & Fondasi)** 🔄: starter `aasallm` disalin penuh & di-rename → `llmbepees` (paket `src/llmbepees`, uji, CI, deploy, hub).
- Kode starter: `src/llmbepees/{np,torch,serve}` + `tests/` + `scripts/` + `deploy/` + catatan `docs/belajar/`.
- CI GitHub Actions (ruff + pytest + gitleaks) sudah ada; repo GitHub menyusul saat push.

## Yang belum selesai / menunggu
| Item | Catatan |
|---|---|
| Repo GitHub | Belum dibuat/push (rencana publik `llmbepees`) |
| Secret hub | `HUB_INGEST_SECRET` di `.env` (dari repo portofolio); tidak di-commit |
| Secret EasyPanel | MCP `easypanel` aktif; project baru ditolak (batas paket) → deploy di project `m`, service `llmbepeesweb` |
| Deploy | Belum (rencana Fase 6) — domain `llmbepees.hgteop.easypanel.host` |
| Domain asisten | Grounding & data dokumen domain belum disiapkan (menunggu dokumen dari pemilik) |

## Keputusan & kesimpulan penting
- **Starter dari `aasallm`** (mirror penuh) lalu rename; paket Python `llmbepees`. Lihat [ADR-0008](docs/adr/0008-starter-dari-aasallm.md).
- **Faktual & anti-halusinasi dicapai dengan grounding** (RAG), bukan mengandalkan bobot model kecil; model menjawab hanya dari konteks dan abstain bila tidak ada.
- **CPU-only** (EasyPanel): model sedang (±5–20 jt param) dilatih sekali, lalu hanya inference.
- PDF buku (`LLM.pdf`, `per bab/`) **tidak** disalin ke repo (hak cipta); hanya catatan turunan.

## Cara menjalankan & menguji
```sh
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\ruff check .
.venv\Scripts\ruff format --check .
.venv\Scripts\pytest
npm run hub:report
```

Menjalankan demo lokal: `.venv\Scripts\python -m uvicorn llmbepees.serve.app:app --port 8000` lalu buka `http://127.0.0.1:8000/`.

## Referensi cepat
- PRD: `PRD.md` · Changelog: `CHANGELOG.md` · ADR: `docs/adr/README.md` · Runbook: `docs/runbooks/` · Catatan: `docs/belajar/` · Panduan membaca: `docs/belajar/PANDUAN.md`
