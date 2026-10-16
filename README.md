# llmbepees

**Asisten domain yang faktual & anti-halusinasi**, dibangun dari nol: tokenizer BPE, model GPT PyTorch, dan **grounding (RAG)** atas dokumen sendiri — jawaban dikutip dari konteks, dan sistem **abstain** ("tidak tahu") bila tidak ada dasar.

> Starter proyek (struktur, kode, uji, CI, deploy, hub, catatan 22 bab) disalin dari [`aasallm`](https://github.com/tooncoderstis/aasallm) lalu diganti nama — lihat [`docs/adr/0008-starter-dari-aasallm.md`](docs/adr/0008-starter-dari-aasallm.md).

## Cara kerja (ringkas)
```
Dokumen → chunk → embed (sendiri) → retriever → konteks → model sendiri (SFT) → jawab HANYA dari konteks → (abstain bila kosong)
```

## Fitur (target)
- Tokenizer subword (BPE) & model GPT yang dilatih sendiri (CPU).
- SFT (loss hanya di jawaban) untuk perilaku tanya-jawab.
- Retrieval + decoding terkendali (abstain) → faktual & anti-halusinasi.
- Evaluasi terukur: akurasi faktual, halusinasi, abstain.
- Layanan FastAPI (`/health`, `/generate`, `/explain`, halaman chat).

## Stack
- Python 3.11 · NumPy (rujukan) · PyTorch (model) · FastAPI/uvicorn (layanan)
- pytest + ruff · GitHub Actions · EasyPanel (source: Docker image)

## Menjalankan & menguji
```sh
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\ruff check .
.venv\Scripts\pytest
```

## Dokumentasi
- Kebutuhan: [`PRD.md`](PRD.md)
- Status: [`STATUS.md`](STATUS.md)
- Riwayat rilis: [`CHANGELOG.md`](CHANGELOG.md)
- Keputusan: [`docs/adr/`](docs/adr/)
- Catatan belajar (rujukan): [`docs/belajar/`](docs/belajar/)
- Runbook: [`docs/runbooks/`](docs/runbooks/)

## Kontribusi
Lihat [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Sumber acuan
Buku *"Memahami Matematika di Balik LLM"* (`LLM.pdf`, `per bab/`, tidak di-commit) dipakai sebagai rujukan belajar; catatan di repo ini ditulis ulang dengan bahasa sendiri.
