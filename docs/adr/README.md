# Architecture Decision Records (ADR)

Keputusan arsitektur & teknologi penting untuk `llmbepees`. Satu keputusan = satu berkas, format Nygard ringkas (lihat [`0000-template.md`](0000-template.md)).

## Status
`Diterima` · `Ditolak` · `Digantikan oleh ADR-XXXX` · `Deprecated`

## Daftar
| # | Judul | Status |
|---|---|---|
| [0001](0001-bahasa-dan-stack.md) | Bahasa & stack: Python, NumPy-first lalu PyTorch | Diterima |
| [0002](0002-strategi-verifikasi.md) | Verifikasi: replikasi eksperimen Bab 21 sebagai acceptance gate | Diterima |
| [0003](0003-bentuk-demo.md) | Bentuk demo: FastAPI (API inference + halaman chat) | Diterima |
| [0004](0004-deploy-easypanel.md) | Deploy EasyPanel: source Docker image via GHCR, programatik MCP | Diterima |
| [0005](0005-repo-dan-registry.md) | Repo GitHub publik + image GHCR publik | Diterima |
| [0006](0006-model-demo-dan-data.md) | Model untuk demo & data latihnya (mini-LLM = verifikasi; demo = model char-level) | Diterima |
| [0007](0007-target-deploy-aktual.md) | Target deploy aktual: project `m`, service `llmbepeesweb` (batas project EasyPanel) | Diterima |
| [0008](0008-starter-dari-aasallm.md) | Starter disalin dari `aasallm` + arah asisten domain (grounding) | Diterima |

## Konvensi
- Nomor urut 4 digit, tanpa lompat.
- ADR lama tidak diedit setelah diterima; buat ADR baru yang menggantikannya.
