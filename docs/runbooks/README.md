# Runbooks

Prosedur operasional. Satu berkas per prosedur.

| Runbook | Status |
|---|---|
| [`deploy.md`](deploy.md) — build image & deploy ke EasyPanel | ⬜ belum dideploy (target https://llmbepees.hgteop.easypanel.host) |
| [`kegunaan-llm.md`](kegunaan-llm.md) — apa gunanya LLM & bagaimana cara kerjanya (edukatif) | ✅ |

## Ringkas: operasional lokal
```sh
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\pytest
.venv\Scripts\ruff check .
```

## Ringkas: menjalankan demo lokal
```sh
.venv\Scripts\python -m pip install -e ".[serve]"
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m uvicorn llmbepees.serve.app:app --port 8000
# buka http://127.0.0.1:8000/
```
