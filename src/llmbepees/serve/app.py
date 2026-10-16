# ruff: noqa: E501
"""Layanan FastAPI demo (Fase 12, ADR-0003).

Endpoint: ``GET /health``, ``POST /generate``, ``POST /explain``, dan halaman chat
``GET /``. Model yang disajikan adalah model char-level latihan sendiri (ADR-0006).
"""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from llmbepees.serve.engine import Engine

PAGE = """<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>llmbepees — demo char-level</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: system-ui, sans-serif; max-width: 820px; margin: 2rem auto; padding: 0 1rem; background:#0f1115; color:#e6e6e6; }
  h1 { font-size: 1.4rem; }
  textarea, input, button { font: inherit; }
  textarea { width: 100%; min-height: 70px; background:#181b22; color:#e6e6e6; border:1px solid #333; border-radius:8px; padding:.6rem; }
  .row { display:flex; gap:.6rem; flex-wrap:wrap; align-items:center; margin:.6rem 0; }
  input[type=number] { width: 5.5rem; background:#181b22; color:#e6e6e6; border:1px solid #333; border-radius:6px; padding:.35rem; }
  button { background:#3b82f6; color:#fff; border:0; border-radius:8px; padding:.5rem 1rem; cursor:pointer; }
  pre { background:#181b22; border:1px solid #333; border-radius:8px; padding:.7rem; white-space:pre-wrap; word-break:break-word; }
  table { border-collapse:collapse; width:100%; margin-top:.4rem; }
  th, td { border-bottom:1px solid #2a2e37; padding:.25rem .4rem; text-align:left; font-size:.9rem; }
  .muted { color:#9aa4b2; font-size:.85rem; }
</style>
</head>
<body>
  <h1>llmbepees — demo model char-level</h1>
  <p class="muted">Model kecil dilatih dari nol (<code>data/corpus.txt</code>). Kualitas teks terbatas — ini demo edukatif, bukan asisten.</p>
  <textarea id="prompt">Model bahasa</textarea>
  <div class="row">
    <label>token baru <input id="maxnew" type="number" value="120" min="1" max="400" /></label>
    <label>temperature <input id="temp" type="number" value="0.8" min="0.05" max="2" step="0.05" /></label>
    <label>top-k <input id="topk" type="number" value="0" min="0" max="50" /></label>
    <button onclick="run('/generate')">Hasilkan</button>
    <button onclick="run('/explain')">Jelaskan langkah</button>
  </div>
  <pre id="out">(keluaran muncul di sini)</pre>
<script>
async function run(path) {
  const body = {
    prompt: document.getElementById('prompt').value,
    max_new_tokens: Number(document.getElementById('maxnew').value),
    temperature: Number(document.getElementById('temp').value),
    top_k: Number(document.getElementById('topk').value) || null,
  };
  const out = document.getElementById('out');
  out.textContent = 'memproses...';
  try {
    const r = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) { out.textContent = 'galat: ' + JSON.stringify(j); return; }
    let s = j.text + '\\n\\n';
    (j.steps || []).slice(0, 8).forEach(function(st) {
      s += 'langkah: dipilih "' + st.chosen + '"  ';
      s += (st.top || []).map(function(t){ return t.token + ' ' + t.prob; }).join(' | ') + '\\n';
    });
    out.textContent = s;
  } catch (e) { out.textContent = 'galat: ' + e; }
}
</script>
</body>
</html>"""


class GenerateRequest(BaseModel):
    prompt: str = Field(default="Model bahasa", max_length=200)
    max_new_tokens: int = Field(default=120, ge=1, le=400)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=1, le=100)
    seed: int | None = None


class ExplainRequest(BaseModel):
    prompt: str = Field(default="Model bahasa", max_length=200)
    max_new_tokens: int = Field(default=12, ge=1, le=64)
    temperature: float = Field(default=1.0, gt=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=1, le=100)


class _RateLimiter:
    """Pembatas laju sederhana per IP (best-effort, cukup untuk demo publik)."""

    def __init__(self, per_minute: int = 30) -> None:
        self.per_minute = per_minute
        self.hits: dict[str, list[float]] = {}

    def allow(self, key: str) -> bool:
        now = time.time()
        window = [t for t in self.hits.get(key, []) if now - t < 60.0]
        if len(window) >= self.per_minute:
            self.hits[key] = window
            return False
        window.append(now)
        self.hits[key] = window
        return True


def create_app(engine: Engine | None = None, *, rate_limit_per_minute: int = 30) -> FastAPI:
    """Bangun aplikasi FastAPI; model default dilatih malas saat pertama dipakai."""
    app = FastAPI(title="llmbepees demo", version="0.1.0")
    state: dict[str, Engine | None] = {"engine": engine}
    limiter = _RateLimiter(rate_limit_per_minute)

    def get_engine() -> Engine:
        if state["engine"] is None:
            state["engine"] = Engine.train_default()
        return state["engine"]

    def guard(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            raise HTTPException(
                status_code=429, detail="terlalu banyak permintaan; coba lagi nanti"
            )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.post("/generate")
    def generate(request: Request, body: GenerateRequest) -> dict:
        guard(request)
        engine_obj = get_engine()
        result = engine_obj.generate(
            body.prompt,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
            seed=body.seed,
        )
        result["model"] = {
            "type": "char-level GPT (latihan sendiri)",
            "parameters": engine_obj.parameter_count,
            "vocab_size": engine_obj.tokenizer.vocab_size,
        }
        return result

    @app.post("/explain")
    def explain(request: Request, body: ExplainRequest) -> dict:
        guard(request)
        engine_obj = get_engine()
        return engine_obj.explain(
            body.prompt,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
        )

    return app


app = create_app()


__all__ = ["GenerateRequest", "ExplainRequest", "app", "create_app"]
