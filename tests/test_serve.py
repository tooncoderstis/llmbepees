"""Uji layanan FastAPI demo (Fase 12, ADR-0003). Dilewati bila fastapi/torch tidak ada."""

import pytest

pytest.importorskip("fastapi")
torch = pytest.importorskip("torch")

from fastapi.testclient import TestClient  # noqa: E402

from llmbepees.serve.app import create_app  # noqa: E402
from llmbepees.serve.engine import Engine  # noqa: E402
from llmbepees.torch.model import GPT, CharTokenizer, GPTConfig  # noqa: E402


def _tiny_engine() -> Engine:
    tokenizer = CharTokenizer("abca caba abcb")
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        n_heads=1,
        n_layers=1,
        d_ff=32,
        max_len=8,
    )
    return Engine(tokenizer, GPT(config), max_len=8)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app(_tiny_engine()))


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "demo" in response.text
    assert "/generate" in response.text


def test_generate(client: TestClient) -> None:
    response = client.post(
        "/generate", json={"prompt": "ab", "max_new_tokens": 5, "temperature": 1.0, "seed": 0}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["prompt"] == "ab"
    assert body["text"].startswith("ab")
    assert len(body["steps"]) == 5
    assert body["model"]["vocab_size"] == 4


def test_explain_returns_probabilities(client: TestClient) -> None:
    response = client.post("/explain", json={"prompt": "ab", "max_new_tokens": 3})
    assert response.status_code == 200
    steps = response.json()["steps"]
    assert len(steps) == 3
    top = steps[0]["top"]
    assert top and "token" in top[0] and "prob" in top[0]


def test_rate_limit() -> None:
    limited = TestClient(create_app(_tiny_engine(), rate_limit_per_minute=1))
    assert limited.post("/explain", json={"prompt": "ab", "max_new_tokens": 1}).status_code == 200
    assert limited.post("/explain", json={"prompt": "ab", "max_new_tokens": 1}).status_code == 429
