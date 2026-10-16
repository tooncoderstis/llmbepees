FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY scripts ./scripts

RUN pip install --no-cache-dir -e ".[serve]" \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "llmbepees.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
