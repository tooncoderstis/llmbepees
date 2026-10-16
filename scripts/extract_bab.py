"""Ekstrak teks satu bab PDF ke stdout — utilitas penulisan catatan belajar.

PDF buku tidak di-commit (hak cipta); ini hanya alat baca lokal. Butuh ``pypdf``
(tersedia di extra ``dev``)::

    .venv\\Scripts\\python scripts\\extract_bab.py "per bab\\bab 09.pdf"
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print('Pakai: python scripts/extract_bab.py "per bab/bab 09.pdf"', file=sys.stderr)
        return 2

    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:
        print('Butuh pypdf. Jalankan: pip install -e ".[dev]"', file=sys.stderr)
        return 1

    sys.stdout.reconfigure(encoding="utf-8")
    reader = PdfReader(argv[1])
    for i, page in enumerate(reader.pages, 1):
        print(f"\n===== PAGE {i} =====")
        print(page.extract_text() or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
