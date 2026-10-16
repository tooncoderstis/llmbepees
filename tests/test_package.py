"""Smoke test paket llmbepees (Fase 0)."""

import llmbepees
import llmbepees.np
import llmbepees.serve
import llmbepees.torch


def test_version() -> None:
    assert llmbepees.__version__ == "0.1.0"


def test_subpackage_docstrings() -> None:
    assert llmbepees.np.__doc__
    assert llmbepees.torch.__doc__
    assert llmbepees.serve.__doc__
