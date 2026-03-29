"""
Unified secret / configuration reader.

Works transparently across three deployment modes:

  1. **Streamlit Cloud** — reads from ``st.secrets`` (set in the app dashboard)
  2. **Docker / local with .env** — reads from environment variables
  3. **Local with .streamlit/secrets.toml** — Streamlit loads this into
     ``st.secrets`` automatically

Call ``get_secret("KEY")`` everywhere instead of ``os.getenv("KEY")``.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st


def get_secret(key: str, default: str | None = None) -> str | None:
    """Return the value for *key*, checking ``st.secrets`` first, then env."""
    try:
        val = st.secrets.get(key)
        if val is not None:
            return str(val)
    except Exception:
        pass

    return os.getenv(key, default)
