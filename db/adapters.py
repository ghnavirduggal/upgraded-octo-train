from __future__ import annotations

import os
import sqlite3
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse, unquote
try:
    from urllib.request import url2pathname
except ImportError:
    url2pathname = lambda s: s  # type: ignore


class DatabaseAdapter:
    """Base adapter used by cap_db/cap_store."""

    driver = "generic"
    row_type = None

    def connect(self):
        raise NotImplementedError

    def info(self) -> dict:
        return {"driver": self.driver}


class SQLiteAdapter(DatabaseAdapter):
    driver = "sqlite"
    row_type = sqlite3.Row

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def connect(self):
        cx = sqlite3.connect(self.path, check_same_thread=False)
        cx.row_factory = sqlite3.Row
        return cx

    def info(self) -> dict:
        base = super().info()
        base.update(path=self.path)
        return base


AdapterFactory = Callable[[str], DatabaseAdapter]

_ADAPTER: Optional[DatabaseAdapter] = None
_ADAPTER_INFO: Dict[str, Any] = {}


def _sqlite_from_url(url: str) -> SQLiteAdapter:
    parsed = urlparse(url)
    if parsed.scheme not in ("sqlite", ""):
        raise ValueError(f"Invalid sqlite URL: {url}")
    if parsed.scheme == "sqlite":
        netloc = parsed.netloc or ""
        path = parsed.path or ""
        if netloc:
            raw = f"//{netloc}{path}"
        else:
            raw = path
    else:
        raw = url
    raw = unquote(raw or "")
    if not raw:
        raise ValueError("SQLite path missing")
    native = url2pathname(raw)
    if os.name == "nt":
        native = native.lstrip("\\/")
    return SQLiteAdapter(native)


def configure_adapter(url: str | None = None, *, default_path: str | None = None) -> DatabaseAdapter:
    """Create global adapter based on CAP_DB_URL or fallback path."""
    global _ADAPTER, _ADAPTER_INFO
    db_url = (url or os.getenv("CAP_DB_URL") or "").strip()
    if not db_url:
        if not default_path:
            raise ValueError("No CAP_DB_URL and no default path provided.")
        db_url = f"sqlite:///{os.path.abspath(default_path)}"
    scheme = urlparse(db_url).scheme or "sqlite"
    if scheme == "sqlite":
        adapter = _sqlite_from_url(db_url)
    else:
        raise NotImplementedError(
            f"No adapter registered for scheme '{scheme}'. Create one in db/adapters.py."
        )
    _ADAPTER = adapter
    _ADAPTER_INFO = adapter.info()
    _ADAPTER_INFO["url"] = db_url
    return adapter


def get_adapter() -> DatabaseAdapter:
    if _ADAPTER is None:
        raise RuntimeError("Database adapter not configured. Call configure_adapter() first.")
    return _ADAPTER


def describe_current_adapter() -> dict:
    return dict(_ADAPTER_INFO)
