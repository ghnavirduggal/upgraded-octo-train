# cap_db.py — backend adapter + dataset helpers
from __future__ import annotations
import os, json, datetime as dt
import pandas as pd
from typing import Optional

from db.adapters import configure_adapter, get_adapter
from db.schema import schema_for_backend

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "capability.sqlite3")
DB_PATH = DEFAULT_DB_PATH

# Configure adapter as soon as module loads so other modules share it
configure_adapter(default_path=DEFAULT_DB_PATH)


def _update_db_path_from_adapter():
    global DB_PATH
    adapter = get_adapter()
    info = adapter.info()
    if adapter.driver == "sqlite":
        DB_PATH = info.get("path", DB_PATH)


_update_db_path_from_adapter()


def _conn():
    return get_adapter().connect()


def init_db(path: str | None = None):
    if path:
        configure_adapter(url=f"sqlite:///{os.path.abspath(path)}")
        _update_db_path_from_adapter()
    adapter = get_adapter()
    schema_sql = schema_for_backend(adapter.driver)
    with _conn() as cx:
        cx.executescript(schema_sql)
        cx.commit()

def save_df(name: str, df: pd.DataFrame):
    csv = df.to_csv(index=False)
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    with _conn() as cx:
        cx.execute(
            "REPLACE INTO datasets(name,csv,updated_at) VALUES(?,?,?)",
            (name, csv, ts)
        )
        cx.commit()

def load_df(name: str) -> Optional[pd.DataFrame]:
    with _conn() as cx:
        row = cx.execute("SELECT csv FROM datasets WHERE name=?", (name,)).fetchone()
    if not row:
        return pd.DataFrame()
    csv_text = row["csv"]
    # Gracefully handle NULL/empty CSV content
    if csv_text is None or str(csv_text).strip() == "":
        return pd.DataFrame()
    import io
    try:
        return pd.read_csv(io.StringIO(csv_text))
    except Exception:
        # Fallback for malformed/empty data
        try:
            # Specific catch improves clarity, but keep broad to avoid import churn
            from pandas.errors import EmptyDataError  # type: ignore
        except Exception:
            EmptyDataError = Exception  # noqa: N806
        try:
            # Re-try to identify EmptyDataError explicitly
            return pd.read_csv(io.StringIO(csv_text))
        except EmptyDataError:
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

def save_kv(key: str, obj):
    ts  = dt.datetime.now(dt.timezone.utc).isoformat()
    js  = json.dumps(obj)
    with _conn() as cx:
        cx.execute("REPLACE INTO kv(key,value,updated_at) VALUES(?,?,?)", (key, js, ts))
        cx.commit()

def load_kv(key: str):
    with _conn() as cx:
        row = cx.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    return (json.loads(row["value"]) if row else None)

def delete_df(name: str) -> None:
    """Delete a dataset row by exact name."""
    with _conn() as cx:
        cx.execute("DELETE FROM datasets WHERE name=?", (name,))
        cx.commit()

# def delete_datasets_by_prefix(prefix: str) -> int:
#     """Delete all dataset rows whose name starts with the given prefix. Returns rows affected."""
#     with _conn() as cx:
#         cur = cx.execute("DELETE FROM datasets WHERE name LIKE ?", (f"{prefix}%",))
#         cx.commit()
#         return cur.rowcount or 0

def delete_datasets_by_prefix(prefix: str) -> int:
    # debug trace before/after
    with _conn() as cx:
        cnt = cx.execute("SELECT COUNT(1) FROM datasets WHERE name LIKE ?", (f"{prefix}%",)).fetchone()[0]
    print(f"[DELETE_PREFIX][pre] prefix={prefix} matches={cnt}")
    with _conn() as cx:
        cur = cx.execute("DELETE FROM datasets WHERE name LIKE ?", (f"{prefix}%",))
        cx.commit()
        rows = cur.rowcount
    with _conn() as cx:
        cnt2 = cx.execute("SELECT COUNT(1) FROM datasets WHERE name LIKE ?", (f"{prefix}%",)).fetchone()[0]
    print(f"[DELETE_PREFIX][post] prefix={prefix} deleted={rows} remaining={cnt2}")
    return rows
