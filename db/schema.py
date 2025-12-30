from __future__ import annotations

# Core schema is defined once so new adapters (Postgres, Redshift, etc.)
# can reuse the table list without grepping through cap_db.py.

SQLITE_BASE_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS datasets (
    name TEXT PRIMARY KEY,
    csv  TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS settings_scoped (
    scope_type TEXT NOT NULL,
    scope_key  TEXT NOT NULL,
    effective_week TEXT NOT NULL,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(scope_type, scope_key, effective_week)
);

CREATE TABLE IF NOT EXISTS clients (
    business_area  TEXT PRIMARY KEY,
    hierarchy_json TEXT
);

-- Forecast runs (audit trail for forecasting workspace)
CREATE TABLE IF NOT EXISTS forecast_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope_key TEXT NOT NULL,
    business_area TEXT,
    sub_business_area TEXT,
    channel TEXT,
    site TEXT,
    model_name TEXT,
    created_by TEXT,
    created_at TEXT NOT NULL,
    forecast_csv TEXT NOT NULL,
    metadata_json TEXT,
    pushed_to_planning INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_forecast_runs_scope ON forecast_runs(scope_key);
CREATE INDEX IF NOT EXISTS idx_forecast_runs_created_at ON forecast_runs(created_at);
"""


def schema_for_backend(backend: str) -> str:
    """Return backend-specific bootstrap SQL."""
    backend = (backend or "").lower()
    if backend in ("sqlite", ""):
        return SQLITE_BASE_SCHEMA
    raise NotImplementedError(
        f"No schema defined for backend '{backend}'. "
        "Create one in db/schema.py when wiring a new adapter."
    )
