# cap_store.py — convenience API over cap_db.py
from __future__ import annotations

import os
import re
import json
import hashlib
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timezone

import pandas as pd

from cap_db import (
    init_db as _init,
    save_df, load_df, save_kv, load_kv
)
from db.adapters import get_adapter


# ─────────────────────────────────────────────────────────────
# Init / Connection
# ─────────────────────────────────────────────────────────────

def _conn():
    return get_adapter().connect()


def ensure_indexes() -> None:
    """Create any missing DB indexes (safe to run repeatedly)."""
    with _conn() as cx:
        cx.execute("""
            CREATE INDEX IF NOT EXISTS idx_cap_plans_vertical_subba_current
            ON capacity_plans(vertical, sub_ba, is_current);
        """)
        cx.commit()


def migrate_capacity_plans_location_site() -> None:
    """Safe migration to add location/site/is_deleted to capacity_plans."""
    with _conn() as cx:
        cols = {r["name"] for r in cx.execute("PRAGMA table_info(capacity_plans)")}
        if "location" not in cols:
            cx.execute("ALTER TABLE capacity_plans ADD COLUMN location TEXT")
        if "site" not in cols:
            cx.execute("ALTER TABLE capacity_plans ADD COLUMN site TEXT")
        if "is_deleted" not in cols:
            cx.execute("ALTER TABLE capacity_plans ADD COLUMN is_deleted INTEGER DEFAULT 0")
        cx.commit()


def ensure_settings_table() -> None:
    """Create/migrate settings_scoped to support effective-week versioning.
    Schema:
        scope_type TEXT in ('location','hier')
        scope_key  TEXT (canonicalized per scope)
        effective_week TEXT (YYYY-MM-DD Monday)
        value      TEXT (JSON)
        updated_at TEXT (ISO)
      PK: (scope_type, scope_key, effective_week)
    """
    with _conn() as cx:
        cx.execute(
            """
            CREATE TABLE IF NOT EXISTS settings_scoped (
                scope_type TEXT NOT NULL,
                scope_key  TEXT NOT NULL,
                effective_week TEXT NOT NULL,
                value      TEXT NOT NULL,
                updated_at TEXT,
                PRIMARY KEY(scope_type, scope_key, effective_week)
            )
            """
        )
        # Migrate legacy table without effective_week if present
        cols = {r["name"] for r in cx.execute("PRAGMA table_info(settings_scoped)")}
        if "effective_week" not in cols:
            try:
                cx.execute("ALTER TABLE settings_scoped ADD COLUMN effective_week TEXT")
            except Exception:
                pass
            try:
                today = pd.Timestamp.today().normalize()
                monday = (today - pd.Timedelta(days=int(today.weekday()))).date().isoformat()
            except Exception:
                monday = pd.Timestamp.today().date().isoformat()
            cx.execute(
                "UPDATE settings_scoped SET effective_week = COALESCE(effective_week, ?)",
                (monday,)
            )
        pk_cols = [r["name"] for r in cx.execute("PRAGMA table_info(settings_scoped)") if r["pk"]]
        if set(pk_cols) != {"scope_type", "scope_key", "effective_week"}:
            cx.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_settings_scoped_unique
                ON settings_scoped(scope_type, scope_key, effective_week)
                """
            )
        cx.commit()

def init_db():
    _init()
    ensure_settings_table()
    ensure_headcount_table()
    # Deprecated: mapping tables are no longer used; leaving functions for backward compat
    # ensure_mapping_tables()
    migrate_capacity_plans_location_site()
    ensure_indexes()


# ─────────────────────────────────────────────────────────────
# Headcount (BRID ⇄ Manager) store
# ─────────────────────────────────────────────────────────────

# --- add these helpers near other utils in cap_store.py ---
def _canon(s: str | None) -> str:
    return (s or "").strip().lower()

def _canon_hier_key(scope_key: str) -> str:
    parts = (scope_key or "").split("|")
    ba   = _canon(parts[0] if len(parts) > 0 else "")
    sub  = _canon(parts[1] if len(parts) > 1 else "")
    lob  = _canon(parts[2] if len(parts) > 2 else "")
    return f"{ba}|{sub}|{lob}"


def _row_value(row, key: str, fallback_index: int | None = None, default=None):
    """Return row value regardless of DB row implementation."""
    try:
        if isinstance(row, dict):
            return row.get(key, default)
        if hasattr(row, "keys") and key in row.keys():  # sqlite3.Row, psycopg rows, etc.
            return row[key]
        return row[key]
    except Exception:
        pass
    if fallback_index is not None and isinstance(row, (list, tuple)):
        try:
            return row[fallback_index]
        except Exception:
            pass
    return default

def _normalize_settings_dict(d: dict | None) -> dict:
    """
    Make settings robust to different field labels:
    - Map many possible keys to canonical: occupancy, target_aht, budgeted_aht, target_sut, budgeted_sut
    - Coerce numbers; for occupancy accept 80, '80', '80%', 0.8, '0.8'
    """
    if not isinstance(d, dict):
        return {}
    low = {str(k).strip().lower(): v for k, v in d.items()}

    def pick(*names):
        for n in names:
            if n in low and low[n] not in (None, ""):
                return low[n]
        return None

    def num(x):
        try:
            s = str(x).replace(",", "").strip()
            if s.endswith("%"):
                return float(s[:-1].strip())
            return float(s)
        except Exception:
            return None

    def pct_to_fraction(x):
        """Return fraction in 0..1 if we can; else None."""
        v = num(x)
        if v is None:
            return None
        if v > 1.0:   # 80 -> 0.8
            return v / 100.0
        return v      # already fractional like 0.8

    out = dict(low)

    # --- Occupancy
    occ_raw = pick("occupancy", "occupancy_pct", "occupancy percent", "occupancy%", "occupancy (%)",
                   "occ", "target_occupancy", "target occupancy", "budgeted_occupancy", "budgeted occupancy")
    occ_frac = pct_to_fraction(occ_raw) if occ_raw is not None else None
    if occ_frac is not None:
        out["occupancy"] = occ_frac                 # fraction 0..1
        out["occupancy_pct"] = round(occ_frac*100)  # percent 0..100

    # --- AHT/SUT canonicals
    canon_map = {
        "target_aht":   ["target_aht", "target aht", "voice_target_aht", "aht_target"],
        "budgeted_aht": ["budgeted_aht", "budgeted aht", "voice_budgeted_aht", "aht_budgeted"],
        "target_sut":   ["target_sut", "target sut", "bo_target_sut", "target_sut_sec", "sut_target"],
        "budgeted_sut": ["budgeted_sut", "budgeted sut", "bo_budgeted_sut", "budgeted_sut_sec", "sut_budgeted"],
    }
    for canon, alts in canon_map.items():
        v = pick(*alts)
        v = num(v) if v is not None else None
        if v is not None:
            out[canon] = v

    return out


HC_CANON_COLS = [
    "level_0", "level_1", "level_2", "level_3", "level_4", "level_5", "level_6",
    "brid", "full_name", "position_description", "hc_operational_status",
    "employee_group_description", "corporate_grade_description",
    "line_manager_brid", "line_manager_full_name",
    "current_org_unit", "current_org_unit_description",
    "position_location_country", "position_location_city", "position_location_building_description",
    "ccid", "cc_name", "journey", "position_group"
]


def ensure_headcount_table():
    with _conn() as cx:
        cx.execute("""
        CREATE TABLE IF NOT EXISTS headcount (
            level_0 TEXT, level_1 TEXT, level_2 TEXT, level_3 TEXT, level_4 TEXT, level_5 TEXT, level_6 TEXT,
            brid TEXT PRIMARY KEY,
            full_name TEXT,
            position_description TEXT,
            hc_operational_status TEXT,
            employee_group_description TEXT,
            corporate_grade_description TEXT,
            line_manager_brid TEXT,
            line_manager_full_name TEXT,
            current_org_unit TEXT,
            current_org_unit_description TEXT,
            position_location_country TEXT,
            position_location_city TEXT,
            position_location_building_description TEXT,
            ccid TEXT,
            cc_name TEXT,
            journey TEXT,
            position_group TEXT,
            updated_at TEXT
        )
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_headcount_lmbrid ON headcount(line_manager_brid)")
        cx.execute("CREATE INDEX IF NOT EXISTS idx_headcount_org ON headcount(current_org_unit)")
        cx.commit()


def _normalize_headcount_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=HC_CANON_COLS)

    L = {str(c).strip().lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            c = L.get(n.lower())
            if c:
                return c
        return None

    rename = {}
    mapping = {
        "level_0": ["level 0"],
        "level_1": ["level 1"],
        "level_2": ["level 2"],
        "level_3": ["level 3"],
        "level_4": ["level 4"],
        "level_5": ["level 5"],
        "level_6": ["level 6"],
        "brid": ["brid", "employee id", "employee number"],
        "full_name": ["full name", "employee name", "name"],
        "position_description": ["position description", "position"],
        "hc_operational_status": ["headcount operational status description", "operational status"],
        "employee_group_description": ["employee group description", "employee group"],
        "corporate_grade_description": ["corporate grade description", "grade"],
        "line_manager_brid": ["line manager brid", "manager brid", "tl brid"],
        "line_manager_full_name": ["line manager full name", "manager name", "tl name", "team manager"],
        "current_org_unit": ["current organisation unit", "current organization unit", "org unit"],
        "current_org_unit_description": ["current organisation unit description", "current organization unit description", "org unit description"],
        "position_location_country": ["position location country", "country"],
        "position_location_city": ["position location city", "city"],
        "position_location_building_description": ["position location building description", "building"],
        "ccid": ["ccid"],
        "cc_name": ["cc name"],
        "journey": ["journey"],
        "position_group": ["position group", "group"],
    }
    for canon, names in mapping.items():
        src = pick(*names)
        if src:
            rename[src] = canon

    dff = df.rename(columns=rename)
    keep = [c for c in HC_CANON_COLS if c in dff.columns]
    dff = dff[keep].copy()

    for c in dff.columns:
        if dff[c].dtype == object:
            dff[c] = dff[c].astype(str).str.strip()

    if "brid" in dff.columns:
        dff["brid"] = dff["brid"].astype(str).str.strip()
        dff = dff.drop_duplicates(subset=["brid"], keep="last")

    return dff


# keep ensure_headcount_table() as-is (it creates headcount with PRIMARY KEY (brid))

def save_headcount_df(df: pd.DataFrame) -> int:
    """
    Append/Upsert headcount:
      - Does NOT truncate the table anymore.
      - Upserts by PRIMARY KEY (brid). New BRIDs are inserted; existing BRIDs are updated.
      - Dedup inside the upload by BRID (last row wins).
    """
    dff = _normalize_headcount_df(df)

    # 1) Dedup within this upload by BRID (last wins)
    if "brid" in dff.columns:
        dff["brid"] = dff["brid"].astype(str).str.strip()
        dff = dff.drop_duplicates(subset=["brid"], keep="last")

    ensure_headcount_table()
    ts = datetime.now(timezone.utc).isoformat()

    # Make sure all expected columns exist in the frame
    all_cols = HC_CANON_COLS + ["updated_at"]
    for c in HC_CANON_COLS:
        if c not in dff.columns:
            dff[c] = None
    dff["updated_at"] = ts

    # 2) Upsert row-by-row (keeps prior uploads; only updates duplicates by BRID)
    col_list = HC_CANON_COLS + ["updated_at"]
    placeholders = ",".join(["?"] * len(col_list))
    assign_sql = ",".join([f"{c}=excluded.{c}" for c in col_list if c != "brid"])

    with _conn() as cx:
        # helpful indexes (idempotent)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_headcount_lmbrid ON headcount(line_manager_brid)")
        cx.execute("CREATE INDEX IF NOT EXISTS idx_headcount_org ON headcount(current_org_unit)")

        stmt = f"""
        INSERT INTO headcount({",".join(col_list)})
        VALUES ({placeholders})
        ON CONFLICT(brid) DO UPDATE SET {assign_sql}
        """
        vals = []
        for _, r in dff.iterrows():
            row_vals = [r.get(c, None) for c in HC_CANON_COLS] + [ts]
            vals.append(row_vals)

        cx.executemany(stmt, vals)
        cx.commit()

    # Optional snapshot (last upload only)
    save_df("headcount_raw", dff[all_cols])
    return int(len(dff))



def load_headcount(limit: int | None = None) -> pd.DataFrame:
    with _conn() as cx:
        q = "SELECT * FROM headcount"
        return pd.read_sql_query(q + (" LIMIT ?" if limit else ""), cx, params=[int(limit)] if limit else None)


def brid_manager_map() -> pd.DataFrame:
    with _conn() as cx:
        try:
            return pd.read_sql_query(
                "SELECT brid, line_manager_brid, line_manager_full_name FROM headcount", cx
            )
        except Exception:
            return pd.DataFrame(columns=["brid", "line_manager_brid", "line_manager_full_name"])

# ─── Journey lookups from Headcount (Level 2 → Journey; dependent Journey → Sites) ───

def level2_to_journey_map(pretty: bool = False) -> pd.DataFrame:
    """
    Returns a mapping between Level 2 and Journey derived from headcount.
    - If pretty=True, columns are ['Level 2','Journey'] (for UI tables).
    - Else columns are ['level_2','journey'].
    Dedupes on Level 2 (keeps first non-null Journey seen).
    """
    with _conn() as cx:
        try:
            df = pd.read_sql_query(
                """
                SELECT DISTINCT
                    COALESCE(level_2,'')       AS level_2,
                    COALESCE(journey,'')       AS journey
                FROM headcount
                WHERE COALESCE(level_2,'') <> ''
                """,
                cx,
            )
        except Exception:
            df = pd.DataFrame(columns=["level_2", "journey"])

    if not df.empty:
        df["level_2"] = df["level_2"].astype(str).str.strip()
        df["journey"] = df["journey"].astype(str).str.strip()
        df = df[df["level_2"] != ""]
        # one row per Level 2
        df = df.sort_values(["level_2", "journey"]).drop_duplicates(subset=["level_2"], keep="first")

    if pretty:
        return df.rename(columns={"level_2": "Level 2", "journey": "Journey"})
    return df


def unique_journeys() -> list[str]:
    """Distinct Journey values from headcount, sorted (empty removed)."""
    with _conn() as cx:
        try:
            df = pd.read_sql_query(
                "SELECT DISTINCT COALESCE(journey,'') AS journey FROM headcount", cx
            )
        except Exception:
            return []
    if df.empty:
        return []
    s = df["journey"].astype(str).str.strip()
    return sorted([x for x in s.unique().tolist() if x])


def journeys_sites_from_headcount() -> dict[str, list[str]]:
    """
    Returns { Journey: [Sites...] } where Sites come from
    'position_location_building_description' in headcount.
    """
    with _conn() as cx:
        try:
            df = pd.read_sql_query(
                """
                SELECT
                    COALESCE(journey,'') AS journey,
                    COALESCE(position_location_building_description,'') AS site
                FROM headcount
                """,
                cx,
            )
        except Exception:
            return {}

    if df.empty:
        return {}

    df["journey"] = df["journey"].astype(str).str.strip()
    df["site"] = df["site"].astype(str).str.strip()
    df = df[(df["journey"] != "") & (df["site"] != "")]
    out: dict[str, list[str]] = {}
    for j, grp in df.groupby("journey"):
        sites = sorted(grp["site"].dropna().astype(str).str.strip().unique().tolist())
        out[j] = sites
    return out


def sites_for_journey(journey: str) -> list[str]:
    """Convenience: list of sites for a given Journey (case-insensitive match)."""
    if not journey:
        return []
    jnorm = str(journey).strip().lower()
    with _conn() as cx:
        try:
            df = pd.read_sql_query(
                """
                SELECT
                    COALESCE(journey,'') AS journey,
                    COALESCE(position_location_building_description,'') AS site
                FROM headcount
                WHERE COALESCE(journey,'') <> ''
                """,
                cx,
            )
        except Exception:
            return []
    if df.empty:
        return []
    df["journey"] = df["journey"].astype(str).str.strip()
    df["site"] = df["site"].astype(str).str.strip()
    df = df[(df["journey"].str.lower() == jnorm) & (df["site"] != "")]
    return sorted(df["site"].unique().tolist())


# Optional helpers to tweak BA sites in clients.hierarchy_json + plans
def _load_clients_hier(ba: str):
    with _conn() as cx:
        row = cx.execute("SELECT hierarchy_json FROM clients WHERE business_area=?", (ba,)).fetchone()
        if not row:
            return None, None
        try:
            h = json.loads(row["hierarchy_json"] or "{}")
        except Exception:
            h = {}
        return h, cx


def rename_site_for_ba(ba: str, old_site: str, new_site: str) -> tuple[bool, str]:
    h, cx = _load_clients_hier(ba)
    if h is None:
        return False, f"Business Area '{ba}' not found in clients."

    sites = [s.strip() for s in (h.get("sites") or []) if str(s).strip()]
    if old_site not in sites and new_site not in sites:
        sites.append(new_site)
    else:
        sites = [new_site if s == old_site else s for s in sites]
        if new_site not in sites:
            sites.append(new_site)
        sites = sorted(set(sites))

    if sites:
        h["sites"] = sites
    else:
        h.pop("sites", None)

    cx.execute("UPDATE clients SET hierarchy_json=? WHERE business_area=?", (json.dumps(h), ba))
    cx.execute("""
        UPDATE capacity_plans SET site=? WHERE vertical=? AND COALESCE(site,'')=?
    """, (new_site, ba, old_site))
    cx.commit()
    return True, f"Renamed site for BA '{ba}': '{old_site}' → '{new_site}'."


def remove_site_for_ba(ba: str, site: str) -> tuple[bool, str]:
    h, cx = _load_clients_hier(ba)
    if h is None:
        return False, f"Business Area '{ba}' not found in clients."

    sites = [s for s in (h.get("sites") or []) if str(s).strip() and s != site]
    if sites:
        h["sites"] = sorted(set(sites))
    else:
        h.pop("sites", None)

    cx.execute("UPDATE clients SET hierarchy_json=? WHERE business_area=?", (json.dumps(h), ba))
    cx.commit()
    return True, f"Removed site '{site}' from BA '{ba}'."


# ─────────────────────────────────────────────────────────────
# Timeseries store by scope (BA|SBA|LOB)
# ─────────────────────────────────────────────────────────────

def _ensure_df(x) -> pd.DataFrame:
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame()

def _canon_scope_key(sk: str) -> str:
    """
    Normalize scope keys ('BA|SubBA|Channel') for storage and lookup.
    We lower-case and strip whitespace to avoid case/space mismatches.
    """
    return str(sk or "").strip().lower()
    
  
def _save_timeseries_impl(kind: str, scope_key: str, df: pd.DataFrame):
    """
    kind ∈ {
      'voice_forecast_volume','voice_actual_volume','voice_forecast_aht','voice_actual_aht',
      'bo_forecast_volume','bo_actual_volume','bo_forecast_sut','bo_actual_sut',
      'voice_tactical_volume','voice_tactical_aht','bo_tactical_volume','bo_tactical_sut'
    }
    """
    sk = _canon_scope_key(scope_key)
    # Enforce 4-part BA|SBA|Channel|Site keys for planner series
    prefixes = ("voice_", "bo_", "chat_", "ob_")
    if any(kind.startswith(p) for p in prefixes):
        parts = sk.split("|")
        # Allow exactly 4 parts; tolerate an empty trailing site as legacy only if explicitly provided
        if len(parts) < 4 or parts[0] == "" or parts[1] == "" or parts[2] == "":
            raise ValueError(
                f"Scope key must be 4-part 'BA|SBA|Channel|Site' for {kind}; got '{scope_key}'."
            )
    key = f"{kind}::{sk}"
    incoming = _ensure_df(df)
    if incoming.empty:
        # No new rows — keep existing snapshot as-is
        return
    existing = _ensure_df(load_df(key))
    if existing.empty:
        save_df(key, incoming)
        return

    combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    # De-duplicate on the natural time keys (date/week/day + interval when present)
    dedupe_keys: list[str] = []
    for candidate in ("date", "week", "day"):
        if candidate in combined.columns:
            dedupe_keys.append(candidate)
            break  # prefer first matching temporal column
    if "interval" in combined.columns:
        dedupe_keys.append("interval")
    if not dedupe_keys:
        # Fallback to all columns except obviously non-key ones
        dedupe_keys = [c for c in combined.columns if c not in ("program", "scope_key")] or combined.columns.tolist()

    combined = combined.drop_duplicates(subset=dedupe_keys, keep="last").reset_index(drop=True)
    save_df(key, combined)


def load_timeseries(kind: str, scope_key: str) -> pd.DataFrame:
    """
    Load by canonical scope key; if not found, attempt case-insensitive recovery
    and migrate to canonical name.
    """
    sk = _canon_scope_key(scope_key)
    df = _ensure_df(load_df(f"{kind}::{sk}"))
    if not isinstance(df, pd.DataFrame) or df.empty:
        # Try to find any saved dataset whose normalized scope matches
        with _conn() as cx:
            rows = cx.execute("SELECT name FROM datasets WHERE name LIKE ?", (f"{kind}::%",)).fetchall()
        for r in rows:
            name = (r["name"] if isinstance(r, dict) else r[0]) if r else ""
            if not name or "::" not in name:
                continue
            _, raw_sk = name.split("::", 1)
            if _canon_scope_key(raw_sk) == sk:
                # migrate to canonical key
                tmp = _ensure_df(load_df(name))
                if not tmp.empty:
                    save_df(f"{kind}::{sk}", tmp)
                    return tmp
    return df


def load_timeseries_any(kind: str, scopes: list[str]) -> pd.DataFrame:
    """
    Load time series for any of the provided scope keys.
    Supports both 3-part (BA|SBA|LOB) and 4-part (BA|SBA|LOB|SITE) keys by:
      1) Trying exact 3-part/4-part canonical fetch via load_timeseries
      2) If empty and a 3-part key is provided, merging all datasets whose stored key
         starts with that 3-part prefix (i.e., all sites under that scope)
    The returned frames will preserve the stored raw scope key in 'scope_key' when using
    prefix-based fallback, so downstream site mapping remains accurate.
    """
    frames: list[pd.DataFrame] = []
    for sk in scopes or []:
        canon = _canon_scope_key(sk)
        # Try exact
        d = load_timeseries(kind, canon)
        if isinstance(d, pd.DataFrame) and not d.empty:
            tmp = d.copy()
            tmp["scope_key"] = canon
            frames.append(tmp)
            continue

        # Try prefix match for 3-part keys (aggregate all matching 4-part stored keys)
        parts = canon.split("|")
        prefix3 = "|".join(parts[:3])
        # If caller already passed 4-part, also try its first 3-part prefix
        with _conn() as cx:
            # Fast path: direct LIKE lookup for names starting with the prefix + '|'
            like_pat = f"{kind}::{prefix3}|%"
            rows = cx.execute("SELECT name FROM datasets WHERE name LIKE ?", (like_pat,)).fetchall()
        for r in rows or []:
            name = (r["name"] if isinstance(r, dict) else r[0]) if r else ""
            if not name or "::" not in name:
                continue
            _, raw_sk = name.split("::", 1)
            df = _ensure_df(load_df(name))
            if isinstance(df, pd.DataFrame) and not df.empty:
                tmp = df.copy()
                # Preserve the stored key (includes site) for downstream mapping
                tmp["scope_key"] = _canon_scope_key(raw_sk)
                frames.append(tmp)

        # As a final lenient fallback, also check for an exact 3-part key name
        if not rows:
            df3 = _ensure_df(load_df(f"{kind}::{prefix3}"))
            if isinstance(df3, pd.DataFrame) and not df3.empty:
                tmp = df3.copy()
                tmp["scope_key"] = prefix3
                frames.append(tmp)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ─────────────────────────────────────────────────────────────


def _canon_holiday_scope(scope_type, scope_key):
    st = str(scope_type or "").strip().lower() or "global"
    if st not in {"global", "location", "hier"}:
        st = "global"
    if st == "global":
        sk = "global"
    elif st == "location":
        sk = str(scope_key or "").strip().lower()
    else:
        parts = [str(p).strip() for p in str(scope_key or "").split("|")]
        while len(parts) < 4:
            parts.append("")
        sk = "|".join(parts[:4]).lower()
    return st, sk


def save_holidays(scope_type, scope_key, df):
    st, sk = _canon_holiday_scope(scope_type, scope_key)
    key = f"holidays::{st}::{sk}"
    if not isinstance(df, pd.DataFrame) or df.empty:
        save_df(key, pd.DataFrame(columns=["date","name"]))
        return
    out = df.copy()
    if "date" not in out.columns:
        raise ValueError("Holiday data must include a 'date' column")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out["date"] = out["date"].dt.date.astype(str)
    name_col = None
    for candidate in ("name","holiday","description","label","event"):
        if candidate in out.columns:
            name_col = candidate
            break
    if name_col:
        out["name"] = out[name_col].astype(str).str.strip()
    else:
        out["name"] = ""
    out = out.drop_duplicates(subset=["date"], keep="last")[ ["date","name"] ]
    save_df(key, out)


def load_holidays(scope_type, scope_key):
    st, sk = _canon_holiday_scope(scope_type, scope_key)
    key = f"holidays::{st}::{sk}"
    df = load_df(key)
    if isinstance(df, pd.DataFrame):
        return df
    return pd.DataFrame(columns=["date","name"])


def resolve_holidays(ba=None, subba=None, lob=None, site=None, location=None):
    def _canon(val):
        return str(val or "").strip()
    ba_c = _canon(ba)
    sba_c = _canon(subba)
    lob_c = _canon(lob)
    site_c = _canon(site)
    if any([ba_c, sba_c, lob_c, site_c]):
        scope_full = "|".join([ba_c, sba_c, lob_c, site_c])
        df = load_holidays("hier", scope_full)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
        if site_c:
            scope_no_site = "|".join([ba_c, sba_c, lob_c])
            df = load_holidays("hier", scope_no_site)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
    loc_c = _canon(location)
    if loc_c:
        df = load_holidays("location", loc_c)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    df = load_holidays("global", "global")
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame(columns=["date","name"])

# Mapping stores (append-only + dedupe)
# ─────────────────────────────────────────────────────────────

def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


def _colpick(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        c = low.get(n.lower())
        if c:
            return c
    return None


def _sha256_of_df(df: pd.DataFrame) -> str:
    csv_bytes = df.to_csv(index=False).encode("utf-8", errors="ignore")
    return hashlib.sha256(csv_bytes).hexdigest()


def ensure_mapping_tables():
    # DEPRECATED: Mapping Sheet 1/2 tables are deprecated. Kept for backward compatibility only.
    with _conn() as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS mapping_files (
              id INTEGER PRIMARY KEY,
              kind TEXT CHECK (kind IN ('map1','map2')) NOT NULL,
              filename TEXT,
              sha256 TEXT UNIQUE,
              uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cx.execute("""
            CREATE TABLE IF NOT EXISTS map1_rows (
              id INTEGER PRIMARY KEY,
              file_id INTEGER NOT NULL REFERENCES mapping_files(id) ON DELETE CASCADE,
              business_area TEXT,
              sub_business_area TEXT,
              channel TEXT,
              location TEXT,
              site TEXT,
              ba_norm TEXT,
              sba_norm TEXT,
              ch_norm TEXT,
              loc_norm TEXT,
              site_norm TEXT,
              UNIQUE (ba_norm, sba_norm, ch_norm, loc_norm, site_norm) ON CONFLICT IGNORE
            );
        """)
        cx.execute("""
            CREATE TABLE IF NOT EXISTS map2_rows (
              id INTEGER PRIMARY KEY,
              file_id INTEGER NOT NULL REFERENCES mapping_files(id) ON DELETE CASCADE,
              business_area TEXT,
              ba_norm TEXT UNIQUE ON CONFLICT IGNORE
            );
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_map1_ba_sba ON map1_rows(ba_norm, sba_norm);")
        cx.execute("CREATE INDEX IF NOT EXISTS idx_map1_site ON map1_rows(site_norm);")
        cx.execute("CREATE INDEX IF NOT EXISTS idx_map2_ba ON map2_rows(ba_norm);")
        cx.commit()


# Ensure mapping tables exist at import
ensure_mapping_tables()
# ─────────────────────────────────────────────────────────────
# Defaults & Scoped Settings
# ─────────────────────────────────────────────────────────────

def load_defaults() -> dict | None:
    return load_kv("defaults")


def save_defaults(cfg: dict):
    save_kv("defaults", _normalize_settings_dict(cfg or {}))


def save_scoped_settings(scope_type: str, scope_key: str, d: dict, effective_week: str | None = None):
    """Versioned save: inserts a new row effective from the given Monday (or current Monday).
    - scope_type: 'hier' or 'location'
    - scope_key: canonicalized key (BA|SBA|LOB for 'hier'; location name for 'location')
    - d: settings payload (normalized)
    - effective_week: Monday date (YYYY-MM-DD). Defaults to current Monday.
    """
    assert scope_type in ("location", "hier")
    if scope_type == "hier":
        scope_key = _canon_hier_key(scope_key)
    else:
        scope_key = _canon(scope_key)

    eff = _monday(effective_week or pd.Timestamp.today())
    blob = json.dumps(_normalize_settings_dict(d or {}))
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as cx:
        try:
            cx.execute(
                """
                INSERT INTO settings_scoped(scope_type, scope_key, effective_week, value, updated_at)
                VALUES (?,?,?,?,?)
                ON CONFLICT(scope_type, scope_key, effective_week)
                DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (scope_type, scope_key, eff, blob, now)
            )
            cx.commit()
        except Exception:
            # Fallback for older DBs where a UNIQUE(scope_type,scope_key) exists (no versioning)
            cur = cx.execute(
                "UPDATE settings_scoped SET value=?, updated_at=?, effective_week=? WHERE scope_type=? AND scope_key=?",
                (blob, now, eff, scope_type, scope_key)
            )
            if (cur.rowcount or 0) == 0:
                cx.execute(
                    "INSERT INTO settings_scoped(scope_type, scope_key, effective_week, value, updated_at) VALUES (?,?,?,?,?)",
                    (scope_type, scope_key, eff, blob, now)
                )
            cx.commit()

def load_scoped_settings(scope_type: str, scope_key: str, for_date: str | None = None) -> dict:
    """Return the settings row whose effective_week is the latest <= for_date (or today)."""
    target_week = _monday(for_date or pd.Timestamp.today())
    with _conn() as cx:
        canon = _canon_hier_key(scope_key) if scope_type == "hier" else _canon(scope_key)
        row = cx.execute(
            """
            SELECT value
              FROM settings_scoped
             WHERE scope_type=? AND scope_key=? AND effective_week <= ?
             ORDER BY effective_week DESC
             LIMIT 1
            """,
            (scope_type, canon, target_week)
        ).fetchone()
        if not row:
            # lenient fallback by lowercase match
            row = cx.execute(
                """
                SELECT value
                  FROM settings_scoped
                 WHERE scope_type=? AND LOWER(scope_key)=LOWER(?) AND effective_week <= ?
                 ORDER BY effective_week DESC
                 LIMIT 1
                """,
                (scope_type, scope_key or "", target_week)
            ).fetchone()
    return _normalize_settings_dict(json.loads(row["value"])) if row else {}

def resolve_settings(location: str | None = None,
                     ba: str | None = None,
                     subba: str | None = None,
                     lob: str | None = None,
                     for_date: str | None = None) -> dict:
    """Resolve settings with effective-week logic.
    - If BA/SubBA/LOB present, pick most specific row with effective_week <= for_date (or today).
    - Else try location. Else fallback to global defaults.
    """
    if ba and subba and lob:
        s = load_scoped_settings("hier", f"{ba}|{subba}|{lob}", for_date=for_date)
        if s:
            return s
    if location:
        s = load_scoped_settings("location", str(location), for_date=for_date)
        if s:
            return s
    d = load_kv("defaults") or {}
    return _normalize_settings_dict(d)



# ─────────────────────────────────────────────────────────────
# Roster / Hiring / Shrinkage / Attrition datasets
# ─────────────────────────────────────────────────────────────

def load_roster() -> pd.DataFrame:
    return _ensure_df(load_df("roster"))


def save_roster(df: pd.DataFrame):
    """
    Saves roster with safe de-duplication:
    - If a 'date' column exists (long format), drop duplicates by (BRID, date).
    - If wide format (YYYY-MM-DD columns), melt to 'roster_long' by (BRID, date).
    - Else, de-dupe by BRID only.
    """
    if df is None or df.empty:
        save_df("roster", pd.DataFrame())
        return

    L = {c.lower(): c for c in df.columns}
    brid_col = L.get("brid") or L.get("employee_id") or "BRID"

    # Long
    if "date" in L:
        date_col = L["date"]
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date.astype(str)
        df = df.dropna(subset=[brid_col, date_col])
        df = df.drop_duplicates(subset=[brid_col, date_col], keep="last")
        save_df("roster", df)
        return

    # Wide → Long
    date_like_cols = [c for c in df.columns if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(c))]
    if date_like_cols:
        static_cols = [c for c in df.columns if c not in date_like_cols]
        long = df.melt(
            id_vars=static_cols,
            value_vars=date_like_cols,
            var_name="date",
            value_name="shift"
        )
        long = long[long["shift"].notna() & (long["shift"].astype(str).str.strip() != "")]
        long["date"] = pd.to_datetime(long["date"], errors="coerce").dt.date.astype(str)
        long = long.dropna(subset=[brid_col, "date"]).drop_duplicates(subset=[brid_col, "date"], keep="last")

        save_df("roster", df)         # keep the wide view here for back-compat
        save_df("roster_long", long)  # normalized
        return

    # Legacy
    save_df("roster", df.drop_duplicates(subset=[brid_col], keep="last"))


def load_roster_wide() -> pd.DataFrame:
    """
    Prefer the dedicated 'roster_wide' key (new), fallback to legacy 'roster'.
    """
    df = load_df("roster_wide")
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    df2 = load_df("roster")
    return df2 if isinstance(df2, pd.DataFrame) else pd.DataFrame()


def save_roster_wide(df: pd.DataFrame):
    save_df("roster_wide", df if isinstance(df, pd.DataFrame) else pd.DataFrame())


def load_roster_long() -> pd.DataFrame:
    df = load_df("roster_long")
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def save_roster_long(df: pd.DataFrame):
    save_df("roster_long", df if isinstance(df, pd.DataFrame) else pd.DataFrame())


def load_hiring() -> pd.DataFrame:
    return _ensure_df(load_df("hiring"))


def save_hiring(df: pd.DataFrame):
    save_df("hiring", df if isinstance(df, pd.DataFrame) else pd.DataFrame())


def load_shrinkage() -> pd.DataFrame:
    return _ensure_df(load_df("shrinkage"))


def save_shrinkage(df: pd.DataFrame):
    save_df("shrinkage", df if isinstance(df, pd.DataFrame) else pd.DataFrame())


def load_attrition() -> pd.DataFrame:
    return _ensure_df(load_df("attrition"))


def save_attrition(df: pd.DataFrame):
    save_df("attrition", df if isinstance(df, pd.DataFrame) else pd.DataFrame())


def load_attrition_raw() -> pd.DataFrame | None:
    return load_df("attrition_raw")


def save_attrition_raw(df: pd.DataFrame):
    save_df("attrition_raw", df if isinstance(df, pd.DataFrame) else pd.DataFrame())


# ─────────────────────────────────────────────────────────────
# Helpers for UI sources (locations / hierarchy)
# ─────────────────────────────────────────────────────────────

def get_roster_locations() -> list[str]:
    df = load_roster()
    if df.empty:
        return []
    vals: set[str] = set()
    for c in ["location", "country", "site", "region"]:
        if c in df.columns:
            vals |= set(
                df[c].dropna().astype(str).str.strip().replace({"": None}).dropna().tolist()
            )
    return sorted(vals)


def _hier_from_rows(rows) -> dict:
    out: dict[str, dict] = {}
    for r in rows:
        ba = _row_value(r, "business_area", fallback_index=0, default="Unknown") or "Unknown"
        hj = _row_value(r, "hierarchy_json", fallback_index=1, default="{}") or "{}"
        try:
            h = json.loads(hj)
        except Exception:
            h = {}
        subs = h.get("sub_business_areas") or ["Default"]
        lobs = h.get("channels") or ["Voice", "Back Office", "Outbound"]
        out.setdefault(ba, {})
        for s in subs:
            out[ba][s] = list(lobs)
    return out


def get_clients_hierarchy() -> dict:
    merged: dict[str, dict] = {}
    with _conn() as cx:
        try:
            merged = _hier_from_rows(cx.execute("SELECT business_area, hierarchy_json FROM clients").fetchall())
        except Exception:
            merged = {}
    return merged


# ─────────────────────────────────────────────────────────────
# Sample template makers (for downloads)
# ─────────────────────────────────────────────────────────────

def roster_template_df() -> pd.DataFrame:
    return pd.DataFrame([
        dict(employee_id="UK0001", name="Alex Doe", status="Active", employment_type="FT",
             fte=1.0, contract_hours_per_week=37.5, country="UK", site="Glasgow", timezone="Europe/London",
             program="WFM", sub_business_area="Retail", lob="Cards", channel="Back Office",
             skill_voice=False, skill_bo=True, skill_ob=False,
             start_date="2025-07-01", end_date=""),
        dict(employee_id="IN0002", name="Priya Singh", status="Active", employment_type="PT",
             fte=0.5, contract_hours_per_week=20, country="India", site="Chennai", timezone="Asia/Kolkata",
             program="WFM", sub_business_area="Retail", lob="Cards", channel="Voice",
             skill_voice=True, skill_bo=False, skill_ob=False,
             start_date="2025-07-08", end_date=""),
    ])


def hiring_template_df() -> pd.DataFrame:
    return pd.DataFrame([
        dict(start_week="2025-07-07", fte=3, program="WFM", country="UK", site="Glasgow"),
        dict(start_week="2025-07-14", fte=5, program="WFM", country="India", site="Chennai"),
    ])


def shrinkage_bo_template_df() -> pd.DataFrame:
    return pd.DataFrame([
        dict(week="2025-07-07", program="WFM", sub_business_area="Retail", lob="Cards", site="Glasgow",
             shrinkage_pct=11.0),
        dict(week="2025-07-14", program="WFM", sub_business_area="Retail", lob="Cards", site="Glasgow",
             shrinkage_pct=10.7),
    ])


def shrinkage_voice_template_df() -> pd.DataFrame:
    return pd.DataFrame([
        dict(week="2025-07-07", program="WFM", queue="Inbound", site="Chennai", shrinkage_pct=12.5),
        dict(week="2025-07-14", program="WFM", queue="Inbound", site="Chennai", shrinkage_pct=11.9),
    ])


def attrition_template_df() -> pd.DataFrame:
    return pd.DataFrame([
        dict(week="2025-07-07", program="WFM", site="Glasgow", attrition_pct=0.8),
        dict(week="2025-07-14", program="WFM", site="Chennai", attrition_pct=1.1),
    ])


# One-time helper: migrate datasets saved with mixed-case scope keys to canonical lower-case.
def migrate_timeseries_scope_keys_to_lower() -> int:
    moved = 0
    with _conn() as cx:
        rows = cx.execute("SELECT name FROM datasets WHERE name LIKE '%::%'").fetchall()
    for r in rows:
        name = _row_value(r, "name", fallback_index=0, default="")
        if "::" not in name:
            continue
        kind, raw_sk = name.split("::", 1)
        canon = _canon_scope_key(raw_sk)
        if canon != raw_sk:
            df = load_df(name)
            if isinstance(df, pd.DataFrame) and not df.empty:
                save_df(f"{kind}::{canon}", df)
                moved += 1
    return moved

def _monday(d) -> str:
    try:
        t = pd.to_datetime(d).normalize()
    except Exception:
        t = pd.Timestamp.today().normalize()
    m = (t - pd.Timedelta(days=int(t.weekday()))).date().isoformat()
    return m

# ---------------------------------------------------------------------------
# Override: timeseries saving should append by date/week and replace overlaps
# ---------------------------------------------------------------------------

def save_timeseries(kind: str, scope_key: str, df: pd.DataFrame):
    """
    Save a time series for a scope, merging by date/week:
      - Appends new dates/weeks
      - Replaces existing rows for overlapping dates/weeks

    Applies to all channels/series, e.g. Voice/BO/Chat/Outbound forecast/actual/tactical.
    """
    sk = _canon_scope_key(scope_key)
    # Enforce 4-part keys for planner series (BA|SBA|Channel|Site)
    prefixes = ("voice_", "bo_", "chat_", "ob_")
    if any(kind.startswith(p) for p in prefixes):
        parts = sk.split("|")
        if len(parts) < 4 or parts[0] == "" or parts[1] == "" or parts[2] == "":
            raise ValueError(
                f"Scope key must be 4-part 'BA|SBA|Channel|Site' for {kind}; got '{scope_key}'."
            )
    name = f"{kind}::{sk}"

    new = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    if not isinstance(new, pd.DataFrame):
        new = pd.DataFrame()

    def _norm_date_cols(_df: pd.DataFrame) -> tuple[pd.DataFrame, str | None, str | None]:
        if not isinstance(_df, pd.DataFrame) or _df.empty:
            return _df, None, None
        d = _df.copy()
        low = {str(c).strip().lower(): c for c in d.columns}
        c_date = low.get("date")
        c_week = low.get("week")
        if c_date and c_date in d.columns:
            d[c_date] = pd.to_datetime(d[c_date], errors="coerce").dt.date.astype(str)
        if (not c_date) and c_week and c_week in d.columns:
            d[c_week] = pd.to_datetime(d[c_week], errors="coerce").dt.date.astype(str)
        return d, c_date, c_week

    new, new_date, new_week = _norm_date_cols(new)

    # Load existing
    existing = load_df(name)
    if not isinstance(existing, pd.DataFrame) or existing.empty:
        save_df(name, new)
        return

    old, old_date, old_week = _norm_date_cols(existing)

    # Choose merge key: prefer date, else week. If incompatible, fallback to overwrite.
    if new_date and old_date:
        kn, ko = new_date, old_date
    elif (not new_date) and (not old_date) and new_week and old_week:
        kn, ko = new_week, old_week
    else:
        # Different schemas (no common date/week); overwrite as legacy behavior
        save_df(name, new)
        return

    # Align columns (union), keep order stable
    all_cols = list(dict.fromkeys(list(old.columns) + list(new.columns)))
    old = old.reindex(columns=all_cols)
    new = new.reindex(columns=all_cols)

    # Remove overlapping keys from old, then append new
    try:
        new_keys = set(pd.Series(new[kn]).dropna().astype(str).unique().tolist())
    except Exception:
        new_keys = set()
    base = old if not new_keys else old[~old[ko].astype(str).isin(new_keys)].copy()
    out = pd.concat([base, new], ignore_index=True)

    # Sort by key and by interval if available for readability
    low_out = {str(c).strip().lower(): c for c in out.columns}
    sort_cols = [ko]
    for cand in ("interval", "interval_start"):
        c = low_out.get(cand)
        if c:
            sort_cols.append(c)
            break
    try:
        out = out.sort_values(sort_cols)
    except Exception:
        pass

    save_df(name, out)


# ---------------------------------------------------------------------------
# Forecasting workspace persistence and audit trail
# ---------------------------------------------------------------------------

def record_forecast_run(
    scope_key: str,
    forecast_df: pd.DataFrame,
    created_by: str,
    model_name: Optional[str] = None,
    metadata: Optional[dict] = None,
    pushed_to_planning: bool = False,
    business_area: Optional[str] = None,
    sub_business_area: Optional[str] = None,
    channel: Optional[str] = None,
    site: Optional[str] = None,
) -> int:
    """
    Persist a forecast output with audit metadata and return the run id.
    Scope key should be BA|SBA|Channel|Site (case-insensitive).
    """
    if not isinstance(forecast_df, pd.DataFrame) or forecast_df.empty:
        raise ValueError("forecast_df must be a non-empty DataFrame")
    sk = _canon_scope_key(scope_key)
    parts = sk.split("|")
    ba = business_area or (parts[0] if len(parts) > 0 else "")
    sba = sub_business_area or (parts[1] if len(parts) > 1 else "")
    ch = channel or (parts[2] if len(parts) > 2 else "")
    st = site or (parts[3] if len(parts) > 3 else "")
    ts = datetime.now(timezone.utc).isoformat()
    meta_js = json.dumps(metadata) if metadata is not None else None
    csv_txt = forecast_df.to_csv(index=False)
    with _conn() as cx:
        cx.execute(
            """
            INSERT INTO forecast_runs(
                scope_key,business_area,sub_business_area,channel,site,
                model_name,created_by,created_at,forecast_csv,metadata_json,pushed_to_planning
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                sk,
                ba,
                sba,
                ch,
                st,
                model_name,
                created_by,
                ts,
                csv_txt,
                meta_js,
                1 if pushed_to_planning else 0,
            ),
        )
        run_id = cx.execute("SELECT last_insert_rowid()").fetchone()[0]
        cx.commit()
    return int(run_id)


def list_forecast_runs(scope_key: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
    """
    Retrieve recent forecast runs (optionally filtered by scope key).
    """
    params: list[Any] = []
    sql = """
    SELECT id, scope_key, business_area, sub_business_area, channel, site,
           model_name, created_by, created_at, pushed_to_planning
      FROM forecast_runs
    """
    if scope_key:
        sql += " WHERE scope_key = ?"
        params.append(_canon_scope_key(scope_key))
    sql += " ORDER BY created_at DESC"
    if limit and limit > 0:
        sql += " LIMIT ?"
        params.append(int(limit))
    with _conn() as cx:
        rows = cx.execute(sql, params).fetchall()
    return pd.DataFrame(rows, columns=[
        "id",
        "scope_key",
        "business_area",
        "sub_business_area",
        "channel",
        "site",
        "model_name",
        "created_by",
        "created_at",
        "pushed_to_planning",
    ])


def load_forecast_run(run_id: int) -> tuple[Optional[dict], pd.DataFrame]:
    """
    Load a stored forecast run (metadata dict, forecast DataFrame).
    """
    with _conn() as cx:
        row = cx.execute(
            """
            SELECT id, scope_key, business_area, sub_business_area, channel, site,
                   model_name, created_by, created_at, forecast_csv, metadata_json, pushed_to_planning
              FROM forecast_runs
             WHERE id = ?
            """,
            (int(run_id),),
        ).fetchone()
    if not row:
        return None, pd.DataFrame()
    meta = {
        "id": row["id"],
        "scope_key": row["scope_key"],
        "business_area": row["business_area"],
        "sub_business_area": row["sub_business_area"],
        "channel": row["channel"],
        "site": row["site"],
        "model_name": row["model_name"],
        "created_by": row["created_by"],
        "created_at": row["created_at"],
        "pushed_to_planning": bool(row["pushed_to_planning"]),
    }
    try:
        extra = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        if isinstance(extra, dict):
            meta["metadata"] = extra
    except Exception:
        meta["metadata"] = {}
    import io
    try:
        df = pd.read_csv(io.StringIO(row["forecast_csv"] or ""))
    except Exception:
        df = pd.DataFrame()
    return meta, df


def push_forecast_to_planning(
    scope_key: str,
    channel: str,
    forecast_df: pd.DataFrame,
    created_by: str,
    model_name: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> tuple[bool, str, Optional[int]]:
    """
    Persist forecast to audit table and push to planning timeseries for the channel.
    Expects forecast_df to contain 'date' and 'volume' columns; 'aht_sec' or 'sut_sec'
    will be saved when present for supported channels.
    """
    sk = _canon_scope_key(scope_key)
    if not isinstance(forecast_df, pd.DataFrame) or forecast_df.empty:
        return False, "No forecast rows to save.", None
    low_cols = {str(c).strip().lower(): c for c in forecast_df.columns}
    c_date = low_cols.get("date")
    c_vol = low_cols.get("volume")
    if not c_date or not c_vol:
        return False, "Forecast must include 'date' and 'volume' columns.", None
    df = forecast_df.copy()
    df[c_date] = pd.to_datetime(df[c_date], errors="coerce").dt.date.astype(str)
    df = df[[c_date, c_vol] + [c for c in df.columns if c not in (c_date, c_vol)]]

    ch_low = (channel or "").strip().lower()
    vol_kind = None
    aht_kind = None
    if ch_low in ("voice", "call", "telephony"):
        vol_kind = "voice_forecast_volume"
        aht_kind = "voice_forecast_aht"
    elif ch_low in ("back office", "bo", "backoffice"):
        vol_kind = "bo_forecast_volume"
        aht_kind = "bo_forecast_sut"
    elif ch_low in ("chat", "messaging", "messageus", "message us"):
        vol_kind = "chat_forecast_volume"
        aht_kind = "chat_forecast_aht"

    if not vol_kind:
        return False, f"Unsupported channel '{channel}' for planning push.", None

    try:
        save_timeseries(vol_kind, sk, df[[c_date, c_vol]].rename(columns={c_date: "date", c_vol: "volume"}))
        if aht_kind:
            target_aht_col = "aht_sec" if "aht" in aht_kind else "sut_sec"
            for aht_col in ("aht_sec", "sut_sec", "aht", "sut"):
                if aht_col in low_cols:
                    save_timeseries(
                        aht_kind,
                        sk,
                        df[[c_date, low_cols[aht_col]]].rename(
                            columns={c_date: "date", low_cols[aht_col]: target_aht_col}
                        ),
                    )
                    break
        run_id = record_forecast_run(
            sk,
            forecast_df,
            created_by=created_by,
            model_name=model_name,
            metadata=metadata,
            pushed_to_planning=True,
        )
        return True, "Forecast saved and pushed to planning.", run_id
    except Exception as exc:
        return False, f"Failed to push forecast: {exc}", None
