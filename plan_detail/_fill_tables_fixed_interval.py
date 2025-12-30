from __future__ import annotations
import math
import re
import datetime as dt
from functools import lru_cache
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from dash import dash_table
from capacity_core import min_agents

from plan_store import get_plan
from cap_store import resolve_settings, load_roster_long
from ._grain_cols import interval_cols_for_day
from ._common import _week_span
from ._calc import _fill_tables_fixed, get_cached_consolidated_calcs
from ._common import (
    _canon_scope,
    _assemble_voice,
    _assemble_chat,
    _assemble_ob,
    _load_ts_with_fallback,
)

# ----------------------------------------------------------------------
# CACHES / HELPERS
# ----------------------------------------------------------------------

# Cache roster to avoid repeated expensive loads
_ROSTER_CACHE: Optional[pd.DataFrame] = None

def _get_roster() -> pd.DataFrame:
    global _ROSTER_CACHE
    if _ROSTER_CACHE is None:
        try:
            df = load_roster_long()
        except Exception:
            df = pd.DataFrame()
        # Pre-parse entry into numeric minutes once per row if possible
        if isinstance(df, pd.DataFrame) and not df.empty and "entry" in df.columns:
            if "start_min" not in df.columns or "end_min" not in df.columns:
                start_mins = []
                end_mins = []
                for s in df["entry"].astype(str):
                    m = re.match(
                        r"^(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})$",
                        s.strip()
                    )
                    if not m:
                        start_mins.append(np.nan)
                        end_mins.append(np.nan)
                        continue
                    sh, sm, eh, em = map(int, m.groups())
                    sh = min(23, max(0, sh))
                    eh = min(24, max(0, eh))
                    start_min = sh * 60 + sm
                    end_min = eh * 60 + em
                    if end_min <= start_min:
                        end_min += 24 * 60
                    start_mins.append(start_min)
                    end_mins.append(end_min)
                df = df.copy()
                df["start_min"] = start_mins
                df["end_min"] = end_mins
        _ROSTER_CACHE = df
    return _ROSTER_CACHE


def _pick_ivl_col(df: pd.DataFrame) -> Optional[str]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    low = {str(c).strip().lower(): c for c in df.columns}
    for k in ("interval", "time", "interval_start", "start_time", "slot"):
        c = low.get(k)
        if c and c in df.columns:
            return c
    return None


def _parse_time_any(s: str) -> Optional[dt.time]:
    try:
        if s is None:
            return None
        t = str(s).strip()
        if not t:
            return None
        # Try common 12h and 24h formats
        for fmt in ("%I:%M:%S %p", "%I:%M %p", "%H:%M:%S", "%H:%M"):
            try:
                return dt.datetime.strptime(t, fmt).time()
            except Exception:
                pass
        # Fallback: pandas parser
        ts = pd.to_datetime(t, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.time()
    except Exception:
        return None


def _fmt_hhmm(t: dt.time) -> str:
    try:
        return f"{t.hour:02d}:{t.minute:02d}"
    except Exception:
        return "00:00"


# Cache of "prepared day frame" per df+day to avoid repeated time parsing
# key: (id(df), day)  value: DataFrame with at least columns: ['lab', <original cols>]
_SLOT_FRAME_CACHE: Dict[Tuple[int, dt.date], pd.DataFrame] = {}


def _prepare_day_slot_frame(df: pd.DataFrame, day: dt.date) -> pd.DataFrame:
    """Return a frame for this df+day with parsed 'lab' (HH:MM) column.
    Cached so multiple metric aggregations reuse the time parsing.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    key = (id(df), day)
    cached = _SLOT_FRAME_CACHE.get(key)
    if cached is not None:
        return cached

    d = df.copy()
    L = {str(c).strip().lower(): c for c in d.columns}
    c_date = L.get("date") or L.get("day")
    c_ivl = _pick_ivl_col(d)
    if not c_ivl:
        res = pd.DataFrame()
        _SLOT_FRAME_CACHE[key] = res
        return res

    if c_date:
        d[c_date] = pd.to_datetime(d[c_date], errors="coerce").dt.date
        d = d[d[c_date].eq(day)]
    if d.empty:
        res = pd.DataFrame()
        _SLOT_FRAME_CACHE[key] = res
        return res

    times = d[c_ivl].astype(str).map(_parse_time_any).dropna()
    if times.empty:
        res = pd.DataFrame()
        _SLOT_FRAME_CACHE[key] = res
        return res

    d = d.loc[times.index].copy()
    d["lab"] = times.map(_fmt_hhmm)
    _SLOT_FRAME_CACHE[key] = d
    return d


def _slot_series_for_day(df: pd.DataFrame, day: dt.date, val_col: str) -> Dict[str, float]:
    """Aggregate df for a single day into slot (HH:MM) -> sum(val_col).
    Uses cached prepared frame to avoid repeated time parsing per metric.
    """
    base = _prepare_day_slot_frame(df, day)
    if not isinstance(base, pd.DataFrame) or base.empty or val_col not in base.columns:
        return {}
    vals = pd.to_numeric(base[val_col], errors="coerce").fillna(0.0)
    tmp = pd.DataFrame({"lab": base["lab"], "val": vals}).dropna(subset=["lab"]).copy()
    if tmp.empty:
        return {}
    g = tmp.groupby("lab", as_index=True)["val"].sum()
    return {str(k): float(v) for k, v in g.to_dict().items()}


def _infer_window(plan: dict, day: dt.date, ch: str, sk: str) -> Tuple[str, Optional[str]]:
    def _window_from(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        try:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return None, None
            d = df.copy()
            L = {str(c).strip().lower(): c for c in d.columns}
            c_date = L.get("date") or L.get("day")
            ivc = _pick_ivl_col(d)
            if not ivc:
                return None, None
            if c_date:
                d[c_date] = pd.to_datetime(d[c_date], errors="coerce").dt.date
                d = d[d[c_date].eq(day)]
            if d.empty:
                return None, None
            times = d[ivc].astype(str).map(_parse_time_any).dropna()
            if times.empty:
                return None, None
            tmin = times.min()
            tmax = times.max()
            return _fmt_hhmm(tmin), _fmt_hhmm(tmax)
        except Exception:
            return None, None

    start = None
    end = None
    try:
        if ch == "voice":
            for df in (_assemble_voice(sk, "forecast"), _assemble_voice(sk, "actual")):
                s, e = _window_from(df)
                start = start or s
                end = end or e
        elif ch == "chat":
            for key in ("chat_forecast_volume", "chat_actual_volume"):
                df = _load_ts_with_fallback(key, sk)
                s, e = _window_from(df)
                start = start or s
                end = end or e
        elif ch in ("outbound", "ob"):
            for key in (
                "ob_forecast_opc", "outbound_forecast_opc", "ob_actual_opc", "outbound_actual_opc",
                "ob_forecast_dials", "outbound_forecast_dials", "ob_actual_dials", "outbound_actual_dials",
                "ob_forecast_calls", "outbound_forecast_calls", "ob_actual_calls", "outbound_actual_calls",
            ):
                df = _load_ts_with_fallback(key, sk)
                s, e = _window_from(df)
                start = start or s
                end = end or e
    except Exception:
        start, end = None, None

    # Fallback: roster window
    if not start or not end:
        try:
            rl = _get_roster()
        except Exception:
            rl = pd.DataFrame()
        if isinstance(rl, pd.DataFrame) and not rl.empty:
            df = rl.copy()

            def _col(opts):
                for c in opts:
                    if c in df.columns:
                        return c
                return None

            c_ba = _col(["Business Area", "business area", "vertical"])
            c_sba = _col(["Sub Business Area", "sub business area", "sub_ba"])
            c_lob = _col(["LOB", "lob", "Channel", "channel"])
            c_site = _col(["Site", "site", "Location", "location", "Country", "country"])
            BA = plan.get("vertical")
            SBA = plan.get("sub_ba")
            LOB = (plan.get("channel") or plan.get("lob") or "").split(",")[0].strip()
            SITE = (plan.get("site") or plan.get("location") or plan.get("country") or "").strip()

            def _match(series, val):
                if not val or not isinstance(series, pd.Series):
                    return pd.Series(True, index=series.index)
                s = series.astype(str).str.strip().str.lower()
                return s.eq(str(val).strip().lower())

            msk = pd.Series(True, index=df.index)
            if c_ba:
                msk &= _match(df[c_ba], BA)
            if c_sba and (SBA not in (None, "")):
                msk &= _match(df[c_sba], SBA)
            if c_lob:
                msk &= _match(df[c_lob], LOB)
            if c_site and (SITE not in (None, "")):
                msk &= _match(df[c_site], SITE)
            df = df[msk]
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
                df = df[df["date"].eq(day)]
            times: List[dt.time] = []
            if "entry" in df.columns:
                for s in df["entry"].astype(str):
                    m = re.match(
                        r"^(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)$",
                        s
                    )
                    if not m:
                        continue
                    t1 = _parse_time_any(m.group(1))
                    t2 = _parse_time_any(m.group(2))
                    if t1:
                        times.append(t1)
                    if t2:
                        times.append(t2)
            if times:
                tmin = min(times)
                tmax = max(times)
                if not start:
                    start = _fmt_hhmm(tmin)
                if not end:
                    end = _fmt_hhmm(tmax)
    return (start or "08:00"), end


def _staff_by_slot_for_day(plan: dict, day: dt.date, ivl_ids: List[str],
                           start_hhmm: str, ivl_min: int) -> Dict[str, float]:
    """Compute staffing by interval, using cached roster and pre-parsed times."""
    try:
        rl = _get_roster()
    except Exception:
        return {lab: 0.0 for lab in ivl_ids}

    if not isinstance(rl, pd.DataFrame) or rl.empty:
        return {lab: 0.0 for lab in ivl_ids}

    df = rl.copy()

    def _col(opts):
        for c in opts:
            if c in df.columns:
                return c
        return None

    c_ba = _col(["Business Area", "business area", "vertical"])
    c_sba = _col(["Sub Business Area", "sub business area", "sub_ba"])
    c_lob = _col(["LOB", "lob", "Channel", "channel"])
    c_site = _col(["Site", "site", "Location", "location", "Country", "country"])
    BA = plan.get("vertical")
    SBA = plan.get("sub_ba")
    LOB = (plan.get("channel") or plan.get("lob") or "").split(",")[0].strip()
    SITE = (plan.get("site") or plan.get("location") or plan.get("country") or "").strip()

    def _match(series, val):
        if not val or not isinstance(series, pd.Series):
            return pd.Series(True, index=series.index)
        s = series.astype(str).str.strip().str.lower()
        return s.eq(str(val).strip().lower())

    msk = pd.Series(True, index=df.index)
    if c_ba:
        msk &= _match(df[c_ba], BA)
    if c_sba and (SBA not in (None, "")):
        msk &= _match(df[c_sba], SBA)
    if c_lob:
        msk &= _match(df[c_lob], LOB)
    if c_site and (SITE not in (None, "")):
        msk &= _match(df[c_site], SITE)

    df = df[msk]

    if "is_leave" in df.columns:
        df = df[~df["is_leave"].astype(bool)]
    if "date" not in df.columns:
        return {lab: 0.0 for lab in ivl_ids}

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df[df["date"].eq(day)]
    slots = {lab: 0.0 for lab in ivl_ids}
    if df.empty:
        return slots

    def _parse_hhmm_to_min(hhmm: str) -> int:
        try:
            h, m = hhmm.split(":", 1)
            return int(h) * 60 + int(m)
        except Exception:
            return 0

    cov_start_min = _parse_hhmm_to_min(start_hhmm)
    ivl_count = len(ivl_ids)
    ivl_length = int(ivl_min)
    # Use numpy array for faster accumulation
    slots_arr = np.zeros(ivl_count, dtype=float)

    # Use pre-parsed start_min/end_min if present; fallback to parsing entry (rare)
    for _, rr in df.iterrows():
        try:
            start_min = rr.get("start_min", np.nan)
            end_min = rr.get("end_min", np.nan)
            if not np.isfinite(start_min) or not np.isfinite(end_min):
                sft = str(rr.get("entry", "")).strip()
                m = re.match(r"^(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})$", sft)
                if not m:
                    continue
                sh, sm, eh, em = map(int, m.groups())
                sh = min(23, max(0, sh))
                eh = min(24, max(0, eh))
                start_min = sh * 60 + sm
                end_min = eh * 60 + em
                if end_min <= start_min:
                    end_min += 24 * 60

            # We keep the original wrap logic by checking each slot relative to shift.
            for idx2 in range(ivl_count):
                slot_abs = cov_start_min + idx2 * ivl_length
                slot_rel = slot_abs
                if slot_rel < start_min:
                    slot_rel += 24 * 60
                if start_min <= slot_rel < end_min:
                    slots_arr[idx2] += 1.0
        except Exception:
            continue

    for idx2, lab2 in enumerate(ivl_ids):
        slots[lab2] = float(slots_arr[idx2])
    return slots


def _erlang_c(A: float, N: int) -> float:
    if N <= 0:
        return 1.0
    if A <= 0:
        return 0.0
    if A >= N:
        return 1.0
    term = 1.0
    ssum = term
    for k in range(1, N):
        term *= A / k
        ssum += term
    term *= A / N
    last = term * (N / (N - A))
    denom = ssum + last
    if denom <= 0:
        return 1.0
    p0 = 1.0 / denom
    return last * p0


def _erlang_sl(A_calls: float, aht: float, agents: float, ivl_sec: float, T_sec: float) -> float:
    if aht <= 0 or ivl_sec <= 0 or agents <= 0:
        return 0.0
    if A_calls <= 0:
        return 1.0
    A = (A_calls * aht) / ivl_sec
    pw = _erlang_c(A, int(max(1, math.floor(agents))))
    return max(
        0.0,
        min(
            1.0,
            1.0 - pw * math.exp(-max(0.0, (agents - A)) * (T_sec / max(1.0, aht))),
        ),
    )


# ----------------------------------------------------------------------
# MEMOIZED WRAPPERS FOR HEAVY CALCS
# ----------------------------------------------------------------------

@lru_cache(maxsize=100_000)
def _erlang_sl_cached(calls_rounded: int, aht_rounded: int,
                      agents_rounded: int, ivl_sec: int, T_sec: int) -> float:
    """Cached version using rounded inputs as key to limit state."""
    return _erlang_sl(
        float(calls_rounded),
        float(aht_rounded),
        float(agents_rounded),
        float(ivl_sec),
        float(T_sec),
    )


@lru_cache(maxsize=100_000)
def _min_agents_cached(calls_rounded: int, aht_rounded: int,
                       ivl_min: int, target_sl: float, T_sec: int,
                       occ_cap_scaled: int) -> Tuple[float, float, float, float]:
    """Cached wrapper for min_agents. occ_cap_scaled = int(occ_cap*1000)."""
    occ_cap = float(occ_cap_scaled) / 1000.0
    return min_agents(
        float(calls_rounded),
        float(aht_rounded),
        int(ivl_min),
        float(target_sl),
        float(T_sec),
        float(occ_cap),
    )


def _make_upper_table(df: pd.DataFrame, ivl_cols: List[dict]):
    if not isinstance(df, pd.DataFrame) or df.empty:
        df = pd.DataFrame({"metric": []})
    return dash_table.DataTable(
        id="tbl-upper",
        data=df.to_dict("records"),
        columns=[{"name": "Metric", "id": "metric", "editable": False}]
        + [{"name": c["name"], "id": c["id"]} for c in ivl_cols if c["id"] != "metric"],
        editable=False,
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={"whiteSpace": "pre"},
    )


def _fill_tables_fixed_interval(
    ptype,
    pid,
    _fw_cols_unused,
    _tick,
    whatif=None,
    ivl_min: int = 30,
    sel_date: Optional[str] = None,
):
    """Interval view (data-first):
    - Render FW intervals exactly as uploaded for the selected date
    - Compute Upper (PHC/SL) and FW Occupancy via Erlang using uploaded intervals + roster
    - Other grids are left empty (or can be loaded from persistence by callers)
    """
    plan = get_plan(pid) or {}
    # pick representative date
    if sel_date:
        try:
            ref_day = pd.to_datetime(sel_date).date()
        except Exception:
            ref_day = dt.date.today()
    else:
        ref_day = dt.date.today()

    ch = (plan.get("channel") or plan.get("lob") or "").split(",")[0].strip().lower()
    sk = _canon_scope(
        plan.get("vertical"),
        plan.get("sub_ba"),
        ch,
        (plan.get("site") or plan.get("location") or plan.get("country") or "").strip(),
    )
    settings = resolve_settings(
        ba=plan.get("vertical"), subba=plan.get("sub_ba"), lob=ch
    )
    calc_bundle = (
        get_cached_consolidated_calcs(
            int(pid),
            settings=settings,
            version_token=_tick,
        )
        if pid
        else {}
    )

    def _from_bundle(key: str) -> pd.DataFrame:
        if not isinstance(calc_bundle, dict):
            return pd.DataFrame()
        val = calc_bundle.get(key)
        return val if isinstance(val, pd.DataFrame) else pd.DataFrame()

    # Prefer cached interval calcs; fallback to assembled raw uploads
    if ch == "voice":
        vF = _from_bundle("voice_ivl_f")
        vA = _from_bundle("voice_ivl_a")
        vT = _from_bundle("voice_ivl_t")
        if vF.empty and vA.empty and vT.empty:
            vF = _assemble_voice(sk, "forecast")
            vA = _assemble_voice(sk, "actual")
            vT = _assemble_voice(sk, "tactical")
    elif ch == "chat":
        cF = _from_bundle("chat_ivl_f")
        cA = _from_bundle("chat_ivl_a")
        cT = _from_bundle("chat_ivl_t")
        if cF.empty and cA.empty and cT.empty:
            cF = _assemble_chat(sk, "forecast")
            cA = _assemble_chat(sk, "actual")
            cT = _assemble_chat(sk, "tactical")
        vF = cF
        vA = cA
        vT = cT
    else:  # outbound
        oF = _from_bundle("ob_ivl_f")
        oA = _from_bundle("ob_ivl_a")
        oT = _from_bundle("ob_ivl_t")
        if oF.empty and oA.empty and oT.empty:
            oF = _assemble_ob(sk, "forecast")
            oA = _assemble_ob(sk, "actual")
            oT = _assemble_ob(sk, "tactical")
        vF = oF
        vA = oA
        vT = oT

    # Infer window using already-loaded interval data; fallback to roster-based inference
    def _window_from_df(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
        try:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return None, None
            d = df.copy()
            L = {str(c).strip().lower(): c for c in d.columns}
            c_date = L.get("date") or L.get("day")
            ivc = _pick_ivl_col(d)
            if not ivc:
                return None, None
            if c_date:
                d[c_date] = pd.to_datetime(d[c_date], errors="coerce").dt.date
                d = d[d[c_date].eq(ref_day)]
            if d.empty:
                return None, None
            times = d[ivc].astype(str).map(_parse_time_any).dropna()
            if times.empty:
                return None, None
            return _fmt_hhmm(times.min()), _fmt_hhmm(times.max())
        except Exception:
            return None, None

    start_hhmm, end_hhmm = None, None
    for df in (vF, vA):
        s, e = _window_from_df(df)
        start_hhmm = start_hhmm or s
        end_hhmm = end_hhmm or e
        if start_hhmm and end_hhmm:
            break
    if not start_hhmm or not end_hhmm:
        start_hhmm, end_hhmm = _infer_window(plan, ref_day, ch, sk)

    ivl_cols, ivl_ids = interval_cols_for_day(
        ref_day, ivl_min=ivl_min, start_hhmm=start_hhmm, end_hhmm=end_hhmm
    )
    cols = ivl_ids  # backward compat alias

    # FW metrics
    fw_metrics: List[str] = []
    weekly = None
    try:
        weeks = _week_span(plan.get("start_week"), plan.get("end_week"))
        weekly_fw_cols = [{"name": "Metric", "id": "metric", "editable": False}] + [
            {"name": w, "id": w} for w in weeks
        ]
        weekly = _fill_tables_fixed(
            ptype, pid, weekly_fw_cols, _tick, whatif=whatif, grain="week"
        )
        (_upper_w, fw_w, *_rest) = weekly
        fw_df = pd.DataFrame(fw_w or [])
        if (
            isinstance(fw_df, pd.DataFrame)
            and not fw_df.empty
            and "metric" in fw_df.columns
        ):
            fw_metrics = fw_df["metric"].astype(str).tolist()
    except Exception:
        fw_metrics = []
    if not fw_metrics:
        fw_metrics = [
            "Forecast",
            "Tactical Forecast",
            "Actual Volume",
            "Forecast AHT/SUT",
            "Actual AHT/SUT",
            "Occupancy",
        ]
    fw_i = pd.DataFrame({"metric": fw_metrics})
    for lab in ivl_ids:
        fw_i[lab] = np.nan

    # Upper rows shaped to match weekly Upper spec (fields/ordering)
    default_upper_rows: List[str] = [
        "FTE Required @ Forecast Volume",
        "FTE Required @ Actual Volume",
        "FTE Over/Under MTP Vs Actual",
        "FTE Over/Under Tactical Vs Actual",
        "FTE Over/Under Budgeted Vs Actual",
        "Projected Supply HC",
        "Projected Handling Capacity (#)",
        "Projected Service Level",
    ]

    upper_rows = default_upper_rows.copy()
    try:
        # Reuse weekly call above (cached in `weekly`) to pick weekly upper spec
        if weekly is None:
            weeks = _week_span(plan.get("start_week"), plan.get("end_week"))
            weekly_fw_cols = [{"name": "Metric", "id": "metric", "editable": False}] + [
                {"name": w, "id": w} for w in weeks
            ]
            weekly = _fill_tables_fixed(ptype, pid, weekly_fw_cols, _tick, whatif=whatif, grain="week")
        (upper_wk, *_rest) = weekly
        upper_df_w = pd.DataFrame(getattr(upper_wk, "data", None) or [])
        if isinstance(upper_df_w, pd.DataFrame) and not upper_df_w.empty and "metric" in upper_df_w.columns:
            wk_rows = upper_df_w["metric"].astype(str).tolist()
            # Keep default order; append any extra weekly rows at the end
            upper_rows = [r for r in default_upper_rows if r in wk_rows] + [
                r for r in wk_rows if r not in default_upper_rows
            ]
    except Exception:
        pass

    upper = pd.DataFrame({"metric": upper_rows})
    for lab in ivl_ids:
        upper[lab] = 0.0


    ivl_sec = max(60, int(ivl_min) * 60)
    T_sec = int(float(settings.get("sl_seconds", 20) or 20.0))
    target_sl = float(settings.get("target_sl", 0.8) or 0.8)

    # Precompute metric masks to avoid repeated astype/eq inside loops
    fw_metric_str = fw_i["metric"].astype(str)
    fw_has_occ = "Occupancy" in fw_metric_str.values
    fw_m_occ = fw_metric_str.eq("Occupancy") if fw_has_occ else None

    upper_metric_str = upper["metric"].astype(str)
    m_req_forecast = upper_metric_str.eq("FTE Required @ Forecast Volume")
    m_req_actual = upper_metric_str.eq("FTE Required @ Actual Volume")
    m_over_under = upper_metric_str.eq("FTE Over/Under (#)")
    m_over_mtp = upper_metric_str.eq("FTE Over/Under MTP Vs Actual")
    m_over_tac = upper_metric_str.eq("FTE Over/Under Tactical Vs Actual")
    m_over_budget = upper_metric_str.eq("FTE Over/Under Budgeted Vs Actual")
    m_supply = upper_metric_str.eq("Projected Supply HC")
    m_cap = upper_metric_str.eq("Projected Handling Capacity (#)")
    m_sl = upper_metric_str.eq("Projected Service Level")

    # Channel-specific fills
    if ch == "voice":
        volF = _slot_series_for_day(vF, ref_day, "volume")
        volA = _slot_series_for_day(
            vA if isinstance(vA, pd.DataFrame) and not vA.empty else vF,
            ref_day,
            "volume",
        )
        volT = _slot_series_for_day(vT, ref_day, "volume")
        ahtF = _slot_series_for_day(vF, ref_day, "aht_sec")
        ahtA = _slot_series_for_day(vA, ref_day, "aht_sec")

        # Budgeted AHT for week of ref_day, if provided
        ahtB_val = None
        try:
            dfp = _load_ts_with_fallback("voice_planned_aht", sk)
        except Exception:
            dfp = pd.DataFrame()
        if isinstance(dfp, pd.DataFrame) and not dfp.empty:
            d = dfp.copy()
            if "week" in d.columns:
                d["week"] = pd.to_datetime(d["week"], errors="coerce").dt.date
            elif "date" in d.columns:
                d["week"] = pd.to_datetime(d["date"], errors="coerce").dt.date
            refw = (pd.to_datetime(ref_day).date() - dt.timedelta(days=ref_day.weekday()))
            dd = d[pd.to_datetime(d["week"], errors="coerce").dt.date.eq(refw)]
            for c in ("aht_sec", "aht", "avg_aht"):
                if c in dd.columns:
                    v = pd.to_numeric(dd[c], errors="coerce").dropna()
                    if not v.empty:
                        ahtB_val = float(v.iloc[-1])
                        break
        if ahtB_val is None:
            try:
                dfb = _load_ts_with_fallback("voice_budget", sk)
            except Exception:
                dfb = pd.DataFrame()
            if isinstance(dfb, pd.DataFrame) and not dfb.empty:
                d = dfb.copy()
                if "week" in d.columns:
                    d["week"] = pd.to_datetime(d["week"], errors="coerce").dt.date
                elif "date" in d.columns:
                    d["week"] = pd.to_datetime(d["date"], errors="coerce").dt.date
                refw = (pd.to_datetime(ref_day).date() - dt.timedelta(days=ref_day.weekday()))
                dd = d[pd.to_datetime(d["week"], errors="coerce").dt.date.eq(refw)]
                for c in ("budget_aht_sec", "aht_sec", "aht"):
                    if c in dd.columns:
                        v = pd.to_numeric(dd[c], errors="coerce").dropna()
                        if not v.empty:
                            ahtB_val = float(v.iloc[-1])
                            break

        mser = fw_metric_str
        has_forecast = "Forecast" in mser.values
        has_tactical = "Tactical Forecast" in mser.values
        has_actual_vol = "Actual Volume" in mser.values
        has_forecast_aht = "Forecast AHT/SUT" in mser.values
        has_actual_aht = "Actual AHT/SUT" in mser.values
        has_budget_aht = "Budgeted AHT/SUT" in mser.values

        for lab in ivl_ids:
            if has_forecast and lab in volF:
                fw_i.loc[mser == "Forecast", lab] = float(volF[lab])
            if has_tactical and lab in volT:
                fw_i.loc[mser == "Tactical Forecast", lab] = float(volT[lab])
            if has_actual_vol and lab in volA:
                fw_i.loc[mser == "Actual Volume", lab] = float(volA[lab])
            if has_forecast_aht and lab in ahtF:
                fw_i.loc[mser == "Forecast AHT/SUT", lab] = float(ahtF[lab])
            if has_actual_aht and lab in ahtA:
                fw_i.loc[mser == "Actual AHT/SUT", lab] = float(ahtA[lab])
            if has_budget_aht and ahtB_val is not None:
                fw_i.loc[mser == "Budgeted AHT/SUT", lab] = float(ahtB_val)

        # Staffing and Erlang rollups
        staff = _staff_by_slot_for_day(plan, ref_day, ivl_ids, start_hhmm, ivl_min)
        occ_cap = float(settings.get("occupancy_cap_voice", 0.85) or 0.85)
        occ_cap_scaled = int(round(occ_cap * 1000))

        for lab in ivl_ids:
            ag = float(staff.get(lab, 0.0) or 0.0)
            calls = float(volF.get(lab, 0.0))
            # use any AHT value, fallback to 300
            default_ahtF_val = ahtF.get(next(iter(ahtF), lab), 300.0) if ahtF else 300.0
            aht = float(ahtF.get(lab, default_ahtF_val) or default_ahtF_val)

            calls_round = int(round(calls))
            aht_round = int(round(aht))

            # FTE Required rows (agents per interval)
            if calls_round > 0 and aht_round > 0:
                Nf, _slN, _occN, _asaN = _min_agents_cached(
                    calls_round, aht_round, int(ivl_min), float(target_sl),
                    int(T_sec), occ_cap_scaled
                )
                if m_req_forecast.any():
                    upper.loc[m_req_forecast, lab] = float(Nf)

            calls_a = float(volA.get(lab, 0.0))
            calls_a_round = int(round(calls_a))
            aht_a = float(ahtA.get(lab, aht))
            aht_a_round = int(round(aht_a)) if aht_a > 0 else 0
            Na = 0.0
            if calls_a_round > 0 and aht_a_round > 0:
                Na, _slNa, _occNa, _asaNa = _min_agents_cached(
                    calls_a_round, aht_a_round, int(ivl_min),
                    float(target_sl), int(T_sec), occ_cap_scaled
                )
                if m_req_actual.any():
                    upper.loc[m_req_actual, lab] = float(Na)

            # Over/Under rows
            req = 0.0
            if calls_a_round > 0 and aht_a_round > 0:
                req = Na
            elif calls_round > 0 and aht_round > 0:
                req = Nf

            if m_over_under.any():
                upper.loc[m_over_under, lab] = float(ag) - float(req)

            if m_over_mtp.any() and (calls_round > 0 or calls_a_round > 0):
                base = Na if (calls_a_round > 0 and aht_a_round > 0) else 0.0
                val = (Nf if (calls_round > 0 and aht_round > 0) else 0.0) - float(
                    base
                )
                upper.loc[m_over_mtp, lab] = float(val)

            if m_over_tac.any() and lab in volT:
                calls_t = float(volT.get(lab, 0.0))
                calls_t_round = int(round(calls_t))
                if calls_t_round > 0 and aht_round > 0:
                    Nt, _slT, _occT, _asaT = _min_agents_cached(
                        calls_t_round, aht_round, int(ivl_min),
                        float(target_sl), int(T_sec), occ_cap_scaled
                    )
                    base = Na if (calls_a_round > 0 and aht_a_round > 0) else 0.0
                    upper.loc[m_over_tac, lab] = float(Nt) - float(base)

            if m_over_budget.any() and ahtB_val:
                base = Na if (calls_a_round > 0 and aht_a_round > 0) else 0.0
                if calls_round > 0:
                    ahtB_round = int(round(float(ahtB_val)))
                    Nb, _slB, _occB, _asaB = _min_agents_cached(
                        calls_round, ahtB_round, int(ivl_min),
                        float(target_sl), int(T_sec), occ_cap_scaled
                    )
                else:
                    Nb = 0.0
                upper.loc[m_over_budget, lab] = float(Nb) - float(base)

            # Projected Supply HC
            if m_supply.any():
                upper.loc[m_supply, lab] = float(ag)

            # Projected capacity & SL using cached Erlang
            cap = 0.0
            sl_val = 0.0
            if aht_round > 0 and ivl_sec > 0:
                # occupancy-limited calls:
                occ_calls = (occ_cap * ag * ivl_sec) / max(1.0, aht)
                max_hi = int(max(1, occ_calls))
                # If even at max_hi you don't meet target SL, cap at occ_calls
                hi_key = max_hi
                sl_at_hi = _erlang_sl_cached(
                    hi_key,
                    aht_round,
                    int(round(ag)),
                    int(ivl_sec),
                    int(T_sec),
                )
                if sl_at_hi < target_sl:
                    cap = float(max_hi)
                else:
                    lo, hi = 0, max_hi
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                            # noinspection PyTypeChecker
                        sl_mid = _erlang_sl_cached(
                            mid,
                            aht_round,
                            int(round(ag)),
                            int(ivl_sec),
                            int(T_sec),
                        )
                        if sl_mid >= target_sl:
                            lo = mid
                        else:
                            hi = mid - 1
                    cap = float(lo)
                # SL at forecast load
                calls_round_for_sl = int(round(calls))
                sl_val = 100.0 * _erlang_sl_cached(
                    calls_round_for_sl,
                    aht_round,
                    int(round(ag)),
                    int(ivl_sec),
                    int(T_sec),
                )

            if m_cap.any():
                upper.loc[m_cap, lab] = cap
            if m_sl.any():
                upper.loc[m_sl, lab] = sl_val

            # FW Occupancy
            A = (calls * aht) / ivl_sec if aht > 0 and ivl_sec > 0 else 0.0
            occ = 100.0 * min(
                occ_cap, (A / max(ag, 1e-6)) if ag > 0 else 0.0
            )
            if fw_has_occ and fw_m_occ is not None:
                fw_i.loc[fw_m_occ, lab] = occ

    elif ch == "chat":
        cF = vF
        cA = vA
        cT = vT
        volF_map = _slot_series_for_day(cF, ref_day, "items") or _slot_series_for_day(
            cF, ref_day, "volume"
        )
        volA_map = _slot_series_for_day(
            cA if isinstance(cA, pd.DataFrame) and not cA.empty else cF,
            ref_day,
            "items",
        )
        volT_map = _slot_series_for_day(cT, ref_day, "items") or {}
        aht_map = _slot_series_for_day(
            cF, ref_day, "aht_sec"
        ) or _slot_series_for_day(cF, ref_day, "aht")

        mser = fw_metric_str
        has_forecast = "Forecast" in mser.values
        has_tactical = "Tactical Forecast" in mser.values
        has_actual_vol = "Actual Volume" in mser.values
        has_forecast_aht = "Forecast AHT/SUT" in mser.values

        for lab in ivl_ids:
            if has_forecast and lab in volF_map:
                fw_i.loc[mser == "Forecast", lab] = float(volF_map[lab])
            if has_tactical and lab in volT_map:
                fw_i.loc[mser == "Tactical Forecast", lab] = float(volT_map[lab])
            if has_actual_vol and lab in volA_map:
                fw_i.loc[mser == "Actual Volume", lab] = float(volA_map[lab])
            if has_forecast_aht and lab in aht_map:
                fw_i.loc[mser == "Forecast AHT/SUT", lab] = float(aht_map[lab])

        staff = _staff_by_slot_for_day(plan, ref_day, ivl_ids, start_hhmm, ivl_min)
        ivl_sec = max(60, int(ivl_min) * 60)
        T_sec_chat = int(
            float(settings.get("chat_sl_seconds", settings.get("sl_seconds", 20)) or 20.0)
        )
        target_sl_chat = float(
            settings.get("chat_target_sl", settings.get("target_sl", 0.8)) or 0.8
        )
        occ_cap_chat = float(
            settings.get(
                "occupancy_cap_chat",
                settings.get("util_chat", settings.get("occupancy_cap_voice", 0.85)),
            )
            or 0.85
        )
        occ_cap_scaled_chat = int(round(occ_cap_chat * 1000))
        conc = float(settings.get("chat_concurrency", 1.5) or 1.0)

        for lab in ivl_ids:
            ag = float(staff.get(lab, 0.0) or 0.0)
            items = float(volF_map.get(lab, 0.0))
            default_aht = aht_map.get(next(iter(aht_map), lab), 240.0) if aht_map else 240.0
            aht_eff = float(aht_map.get(lab, default_aht) or default_aht) / max(0.1, conc)
            items_round = int(round(items))
            aht_round = int(round(aht_eff)) if aht_eff > 0 else 0

            # FTE Required rows (agents per interval) using effective AHT
            if items_round > 0 and aht_round > 0:
                Nf, _slN, _occN, _asaN = _min_agents_cached(
                    items_round,
                    aht_round,
                    int(ivl_min),
                    float(target_sl_chat),
                    int(T_sec_chat),
                    occ_cap_scaled_chat,
                )
                if m_req_forecast.any():
                    upper.loc[m_req_forecast, lab] = float(Nf)

            items_a = float(volA_map.get(lab, 0.0))
            items_a_round = int(round(items_a))
            Na = 0.0
            if items_a_round > 0 and aht_round > 0:
                Na, _slNa, _occNa, _asaNa = _min_agents_cached(
                    items_a_round,
                    aht_round,
                    int(ivl_min),
                    float(target_sl_chat),
                    int(T_sec_chat),
                    occ_cap_scaled_chat,
                )
                if m_req_actual.any():
                    upper.loc[m_req_actual, lab] = float(Na)

            # Deltas vs actual + supply row
            if m_over_mtp.any() and (items_round > 0 or items_a_round > 0):
                base = Na if items_a_round > 0 else 0.0
                upper.loc[m_over_mtp, lab] = float(Nf if items_round > 0 else 0.0) - float(
                    base
                )
            if m_over_tac.any() and lab in volT_map:
                it = float(volT_map.get(lab, 0.0))
                it_round = int(round(it))
                if it_round > 0 and aht_round > 0:
                    Nt, _slT, _occT, _asaT = _min_agents_cached(
                        it_round,
                        aht_round,
                        int(ivl_min),
                        float(target_sl_chat),
                        int(T_sec_chat),
                        occ_cap_scaled_chat,
                    )
                    base = Na if items_a_round > 0 else 0.0
                    upper.loc[m_over_tac, lab] = float(Nt) - float(base)
            if m_over_budget.any():
                base = Na if items_a_round > 0 else 0.0
                upper.loc[m_over_budget, lab] = float(
                    Nf if items_round > 0 else 0.0
                ) - float(base)
            if m_supply.any():
                upper.loc[m_supply, lab] = float(ag)

            # Over/Under if row present
            if m_over_under.any():
                req = (
                    Na
                    if (items_a_round > 0 and aht_round > 0)
                    else (Nf if (items_round > 0 and aht_round > 0) else 0.0)
                )
                upper.loc[m_over_under, lab] = float(ag) - float(req)

            # capacity search (as voice)
            cap = 0.0
            occ_calls = (occ_cap_chat * ag * ivl_sec) / max(1.0, aht_eff)
            max_hi = int(max(1, occ_calls))

            if aht_round > 0 and ivl_sec > 0:
                sl_hi = _erlang_sl_cached(
                    max_hi,
                    aht_round,
                    int(round(ag)),
                    int(ivl_sec),
                    int(T_sec_chat),
                )
                if sl_hi < target_sl_chat:
                    cap = float(max_hi)
                else:
                    lo, hi = 0, max_hi
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        sl_mid = _erlang_sl_cached(
                            mid,
                            aht_round,
                            int(round(ag)),
                            int(ivl_sec),
                            int(T_sec_chat),
                        )
                        if sl_mid >= target_sl_chat:
                            lo = mid
                        else:
                            hi = mid - 1
                    cap = float(lo)
                sl_val = 100.0 * _erlang_sl_cached(
                    int(round(items)),
                    aht_round,
                    int(round(ag)),
                    int(ivl_sec),
                    int(T_sec_chat),
                )
            else:
                sl_val = 0.0

            if m_cap.any():
                upper.loc[m_cap, lab] = cap
            if m_sl.any():
                upper.loc[m_sl, lab] = sl_val

            A = (items * aht_eff) / ivl_sec if aht_eff > 0 and ivl_sec > 0 else 0.0
            occ = 100.0 * min(
                occ_cap_chat, (A / max(ag, 1e-6)) if ag > 0 else 0.0
            )
            if fw_has_occ and fw_m_occ is not None:
                fw_i.loc[fw_m_occ, lab] = occ

    elif ch in ("outbound", "ob"):
        def _alias_opc(df: pd.DataFrame) -> pd.DataFrame:
            if not isinstance(df, pd.DataFrame):
                return pd.DataFrame()
            if "opc" not in df.columns and "items" in df.columns:
                return df.rename(columns={"items": "opc"})
            return df

        oF = _alias_opc(vF)
        oA = _alias_opc(vA)
        oT = _alias_opc(vT)
        volF = (
            _slot_series_for_day(oF, ref_day, "opc")
            or _slot_series_for_day(oF, ref_day, "dials")
            or _slot_series_for_day(oF, ref_day, "calls")
            or _slot_series_for_day(oF, ref_day, "volume")
            or _slot_series_for_day(oF, ref_day, "items")
        )
        volA = (
            _slot_series_for_day(oA, ref_day, "opc")
            or _slot_series_for_day(oA, ref_day, "dials")
            or _slot_series_for_day(oA, ref_day, "calls")
            or _slot_series_for_day(oA, ref_day, "volume")
            or _slot_series_for_day(oA, ref_day, "items")
        )
        volT = (
            _slot_series_for_day(oT, ref_day, "opc")
            or _slot_series_for_day(oT, ref_day, "items")
            or {}
        )
        aht_map = _slot_series_for_day(oF, ref_day, "aht_sec") or _slot_series_for_day(
            oF, ref_day, "aht"
        )

        mser = fw_metric_str
        has_forecast = "Forecast" in mser.values
        has_tactical = "Tactical Forecast" in mser.values
        has_actual_vol = "Actual Volume" in mser.values
        has_forecast_aht = "Forecast AHT/SUT" in mser.values

        for lab in ivl_ids:
            if has_forecast and lab in volF:
                fw_i.loc[mser == "Forecast", lab] = float(volF[lab])
            if has_tactical and lab in volT:
                fw_i.loc[mser == "Tactical Forecast", lab] = float(volT[lab])
            if has_actual_vol and lab in volA:
                fw_i.loc[mser == "Actual Volume", lab] = float(volA[lab])
            if has_forecast_aht and lab in aht_map:
                fw_i.loc[mser == "Forecast AHT/SUT", lab] = float(aht_map[lab])

        staff = _staff_by_slot_for_day(plan, ref_day, ivl_ids, start_hhmm, ivl_min)
        ivl_sec = max(60, int(ivl_min) * 60)
        T_sec_ob = int(
            float(settings.get("ob_sl_seconds", settings.get("sl_seconds", 20)) or 20.0)
        )
        target_sl_ob = float(
            settings.get("ob_target_sl", settings.get("target_sl", 0.8)) or 0.8
        )
        occ_cap_ob = float(
            settings.get(
                "occupancy_cap_ob",
                settings.get("util_ob", settings.get("occupancy_cap_voice", 0.85)),
            )
            or 0.85
        )
        occ_cap_scaled_ob = int(round(occ_cap_ob * 1000))

        for lab in ivl_ids:
            ag = float(staff.get(lab, 0.0) or 0.0)
            calls = float(volF.get(lab, 0.0))
            default_aht = aht_map.get(next(iter(aht_map), lab), 240.0) if aht_map else 240.0
            aht = float(aht_map.get(lab, default_aht) or default_aht)
            calls_round = int(round(calls))
            aht_round = int(round(aht)) if aht > 0 else 0

            # FTE Required rows (agents per interval)
            if calls_round > 0 and aht_round > 0:
                Nf, _slN, _occN, _asaN = _min_agents_cached(
                    calls_round,
                    aht_round,
                    int(ivl_min),
                    float(target_sl_ob),
                    int(T_sec_ob),
                    occ_cap_scaled_ob,
                )
                if m_req_forecast.any():
                    upper.loc[m_req_forecast, lab] = float(Nf)

            calls_a = float(volA.get(lab, 0.0))
            calls_a_round = int(round(calls_a))
            Na = 0.0
            if calls_a_round > 0 and aht_round > 0:
                Na, _slNa, _occNa, _asaNa = _min_agents_cached(
                    calls_a_round,
                    aht_round,
                    int(ivl_min),
                    float(target_sl_ob),
                    int(T_sec_ob),
                    occ_cap_scaled_ob,
                )
                if m_req_actual.any():
                    upper.loc[m_req_actual, lab] = float(Na)

            # Deltas vs actual + supply row
            if m_over_mtp.any() and (calls_round > 0 or calls_a_round > 0):
                base = Na if calls_a_round > 0 else 0.0
                upper.loc[m_over_mtp, lab] = float(Nf if calls_round > 0 else 0.0) - float(
                    base
                )
            if m_over_tac.any() and lab in volT:
                ct = float(volT.get(lab, 0.0))
                ct_round = int(round(ct))
                if ct_round > 0 and aht_round > 0:
                    Nt, _slT, _occT, _asaT = _min_agents_cached(
                        ct_round,
                        aht_round,
                        int(ivl_min),
                        float(target_sl_ob),
                        int(T_sec_ob),
                        occ_cap_scaled_ob,
                    )
                    base = Na if calls_a_round > 0 else 0.0
                    upper.loc[m_over_tac, lab] = float(Nt) - float(base)
            if m_over_budget.any():
                base = Na if calls_a_round > 0 else 0.0
                upper.loc[m_over_budget, lab] = float(
                    Nf if calls_round > 0 else 0.0
                ) - float(base)
            if m_supply.any():
                upper.loc[m_supply, lab] = float(ag)

            # Over/Under if row present
            if m_over_under.any():
                req = (
                    Na
                    if (calls_a_round > 0 and aht_round > 0)
                    else (Nf if (calls_round > 0 and aht_round > 0) else 0.0)
                )
                upper.loc[m_over_under, lab] = float(ag) - float(req)

            # capacity & SL
            cap = 0.0
            occ_calls = (occ_cap_ob * ag * ivl_sec) / max(1.0, aht)
            max_hi = int(max(1, occ_calls))
            if aht_round > 0 and ivl_sec > 0:
                sl_hi = _erlang_sl_cached(
                    max_hi,
                    aht_round,
                    int(round(ag)),
                    int(ivl_sec),
                    int(T_sec_ob),
                )
                if sl_hi < target_sl_ob:
                    cap = float(max_hi)
                else:
                    lo, hi = 0, max_hi
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        sl_mid = _erlang_sl_cached(
                            mid,
                            aht_round,
                            int(round(ag)),
                            int(ivl_sec),
                            int(T_sec_ob),
                        )
                        if sl_mid >= target_sl_ob:
                            lo = mid
                        else:
                            hi = mid - 1
                    cap = float(lo)
                sl_val = 100.0 * _erlang_sl_cached(
                    calls_round,
                    aht_round,
                    int(round(ag)),
                    int(ivl_sec),
                    int(T_sec_ob),
                )
            else:
                sl_val = 0.0

            if m_cap.any():
                upper.loc[m_cap, lab] = cap
            if m_sl.any():
                upper.loc[m_sl, lab] = sl_val

            A = (calls * aht) / ivl_sec if aht > 0 and ivl_sec > 0 else 0.0
            occ = 100.0 * min(
                occ_cap_ob, (A / max(ag, 1e-6)) if ag > 0 else 0.0
            )
            if fw_has_occ and fw_m_occ is not None:
                fw_i.loc[fw_m_occ, lab] = occ

    # For future dates, override Occupancy with settings value
    try:
        today = pd.Timestamp("today").date()
        ch_low = str(ch).strip().lower()
        if isinstance(ref_day, dt.date) and ref_day > today and "metric" in fw_i.columns:
            m_occ_future = fw_i["metric"].astype(str).eq("Occupancy")
            if m_occ_future.any():
                if ch_low == "voice":
                    base = settings.get(
                        "occupancy_cap_voice", settings.get("occupancy", 0.85)
                    )
                elif ch_low == "chat":
                    base = settings.get(
                        "util_chat", settings.get("util_bo", 0.85)
                    )
                else:
                    base = settings.get(
                        "util_ob", settings.get("occupancy", 0.85)
                    )
                try:
                    occ_pct = float(base)
                    if occ_pct <= 1.0:
                        occ_pct *= 100.0
                except Exception:
                    occ_pct = 85.0
                for lab in ivl_ids:
                    fw_i.loc[m_occ_future, lab] = occ_pct
    except Exception:
        pass

    # Rounding: 1 decimal for Occupancy, Service Level, and Handling Capacity
    try:
        # Round FW Occupancy row to 1 decimal
        if "metric" in fw_i.columns:
            m_occ_round = fw_i["metric"].astype(str).eq("Occupancy")
            if m_occ_round.any():
                for lab in ivl_ids:
                    try:
                        fw_i[lab] = fw_i[lab].astype(object)
                    except Exception:
                        pass
                    vals = pd.to_numeric(
                        fw_i.loc[m_occ_round, lab], errors="coerce"
                    ).round(1)
                    fw_i.loc[m_occ_round, lab] = vals.astype(str) + "%"

        # Round upper SL and Capacity rows to 1 decimal
        if "metric" in upper.columns:
            for row_name in [
                "Projected Handling Capacity (#)",
                "Projected Service Level",
            ]:
                m = upper["metric"].astype(str).eq(row_name)
                if m.any():
                    for lab in ivl_ids:
                        if row_name == "Projected Service Level":
                            try:
                                upper[lab] = upper[lab].astype(object)
                            except Exception:
                                pass
                            vals = pd.to_numeric(
                                upper.loc[m, lab], errors="coerce"
                            ).round(1)
                            upper.loc[m, lab] = vals.astype(str) + "%"
                        else:
                            upper.loc[m, lab] = pd.to_numeric(
                                upper.loc[m, lab], errors="coerce"
                            ).round(1)
    except Exception:
        pass

    # Upper table component
    upper_tbl = _make_upper_table(upper, ivl_cols)

    # Other tabs: leave empty (callers can persist/load as needed)
    empty: List[dict] = []
    return (
        upper_tbl,
        fw_i.to_dict("records"),
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
    )
