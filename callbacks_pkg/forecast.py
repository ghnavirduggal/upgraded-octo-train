from __future__ import annotations
import base64
import io
import json
import re
import logging
import calendar
from typing import Any, Optional, Tuple
from prophet import Prophet
import dash
from dash import Input, Output, State, no_update, dcc, html, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from app_instance import app
from plan_detail._common import current_user_fallback
import cap_store
from forecasting.process_and_IQ_data import (
    IQ_data,
    forecast_group_pivot_and_long_style,
    process_forecast_results,
    plot_contact_ratio_seasonality,
    accuracy_phase1,
    fill_final_smoothed_row,
    create_download_csv_with_metadata,
    unpivot_iq_summary,
    map_original_volume_to_phase2_forecast,
    map_normalized_volume_to_forecast,
    fmt_percent1,
    fmt_millions_1,
    clean_and_convert_percentage,
    clean_and_convert_millions,
    add_editable_base_volume,
)
from forecasting.contact_ratio_dash import (
    run_contact_ratio_forecast,
    run_phase2_forecast,
    iterative_tuning,
    train_and_evaluate_func,
)
import config_manager
import os

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

_FORECAST_STORE_IDS = [
    "vs-data-store",
    "vs-results-store",
    "vs-iq-store",
    "vs-iq-summary-store",
    "vs-seasonality-store",
    "vs-prophet-store",
    "vs-holiday-store",
    "vs-phase1-config-store",
    "vs-phase1-download-store",
    "vs-phase2-store",
    "vs-adjusted-store",
    "sa-raw-store",
    "sa-results-store",
    "sa-seasonality-store",
    "fc-data-store",
    "fc-saved-list-store",
    "fc-saved-selected",
    "fc-config-store",
    "tp-raw-store",
    "tp-filtered-store",
    "tp-final-store",
    "di-page-init",
    "di-forecast-dates-store",
    "di-holidays-store",
    "di-transform-store",
    "di-interval-store",
    "di-results-store",
    "di-distribution-store",
]


@app.callback(
    [Output(store_id, "data") for store_id in _FORECAST_STORE_IDS],
    Input("url-router", "pathname"),
    prevent_initial_call=False,
)
def _clear_forecast_stores_on_nav(pathname):
    path = (pathname or "").rstrip("/")
    if path.startswith("/forecast"):
        raise dash.exceptions.PreventUpdate
    return [None] * len(_FORECAST_STORE_IDS)


def _parse_upload(contents: str, filename: str, decoded_bytes: Optional[bytes] = None) -> Tuple[pd.DataFrame, str]:
    if not contents or "," not in contents:
        return pd.DataFrame(), "No file supplied."
    sheet_note = ""
    decoded = decoded_bytes
    if decoded is None:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
    try:
        lower = filename.lower()
        if lower.endswith(".csv"):
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        else:
            try:
                xl = pd.ExcelFile(io.BytesIO(decoded))
                sheet_map = {
                    re.sub(r"[^a-z0-9]", "", str(name).lower()): name
                    for name in xl.sheet_names
                }
                volume_sheet = sheet_map.get("volume")
                if volume_sheet:
                    df = xl.parse(volume_sheet)
                    sheet_note = f" (sheet '{volume_sheet}')"
                else:
                    first_sheet = xl.sheet_names[0] if xl.sheet_names else None
                    df = xl.parse(first_sheet) if first_sheet else pd.DataFrame()
                    if first_sheet:
                        sheet_note = f" (sheet '{first_sheet}')"
            except Exception:
                try:
                    df = pd.read_excel(io.BytesIO(decoded), engine="openpyxl")
                except Exception:
                    # Fallback: attempt CSV decode
                    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        msg = f"Loaded {len(df):,} rows from {filename}{sheet_note}."
        return df, msg
    except Exception as exc:
        return pd.DataFrame(), f"Failed to read {filename}: {exc}"


def _pick_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    lookup: dict[str, str] = {}
    for c in df.columns:
        base = str(c).strip().lower()
        variants = {
            base,
            base.replace(" ", "_"),
            base.replace(" ", ""),
            base.replace("-", "_"),
            base.replace("-", ""),
            base.replace("_", ""),
            re.sub(r"[^a-z0-9]", "", base),
        }
        for v in variants:
            lookup.setdefault(v, c)

    for nm in candidates:
        key = str(nm).strip().lower()
        candidates_norm = {
            key,
            key.replace(" ", "_"),
            key.replace(" ", ""),
            key.replace("-", "_"),
            key.replace("-", ""),
            key.replace("_", ""),
            re.sub(r"[^a-z0-9]", "", key),
        }
        for cand in candidates_norm:
            col = lookup.get(cand)
            if col:
                return col
    return None


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    pivot = _category_month_pivot(df)
    if not pivot.empty:
        return pivot

    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    d = df.copy()
    date_col = _pick_col(d, ("date", "ds", "datetime", "timestamp", "month_start"))
    val_col = _pick_col(d, ("volume", "items", "calls", "count", "value"))

    if date_col and val_col:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.dropna(subset=[date_col])
        d["_month"] = d[date_col].dt.to_period("M").dt.to_timestamp()
        d[val_col] = pd.to_numeric(d[val_col], errors="coerce")
        grouped = (
            d.groupby("_month", as_index=False)[val_col]
            .sum()
            .rename(columns={val_col: "Total"})
        )
        grouped["Month"] = grouped["_month"].dt.strftime("%b-%y")
        grouped = grouped[["Month", "Total"]]
        return grouped

    stats = d.describe(include="all").reset_index().rename(columns={"index": "metric"})
    return stats


def _normalize_volume_df(df: pd.DataFrame, cat_hint: Optional[str] = None) -> pd.DataFrame:
    """Standardize common column names so downstream helpers can work with varied uploads."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    d = df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]

    # category / forecast group
    cat_col = None
    if cat_hint:
        cat_hint_norm = str(cat_hint).strip().lower()
        if cat_hint_norm in d.columns:
            cat_col = cat_hint_norm
    if not cat_col:
        cat_col = _pick_col(d, ("category", "forecast_group", "queue_name"))
    if cat_col and cat_col != "category":
        d = d.rename(columns={cat_col: "category"})
    if "category" not in d.columns:
        d["category"] = "All"
    if "forecast_group" not in d.columns:
        d["forecast_group"] = d["category"]
    d["category"] = d["category"].astype(str).str.strip()
    d["forecast_group"] = d["forecast_group"].astype(str).str.strip()
    d.loc[d["category"].str.lower() == "nan", "category"] = None
    d.loc[d["forecast_group"].str.lower() == "nan", "forecast_group"] = None

    # date and volume
    date_col = _pick_col(d, ("date", "ds", "datetime", "timestamp", "month_start"))
    if date_col and date_col != "date":
        d = d.rename(columns={date_col: "date"})
    vol_col = _pick_col(d, ("volume", "items", "calls", "count", "value", "y"))
    if vol_col and vol_col != "volume":
        d = d.rename(columns={vol_col: "volume"})

    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
    if "volume" in d.columns:
        d["volume"] = pd.to_numeric(d["volume"], errors="coerce")

    if "date" in d.columns and "volume" in d.columns:
        d = d.dropna(subset=["date", "volume"])

    return d


def _aggregate_monthly(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Aggregate uploaded history to month-level for downstream calculations."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(), "No data to aggregate."
    d = df.copy()
    # try to find key columns, fallback to first convertible
    date_col = _pick_col(d, ("date", "ds", "datetime", "timestamp"))
    if not date_col:
        for c in d.columns:
            try:
                test = pd.to_datetime(d[c], errors="coerce")
                if test.notna().any():
                    date_col = c
                    break
            except Exception:
                continue
    val_col = _pick_col(d, ("volume", "items", "calls", "count", "value", "y"))
    if not val_col:
        for c in d.columns:
            if pd.api.types.is_numeric_dtype(d[c]):
                val_col = c
                break
    cat_col = _pick_col(d, ("category", "forecast_group", "queue_name"))
    if not date_col or not val_col:
        return pd.DataFrame(), "Upload must include a date and volume column for monthly aggregation."

    d["_date"] = pd.to_datetime(d[date_col], errors="coerce")
    d["_volume"] = pd.to_numeric(d[val_col], errors="coerce")
    d = d.dropna(subset=["_date", "_volume"])
    d["Month_Start"] = d["_date"].dt.to_period("M").dt.to_timestamp()
    group_keys = ["Month_Start"]
    if cat_col:
        d["Category"] = d[cat_col].astype(str)
        group_keys.insert(0, "Category")
    agg = d.groupby(group_keys, as_index=False)["_volume"].sum()
    agg = agg.rename(columns={"_volume": "Volume"})
    if "Month_Start" in agg.columns:
        agg["Month_Start"] = pd.to_datetime(agg["Month_Start"], errors="coerce").dt.strftime("%b-%y")
    msg = f"Aggregated to {len(agg):,} monthly rows."
    return agg, msg


def _category_month_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot volumes to Category rows with month columns for UI previews."""
    norm = _normalize_volume_df(df)
    if norm.empty or "date" not in norm.columns or "volume" not in norm.columns:
        return pd.DataFrame()

    norm = norm.dropna(subset=["date", "volume"])
    if norm.empty:
        return pd.DataFrame()

    norm["__month_period"] = norm["date"].dt.to_period("M")
    grouped = norm.groupby(["category", "__month_period"], as_index=False)["volume"].sum()
    if grouped.empty:
        return pd.DataFrame()

    ordered_months = sorted(grouped["__month_period"].unique())
    month_labels = {p: p.strftime("%b-%y") for p in ordered_months}

    pivot = grouped.pivot_table(
        index="category",
        columns="__month_period",
        values="volume",
        aggfunc="sum",
        fill_value=0,
    )
    pivot = pivot.rename(columns=month_labels).reset_index()

    ordered_cols = ["category"] + [month_labels[p] for p in ordered_months]
    pivot = pivot.loc[:, ordered_cols]
    pivot = pivot.rename(columns={"category": "Category"})

    month_cols = [c for c in pivot.columns if c != "Category"]
    for col in month_cols:
        pivot[col] = pd.to_numeric(pivot[col], errors="coerce").round(0)

    pivot = pivot.sort_values("Category").reset_index(drop=True)
    return pivot


def _empty_fig(msg: str = ""):
    fig = go.Figure()
    if msg:
        fig.add_annotation(text=msg, showarrow=False)
    fig.update_layout(margin=dict(t=30, l=20, r=20, b=20))
    return fig


def _cols(df: pd.DataFrame):
    return [{"name": c, "id": c} for c in df.columns]


def _serialize_iq_results(results: dict) -> Optional[str]:
    if not results:
        return None
    payload = {}
    for cat, group in results.items():
        if not isinstance(group, dict):
            continue
        cat_key = str(cat)
        payload[cat_key] = {}
        for key, df in group.items():
            if isinstance(df, pd.DataFrame):
                payload[cat_key][key] = df.to_json(date_format="iso", orient="split")
    if not payload:
        return None
    return json.dumps(payload)


def _iq_payload_for_category(payload: dict, cat: str) -> dict:
    if not payload or not cat:
        return {}
    if cat in payload:
        return payload.get(cat, {})
    for key, value in payload.items():
        if str(key).strip().lower() == str(cat).strip().lower():
            return value
    return {}


def _read_split_df(data_json: Optional[str]) -> pd.DataFrame:
    if not data_json:
        return pd.DataFrame()
    try:
        return pd.read_json(io.StringIO(data_json), orient="split")
    except Exception:
        return pd.DataFrame()


def _iq_tables_from_store(iq_store: Optional[str], cat: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not iq_store or not cat:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    try:
        payload = json.loads(iq_store) if isinstance(iq_store, str) else iq_store
    except Exception:
        payload = {}
    cat_payload = _iq_payload_for_category(payload, cat)
    if not cat_payload:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return (
        _read_split_df(cat_payload.get("IQ")),
        _read_split_df(cat_payload.get("Volume")),
        _read_split_df(cat_payload.get("Contact_Ratio")),
    )


def _ratio_fig(df: pd.DataFrame, title: str):
    if df is None or df.empty:
        return _empty_fig("No ratio data")
    melted = df.melt(id_vars="Year", var_name="Month", value_name="Ratio")
    melted = melted.dropna(subset=["Ratio"])
    if melted.empty:
        return _empty_fig("No ratio data")
    fig = px.line(melted, x="Month", y="Ratio", color="Year", markers=True, title=title)
    fig.update_traces(mode="lines+markers")
    return fig


def _clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN/NA and 'nan' strings with blanks for display tables."""
    if df is None or df.empty:
        return df

    def _strip_nan(val):
        if isinstance(val, str):
            lowered = val.strip().lower()
            if lowered.startswith("nan"):
                return ""
        return val

    cleaned = df.replace({pd.NA: "", np.nan: ""})
    mapper = getattr(cleaned, "map", None)
    if callable(mapper):
        return mapper(_strip_nan)
    return cleaned.applymap(_strip_nan)


def _format_ratio_wide(df: pd.DataFrame, add_avg: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    value_cols = [c for c in out.columns if c not in ("Model", "Avg")]
    if not value_cols:
        return out
    numeric = out[value_cols].apply(pd.to_numeric, errors="coerce")
    if add_avg:
        avg_series = numeric.mean(axis=1, skipna=True).round(4)
        if "Avg" in out.columns:
            out["Avg"] = avg_series
        else:
            insert_loc = 1 if "Model" in out.columns else 0
            out.insert(insert_loc, "Avg", avg_series)
    for col in [c for c in out.columns if c != "Model"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").apply(
            lambda v: f"{v * 100:.2f}%" if pd.notna(v) else ""
        )
    month_cols = [c for c in out.columns if c not in ["Model", "Avg"]]
    ordered_months = _sort_month_year_columns(month_cols)
    ordered_cols = ["Model"]
    if "Avg" in out.columns:
        ordered_cols.append("Avg")
    ordered_cols.extend(ordered_months)
    ordered_cols.extend([c for c in out.columns if c not in ordered_cols])
    out = out[ordered_cols]
    return out


def _sort_month_year_columns(cols: list[str]) -> list[str]:
    parsed: list[tuple[Optional[pd.Timestamp], int, str]] = []
    for idx, col in enumerate(cols):
        dt = pd.to_datetime(col, format="%b-%y", errors="coerce")
        parsed.append((dt, idx, col))
    parsed.sort(key=lambda item: (pd.isna(item[0]), item[0] if pd.notna(item[0]) else pd.Timestamp.max, item[1]))
    return [col for _, _, col in parsed]


def _sort_year_month(df: pd.DataFrame, year_col: str = "Year", month_col: str = "Month") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if year_col not in df.columns or month_col not in df.columns:
        return df
    out = df.copy()
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    out[month_col] = out[month_col].astype(str).str.strip()
    out["_month_order"] = pd.Categorical(out[month_col], categories=month_order, ordered=True)
    out = out.sort_values([year_col, "_month_order"]).drop(columns=["_month_order"])
    return out


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _clean_contact_ratio_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    cols = ["Year"] + [m for m in months if m in df.columns]
    if "Year" not in df.columns:
        return pd.DataFrame()
    out = df[cols].copy()
    for col in out.columns:
        if col != "Year":
            out[col] = clean_and_convert_percentage(out[col])
    return out


def _month_name_to_num(name: str) -> Optional[int]:
    month_map = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    if not name:
        return None
    return month_map.get(str(name).strip().lower()[:3])


def _table_to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    if df is None or df.empty or "Year" not in df.columns:
        return pd.DataFrame()
    month_cols = [c for c in df.columns if c != "Year"]
    if not month_cols:
        return pd.DataFrame()
    long_df = df.melt(id_vars=["Year"], value_vars=month_cols, var_name="Month", value_name=value_name)
    long_df["Month_num"] = long_df["Month"].apply(_month_name_to_num)
    long_df = long_df.dropna(subset=["Month_num"])
    long_df["ds"] = pd.to_datetime(
        long_df["Year"].astype(str) + "-" + long_df["Month_num"].astype(int).astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )
    return long_df.dropna(subset=["ds"])


def _iq_table_to_long(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "Year" not in df.columns:
        return pd.DataFrame()
    month_cols = [c for c in df.columns if c != "Year"]
    clean_df = df.copy()
    for col in month_cols:
        clean_df[col] = clean_and_convert_millions(clean_df[col])
    long_df = clean_df.melt(id_vars=["Year"], value_vars=month_cols, var_name="Month", value_name="IQ_value")
    long_df["Month_num"] = long_df["Month"].apply(_month_name_to_num)
    long_df = long_df.dropna(subset=["Month_num"])
    long_df["ds"] = pd.to_datetime(
        long_df["Year"].astype(str) + "-" + long_df["Month_num"].astype(int).astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )
    return long_df.dropna(subset=["ds"])


def _prophet_param_grid(include_holidays: bool) -> list[dict]:
    cps = [0.003, 0.01, 0.03, 0.1, 0.3]
    crange = [0.8, 0.9, 0.95]
    sps = [0.1, 0.3, 1, 3, 10]
    smode = ["additive", "multiplicative"]
    n_chg = [3, 5, 8, 10]
    y_fourier = [3, 4, 5, 6, 8]
    hps = [0.1, 0.3, 1, 3] if include_holidays else [None]

    combos = []
    for cp in cps:
        for cr in crange:
            for sp in sps:
                for mode in smode:
                    for ncp in n_chg:
                        for yf in y_fourier:
                            for hp in hps:
                                combos.append(
                                    {
                                        "changepoint_prior_scale": cp,
                                        "changepoint_range": cr,
                                        "seasonality_prior_scale": sp,
                                        "seasonality_mode": mode,
                                        "n_changepoints": ncp,
                                        "yearly_fourier_order": yf,
                                        "holidays_prior_scale": hp,
                                    }
                                )
    max_candidates = 60
    if len(combos) > max_candidates:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(combos), size=max_candidates, replace=False)
        combos = [combos[i] for i in idx]
    return combos


def _prophet_cv_splits(n: int) -> list[tuple[int, int]]:
    splits = []
    for train_len in [15, 18, 21]:
        if n >= train_len + 3:
            splits.append((train_len, 3))
    if not splits and n >= 6:
        splits.append((n - 3, 3))
    return splits


def _residual_anomaly_rate(residuals: np.ndarray, z_thresh: float = 3.5) -> float:
    if residuals.size == 0:
        return 0.0
    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median))
    if mad == 0:
        std = np.std(residuals)
        if std == 0:
            return 0.0
        zscores = np.abs((residuals - np.mean(residuals)) / std)
    else:
        zscores = 0.6745 * np.abs(residuals - median) / mad
    return float(np.mean(zscores > z_thresh))


def _prophet_score_candidate(
    df: pd.DataFrame,
    params: dict,
    splits: list[tuple[int, int]],
    regressors: list[str],
    holiday_df: Optional[pd.DataFrame],
) -> float:
    scores = []
    anomaly_threshold = 0.5
    for train_len, horizon in splits:
        train = df.iloc[:train_len].copy()
        val = df.iloc[train_len : train_len + horizon].copy()
        if train.empty or val.empty:
            continue

        m = Prophet(
            changepoint_prior_scale=params["changepoint_prior_scale"],
            changepoint_range=params["changepoint_range"],
            seasonality_prior_scale=params["seasonality_prior_scale"],
            seasonality_mode=params["seasonality_mode"],
            n_changepoints=params["n_changepoints"],
            yearly_seasonality=int(params["yearly_fourier_order"]),
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=holiday_df if params.get("holidays_prior_scale") is not None else None,
            holidays_prior_scale=params.get("holidays_prior_scale") or 0.1,
        )
        reg_cols = []
        for reg in regressors:
            if reg in train.columns:
                m.add_regressor(reg)
                reg_cols.append(reg)
        fit_cols = ["ds", "y"] + reg_cols
        m.fit(train[fit_cols])

        pred = m.predict(val[["ds"] + reg_cols])
        yhat = pred["yhat"].values
        y = val["y"].values
        residuals = y - yhat
        anomaly_rate = _residual_anomaly_rate(residuals)
        if anomaly_rate > anomaly_threshold:
            return float("inf")
        denom = np.sum(np.abs(y)) or 1e-9
        wape = np.sum(np.abs(y - yhat)) / denom
        bias_pct = np.sum(yhat - y) / denom
        scores.append(wape + 0.5 * abs(bias_pct))
    if not scores:
        return float("inf")
    return float(np.mean(scores))


def _prophet_cv_best(
    df: pd.DataFrame,
    holiday_df: Optional[pd.DataFrame],
    regressors: list[str],
) -> tuple[dict, float]:
    candidates = _prophet_param_grid(include_holidays=holiday_df is not None)
    splits = _prophet_cv_splits(len(df))
    best_score = float("inf")
    best_params = candidates[0] if candidates else {}
    for params in candidates:
        score = _prophet_score_candidate(df, params, splits, regressors, holiday_df)
        if score < best_score:
            best_score = score
            best_params = params
    return best_params, best_score


def _prophet_fit_full(
    df: pd.DataFrame,
    params: dict,
    regressors: list[str],
    holiday_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    model = Prophet(
        changepoint_prior_scale=params["changepoint_prior_scale"],
        changepoint_range=params["changepoint_range"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        seasonality_mode=params["seasonality_mode"],
        n_changepoints=params["n_changepoints"],
        yearly_seasonality=int(params["yearly_fourier_order"]),
        weekly_seasonality=False,
        daily_seasonality=False,
        holidays=holiday_df if params.get("holidays_prior_scale") is not None else None,
        holidays_prior_scale=params.get("holidays_prior_scale") or 0.1,
    )
    reg_cols = []
    for reg in regressors:
        if reg in df.columns:
            model.add_regressor(reg)
            reg_cols.append(reg)
    fit_cols = ["ds", "y"] + reg_cols
    model.fit(df[fit_cols])
    pred = model.predict(df[["ds"] + reg_cols])
    return pred
def _apply_caps(df: pd.DataFrame, lower: Optional[float], upper: Optional[float]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    capped = df.copy()
    month_cols = [c for c in capped.columns if c not in ["Year", "Avg"]]
    for col in month_cols:
        capped[col] = pd.to_numeric(capped[col], errors="coerce")
    low = _safe_float(lower, None)
    high = _safe_float(upper, None)
    if low is not None:
        capped[month_cols] = capped[month_cols].clip(lower=low)
    if high is not None:
        capped[month_cols] = capped[month_cols].clip(upper=high)
    return capped


def _recalculate_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    month_cols = [c for c in out.columns if c not in ["Year", "Avg"]]
    for col in month_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    row_means = out[month_cols].mean(axis=1)
    row_means = row_means.replace(0, np.nan)
    out[month_cols] = out[month_cols].div(row_means, axis=0).round(2)
    out[month_cols] = out[month_cols].fillna(0)
    out["Avg"] = out[month_cols].mean(axis=1).round(2)
    return out


def _normalized_ratio_table(df: pd.DataFrame, base_volume: Optional[float]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    base = _safe_float(base_volume, 0.0)
    out = df.copy()
    month_cols = [c for c in out.columns if c not in ["Year", "Avg"]]
    out[month_cols] = (out[month_cols].astype(float) * base).round(2)
    out["Avg"] = out[month_cols].mean(axis=1).round(2)
    return out


def _normalize_phase_store(phase_store: Any) -> dict:
    if not phase_store:
        return {}
    if isinstance(phase_store, dict):
        return phase_store
    if isinstance(phase_store, str):
        try:
            return json.loads(phase_store)
        except Exception:
            return {}
    try:
        return dict(phase_store)
    except Exception:
        return {}


def _phase1_compact_from_results(res: dict) -> dict:
    """Build a compact, navigation-friendly payload for Phase 2."""
    smoothed = pd.DataFrame(res.get("smoothed", []))
    if smoothed.empty:
        return {"smoothed": []}
    compact = smoothed.copy()
    # Normalize ds and final smoothed columns
    if "ds" in compact.columns:
        compact["ds"] = pd.to_datetime(compact["ds"]).dt.strftime("%Y-%m-%d")
    if "Final_Smoothed_Value" not in compact.columns:
        if "smoothed" in compact.columns:
            compact["Final_Smoothed_Value"] = compact["smoothed"]
        elif "y" in compact.columns:
            compact["Final_Smoothed_Value"] = compact["y"]
    keep_cols = [c for c in ("ds", "Final_Smoothed_Value", "IQ_value") if c in compact.columns]
    compact = compact[keep_cols]
    return {"smoothed": compact.to_dict("records")}


@app.callback(Output("global-loading", "data", allow_duplicate=True), Input("vs-upload", "contents"), prevent_initial_call=True)
def _vs_upload_show_loader(_contents):
    logger.info("vs-upload: show loader")
    return True


@app.callback(Output("global-loading", "data", allow_duplicate=True), Input("vs-upload-msg", "children"), prevent_initial_call=True)
def _vs_upload_hide_loader(_msg):
    logger.info("vs-upload: hide loader")
    return False


@app.callback(Output("global-loading", "data", allow_duplicate=True), Input("vs-run-btn", "n_clicks"), prevent_initial_call=True)
def _vs_show_loader(_n):
    ctx = dash.callback_context
    triggered_val = ctx.triggered[0].get("value") if ctx.triggered else None
    if not triggered_val:
        logger.info("vs-run: show loader skipped (no click)")
        raise dash.exceptions.PreventUpdate
    logger.info("vs-run: show loader n_clicks=%s", _n)
    return True


@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("vs-alert", "children", allow_optional=True),
    Input("vs-results-store", "data"),
    prevent_initial_call=True,
)
def _vs_hide_loader(_msg, _results):
    logger.info("vs-run: hide loader")
    return False


@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("vs-run-prophet", "n_clicks"),
    prevent_initial_call=True,
)
def _vs_prophet_show_loader(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    logger.info("vs-prophet: show loader n_clicks=%s", n_clicks)
    return True


@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("vs-prophet-status", "children", allow_optional=True),
    Input("vs-prophet-store", "data"),
    prevent_initial_call=True,
)
def _vs_prophet_hide_loader(_status, _store):
    logger.info("vs-prophet: hide loader")
    return False


@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("vs-apply-seasonality", "n_clicks"),
    prevent_initial_call=True,
)
def _vs_apply_seasonality_show_loader(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    logger.info("vs-seasonality: show loader n_clicks=%s", n_clicks)
    return True


@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("vs-seasonality-status", "children", allow_optional=True),
    Input("vs-seasonality-store", "data"),
    prevent_initial_call=True,
)
def _vs_apply_seasonality_hide_loader(_status, _store):
    logger.info("vs-seasonality: hide loader")
    return False


@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("vs-save-prophet", "n_clicks"),
    prevent_initial_call=True,
)
def _vs_save_prophet_show_loader(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    logger.info("vs-prophet-save: show loader n_clicks=%s", n_clicks)
    return True


@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("vs-prophet-save-status", "children", allow_optional=True),
    Input("vs-prophet-store", "data"),
    prevent_initial_call=True,
)
def _vs_save_prophet_hide_loader(_status, _store):
    logger.info("vs-prophet-save: hide loader")
    return False


@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("vs-run-phase1", "n_clicks"),
    prevent_initial_call=True,
)
def _vs_phase1_show_loader(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    logger.info("vs-phase1: show loader n_clicks=%s", n_clicks)
    return True


@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("vs-phase1-status", "children"),
    Input("vs-phase1-results", "data"),
    prevent_initial_call=True,
)
def _vs_phase1_hide_loader(_status, _results):
    logger.info("vs-phase1: hide loader")
    return False


@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("vs-run-phase2", "n_clicks"),
    prevent_initial_call=True,
)
def _vs_phase2_show_loader(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    logger.info("vs-phase2: show loader n_clicks=%s", n_clicks)
    return True


@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("vs-phase2-status", "children", allow_optional=True),
    Input("vs-phase2-store", "data"),
    prevent_initial_call=True,
)
def _vs_phase2_hide_loader(_status, _store):
    logger.info("vs-phase2: hide loader")
    return False


@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("sa-run-smoothing", "n_clicks"),
    Input("sa-run-prophet", "n_clicks"),
    prevent_initial_call=True,
)
def _sa_show_loader(*_):
    return True


@app.callback(Output("global-loading", "data", allow_duplicate=True), Input("sa-alert", "children"), prevent_initial_call=True)
def _sa_hide_loader(_msg):
    return False


@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("fc-run-btn", "n_clicks"),
    Input("fc-load-phase1", "n_clicks"),
    Input("fc-load-saved-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _fc_show_loader(*_):
    return True


@app.callback(Output("global-loading", "data", allow_duplicate=True), Input("fc-alert", "children"), prevent_initial_call=True)
def _fc_hide_loader(_msg):
    return False


@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("tp-apply-selection", "n_clicks"),
    Input("tp-apply-transform", "n_clicks"),
    prevent_initial_call=True,
)
def _tp_show_loader(*_):
    return True


@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("tp-selection-status", "children"),
    Input("tp-transform-status", "children"),
    prevent_initial_call=True,
)
def _tp_hide_loader(*_):
    return False


@app.callback(Output("global-loading", "data", allow_duplicate=True), Input("di-run-btn", "n_clicks"), prevent_initial_call=True)
def _di_show_loader(_n):
    return True


@app.callback(Output("global-loading", "data", allow_duplicate=True), Input("di-run-status", "children"), prevent_initial_call=True)
def _di_hide_loader(_msg):
    return False


def _fallback_pivots(df: pd.DataFrame, cat: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lightweight, defensive pivot builders used when the primary helper fails."""
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    d = df.copy()
    if "category" not in d.columns:
        return pd.DataFrame(), pd.DataFrame()
    d = d[d["category"] == cat]
    if d.empty or "date" not in d.columns or "volume" not in d.columns:
        return pd.DataFrame(), pd.DataFrame()

    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce")
    d = d.dropna(subset=["date", "volume"])
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()

    d["Year"] = d["date"].dt.year
    d["Month_Num"] = d["date"].dt.month
    d["Month"] = d["date"].dt.strftime("%b")
    if "forecast_group" not in d.columns:
        d["forecast_group"] = d["category"]

    months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot_raw = (
        d.pivot_table(index=["Year", "forecast_group"], columns="Month", values="volume", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    month_cols = [m for m in months_order if m in pivot_raw.columns]

    # Display pivot (k formatting + growth)
    display = pivot_raw.copy()
    if month_cols:
        display["Yearly_Avg"] = display[month_cols].mean(axis=1)
    else:
        display["Yearly_Avg"] = 0
    display["Growth_%"] = None
    for fg in display["forecast_group"].unique():
        idx = display["forecast_group"] == fg
        growth = display[idx].sort_values("Year")["Yearly_Avg"].pct_change().values
        display.loc[idx, "Growth_%"] = growth
    for col in month_cols + ["Yearly_Avg"]:
        display[col] = display[col].apply(lambda x: f"{float(x) / 1_000:,.1f}k" if pd.notna(x) else "")
    display["Growth_%"] = display["Growth_%"].apply(
        lambda x: f"{float(x) * 100:.1f}%" if pd.notna(x) else ""
    )
    display = display[["Year", "forecast_group"] + month_cols + ["Yearly_Avg", "Growth_%"]]

    # Volume split (% per year+fg across its months)
    split = pivot_raw.copy()
    row_totals = split[month_cols].sum(axis=1) if month_cols else pd.Series(dtype=float)
    for col in month_cols:
        split[col] = split[col].where(row_totals == 0, split[col] / row_totals * 100).round(2)

    def _row_avg(row):
        vals = [row[m] for m in month_cols if pd.notna(row[m]) and row[m] > 0]
        return round(sum(vals) / len(vals), 1) if vals else pd.NA

    split["Avg"] = split.apply(_row_avg, axis=1) if month_cols else pd.NA

    def _last3(row):
        vals = [row[m] for m in month_cols if pd.notna(row[m]) and row[m] > 0]
        if len(vals) >= 3:
            return round(sum(vals[-3:]) / 3, 1)
        return round(sum(vals) / len(vals), 1) if vals else pd.NA

    split["Vol_Split_Last_3M"] = split.apply(_last3, axis=1) if month_cols else pd.NA
    for col in month_cols + ["Avg", "Vol_Split_Last_3M"]:
        split[col] = split[col].apply(lambda x: f"{float(x):.2f}%" if pd.notna(x) else "")
    split = split[["Year", "forecast_group"] + month_cols + ["Avg", "Vol_Split_Last_3M"]]
    return display, split


def _smoothing_core(
    df: pd.DataFrame,
    window: int,
    threshold: float,
    prophet_order: Optional[int] = None,
    build_figs: bool = True,
):
    """Run EWMA smoothing (or Prophet) + anomaly detection and seasonality pivots."""
    date_col = _pick_col(df, ("date", "ds", "timestamp"))
    val_col = _pick_col(df, ("final_smoothed_value", "volume", "value", "items", "calls", "count", "y"))
    if not date_col or not val_col:
        raise ValueError("Expected columns for date and volume.")

    d = df.copy()
    d["ds"] = pd.to_datetime(d[date_col], errors="coerce")
    d["y"] = pd.to_numeric(d[val_col], errors="coerce")
    d = d.dropna(subset=["ds", "y"]).sort_values("ds")

    if prophet_order:
        from prophet import Prophet

        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.add_seasonality(name="monthly", period=30.5, fourier_order=int(prophet_order or 5))
        m.fit(d[["ds", "y"]])
        preds = m.predict(d[["ds"]])
        d["smoothed"] = preds["yhat"]
    else:
        span = max(int(window or 6), 1)
        d["smoothed"] = d["y"].ewm(span=span, adjust=False).mean()

    resid = d["y"] - d["smoothed"]
    std = resid.std() or 1e-9
    d["zscore"] = (resid - resid.mean()) / std
    d["is_anomaly"] = d["zscore"].abs() > float(threshold or 3.0)

    d["Year"] = d["ds"].dt.year
    d["Month"] = d["ds"].dt.strftime("%b")
    pivot = d.pivot_table(index="Year", columns="Month", values="smoothed", aggfunc="mean").reset_index()
    pivot = pivot.fillna(0)

    ratio_fig, capped, ratio = None, pd.DataFrame(), pd.DataFrame()
    try:
        ratio_fig, capped, ratio = plot_contact_ratio_seasonality(pivot)
    except Exception:
        ratio_fig = _empty_fig("Seasonality not available")

    ratio_disp = ratio.copy()
    capped_disp = capped.copy()
    for col in ratio_disp.columns:
        if col != "Year":
            ratio_disp[col] = pd.to_numeric(ratio_disp[col], errors="coerce").round(2)
    for col in capped_disp.columns:
        if col != "Year":
            capped_disp[col] = pd.to_numeric(capped_disp[col], errors="coerce").round(2)

    anomalies = d[d["is_anomaly"]][["ds", "y", "smoothed", "zscore"]]
    smoothed_tbl = d[["ds", "y", "smoothed", "zscore", "is_anomaly"]]

    if build_figs:
        fig_series = px.line(
            d,
            x="ds",
            y=["y", "smoothed"],
            labels={"value": "Volume", "ds": "Date", "variable": "Series"},
            title="Smoothed vs Original",
        )
        if not anomalies.empty:
            fig_series.add_scatter(
                x=anomalies["ds"],
                y=anomalies["y"],
                mode="markers",
                name="Anomaly",
                marker=dict(color="red", size=10, symbol="x"),
            )
    else:
        fig_series = _empty_fig()

    return {
        "fig_series": fig_series,
        "fig_ratio1": _ratio_fig(ratio_disp, "Normalized Ratio 1") if build_figs else _empty_fig(),
        "fig_ratio2": _ratio_fig(capped_disp, "Normalized Ratio 2 (capped)") if build_figs else _empty_fig(),
        "ratio": ratio_disp,
        "capped": capped_disp,
        "smoothed": smoothed_tbl,
        "anomalies": anomalies,
        "pivot": pivot,
    }


def _score_smoothing_candidate(smoothed_tbl: pd.DataFrame) -> tuple[float, dict]:
    """Score a smoothing run to pick a balanced span/threshold combination."""
    if smoothed_tbl is None or smoothed_tbl.empty:
        return float("inf"), {}
    y = pd.to_numeric(smoothed_tbl.get("y"), errors="coerce")
    s = pd.to_numeric(smoothed_tbl.get("smoothed"), errors="coerce")
    mask = y.notna() & s.notna()
    if mask.sum() < 3:
        return float("inf"), {}

    y = y[mask]
    s = s[mask]
    resid = y - s
    rmse = float(np.sqrt(np.mean(resid**2)))
    y_std = float(y.std()) or 1e-9
    rmse_norm = rmse / y_std

    y_diff = y.diff().abs().mean()
    s_diff = s.diff().abs().mean()
    diff_base = float(y_diff) if pd.notna(y_diff) else 0.0
    smooth_ratio = float(s_diff) / (diff_base + 1e-9) if diff_base else 0.0

    anomaly_rate = 0.0
    if "is_anomaly" in smoothed_tbl.columns:
        anomaly_rate = float(smoothed_tbl["is_anomaly"].fillna(False).mean())

    target_smooth = 0.6
    target_anomaly = 0.03
    smooth_penalty = abs(smooth_ratio - target_smooth)
    anomaly_penalty = abs(anomaly_rate - target_anomaly)

    flat_penalty = 0.0
    if diff_base and smooth_ratio < 0.1:
        flat_penalty = 0.3
    if anomaly_rate > 0.2:
        anomaly_penalty += 0.3

    score = rmse_norm + 0.8 * smooth_penalty + 1.2 * anomaly_penalty + flat_penalty
    metrics = {
        "rmse_norm": rmse_norm,
        "smooth_ratio": smooth_ratio,
        "anomaly_rate": anomaly_rate,
    }
    return score, metrics


def _format_smoothing_payload(res: dict, source_tag: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    smoothed_tbl = res["smoothed"].copy()
    anomalies_tbl = res["anomalies"].copy()
    for tbl in (smoothed_tbl, anomalies_tbl):
        if "ds" in tbl.columns:
            tbl["ds"] = pd.to_datetime(tbl["ds"], errors="coerce").dt.strftime("%Y-%m-%d")

    payload = {
        "smoothed": smoothed_tbl.to_dict("records"),
        "anomalies": anomalies_tbl.to_dict("records"),
        "ratio": res["ratio"].to_dict("records"),
        "capped": res["capped"].to_dict("records"),
        "pivot": res["pivot"].to_dict("records"),
        "source": source_tag,
    }
    return smoothed_tbl, anomalies_tbl, payload


def _auto_smoothing_sweep(
    df: pd.DataFrame,
    windows: Optional[list[int]] = None,
    thresholds: Optional[list[float]] = None,
) -> tuple[Optional[dict], dict]:
    """Try multiple span/threshold permutations and pick the best-scoring run."""
    if df is None or df.empty:
        return None, {"error": "no-data"}

    default_windows = [3, 4, 5, 6, 8, 10, 12]
    default_thresholds = [2.0, 2.5, 3.0, 3.5]
    windows = windows or default_windows
    thresholds = thresholds or default_thresholds

    data_len = len(df)
    max_win = max(2, min(12, data_len))
    windows = sorted({int(w) for w in windows if int(w) > 1 and int(w) <= max_win})
    if not windows:
        windows = [max(2, min(6, data_len))]
    thresholds = sorted({float(t) for t in thresholds if float(t) > 0})
    if not thresholds:
        thresholds = [2.5]

    best = None
    candidates: list[dict] = []
    for window in windows:
        for threshold in thresholds:
            try:
                res = _smoothing_core(df, window, threshold, None, build_figs=False)
            except Exception:
                continue
            score, metrics = _score_smoothing_candidate(res.get("smoothed"))
            if not np.isfinite(score):
                continue
            meta = {
                "method": "ewma",
                "window": int(window),
                "threshold": float(threshold),
                "score": float(score),
                "metrics": metrics,
            }
            candidates.append(meta)
            if best is None or score < best["meta"]["score"]:
                best = {"res": res, "meta": meta}

    if best is None:
        return None, {"tested": len(candidates)}

    candidates_sorted = sorted(candidates, key=lambda m: m["score"])
    top_candidates = candidates_sorted[:5]
    smoothed_tbl, anomalies_tbl, payload = _format_smoothing_payload(best["res"], "auto-ewma")
    auto_meta = best["meta"].copy()
    auto_meta["tested"] = len(candidates)
    auto_meta["candidates"] = top_candidates
    auto_meta["ts"] = pd.Timestamp.utcnow().isoformat()
    payload["auto_meta"] = auto_meta
    return payload, auto_meta


def _render_smoothing_payload(payload: dict):
    smoothed_tbl = pd.DataFrame(payload.get("smoothed", []))
    anomalies_tbl = pd.DataFrame(payload.get("anomalies", []))
    ratio_df = pd.DataFrame(payload.get("ratio", []))
    capped_df = pd.DataFrame(payload.get("capped", []))

    fig_series = _empty_fig("No data")
    if not smoothed_tbl.empty and {"ds", "y", "smoothed"}.issubset(smoothed_tbl.columns):
        plot_tbl = smoothed_tbl.copy()
        plot_tbl["ds"] = pd.to_datetime(plot_tbl["ds"], errors="coerce")
        fig_series = px.line(
            plot_tbl,
            x="ds",
            y=["y", "smoothed"],
            labels={"value": "Volume", "ds": "Date", "variable": "Series"},
            title="Smoothed vs Original",
        )
        if not anomalies_tbl.empty and {"ds", "y"}.issubset(anomalies_tbl.columns):
            a = anomalies_tbl.copy()
            a["ds"] = pd.to_datetime(a["ds"], errors="coerce")
            fig_series.add_scatter(
                x=a["ds"],
                y=pd.to_numeric(a["y"], errors="coerce"),
                mode="markers",
                name="Anomaly",
                marker=dict(color="red", size=10, symbol="x"),
            )

    fig_ratio1 = _ratio_fig(ratio_df, "Normalized Ratio 1")
    fig_ratio2 = _ratio_fig(capped_df, "Normalized Ratio 2 (capped)")
    return fig_series, fig_ratio1, fig_ratio2, smoothed_tbl, anomalies_tbl, ratio_df, capped_df


_TRANSFORMATION_COLS = [
    "Transformation 1",
    "Remarks_Tr 1",
    "Transformation 2",
    "Remarks_Tr 2",
    "Transformation 3",
    "Remarks_Tr 3",
    "IA 1",
    "Remarks_IA 1",
    "IA 2",
    "Remarks_IA 2",
    "IA 3",
    "Remarks_IA 3",
    "Marketing Campaign 1",
    "Remarks_Mkt 1",
    "Marketing Campaign 2",
    "Remarks_Mkt 2",
    "Marketing Campaign 3",
    "Remarks_Mkt 3",
]


def _apply_transformations(df: pd.DataFrame) -> pd.DataFrame:
    def calculate_sequential(row: pd.Series, field: str, base_col: str):
        try:
            value = row.get(field, "")
            if pd.notna(value) and str(value).strip() != "":
                adj_percent = float(str(value).replace("%", "").strip())
                return round(row[base_col] * (1 + adj_percent / 100), 0)
        except Exception:
            pass
        return round(row[base_col], 0)

    def apply_sequential_adjustments(df_in: pd.DataFrame) -> pd.DataFrame:
        df_copy = df_in.copy()
        adjustment_fields = [
            "Transformation 1",
            "Transformation 2",
            "Transformation 3",
            "IA 1",
            "IA 2",
            "IA 3",
            "Marketing Campaign 1",
            "Marketing Campaign 2",
            "Marketing Campaign 3",
        ]
        prev_col = "Base_Forecast_for_Forecast_Group"
        for field in adjustment_fields:
            new_col = f"Forecast_{field}"
            df_copy[new_col] = df_copy.apply(lambda row: calculate_sequential(row, field, prev_col), axis=1)
            prev_col = new_col
        return df_copy

    processed = apply_sequential_adjustments(df)
    if "Forecast_Marketing Campaign 3" in processed.columns:
        processed["Final_Forecast_Post_Transformations"] = processed["Forecast_Marketing Campaign 3"]
        processed["Final_Forecast"] = processed["Forecast_Marketing Campaign 3"]
    return processed


@app.callback(
    Output("vs-upload-msg", "children"),
    Output("vs-preview", "data"),
    Output("vs-preview", "columns"),
    Output("vs-data-store", "data", allow_duplicate=True),
    Output("vs-iq-store", "data", allow_duplicate=True),
    Output("vs-iq-summary-store", "data", allow_duplicate=True),
    Output("vs-holiday-store", "data", allow_duplicate=True),
    Input("vs-upload", "contents"),
    State("vs-upload", "filename"),
    prevent_initial_call=True,
)
def _on_vs_upload(contents, filename):
    if not contents or not filename:
        return "No file supplied.", [], [], None, None, None, None

    try:
        logger.info("vs-upload: start filename=%s", filename)
        decoded_bytes = None
        try:
            _, content_string = contents.split(",", 1)
            decoded_bytes = base64.b64decode(content_string)
        except Exception:
            decoded_bytes = None

        df, msg = _parse_upload(contents, filename, decoded_bytes=decoded_bytes)
        monthly_df, agg_msg = _aggregate_monthly(df)
        pivot_preview = _category_month_pivot(df)
        if not pivot_preview.empty:
            preview = pivot_preview
        elif isinstance(monthly_df, pd.DataFrame) and not monthly_df.empty:
            preview = monthly_df.head(50)
        else:
            preview = df.head(50)
        cols = _cols(preview)
        store = df.to_json(date_format="iso", orient="split") if not df.empty else None
        iq_store = None
        iq_summary_store = None
        holiday_store = None
        if decoded_bytes is not None and filename.lower().endswith((".xlsx", ".xls", ".xlsm")):
            try:
                xl = pd.ExcelFile(io.BytesIO(decoded_bytes))
                sheet_map = {
                    re.sub(r"[^a-z0-9]", "", str(name).lower()): name
                    for name in xl.sheet_names
                }
                iq_sheet = sheet_map.get("iqdata") or next(
                    (s for s in xl.sheet_names if "iq" in str(s).lower()),
                    None,
                )
                if iq_sheet:
                    iq_df = xl.parse(iq_sheet)
                    if not iq_df.empty:
                        iq_store = iq_df.to_json(date_format="iso", orient="split")
                        msg = f"{msg} | IQ sheet '{iq_sheet}' loaded."
            except Exception:
                pass
            try:
                iq_results = IQ_data(io.BytesIO(decoded_bytes))
                iq_summary_store = _serialize_iq_results(iq_results)
                if iq_results:
                    msg = f"{msg} | IQ summary ready ({len(iq_results)} categories)."
            except Exception:
                iq_summary_store = None
            try:
                holiday_sheet = next(
                    (s for s in xl.sheet_names if "holiday" in str(s).lower()),
                    None,
                )
                if holiday_sheet:
                    df_holidays = xl.parse(holiday_sheet)
                    df_holidays.columns = [
                        str(col).strip().lower().replace(" ", "_")
                        for col in df_holidays.columns
                    ]
                    date_col = _pick_col(df_holidays, ("date", "holiday_date"))
                    name_col = _pick_col(df_holidays, ("holidays", "holiday", "name"))
                    if date_col and name_col:
                        df_holidays[date_col] = pd.to_datetime(df_holidays[date_col], errors="coerce")
                        df_holidays = df_holidays.dropna(subset=[date_col, name_col])
                        holiday_store = {
                            "mapping": {
                                str(k): v for k, v in zip(df_holidays[date_col], df_holidays[name_col])
                            }
                        }
            except Exception:
                holiday_store = None
        pivot_msg = "Preview shows Category by Month view." if not pivot_preview.empty else ""
        combined_msg = " ".join(part for part in [msg, agg_msg, pivot_msg] if part).strip()
        logger.info("vs-upload: complete rows=%s cols=%s", len(df.index), len(df.columns))
        return combined_msg, preview.to_dict("records"), cols, store, iq_store, iq_summary_store, json.dumps(holiday_store) if holiday_store else None
    except Exception as exc:
        logger.exception("vs-upload: failed")
        return f"Upload failed: {exc}", [], [], None, None, None, None


@app.callback(
    Output("vs-summary", "data"),
    Output("vs-summary", "columns"),
    Output("vs-alert", "children"),
    Output("vs-alert", "is_open"),
    Output("vs-category-tabs", "children"),
    Output("vs-category-tabs", "value"),
    Output("vs-pivot", "data"),
    Output("vs-pivot", "columns"),
    Output("vs-volume-split", "data"),
    Output("vs-volume-split", "columns"),
    Output("vs-iq-summary", "data"),
    Output("vs-iq-summary", "columns"),
    Output("vs-volume-summary", "data"),
    Output("vs-volume-summary", "columns"),
    Output("vs-contact-summary", "data"),
    Output("vs-contact-summary", "columns"),
    Output("vs-results-store", "data", allow_duplicate=True),
    Output("forecast-phase-store", "data", allow_duplicate=True),
    Input("vs-run-btn", "n_clicks"),
    State("vs-data-store", "data"),
    State("vs-iq-summary-store", "data"),
    State("forecast-phase-store", "data"),
    prevent_initial_call=True,
)
def _run_volume_summary(n_clicks, data_json, iq_summary_store, phase_store):
    logger.info("vs-run: start n_clicks=%s data_json=%s", n_clicks, bool(data_json))
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not data_json:
        logger.info("vs-run: missing data_json, returning early")
        return (
            [],
            [],
            "Upload data to run the summary.",
            True,
            [],
            None,
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            None,
            phase_store,
        )
    try:
        df = pd.read_json(io.StringIO(data_json), orient="split")
        logger.info("vs-run: loaded df rows=%s cols=%s", len(df.index), len(df.columns))

        summary = _clean_table(_summarize(df))
        cols = [{"name": c, "id": c} for c in summary.columns]
        alert_text = "Summary complete. Review results below."

        # Normalize column names for downstream helpers
        df_norm = _normalize_volume_df(df)
        logger.info("vs-run: normalized rows=%s cols=%s", len(df_norm.index), len(df_norm.columns))
        if df_norm.empty:
            alert_text = "Summary complete. Could not build forecast group view (missing valid date/volume rows)."
        categories = sorted(df_norm["category"].dropna().astype(str).unique().tolist()) if not df_norm.empty else []
        iq_categories = []
        if iq_summary_store:
            try:
                iq_payload = json.loads(iq_summary_store) if isinstance(iq_summary_store, str) else iq_summary_store
                iq_categories = sorted(str(k) for k in iq_payload.keys())
            except Exception:
                iq_payload = {}
        else:
            iq_payload = {}
        if not categories and iq_categories:
            categories = iq_categories
        tabs = [dcc.Tab(label=c, value=c) for c in categories]
        chosen = categories[0] if categories else None

        def _safe_pivots(cat: str):
            try:
                piv, split, long_orig, long_monthly, long_daily = forecast_group_pivot_and_long_style(df_norm, cat)
                if piv is None or split is None or piv.empty or split.empty:
                    raise ValueError("Primary pivot empty")
                return piv, split
            except Exception:
                return _fallback_pivots(df_norm, cat)

        piv0, split0 = _safe_pivots(chosen) if chosen else (pd.DataFrame(), pd.DataFrame())
        logger.info(
            "vs-run: pivots ready chosen=%s piv_rows=%s split_rows=%s",
            chosen,
            len(piv0.index) if piv0 is not None else 0,
            len(split0.index) if split0 is not None else 0,
        )
        piv0 = _clean_table(piv0)
        split0 = _clean_table(split0)
        pivot_cols = [{"name": c, "id": c} for c in (piv0.columns if not piv0.empty else [])]
        split_cols = [{"name": c, "id": c} for c in (split0.columns if not split0.empty else [])]
        iq_df, vol_df, ratio_df = _iq_tables_from_store(iq_summary_store, chosen)
        logger.info(
            "vs-run: iq tables rows iq=%s vol=%s ratio=%s",
            len(iq_df.index) if iq_df is not None else 0,
            len(vol_df.index) if vol_df is not None else 0,
            len(ratio_df.index) if ratio_df is not None else 0,
        )
        iq_df = _clean_table(iq_df)
        vol_df = _clean_table(vol_df)
        ratio_df = _clean_table(ratio_df)
        iq_cols = [{"name": c, "id": c} for c in (iq_df.columns if not iq_df.empty else [])]
        vol_cols = [{"name": c, "id": c} for c in (vol_df.columns if not vol_df.empty else [])]
        ratio_cols = [{"name": c, "id": c} for c in (ratio_df.columns if not ratio_df.empty else [])]

        phase_data = _normalize_phase_store(phase_store)
        auto_payload = None
        auto_meta = None
        if not df_norm.empty and {"date", "volume"}.issubset(df_norm.columns):
            logger.info("vs-run: starting auto smoothing")
            auto_payload, auto_meta = _auto_smoothing_sweep(df_norm)
            if auto_payload:
                phase_data["auto_smoothing"] = json.dumps(auto_payload)
                phase_data["phase1"] = json.dumps(_phase1_compact_from_results(auto_payload))
                phase_data["phase1_meta"] = {
                    "source": "auto-volume-summary",
                    "ts": pd.Timestamp.utcnow().isoformat(),
                }
                alert_text = (
                    f"{alert_text} Auto smoothing complete (EWMA span {auto_meta.get('window')},"
                    f" z {auto_meta.get('threshold')})."
                )
            else:
                alert_text = f"{alert_text} Auto smoothing skipped (insufficient data)."
        else:
            logger.info("vs-run: auto smoothing skipped (missing date/volume)")

        results_store = {
            "categories": categories,
            "cat_col": "category",
            "data": df_norm.to_json(date_format="iso", orient="split") if not df_norm.empty else None,
            "auto_meta": auto_meta,
        }

        return (
            summary.to_dict("records"),
            cols,
            alert_text,
            True,
            tabs,
            chosen,
            (piv0.to_dict("records") if not piv0.empty else []),
            pivot_cols,
            (split0.to_dict("records") if not split0.empty else []),
            split_cols,
            (iq_df.to_dict("records") if not iq_df.empty else []),
            iq_cols,
            (vol_df.to_dict("records") if not vol_df.empty else []),
            vol_cols,
            (ratio_df.to_dict("records") if not ratio_df.empty else []),
            ratio_cols,
            json.dumps(results_store),
            phase_data,
        )
    except Exception as exc:
        logger.exception("vs-run: failed")
        return (
            [],
            [],
            f"Volume summary failed: {exc}",
            True,
            [],
            None,
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            None,
            phase_store,
        )


def _toggle_modal(btn_id: str, modal_id: str):
    @app.callback(
        Output(modal_id, "is_open"),
        Input(btn_id, "n_clicks"),
        Input(f"{btn_id}-close", "n_clicks"),
        State(modal_id, "is_open"),
        prevent_initial_call=True,
    )
    def _toggle(n_open, n_close, is_open):
        if not n_open and not n_close:
            raise dash.exceptions.PreventUpdate
        if dash.callback_context.triggered_id == f"{btn_id}-close":
            return False
        return not bool(is_open)


for _btn, _modal in [
    ("sa-complete", "sa-complete-modal"),
    ("fc-complete", "fc-complete-modal"),
    ("tp-complete", "tp-complete-modal"),
    ("di-complete", "di-complete-modal"),
]:
    _toggle_modal(_btn, _modal)


@app.callback(
    Output("vs-category-heading", "children"),
    Input("vs-category-tabs", "value", allow_optional=True),
    prevent_initial_call=True,
)
def _update_category_heading(cat: Optional[str]):
    if not cat:
        return "Select a category to view its tables."
    return f"Category: {cat}"


@app.callback(
    Output("vs-pivot-title", "children"),
    Output("vs-volume-split-title", "children"),
    Input("vs-category-tabs", "value", allow_optional=True),
    prevent_initial_call=True,
)
def _update_category_section_titles(cat: Optional[str]):
    if not cat:
        return "Forecast Group Pivot Analysis", "Volume Split Percentage"
    return f"Forecast Group Pivot Analysis for {cat}", f"Volume Split Percentage for {cat}"


@app.callback(
    Output("vs-pivot", "data", allow_duplicate=True),
    Output("vs-pivot", "columns", allow_duplicate=True),
    Output("vs-volume-split", "data", allow_duplicate=True),
    Output("vs-volume-split", "columns", allow_duplicate=True),
    Output("vs-iq-summary", "data", allow_duplicate=True),
    Output("vs-iq-summary", "columns", allow_duplicate=True),
    Output("vs-volume-summary", "data", allow_duplicate=True),
    Output("vs-volume-summary", "columns", allow_duplicate=True),
    Output("vs-contact-summary", "data", allow_duplicate=True),
    Output("vs-contact-summary", "columns", allow_duplicate=True),
    Input("vs-category-tabs", "value", allow_optional=True),
    State("vs-results-store", "data"),
    State("vs-iq-summary-store", "data"),
    prevent_initial_call=True,
)
def _on_category_change(cat, store_json, iq_summary_store):
    if not cat:
        raise dash.exceptions.PreventUpdate
    piv_rows, piv_cols, split_rows, split_cols = [], [], [], []
    if store_json:
        try:
            payload = json.loads(store_json)
            data_json = payload.get("data")
            cat_col = payload.get("cat_col", "category")
            if data_json:
                df = pd.read_json(io.StringIO(data_json), orient="split")
                df_norm = _normalize_volume_df(df, cat_col)
                if not df_norm.empty:
                    try:
                        piv, split, _, _, _ = forecast_group_pivot_and_long_style(df_norm, cat)
                        if piv is None or split is None or piv.empty or split.empty:
                            raise ValueError("Primary pivot empty")
                    except Exception:
                        piv, split = _fallback_pivots(df_norm, cat)
                    if piv is not None and not piv.empty:
                        piv = _clean_table(piv)
                        piv_rows = piv.to_dict("records")
                        piv_cols = [{"name": c, "id": c} for c in piv.columns]
                    if split is not None and not split.empty:
                        split = _clean_table(split)
                        split_rows = split.to_dict("records")
                        split_cols = [{"name": c, "id": c} for c in split.columns]
        except Exception:
            piv_rows, piv_cols, split_rows, split_cols = [], [], [], []

    iq_df, vol_df, ratio_df = _iq_tables_from_store(iq_summary_store, cat)
    iq_df = _clean_table(iq_df)
    vol_df = _clean_table(vol_df)
    ratio_df = _clean_table(ratio_df)
    iq_rows = iq_df.to_dict("records") if not iq_df.empty else []
    vol_rows = vol_df.to_dict("records") if not vol_df.empty else []
    ratio_rows = ratio_df.to_dict("records") if not ratio_df.empty else []
    iq_cols = [{"name": c, "id": c} for c in iq_df.columns] if not iq_df.empty else []
    vol_cols = [{"name": c, "id": c} for c in vol_df.columns] if not vol_df.empty else []
    ratio_cols = [{"name": c, "id": c} for c in ratio_df.columns] if not ratio_df.empty else []

    return (
        piv_rows,
        piv_cols,
        split_rows,
        split_cols,
        iq_rows,
        iq_cols,
        vol_rows,
        vol_cols,
        ratio_rows,
        ratio_cols,
    )


@app.callback(
    Output("vs-seasonality-table", "data"),
    Output("vs-seasonality-table", "columns"),
    Output("vs-seasonality-chart", "figure"),
    Output("vs-capped-editor", "data"),
    Output("vs-capped-editor", "columns"),
    Output("vs-recalc-table", "data"),
    Output("vs-recalc-table", "columns"),
    Output("vs-capped-chart", "figure"),
    Output("vs-base-volume", "value"),
    Output("vs-normalized-table", "data"),
    Output("vs-normalized-table", "columns"),
    Output("vs-seasonality-status", "children"),
    Output("vs-seasonality-store", "data", allow_duplicate=True),
    Input("vs-category-tabs", "value", allow_optional=True),
    State("vs-iq-summary-store", "data"),
    prevent_initial_call=True,
)
def _build_volume_seasonality(cat, iq_summary_store):
    if not cat or not iq_summary_store:
        raise dash.exceptions.PreventUpdate
    try:
        _, _, ratio_df = _iq_tables_from_store(iq_summary_store, cat)
        if ratio_df is None or ratio_df.empty:
            return [], [], _empty_fig("No ratio data"), [], [], [], [], _empty_fig("No data"), None, [], [], "No ratio data.", None
        cleaned_ratio = _clean_contact_ratio_table(ratio_df)
        if cleaned_ratio.empty:
            return [], [], _empty_fig("No ratio data"), [], [], [], [], _empty_fig("No data"), None, [], [], "No ratio data.", None
        seasonality_fig, capped_df, ratio_seasonality_df = plot_contact_ratio_seasonality(cleaned_ratio, unique_key=str(cat))
        ratio_seasonality_df = ratio_seasonality_df.round(2)
        capped_df = capped_df.round(2)

        _, base_volume = add_editable_base_volume(ratio_df)
        recalc_df = _recalculate_seasonality(capped_df)
        normalized_df = _normalized_ratio_table(recalc_df, base_volume)

        seasonality_store = {
            "ratio": ratio_seasonality_df.to_json(date_format="iso", orient="split"),
            "capped": capped_df.to_json(date_format="iso", orient="split"),
            "recalc": recalc_df.to_json(date_format="iso", orient="split"),
            "normalized": normalized_df.to_json(date_format="iso", orient="split"),
            "base_volume": base_volume,
        }

        return (
            ratio_seasonality_df.to_dict("records"),
            _cols(ratio_seasonality_df),
            seasonality_fig,
            capped_df.to_dict("records"),
            _cols(capped_df),
            recalc_df.to_dict("records"),
            _cols(recalc_df),
            _ratio_fig(recalc_df, "Capped Seasonality Chart"),
            base_volume,
            normalized_df.to_dict("records"),
            _cols(normalized_df),
            "Seasonality loaded.",
            json.dumps(seasonality_store),
        )
    except Exception:
        logger.exception("vs-seasonality: failed")
        return [], [], _empty_fig("Error"), [], [], [], [], _empty_fig("Error"), None, [], [], "Seasonality failed.", None


@app.callback(
    Output("vs-capped-editor", "data", allow_duplicate=True),
    Output("vs-recalc-table", "data", allow_duplicate=True),
    Output("vs-recalc-table", "columns", allow_duplicate=True),
    Output("vs-capped-chart", "figure", allow_duplicate=True),
    Output("vs-normalized-table", "data", allow_duplicate=True),
    Output("vs-normalized-table", "columns", allow_duplicate=True),
    Output("vs-seasonality-status", "children", allow_duplicate=True),
    Output("vs-seasonality-store", "data", allow_duplicate=True),
    Input("vs-apply-seasonality", "n_clicks"),
    State("vs-capped-editor", "data"),
    State("vs-lower-cap", "value"),
    State("vs-upper-cap", "value"),
    State("vs-base-volume", "value"),
    State("vs-seasonality-store", "data"),
    prevent_initial_call=True,
)
def _apply_seasonality_changes(n_clicks, capped_rows, lower_cap, upper_cap, base_volume, seasonality_store):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    capped_df = pd.DataFrame(capped_rows) if capped_rows else pd.DataFrame()
    if capped_df.empty and seasonality_store:
        try:
            payload = json.loads(seasonality_store) if isinstance(seasonality_store, str) else seasonality_store
            cached = payload.get("capped")
            if cached:
                capped_df = pd.read_json(io.StringIO(cached), orient="split")
        except Exception:
            capped_df = pd.DataFrame()

    if capped_df.empty:
        return no_update, no_update, no_update, no_update, no_update, no_update, "No seasonality data to apply.", no_update

    capped_df = _apply_caps(capped_df, lower_cap, upper_cap)
    recalc_df = _recalculate_seasonality(capped_df)

    if base_volume is None and seasonality_store:
        try:
            payload = json.loads(seasonality_store) if isinstance(seasonality_store, str) else seasonality_store
            base_volume = payload.get("base_volume")
        except Exception:
            base_volume = None

    normalized_df = _normalized_ratio_table(recalc_df, base_volume)

    seasonality_store = {
        "capped": capped_df.to_json(date_format="iso", orient="split"),
        "recalc": recalc_df.to_json(date_format="iso", orient="split"),
        "normalized": normalized_df.to_json(date_format="iso", orient="split"),
        "base_volume": _safe_float(base_volume, 0.0),
    }

    return (
        capped_df.to_dict("records"),
        recalc_df.to_dict("records"),
        _cols(recalc_df),
        _ratio_fig(recalc_df, "Capped Seasonality Chart"),
        normalized_df.to_dict("records"),
        _cols(normalized_df),
        "Seasonality updated.",
        json.dumps(seasonality_store),
    )


@app.callback(
    Output("vs-prophet-status", "children"),
    Output("vs-prophet-table", "data"),
    Output("vs-prophet-table", "columns"),
    Output("vs-norm2-table", "data"),
    Output("vs-norm2-table", "columns"),
    Output("vs-norm2-chart", "figure"),
    Output("vs-prophet-line", "figure"),
    Output("vs-prophet-store", "data", allow_duplicate=True),
    Input("vs-run-prophet", "n_clicks"),
    State("vs-seasonality-store", "data"),
    State("vs-iq-summary-store", "data"),
    State("vs-category-tabs", "value"),
    State("vs-holiday-store", "data"),
    prevent_initial_call=True,
)
def _run_prophet_smoothing(n_clicks, seasonality_store, iq_summary_store, cat, holiday_store):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not seasonality_store:
        return "Run seasonality adjustments first.", [], [], [], [], _empty_fig("No data"), _empty_fig(), None
    if not iq_summary_store or not cat:
        return "Select a category with IQ data.", [], [], [], [], _empty_fig("No data"), _empty_fig(), None

    try:
        season_payload = json.loads(seasonality_store) if isinstance(seasonality_store, str) else seasonality_store
        normalized_json = season_payload.get("normalized")
        ratio_json = season_payload.get("ratio")
        if not normalized_json:
            return "Normalized ratio not found.", [], [], [], [], _empty_fig("No data"), _empty_fig(), None
        normalized_df = pd.read_json(io.StringIO(normalized_json), orient="split")
        normalized_long = _table_to_long(normalized_df, "Normalized_Ratio_1")
        if normalized_long.empty:
            return "Normalized ratio is empty.", [], [], [], [], _empty_fig("No data"), _empty_fig(), None
        ratio_long = pd.DataFrame()
        if ratio_json:
            ratio_df = pd.read_json(io.StringIO(ratio_json), orient="split")
            ratio_long = _table_to_long(ratio_df, "Original_Contact_Ratio")

        iq_df, _, _ = _iq_tables_from_store(iq_summary_store, cat)
        iq_long = _iq_table_to_long(iq_df)
        df_long = normalized_long.merge(iq_long[["ds", "IQ_value"]], on="ds", how="left")
        if not ratio_long.empty:
            df_long = df_long.merge(ratio_long[["ds", "Original_Contact_Ratio"]], on="ds", how="left")
        df_long["IQ_value"] = pd.to_numeric(df_long["IQ_value"], errors="coerce").fillna(1.0)
        scaler = MinMaxScaler()
        df_long["IQ_value_scaled"] = scaler.fit_transform(df_long[["IQ_value"]]).round(4)
        df_long = df_long.sort_values("ds").reset_index(drop=True)
        df_long["y"] = df_long["Normalized_Ratio_1"]

        holiday_df = None
        if holiday_store:
            try:
                h_payload = json.loads(holiday_store) if isinstance(holiday_store, str) else holiday_store
                mapping = h_payload.get("mapping", {})
                if mapping:
                    holiday_df = pd.DataFrame(
                        {"ds": pd.to_datetime(list(mapping.keys()), errors="coerce"), "holiday": list(mapping.values())}
                    )
                    holiday_df = holiday_df.dropna(subset=["ds"])
            except Exception:
                holiday_df = None

        regressors = ["IQ_value_scaled"]
        best_params, best_score = _prophet_cv_best(df_long, holiday_df, regressors)
        pred = _prophet_fit_full(df_long, best_params, regressors, holiday_df)

        df_long["Normalized_Ratio_Post_Prophet"] = pred["yhat"].values.round(4)
        df_long["Normalized_Volume"] = (
            pd.to_numeric(df_long["Normalized_Ratio_Post_Prophet"], errors="coerce")
            * pd.to_numeric(df_long["IQ_value"], errors="coerce")
        ).round(4)
        df_long["Final_Smoothed_Value"] = df_long["Normalized_Ratio_Post_Prophet"].round(4)
        df_long["Year"] = df_long["ds"].dt.year
        df_long["Month"] = df_long["ds"].dt.strftime("%b")
        df_long["Month_Year"] = df_long["ds"].dt.strftime("%b-%Y")
        if "Original_Contact_Ratio" in df_long.columns:
            df_long["Contact_Ratio"] = df_long["Original_Contact_Ratio"]
        else:
            df_long["Contact_Ratio"] = np.nan
        df_long["IQ_Value_Scaled"] = df_long["IQ_value_scaled"]
        holiday_name_map = {}
        if holiday_df is not None and not holiday_df.empty and "ds" in holiday_df.columns:
            holiday_df = holiday_df.copy()
            holiday_df["Month_Year"] = pd.to_datetime(holiday_df["ds"], errors="coerce").dt.strftime("%b-%Y")
            name_col = "holiday" if "holiday" in holiday_df.columns else None
            if name_col:
                holiday_name_map = (
                    holiday_df.groupby("Month_Year")[name_col]
                    .apply(lambda x: ", ".join(sorted({str(v) for v in x if str(v).strip()})))
                    .to_dict()
                )
        df_long["Holiday_Name"] = df_long["Month_Year"].map(holiday_name_map).fillna("")

        norm2_pivot = (
            df_long.pivot_table(index="Year", columns="Month", values="Final_Smoothed_Value")
            .reset_index()
        )
        months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_cols = [m for m in months_order if m in norm2_pivot.columns]
        norm2_pivot["Avg"] = norm2_pivot[month_cols].mean(axis=1).round(2)
        cols = ["Year"] + month_cols + ["Avg"]
        norm2_pivot = norm2_pivot[cols]

        line_fig = go.Figure()
        line_fig.add_trace(
            go.Scatter(x=df_long["ds"], y=df_long["Normalized_Ratio_1"], name="Normalized Ratio 1", mode="lines+markers")
        )
        line_fig.add_trace(
            go.Scatter(x=df_long["ds"], y=df_long["Final_Smoothed_Value"], name="Final Smoothed Value", mode="lines+markers")
        )
        line_fig.update_layout(title="Prophet Smoothing", xaxis_title="Month", yaxis_title="Value")

        table_cols = [
            "Year",
            "Month",
            "Contact_Ratio",
            "Month_Year",
            "Holiday_Name",
            "Normalized_Ratio_1",
            "IQ_Value_Scaled",
            "Normalized_Ratio_Post_Prophet",
            "Normalized_Volume",
            "Final_Smoothed_Value",
        ]
        table_df = df_long[table_cols].copy()
        table_df = table_df.rename(columns={"Normalized_Ratio_1": "Normalized Ratio 1"})
        columns = []
        for col in table_df.columns:
            columns.append({"name": col, "id": col, "editable": col == "Final_Smoothed_Value"})

        store_payload = {
            "prophet_table": table_df.to_json(date_format="iso", orient="split"),
            "norm2": norm2_pivot.to_json(date_format="iso", orient="split"),
            "params": best_params,
            "score": best_score,
            "ready": True,
        }

        status = f"Prophet smoothing complete. Best score (WAPE + 0.5*bias): {best_score:.4f}"

        return (
            status,
            table_df.to_dict("records"),
            columns,
            norm2_pivot.to_dict("records"),
            _cols(norm2_pivot),
            _ratio_fig(norm2_pivot, "Normalized Contact Ratio 2"),
            line_fig,
            json.dumps(store_payload),
        )
    except Exception:
        logger.exception("vs-prophet: failed")
        return "Prophet smoothing failed.", [], [], [], [], _empty_fig("Error"), _empty_fig(), None


@app.callback(
    Output("vs-prophet-save-status", "children"),
    Output("vs-prophet-store", "data", allow_duplicate=True),
    Output("vs-norm2-table", "data", allow_duplicate=True),
    Output("vs-norm2-table", "columns", allow_duplicate=True),
    Output("vs-norm2-chart", "figure", allow_duplicate=True),
    Input("vs-save-prophet", "n_clicks"),
    State("vs-prophet-table", "data"),
    State("vs-prophet-store", "data"),
    prevent_initial_call=True,
)
def _save_prophet_changes(n_clicks, table_rows, prophet_store):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not prophet_store:
        return "Run Prophet smoothing first.", no_update, no_update, no_update, no_update
    try:
        payload = json.loads(prophet_store) if isinstance(prophet_store, str) else prophet_store
        orig_json = payload.get("prophet_table")
        orig_df = pd.read_json(io.StringIO(orig_json), orient="split") if orig_json else pd.DataFrame()
    except Exception:
        orig_df = pd.DataFrame()
        payload = {}

    new_df = pd.DataFrame(table_rows) if table_rows else pd.DataFrame()
    if orig_df.empty or new_df.empty:
        return "No data to save.", no_update, no_update, no_update, no_update

    def _ensure_ds(df_in: pd.DataFrame) -> pd.DataFrame:
        df_work = df_in.copy()
        if "ds" in df_work.columns and df_work["ds"].notna().any():
            df_work["ds"] = pd.to_datetime(df_work["ds"], errors="coerce")
            return df_work
        if "Month_Year" in df_work.columns:
            df_work["ds"] = pd.to_datetime(df_work["Month_Year"], format="%b-%Y", errors="coerce")
            if df_work["ds"].isna().all():
                df_work["ds"] = pd.to_datetime(df_work["Month_Year"], errors="coerce")
            return df_work
        if "Year" in df_work.columns and "Month" in df_work.columns:
            month_num = df_work["Month"].apply(_month_name_to_num)
            date_str = (
                df_work["Year"].astype(str)
                + "-"
                + month_num.astype("Int64").astype(str).str.zfill(2)
                + "-01"
            )
            df_work["ds"] = pd.to_datetime(date_str, errors="coerce")
        return df_work

    changed = False
    if "Final_Smoothed_Value" in new_df.columns:
        try:
            changed = not np.allclose(
                pd.to_numeric(new_df["Final_Smoothed_Value"], errors="coerce"),
                pd.to_numeric(orig_df["Final_Smoothed_Value"], errors="coerce"),
                equal_nan=True,
            )
        except Exception:
            changed = True

    new_df = _ensure_ds(new_df)
    new_df["Year"] = new_df["ds"].dt.year
    new_df["Month"] = new_df["ds"].dt.strftime("%b")
    norm2_pivot = (
        new_df.pivot_table(index="Year", columns="Month", values="Final_Smoothed_Value")
        .reset_index()
    )
    months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_cols = [m for m in months_order if m in norm2_pivot.columns]
    norm2_pivot["Avg"] = norm2_pivot[month_cols].mean(axis=1).round(2)
    cols = ["Year"] + month_cols + ["Avg"]
    norm2_pivot = norm2_pivot[cols]

    payload["prophet_table"] = new_df.to_json(date_format="iso", orient="split")
    payload["norm2"] = norm2_pivot.to_json(date_format="iso", orient="split")
    payload["ready"] = True

    if not changed:
        status = (
            "No changes detected - Phase 1 is ready immediately! "
            "You can run Phase 1 below (no need to Save Changes if there aren't any!)."
        )
    else:
        status = "Changes saved. Phase 1 is ready."

    return (
        status,
        json.dumps(payload),
        norm2_pivot.to_dict("records"),
        _cols(norm2_pivot),
        _ratio_fig(norm2_pivot, "Normalized Contact Ratio 2"),
    )


@app.callback(
    Output("vs-phase1-status", "children"),
    Output("vs-phase1-results", "data"),
    Output("vs-phase1-results", "columns"),
    Output("vs-phase1-accuracy", "data"),
    Output("vs-phase1-accuracy", "columns"),
    Output("vs-phase1-tuning", "data"),
    Output("vs-phase1-tuning", "columns"),
    Output("vs-final-accuracy", "data"),
    Output("vs-final-accuracy", "columns"),
    Output("vs-phase1-config-store", "data", allow_duplicate=True),
    Output("vs-phase1-download-store", "data", allow_duplicate=True),
    Output("forecast-phase-store", "data", allow_duplicate=True),
    Input("vs-run-phase1", "n_clicks"),
    State("vs-prophet-store", "data"),
    State("forecast-phase-store", "data"),
    State("vs-holiday-store", "data"),
    prevent_initial_call=True,
)
def _run_phase1_from_volume(n_clicks, prophet_store, phase_store, holiday_store):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not prophet_store:
        return "Run Prophet smoothing first.", [], [], [], [], [], [], [], [], None, None, phase_store

    payload = _normalize_phase_store(prophet_store)
    prophet_json = payload.get("prophet_table") if isinstance(payload, dict) else None
    if not prophet_json:
        return "Prophet data missing.", [], [], [], [], [], [], [], [], None, None, phase_store
    df = pd.read_json(io.StringIO(prophet_json), orient="split")
    if df.empty:
        return "Prophet data missing.", [], [], [], [], [], [], [], [], None, None, phase_store
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    elif "Month_Year" in df.columns:
        df["ds"] = pd.to_datetime(df["Month_Year"], format="%b-%Y", errors="coerce")
        if df["ds"].isna().all():
            df["ds"] = pd.to_datetime(df["Month_Year"], errors="coerce")
    elif "Year" in df.columns and "Month" in df.columns:
        month_num = df["Month"].apply(_month_name_to_num)
        date_str = (
            df["Year"].astype(str)
            + "-"
            + month_num.astype("Int64").astype(str).str.zfill(2)
            + "-01"
        )
        df["ds"] = pd.to_datetime(date_str, errors="coerce")
    df["Final_Smoothed_Value"] = pd.to_numeric(df["Final_Smoothed_Value"], errors="coerce")
    df["y"] = df["Final_Smoothed_Value"] / 100

    holiday_df = None
    if holiday_store:
        try:
            h_payload = json.loads(holiday_store) if isinstance(holiday_store, str) else holiday_store
            mapping = h_payload.get("mapping", {})
            if mapping:
                holiday_df = pd.DataFrame(
                    {"ds": pd.to_datetime(list(mapping.keys()), errors="coerce"), "holiday": list(mapping.values())}
                )
                holiday_df = holiday_df.dropna(subset=["ds"])
        except Exception:
            holiday_df = None

    config = config_manager.load_config()
    forecast_res = run_contact_ratio_forecast(
        df,
        12,
        holiday_mapping=None,
        holiday_original_date=holiday_df,
        config=config,
    )
    train = forecast_res.get("train", pd.DataFrame())
    test = forecast_res.get("test", pd.DataFrame())
    total_points = len(df)
    status_lines = [
        f"Starting Phase 1 Processing",
        f"Total data points: {total_points} months",
    ]
    if not train.empty:
        status_lines.append(
            f"Train Period: {train['ds'].min().strftime('%b %Y')} to {train['ds'].max().strftime('%b %Y')} ({len(train)} months)"
        )
    if not test.empty:
        status_lines.append(
            f"Test Period: {test['ds'].min().strftime('%b %Y')} to {test['ds'].max().strftime('%b %Y')} ({len(test)} months)"
        )
    results = forecast_res.get("forecast_results", {})
    combined = pd.DataFrame()
    wide = pd.DataFrame()
    pivot_smoothed = pd.DataFrame()
    accuracy_tbl = pd.DataFrame()
    tuning_tbl = pd.DataFrame()
    final_accuracy_tbl = pd.DataFrame()
    tuned_config = config
    download_csv = None

    if results:
        results_with_smoothed = dict(results)
        smoothed_vals = forecast_res.get("final_smoothed_values")
        if smoothed_vals is not None:
            results_with_smoothed["final_smoothed_values"] = smoothed_vals
        combined, wide, pivot_smoothed = process_forecast_results(results_with_smoothed)
        if not wide.empty and not pivot_smoothed.empty:
            wide = fill_final_smoothed_row(wide.copy(), pivot_smoothed)
        if not wide.empty and not pivot_smoothed.empty:
            try:
                accuracy_tbl = accuracy_phase1(wide, pivot_smoothed)
            except Exception:
                accuracy_tbl = pd.DataFrame()

    def _config_table(cfg: dict) -> pd.DataFrame:
        rows = []
        for model_name, params in (cfg or {}).items():
            if not isinstance(params, dict):
                continue
            for key, val in params.items():
                if isinstance(val, (dict, list, tuple)):
                    val = json.dumps(val)
                rows.append({"Model": model_name, "Parameter": key, "Value": val})
        return pd.DataFrame(rows)

    forecast_horizon = len(test) if not test.empty else 0
    if not train.empty and forecast_horizon > 0:
        try:
            actual_data = df[["ds", "y"]].dropna()
            initial_accuracy_df, _, _ = train_and_evaluate_func(
                config,
                train,
                actual_data,
                forecast_horizon,
                show_details=False,
            )
            tuned_config, _acc_before, tuned_accuracy_df = iterative_tuning(
                config,
                initial_accuracy_df,
                train_and_evaluate_func,
                train,
                forecast_horizon,
                actual_data,
            )
            tuning_tbl = _config_table(tuned_config)
            final_accuracy_tbl = tuned_accuracy_df.copy() if tuned_accuracy_df is not None else pd.DataFrame()

            tuned_forecast = run_contact_ratio_forecast(
                df,
                forecast_horizon,
                holiday_mapping=None,
                holiday_original_date=holiday_df,
                config=tuned_config,
            )
            tuned_results = tuned_forecast.get("forecast_results", {})
            if tuned_results:
                tuned_with_smoothed = dict(tuned_results)
                tuned_smoothed_vals = tuned_forecast.get("final_smoothed_values")
                if tuned_smoothed_vals is not None:
                    tuned_with_smoothed["final_smoothed_values"] = tuned_smoothed_vals
                combined_tuned, wide_tuned, pivot_tuned = process_forecast_results(tuned_with_smoothed)
                if not wide_tuned.empty and not pivot_tuned.empty:
                    wide_tuned = fill_final_smoothed_row(wide_tuned.copy(), pivot_tuned)
                if not wide_tuned.empty and not pivot_tuned.empty:
                    final_accuracy_tbl = accuracy_phase1(wide_tuned, pivot_tuned)
                if not wide_tuned.empty:
                    download_csv = create_download_csv_with_metadata(wide_tuned, tuned_config)
        except Exception:
            logger.exception("vs-phase1: tuning failed")

    if final_accuracy_tbl.empty:
        final_accuracy_tbl = accuracy_tbl.copy()

    if download_csv is None and not wide.empty:
        try:
            download_csv = create_download_csv_with_metadata(wide, tuned_config)
        except Exception:
            download_csv = None

    display_results = pd.DataFrame()
    if not wide.empty:
        display_results = _format_ratio_wide(wide, add_avg=True)
    elif not combined.empty:
        display_results = combined.copy()
        if "Forecast" in display_results.columns:
            display_results["Forecast"] = pd.to_numeric(
                display_results["Forecast"], errors="coerce"
            ).apply(lambda v: f"{v * 100:.2f}%" if pd.notna(v) else "")
    if results:
        status_lines.append("Prophet Forecast Done" if results.get("prophet") is not None else "Prophet Forecast Failed")
        status_lines.append("RF Forecast Done" if results.get("random_forest") is not None else "RF Forecast Failed")
        status_lines.append("XGBoost Forecast Done" if results.get("xgboost") is not None else "XGBoost Forecast Failed")
        status_lines.append("VAR Forecast Done" if results.get("var") is not None else "VAR Forecast Failed")
        status_lines.append("Sarimax Forecast Done" if results.get("sarimax") is not None else "Sarimax Forecast Failed")

    phase_data = _normalize_phase_store(phase_store)
    phase_data["phase1"] = json.dumps(
        _phase1_compact_from_results(
            {"smoothed": df[["ds", "Final_Smoothed_Value"]].rename(columns={"Final_Smoothed_Value": "smoothed"}).to_dict("records")}
        )
    )
    phase_data["phase1_meta"] = {"source": "prophet-volume-summary", "ts": pd.Timestamp.utcnow().isoformat()}

    config_store = json.dumps(tuned_config)
    return (
        html.Ul([html.Li(line) for line in status_lines]),
        display_results.to_dict("records") if not display_results.empty else [],
        _cols(display_results) if not display_results.empty else [],
        accuracy_tbl.to_dict("records") if isinstance(accuracy_tbl, pd.DataFrame) and not accuracy_tbl.empty else [],
        _cols(accuracy_tbl) if isinstance(accuracy_tbl, pd.DataFrame) and not accuracy_tbl.empty else [],
        tuning_tbl.to_dict("records") if not tuning_tbl.empty else [],
        _cols(tuning_tbl) if not tuning_tbl.empty else [],
        final_accuracy_tbl.to_dict("records") if isinstance(final_accuracy_tbl, pd.DataFrame) and not final_accuracy_tbl.empty else [],
        _cols(final_accuracy_tbl) if isinstance(final_accuracy_tbl, pd.DataFrame) and not final_accuracy_tbl.empty else [],
        config_store,
        download_csv,
        phase_data,
    )


@app.callback(
    Output("vs-phase1-configs", "children"),
    Input("vs-phase1-config-store", "data"),
)
def _render_phase1_configs(config_json):
    if not config_json:
        return html.Div("Run Phase 1 to view model configurations.", className="small text-muted")
    try:
        cfg = json.loads(config_json) if isinstance(config_json, str) else config_json
    except Exception:
        return html.Div("Could not parse configuration data.", className="small text-muted")
    if not isinstance(cfg, dict) or not cfg:
        return html.Div("No configuration data found.", className="small text-muted")

    items = []
    for model_name, params in cfg.items():
        if not isinstance(params, dict):
            continue
        rows = []
        for key, val in params.items():
            if isinstance(val, (dict, list, tuple)):
                val = json.dumps(val)
            rows.append({"Parameter": key, "Value": val})
        df = pd.DataFrame(rows)
        table = dash_table.DataTable(
            data=df.to_dict("records"),
            columns=_cols(df),
            page_size=8,
            page_action="none",
            fixed_rows={"headers": True},
            style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "360px"},
            style_cell={"fontSize": 12, "padding": "6px 8px"},
            style_header={"fontWeight": "600"},
        )
        items.append(dbc.AccordionItem(table, title=str(model_name)))

    if not items:
        return html.Div("No model configuration tables to display.", className="small text-muted")

    return dbc.Accordion(items, always_open=True, flush=True, start_collapsed=True)


@app.callback(
    Output("vs-config-download", "data"),
    Input("vs-config-download-btn", "n_clicks"),
    State("vs-phase1-config-store", "data"),
    prevent_initial_call=True,
)
def _download_phase1_config(n_clicks, config_json):
    if not n_clicks or not config_json:
        raise dash.exceptions.PreventUpdate
    try:
        cfg = json.loads(config_json) if isinstance(config_json, str) else config_json
    except Exception:
        raise dash.exceptions.PreventUpdate

    rows = []
    for model_name, params in (cfg or {}).items():
        if not isinstance(params, dict):
            continue
        for key, val in params.items():
            if isinstance(val, (dict, list, tuple)):
                val = json.dumps(val)
            rows.append({"Model": model_name, "Parameter": key, "Value": val})
    df = pd.DataFrame(rows)
    if df.empty:
        raise dash.exceptions.PreventUpdate
    return dcc.send_data_frame(df.to_csv, "phase1_config_summary.csv", index=False)


@app.callback(
    Output("vs-phase1-download", "data"),
    Input("vs-phase1-download-btn", "n_clicks"),
    State("vs-phase1-download-store", "data"),
    prevent_initial_call=True,
)
def _download_phase1_results(n_clicks, csv_text):
    if not n_clicks or not csv_text:
        raise dash.exceptions.PreventUpdate
    return dcc.send_string(lambda: csv_text, "phase1_results_with_config.csv")


@app.callback(
    Output("vs-phase2-status", "children"),
    Output("vs-phase2-forecast", "data"),
    Output("vs-phase2-forecast", "columns"),
    Output("vs-phase2-base", "data"),
    Output("vs-phase2-base", "columns"),
    Output("vs-phase2-fg-summary", "data"),
    Output("vs-phase2-fg-summary", "columns"),
    Output("vs-phase2-volume-split", "data"),
    Output("vs-phase2-volume-split", "columns"),
    Output("vs-volume-split-edit", "data"),
    Output("vs-volume-split-edit", "columns"),
    Output("vs-volume-split-info", "children"),
    Output("vs-phase2-store", "data", allow_duplicate=True),
    Output("vs-adjusted-forecast", "data", allow_duplicate=True),
    Output("vs-adjusted-forecast", "columns", allow_duplicate=True),
    Output("vs-adjusted-verify", "data", allow_duplicate=True),
    Output("vs-adjusted-verify", "columns", allow_duplicate=True),
    Output("vs-adjusted-status", "children", allow_duplicate=True),
    Output("vs-adjusted-store", "data", allow_duplicate=True),
    Output("vs-save-adjusted-status", "children", allow_duplicate=True),
    Input("vs-run-phase2", "n_clicks"),
    Input("vs-clear-phase2", "n_clicks"),
    State("vs-phase2-start", "date"),
    State("vs-phase2-end", "date"),
    State("vs-prophet-store", "data"),
    State("vs-phase1-config-store", "data"),
    State("vs-results-store", "data"),
    State("vs-iq-summary-store", "data"),
    State("vs-category-tabs", "value"),
    prevent_initial_call=True,
)
def _run_phase2_from_volume(
    n_clicks,
    n_clear,
    start_date,
    end_date,
    prophet_store,
    config_store,
    results_store,
    iq_summary_store,
    cat,
):
    ctx = dash.callback_context
    trigger = ctx.triggered_id if ctx.triggered_id else None

    def _empty_phase2(status_text: str):
        return (
            status_text,
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            "",
            None,
            [],
            [],
            [],
            [],
            "",
            None,
            "",
        )

    if trigger == "vs-clear-phase2":
        return _empty_phase2("Phase 2 cache cleared.")
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not start_date or not end_date:
        return _empty_phase2("Select start and end dates for Phase 2.")

    try:
        start = pd.to_datetime(start_date).to_period("M").to_timestamp()
        end = pd.to_datetime(end_date).to_period("M").to_timestamp()
    except Exception:
        return _empty_phase2("Invalid dates for Phase 2.")
    if end < start:
        return _empty_phase2("End date must be after start date.")
    forecast_months = (end.year - start.year) * 12 + (end.month - start.month) + 1

    if not prophet_store:
        return _empty_phase2("Run Phase 1 first to generate smoothing data.")

    payload = _normalize_phase_store(prophet_store)
    prophet_json = payload.get("prophet_table") if isinstance(payload, dict) else None
    if not prophet_json:
        return _empty_phase2("Prophet smoothing data not found.")

    df_smooth = pd.read_json(io.StringIO(prophet_json), orient="split")
    if df_smooth.empty:
        return _empty_phase2("Prophet smoothing data is empty.")

    if "ds" in df_smooth.columns:
        df_smooth["ds"] = pd.to_datetime(df_smooth["ds"], errors="coerce")
    elif "Month_Year" in df_smooth.columns:
        df_smooth["ds"] = pd.to_datetime(df_smooth["Month_Year"], format="%b-%Y", errors="coerce")
        if df_smooth["ds"].isna().all():
            df_smooth["ds"] = pd.to_datetime(df_smooth["Month_Year"], errors="coerce")
    elif "Year" in df_smooth.columns and "Month" in df_smooth.columns:
        month_num = df_smooth["Month"].apply(_month_name_to_num)
        date_str = (
            df_smooth["Year"].astype(str)
            + "-"
            + month_num.astype("Int64").astype(str).str.zfill(2)
            + "-01"
        )
        df_smooth["ds"] = pd.to_datetime(date_str, errors="coerce")

    df_smooth["Final_Smoothed_Value"] = pd.to_numeric(df_smooth.get("Final_Smoothed_Value"), errors="coerce")

    iq_df, vol_df, _ = _iq_tables_from_store(iq_summary_store, cat or "")
    iq_long = unpivot_iq_summary(iq_df.copy()) if iq_df is not None and not iq_df.empty else pd.DataFrame()
    if not iq_long.empty:
        iq_long["Year"] = iq_long["Year"].astype(int)
        iq_long["Month"] = iq_long["Month"].astype(str).str.strip()
        df_smooth["Year"] = df_smooth["ds"].dt.year
        df_smooth["Month"] = df_smooth["ds"].dt.strftime("%b")
        df_smooth = df_smooth.merge(iq_long, on=["Year", "Month"], how="left")
    if "IQ_value" not in df_smooth.columns or df_smooth["IQ_value"].isna().all():
        if "Normalized_Volume" in df_smooth.columns and "Normalized_Ratio_Post_Prophet" in df_smooth.columns:
            denom = pd.to_numeric(df_smooth["Normalized_Ratio_Post_Prophet"], errors="coerce").replace(0, np.nan)
            df_smooth["IQ_value"] = pd.to_numeric(df_smooth["Normalized_Volume"], errors="coerce") / denom
        else:
            df_smooth["IQ_value"] = 1.0

    cfg = config_manager.load_config()
    if config_store:
        try:
            cfg = json.loads(config_store) if isinstance(config_store, str) else config_store
        except Exception:
            cfg = config_manager.load_config()

    phase2_res = run_phase2_forecast(df_smooth, forecast_months, config=cfg)
    forecast_results = phase2_res.get("forecast_results", {})
    if not forecast_results:
        return _empty_phase2("Phase 2 forecast failed to produce results.")

    combined, wide, pivot_smoothed = process_forecast_results(forecast_results)
    if combined.empty:
        return _empty_phase2("Phase 2 forecast returned no rows.")

    combined["ds"] = pd.to_datetime(combined["Month_Year"], format="%b-%y", errors="coerce")
    combined["Year"] = combined["ds"].dt.year
    combined["Month"] = combined["ds"].dt.strftime("%b")

    history_df = pd.DataFrame()
    if "ds" in df_smooth.columns:
        history_df = df_smooth.copy()
        history_df["ds"] = pd.to_datetime(history_df["ds"], errors="coerce")
        history_df = history_df.dropna(subset=["ds"])
        if not history_df.empty:
            history_df["Year"] = history_df["ds"].dt.year
            history_df["Month"] = history_df["ds"].dt.strftime("%b")
            history_df["Month_Year"] = history_df["ds"].dt.strftime("%b-%y")
            ratio_vals = pd.to_numeric(history_df.get("Final_Smoothed_Value"), errors="coerce")
            if ratio_vals.notna().any() and ratio_vals.dropna().median() > 1:
                ratio_vals = ratio_vals / 100.0
            history_df["Forecast"] = ratio_vals
            history_df["Model"] = "Final_smoothed"
            keep_cols = ["ds", "Year", "Month", "Month_Year", "Model", "Forecast"]
            for col in ("IQ_value", "Normalized_Volume"):
                if col in history_df.columns:
                    keep_cols.append(col)
            history_df = history_df[keep_cols]

    merged_df = pd.concat([history_df, combined], ignore_index=True) if not history_df.empty else combined.copy()
    if not iq_long.empty:
        merged_df = merged_df.merge(iq_long, on=["Year", "Month"], how="left", suffixes=("", "_iq"))
        if "IQ_value_iq" in merged_df.columns:
            merged_df["IQ_value"] = merged_df["IQ_value"].fillna(merged_df["IQ_value_iq"])
            merged_df = merged_df.drop(columns=["IQ_value_iq"])
    merged_df["ds"] = pd.to_datetime(merged_df.get("ds"), errors="coerce")
    merged_df = merged_df.sort_values("ds").reset_index(drop=True)
    merged_df["IQ_value"] = pd.to_numeric(merged_df.get("IQ_value"), errors="coerce")
    if merged_df["IQ_value"].notna().any():
        merged_df["IQ_value"] = merged_df.groupby(["Year", "Month"])["IQ_value"].transform(
            lambda s: s.fillna(s.dropna().iloc[0]) if s.dropna().any() else s
        )
        merged_df["IQ_value"] = merged_df["IQ_value"].ffill()
    merged_df["Forecast"] = pd.to_numeric(merged_df.get("Forecast"), errors="coerce")
    merged_df["Base_Forecast_Category"] = (
        merged_df["Forecast"] * merged_df["IQ_value"] * 1_000_000
    ).round().astype("Int64")
    merged_df["Contact_Ratio_Forecast_Category"] = merged_df["Forecast"].apply(fmt_percent1)
    merged_df["IQ_value_Category"] = merged_df["IQ_value"].apply(fmt_millions_1)

    volume_long = pd.DataFrame()
    if vol_df is not None and not vol_df.empty:
        month_cols = [c for c in vol_df.columns if c != "Year"]
        volume_long = vol_df.melt(id_vars="Year", value_vars=month_cols, var_name="Month", value_name="volume")
        volume_long["Year"] = pd.to_numeric(volume_long["Year"], errors="coerce")
        volume_long["Month"] = volume_long["Month"].astype(str).str.strip()
        volume_long["volume"] = (
            volume_long["volume"].astype(str).str.replace(",", "", regex=False).str.strip()
        )

        def _parse_vol(x):
            if x is None:
                return None
            s = str(x).lower()
            try:
                if s.endswith("k"):
                    return float(s[:-1]) * 1000
                if s.endswith("m"):
                    return float(s[:-1]) * 1_000_000
                return float(s)
            except Exception:
                return None

        volume_long["volume"] = volume_long["volume"].apply(_parse_vol)
        volume_long = volume_long.dropna(subset=["Year", "Month", "volume"])
        volume_long = volume_long.rename(columns={"Year": "year", "Month": "month"})

    base_df = merged_df.copy()
    if not volume_long.empty:
        base_df = map_original_volume_to_phase2_forecast(base_df, volume_long)
    smoothing_norm = pd.DataFrame()
    if "Normalized_Volume" in df_smooth.columns:
        smoothing_norm = df_smooth.copy()
        smoothing_norm["Year"] = smoothing_norm["ds"].dt.year
        smoothing_norm["Month"] = smoothing_norm["ds"].dt.strftime("%b")
        smoothing_norm = smoothing_norm[["Year", "Month", "Normalized_Volume"]]
    if not smoothing_norm.empty:
        base_df = map_normalized_volume_to_forecast(base_df, smoothing_norm)

    if "volume" in base_df.columns:
        base_df = base_df.rename(columns={"volume": "Original_volume"})
    base_df["Original_volume_Category"] = pd.to_numeric(base_df.get("Original_volume"), errors="coerce")
    base_df["Normalized_Volume_Category"] = pd.to_numeric(base_df.get("Normalized_Volume"), errors="coerce")

    display_phase2 = pd.DataFrame()
    if not wide.empty:
        display_phase2 = _format_ratio_wide(wide)
    elif not combined.empty:
        display_phase2 = combined.copy()
        if "Forecast" in display_phase2.columns:
            display_phase2["Forecast"] = pd.to_numeric(
                display_phase2["Forecast"], errors="coerce"
            ).apply(lambda v: f"{v * 100:.2f}%" if pd.notna(v) else "")

    base_cols = [
        "Year",
        "Month",
        "Model",
        "Contact_Ratio_Forecast_Category",
        "IQ_value_Category",
        "Base_Forecast_Category",
        "Original_volume_Category",
        "Normalized_Volume_Category",
    ]
    base_display = base_df[[c for c in base_cols if c in base_df.columns]].copy()
    if "Original_volume_Category" in base_display.columns:
        base_display = base_display.rename(columns={"Original_volume_Category": "Original_Volume_Category"})
    base_display = _sort_year_month(base_display)
    base_display = _clean_table(base_display)

    fg_summary = pd.DataFrame()
    fg_split = pd.DataFrame()
    fg_monthly = pd.DataFrame()
    volume_split_edit = pd.DataFrame()
    volume_split_info = ""
    if results_store and cat:
        try:
            rs_payload = json.loads(results_store) if isinstance(results_store, str) else results_store
            data_json = rs_payload.get("data")
        except Exception:
            data_json = None
        if data_json:
            df_norm = pd.read_json(io.StringIO(data_json), orient="split")
            try:
                fg_summary, fg_split, _, _, _ = forecast_group_pivot_and_long_style(df_norm, cat)
            except Exception:
                fg_summary, fg_split = _fallback_pivots(df_norm, cat)
            if not df_norm.empty and {"date", "volume", "forecast_group", "category"}.issubset(df_norm.columns):
                df_cat = df_norm[df_norm["category"] == cat].copy()
                df_cat["ds"] = pd.to_datetime(df_cat["date"], errors="coerce")
                df_cat["Year"] = df_cat["ds"].dt.year
                df_cat["Month"] = df_cat["ds"].dt.strftime("%b")
                df_cat["volume"] = pd.to_numeric(df_cat["volume"], errors="coerce")
                df_cat = df_cat.dropna(subset=["Year", "Month", "forecast_group", "volume"])
                if not df_cat.empty:
                    fg_monthly = (
                        df_cat.groupby(["Year", "Month", "forecast_group"], as_index=False)["volume"].sum()
                    )

    if fg_split is not None and not fg_split.empty:
        split_clean = fg_split.copy()
        split_clean = split_clean[split_clean["Year"] != "--------"].copy() if "Year" in split_clean.columns else split_clean
        split_clean["Year_Numeric"] = pd.to_numeric(split_clean["Year"], errors="coerce")
        def _pick_latest(group: pd.DataFrame) -> pd.Series:
            year_numeric = group["Year_Numeric"]
            if year_numeric.notna().any():
                return group.loc[year_numeric.idxmax()]
            return group.iloc[-1]

        latest_data = split_clean.groupby("forecast_group").apply(_pick_latest).reset_index(drop=True)
        if "Vol_Split_Last_3M" in latest_data.columns:
            latest_data["Vol_Split_Last_3M_Numeric"] = (
                latest_data["Vol_Split_Last_3M"].astype(str).str.replace("%", "", regex=False)
            )
            latest_data["Vol_Split_Last_3M_Numeric"] = pd.to_numeric(
                latest_data["Vol_Split_Last_3M_Numeric"], errors="coerce"
            )
        else:
            latest_data["Vol_Split_Last_3M_Numeric"] = 0.0
        total_original = latest_data["Vol_Split_Last_3M_Numeric"].sum()
        if total_original > 0:
            latest_data["Vol_Split_Normalized"] = (
                latest_data["Vol_Split_Last_3M_Numeric"] / total_original * 100
            ).round(1)
        else:
            latest_data["Vol_Split_Normalized"] = 0.0
        volume_split_edit = latest_data[
            ["forecast_group", "Year", "Vol_Split_Last_3M_Numeric", "Vol_Split_Normalized"]
        ].copy()
        volume_split_info = (
            f"Volume Split% last 3 Months total: {total_original:.1f}% | "
            f"Final normalized: {volume_split_edit['Vol_Split_Normalized'].sum():.1f}%"
        )

    if fg_summary is not None and not fg_summary.empty:
        fg_summary = _clean_table(fg_summary)
    if fg_split is not None and not fg_split.empty:
        fg_split = _clean_table(fg_split)

    edit_cols = [
        {"name": "Forecast Group", "id": "forecast_group", "editable": False},
        {"name": "Year", "id": "Year", "editable": False},
        {"name": "Volume Split% last 3Months", "id": "Vol_Split_Last_3M_Numeric", "editable": False},
        {"name": "Vol_Split_Normalized", "id": "Vol_Split_Normalized", "editable": True},
    ]

    store_payload = {
        "base_df": base_df.to_json(date_format="iso", orient="split"),
        "forecast_group_split": fg_split.to_json(date_format="iso", orient="split") if fg_split is not None else None,
        "volume_split_edit": volume_split_edit.to_json(date_format="iso", orient="split")
        if not volume_split_edit.empty
        else None,
        "forecast_group_monthly": fg_monthly.to_json(date_format="iso", orient="split")
        if fg_monthly is not None and not fg_monthly.empty
        else None,
    }

    return (
        f"Phase 2 forecast ready ({forecast_months} months).",
        display_phase2.to_dict("records") if not display_phase2.empty else [],
        _cols(display_phase2) if not display_phase2.empty else [],
        base_display.to_dict("records"),
        _cols(base_display),
        fg_summary.to_dict("records") if fg_summary is not None and not fg_summary.empty else [],
        _cols(fg_summary) if fg_summary is not None and not fg_summary.empty else [],
        fg_split.to_dict("records") if fg_split is not None and not fg_split.empty else [],
        _cols(fg_split) if fg_split is not None and not fg_split.empty else [],
        volume_split_edit.to_dict("records") if not volume_split_edit.empty else [],
        edit_cols if not volume_split_edit.empty else [],
        volume_split_info,
        json.dumps(store_payload),
        [],
        [],
        [],
        [],
        "",
        None,
        "",
    )


@app.callback(
    Output("vs-adjusted-forecast", "data", allow_duplicate=True),
    Output("vs-adjusted-forecast", "columns", allow_duplicate=True),
    Output("vs-adjusted-verify", "data", allow_duplicate=True),
    Output("vs-adjusted-verify", "columns", allow_duplicate=True),
    Output("vs-adjusted-status", "children", allow_duplicate=True),
    Output("vs-adjusted-store", "data", allow_duplicate=True),
    Input("vs-apply-volume-split", "n_clicks"),
    State("vs-volume-split-edit", "data"),
    State("vs-phase2-store", "data"),
    prevent_initial_call=True,
)
def _apply_volume_split(n_clicks, split_rows, phase2_store):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not phase2_store:
        return [], [], [], [], "Run Phase 2 first.", None

    try:
        payload = json.loads(phase2_store) if isinstance(phase2_store, str) else phase2_store
        base_json = payload.get("base_df")
        fg_monthly_json = payload.get("forecast_group_monthly")
    except Exception:
        base_json = None
        fg_monthly_json = None
    if not base_json:
        return [], [], [], [], "Phase 2 base forecast missing.", None
    base_df = pd.read_json(io.StringIO(base_json), orient="split")
    if base_df.empty:
        return [], [], [], [], "Phase 2 base forecast missing.", None

    split_df = pd.DataFrame(split_rows) if split_rows else pd.DataFrame()
    if split_df.empty or "forecast_group" not in split_df.columns:
        return [], [], [], [], "Volume split data missing.", None

    split_df["Vol_Split_Normalized"] = pd.to_numeric(split_df["Vol_Split_Normalized"], errors="coerce").fillna(0.0)
    total_norm = split_df["Vol_Split_Normalized"].sum()
    if total_norm > 0:
        split_df["Vol_Split_Final"] = (split_df["Vol_Split_Normalized"] / total_norm * 100).round(1)
    else:
        split_df["Vol_Split_Final"] = 0.0

    mapping = dict(zip(split_df["forecast_group"], split_df["Vol_Split_Final"] / 100.0))

    adjusted_results = []
    for fg, split_pct in mapping.items():
        fg_forecast = base_df.copy()
        fg_forecast["forecast_group"] = fg
        fg_forecast["Volume_Split_%Fg"] = split_pct * 100
        fg_forecast["Base_Forecast_for_Forecast_Group"] = (
            pd.to_numeric(fg_forecast["Base_Forecast_Category"], errors="coerce") * split_pct
        ).round().astype("Int64")
        adjusted_results.append(fg_forecast)

    adjusted_df = pd.concat(adjusted_results, ignore_index=True) if adjusted_results else pd.DataFrame()
    if adjusted_df.empty:
        return [], [], [], [], "No adjusted forecast generated.", None

    if "Contact_Ratio_Forecast_Category" in adjusted_df.columns:
        adjusted_df["Contact_Ratio_Forecast_Group"] = adjusted_df["Contact_Ratio_Forecast_Category"]
    if "IQ_value_Category" in adjusted_df.columns:
        adjusted_df["IQ_Value_Category"] = adjusted_df["IQ_value_Category"]
    adjusted_df["Volume_Split%_Forecast_Group"] = pd.to_numeric(
        adjusted_df.get("Volume_Split_%Fg"), errors="coerce"
    )

    fg_monthly = pd.DataFrame()
    if fg_monthly_json:
        try:
            fg_monthly = pd.read_json(io.StringIO(fg_monthly_json), orient="split")
        except Exception:
            fg_monthly = pd.DataFrame()
    if not fg_monthly.empty:
        fg_monthly["Year"] = pd.to_numeric(fg_monthly["Year"], errors="coerce")
        fg_monthly["Month"] = fg_monthly["Month"].astype(str).str.strip()
        fg_monthly["volume"] = pd.to_numeric(fg_monthly.get("volume"), errors="coerce")
        fg_monthly = fg_monthly.dropna(subset=["Year", "Month", "forecast_group", "volume"])
        fg_monthly = fg_monthly.rename(columns={"volume": "Actual_Forecast_Group_Original_Volume"})
        adjusted_df = adjusted_df.merge(
            fg_monthly[["Year", "Month", "forecast_group", "Actual_Forecast_Group_Original_Volume"]],
            on=["Year", "Month", "forecast_group"],
            how="left",
        )

    verify_df = adjusted_df.groupby(["Year", "Month", "Model"], as_index=False).agg(
        Base_Forecast_Category=("Base_Forecast_Category", "first"),
        Base_Forecast_for_Forecast_Group=("Base_Forecast_for_Forecast_Group", "sum"),
    )
    verify_df["Difference"] = (
        pd.to_numeric(verify_df["Base_Forecast_for_Forecast_Group"], errors="coerce")
        - pd.to_numeric(verify_df["Base_Forecast_Category"], errors="coerce")
    )

    display_cols = [
        "Year",
        "Month",
        "Model",
        "forecast_group",
        "Volume_Split%_Forecast_Group",
        "Contact_Ratio_Forecast_Group",
        "IQ_Value_Category",
        "Base_Forecast_Category",
        "Base_Forecast_for_Forecast_Group",
        "Actual_Forecast_Group_Original_Volume",
    ]
    adjusted_display = adjusted_df[[c for c in display_cols if c in adjusted_df.columns]].copy()
    adjusted_display = adjusted_display.rename(columns={"forecast_group": "Forecast_Group"})
    required_cols = [
        "Year",
        "Month",
        "Model",
        "Forecast_Group",
        "Volume_Split%_Forecast_Group",
        "Contact_Ratio_Forecast_Group",
        "IQ_Value_Category",
        "Base_Forecast_Category",
        "Base_Forecast_for_Forecast_Group",
        "Actual_Forecast_Group_Original_Volume",
    ]
    for col in required_cols:
        if col not in adjusted_display.columns:
            adjusted_display[col] = None
    adjusted_display = adjusted_display[required_cols]
    adjusted_display = _sort_year_month(adjusted_display)
    verify_df = _sort_year_month(verify_df)
    verify_df = _clean_table(verify_df)
    adjusted_store = adjusted_df.to_json(date_format="iso", orient="split")
    status = "Volume Split Applied successfully to base forecast."
    return (
        adjusted_display.to_dict("records"),
        _cols(adjusted_display),
        verify_df.to_dict("records"),
        _cols(verify_df),
        status,
        adjusted_store,
    )


@app.callback(
    Output("vs-download-adjusted-file", "data"),
    Input("vs-download-adjusted", "n_clicks"),
    State("vs-adjusted-store", "data"),
    prevent_initial_call=True,
)
def _download_adjusted_forecast(n_clicks, adjusted_json):
    if not n_clicks or not adjusted_json:
        raise dash.exceptions.PreventUpdate
    df = pd.read_json(io.StringIO(adjusted_json), orient="split")
    if df.empty:
        raise dash.exceptions.PreventUpdate
    return dcc.send_data_frame(df.to_csv, "adjusted_forecast_by_group.csv", index=False)


@app.callback(
    Output("vs-save-adjusted-status", "children", allow_duplicate=True),
    Output("vs-save-adjusted-download", "data"),
    Output("vs-next-step", "style"),
    Input("vs-save-adjusted", "n_clicks"),
    State("vs-adjusted-store", "data"),
    State("vs-category-tabs", "value"),
    prevent_initial_call=True,
)
def _save_adjusted_to_db(n_clicks, adjusted_json, category):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    hidden_style = {"display": "none"}
    shown_style = {"display": "inline-block"}
    if not adjusted_json:
        return "No adjusted forecast to save.", no_update, hidden_style
    try:
        df = pd.read_json(io.StringIO(adjusted_json), orient="split")
    except Exception as exc:
        return f"Could not parse adjusted forecast: {exc}", no_update, hidden_style
    if df.empty:
        return "Adjusted forecast is empty.", no_update, hidden_style

    def _safe_filename_part(value: Optional[str], fallback: str) -> str:
        if not value:
            return fallback
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
        return cleaned or fallback

    group_name = None
    if "forecast_group" in df.columns:
        unique_groups = [str(v) for v in df["forecast_group"].dropna().unique()]
        if len(unique_groups) == 1:
            group_name = unique_groups[0]
        elif len(unique_groups) > 1:
            group_name = "multiple_groups"
    if not group_name:
        group_name = category or "forecast_group"
    try:
        user = current_user_fallback() or "unknown"
    except Exception:
        user = "unknown"
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"Monthly_Forecast_{_safe_filename_part(group_name, 'forecast_group')}"
        f"_{ts}_{_safe_filename_part(user, 'user')}.csv"
    )
    download_data = dcc.send_data_frame(df.to_csv, filename, index=False)
    scope_key = "forecast|workspace|global|"
    metadata = {
        "source": "volume-summary-phase2",
        "saved_at": pd.Timestamp.utcnow().isoformat(),
    }
    try:
        run_id = cap_store.record_forecast_run(
            scope_key=scope_key,
            forecast_df=df,
            created_by=user,
            model_name="volume-summary",
            metadata=metadata,
            pushed_to_planning=False,
        )
        status = f"Saved forecast run #{run_id} by {user}. Downloading {filename}."
        return status, download_data, shown_style
    except Exception as exc:
        status = f"Download ready ({filename}). DB save failed: {exc}"
        return status, download_data, shown_style


@app.callback(
    Output("sa-upload-msg", "children"),
    Output("sa-preview", "data"),
    Output("sa-preview", "columns"),
    Output("sa-raw-store", "data", allow_duplicate=True),
    Input("sa-upload", "contents"),
    State("sa-upload", "filename"),
    prevent_initial_call=True,
)
def _on_sa_upload(contents, filename):
    if not contents or not filename:
        raise dash.exceptions.PreventUpdate
    df, msg = _parse_upload(contents, filename)
    preview = df.head(50)
    return msg, preview.to_dict("records"), _cols(preview), df.to_json(date_format="iso", orient="split")


@app.callback(
    Output("sa-alert", "children"),
    Output("sa-alert", "is_open"),
    Output("sa-smooth-chart", "figure"),
    Output("sa-anomaly-table", "data"),
    Output("sa-anomaly-table", "columns"),
    Output("sa-smooth-table", "data"),
    Output("sa-smooth-table", "columns"),
    Output("sa-ratio-table", "data"),
    Output("sa-ratio-table", "columns"),
    Output("sa-seasonality-table", "data"),
    Output("sa-seasonality-table", "columns"),
    Output("sa-norm1-chart", "figure"),
    Output("sa-norm2-chart", "figure", allow_duplicate=True),
    Output("sa-results-store", "data", allow_duplicate=True),
    Output("forecast-phase-store", "data", allow_duplicate=True),
    Input("sa-run-smoothing", "n_clicks"),
    Input("sa-run-prophet", "n_clicks"),
    State("sa-raw-store", "data"),
    State("sa-window", "value"),
    State("sa-threshold", "value"),
    State("sa-prophet-order", "value"),
    State("sa-holdout", "value"),
    State("forecast-phase-store", "data"),
    State("vs-data-store", "data"),
    State("vs-iq-store", "data"),
    prevent_initial_call=True,
)
def _run_smoothing(n_basic, n_prophet, raw_json, window, threshold, prophet_order, holdout, phase_store, vs_data_json, vs_iq_json):
    if not n_basic and not n_prophet:
        raise dash.exceptions.PreventUpdate
    _ = holdout  # placeholder for future train/test split logic
    if not raw_json and not vs_data_json:
        return (
            "Upload data to smooth.",
            True,
            _empty_fig("No data"),
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            _empty_fig(),
            _empty_fig(),
            None,
            phase_store,
        )

    def _load_df(data_json):
        try:
            return pd.read_json(io.StringIO(data_json), orient="split")
        except Exception:
            return pd.DataFrame()

    df = _load_df(raw_json) if raw_json else pd.DataFrame()
    if df.empty and vs_data_json:
        df = _load_df(vs_data_json)
    if df.empty:
        return (
            "Could not read uploaded data.",
            True,
            _empty_fig("Bad input"),
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            _empty_fig(),
            _empty_fig(),
            None,
            phase_store,
        )

    def _attach_iq(base_df: pd.DataFrame, iq_json: str) -> pd.DataFrame:
        if not iq_json or "IQ_value" in base_df.columns:
            return base_df
        try:
            iq_df = pd.read_json(io.StringIO(iq_json), orient="split")
        except Exception:
            return base_df
        base_date_col = _pick_col(base_df, ("date", "ds", "timestamp"))
        iq_date_col = _pick_col(iq_df, ("date", "ds", "timestamp"))
        iq_val_col = _pick_col(iq_df, ("iq_value", "iq", "value"))
        if not base_date_col or not iq_date_col or not iq_val_col:
            return base_df
        iq_use = iq_df[[iq_date_col, iq_val_col]].copy()
        iq_use["__iq_date"] = pd.to_datetime(iq_use[iq_date_col], errors="coerce").dt.normalize()
        iq_use["IQ_value"] = pd.to_numeric(iq_use[iq_val_col], errors="coerce")
        base_df["__iq_date"] = pd.to_datetime(base_df[base_date_col], errors="coerce").dt.normalize()
        base_df = base_df.merge(iq_use[["__iq_date", "IQ_value"]], on="__iq_date", how="left")
        base_df = base_df.drop(columns=["__iq_date"])
        return base_df

    df = _attach_iq(df, vs_iq_json)

    triggered = dash.callback_context.triggered_id
    use_prophet = triggered == "sa-run-prophet"

    message_text = ""
    source_tag = "prophet" if use_prophet else "ewma"
    try:
        res = _smoothing_core(df, window, threshold, prophet_order if use_prophet else None)
        message_text = f"Smoothing complete ({'Prophet' if use_prophet else 'EWMA'})."
    except Exception as exc:
        if use_prophet:
            # Try a graceful fallback to EWMA if Prophet init fails
            try:
                res = _smoothing_core(df, window, threshold, None)
                message_text = f"Prophet smoothing failed ({exc}). Fell back to EWMA."
                use_prophet = False
                source_tag = "ewma-fallback"
            except Exception as exc2:
                return (
                    f"Smoothing failed (Prophet): {exc}; fallback error: {exc2}",
                    True,
                    _empty_fig("Error"),
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    _empty_fig(),
                    _empty_fig(),
                    None,
                    phase_store,
                )
        else:
            return (
                f"Smoothing failed: {exc}",
                True,
                _empty_fig("Error"),
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                _empty_fig(),
                _empty_fig(),
                None,
                phase_store,
            )

    smoothed_tbl, anomalies_tbl, payload = _format_smoothing_payload(res, source_tag)

    phase_data = _normalize_phase_store(phase_store)
    phase_data["phase1"] = json.dumps(_phase1_compact_from_results(payload))
    phase_data["phase1_meta"] = {"source": payload["source"], "ts": pd.Timestamp.utcnow().isoformat()}

    return (
        message_text or f"Smoothing complete ({'Prophet' if use_prophet else 'EWMA'}).",
        True,
        res["fig_series"],
        anomalies_tbl.to_dict("records"),
        _cols(anomalies_tbl),
        smoothed_tbl.to_dict("records"),
        _cols(smoothed_tbl),
        res["ratio"].to_dict("records"),
        _cols(res["ratio"]),
        res["capped"].to_dict("records"),
        _cols(res["capped"]),
        res["fig_ratio1"],
        res["fig_ratio2"],
        json.dumps(payload),
        phase_data,
    )


@app.callback(
    Output("sa-alert", "children", allow_duplicate=True),
    Output("sa-alert", "is_open", allow_duplicate=True),
    Output("sa-smooth-chart", "figure", allow_duplicate=True),
    Output("sa-anomaly-table", "data", allow_duplicate=True),
    Output("sa-anomaly-table", "columns", allow_duplicate=True),
    Output("sa-smooth-table", "data", allow_duplicate=True),
    Output("sa-smooth-table", "columns", allow_duplicate=True),
    Output("sa-ratio-table", "data", allow_duplicate=True),
    Output("sa-ratio-table", "columns", allow_duplicate=True),
    Output("sa-seasonality-table", "data", allow_duplicate=True),
    Output("sa-seasonality-table", "columns", allow_duplicate=True),
    Output("sa-norm1-chart", "figure", allow_duplicate=True),
    Output("sa-norm2-chart", "figure", allow_duplicate=True),
    Output("sa-results-store", "data", allow_duplicate=True),
    Output("sa-window", "value"),
    Output("sa-threshold", "value"),
    Output("sa-prophet-order", "value"),
    Input("sa-upload", "contents", allow_optional=True),
    State("sa-raw-store", "data"),
    State("forecast-phase-store", "data"),
    State("sa-results-store", "data"),
    prevent_initial_call=True,
)
def _load_auto_smoothing(_contents, raw_json, phase_store, current_results):
    if raw_json:
        raise dash.exceptions.PreventUpdate
    if current_results:
        raise dash.exceptions.PreventUpdate
    if not phase_store:
        raise dash.exceptions.PreventUpdate
    phase_payload = _normalize_phase_store(phase_store)
    if not phase_payload:
        raise dash.exceptions.PreventUpdate
    auto_payload = phase_payload.get("auto_smoothing") if isinstance(phase_payload, dict) else None
    if not auto_payload:
        raise dash.exceptions.PreventUpdate
    if isinstance(auto_payload, str):
        try:
            auto_payload = json.loads(auto_payload)
        except Exception:
            raise dash.exceptions.PreventUpdate

    fig_series, fig_ratio1, fig_ratio2, smoothed_tbl, anomalies_tbl, ratio_df, capped_df = _render_smoothing_payload(
        auto_payload
    )
    meta = auto_payload.get("auto_meta", {}) if isinstance(auto_payload, dict) else {}
    msg = "Auto smoothing loaded from Volume Summary. Adjust settings and rerun if needed."
    if meta:
        msg = (
            f"Auto smoothing loaded (EWMA span {meta.get('window')}, z {meta.get('threshold')}). "
            "Adjust settings and rerun if needed."
        )

    window_val = meta.get("window") if meta else no_update
    threshold_val = meta.get("threshold") if meta else no_update
    prophet_val = meta.get("prophet_order") if meta else no_update

    return (
        msg,
        True,
        fig_series,
        anomalies_tbl.to_dict("records"),
        _cols(anomalies_tbl),
        smoothed_tbl.to_dict("records"),
        _cols(smoothed_tbl),
        ratio_df.to_dict("records"),
        _cols(ratio_df),
        capped_df.to_dict("records"),
        _cols(capped_df),
        fig_ratio1,
        fig_ratio2,
        json.dumps(auto_payload),
        window_val,
        threshold_val,
        prophet_val,
    )


@app.callback(
    Output("sa-norm2-chart", "figure", allow_duplicate=True),
    Output("sa-seasonality-store", "data", allow_duplicate=True),
    Input("sa-seasonality-table", "data"),
    prevent_initial_call=True,
)
def _edit_seasonality(rows):
    if not rows:
        raise dash.exceptions.PreventUpdate
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col != "Year":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return _ratio_fig(df, "Normalized Ratio 2 (edited)"), df.to_json(date_format="iso", orient="split")


@app.callback(
    Output("sa-download-smoothed", "data"),
    Input("sa-download-btn", "n_clicks"),
    State("sa-results-store", "data"),
    prevent_initial_call=True,
)
def _download_smoothing(n, data_json):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not data_json:
        return dash.no_update
    try:
        payload = json.loads(data_json)
    except Exception:
        return dash.no_update

    smoothed = pd.DataFrame(payload.get("smoothed", []))
    seasonality = pd.DataFrame(payload.get("capped", []))

    def _bundle():
        buf = io.StringIO()
        buf.write("### Smoothed Series\n")
        smoothed.to_csv(buf, index=False)
        buf.write("\n\n### Seasonality (capped)\n")
        seasonality.to_csv(buf, index=False)
        return buf.getvalue()

    return dcc.send_string(_bundle, "smoothing_results.txt")


@app.callback(
    Output("sa-save-status", "children"),
    Input("sa-save-btn", "n_clicks"),
    State("sa-results-store", "data"),
    prevent_initial_call=True,
)
def _save_smoothing(n, data_json):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not data_json:
        return "Run smoothing first."
    try:
        payload = json.loads(data_json)
    except Exception as exc:
        return f"Could not parse cached data: {exc}"
    outdir = Path(__file__).resolve().parent.parent / "exports"
    outdir.mkdir(exist_ok=True)
    smoothed = pd.DataFrame(payload.get("smoothed", []))
    seasonality = pd.DataFrame(payload.get("capped", []))
    smoothed_path = outdir / "smoothing_smoothed.csv"
    seasonality_path = outdir / "smoothing_seasonality.csv"
    smoothed.to_csv(smoothed_path, index=False)
    seasonality.to_csv(seasonality_path, index=False)
    return f"Saved to {smoothed_path} and {seasonality_path}."


@app.callback(
    Output("sa-phase-status", "children"),
    Output("forecast-phase-store", "data", allow_duplicate=True),
    Input("sa-send-phase2", "n_clicks"),
    State("sa-results-store", "data"),
    State("forecast-phase-store", "data"),
    prevent_initial_call=True,
)
def _send_phase2(n, data_json, phase_store):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not data_json:
        return "Nothing to stage yet.", phase_store
    payload = _normalize_phase_store(phase_store)
    try:
        base = json.loads(data_json) if isinstance(data_json, str) else data_json
    except Exception:
        base = {}
    payload["phase1"] = json.dumps(_phase1_compact_from_results(base))
    payload["phase1_meta"] = {"ts": pd.Timestamp.utcnow().isoformat(), "source": "manual-stage"}
    return "Phase 1 staged for Phase 2 forecasting.", payload


def _config_fields():
    return {
        "prophet": {
            "changepoint_prior_scale": "fc-prophet-cps",
            "seasonality_prior_scale": "fc-prophet-sps",
            "holidays_prior_scale": "fc-prophet-hps",
            "yearly_fourier_order": "fc-prophet-fourier",
            "use_holidays": "fc-prophet-holidays",
            "use_iq_value_scaled": "fc-prophet-iq",
        },
        "random_forest": {
            "n_estimators": "fc-rf-n",
            "max_depth": "fc-rf-depth",
            "use_holidays": "fc-rf-holidays",
            "use_iq_value_scaled": "fc-rf-iq",
        },
        "xgboost": {
            "n_estimators": "fc-xgb-n",
            "learning_rate": "fc-xgb-lr",
            "max_depth": "fc-xgb-depth",
            "use_holidays": "fc-xgb-holidays",
            "use_iq_value_scaled": "fc-xgb-iq",
        },
        "var": {
            "lags": "fc-var-lags",
            "use_holidays": "fc-var-holidays",
            "use_iq_value_scaled": "fc-var-iq",
        },
        "sarimax": {
            "order": "fc-sarimax-order",
            "seasonal_order": "fc-sarimax-seasonal",
            "use_holidays": "fc-sarimax-holidays",
            "use_iq_value_scaled": "fc-sarimax-iq",
        },
        "general": {
            "use_seasonality": "fc-general-seasonality",
        },
    }


@app.callback(
    Output("fc-config-store", "data", allow_duplicate=True),
    Output("fc-config-status", "children"),
    Output("fc-config-status", "is_open"),
    Input("fc-config-loader", "n_intervals"),
    Input("fc-config-save", "n_clicks"),
    Input("fc-config-reset", "n_clicks"),
    State("fc-prophet-cps", "value"),
    State("fc-prophet-sps", "value"),
    State("fc-prophet-hps", "value"),
    State("fc-prophet-fourier", "value"),
    State("fc-prophet-holidays", "value"),
    State("fc-prophet-iq", "value"),
    State("fc-rf-n", "value"),
    State("fc-rf-depth", "value"),
    State("fc-rf-holidays", "value"),
    State("fc-rf-iq", "value"),
    State("fc-xgb-n", "value"),
    State("fc-xgb-lr", "value"),
    State("fc-xgb-depth", "value"),
    State("fc-xgb-holidays", "value"),
    State("fc-xgb-iq", "value"),
    State("fc-var-lags", "value"),
    State("fc-var-holidays", "value"),
    State("fc-var-iq", "value"),
    State("fc-sarimax-order", "value"),
    State("fc-sarimax-seasonal", "value"),
    State("fc-sarimax-holidays", "value"),
    State("fc-sarimax-iq", "value"),
    State("fc-general-seasonality", "value"),
    prevent_initial_call=True,
)
def _config_save_load(
    loader,
    save_click,
    reset_click,
    p_cps,
    p_sps,
    p_hps,
    p_fourier,
    p_holidays,
    p_iq,
    rf_n,
    rf_depth,
    rf_holidays,
    rf_iq,
    xgb_n,
    xgb_lr,
    xgb_depth,
    xgb_holidays,
    xgb_iq,
    var_lags,
    var_holidays,
    var_iq,
    sar_order,
    sar_seasonal,
    sar_holidays,
    sar_iq,
    gen_season,
):
    triggered = dash.callback_context.triggered_id
    cfg = config_manager.load_config()
    status = ""
    if triggered == "fc-config-reset":
        cfg = config_manager.reset_to_default()
        status = "Config reset to defaults."
    elif triggered == "fc-config-save":
        try:
            cfg["prophet"].update(
                {
                    "changepoint_prior_scale": float(p_cps or cfg["prophet"]["changepoint_prior_scale"]),
                    "seasonality_prior_scale": float(p_sps or cfg["prophet"]["seasonality_prior_scale"]),
                    "holidays_prior_scale": float(p_hps or cfg["prophet"]["holidays_prior_scale"]),
                    "yearly_fourier_order": int(
                        p_fourier
                        or cfg["prophet"].get("yearly_fourier_order")
                        or cfg["prophet"].get("monthly_fourier_order", 5)
                    ),
                    "use_holidays": bool(p_holidays),
                    "use_iq_value_scaled": bool(p_iq),
                }
            )
            cfg["random_forest"].update(
                {
                    "n_estimators": int(rf_n or cfg["random_forest"]["n_estimators"]),
                    "max_depth": int(rf_depth or cfg["random_forest"]["max_depth"]),
                    "use_holidays": bool(rf_holidays),
                    "use_iq_value_scaled": bool(rf_iq),
                }
            )
            cfg["xgboost"].update(
                {
                    "n_estimators": int(xgb_n or cfg["xgboost"]["n_estimators"]),
                    "learning_rate": float(xgb_lr or cfg["xgboost"]["learning_rate"]),
                    "max_depth": int(xgb_depth or cfg["xgboost"]["max_depth"]),
                    "use_holidays": bool(xgb_holidays),
                    "use_iq_value_scaled": bool(xgb_iq),
                }
            )
            cfg["var"].update(
                {
                    "lags": int(var_lags or cfg["var"]["lags"]),
                    "use_holidays": bool(var_holidays),
                    "use_iq_value_scaled": bool(var_iq),
                }
            )
            cfg["sarimax"].update(
                {
                    "order": json.loads(sar_order) if sar_order else cfg["sarimax"]["order"],
                    "seasonal_order": json.loads(sar_seasonal) if sar_seasonal else cfg["sarimax"]["seasonal_order"],
                    "use_holidays": bool(sar_holidays),
                    "use_iq_value_scaled": bool(sar_iq),
                }
            )
            cfg["general"].update({"use_seasonality": bool(gen_season)})
            config_manager.save_config(cfg)
            status = "Config saved."
        except Exception as exc:
            status = f"Save failed: {exc}"
    return cfg, status, bool(status)


@app.callback(
    Output("fc-prophet-cps", "value"),
    Output("fc-prophet-sps", "value"),
    Output("fc-prophet-hps", "value"),
    Output("fc-prophet-fourier", "value"),
    Output("fc-prophet-holidays", "value"),
    Output("fc-prophet-iq", "value"),
    Output("fc-rf-n", "value"),
    Output("fc-rf-depth", "value"),
    Output("fc-rf-holidays", "value"),
    Output("fc-rf-iq", "value"),
    Output("fc-xgb-n", "value"),
    Output("fc-xgb-lr", "value"),
    Output("fc-xgb-depth", "value"),
    Output("fc-xgb-holidays", "value"),
    Output("fc-xgb-iq", "value"),
    Output("fc-var-lags", "value"),
    Output("fc-var-holidays", "value"),
    Output("fc-var-iq", "value"),
    Output("fc-sarimax-order", "value"),
    Output("fc-sarimax-seasonal", "value"),
    Output("fc-sarimax-holidays", "value"),
    Output("fc-sarimax-iq", "value"),
    Output("fc-general-seasonality", "value"),
    Input("fc-config-store", "data"),
    prevent_initial_call=False,
)
def _populate_config(cfg):
    cfg = cfg or config_manager.load_config()
    p = cfg.get("prophet", {})
    rf = cfg.get("random_forest", {})
    xgb = cfg.get("xgboost", {})
    var_cfg = cfg.get("var", {})
    sar = cfg.get("sarimax", {})
    gen = cfg.get("general", {})
    return (
        p.get("changepoint_prior_scale"),
        p.get("seasonality_prior_scale"),
        p.get("holidays_prior_scale"),
        p.get("yearly_fourier_order", p.get("monthly_fourier_order")),
        p.get("use_holidays"),
        p.get("use_iq_value_scaled"),
        rf.get("n_estimators"),
        rf.get("max_depth"),
        rf.get("use_holidays"),
        rf.get("use_iq_value_scaled"),
        xgb.get("n_estimators"),
        xgb.get("learning_rate"),
        xgb.get("max_depth"),
        xgb.get("use_holidays"),
        xgb.get("use_iq_value_scaled"),
        var_cfg.get("lags"),
        var_cfg.get("use_holidays"),
        var_cfg.get("use_iq_value_scaled"),
        json.dumps(sar.get("order")),
        json.dumps(sar.get("seasonal_order")),
        sar.get("use_holidays"),
        sar.get("use_iq_value_scaled"),
        gen.get("use_seasonality"),
    )


@app.callback(
    Output("fc-alert", "children"),
    Output("fc-alert", "is_open"),
    Output("fc-combined", "data"),
    Output("fc-combined", "columns"),
    Output("fc-pivot", "data"),
    Output("fc-pivot", "columns"),
    Output("fc-errors", "children"),
    Output("fc-data-store", "data", allow_duplicate=True),
    Output("fc-line-chart", "figure"),
    Output("fc-ratio1-chart", "figure"),
    Output("fc-ratio2-chart", "figure"),
    Output("fc-accuracy-table", "data"),
    Output("fc-accuracy-table", "columns"),
    Output("fc-accuracy-chart", "figure"),
    Output("fc-phase2-meta", "children"),
    Input("fc-run-btn", "n_clicks"),
    Input("fc-load-phase1", "n_clicks"),
    Input("fc-load-saved-btn", "n_clicks"),
    State("fc-upload", "contents"),
    State("fc-upload", "filename"),
    State("fc-months", "value"),
    State("forecast-phase-store", "data"),
    State("fc-saved-selected", "data"),
    prevent_initial_call=True,
)
def _run_forecast(n, n_phase, n_saved, contents, filename, months, phase_store, saved_run_id):
    if not n and not n_phase and not n_saved:
        raise dash.exceptions.PreventUpdate
    try:
        months = int(months or 12)
    except Exception:
        months = 12

    ctx = dash.callback_context
    triggered = ctx.triggered_id if ctx.triggered_id else "fc-run-btn"
    use_phase_data = triggered == "fc-load-phase1"
    use_saved = triggered == "fc-load-saved-btn"
    df = None
    source = ""
    store_from_meta = None

    if use_saved:
        if not saved_run_id:
            raise dash.exceptions.PreventUpdate
        meta, df_saved = cap_store.load_forecast_run(int(saved_run_id))
        meta_js = {}
        try:
            meta_js = json.loads(meta.get("metadata_json") or "{}") if meta else {}
        except Exception:
            meta_js = {}
        store_from_meta = meta_js.get("fc_store")
        df = df_saved if isinstance(df_saved, pd.DataFrame) else None
        source = f"saved-run-{saved_run_id}"

    if contents and filename and not use_phase_data and not use_saved:
        try:
            df, msg = _parse_upload(contents, filename)
            source = f"upload:{filename}"
        except Exception as exc:
            return (
                f"Upload failed: {exc}",
                True,
                [],
                [],
                [],
                [],
                "",
                None,
                _empty_fig(),
                _empty_fig(),
                _empty_fig(),
                [],
                [],
                _empty_fig(),
                "",
            )
    if (df is None or df.empty) and phase_store and not use_saved:
        phase_payload = _normalize_phase_store(phase_store)
        phase1_json = phase_payload.get("phase1") if isinstance(phase_payload, dict) else None
        if phase1_json:
            try:
                phase1_payload = json.loads(phase1_json) if isinstance(phase1_json, str) else phase1_json
                smoothed_records = phase1_payload.get("smoothed", [])
                df = pd.DataFrame(smoothed_records)
                if "Final_Smoothed_Value" not in df.columns and "smoothed" in df.columns:
                    df["Final_Smoothed_Value"] = df["smoothed"]
                source = "phase1-staged"
            except Exception:
                df = None

    if df is None or df.empty:
        return (
            "Upload smoothed data (or stage Phase 1) to run the forecast.",
            True,
            [],
            [],
            [],
            [],
            "",
            None,
            _empty_fig("No data"),
            _empty_fig(),
            _empty_fig(),
            [],
            [],
            _empty_fig(),
            "",
        )

    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    if "Final_Smoothed_Value" not in df.columns and "y" in df.columns:
        df["Final_Smoothed_Value"] = df["y"]

    try:
        if use_saved and store_from_meta:
            res = json.loads(store_from_meta) if isinstance(store_from_meta, str) else store_from_meta
        elif use_saved:
            res = {"combined": df.to_dict("records") if isinstance(df, pd.DataFrame) else []}
        else:
            res = run_phase2_forecast(df, months)

        forecast_results = res.get("forecast_results", {}) if not use_saved else {}
        if not use_saved and forecast_results:
            combined, wide, pivot_smoothed = process_forecast_results(forecast_results)
        else:
            combined = pd.DataFrame(res.get("combined", []))
            wide = pd.DataFrame(res.get("wide", []))
            pivot_smoothed = pd.DataFrame(res.get("pivot_smoothed", []))

        errors = res.get("errors", {})
        err_txt = "; ".join([f"{k}: {v}" for k, v in errors.items()]) if errors else ""
        line_fig = _empty_fig("No forecast data")
        if not combined.empty:
            # Normalize Month/Month_Year for plotting
            plot_df = combined.copy()
            if "Month_Year" in plot_df.columns:
                try:
                    plot_df["Month_Parsed"] = pd.to_datetime("1 " + plot_df["Month_Year"].astype(str), errors="coerce")
                except Exception:
                    plot_df["Month_Parsed"] = pd.NaT
            elif "Month" in plot_df.columns:
                try:
                    plot_df["Month_Parsed"] = pd.to_datetime(plot_df["Month"], errors="coerce")
                except Exception:
                    plot_df["Month_Parsed"] = pd.NaT
            else:
                plot_df["Month_Parsed"] = pd.NaT

            line_fig = px.line(
                plot_df,
                x="Month_Parsed",
                y="Forecast",
                color="Model" if "Model" in plot_df.columns else None,
                markers=True,
                title="Forecast by model",
            )
            line_fig.update_layout(xaxis_title="Month")

            if not pivot_smoothed.empty:
                baseline_long = pivot_smoothed.melt(id_vars="Year", var_name="Month", value_name="Final_Smoothed_Value")
                baseline_long["Month_Parsed"] = pd.to_datetime(
                    "1 " + baseline_long["Month"].astype(str) + " " + baseline_long["Year"].astype(str),
                    errors="coerce",
                )
                line_fig.add_scatter(
                    x=baseline_long["Month_Parsed"],
                    y=baseline_long["Final_Smoothed_Value"],
                    mode="lines+markers",
                    name="Final Smoothed",
                    line=dict(color="gray", dash="dot"),
                )

        ratio_fig1, ratio_fig2 = _empty_fig("No ratio data"), _empty_fig("No ratio data")
        ratio_df = pd.DataFrame(res.get("ratio", []))
        capped_df = pd.DataFrame(res.get("capped", []))
        if not pivot_smoothed.empty and (ratio_df.empty or capped_df.empty):
            try:
                _raw_fig, capped_df, ratio_df = plot_contact_ratio_seasonality(pivot_smoothed)
            except Exception:
                pass
        if not ratio_df.empty:
            ratio_fig1 = _ratio_fig(ratio_df, "Normalized Ratio 1")
        if not capped_df.empty:
            ratio_fig2 = _ratio_fig(capped_df, "Normalized Ratio 2 (capped)")

        accuracy_tbl = pd.DataFrame(res.get("accuracy", []))
        accuracy_fig = _empty_fig("Accuracy not available")
        if accuracy_tbl.empty and not pivot_smoothed.empty and not wide.empty:
            try:
                wide_with_base = fill_final_smoothed_row(wide.copy(), pivot_smoothed)
                accuracy_tbl = accuracy_phase1(wide_with_base, pivot_smoothed)
            except Exception:
                accuracy_tbl = pd.DataFrame()
        if not accuracy_tbl.empty:
            plot_df = accuracy_tbl.copy()
            for col in ["Accuracy(+−5%)", "Accuracy(+−7%)", "Accuracy(+−10%)"]:
                if col in plot_df.columns:
                    plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
            accuracy_fig = px.bar(
                plot_df,
                x="Model",
                y=[c for c in ["Accuracy(+−5%)", "Accuracy(+−7%)", "Accuracy(+−10%)"] if c in plot_df.columns],
                barmode="group",
                title="Accuracy by model (%)",
            )

        store = res | {
            "combined": combined.to_dict("records"),
            "wide": wide.to_dict("records"),
            "pivot_smoothed": pivot_smoothed.to_dict("records"),
            "accuracy": accuracy_tbl.to_dict("records") if isinstance(accuracy_tbl, pd.DataFrame) else [],
            "ratio": ratio_df.to_dict("records") if isinstance(ratio_df, pd.DataFrame) else [],
            "capped": capped_df.to_dict("records") if isinstance(capped_df, pd.DataFrame) else [],
        }
        model_names = [m for m in forecast_results.keys() if m not in ("final_smoothed_values",)] if forecast_results else []
        meta_txt = ""
        if use_saved:
            meta_txt = f"Loaded saved run #{saved_run_id}"
        else:
            meta_txt = f"Phase 2 run ({source or 'upload'}) | horizon {months} months"
            if model_names:
                meta_txt = f"{meta_txt} | models: {', '.join(model_names)}"
        return (
            "Forecast complete." if not use_saved else f"Loaded saved run #{saved_run_id}",
            True,
            combined.to_dict("records"),
            [{"name": c, "id": c} for c in combined.columns],
            wide.to_dict("records"),
            [{"name": c, "id": c} for c in wide.columns],
            err_txt,
            json.dumps(store),
            line_fig,
            ratio_fig1,
            ratio_fig2,
            accuracy_tbl.to_dict("records") if isinstance(accuracy_tbl, pd.DataFrame) else [],
            _cols(accuracy_tbl) if isinstance(accuracy_tbl, pd.DataFrame) else [],
            accuracy_fig,
            meta_txt,
        )
    except Exception as exc:
        return (
            f"Forecast failed: {exc}",
            True,
            [],
            [],
            [],
            [],
            str(exc),
            None,
            _empty_fig("Error"),
            _empty_fig(),
            _empty_fig(),
            [],
            [],
            _empty_fig(),
            "",
        )


@app.callback(
    Output("fc-download-forecast", "data"),
    Input("fc-download-btn", "n_clicks"),
    State("fc-data-store", "data"),
    prevent_initial_call=True,
)
def _download_forecast_results(n, store_json):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not store_json:
        return dash.no_update
    try:
        payload = json.loads(store_json)
    except Exception:
        return dash.no_update
    combined = pd.DataFrame(payload.get("combined", []))
    if combined.empty:
        return dash.no_update
    return dcc.send_data_frame(combined.to_csv, "forecast_results.csv", index=False)


@app.callback(
    Output("fc-download-config", "data"),
    Input("fc-download-config-btn", "n_clicks"),
    State("fc-data-store", "data"),
    prevent_initial_call=True,
)
def _download_forecast_config(n, store_json):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not store_json:
        return dash.no_update
    try:
        payload = json.loads(store_json)
    except Exception:
        return dash.no_update
    cfg = payload.get("config", {})
    combined = pd.DataFrame(payload.get("combined", []))
    txt = create_download_csv_with_metadata(combined, cfg)
    return dcc.send_string(lambda: txt, "forecast_with_config.txt")


@app.callback(
    Output("fc-save-status", "children"),
    Input("fc-save-btn", "n_clicks"),
    State("fc-data-store", "data"),
    prevent_initial_call=True,
)
def _save_forecast_to_disk(n, store_json):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not store_json:
        return "Run a forecast first."
    try:
        payload = json.loads(store_json)
    except Exception as exc:
        return f"Could not read cached forecast: {exc}"
    outdir = Path(__file__).resolve().parent.parent / "exports"
    outdir.mkdir(exist_ok=True)
    combined = pd.DataFrame(payload.get("combined", []))
    accuracy = pd.DataFrame(payload.get("accuracy", []))
    combined_path = outdir / "forecast_results.csv"
    accuracy_path = outdir / "forecast_accuracy.csv"
    combined.to_csv(combined_path, index=False)
    accuracy.to_csv(accuracy_path, index=False)
    return f"Saved to {combined_path} and {accuracy_path}."


@app.callback(
    Output("fc-save-db-status", "children"),
    Input("fc-save-db-btn", "n_clicks"),
    State("fc-data-store", "data"),
    State("fc-phase2-meta", "children"),
    prevent_initial_call=True,
)
def _save_forecast_to_db(n, store_json, meta_text):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not store_json:
        return "Run a forecast first."
    try:
        payload = json.loads(store_json)
    except Exception as exc:
        return f"Could not read cached forecast: {exc}"
    combined = pd.DataFrame(payload.get("combined", []))
    if combined.empty:
        return "No forecast data to save."
    try:
        user = current_user_fallback() or "unknown"
    except Exception:
        user = "unknown"
    scope_key = "forecast|workspace|global|"
    metadata = {"fc_store": store_json, "meta": meta_text}
    try:
        run_id = cap_store.record_forecast_run(
            scope_key=scope_key,
            forecast_df=combined,
            created_by=user,
            model_name="workspace",
            metadata=metadata,
            pushed_to_planning=False,
        )
        return f"Saved forecast run #{run_id} by {user}."
    except Exception as exc:
        return f"DB save failed: {exc}"


@app.callback(
    Output("fc-saved-table", "data"),
    Output("fc-saved-table", "columns"),
    Output("fc-saved-list-store", "data", allow_duplicate=True),
    Output("fc-saved-status", "children"),
    Input("fc-refresh-saved", "n_clicks"),
    prevent_initial_call=True,
)
def _refresh_saved_runs(n):
    if not n:
        raise dash.exceptions.PreventUpdate
    try:
        df = cap_store.list_forecast_runs(limit=50)
    except Exception as exc:
        return [], [], None, f"Could not load saved forecasts: {exc}"
    data = df.to_dict("records")
    cols = _cols(df) if not df.empty else []
    return data, cols, df.to_json(date_format="iso", orient="split"), f"Loaded {len(df)} saved forecasts."


@app.callback(
    Output("fc-saved-selected", "data", allow_duplicate=True),
    Input("fc-saved-table", "active_cell"),
    State("fc-saved-table", "data"),
    prevent_initial_call=True,
)
def _select_saved_run(cell, rows):
    if not cell or not rows:
        raise dash.exceptions.PreventUpdate
    try:
        row = rows[cell["row"]]
        return row.get("id")
    except Exception:
        raise dash.exceptions.PreventUpdate


# ---------------------------------------------------------------------------
# Transformation / Projects (Phase 2+ adjustments)
# ---------------------------------------------------------------------------


def _load_latest_forecast_file() -> tuple[pd.DataFrame, str]:
    txt_path = Path(__file__).resolve().parent.parent / "latest_forecast_full_path.txt"
    if txt_path.exists():
        try:
            file_path = txt_path.read_text().strip()
            if file_path and Path(file_path).exists():
                df = pd.read_csv(file_path)
                return df, f"Loaded {Path(file_path).name}."
            return pd.DataFrame(), f"File not found: {file_path}"
        except Exception as exc:
            return pd.DataFrame(), f"Could not load latest forecast: {exc}"
    return pd.DataFrame(), "No latest forecast path found."


def _options_from_df(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return []
    vals = sorted(pd.unique(df[col].dropna()).tolist())
    return [{"label": str(v), "value": v} for v in vals]


@app.callback(
    Output("tp-load-status", "children"),
    Output("tp-raw-store", "data", allow_duplicate=True),
    Output("tp-group", "options"),
    Output("tp-model", "options"),
    Output("tp-year", "options"),
    Output("tp-raw-table", "data"),
    Output("tp-raw-table", "columns"),
    Input("tp-load-latest", "n_clicks"),
    Input("tp-use-phase", "n_clicks"),
    State("fc-data-store", "data"),
    prevent_initial_call=True,
)
def _tp_load_source(n_file, n_phase, fc_store_json):
    if not n_file and not n_phase:
        raise dash.exceptions.PreventUpdate
    ctx = dash.callback_context
    trig = ctx.triggered_id if ctx.triggered_id else "tp-load-latest"

    df = pd.DataFrame()
    msg = ""
    if trig == "tp-load-latest":
        df, msg = _load_latest_forecast_file()
    elif trig == "tp-use-phase":
        if fc_store_json:
            try:
                payload = json.loads(fc_store_json)
                combined = pd.DataFrame(payload.get("combined", []))
                pivot = pd.DataFrame(payload.get("pivot_smoothed", []))
                if not combined.empty:
                    # attempt to reconstruct Month_Year and Year/Month
                    combined["Month_Year"] = pd.to_datetime(combined["Month"]).dt.strftime("%b-%y")
                    combined["Year"] = pd.to_datetime(combined["Month"]).dt.year
                    combined["Month"] = pd.to_datetime(combined["Month"]).dt.strftime("%b")
                    combined.rename(columns={"Forecast": "Base_Forecast_for_Forecast_Group"}, inplace=True)
                    df = combined
                    msg = "Loaded from Phase 2 results."
                elif not pivot.empty:
                    pivot = pivot.copy()
                    df = pivot
                    msg = "Loaded pivot smoothed from Phase 2."
                else:
                    msg = "No Phase 2 data available."
            except Exception as exc:
                msg = f"Could not parse Phase 2 data: {exc}"
        else:
            msg = "Phase 2 results not found."

    opts_group = _options_from_df(df, "forecast_group")
    opts_model = _options_from_df(df, "Model")
    opts_year = _options_from_df(df, "Year")
    preview = df.head(200)
    return msg, df.to_json(date_format="iso", orient="split"), opts_group, opts_model, opts_year, preview.to_dict("records"), _cols(preview)


@app.callback(
    Output("tp-selection-status", "children"),
    Output("tp-selection-status", "is_open"),
    Output("tp-filtered-table", "data"),
    Output("tp-filtered-table", "columns"),
    Output("tp-transform-table", "data"),
    Output("tp-transform-table", "columns"),
    Output("tp-filtered-store", "data", allow_duplicate=True),
    Input("tp-apply-selection", "n_clicks"),
    State("tp-raw-store", "data"),
    State("tp-group", "value"),
    State("tp-model", "value"),
    State("tp-year", "value"),
    prevent_initial_call=True,
)
def _tp_apply_selection(n, raw_json, group, model, year):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not raw_json:
        return "Load data first.", True, [], [], [], [], None
    try:
        df = pd.read_json(io.StringIO(raw_json), orient="split")
    except Exception:
        return "Could not read loaded data.", True, [], [], [], [], None

    filtered = df.copy()
    if group and "forecast_group" in filtered.columns:
        filtered = filtered[filtered["forecast_group"] == group]
    if model and "Model" in filtered.columns:
        filtered = filtered[filtered["Model"] == model]
    if year and "Year" in filtered.columns:
        filtered = filtered[filtered["Year"] >= year]
    if filtered.empty:
        return "No data for that selection.", True, [], [], [], [], None

    # ensure Month_Year present
    if "Month_Year" not in filtered.columns and "Month" in filtered.columns and "Year" in filtered.columns:
        filtered["Month_Year"] = filtered.apply(
            lambda r: f"{str(r['Month'])[:3]}-{str(r['Year'])[-2:]}", axis=1
        )

    filtered = filtered.sort_values(["Year", "Month_Year"]) if "Year" in filtered.columns else filtered

    forecast_notice = ""
    if "Month_Year" in filtered.columns:
        try:
            month_year = pd.to_datetime(
                filtered["Month_Year"].astype(str),
                format="%b-%y",
                errors="coerce",
            )
            valid = month_year.dropna()
            if not valid.empty:
                start = valid.min().to_period("M").to_timestamp()
                end = valid.max().to_period("M").to_timestamp()
                forecast_dates = pd.date_range(start=start, end=end, freq="MS")
                csv_content = "Forecast_Dates\n" + "\n".join(
                    forecast_dates.strftime("%Y-%m-%d").tolist()
                )
                out_path = Path(__file__).resolve().parent.parent / "forecast_dates.csv"
                out_path.write_text(csv_content)
                forecast_notice = (
                    f" Forecast dates saved: {forecast_dates[0]:%Y-%m-%d}"
                    f" to {forecast_dates[-1]:%Y-%m-%d} ({len(forecast_dates)} months)."
                )
        except Exception as exc:
            forecast_notice = f" Could not write forecast_dates.csv: {exc}"

    # add transformation columns
    display_cols = ["Month_Year", "Year", "Month", "Base_Forecast_for_Forecast_Group"]
    base_cols = [c for c in display_cols if c in filtered.columns]
    transform_df = filtered[base_cols].copy()
    for col in _TRANSFORMATION_COLS:
        transform_df[col] = ""

    locked_cols = {"Month_Year", "Year", "Month", "Base_Forecast_for_Forecast_Group"}
    transform_cols = _cols(transform_df)
    for col in transform_cols:
        col["editable"] = col.get("id") not in locked_cols

    message = f"Loaded {len(filtered)} rows for {group or 'All'} | {model or 'All'} | {year or 'All'}+."
    if forecast_notice:
        message = message.rstrip(".") + f".{forecast_notice}"

    return (
        message,
        True,
        filtered.head(200).to_dict("records"),
        _cols(filtered.head(200)),
        transform_df.to_dict("records"),
        transform_cols,
        filtered.to_json(date_format="iso", orient="split"),
    )


@app.callback(
    Output("tp-transform-status", "children"),
    Output("tp-final-store", "data", allow_duplicate=True),
    Output("tp-transposed-table", "data"),
    Output("tp-transposed-table", "columns"),
    Output("tp-final-forecast-table", "data"),
    Output("tp-final-forecast-table", "columns"),
    Output("tp-summary-json", "children"),
    Input("tp-apply-transform", "n_clicks"),
    State("tp-transform-table", "data"),
    State("tp-filtered-store", "data"),
    prevent_initial_call=True,
)
def _tp_apply_transform(n, edited_rows, filtered_json):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not edited_rows or not filtered_json:
        return "Nothing to transform yet.", None, [], [], [], [], ""
    try:
        base_filtered = pd.read_json(io.StringIO(filtered_json), orient="split")
    except Exception:
        base_filtered = pd.DataFrame()
    edited_df = pd.DataFrame(edited_rows)
    full_df = base_filtered.copy()
    for col in edited_df.columns:
        if col in full_df.columns:
            # keep base cols as-is
            continue
        full_df[col] = edited_df[col]

    processed = _apply_transformations(full_df)

    # transposed view
    transpose_cols = [
        "Month_Year",
        "Base_Forecast_for_Forecast_Group",
        "Transformation 1",
        "Remarks_Tr 1",
        "Forecast_Transformation 1",
        "Transformation 2",
        "Remarks_Tr 2",
        "Forecast_Transformation 2",
        "Transformation 3",
        "Remarks_Tr 3",
        "Forecast_Transformation 3",
        "IA 1",
        "Remarks_IA 1",
        "Forecast_IA 1",
        "IA 2",
        "Remarks_IA 2",
        "Forecast_IA 2",
        "IA 3",
        "Remarks_IA 3",
        "Forecast_IA 3",
        "Marketing Campaign 1",
        "Remarks_Mkt 1",
        "Forecast_Marketing Campaign 1",
        "Marketing Campaign 2",
        "Remarks_Mkt 2",
        "Forecast_Marketing Campaign 2",
        "Marketing Campaign 3",
        "Remarks_Mkt 3",
        "Forecast_Marketing Campaign 3",
    ]
    available_cols = [c for c in transpose_cols if c in processed.columns]
    transposed = pd.DataFrame()
    if available_cols:
        transposed = processed[available_cols].copy()
        if "Month_Year" in transposed.columns:
            t = transposed.set_index("Month_Year").transpose().reset_index()
            t.rename(columns={"index": "Category"}, inplace=True)
            if "Forecast_Marketing Campaign 3" in processed.columns:
                final_forecast_values = processed.set_index("Month_Year")["Forecast_Marketing Campaign 3"]
                t.loc[len(t)] = ["Final Forecast"] + final_forecast_values.tolist()
            transposed = t

    # final forecast table
    final_cols = ["Month_Year", "Forecast_Marketing Campaign 3"]
    if "forecast_group" in processed.columns:
        final_cols.insert(0, "forecast_group")
    if "Model" in processed.columns:
        final_cols.insert(-1, "Model")
    if "Year" in processed.columns:
        final_cols.insert(-1, "Year")
    final_cols = [c for c in final_cols if c in processed.columns]
    final_tbl = processed[final_cols].copy()
    if "Forecast_Marketing Campaign 3" in final_tbl.columns:
        final_tbl = final_tbl.rename(columns={"Forecast_Marketing Campaign 3": "Final_Forecast"})

    # summary
    summary = {}
    try:
        summary = {
            "Forecast Group": processed["forecast_group"].iloc[0] if "forecast_group" in processed.columns else None,
            "Model": processed["Model"].iloc[0] if "Model" in processed.columns else None,
            "Selected Year": processed["Year"].min() if "Year" in processed.columns else None,
            "Years Included": sorted(processed["Year"].unique().tolist()) if "Year" in processed.columns else [],
            "Total Rows": len(processed),
            "Base Forecast Total": float(processed["Base_Forecast_for_Forecast_Group"].sum()) if "Base_Forecast_for_Forecast_Group" in processed.columns else None,
            "Final Forecast Total": float(processed["Final_Forecast"].sum()) if "Final_Forecast" in processed.columns else None,
        }
    except Exception:
        summary = {}

    return (
        "Transformations applied.",
        processed.to_json(date_format="iso", orient="split"),
        transposed.to_dict("records"),
        _cols(transposed),
        final_tbl.to_dict("records"),
        _cols(final_tbl),
        json.dumps(summary, indent=2),
    )


@app.callback(
    Output("tp-download-final", "data"),
    Input("tp-download-final-btn", "n_clicks"),
    State("tp-final-store", "data"),
    prevent_initial_call=True,
)
def _tp_download_final(n, final_json):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not final_json:
        return dash.no_update
    try:
        df = pd.read_json(io.StringIO(final_json), orient="split")
    except Exception:
        return dash.no_update
    final_col = None
    if "Final_Forecast" in df.columns:
        final_col = "Final_Forecast"
    elif "Forecast_Marketing Campaign 3" in df.columns:
        final_col = "Forecast_Marketing Campaign 3"
    if not final_col:
        return dash.no_update

    cols = ["Month_Year", final_col]
    if "forecast_group" in df.columns:
        cols.insert(0, "forecast_group")
    if "Model" in df.columns:
        cols.insert(-1, "Model")
    if "Year" in df.columns:
        cols.insert(-1, "Year")
    cols = [c for c in cols if c in df.columns]
    use_df = df[cols].copy()
    if final_col != "Final_Forecast":
        use_df = use_df.rename(columns={final_col: "Final_Forecast"})

    filename = "final_forecast.csv"
    try:
        grp = str(use_df["forecast_group"].iloc[0]) if "forecast_group" in use_df.columns else ""
        mod = str(use_df["Model"].iloc[0]) if "Model" in use_df.columns else ""
        yr = str(use_df["Year"].iloc[0]) if "Year" in use_df.columns else ""
        parts = [p for p in [grp, mod, yr] if p]
        if parts:
            safe = "_".join(p.replace(" ", "_") for p in parts)
            filename = f"final_forecast_{safe}.csv"
    except Exception:
        filename = "final_forecast.csv"

    return dcc.send_data_frame(use_df.to_csv, filename, index=False)


@app.callback(
    Output("tp-download-full", "data"),
    Input("tp-download-full-btn", "n_clicks"),
    State("tp-final-store", "data"),
    prevent_initial_call=True,
)
def _tp_download_full(n, final_json):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not final_json:
        return dash.no_update
    try:
        df = pd.read_json(io.StringIO(final_json), orient="split")
    except Exception:
        return dash.no_update
    return dcc.send_data_frame(df.to_csv, "transformation_full.csv", index=False)


@app.callback(
    Output("tp-save-status", "children", allow_duplicate=True),
    Input("tp-save-btn", "n_clicks"),
    State("tp-final-store", "data"),
    prevent_initial_call=True,
)
def _tp_save_to_disk(n, final_json):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not final_json:
        return "Run transformations first."
    try:
        df = pd.read_json(io.StringIO(final_json), orient="split")
    except Exception as exc:
        return f"Could not read results: {exc}"

    base_dir_txt = Path(__file__).resolve().parent.parent / "latest_forecast_base_dir.txt"
    outdir = Path(__file__).resolve().parent.parent / "exports"
    outdir.mkdir(exist_ok=True)
    target_dir = outdir
    if base_dir_txt.exists():
        try:
            bd = base_dir_txt.read_text().strip()
            if bd:
                target_dir = Path(bd)
                target_dir.mkdir(exist_ok=True, parents=True)
        except Exception:
            pass

    if "Forecast_Marketing Campaign 3" in df.columns and "Final_Forecast_Post_Transformations" not in df.columns:
        df["Final_Forecast_Post_Transformations"] = df["Forecast_Marketing Campaign 3"]

    transformation_columns = [
        "Month_Year",
        "Base_Forecast_for_Forecast_Group",
        "Final_Forecast_Post_Transformations",
        "Transformation 1",
        "Remarks_Tr 1",
        "Forecast_Transformation 1",
        "Transformation 2",
        "Remarks_Tr 2",
        "Forecast_Transformation 2",
        "Transformation 3",
        "Remarks_Tr 3",
        "Forecast_Transformation 3",
        "IA 1",
        "Remarks_IA 1",
        "Forecast_IA 1",
        "IA 2",
        "Remarks_IA 2",
        "Forecast_IA 2",
        "IA 3",
        "Remarks_IA 3",
        "Forecast_IA 3",
        "Marketing Campaign 1",
        "Remarks_Mkt 1",
        "Forecast_Marketing Campaign 1",
        "Marketing Campaign 2",
        "Remarks_Mkt 2",
        "Forecast_Marketing Campaign 2",
        "Marketing Campaign 3",
        "Remarks_Mkt 3",
        "Forecast_Marketing Campaign 3",
    ]
    if "forecast_group" in df.columns:
        transformation_columns.insert(0, "forecast_group")
    available_transform_cols = [col for col in transformation_columns if col in df.columns]
    transformation_df = df[available_transform_cols].copy()

    owner = current_user_fallback() or os.getenv("USERNAME") or os.getenv("USER") or "user"
    owner_clean = re.sub(r"[^A-Za-z0-9_-]+", "_", str(owner)).strip("_") or "user"
    timestamp = pd.Timestamp.now()
    date_str = timestamp.strftime("%d_%b_%Y")
    time_str = timestamp.strftime("%H-%M-%S")
    fname = f"Monthly_Forecast_with_Adjustments_{date_str}_{time_str}_{owner_clean}.csv"
    fpath = target_dir / fname
    try:
        transformation_df.to_csv(fpath, index=False)
        return f"Saved to {fpath}"
    except Exception as exc:
        return f"Save failed: {exc}"


def _di_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _di_base_dir() -> Path:
    repo_root = _di_repo_root()
    base_dir_txt = repo_root / "latest_forecast_base_dir.txt"
    if base_dir_txt.exists():
        try:
            txt = base_dir_txt.read_text().strip()
            if txt:
                candidate = Path(txt)
                if candidate.exists():
                    return candidate
        except Exception:
            pass
    return repo_root


def _di_list_transform_files(base_dir: Path) -> list[dict]:
    files = sorted(
        base_dir.glob("Monthly_Forecast_with_Adjustments_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    options = []
    for path in files:
        try:
            ts = pd.to_datetime(path.stat().st_mtime, unit="s").strftime("%d-%b-%Y %H:%M")
        except Exception:
            ts = "unknown time"
        options.append({"label": f"{path.name} (created {ts})", "value": str(path)})
    return options


def _di_load_forecast_dates() -> tuple[pd.Series, str]:
    repo_root = _di_repo_root()
    candidates = [
        repo_root / "forecast_dates.csv",
        _di_base_dir() / "forecast_dates.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if "Forecast_Dates" not in df.columns:
                continue
            dates = pd.to_datetime(df["Forecast_Dates"], errors="coerce").dropna()
            if dates.empty:
                return pd.Series(dtype="datetime64[ns]"), f"No usable dates in {path.name}."
            return dates, f"Loaded forecast dates from {path.name}."
        except Exception as exc:
            return pd.Series(dtype="datetime64[ns]"), f"Failed to read {path.name}: {exc}"
    return pd.Series(dtype="datetime64[ns]"), "forecast_dates.csv not found. Generate it from Transformation Projects."


def _di_load_holidays() -> tuple[pd.DataFrame, str]:
    repo_root = _di_repo_root()
    candidates = [
        repo_root / "holidays_list.csv",
        _di_base_dir() / "holidays_list.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if "holiday_date" in df.columns:
                df["holiday_date"] = pd.to_datetime(df["holiday_date"], errors="coerce")
            return df, f"Loaded holidays from {path.name}."
        except Exception as exc:
            return pd.DataFrame(), f"Failed to read {path.name}: {exc}"
    return pd.DataFrame(), "holidays_list.csv not found."


def _di_parse_forecast_month(year_val: Any, month_val: Any) -> Optional[pd.Timestamp]:
    if not year_val or not month_val:
        return None
    try:
        dt = pd.to_datetime(f"{month_val} {year_val}", errors="coerce")
    except Exception:
        dt = pd.NaT
    if pd.isna(dt):
        return None
    return dt.replace(day=1)


def _di_normalize_original_data(df: pd.DataFrame, group_value: Optional[str]) -> pd.DataFrame:
    d = _normalize_volume_df(df)
    if d.empty:
        return d
    if group_value and "forecast_group" in d.columns:
        d = d[d["forecast_group"] == group_value]
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
    if "volume" in d.columns:
        d["volume"] = pd.to_numeric(d["volume"], errors="coerce")
    d = d.dropna(subset=["date", "volume"])
    return d


def _di_load_original_data(group_value: Optional[str]) -> tuple[pd.DataFrame, str]:
    base_dir = _di_base_dir()
    repo_root = _di_repo_root()
    candidates = [
        base_dir / "original_data.csv",
        repo_root / "original_data.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, thousands=",")
            df = _di_normalize_original_data(df, group_value)
            if df.empty:
                return df, f"original_data.csv loaded from {path.name} but no usable rows."
            return df, f"Loaded original data from {path.name}."
        except Exception as exc:
            return pd.DataFrame(), f"Failed to read {path.name}: {exc}"
    return pd.DataFrame(), "original_data.csv not found."


def _di_match_months(df_daily: pd.DataFrame, target_month: pd.Timestamp) -> list[pd.Timestamp]:
    if df_daily.empty:
        return []
    month_starts = sorted(pd.unique(df_daily["month_start"].dropna()))
    target_weekday = target_month.weekday()
    matching = [m for m in month_starts if pd.Timestamp(m).weekday() == target_weekday]
    matching = [pd.Timestamp(m) for m in matching if pd.Timestamp(m) != target_month]
    if not matching:
        matching = [pd.Timestamp(m) for m in month_starts]
    return matching[-3:]


def _di_compute_distribution(df_daily: pd.DataFrame, target_month: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    days_in_month = calendar.monthrange(target_month.year, target_month.month)[1]
    matching_months = _di_match_months(df_daily, target_month)
    msg = "No matching months found."
    if matching_months:
        msg = "Using months: " + ", ".join(m.strftime("%b %Y") for m in matching_months)

    day_vals = []
    for day in range(1, days_in_month + 1):
        values = []
        for month in matching_months:
            md = df_daily[df_daily["month_start"] == month]
            val = md.loc[md["day"] == day, "volume_pct"]
            if not val.empty:
                values.append(float(val.iloc[0]))
        avg = float(np.nanmean(values)) if values else 0.0
        day_vals.append(avg)

    total = sum(day_vals)
    if total > 0:
        day_vals = [v / total * 100 for v in day_vals]

    dates = [target_month + pd.Timedelta(days=i) for i in range(days_in_month)]
    dist = pd.DataFrame(
        {
            "Date": [d.date() for d in dates],
            "Weekday": [d.day_name() for d in dates],
            "Distribution_Pct": [round(v, 2) for v in day_vals],
        }
    )
    return dist, msg


def _di_normalize_distribution(dist: pd.DataFrame) -> pd.DataFrame:
    if dist.empty or "Distribution_Pct" not in dist.columns:
        return dist
    vals = pd.to_numeric(dist["Distribution_Pct"], errors="coerce").fillna(0.0)
    total = float(vals.sum())
    if total > 0:
        dist["Distribution_Pct"] = (vals / total * 100).round(2)
    return dist


def _di_interval_ratios(interval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if interval_df.empty:
        return pd.DataFrame(), pd.DataFrame(), "Interval data is empty."
    date_col = _pick_col(interval_df, ("date", "ds", "timestamp"))
    ivl_col = _pick_col(interval_df, ("interval", "time", "interval_start"))
    vol_col = _pick_col(interval_df, ("volume", "calls", "items", "count"))
    if not date_col or not ivl_col or not vol_col:
        return pd.DataFrame(), pd.DataFrame(), "Interval data must have date, interval, and volume columns."
    d = interval_df.copy()
    d["date"] = pd.to_datetime(d[date_col], errors="coerce")
    d["interval"] = d[ivl_col].astype(str)
    d["volume"] = pd.to_numeric(d[vol_col], errors="coerce")
    d = d.dropna(subset=["date", "interval", "volume"])
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), "Interval data empty after cleaning."
    max_date = d["date"].max()
    if pd.notna(max_date):
        cutoff = max_date - pd.DateOffset(months=3)
        d = d[d["date"] >= cutoff]
    daily_totals = d.groupby("date")["volume"].sum()
    d = d.merge(daily_totals.rename("day_total"), on="date", how="left")
    d = d[d["day_total"] > 0]
    d["ratio"] = d["volume"] / d["day_total"]
    d["weekday"] = d["date"].dt.day_name()
    interval_ratio = d.groupby(["interval", "weekday"])["ratio"].mean().reset_index()
    overall_ratio = d.groupby("interval")["ratio"].mean().reset_index()
    return interval_ratio, overall_ratio, "Interval ratios computed."


def _di_build_forecasts(
    transform_df: pd.DataFrame,
    interval_df: pd.DataFrame,
    forecast_month: pd.Timestamp,
    group_value: Optional[str],
    model_value: Optional[str],
    month_value: Optional[str],
    distribution_override: Optional[pd.DataFrame] = None,
) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if transform_df.empty:
        return "Load a transformed forecast first.", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    tf = transform_df.copy()
    tf_cols = {c.lower(): c for c in tf.columns}
    fg_col = tf_cols.get("forecast_group") or tf_cols.get("queue_name") or tf_cols.get("category")
    model_col = tf_cols.get("model")

    if fg_col and group_value:
        tf = tf[tf[fg_col] == group_value]
    if model_col and model_value:
        tf = tf[tf[model_col] == model_value]

    val_col = None
    for cand in [
        "Final_Forecast_Post_Transformations",
        "Final_Forecast",
        "Forecast_Marketing Campaign 3",
        "Forecast_Marketing Campaign 2",
        "Forecast_Marketing Campaign 1",
    ]:
        if cand in tf.columns:
            val_col = cand
            break
    if not val_col:
        return "No forecast value column found.", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    if forecast_month is not None:
        month_num = forecast_month.month
        year_val = forecast_month.year
        if "Year" in tf.columns:
            tf["Year"] = pd.to_numeric(tf["Year"], errors="coerce")
        if "Month" in tf.columns:
            tf["_month_num"] = pd.to_datetime(tf["Month"], errors="coerce").dt.month
            tf["_month_num"] = tf["_month_num"].fillna(pd.to_numeric(tf["Month"], errors="coerce"))
            tf = tf[tf["_month_num"] == month_num]
        if "Year" in tf.columns:
            tf = tf[tf["Year"] == year_val]
        if "Month_Year" in tf.columns:
            tf["_month_year_dt"] = pd.to_datetime(tf["Month_Year"], errors="coerce")
            tf = tf[(tf["_month_year_dt"].dt.year == year_val) & (tf["_month_year_dt"].dt.month == month_num)]
    elif month_value and "Month_Year" in tf.columns:
        tf = tf[tf["Month_Year"].astype(str) == str(month_value)]

    if tf.empty:
        return "No matching row for the selected filters.", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    forecast_val = pd.to_numeric(tf[val_col], errors="coerce").dropna()
    if forecast_val.empty:
        return "Selected forecast value is missing.", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    monthly_forecast = float(forecast_val.iloc[0])

    if distribution_override is not None and not distribution_override.empty:
        distribution = distribution_override.copy()
        dist_msg = "Using edited distribution."
    else:
        orig_df, orig_msg = _di_load_original_data(group_value)
        if orig_df.empty:
            return orig_msg, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
        daily = (
            orig_df.groupby("date", as_index=False)["volume"]
            .sum()
            .dropna(subset=["date"])
        )
        daily["month_start"] = daily["date"].dt.to_period("M").dt.to_timestamp()
        daily["day"] = daily["date"].dt.day
        daily_totals = daily.groupby("month_start")["volume"].transform("sum")
        daily["volume_pct"] = np.where(daily_totals > 0, daily["volume"] / daily_totals * 100, np.nan)
        distribution, dist_msg = _di_compute_distribution(daily, forecast_month)

    distribution = _di_normalize_distribution(distribution)
    dist_msg = f"{dist_msg} Normalized (sum {distribution['Distribution_Pct'].sum():.2f}%)."

    daily_tbl = distribution.copy()
    daily_tbl["Daily_Forecast"] = (daily_tbl["Distribution_Pct"] / 100 * monthly_forecast).round(0)

    interval_ratio, overall_ratio, interval_msg = _di_interval_ratios(interval_df)
    interval_rows = []
    if not interval_ratio.empty or not overall_ratio.empty:
        for _, row in daily_tbl.iterrows():
            wd = row["Weekday"]
            dv = float(row["Daily_Forecast"])
            dist = interval_ratio[interval_ratio["weekday"] == wd][["interval", "ratio"]].copy()
            if dist.empty:
                dist = overall_ratio.copy()
            if dist.empty:
                continue
            total = dist["ratio"].sum()
            if total <= 0:
                continue
            dist["ratio"] = dist["ratio"] / total
            dist["Interval_Forecast"] = (dist["ratio"] * dv).round(0)
            dist["Date"] = row["Date"]
            dist["Weekday"] = wd
            interval_rows.append(dist[["Date", "Weekday", "interval", "Interval_Forecast"]])

    interval_tbl = pd.concat(interval_rows, ignore_index=True) if interval_rows else pd.DataFrame()
    if not interval_tbl.empty:
        interval_tbl = interval_tbl.rename(columns={"interval": "Interval"})

    meta = {
        "monthly_forecast": monthly_forecast,
        "distribution_sum": float(distribution["Distribution_Pct"].sum()) if not distribution.empty else 0.0,
        "interval_msg": interval_msg,
        "dist_msg": dist_msg,
    }
    status = f"Daily/interval forecast ready | monthly {monthly_forecast:,.0f}"
    return status, distribution, daily_tbl, interval_tbl, meta


@app.callback(
    Output("di-forecast-dates-msg", "children"),
    Output("di-forecast-year", "options"),
    Output("di-forecast-year", "value"),
    Output("di-forecast-month", "options"),
    Output("di-forecast-month", "value"),
    Output("di-forecast-dates-store", "data", allow_duplicate=True),
    Output("di-holidays-store", "data", allow_duplicate=True),
    Output("di-transform-file", "options"),
    Output("di-transform-file", "value"),
    Input("di-refresh-forecast-dates", "n_clicks"),
    prevent_initial_call=True,
)
def _di_load_forecast_dates_cb(_refresh):
    dates, msg = _di_load_forecast_dates()
    holidays_df, holidays_msg = _di_load_holidays()

    years_opts = []
    year_val = None
    month_opts = []
    month_val = None
    payload = {}
    if not dates.empty:
        years = sorted(dates.dt.year.unique().tolist())
        years_opts = [{"label": str(y), "value": int(y)} for y in years]
        year_val = years_opts[0]["value"] if years_opts else None
        months_by_year = {}
        for y in years:
            months = sorted(dates[dates.dt.year == y].dt.month.unique().tolist())
            months_by_year[int(y)] = [calendar.month_name[m] for m in months if m]
        payload = {"months_by_year": months_by_year}
        month_opts = [{"label": m, "value": m} for m in months_by_year.get(year_val, [])]
        month_val = month_opts[0]["value"] if month_opts else None

    msg_full = msg
    if holidays_msg:
        msg_full = f"{msg} | {holidays_msg}"

    files_opts = _di_list_transform_files(_di_base_dir())
    file_val = files_opts[0]["value"] if files_opts else None

    dates_json = json.dumps(payload) if payload else None
    holidays_json = holidays_df.to_json(date_format="iso", orient="split") if not holidays_df.empty else None
    return (
        msg_full,
        years_opts,
        year_val,
        month_opts,
        month_val,
        dates_json,
        holidays_json,
        files_opts,
        file_val,
    )


@app.callback(
    Output("di-forecast-month", "options", allow_duplicate=True),
    Output("di-forecast-month", "value", allow_duplicate=True),
    Input("di-forecast-year", "value"),
    State("di-forecast-dates-store", "data"),
    prevent_initial_call=True,
)
def _di_update_month_options(year_val, store_json):
    if not year_val or not store_json:
        return [], None
    try:
        payload = json.loads(store_json)
    except Exception:
        return [], None
    months_by_year = payload.get("months_by_year", {})
    try:
        year_int = int(year_val)
    except Exception:
        year_int = None
    months = months_by_year.get(str(year_val)) or (months_by_year.get(year_int) if year_int is not None else None)
    months = months or []
    month_opts = [{"label": m, "value": m} for m in months]
    month_val = month_opts[0]["value"] if month_opts else None
    return month_opts, month_val


@app.callback(
    Output("di-transform-msg", "children"),
    Output("di-transform-preview", "data"),
    Output("di-transform-preview", "columns"),
    Output("di-transform-group", "options"),
    Output("di-transform-group", "value"),
    Output("di-transform-model", "options"),
    Output("di-transform-model", "value"),
    Output("di-transform-month", "options"),
    Output("di-transform-month", "value"),
    Output("di-transform-store", "data", allow_duplicate=True),
    Input("di-load-transform", "n_clicks"),
    Input("di-load-transform-selected", "n_clicks"),
    Input("di-upload-transform", "contents"),
    State("di-transform-file", "value"),
    State("di-upload-transform", "filename"),
    prevent_initial_call=True,
)
def _di_load_transform(n_load, n_selected, contents, selected_path, filename):
    ctx = dash.callback_context
    trig = ctx.triggered_id if ctx.triggered_id else None

    def _load_latest():
        base_dir_txt = Path(__file__).resolve().parent.parent / "latest_forecast_base_dir.txt"
        search_dir = Path(__file__).resolve().parent.parent / "exports"
        if base_dir_txt.exists():
            try:
                txt = base_dir_txt.read_text().strip()
                if txt:
                    candidate = Path(txt)
                    if candidate.exists():
                        search_dir = candidate
            except Exception:
                pass
        files = sorted(search_dir.glob("Monthly_Forecast_with_Adjustments_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return pd.DataFrame(), "No transformed forecast files found."
        try:
            df_latest = pd.read_csv(files[0])
            return df_latest, f"Loaded {files[0].name}."
        except Exception as exc:
            return pd.DataFrame(), f"Could not load latest file: {exc}"

    if trig == "di-load-transform" and n_load:
        df, msg = _load_latest()
    elif trig == "di-load-transform-selected" and n_selected and selected_path:
        try:
            df = pd.read_csv(selected_path)
            msg = f"Loaded {Path(selected_path).name}."
        except Exception as exc:
            df = pd.DataFrame()
            msg = f"Could not load selected file: {exc}"
    elif trig == "di-upload-transform" and contents and filename:
        df, msg = _parse_upload(contents, filename)
    else:
        raise dash.exceptions.PreventUpdate

    if df is None or df.empty:
        return msg, [], [], [], None, [], None, [], None, None

    def _opts(colname: str):
        uniq = sorted(pd.unique(df[colname].dropna()).tolist())
        return [{"label": str(u), "value": u} for u in uniq]

    df_cols = {c.lower(): c for c in df.columns}
    fg_col = df_cols.get("forecast_group") or df_cols.get("queue_name") or df_cols.get("category")
    model_col = df_cols.get("model")
    group_opts = _opts(fg_col) if fg_col else []
    group_val = group_opts[0]["value"] if group_opts else None

    model_opts = _opts(model_col) if model_col else []
    model_val = model_opts[0]["value"] if model_opts else None

    month_val = None
    month_opts: list[dict] = []
    if "Month_Year" in df.columns:
        df["Month_Year"] = df["Month_Year"].astype(str)
        month_opts = _opts("Month_Year")
        month_val = month_opts[0]["value"] if month_opts else None
    else:
        month_col = df_cols.get("month")
        year_col = df_cols.get("year")
        if month_col and year_col:
            df["_month_year"] = df[month_col].astype(str).str[:3] + " " + df[year_col].astype(str)
            month_opts = _opts("_month_year")
            month_val = month_opts[0]["value"] if month_opts else None

    preview = df.head(200)
    return (
        msg,
        preview.to_dict("records"),
        _cols(preview),
        group_opts,
        group_val,
        model_opts,
        model_val,
        month_opts,
        month_val,
        df.to_json(date_format="iso", orient="split"),
    )


@app.callback(
    Output("di-upload-msg", "children"),
    Output("di-preview", "data"),
    Output("di-preview", "columns"),
    Output("di-interval-store", "data", allow_duplicate=True),
    Input("di-upload", "contents"),
    State("di-upload", "filename"),
    prevent_initial_call=True,
)
def _di_on_interval_upload(contents, filename):
    if not contents or not filename:
        raise dash.exceptions.PreventUpdate
    df, msg = _parse_upload(contents, filename)
    preview = df.head(200)
    return msg, preview.to_dict("records"), _cols(preview), df.to_json(date_format="iso", orient="split")


@app.callback(
    Output("di-run-status", "children"),
    Output("di-distribution-table", "data"),
    Output("di-distribution-table", "columns"),
    Output("di-distribution-msg", "children"),
    Output("di-daily-table", "data"),
    Output("di-daily-table", "columns"),
    Output("di-interval-forecast-table", "data"),
    Output("di-interval-forecast-table", "columns"),
    Output("di-results-store", "data", allow_duplicate=True),
    Output("di-distribution-store", "data", allow_duplicate=True),
    Input("di-run-btn", "n_clicks"),
    State("di-transform-store", "data"),
    State("di-interval-store", "data"),
    State("di-transform-group", "value"),
    State("di-transform-model", "value"),
    State("di-transform-month", "value"),
    State("di-forecast-year", "value"),
    State("di-forecast-month", "value"),
    State("di-holidays-store", "data"),
    prevent_initial_call=True,
)
def _di_run_interval_forecast(
    n,
    transform_json,
    interval_json,
    group_value,
    model_value,
    month_value,
    forecast_year,
    forecast_month,
    holidays_json,
):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not transform_json:
        return ("Load a transformed forecast first.", [], [], "", [], [], [], [], None, None)
    try:
        tf = pd.read_json(io.StringIO(transform_json), orient="split")
    except Exception:
        return ("Could not read transformed forecast.", [], [], "", [], [], [], [], None, None)

    if interval_json:
        try:
            iv = pd.read_json(io.StringIO(interval_json), orient="split")
        except Exception:
            iv = pd.DataFrame()
    else:
        iv = pd.DataFrame()

    if iv.empty:
        orig_df, _ = _di_load_original_data(group_value)
        if not orig_df.empty and _pick_col(orig_df, ("interval", "time", "interval_start")):
            iv = orig_df.copy()

    forecast_month_dt = _di_parse_forecast_month(forecast_year, forecast_month)
    if forecast_month_dt is None and month_value:
        try:
            fallback = pd.to_datetime(month_value, errors="coerce")
        except Exception:
            fallback = pd.NaT
        if pd.notna(fallback):
            forecast_month_dt = fallback.replace(day=1)
    if forecast_month_dt is None:
        return ("Select a forecast month.", [], [], "", [], [], [], [], None, None)

    status, dist_df, daily_tbl, interval_tbl, meta = _di_build_forecasts(
        tf,
        iv,
        forecast_month_dt,
        group_value,
        model_value,
        month_value,
    )
    if dist_df.empty and daily_tbl.empty:
        return (status, [], [], meta.get("dist_msg", ""), [], [], [], [], None, None)

    results = {
        "distribution": dist_df.to_dict("records"),
        "daily": daily_tbl.to_dict("records"),
        "interval": interval_tbl.to_dict("records"),
        "meta": {
            "group": group_value,
            "model": model_value,
            "month": str(forecast_month_dt.strftime("%b %Y")),
            "monthly_forecast": meta.get("monthly_forecast"),
            "dist_msg": meta.get("dist_msg"),
            "interval_msg": meta.get("interval_msg"),
        },
    }
    dist_msg = meta.get("dist_msg", "")
    if holidays_json:
        try:
            hdf = pd.read_json(io.StringIO(holidays_json), orient="split")
        except Exception:
            hdf = pd.DataFrame()
        if not hdf.empty and "holiday_date" in hdf.columns and forecast_month_dt is not None:
            hdf["holiday_date"] = pd.to_datetime(hdf["holiday_date"], errors="coerce")
            month_mask = (
                hdf["holiday_date"].dt.year == forecast_month_dt.year
            ) & (hdf["holiday_date"].dt.month == forecast_month_dt.month)
            month_holidays = hdf[month_mask]
            if not month_holidays.empty:
                names = []
                for _, row in month_holidays.iterrows():
                    name = row.get("holiday_name", "")
                    dt = row.get("holiday_date")
                    if pd.notna(dt):
                        names.append(f"{dt.strftime('%d %b')}: {name}".strip())
                if names:
                    dist_msg = f"{dist_msg} | Holidays: {', '.join(names)}"
    return (
        status,
        dist_df.to_dict("records"),
        _cols(dist_df),
        dist_msg,
        daily_tbl.to_dict("records"),
        _cols(daily_tbl),
        interval_tbl.to_dict("records"),
        _cols(interval_tbl),
        json.dumps(results),
        dist_df.to_json(date_format="iso", orient="split"),
    )


@app.callback(
    Output("di-distribution-table", "data", allow_duplicate=True),
    Output("di-distribution-table", "columns", allow_duplicate=True),
    Output("di-distribution-msg", "children", allow_duplicate=True),
    Output("di-daily-table", "data", allow_duplicate=True),
    Output("di-daily-table", "columns", allow_duplicate=True),
    Output("di-interval-forecast-table", "data", allow_duplicate=True),
    Output("di-interval-forecast-table", "columns", allow_duplicate=True),
    Output("di-results-store", "data", allow_duplicate=True),
    Output("di-distribution-store", "data", allow_duplicate=True),
    Input("di-apply-distribution-btn", "n_clicks"),
    State("di-distribution-table", "data"),
    State("di-transform-store", "data"),
    State("di-interval-store", "data"),
    State("di-transform-group", "value"),
    State("di-transform-model", "value"),
    State("di-transform-month", "value"),
    State("di-forecast-year", "value"),
    State("di-forecast-month", "value"),
    prevent_initial_call=True,
)
def _di_apply_distribution_edits(
    n,
    dist_data,
    transform_json,
    interval_json,
    group_value,
    model_value,
    month_value,
    forecast_year,
    forecast_month,
):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not transform_json:
        return [], [], "Load a transformed forecast first.", [], [], [], [], None, None
    if not dist_data:
        return [], [], "No distribution data to apply.", [], [], [], [], None, None

    try:
        tf = pd.read_json(io.StringIO(transform_json), orient="split")
    except Exception:
        return [], [], "Could not read transformed forecast.", [], [], [], [], None, None

    dist_df = pd.DataFrame(dist_data)
    if dist_df.empty:
        return [], [], "No distribution data to apply.", [], [], [], [], None, None
    if "Distribution_Pct" not in dist_df.columns:
        for col in dist_df.columns:
            if str(col).strip().lower() in {"distribution_pct", "distribution", "pct", "percent"}:
                dist_df = dist_df.rename(columns={col: "Distribution_Pct"})
                break

    forecast_month_dt = _di_parse_forecast_month(forecast_year, forecast_month)
    if forecast_month_dt is None and "Date" in dist_df.columns:
        dt = pd.to_datetime(dist_df["Date"], errors="coerce").dropna()
        if not dt.empty:
            forecast_month_dt = dt.iloc[0].replace(day=1)

    if interval_json:
        try:
            iv = pd.read_json(io.StringIO(interval_json), orient="split")
        except Exception:
            iv = pd.DataFrame()
    else:
        iv = pd.DataFrame()

    if iv.empty:
        orig_df, _ = _di_load_original_data(group_value)
        if not orig_df.empty and _pick_col(orig_df, ("interval", "time", "interval_start")):
            iv = orig_df.copy()

    status, dist_norm, daily_tbl, interval_tbl, meta = _di_build_forecasts(
        tf,
        iv,
        forecast_month_dt,
        group_value,
        model_value,
        month_value,
        distribution_override=dist_df,
    )
    if dist_norm.empty:
        return [], [], status, [], [], [], [], None, None

    results = {
        "distribution": dist_norm.to_dict("records"),
        "daily": daily_tbl.to_dict("records"),
        "interval": interval_tbl.to_dict("records"),
        "meta": {
            "group": group_value,
            "model": model_value,
            "month": str(forecast_month_dt.strftime("%b %Y")) if forecast_month_dt is not None else "",
            "monthly_forecast": meta.get("monthly_forecast"),
            "dist_msg": meta.get("dist_msg"),
            "interval_msg": meta.get("interval_msg"),
        },
    }
    return (
        dist_norm.to_dict("records"),
        _cols(dist_norm),
        meta.get("dist_msg", status),
        daily_tbl.to_dict("records"),
        _cols(daily_tbl),
        interval_tbl.to_dict("records"),
        _cols(interval_tbl),
        json.dumps(results),
        dist_norm.to_json(date_format="iso", orient="split"),
    )


@app.callback(
    Output("di-download-daily", "data"),
    Input("di-download-daily-btn", "n_clicks"),
    State("di-results-store", "data"),
    prevent_initial_call=True,
)
def _di_download_daily(n, results_json):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not results_json:
        return dash.no_update
    try:
        payload = json.loads(results_json)
    except Exception:
        return dash.no_update
    daily = pd.DataFrame(payload.get("daily", []))
    if daily.empty:
        return dash.no_update
    return dcc.send_data_frame(daily.to_csv, "daily_forecast.csv", index=False)


@app.callback(
    Output("di-download-interval", "data"),
    Input("di-download-interval-btn", "n_clicks"),
    State("di-results-store", "data"),
    prevent_initial_call=True,
)
def _di_download_interval(n, results_json):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not results_json:
        return dash.no_update
    try:
        payload = json.loads(results_json)
    except Exception:
        return dash.no_update
    interval = pd.DataFrame(payload.get("interval", []))
    if interval.empty:
        return dash.no_update
    return dcc.send_data_frame(interval.to_csv, "interval_forecast.csv", index=False)


@app.callback(
    Output("di-save-status", "children"),
    Input("di-save-btn", "n_clicks"),
    State("di-results-store", "data"),
    prevent_initial_call=True,
)
def _di_save_results(n, results_json):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not results_json:
        return "Run interval forecast first."
    try:
        payload = json.loads(results_json)
    except Exception as exc:
        return f"Could not parse results: {exc}"
    daily = pd.DataFrame(payload.get("daily", []))
    interval = pd.DataFrame(payload.get("interval", []))
    meta = payload.get("meta", {})
    month_label = meta.get("month") or ""
    month_folder = None
    if month_label:
        try:
            month_dt = pd.to_datetime(month_label, errors="coerce")
        except Exception:
            month_dt = pd.NaT
        if pd.notna(month_dt):
            month_folder = month_dt.strftime("%b_%Y")

    base_dir = None
    repo_root = Path(__file__).resolve().parent.parent
    saved_path_txt = repo_root / "saved_folder_path.txt"
    if saved_path_txt.exists():
        try:
            saved_path = saved_path_txt.read_text().strip()
            if saved_path:
                base_dir = Path(saved_path)
        except Exception:
            base_dir = None

    if base_dir is None:
        base_dir = _di_base_dir()
    if base_dir is None:
        base_dir = repo_root / "exports"
    base_dir.mkdir(exist_ok=True, parents=True)

    output_dir = base_dir / month_folder if month_folder else base_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    ts = pd.Timestamp.now()
    try:
        daily_path = output_dir / f"Final_Daily_Forecast_{ts:%Y%m%d_%H%M%S}.csv"
        interval_path = output_dir / f"Interval_Daily_Forecast_{ts:%Y%m%d_%H%M%S}.csv"
        daily.to_csv(daily_path, index=False)
        interval.to_csv(interval_path, index=False)
        try:
            (output_dir / "latest_forecast_path.txt").write_text(str(daily_path))
        except Exception:
            pass
        return f"Saved to {daily_path} and {interval_path}"
    except Exception as exc:
        return f"Save failed: {exc}"


@app.callback(
    Output("tp-raw-store", "data", allow_duplicate=True),
    Output("tp-filtered-store", "data", allow_duplicate=True),
    Output("tp-final-store", "data", allow_duplicate=True),
    Output("tp-raw-table", "data", allow_duplicate=True),
    Output("tp-raw-table", "columns", allow_duplicate=True),
    Output("tp-filtered-table", "data", allow_duplicate=True),
    Output("tp-filtered-table", "columns", allow_duplicate=True),
    Output("tp-transform-table", "data", allow_duplicate=True),
    Output("tp-transform-table", "columns", allow_duplicate=True),
    Output("tp-transposed-table", "data", allow_duplicate=True),
    Output("tp-transposed-table", "columns", allow_duplicate=True),
    Output("tp-final-forecast-table", "data", allow_duplicate=True),
    Output("tp-final-forecast-table", "columns", allow_duplicate=True),
    Output("tp-selection-status", "is_open", allow_duplicate=True),
    Output("tp-selection-status", "children", allow_duplicate=True),
    Output("tp-transform-status", "children", allow_duplicate=True),
    Input("tp-reset", "n_clicks"),
    prevent_initial_call=True,
)
def _tp_reset(n):
    if not n:
        raise dash.exceptions.PreventUpdate
    empty_cols = []
    return (None, None, None, [], empty_cols, [], empty_cols, [], empty_cols, [], empty_cols, [], empty_cols, False, "", "")
