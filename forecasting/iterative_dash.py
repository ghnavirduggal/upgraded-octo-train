from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Optional


def apply_cyclical_contact_ratio_pattern(
    contact_ratio_template: pd.DataFrame,
    future_iq_data: pd.DataFrame,
    future_holidays: Optional[dict] = None,
) -> pd.DataFrame:
    """Repeat historical contact ratios by month over future IQ values."""
    if contact_ratio_template is None or contact_ratio_template.empty:
        return pd.DataFrame(columns=["ds", "yhat"])

    monthly_ratios: dict[int, float] = {}
    for _, row in contact_ratio_template.iterrows():
        try:
            m = pd.to_datetime(row["Date"]).month
            monthly_ratios[m] = float(row["Contact_Ratio"])
        except Exception:
            continue

    future = future_iq_data.copy()
    date_col = "Date"
    if date_col not in future.columns:
        date_col = "ds" if "ds" in future.columns else "date" if "date" in future.columns else ""
    if not date_col:
        return pd.DataFrame(columns=["ds", "yhat"])

    future["Date"] = pd.to_datetime(future[date_col], errors="coerce")
    future = future.dropna(subset=["Date"])
    future["Month"] = future["Date"].dt.month
    future["Year_Month"] = future["Date"].dt.strftime("%Y-%m")
    future["Contact_Ratio"] = future["Month"].map(monthly_ratios)

    iq_col = None
    for cand in ("IQ_value", "iq_value", "iq", "value"):
        if cand in future.columns:
            iq_col = cand
            break
    if iq_col:
        future["Volume"] = pd.to_numeric(future[iq_col], errors="coerce") * future["Contact_Ratio"]
    else:
        future["Volume"] = pd.NA

    if future_holidays:
        holiday_multipliers = {
            "Diwali": 1.15,
            "Christmas": 1.12,
            "Eid": 1.08,
            "New Year": 1.10,
        }
        for idx, row in future.iterrows():
            hm = row.get("Year_Month")
            if hm in future_holidays:
                mult = holiday_multipliers.get(future_holidays[hm], 1.0)
                future.loc[idx, "Volume"] = float(future.loc[idx, "Volume"] or 0.0) * mult

    result = future[["Date", "Contact_Ratio", "Volume"]].rename(
        columns={"Date": "ds", "Contact_Ratio": "yhat"}
    )
    return result.dropna(subset=["ds"])


def prophet_forecast_phase2_static(contact_ratio_template, future_iq_data, periods, static_config, future_holidays=None):
    result = apply_cyclical_contact_ratio_pattern(contact_ratio_template, future_iq_data, future_holidays)
    if result is None or result.empty:
        return pd.DataFrame(columns=["ds", "yhat"])
    boost = float(static_config.get("seasonality_strength", 1.0) or 1.0)
    if "yhat" not in result.columns:
        return pd.DataFrame(columns=["ds", "yhat"])
    result["yhat"] = pd.to_numeric(result["yhat"], errors="coerce") * boost
    return result[["ds", "yhat"]]


def rf_forecast_phase2_static(contact_ratio_template, future_iq_data, periods, static_config, future_holidays=None):
    result = apply_cyclical_contact_ratio_pattern(contact_ratio_template, future_iq_data, future_holidays)
    if result is None or result.empty:
        return pd.DataFrame(columns=["ds", "yhat"])
    boost = float(static_config.get("feature_importance_boost", 1.0) or 1.0)
    result["yhat"] = pd.to_numeric(result["yhat"], errors="coerce") * boost
    return result[["ds", "yhat"]]


def xgb_forecast_phase2_static(contact_ratio_template, future_iq_data, periods, static_config, future_holidays=None):
    result = apply_cyclical_contact_ratio_pattern(contact_ratio_template, future_iq_data, future_holidays)
    if result is None or result.empty:
        return pd.DataFrame(columns=["ds", "yhat"])
    boost = float(static_config.get("learning_rate_boost", 1.0) or 1.0)
    result["yhat"] = pd.to_numeric(result["yhat"], errors="coerce") * boost
    return result[["ds", "yhat"]]


def var_forecast_phase2_static(contact_ratio_template, future_iq_data, periods, static_config, future_holidays=None):
    result = apply_cyclical_contact_ratio_pattern(contact_ratio_template, future_iq_data, future_holidays)
    if result is None or result.empty:
        return pd.DataFrame(columns=["ds", "yhat"])
    lag_boost = float(static_config.get("lag_strength", 1.0) or 1.0)
    multi_boost = float(static_config.get("multivariate_strength", 1.0) or 1.0)
    result["yhat"] = pd.to_numeric(result["yhat"], errors="coerce") * (lag_boost + multi_boost)
    return result[["ds", "yhat"]]


def sarimax_forecast_phase2_static(contact_ratio_template, future_iq_data, periods, static_config, future_holidays=None):
    result = apply_cyclical_contact_ratio_pattern(contact_ratio_template, future_iq_data, future_holidays)
    if result is None or result.empty:
        return pd.DataFrame(columns=["ds", "yhat"])
    seas = float(static_config.get("seasonality_strength", 1.0) or 1.0)
    trend = float(static_config.get("trend_strength", 1.0) or 1.0)
    result["yhat"] = pd.to_numeric(result["yhat"], errors="coerce") * (seas + trend)
    return result[["ds", "yhat"]]


def run_phase2_with_static_config(
    contact_ratio_template: pd.DataFrame,
    future_iq_data: pd.DataFrame,
    periods: int,
    phase1_static_configs: Dict[str, dict],
    future_holidays: Optional[dict] = None,
    selected_model: str = "all",
) -> Dict[str, pd.DataFrame]:
    models = {
        "prophet": prophet_forecast_phase2_static,
        "rf": rf_forecast_phase2_static,
        "xgb": xgb_forecast_phase2_static,
        "var": var_forecast_phase2_static,
        "sarimax": sarimax_forecast_phase2_static,
    }
    if selected_model != "all":
        models = {selected_model: models[selected_model]} if selected_model in models else {}
    out: Dict[str, pd.DataFrame] = {}
    for name, func in models.items():
        cfg = phase1_static_configs.get(name, {})
        try:
            out[name] = func(contact_ratio_template, future_iq_data, periods, cfg, future_holidays)
        except Exception:
            out[name] = pd.DataFrame(columns=["ds", "yhat"])
    return out


def run_all_models_separately(
    merged_filtered: pd.DataFrame,
    future_iq_data: pd.DataFrame,
    forecast_months: int,
    all_model_configs: Dict[str, dict],
    future_holidays: Optional[dict],
) -> Dict[str, pd.DataFrame]:
    """Apply static templates per model row to future IQ data."""
    results: Dict[str, pd.DataFrame] = {}
    month_map = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
        "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
        "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
    }

    for _, row in merged_filtered.iterrows():
        model_name = str(row.get("Model", "")).strip().lower()
        month_cols = [
            col for col in merged_filtered.columns
            if "-" in str(col) and any(m in str(col) for m in month_map)
        ]
        data = []
        for mc in month_cols:
            try:
                val = row[mc]
                if pd.isna(val):
                    continue
                mname, yr = str(mc).split("-", 1)
                if mname not in month_map:
                    continue
                date_str = f"20{yr}-{month_map[mname]}-01"
                data.append({"Date": pd.to_datetime(date_str), "Contact_Ratio": float(val)})
            except Exception:
                continue
        tmpl = pd.DataFrame(data).sort_values("Date")
        cfg = {model_name: all_model_configs.get(model_name, {})}
        res = run_phase2_with_static_config(tmpl, future_iq_data, forecast_months, cfg, future_holidays, selected_model=model_name)
        results[model_name] = res.get(model_name, pd.DataFrame())
    return results


def transform_merged_filtered_for_phase2(merged_filtered: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    month_map = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
        "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
        "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
    }
    month_cols = [
        col for col in merged_filtered.columns
        if "-" in str(col) and any(m in str(col) for m in month_map)
    ]
    out: Dict[str, pd.DataFrame] = {}
    for _, row in merged_filtered.iterrows():
        model_name = str(row.get("Model", "")).strip().lower()
        data = []
        for mc in month_cols:
            try:
                val = row[mc]
                if pd.isna(val):
                    continue
                mname, yr = str(mc).split("-", 1)
                if mname not in month_map:
                    continue
                date_str = f"20{yr}-{month_map[mname]}-01"
                data.append({"Date": pd.to_datetime(date_str), "Contact_Ratio": float(val)})
            except Exception:
                continue
        out[model_name] = pd.DataFrame(data).sort_values("Date") if data else pd.DataFrame()
    return out


def create_single_template_from_dict(model_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "prophet" in model_dict and isinstance(model_dict["prophet"], pd.DataFrame) and not model_dict["prophet"].empty:
        return model_dict["prophet"].copy()
    frames = [df for df in model_dict.values() if isinstance(df, pd.DataFrame) and not df.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby("Date", as_index=False)["Contact_Ratio"].mean()
