from __future__ import annotations

import copy
import math
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import VAR, SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from xgboost import XGBRegressor

import config_manager


def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalize_col_name(name: Any) -> str:
    return str(name).replace("\u2212", "-").strip().lower()


def _find_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lookup = {_normalize_col_name(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_col_name(cand)
        if key in lookup:
            return lookup[key]
    return None


def _normalize_model_name(name: Any) -> str:
    raw = str(name).strip().lower()
    cleaned = "".join(ch for ch in raw if ch.isalnum())
    mapping = {
        "prophet": "prophet",
        "randomforest": "random_forest",
        "rf": "random_forest",
        "xgboost": "xgboost",
        "xgb": "xgboost",
        "sarimax": "sarimax",
        "var": "var",
    }
    return mapping.get(cleaned, cleaned)


def _ensure_iq_scaled(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "IQ_value_scaled" in df.columns:
        return df
    if "IQ_value" not in df.columns:
        return df
    d = df.copy()
    scaler = MinMaxScaler()
    d["IQ_value_scaled"] = scaler.fit_transform(d[["IQ_value"]]).round(4)
    return d


def _normalize_holidays(holiday_original_date: Any) -> Optional[pd.DataFrame]:
    if holiday_original_date is None:
        return None
    if isinstance(holiday_original_date, pd.DataFrame):
        if "ds" in holiday_original_date.columns:
            return holiday_original_date[["ds"] + (["holiday"] if "holiday" in holiday_original_date.columns else [])]
        return None
    if isinstance(holiday_original_date, dict):
        return pd.DataFrame({"ds": list(holiday_original_date.keys()), "holiday": list(holiday_original_date.values())})
    return None


def _prep_input(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    cols = {str(c).strip().lower(): c for c in d.columns}
    ds_col = cols.get("ds") or cols.get("date")
    if not ds_col:
        raise ValueError("Input must contain a 'ds' or 'date' column.")
    d["ds"] = pd.to_datetime(d[ds_col], errors="coerce")
    d = d.dropna(subset=["ds"])
    if "final_smoothed_value" not in cols and "y" in cols:
        d["Final_Smoothed_Value"] = _safe_num(d[cols["y"]]) * 100.0
    else:
        d["Final_Smoothed_Value"] = _safe_num(d.get(cols.get("final_smoothed_value", "Final_Smoothed_Value"), 0.0))
    d["IQ_value"] = _safe_num(d.get(cols.get("iq_value", "IQ_value"), 1.0))
    hcol = cols.get("holiday_month_start") or cols.get("holiday")
    if hcol:
        d["holiday_month_start"] = pd.to_datetime(d[hcol], errors="coerce")
    else:
        d["holiday_month_start"] = pd.NaT
    d["Year"] = d["ds"].dt.year
    d["Month"] = d["ds"].dt.strftime("%b")
    d = _ensure_iq_scaled(d)
    return d.reset_index(drop=True)


def _prep_smoothed_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    if "ds" not in d.columns:
        if {"Year", "Month"}.issubset(d.columns):
            d["ds"] = pd.to_datetime(
                d["Year"].astype(str) + "-" + d["Month"].astype(str) + "-01",
                errors="coerce",
            )
    d["ds"] = pd.to_datetime(d["ds"], errors="coerce")
    if "Final_Smoothed_Value" not in d.columns and "y" in d.columns:
        d["Final_Smoothed_Value"] = _safe_num(d["y"])
    iq_col = _find_col(d, ("IQ_value",))
    if iq_col and iq_col != "IQ_value":
        d["IQ_value"] = _safe_num(d[iq_col])
    elif "IQ_value" in d.columns:
        d["IQ_value"] = _safe_num(d["IQ_value"])
    scaled_col = _find_col(d, ("IQ_value_scaled",))
    if scaled_col and scaled_col != "IQ_value_scaled":
        d["IQ_value_scaled"] = _safe_num(d[scaled_col])
    d = _ensure_iq_scaled(d)
    if "holiday_month_start" not in d.columns:
        if "holiday" in d.columns:
            d["holiday_month_start"] = pd.to_datetime(d["holiday"], errors="coerce")
        else:
            d["holiday_month_start"] = pd.NaT
    d = d.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
    d["holiday_month_start_flag"] = d["holiday_month_start"].notna().astype(int)
    return d


def prophet_forecast(
    train: pd.DataFrame,
    periods: int,
    config: Optional[dict] = None,
    holidays_df: Optional[pd.DataFrame] = None,
    *,
    seasonality: bool = True,
    regressors: Optional[list[str]] = None,
    prop_config: Optional[dict] = None,
    holidays: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    cfg = prop_config or config or {}
    holiday_df = holidays if holidays is not None else holidays_df
    use_holidays = cfg.get("use_holidays", True)
    yearly_value = cfg.get("yearly_fourier_order")
    if yearly_value is None:
        yearly_value = cfg.get("monthly_fourier_order", cfg.get("yearly_seasonality", seasonality))
    m = Prophet(
        changepoint_prior_scale=cfg.get("changepoint_prior_scale", 0.05),
        seasonality_prior_scale=cfg.get("seasonality_prior_scale", 0.1),
        holidays_prior_scale=cfg.get("holidays_prior_scale", 0.1),
        yearly_seasonality=yearly_value,
        weekly_seasonality=cfg.get("weekly_seasonality", False),
        daily_seasonality=cfg.get("daily_seasonality", False),
        holidays=holiday_df if use_holidays else None,
    )

    reg_cols: list[str] = []
    if regressors:
        reg_cols = [r for r in regressors if r in train.columns]
    elif cfg.get("use_iq_value_scaled", True) and "IQ_value_scaled" in train.columns:
        reg_cols = ["IQ_value_scaled"]

    for reg in reg_cols:
        m.add_regressor(reg)

    fit_cols = ["ds", "y"] + reg_cols
    m.fit(train[fit_cols])
    future = m.make_future_dataframe(periods=periods, freq="MS")
    for reg in reg_cols:
        last_val = train[reg].iloc[-1]
        future[reg] = list(train[reg]) + [last_val] * periods
    fc = m.predict(future)
    return fc[["ds", "yhat"]].iloc[len(train) :]


def add_monthly_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out["ds"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def _add_monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_monthly_seasonal_features(df)


def rf_forecast(
    train: pd.DataFrame,
    periods: int,
    config: Optional[dict] = None,
    *,
    regressors: Optional[list[str]] = None,
    holiday_feature: Optional[str] = None,
    rf_config: Optional[dict] = None,
) -> pd.DataFrame:
    cfg = rf_config or config or {}
    t = add_monthly_seasonal_features(train)
    X_train = np.arange(len(t)).reshape(-1, 1)
    seasonal = t[["month_sin", "month_cos"]].values
    X_train = np.hstack([X_train, seasonal])

    if regressors is None and cfg.get("use_iq_value_scaled", True) and "IQ_value_scaled" in t.columns:
        regressors = ["IQ_value_scaled"]
    if regressors:
        for reg in regressors:
            if reg in t.columns:
                X_train = np.hstack([X_train, t[reg].values.reshape(-1, 1)])

    if holiday_feature and cfg.get("use_holidays", True) and holiday_feature in t.columns:
        X_train = np.hstack([X_train, t[holiday_feature].values.reshape(-1, 1)])

    y_train = t["y"].values
    future_dates = pd.date_range(start=t["ds"].iloc[-1] + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    future_df = pd.DataFrame({"ds": future_dates})
    future_df = add_monthly_seasonal_features(future_df)
    X_future = np.arange(len(t), len(t) + periods).reshape(-1, 1)
    future_seasonal = future_df[["month_sin", "month_cos"]].values
    X_future = np.hstack([X_future, future_seasonal])

    if regressors:
        for reg in regressors:
            if reg in t.columns:
                last_val = t[reg].iloc[-1]
                forecast_vals = np.array([last_val] * periods).reshape(-1, 1)
                X_future = np.hstack([X_future, forecast_vals])

    if holiday_feature and cfg.get("use_holidays", True) and holiday_feature in t.columns:
        last_val = t[holiday_feature].iloc[-1]
        forecast_vals = np.array([last_val] * periods).reshape(-1, 1)
        X_future = np.hstack([X_future, forecast_vals])

    model = RandomForestRegressor(
        n_estimators=int(cfg.get("n_estimators", 200)),
        max_depth=int(cfg.get("max_depth", 5)),
        random_state=int(cfg.get("random_state", 42)),
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_future)
    return pd.DataFrame({"ds": future_dates, "yhat": y_pred})


def xgb_forecast(
    train: pd.DataFrame,
    periods: int,
    config: Optional[dict] = None,
    *,
    regressors: Optional[list[str]] = None,
    holiday_feature: Optional[str] = None,
    xgb_config: Optional[dict] = None,
) -> pd.DataFrame:
    cfg = xgb_config or config or {}
    t = add_monthly_seasonal_features(train)
    X_train = np.arange(len(t)).reshape(-1, 1)
    seasonal = t[["month_sin", "month_cos"]].values
    X_train = np.hstack([X_train, seasonal])

    if regressors is None and cfg.get("use_iq_value_scaled", True) and "IQ_value_scaled" in t.columns:
        regressors = ["IQ_value_scaled"]
    if regressors:
        for reg in regressors:
            if reg in t.columns:
                X_train = np.hstack([X_train, t[reg].values.reshape(-1, 1)])

    if holiday_feature and cfg.get("use_holidays", True) and holiday_feature in t.columns:
        X_train = np.hstack([X_train, t[holiday_feature].values.reshape(-1, 1)])

    y_train = t["y"].values
    future_dates = pd.date_range(start=t["ds"].iloc[-1] + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    future_df = pd.DataFrame({"ds": future_dates})
    future_df = add_monthly_seasonal_features(future_df)
    X_future = np.arange(len(t), len(t) + periods).reshape(-1, 1)
    future_seasonal = future_df[["month_sin", "month_cos"]].values
    X_future = np.hstack([X_future, future_seasonal])

    if regressors:
        for reg in regressors:
            if reg in t.columns:
                last_val = t[reg].iloc[-1]
                forecast_vals = np.array([last_val] * periods).reshape(-1, 1)
                X_future = np.hstack([X_future, forecast_vals])

    if holiday_feature and cfg.get("use_holidays", True) and holiday_feature in t.columns:
        last_val = t[holiday_feature].iloc[-1]
        forecast_vals = np.array([last_val] * periods).reshape(-1, 1)
        X_future = np.hstack([X_future, forecast_vals])

    model = XGBRegressor(
        n_estimators=int(cfg.get("n_estimators", 200)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        max_depth=int(cfg.get("max_depth", 3)),
        random_state=int(cfg.get("random_state", 42)),
        subsample=float(cfg.get("subsample", 1.0)),
        colsample_bytree=float(cfg.get("colsample_bytree", 1.0)),
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_future)
    return pd.DataFrame({"ds": future_dates, "yhat": y_pred})


def var_forecast(
    train: pd.DataFrame,
    periods: int,
    config: Optional[dict] = None,
    *,
    regressors: Optional[list[str]] = None,
    var_config: Optional[dict] = None,
) -> pd.DataFrame:
    cfg = var_config or config or {}
    t = add_monthly_seasonal_features(train)
    vars_use = ["y", "month_sin", "month_cos"]
    if regressors:
        vars_use.extend([r for r in regressors if r in t.columns])
    else:
        if cfg.get("use_iq_value_scaled", True) and "IQ_value_scaled" in t.columns:
            vars_use.append("IQ_value_scaled")
        if cfg.get("use_holidays", True) and "holiday_month_start_flag" in t.columns:
            vars_use.append("holiday_month_start_flag")

    data = t[vars_use].dropna()
    if data.empty or len(data) < 3:
        return pd.DataFrame(columns=["ds", "yhat"])

    lag = int(cfg.get("lags", 12))
    lag = max(1, min(lag, len(data) - 1))
    model = VAR(data)
    results = model.fit(lag)
    fc = results.forecast(data.iloc[-lag:].values, steps=periods)
    future_dates = pd.date_range(t["ds"].iloc[-1] + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    return pd.DataFrame({"ds": future_dates, "yhat": fc[:, 0]})


def sarimax_forecast(
    train: pd.DataFrame,
    periods: int,
    config: Optional[dict] = None,
    exog_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    cfg = config or {}
    exog_cols = exog_cols or []
    exog_train = train[exog_cols] if exog_cols else None

    order = cfg.get("order", (1, 1, 1))
    seasonal = cfg.get("seasonal_order", (1, 1, 1, 12))
    if isinstance(order, str):
        order = tuple(int(x) for x in order.strip("()[]").split(",") if str(x).strip())
    if isinstance(seasonal, str):
        seasonal = tuple(int(x) for x in seasonal.strip("()[]").split(",") if str(x).strip())

    model = SARIMAX(train["y"], exog=exog_train, order=tuple(order), seasonal_order=tuple(seasonal))
    maxiter = int(cfg.get("maxiter", 200))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        results = model.fit(disp=False, maxiter=maxiter)

    exog_forecast = None
    if exog_cols:
        last_vals = {c: train[c].iloc[-1] for c in exog_cols}
        exog_forecast = pd.DataFrame({c: [last_vals[c]] * periods for c in exog_cols})
    fc = results.get_forecast(steps=periods, exog=exog_forecast)
    future_dates = pd.date_range(train["ds"].iloc[-1] + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    return pd.DataFrame({"ds": future_dates, "yhat": fc.predicted_mean})


def create_flexible_12_month_test_split(df_smoothed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    df_smoothed = df_smoothed.sort_values("ds").reset_index(drop=True)
    total_months = len(df_smoothed)
    test_size = min(12, max(1, total_months // 3))
    train_size = max(1, total_months - test_size)
    train = df_smoothed.iloc[:train_size].copy()
    test = df_smoothed.iloc[train_size:].copy()
    forecast_months_test_range = len(test)
    return train, test, forecast_months_test_range


def calculate_accuracy_for_fine_tuning(model_name: str, actual_data: pd.DataFrame, forecast_df: pd.DataFrame) -> dict:
    if actual_data is None or forecast_df is None or actual_data.empty or forecast_df.empty:
        return {"Model": model_name, "Count_within(+/-5%)": 0, "Total_Months": 0, "Month_Level_Details": pd.DataFrame()}

    actuals_df = actual_data[actual_data["ds"].isin(forecast_df["ds"])].copy()
    actuals_df = actuals_df.sort_values("ds").reset_index(drop=True)
    forecasts_df = forecast_df.sort_values("ds").reset_index(drop=True)
    actuals = actuals_df["y"].reset_index(drop=True)
    forecasts = forecasts_df["yhat"].reset_index(drop=True)

    within_5_flag = (forecasts >= actuals * 0.945) & (forecasts <= actuals * 1.055)
    count_within_5 = int(np.sum(within_5_flag))
    total_months = int(len(actuals))

    month_year = forecasts_df["ds"].dt.strftime("%b-%y")
    month_level_df = pd.DataFrame(
        {
            "Model": model_name,
            "Month_Year": month_year,
            "Actual": actuals,
            "Forecast": forecasts,
            "Within_5pct": within_5_flag,
        }
    )

    pivot_data = []
    for model in month_level_df["Model"].unique():
        model_data = month_level_df[month_level_df["Model"] == model].set_index("Month_Year")
        actual_row = {"Model": model, "Metric": "Actual"}
        actual_row.update({k: round(v, 2) for k, v in model_data["Actual"].to_dict().items()})
        pivot_data.append(actual_row)

        forecast_row = {"Model": model, "Metric": "Forecast"}
        forecast_row.update({k: round(v, 2) for k, v in model_data["Forecast"].to_dict().items()})
        pivot_data.append(forecast_row)

        within5_row = {"Model": model, "Metric": "Within_5pct"}
        within5_row.update(model_data["Within_5pct"].to_dict())
        pivot_data.append(within5_row)

    final_df = pd.DataFrame(pivot_data)
    month_cols = sorted(
        [col for col in final_df.columns if col not in ["Model", "Metric"]],
        key=lambda x: pd.to_datetime(x, format="%b-%y") if x not in ["Model", "Metric"] else x,
    )
    final_df = final_df[["Model", "Metric"] + month_cols]

    return {
        "Model": model_name,
        "Count_within(+/-5%)": count_within_5,
        "Total_Months": total_months,
        "Month_Level_Details": final_df,
    }


def train_and_evaluate_func(
    config: dict,
    full_train_data: pd.DataFrame,
    actual_data: pd.DataFrame,
    forecast_horizon: int,
    show_details: bool = False,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    accuracy_records = []
    all_month_level_data = []

    prop_config = config.get("prophet", {})
    rf_config = config.get("random_forest", {})
    xgb_config = config.get("xgboost", {})
    sarimax_config = config.get("sarimax", {})
    var_config = config.get("var", {})
    train = full_train_data.copy()
    train = _ensure_iq_scaled(train)
    if "holiday_month_start" in train.columns and "holiday_month_start_flag" not in train.columns:
        train["holiday_month_start_flag"] = train["holiday_month_start"].notna().astype(int)
    if "y" not in train.columns and "Final_Smoothed_Value" in train.columns:
        train["y"] = train["Final_Smoothed_Value"] / 100

    try:
        prophet_regressors = build_regressors(prop_config, train)
        holiday_df = None
        if prop_config.get("use_holidays", True) and "holiday_month_start" in train.columns:
            holiday_df = pd.DataFrame({"ds": train["holiday_month_start"].dropna().unique()})
        prophet_fc = prophet_forecast(
            train=train,
            periods=forecast_horizon,
            prop_config=prop_config,
            regressors=prophet_regressors,
            holidays=holiday_df,
        )
        result = calculate_accuracy_for_fine_tuning("Prophet", actual_data, prophet_fc)
        accuracy_records.append(result)
        all_month_level_data.append(result["Month_Level_Details"])
    except Exception:
        pass

    try:
        rf_regressors = build_regressors(rf_config, train)
        rf_holiday_feature = "holiday_month_start_flag" if rf_config.get("use_holidays", True) else None
        rf_fc = rf_forecast(
            train=train,
            periods=forecast_horizon,
            rf_config=rf_config,
            regressors=rf_regressors,
            holiday_feature=rf_holiday_feature,
        )
        result = calculate_accuracy_for_fine_tuning("Random Forest", actual_data, rf_fc)
        accuracy_records.append(result)
        all_month_level_data.append(result["Month_Level_Details"])
    except Exception:
        pass

    try:
        xgb_regressors = build_regressors(xgb_config, train)
        xgb_holiday_feature = "holiday_month_start_flag" if xgb_config.get("use_holidays", True) else None
        xgb_fc = xgb_forecast(
            train=train,
            periods=forecast_horizon,
            xgb_config=xgb_config,
            regressors=xgb_regressors,
            holiday_feature=xgb_holiday_feature,
        )
        result = calculate_accuracy_for_fine_tuning("XGBoost", actual_data, xgb_fc)
        accuracy_records.append(result)
        all_month_level_data.append(result["Month_Level_Details"])
    except Exception:
        pass

    try:
        sarimax_exog = build_regressors(sarimax_config, train)
        sarimax_fc = sarimax_forecast(
            train=train,
            periods=forecast_horizon,
            config=sarimax_config,
            exog_cols=sarimax_exog,
        )
        result = calculate_accuracy_for_fine_tuning("SARIMAX", actual_data, sarimax_fc)
        accuracy_records.append(result)
        all_month_level_data.append(result["Month_Level_Details"])
    except Exception:
        pass

    try:
        var_regressors = build_regressors(var_config, train)
        var_fc = var_forecast(
            train=train,
            periods=forecast_horizon,
            var_config=var_config,
            regressors=var_regressors,
        )
        result = calculate_accuracy_for_fine_tuning("VAR", actual_data, var_fc)
        accuracy_records.append(result)
        all_month_level_data.append(result["Month_Level_Details"])
    except Exception:
        pass

    if all_month_level_data and show_details:
        combined_month_level_df = pd.concat(all_month_level_data, ignore_index=True)
    else:
        combined_month_level_df = None

    def enhance_with_accuracy_and_separators(df: pd.DataFrame) -> pd.DataFrame:
        month_cols = [col for col in df.columns if col not in ["Model", "Metric"]]
        models = df["Model"].unique()
        enhanced_rows = []

        for i, model in enumerate(models):
            model_data = df[df["Model"] == model]
            for _, row in model_data.iterrows():
                enhanced_rows.append(row.to_dict())

            actual_row = model_data[model_data["Metric"] == "Actual"]
            forecast_row = model_data[model_data["Metric"] == "Forecast"]
            if len(actual_row) > 0 and len(forecast_row) > 0:
                ratio_values = []
                for col in month_cols:
                    actual_val = actual_row[col].iloc[0]
                    forecast_val = forecast_row[col].iloc[0]
                    if pd.notnull(actual_val) and pd.notnull(forecast_val) and actual_val != 0:
                        ratio_values.append((forecast_val / actual_val) * 100)
                    else:
                        ratio_values.append(None)

                def filter_and_summarize(values, lower, upper):
                    filtered = [val for val in values if pd.notnull(val) and lower <= val <= upper]
                    count = len(filtered)
                    total = len([val for val in values if pd.notnull(val)])
                    accuracy = count / total if total > 0 else None
                    return accuracy, count, total

                ranges = [(94.5, 105.5), (92.5, 107.5), (89.5, 110.5)]
                results = [filter_and_summarize(ratio_values, r[0], r[1]) for r in ranges]

                accuracy_summary = {
                    "Model": model,
                    "Metric": "Summary",
                    "Total_Months": results[0][2],
                    "Count_within(+/-5%)": results[0][1] if results[0][0] is not None else None,
                    "Accuracy(+/-5%)": results[0][0] * 100 if results[0][0] is not None else None,
                    "Accuracy(+/-7%)": results[1][0] * 100 if results[1][0] is not None else None,
                    "Accuracy(+/-10%)": results[2][0] * 100 if results[2][0] is not None else None,
                }

                for col in month_cols:
                    accuracy_summary[col] = None

                enhanced_rows.append(accuracy_summary)

            if i < len(models) - 1:
                separator_row = {"Model": "-" * 20, "Metric": "-" * 10}
                for col in month_cols + [
                    "Total_Months",
                    "Count_within(+/-5%)",
                    "Accuracy(+/-5%)",
                    "Accuracy(+/-7%)",
                    "Accuracy(+/-10%)",
                ]:
                    separator_row[col] = "-" * 8
                enhanced_rows.append(separator_row)

        enhanced_df = pd.DataFrame(enhanced_rows)
        ordered_cols = [
            "Model",
            "Metric",
            "Total_Months",
            "Count_within(+/-5%)",
            "Accuracy(+/-5%)",
            "Accuracy(+/-7%)",
            "Accuracy(+/-10%)",
        ]
        month_cols_sorted = sorted(month_cols, key=lambda x: pd.to_datetime(x, format="%b-%y"))
        final_cols = ordered_cols + month_cols_sorted

        for col in final_cols:
            if col not in enhanced_df.columns:
                enhanced_df[col] = None
        enhanced_df = enhanced_df[final_cols]
        return enhanced_df

    final_enhanced_df = enhance_with_accuracy_and_separators(combined_month_level_df) if combined_month_level_df is not None else None
    styled_data_frame = None
    if final_enhanced_df is not None:
        try:
            from forecasting.process_and_IQ_data import simple_style_month_forecast
            styled_data_frame = simple_style_month_forecast(final_enhanced_df)
        except Exception:
            styled_data_frame = final_enhanced_df

    accuracy_df = pd.DataFrame(accuracy_records)
    return accuracy_df, final_enhanced_df, styled_data_frame


def auto_tune_config(config: dict, accuracy_df: pd.DataFrame) -> dict:
    tuned = copy.deepcopy(config)
    if accuracy_df is None or accuracy_df.empty:
        return tuned

    model_col = _find_col(accuracy_df, ("Model", "model"))
    count_col = _find_col(accuracy_df, ("Count_within(+/-5%)", "Count_Within(+/-5%)", "Count_Within(+-5%)"))
    total_col = _find_col(accuracy_df, ("Total_Months", "Total Months", "total_months"))
    if not model_col or not count_col or not total_col:
        return tuned

    df = accuracy_df.copy()
    df[count_col] = pd.to_numeric(df[count_col], errors="coerce")
    df[total_col] = pd.to_numeric(df[total_col], errors="coerce")

    for _, row in df.iterrows():
        model = _normalize_model_name(row[model_col])
        count_within = row[count_col]
        total_months = row[total_col]
        if pd.isna(count_within) or pd.isna(total_months) or total_months <= 0:
            continue
        accuracy_goal = math.floor(0.95 * total_months)
        if count_within < accuracy_goal:
            adjustment_factor = 1 + (accuracy_goal - count_within) / total_months
        else:
            adjustment_factor = 1 - (count_within - accuracy_goal) / total_months

        if model == "prophet" and "prophet" in tuned:
            if count_within < accuracy_goal:
                tuned["prophet"]["changepoint_prior_scale"] = min(
                    1.0, tuned["prophet"]["changepoint_prior_scale"] * adjustment_factor
                )
                tuned["prophet"]["seasonality_prior_scale"] = min(
                    20, tuned["prophet"]["seasonality_prior_scale"] * adjustment_factor
                )
                tuned["prophet"]["changepoint_range"] = min(
                    1.0, tuned["prophet"].get("changepoint_range", 0.5) + 0.05
                )
            else:
                tuned["prophet"]["changepoint_prior_scale"] = max(
                    0.001, tuned["prophet"]["changepoint_prior_scale"] * adjustment_factor
                )
                tuned["prophet"]["seasonality_prior_scale"] = max(
                    1.0, tuned["prophet"]["seasonality_prior_scale"] * adjustment_factor
                )
                tuned["prophet"]["changepoint_range"] = max(
                    0.1, tuned["prophet"].get("changepoint_range", 0.5) - 0.05
                )
        elif model == "random_forest" and "random_forest" in tuned:
            if count_within < accuracy_goal:
                increase = max(1, int((accuracy_goal - count_within) // 2))
                tuned["random_forest"]["max_depth"] = min(
                    50, int(tuned["random_forest"]["max_depth"]) + increase
                )
            else:
                decrease = max(1, int((count_within - accuracy_goal) // 2))
                tuned["random_forest"]["max_depth"] = max(
                    3, int(tuned["random_forest"]["max_depth"]) - decrease
                )
        elif model == "xgboost" and "xgboost" in tuned:
            if count_within < accuracy_goal:
                increase = max(1, int((accuracy_goal - count_within) // 2))
                tuned["xgboost"]["max_depth"] = min(
                    20, int(tuned["xgboost"]["max_depth"]) + increase
                )
                tuned["xgboost"]["learning_rate"] = max(
                    0.01, tuned["xgboost"]["learning_rate"] * 0.9
                )
            else:
                decrease = max(1, int((count_within - accuracy_goal) // 2))
                tuned["xgboost"]["max_depth"] = max(
                    1, int(tuned["xgboost"]["max_depth"]) - decrease
                )
                tuned["xgboost"]["learning_rate"] = min(
                    0.2, tuned["xgboost"]["learning_rate"] * 1.1
                )
        elif model == "sarimax" and "sarimax" in tuned:
            order = list(tuned["sarimax"].get("order", (1, 1, 1)))
            seasonal_order = list(tuned["sarimax"].get("seasonal_order", (1, 1, 1, 12)))
            if count_within < accuracy_goal:
                order[0] = min(5, order[0] + 1)
                seasonal_order[0] = min(5, seasonal_order[0] + 1)
            else:
                order[0] = max(1, order[0] - 1)
                seasonal_order[0] = max(1, seasonal_order[0] - 1)
            tuned["sarimax"]["order"] = tuple(order)
            tuned["sarimax"]["seasonal_order"] = tuple(seasonal_order)
        elif model == "var" and "var" in tuned:
            if count_within < accuracy_goal:
                tuned["var"]["lags"] = min(24, int(tuned["var"]["lags"]) + 1)
            else:
                tuned["var"]["lags"] = max(1, int(tuned["var"]["lags"]) - 1)
    return tuned


def record_config_random_states(config: dict) -> dict:
    random_states = {}
    for model, params in config.items():
        if isinstance(params, dict) and "random_state" in params:
            random_states[model] = params["random_state"]
    return random_states


def update_random_states_for_iteration(config: dict, iteration: int, base_seed: int = 42) -> dict:
    updated_states = {}
    for model in config:
        if isinstance(config[model], dict) and "random_state" in config[model]:
            new_seed = base_seed + iteration
            config[model]["random_state"] = new_seed
            updated_states[model] = new_seed
    return updated_states


def iterative_tuning(
    initial_config: dict,
    initial_accuracy_df: pd.DataFrame,
    train_and_evaluate_func,
    full_train_data: pd.DataFrame,
    forecast_horizon: Optional[int] = None,
    actual_data: Optional[pd.DataFrame] = None,
    max_iterations: int = 1,
    improvement_threshold: float = 0.05,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if initial_accuracy_df is None or initial_accuracy_df.empty:
        return initial_config, initial_accuracy_df, initial_accuracy_df

    config_history = []
    first_config = copy.deepcopy(initial_config)

    accuracy_df = initial_accuracy_df.copy()
    model_col = _find_col(accuracy_df, ("Model", "model"))
    count_col = _find_col(accuracy_df, ("Count_within(+/-5%)", "Count_Within(+/-5%)", "Count_Within(+-5%)"))
    total_col = _find_col(accuracy_df, ("Total_Months", "Total Months", "total_months"))

    if not model_col or not count_col or not total_col:
        return initial_config, initial_accuracy_df, initial_accuracy_df

    accuracy_df["Model_norm"] = accuracy_df[model_col].apply(_normalize_model_name)
    accuracy_df[count_col] = pd.to_numeric(accuracy_df[count_col], errors="coerce")
    accuracy_df[total_col] = pd.to_numeric(accuracy_df[total_col], errors="coerce")

    goal_count = np.floor(0.95 * accuracy_df[total_col])
    meeting_goal = accuracy_df[accuracy_df[count_col] >= goal_count]["Model_norm"].dropna().unique().tolist()
    models_meeting_goal = set(meeting_goal)

    best_config = {k.lower(): v for k, v in first_config.items() if k != "general"}

    for iteration in range(max_iterations):
        models_to_tune = set(best_config.keys()) - models_meeting_goal
        if not models_to_tune:
            break

        config_subset = {m: copy.deepcopy(best_config[m]) for m in models_to_tune if m in best_config}
        acc_subset = accuracy_df[accuracy_df["Model_norm"].isin(models_to_tune)].copy()
        if not acc_subset.empty:
            acc_subset = acc_subset.rename(columns={model_col: "Model"})
            acc_subset["Model"] = acc_subset["Model_norm"]

        tuned_config = auto_tune_config(config_subset, acc_subset)
        config_history.append({"iteration": iteration + 1, "config": copy.deepcopy(tuned_config)})

        if forecast_horizon is None:
            forecast_horizon = 12
        new_accuracy_df, _, _ = train_and_evaluate_func(
            {**best_config, **tuned_config}, full_train_data, actual_data, forecast_horizon, show_details=False
        )
        if new_accuracy_df is None or new_accuracy_df.empty:
            break

        new_accuracy_df["Model_norm"] = new_accuracy_df["Model"].apply(_normalize_model_name)
        new_accuracy_df[count_col] = pd.to_numeric(new_accuracy_df[count_col], errors="coerce")
        new_accuracy_df[total_col] = pd.to_numeric(new_accuracy_df[total_col], errors="coerce")

        for model in models_to_tune:
            prev_row = accuracy_df[accuracy_df["Model_norm"] == model]
            new_row = new_accuracy_df[new_accuracy_df["Model_norm"] == model]
            if prev_row.empty or new_row.empty:
                continue
            prev_acc = (prev_row[count_col] / prev_row[total_col]).values[0]
            new_acc = (new_row[count_col] / new_row[total_col]).values[0]
            if new_acc - prev_acc >= improvement_threshold:
                best_config[model] = tuned_config.get(model, best_config.get(model))

        accuracy_df = new_accuracy_df.copy()

    filtered_tuned_accuracy_df = accuracy_df.copy()
    return best_config, initial_accuracy_df, filtered_tuned_accuracy_df


def create_styled_percentage_chart(
    df: pd.DataFrame,
    metric: str,
    title: str,
    custom_colors: Optional[list[str]] = None,
    line_styles: Optional[list[str]] = None,
    line_width: int = 3,
    show_text: bool = True,
) -> go.Figure:
    if custom_colors is None:
        custom_colors = ["teal", "orange", "purple", "darkgoldenrod", "green", "red"]
    if line_styles is None:
        line_styles = ["dashdot"] * len(custom_colors)
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    df = df.copy()
    df["Month"] = pd.Categorical(df["Month"], categories=month_order, ordered=True)
    all_years = df["Year"].unique()
    full_index = pd.MultiIndex.from_product([all_years, month_order], names=["Year", "Month"])
    df_full = pd.DataFrame(index=full_index).reset_index().merge(df, on=["Year", "Month"], how="left")
    fig = go.Figure()

    for i, year in enumerate(all_years):
        group = df_full[df_full["Year"] == year]
        if i <= 1:
            offset = 0.0005
            text_position = "top center"
        else:
            offset = -0.0005
            text_position = "bottom left"

        fig.add_trace(
            go.Scatter(
                x=group["Month"],
                y=group[metric],
                mode="lines+markers",
                name=str(year),
                line=dict(color=custom_colors[i % len(custom_colors)], width=line_width, dash=line_styles[i % len(line_styles)]),
                marker=dict(size=6),
            )
        )

        if show_text:
            fig.add_trace(
                go.Scatter(
                    x=group["Month"],
                    y=group[metric] + offset,
                    mode="text",
                    text=[f"{float(v) * 100:.2f}%" if pd.notna(v) else "" for v in group[metric]],
                    textposition=text_position,
                    textfont=dict(size=14, color="darkblue", family="Arial"),
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Percentage",
        legend_title="Year",
        template="plotly_white",
        width=900,
        height=500,
    )
    return fig


def build_regressors(config_section: dict, df: pd.DataFrame) -> list[str]:
    regressors = []
    if config_section.get("use_iq_value_scaled", True) and "IQ_value_scaled" in df.columns:
        regressors.append("IQ_value_scaled")
    if config_section.get("use_holidays", True) and "holiday_month_start_flag" in df.columns:
        regressors.append("holiday_month_start_flag")
    return regressors


def run_prophet_smoothing(
    df_long: pd.DataFrame,
    forecast_months: int,
    holiday_original_date: Optional[Any],
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_long = df_long.copy()
    if "Normalized_Ratio_1" not in df_long.columns:
        raise ValueError("'Normalized_Ratio_1' column not found in input data")
    df_long["y"] = df_long["Normalized_Ratio_1"].copy()
    scaler = MinMaxScaler()
    df_long["IQ_value_scaled"] = scaler.fit_transform(df_long[["IQ_value"]]).round(2)

    prop_config = config.get("prophet", {}) if config else {}
    prophet_regressors = []
    if prop_config.get("use_iq_value_scaled", True):
        prophet_regressors.append("IQ_value_scaled")

    holiday_df = None
    if holiday_original_date is not None and prop_config.get("use_holidays", True):
        holiday_df = _normalize_holidays(holiday_original_date)

    yearly_value = prop_config.get("yearly_fourier_order")
    if yearly_value is None:
        yearly_value = prop_config.get("monthly_fourier_order", prop_config.get("yearly_seasonality", True))
    model = Prophet(
        changepoint_prior_scale=prop_config.get("changepoint_prior_scale", 0.05),
        seasonality_prior_scale=prop_config.get("seasonality_prior_scale", 0.1),
        holidays_prior_scale=prop_config.get("holidays_prior_scale", 0.1),
        yearly_seasonality=yearly_value,
        weekly_seasonality=prop_config.get("weekly_seasonality", False),
        daily_seasonality=prop_config.get("daily_seasonality", False),
        holidays=holiday_df,
    )
    for reg in prophet_regressors:
        if reg in df_long.columns:
            model.add_regressor(reg)

    model.fit(df_long[["ds", "y"] + prophet_regressors])
    future = model.make_future_dataframe(periods=forecast_months, freq="M")
    for reg in prophet_regressors:
        if reg in df_long.columns:
            last_val = df_long[reg].iloc[-1]
            future[reg] = list(df_long[reg]) + [last_val] * forecast_months

    forecast = model.predict(future)
    df_long["Normalized_Ratio_Post_Prophet"] = forecast["yhat"][: len(df_long)].round(2)
    df_long["Normalized_Volume"] = df_long["Normalized_Ratio_Post_Prophet"] * df_long["IQ_value"].round(0)
    df_long["Final_Smoothed_Value"] = df_long["Normalized_Ratio_Post_Prophet"].round(2)
    if "Contact_Ratio" in df_long.columns:
        df_long["Contact_Ratio"] = (df_long["Contact_Ratio"] * 100).round(2)

    return df_long, forecast


def run_contact_ratio_forecast(
    smoothing_results: pd.DataFrame,
    forecast_months: int,
    holiday_mapping: Optional[dict],
    holiday_original_date: Optional[Any],
    config: dict,
) -> dict:
    df_smoothed = _prep_smoothed_df(smoothing_results)
    if df_smoothed.empty:
        return {"errors": {"data": "No data supplied"}}

    df_smoothed["IQ_value_scaled"] = pd.to_numeric(df_smoothed["IQ_value_scaled"], errors="coerce")
    nan_count = df_smoothed["IQ_value_scaled"].isna().sum()
    if nan_count > 0:
        df_smoothed["IQ_value_scaled"].fillna(method="ffill", inplace=True)
        if df_smoothed["IQ_value_scaled"].isna().sum() > 0:
            df_smoothed["IQ_value_scaled"].fillna(0, inplace=True)

    train, test, forecast_months_test_range = create_flexible_12_month_test_split(df_smoothed)
    train["holiday_month_start_flag"] = train["holiday_month_start"].notna().astype(int)
    test["holiday_month_start_flag"] = test["holiday_month_start"].notna().astype(int)

    rf_config = config.get("random_forest", {})
    xgb_config = config.get("xgboost", {})
    var_config = config.get("var", {})
    sarimax_config = config.get("sarimax", {})
    general_config = config.get("general", {})
    prop_config = config.get("prophet", {})

    rf_regressors = []
    xgb_regressors = []
    var_regressors = []
    sarimax_exog = []

    if rf_config.get("use_iq_value_scaled", True):
        rf_regressors.append("IQ_value_scaled")
    rf_holiday_feature = "holiday_month_start_flag" if rf_config.get("use_holidays", True) else None

    if xgb_config.get("use_iq_value_scaled", True):
        xgb_regressors.append("IQ_value_scaled")
    xgb_holiday_feature = "holiday_month_start_flag" if xgb_config.get("use_holidays", True) else None

    if var_config.get("use_iq_value_scaled", True):
        var_regressors.append("IQ_value_scaled")
    if var_config.get("use_holidays", True):
        var_regressors.append("holiday_month_start_flag")

    if sarimax_config.get("use_iq_value_scaled", True):
        sarimax_exog.append("IQ_value_scaled")
    if sarimax_config.get("use_holidays", True):
        sarimax_exog.append("holiday_month_start_flag")

    train["y"] = train["Final_Smoothed_Value"] / 100
    prophet_regressors = []
    if prop_config.get("use_iq_value_scaled", True):
        prophet_regressors.append("IQ_value_scaled")
    prophet_holidays = _normalize_holidays(holiday_original_date) if prop_config.get("use_holidays", True) else None

    forecast_results: dict[str, pd.DataFrame] = {}
    error_messages: dict[str, str] = {}

    try:
        forecast_results["prophet"] = prophet_forecast(
            train,
            forecast_months_test_range,
            seasonality=general_config.get("use_seasonality", True),
            regressors=prophet_regressors,
            prop_config=prop_config,
            holidays=prophet_holidays,
        )
    except Exception as exc:
        error_messages["prophet"] = str(exc)

    try:
        forecast_results["random_forest"] = rf_forecast(
            train,
            forecast_months_test_range,
            regressors=rf_regressors,
            holiday_feature=rf_holiday_feature,
            rf_config=rf_config,
        )
    except Exception as exc:
        error_messages["rf"] = str(exc)

    try:
        forecast_results["xgboost"] = xgb_forecast(
            train,
            forecast_months_test_range,
            regressors=xgb_regressors,
            holiday_feature=xgb_holiday_feature,
            xgb_config=xgb_config,
        )
    except Exception as exc:
        error_messages["xgb"] = str(exc)

    try:
        forecast_results["var"] = var_forecast(
            train,
            forecast_months_test_range,
            regressors=var_regressors,
            var_config=var_config,
        )
    except Exception as exc:
        error_messages["var"] = str(exc)

    try:
        forecast_results["sarimax"] = sarimax_forecast(
            train,
            forecast_months_test_range,
            config=sarimax_config,
            exog_cols=sarimax_exog,
        )
    except Exception as exc:
        error_messages["sarimax"] = str(exc)

    df_smoothed["Date"] = pd.to_datetime(df_smoothed["ds"]).dt.to_period("M").dt.to_timestamp()

    return {
        "final_smoothed_values": df_smoothed[["Final_Smoothed_Value", "Date"]],
        "train": train,
        "test": test,
        "forecast_results": forecast_results,
        "errors": error_messages,
        "debug": {
            "df_smoothed_head": df_smoothed.head().to_dict(),
            "train_head": train.head().to_dict(),
            "test_head": test.head().to_dict(),
        },
    }


def run_contact_ratio_forecast_phase2(
    smoothing_results: pd.DataFrame,
    forecast_months: int,
    holiday_mapping: Optional[dict],
    holiday_original_date: Optional[Any],
    config: dict,
) -> dict:
    df_smoothed = _prep_smoothed_df(smoothing_results)
    if df_smoothed.empty:
        return {"forecast_results": {}, "error_messages": {"data": "No data supplied"}}

    df_smoothed["IQ_value_scaled"] = pd.to_numeric(df_smoothed["IQ_value_scaled"], errors="coerce")
    if df_smoothed["IQ_value_scaled"].isna().sum() > 0:
        df_smoothed["IQ_value_scaled"].fillna(method="ffill", inplace=True)
        if df_smoothed["IQ_value_scaled"].isna().sum() > 0:
            df_smoothed["IQ_value_scaled"].fillna(0, inplace=True)

    full_train = df_smoothed.copy()
    full_train["holiday_month_start_flag"] = full_train["holiday_month_start"].notna().astype(int)
    full_train["y"] = full_train["Final_Smoothed_Value"] / 100

    rf_config = config.get("random_forest", {})
    xgb_config = config.get("xgboost", {})
    var_config = config.get("var", {})
    sarimax_config = config.get("sarimax", {})
    prop_config = config.get("prophet", {})
    general_config = config.get("general", {})

    rf_regressors = []
    xgb_regressors = []
    var_regressors = []
    sarimax_exog = []

    if rf_config.get("use_iq_value_scaled", True):
        rf_regressors.append("IQ_value_scaled")
    rf_holiday_feature = "holiday_month_start_flag" if rf_config.get("use_holidays", True) else None

    if xgb_config.get("use_iq_value_scaled", True):
        xgb_regressors.append("IQ_value_scaled")
    xgb_holiday_feature = "holiday_month_start_flag" if xgb_config.get("use_holidays", True) else None

    if var_config.get("use_iq_value_scaled", True):
        var_regressors.append("IQ_value_scaled")
    if var_config.get("use_holidays", True):
        var_regressors.append("holiday_month_start_flag")

    if sarimax_config.get("use_iq_value_scaled", True):
        sarimax_exog.append("IQ_value_scaled")
    if sarimax_config.get("use_holidays", True):
        sarimax_exog.append("holiday_month_start_flag")

    prophet_regressors = []
    if prop_config.get("use_iq_value_scaled", True):
        prophet_regressors.append("IQ_value_scaled")
    prophet_holidays = _normalize_holidays(holiday_original_date) if prop_config.get("use_holidays", True) else None

    forecast_results: dict[str, pd.DataFrame] = {}
    error_messages: dict[str, str] = {}

    try:
        forecast_results["prophet"] = prophet_forecast(
            full_train,
            forecast_months,
            seasonality=general_config.get("use_seasonality", True),
            regressors=prophet_regressors,
            prop_config=prop_config,
            holidays=prophet_holidays,
        )
    except Exception as exc:
        error_messages["prophet"] = str(exc)

    try:
        forecast_results["random_forest"] = rf_forecast(
            full_train,
            forecast_months,
            regressors=rf_regressors,
            holiday_feature=rf_holiday_feature,
            rf_config=rf_config,
        )
    except Exception as exc:
        error_messages["rf"] = str(exc)

    try:
        forecast_results["xgboost"] = xgb_forecast(
            full_train,
            forecast_months,
            regressors=xgb_regressors,
            holiday_feature=xgb_holiday_feature,
            xgb_config=xgb_config,
        )
    except Exception as exc:
        error_messages["xgb"] = str(exc)

    try:
        forecast_results["var"] = var_forecast(
            full_train,
            forecast_months,
            regressors=var_regressors,
            var_config=var_config,
        )
    except Exception as exc:
        error_messages["var"] = str(exc)

    try:
        forecast_results["sarimax"] = sarimax_forecast(
            full_train,
            forecast_months,
            config=sarimax_config,
            exog_cols=sarimax_exog,
        )
    except Exception as exc:
        error_messages["sarimax"] = str(exc)

    df_smoothed["Date"] = pd.to_datetime(df_smoothed["ds"]).dt.to_period("M").dt.to_timestamp()

    return {
        "final_smoothed_values": df_smoothed[["Final_Smoothed_Value", "Date"]],
        "full_training_data": full_train,
        "training_data_size": len(full_train),
        "forecast_horizon": forecast_months,
        "forecast_results": forecast_results,
        "error_messages": error_messages,
        "config_used": config,
        "phase": "Phase 2 - Production Forecasting",
    }


def run_phase2_forecast(df: pd.DataFrame, forecast_months: int, config: Optional[dict] = None) -> Dict[str, Any]:
    cfg = config or config_manager.load_config()
    prepared = _prep_input(df)
    prepared["holiday_month_start_flag"] = prepared["holiday_month_start"].notna().astype(int)
    prepared["y"] = _safe_num(prepared["Final_Smoothed_Value"]) / 100.0

    holiday_original_date = None
    if "holiday_month_start" in prepared.columns and prepared["holiday_month_start"].notna().any():
        holiday_original_date = pd.DataFrame({"ds": prepared["holiday_month_start"].dropna().unique()})

    # Ensure supported datetime index for downstream models
    try:
        idx = pd.DatetimeIndex(prepared["ds"], name="ds_idx")
        if idx.freq is None:
            inferred = pd.infer_freq(idx)
            if inferred:
                idx = pd.DatetimeIndex(idx, freq=inferred, name="ds_idx")
            else:
                idx = pd.DatetimeIndex(idx.to_period("M").to_timestamp(), freq="MS", name="ds_idx")
        prepared = prepared.set_index(idx, drop=False)
    except Exception:
        prepared = prepared.reset_index(drop=True)

    res = run_contact_ratio_forecast_phase2(
        prepared,
        forecast_months,
        holiday_mapping=None,
        holiday_original_date=holiday_original_date,
        config=cfg,
    )
    forecast_results = res.get("forecast_results", {})
    if "final_smoothed_values" in res:
        forecast_results["final_smoothed_values"] = res["final_smoothed_values"]

    return {
        "forecast_results": forecast_results,
        "errors": res.get("error_messages", {}),
        "config": cfg,
    }
