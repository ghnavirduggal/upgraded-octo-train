from __future__ import annotations
import calendar
import io
import math
import re
import traceback
import warnings
from datetime import datetime
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# LOAD + PREPROCESS HELPERS
# ---------------------------------------------------------------------------

def _load_volume_sheet(file) -> pd.DataFrame:
    df_raw = pd.read_excel(file, sheet_name="Volume")
    df_raw.columns = [str(col).strip().lower().replace(" ", "_") for col in df_raw.columns]

    if "date" not in df_raw.columns or "volume" not in df_raw.columns:
        raise ValueError("Dataset must contain 'date' and 'volume' columns")

    df = df_raw.copy()
    df["volume"] = df["volume"].astype(str).str.replace(",", "", regex=False).str.strip()
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.strftime("%b")
    df["day"] = df["date"].dt.strftime("%a")
    df["date_label"] = df["date"].dt.strftime("%d-%b-%Y")
    df["week"] = df["date"].dt.to_period("W").apply(
        lambda r: r.start_time.strftime("%d-%b-%Y")
    )
    return df


def load_and_preprocess(file) -> tuple[pd.DataFrame, dict]:
    df = _load_volume_sheet(file)
    df["week"] = df["date"].dt.to_period("W").apply(
        lambda r: r.start_time.strftime("%d-%b-%Y")
    )

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError(
            "'date' column could not be converted to datetime. "
            "Please check the format in your Excel file."
        )

    holiday_mapping = {}
    try:
        df_holidays = pd.read_excel(file, sheet_name=1)
        df_holidays.columns = [
            str(col).strip().lower().replace(" ", "_")
            for col in df_holidays.columns
        ]
        if "date" not in df_holidays.columns or "holidays" not in df_holidays.columns:
            raise ValueError("Holidays sheet must have 'Date' and 'Holidays' columns")

        df_holidays["date"] = pd.to_datetime(df_holidays["date"], errors="coerce")
        df_holidays = df_holidays.dropna(subset=["date", "holidays"])
        holiday_mapping = dict(zip(df_holidays["date"], df_holidays["holidays"]))
    except Exception:
        holiday_mapping = {}

    return df, holiday_mapping


def IQ_data(file, font_color: str = "darkblue", font_size: int = 30) -> Dict[str, pd.DataFrame]:
    try:
        try:
            df_iq = pd.read_excel(file, sheet_name="IQ_Data")
        except ValueError:
            df_iq = pd.read_excel(file, sheet_name="IQ Data")
        df_vol = pd.read_excel(file, sheet_name="Volume")
    except ValueError as exc:
        warnings.warn(f"Required sheet not found: {exc}")
        return {}

    if "Category" not in df_iq.columns:
        warnings.warn("Missing 'Category' in IQ Data")
        return {}

    if "Service_Category_Sub_Service" not in df_vol.columns and "Category" not in df_vol.columns:
        warnings.warn("Missing 'Category' in Volume sheet")
        return {}

    id_cols = ["Category", "Category_Sub_Service"] if "Category_Sub_Service" in df_iq.columns else ["Category"]
    value_cols = [c for c in df_iq.columns if c not in id_cols]

    df_long = df_iq.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="Period",
        value_name="Value",
    )
    df_long["Period_dt"] = pd.to_datetime(df_long["Period"], errors="coerce")
    invalid_periods = df_long[df_long["Period_dt"].isna()]["Period"].unique().tolist()
    if invalid_periods:
        warnings.warn(f"Invalid Period values: {invalid_periods[:5]}")

    df_long["Year"] = df_long["Period_dt"].dt.year
    df_long["Month"] = df_long["Period_dt"].dt.month

    group_cols = id_cols + ["Year", "Month"]
    df_long = df_long.dropna(subset=["Year", "Month"])
    grouped_iq = df_long.groupby(group_cols)["Value"].mean().reset_index(name="Avg_vol")

    month_map = {i: calendar.month_abbr[i] for i in range(1, 13)}
    result_dfs: Dict[str, pd.DataFrame] = {}

    for cat in grouped_iq["Category"].unique():
        df_cat = grouped_iq[grouped_iq["Category"] == cat].copy()

        pivot_iq = df_cat.pivot_table(
            index=["Year", "Category_Sub_Service"] if "Category_Sub_Service" in df_cat.columns else ["Year"],
            columns="Month",
            values="Avg_vol",
            aggfunc="mean",
        ).reset_index()

        pivot_iq.rename(columns=month_map, inplace=True)
        month_cols = [m for m in month_map.values() if m in pivot_iq.columns]
        pivot_iq["Yearly_Avg"] = pivot_iq[month_cols].mean(axis=1)

        pivot_iq["Growth_%"] = None
        if "Category_Sub_Service" in pivot_iq.columns:
            for sub in pivot_iq["Category_Sub_Service"].unique():
                sub_idx = pivot_iq["Category_Sub_Service"] == sub
                growth = pivot_iq[sub_idx].sort_values("Year")["Yearly_Avg"].pct_change().values
                pivot_iq.loc[sub_idx, "Growth_%"] = growth
        else:
            growth = pivot_iq.sort_values("Year")["Yearly_Avg"].pct_change().values
            pivot_iq["Growth_%"] = growth

        df_vol["Date"] = pd.to_datetime(df_vol["Date"], errors="coerce")
        df_vol["Year"] = df_vol["Date"].dt.year
        df_vol["Month"] = df_vol["Date"].dt.month

        df_vol_cat = df_vol[
            df_vol["Category"].str.strip().str.lower() == str(cat).strip().lower()
        ].copy()
        df_vol_cat["Volume"] = (
            df_vol_cat["Volume"].astype(str).str.replace(",", "", regex=False).astype(float)
        )

        if not df_vol_cat.empty:
            pivot_vol = df_vol_cat.pivot_table(
                index=["Year", "Category"],
                columns="Month",
                values="Volume",
                aggfunc="sum",
            ).reset_index()

            pivot_vol.rename(columns=month_map, inplace=True)
            existing_months = [m for m in month_map.values() if m in pivot_vol.columns]
            pivot_vol["Yearly_Avg"] = pivot_vol[existing_months].mean(axis=1)

            pivot_vol["Growth_%"] = None
            for sub in pivot_vol["Category"].unique():
                sub_idx = pivot_vol["Category"] == sub
                growth = pivot_vol[sub_idx].sort_values("Year")["Yearly_Avg"].pct_change().values
                pivot_vol.loc[sub_idx, "Growth_%"] = growth
        else:
            pivot_vol = pd.DataFrame(columns=pivot_iq.columns)

        pivot_iq_cat = pivot_iq.copy()
        pivot_vol_cat = pivot_vol.copy()
        pivot_ratio = pivot_vol_cat.copy()

        for col in month_cols + ["Yearly_Avg"]:
            if col in pivot_vol_cat.columns and col in pivot_iq_cat.columns:

                def _calc_ratio(row, col=col):
                    try:
                        v_val = row[col]
                        iq_val = pivot_iq_cat.loc[pivot_iq_cat["Year"] == row["Year"], col].values[0]
                        if pd.isna(v_val) or pd.isna(iq_val) or iq_val == 0:
                            return None
                        return v_val / iq_val
                    except Exception:
                        return None

                pivot_ratio[col] = pivot_ratio.apply(_calc_ratio, axis=1)

        pivot_ratio["Growth_%"] = ""
        pivot_ratio["Growth_%"] = pivot_ratio["Yearly_Avg"]

        def fmt_millions(x):
            try:
                return f"{float(x) / 1_000_000:.2f}M" if pd.notna(x) else ""
            except Exception:
                return ""

        def fmt_thousands(x):
            try:
                return f"{float(x) / 1_000:.1f}K" if pd.notna(x) else ""
            except Exception:
                return ""

        def fmt_percent(x):
            try:
                return f"{float(x) * 100:.1f}%" if pd.notna(x) else ""
            except Exception:
                return ""

        display_iq = pivot_iq.copy()
        display_vol = pivot_vol.copy()
        display_ratio = pivot_ratio.copy()

        for col in month_cols + ["Yearly_Avg"]:
            if col in display_iq.columns:
                display_iq[col] = display_iq[col].apply(fmt_millions)
        if "Growth_%" in display_iq.columns:
            display_iq["Growth_%"] = display_iq["Growth_%"].apply(fmt_percent)

        for col in month_cols + ["Yearly_Avg"]:
            if col in display_vol.columns:
                display_vol[col] = display_vol[col].apply(fmt_thousands)
        if "Growth_%" in display_vol.columns:
            display_vol["Growth_%"] = display_vol["Growth_%"].apply(fmt_percent)

        for col in month_cols + ["Yearly_Avg"]:
            if col in display_ratio.columns:
                display_ratio[col] = display_ratio[col].apply(fmt_percent)

        ordered_cols = ["Year"] + month_cols + ["Yearly_Avg", "Growth_%"]
        display_iq = display_iq[[c for c in ordered_cols if c in display_iq.columns]]
        display_vol = display_vol[[c for c in ordered_cols if c in display_vol.columns]]
        display_ratio = display_ratio[[c for c in ordered_cols if c in display_ratio.columns]]

        result_dfs[cat] = {
            "IQ": display_iq,
            "Volume": display_vol,
            "Contact_Ratio": display_ratio,
        }

    return result_dfs


# ---------------------------------------------------------------------------
# MAIN PIVOT + LONG STYLE FOR FORECAST GROUP
# ---------------------------------------------------------------------------

def forecast_group_pivot_and_long_style(df: pd.DataFrame, cat: str):
    """
    Build:
      1) Display pivot by forecast_group x Year (with growth%)
      2) Volume-split pivot (% by month, incl. Avg & last-3m)
      3) Long-style original data
      4) Long-style monthly aggregated
      5) Long-style daily aggregated
    Filtered to a single category.
    """
    forecast_group_data = df[df["category"] == cat].copy()
    if forecast_group_data.empty:
        return None, None, None, None, None

    forecast_group_data["Date"] = pd.to_datetime(
        forecast_group_data["date"], errors="coerce"
    )
    forecast_group_data["Year"] = forecast_group_data["Date"].dt.year
    forecast_group_data["Month"] = forecast_group_data["Date"].dt.month
    forecast_group_data["Day"] = forecast_group_data["Date"].dt.day
    forecast_group_data["Month_Name"] = forecast_group_data["Date"].dt.strftime("%b")

    forecast_group_data["Volume"] = (
        forecast_group_data["volume"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    month_map = {i: calendar.month_abbr[i] for i in range(1, 13)}

    pivot_forecast_group = forecast_group_data.pivot_table(
        index=["Year", "forecast_group"],
        columns="Month",
        values="Volume",
        aggfunc="sum",
    ).reset_index()

    pivot_forecast_group.rename(columns=month_map, inplace=True)
    month_cols = [m for m in month_map.values() if m in pivot_forecast_group.columns]

    pivot_forecast_group["Yearly_Avg"] = pivot_forecast_group[month_cols].mean(axis=1)

    pivot_forecast_group["Growth_%"] = None
    for fg in pivot_forecast_group["forecast_group"].unique():
        fg_idx = pivot_forecast_group["forecast_group"] == fg
        growth = (
            pivot_forecast_group[fg_idx]
            .sort_values("Year")["Yearly_Avg"]
            .pct_change()
            .values
        )
        pivot_forecast_group.loc[fg_idx, "Growth_%"] = growth

    pivot_forecast_group = pivot_forecast_group.sort_values(
        ["forecast_group", "Year"]
    ).reset_index(drop=True)

    def insert_blank_rows(df_in: pd.DataFrame) -> pd.DataFrame:
        """Insert separator rows between forecast groups."""
        result_rows = []
        prev_fg = None

        for _, row in df_in.iterrows():
            curr_fg = row["forecast_group"]
            if prev_fg is not None and curr_fg != prev_fg:
                blank_row = pd.Series(
                    ["-----------"] * len(df_in.columns), index=df_in.columns
                )
                result_rows.append(blank_row)
            result_rows.append(row)
            prev_fg = curr_fg

        result_df = pd.DataFrame(result_rows).reset_index(drop=True)
        return result_df.astype(str)

    pivot_with_blanks = insert_blank_rows(pivot_forecast_group)

    def fmt_thousands(x: Any) -> str:
        try:
            if str(x) == "-----------":
                return "-----------"
            return f"{float(x) / 1_000:,.1f}k" if pd.notna(x) and x != "" else ""
        except Exception:
            return ""

    def fmt_percent(x: Any) -> str:
        try:
            if str(x) == "-----------":
                return "-----------"
            return f"{float(x) * 100:.1f}%" if pd.notna(x) and x != "" else ""
        except Exception:
            return ""

    forecast_group_display_pivot = pivot_with_blanks.copy()
    for col in month_cols + ["Yearly_Avg"]:
        if col in forecast_group_display_pivot.columns:
            forecast_group_display_pivot[col] = forecast_group_display_pivot[col].apply(
                fmt_thousands
            )

    if "Growth_%" in forecast_group_display_pivot.columns:
        forecast_group_display_pivot["Growth_%"] = forecast_group_display_pivot[
            "Growth_%"
        ].apply(fmt_percent)

    ordered_cols = ["Year", "forecast_group"] + month_cols + ["Yearly_Avg", "Growth_%"]
    forecast_group_display_pivot = forecast_group_display_pivot[
        [c for c in ordered_cols if c in forecast_group_display_pivot.columns]
    ]

    volume_split_data = (
        forecast_group_data.groupby(["Year", "forecast_group", "Month"])
        .agg({"Volume": "sum"})
        .reset_index()
    )

    volume_split_pivot = volume_split_data.pivot_table(
        index=["Year", "forecast_group"],
        columns="Month",
        values="Volume",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    volume_split_pivot.rename(columns=month_map, inplace=True)
    available_months = [m for m in month_cols if m in volume_split_pivot.columns]

    for year in volume_split_pivot["Year"].unique():
        year_mask = volume_split_pivot["Year"] == year
        for month in available_months:
            if month in volume_split_pivot.columns:
                month_total = volume_split_pivot.loc[year_mask, month].sum()
                if month_total > 0:
                    volume_split_pivot.loc[year_mask, month] = (
                        volume_split_pivot.loc[year_mask, month] / month_total * 100
                    ).round(1)
                else:
                    volume_split_pivot.loc[year_mask, month] = pd.NA

    def _avg_excluding_na(row: pd.Series, months: Sequence[str]):
        values = [row[m] for m in months if pd.notna(row[m]) and row[m] > 0]
        return round(sum(values) / len(values), 1) if values else pd.NA

    volume_split_pivot["Avg"] = volume_split_pivot.apply(
        lambda r: _avg_excluding_na(r, available_months), axis=1
    )

    def _last_3m(row: pd.Series, months: Sequence[str]):
        months_with_data = [m for m in months if pd.notna(row[m]) and row[m] > 0]
        if len(months_with_data) >= 3:
            last3 = months_with_data[-3:]
            values = [row[m] for m in last3]
            return round(sum(values) / len(values), 1)
        if months_with_data:
            values = [row[m] for m in months_with_data]
            return round(sum(values) / len(values), 1)
        return pd.NA

    volume_split_pivot["Vol_Split_Last_3M"] = volume_split_pivot.apply(
        lambda r: _last_3m(r, available_months), axis=1
    )

    volume_split_pivot = volume_split_pivot.sort_values(
        ["forecast_group", "Year"]
    ).reset_index(drop=True)

    volume_split_with_blanks = insert_blank_rows(volume_split_pivot)

    def fmt_percentage(x: Any) -> str:
        try:
            if str(x) == "-----------":
                return "-----------"
            if pd.isna(x):
                return ""
            return f"{float(x):.2f}%"
        except Exception:
            return ""

    forecast_group_volume_split = volume_split_with_blanks.copy()
    for col in available_months + ["Avg", "Vol_Split_Last_3M"]:
        if col in forecast_group_volume_split.columns:
            forecast_group_volume_split[col] = forecast_group_volume_split[col].apply(
                fmt_percentage
            )

    split_cols = ["Year", "forecast_group"] + available_months + [
        "Avg",
        "Vol_Split_Last_3M",
    ]
    forecast_group_volume_split = forecast_group_volume_split[
        [c for c in split_cols if c in forecast_group_volume_split.columns]
    ]

    forecast_group_long_original = forecast_group_data.copy()

    forecast_group_long_monthly = (
        forecast_group_data.groupby(["Year", "Month", "Month_Name", "forecast_group"])
        .agg({"Volume": "sum", "Date": "count"})
        .reset_index()
    )
    forecast_group_long_monthly.rename(columns={"Date": "Days_Count"}, inplace=True)
    forecast_group_long_monthly = forecast_group_long_monthly.sort_values(
        ["forecast_group", "Year", "Month"]
    ).reset_index(drop=True)

    forecast_group_long_daily = (
        forecast_group_data.groupby(
            ["Year", "Month", "Month_Name", "Day", "forecast_group"]
        )
        .agg({"Volume": "sum"})
        .reset_index()
    )
    forecast_group_long_daily = forecast_group_long_daily.sort_values(
        ["forecast_group", "Year", "Month", "Day"]
    ).reset_index(drop=True)

    return (
        forecast_group_display_pivot,
        forecast_group_volume_split,
        forecast_group_long_original,
        forecast_group_long_monthly,
        forecast_group_long_daily,
    )


# ---------------------------------------------------------------------------
# MAP ORIGINAL VOLUME TO PHASE 2 FORECAST
# ---------------------------------------------------------------------------

def map_original_volume_to_phase2_forecast(forecast_df: pd.DataFrame, volume_long_df: pd.DataFrame) -> pd.DataFrame:
    forecast_df = forecast_df.copy()
    volume_long_df = volume_long_df.copy()

    volume_long_df["year"] = volume_long_df["year"].astype(int)
    forecast_df["Year"] = forecast_df["Year"].astype(int)

    volume_long_df["month"] = volume_long_df["month"].str.strip()
    forecast_df["Month"] = forecast_df["Month"].str.strip()

    group_keys = ["year", "month"]
    if "category" in forecast_df.columns and "category" in volume_long_df.columns:
        group_keys.append("category")

    volume_agg_long = volume_long_df.groupby(group_keys, as_index=False)["volume"].sum()

    left_on = ["Year", "Month"]
    right_on = ["year", "month"]
    if "category" in forecast_df.columns and "category" in volume_long_df.columns:
        left_on.append("category")
        right_on.append("category")

    merge_volume_df = volume_agg_long[right_on + ["volume"]]

    merged_df = pd.merge(
        forecast_df,
        merge_volume_df,
        left_on=left_on,
        right_on=right_on,
        how="left",
    )

    for col in right_on:
        if col not in forecast_df.columns:
            merged_df.drop(columns=col, inplace=True)

    return merged_df


# ---------------------------------------------------------------------------
# CLEAN + CONVERT HELPERS (M / K / %)
# ---------------------------------------------------------------------------

def clean_and_convert_millions(col: pd.Series) -> pd.Series:
    col_str = col.astype(str).str.replace(",", "", regex=False).str.strip().str.upper()

    def convert_val(val: str):
        if pd.isna(val):
            return None
        try:
            if val.endswith("M"):
                return float(val[:-1]) * 1_000_000
            if val.endswith("K"):
                return float(val[:-1]) * 1_000
            return float(val)
        except Exception:
            return None

    return col_str.apply(convert_val)


def clean_and_convert_thousands(col: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(col.astype(str).str.replace(",", "", regex=False), errors="coerce")
        * 1_000
    )


def clean_and_convert_percentage(col: pd.Series) -> pd.Series:
    return pd.to_numeric(
        col.astype(str).str.replace("%", "", regex=False), errors="coerce"
    ) / 100


# ---------------------------------------------------------------------------
# FORMAT FORECAST PIVOT (NO DATE) + IQ UNPIVOT + STYLING
# ---------------------------------------------------------------------------

def format_forecast_pivot_no_date(
    df: pd.DataFrame,
    model_col: str = "Model",
    month_year_col: str = "Month_Year",
    forecast_col: str = "Forecast",
    font_color: str = "darkblue",
    font_size: str = "16px",
) -> pd.DataFrame:
    wide_df = (
        df.pivot(index=model_col, columns=month_year_col, values=forecast_col)
        .reset_index()
    )

    month_cols = [c for c in wide_df.columns if c != model_col]
    date_cols = pd.to_datetime(month_cols, format="%b-%y", errors="coerce").dropna()
    sorted_month_cols = [d.strftime("%b-%y") for d in sorted(date_cols)]
    sorted_columns = [model_col] + sorted_month_cols
    wide_df = wide_df.reindex(columns=sorted_columns)

    for col in sorted_month_cols:
        wide_df[col] = pd.to_numeric(wide_df[col], errors="coerce")
        wide_df[col] = wide_df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
    return wide_df


def unpivot_iq_summary(iq_df: pd.DataFrame) -> pd.DataFrame:
    month_cols = [c for c in iq_df.columns if c != "Year"]
    for col in month_cols:
        iq_df[col] = iq_df[col].astype(str)

    iq_long = iq_df.melt(
        id_vars="Year", value_vars=month_cols, var_name="Month", value_name="IQ_value"
    )

    iq_long["IQ_value"] = iq_long["IQ_value"].replace(["", " "], pd.NA)
    iq_long["IQ_value"] = (
        iq_long["IQ_value"].str.replace("M", "", regex=False).str.replace(",", "", regex=False)
    )
    iq_long["IQ_value"] = pd.to_numeric(iq_long["IQ_value"], errors="coerce")
    return iq_long


def fmt_percent1(x: Any) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return ""
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return ""


def fmt_millions_1(x: Any) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return ""
        return f"{float(x):.1f}M"
    except Exception:
        return ""


def apply_font_style1(df: pd.DataFrame, color: str = "darkblue", size: int = 30) -> pd.DataFrame:
    # In Dash, styling is handled in components; return df unchanged.
    return df.copy()


# ---------------------------------------------------------------------------
# IMPUTATION / SEASONALITY HELPERS
# ---------------------------------------------------------------------------

def impute_blanks_with_directional_average(
    df: pd.DataFrame, month_columns: Sequence[str], skip_first_row: bool = True
) -> pd.DataFrame:
    df_copy = df.copy()
    n_rows = len(df_copy)

    for col_idx, curr_month in enumerate(month_columns):
        for i in range(n_rows):
            val = df_copy.iloc[i][curr_month]
            if skip_first_row and i == 0:
                continue

            if pd.isna(val) or val == 0:
                if i == n_rows - 1:
                    prev_row = i - 1
                    prev2_row = i - 2
                    ref_month_idx = col_idx - 1
                    ref_month = month_columns[ref_month_idx] if ref_month_idx >= 0 else None

                    prev_row_curr_month = (
                        df_copy.iloc[prev_row][curr_month] if prev_row >= 0 else None
                    )
                    prev2_row_curr_month = (
                        df_copy.iloc[prev2_row][curr_month] if prev2_row >= 0 else None
                    )
                    prev_row_prev_month = (
                        df_copy.iloc[prev_row][ref_month] if prev_row >= 0 and ref_month else None
                    )
                    curr_row_prev_month = (
                        df_copy.iloc[i][ref_month] if ref_month else None
                    )

                    imput_val = np.nan
                    if prev_row >= 0 and prev2_row >= 0:
                        if pd.isna(prev2_row_curr_month) or prev2_row_curr_month == 0:
                            if prev_row_prev_month and not pd.isna(prev_row_prev_month) and prev_row_prev_month != 0:
                                ratio2 = prev_row_curr_month / prev_row_prev_month
                                if ratio2 < 0.8 or ratio2 > 1.2:
                                    previous_3_month_indices = [
                                        col_idx - j for j in range(1, 4) if col_idx - j >= 0
                                    ]
                                    prev_3_month_values = [
                                        df_copy.iloc[i][month_columns[idx]]
                                        for idx in previous_3_month_indices
                                        if pd.notna(df_copy.iloc[i][month_columns[idx]])
                                        and df_copy.iloc[i][month_columns[idx]] != 0
                                    ]
                                    imput_val = np.mean(prev_3_month_values) if prev_3_month_values else np.nan
                                else:
                                    imput_val = (
                                        (prev_row_curr_month / prev_row_prev_month) * curr_row_prev_month
                                        if not pd.isna(curr_row_prev_month) and curr_row_prev_month != 0
                                        else np.nan
                                    )
                            else:
                                imput_val = np.nan
                        else:
                            test1 = prev_row_curr_month / prev2_row_curr_month
                            if test1 < 0.8 or test1 > 1.2:
                                if prev_row_prev_month and not pd.isna(prev_row_prev_month) and prev_row_prev_month != 0:
                                    test2 = prev_row_curr_month / prev_row_prev_month
                                    if test2 < 0.8 or test2 > 1.2:
                                        values_for_avg = [
                                            df_copy.iloc[j][curr_month]
                                            for j in [i - 1, i - 2, i - 3]
                                            if j >= 0
                                            and pd.notna(df_copy.iloc[j][curr_month])
                                            and df_copy.iloc[j][curr_month] != 0
                                        ]
                                        imput_val = np.mean(values_for_avg) if values_for_avg else np.nan
                                    else:
                                        imput_val = (
                                            (prev_row_curr_month / prev_row_prev_month) * curr_row_prev_month
                                            if not pd.isna(curr_row_prev_month) and curr_row_prev_month != 0
                                            else np.nan
                                        )
                                else:
                                    imput_val = np.nan
                            else:
                                avg_num = (
                                    np.mean([prev_row_curr_month, prev_row_prev_month])
                                    if not pd.isna(prev_row_prev_month) and prev_row_prev_month != 0
                                    else prev_row_curr_month
                                )
                                avg_den = (
                                    np.mean([prev_row_prev_month, prev2_row_curr_month])
                                    if not pd.isna(prev2_row_curr_month) and prev2_row_curr_month != 0
                                    else prev2_row_curr_month
                                )
                                if avg_den != 0 and not pd.isna(avg_den) and curr_row_prev_month != 0:
                                    imput_val = (avg_num / avg_den) * curr_row_prev_month
                                else:
                                    imput_val = np.nan

                    df_copy.iat[i, df_copy.columns.get_loc(curr_month)] = imput_val
                else:
                    avg_below = (
                        df_copy.iloc[i + 1 :][curr_month].replace(0, np.nan).mean(skipna=True)
                    )
                    df_copy.iat[i, df_copy.columns.get_loc(curr_month)] = avg_below

    return df_copy


def calculate_seasonality(avg_daily_df: pd.DataFrame) -> pd.DataFrame:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    df = avg_daily_df.copy()
    df.set_index("Year", inplace=True)

    if "Avg" in df.columns:
        df = df.drop(columns=["Avg"], inplace=False)

    df = impute_blanks_with_directional_average(df.reset_index(), months)
    df.set_index("Year", inplace=True)

    yearly_avg = df.iloc[1:].mean(axis=1, skipna=True)

    ratio_df = pd.DataFrame(index=df.index, columns=months, dtype=float)
    ratio_df.iloc[1:] = df.iloc[1:].div(yearly_avg, axis=0)

    first_row = df.iloc[0]
    non_blank_cols = [c for c in months if pd.notna(first_row[c]) and first_row[c] != 0]
    avg_non_blank = first_row[non_blank_cols].mean() if len(non_blank_cols) > 0 else np.nan
    first_row_ratio = first_row.copy()
    for col in months:
        val = first_row[col]
        if pd.notna(val) and val != 0:
            first_row_ratio[col] = val / avg_non_blank
        else:
            first_row_ratio[col] = np.nan
    ratio_df.iloc[0] = first_row_ratio

    ratio_df["Avg"] = ratio_df.mean(axis=1)

    if len(ratio_df) >= 2:
        last_row_index = len(ratio_df) - 1
        prev_row_index = last_row_index - 1
        avg_current = ratio_df.iloc[last_row_index]["Avg"]
        avg_prev = ratio_df.iloc[prev_row_index]["Avg"]

        if avg_prev != 0 and (avg_current / avg_prev < 0.8 or avg_current / avg_prev > 1.2):
            for col in months:
                ratio_df.iat[prev_row_index, ratio_df.columns.get_loc(col)] = ratio_df.iat[
                    last_row_index, ratio_df.columns.get_loc(col)
                ]
            ratio_df.iat[prev_row_index, ratio_df.columns.get_loc("Avg")] = ratio_df.iat[
                last_row_index, ratio_df.columns.get_loc("Avg")
            ]

    ratio_df.reset_index(inplace=True)
    return ratio_df


def impute_first_row_with_seasonality(
    seasonality_df: pd.DataFrame, month_columns: Sequence[str]
) -> pd.DataFrame:
    df_copy = seasonality_df.copy()
    n_rows = len(df_copy)

    for i in range(n_rows - 1, 0, -1):
        current_row = i
        prev_row = i - 1
        for col in month_columns:
            prev_val = df_copy.iloc[prev_row][col]

            if pd.isna(prev_val) or prev_val == 0:
                avg_below = (
                    df_copy.iloc[prev_row + 1 :][col].replace(0, np.nan).mean(skipna=True)
                )
                if pd.notna(avg_below):
                    df_copy.iat[prev_row, df_copy.columns.get_loc(col)] = avg_below
            else:
                current_val = df_copy.iloc[current_row][col]
                ratio = current_val / prev_val if prev_val != 0 else np.nan
                if pd.notna(ratio) and (ratio < 0.8 or ratio > 1.2):
                    df_copy.iat[prev_row, df_copy.columns.get_loc(col)] = current_val

    return df_copy


# ---------------------------------------------------------------------------
# SEASONALITY PLOT + PROCESSING (CONTACT RATIO)
# ---------------------------------------------------------------------------

def plot_contact_ratio_seasonality(
    contact_ratio_df: pd.DataFrame,
    unique_key: str = "default",
    stage_store: MutableMapping[str, Any] | None = None,
) -> tuple[go.Figure, pd.DataFrame, pd.DataFrame]:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    columns_to_use = ["Year"] + [col for col in months if col in contact_ratio_df.columns]

    df = contact_ratio_df[columns_to_use].copy()
    seasonality = calculate_seasonality(df)
    seasonality_imputed = impute_first_row_with_seasonality(seasonality, months)
    ratio_df = seasonality_imputed.copy()

    melted_ratio_df = ratio_df.melt(id_vars="Year", var_name="Month", value_name="Ratio")
    melted_ratio_df["Month"] = pd.Categorical(
        melted_ratio_df["Month"],
        categories=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Avg"],
        ordered=True,
    )
    melted_ratio_df.sort_values(by=["Year", "Month"], inplace=True)

    custom_colors = ["#1f77b4", "orange", "green", "#d62728", "#9467bd", "#8c564b"]
    alignments = ["bottom left", "top right", "top right"]
    offsets = [-10, 0, 10]

    fig = go.Figure()
    for i, year in enumerate(melted_ratio_df["Year"].unique()):
        df_year = melted_ratio_df[melted_ratio_df["Year"] == year]
        color = custom_colors[i % len(custom_colors)]
        offset = offsets[i % len(offsets)]
        alignment = alignments[i % len(alignments)]

        fig.add_trace(
            go.Scatter(
                x=df_year["Month"],
                y=df_year["Ratio"],
                mode="lines+markers+text",
                name=str(year),
                line=dict(color=color, width=3, dash="dash"),
                marker=dict(size=6),
                text=[f"{r:.2f}" for r in df_year["Ratio"]],
                textposition=alignment,
                textfont=dict(color=color, family="Arial Bold"),
            )
        )

    fig.update_layout(
        title="Seasonality Ratio Chart",
        xaxis_title="Month",
        yaxis_title="Contact Ratio / Yearly Average",
        yaxis=dict(range=[0, melted_ratio_df["Ratio"].max() + 0.2]),
        plot_bgcolor="white",
        width=900,
        height=500,
        legend_title="Year",
    )

    lower_cap = 0.80
    upper_cap = 1.15

    capped_seasonality = ratio_df.copy()
    for col in capped_seasonality.columns:
        if col == "Year":
            continue
        capped_seasonality[col] = capped_seasonality[col].clip(lower=lower_cap, upper=upper_cap)

    for col in capped_seasonality.columns:
        if col == "Year":
            continue
        capped_seasonality[col] = pd.to_numeric(capped_seasonality[col], errors="coerce")

    month_cols = [c for c in capped_seasonality.columns if c not in ["Year", "Avg"]]

    def normalize_row(row: pd.Series):
        avg = row[month_cols].mean()
        if avg == 0:
            return row[month_cols]
        factor = 1.0 / avg
        return row[month_cols] * factor

    capped_seasonality[month_cols] = capped_seasonality.apply(normalize_row, axis=1).round(2)
    capped_seasonality["Avg"] = capped_seasonality[month_cols].mean(axis=1).round(2)

    if stage_store is not None:
        stage_store[f"stage_data_{unique_key}"] = {
            "imputed_df": seasonality_imputed,
            "seasonality": seasonality,
            "seasonality_imputed": seasonality_imputed,
        }
        stage_store[f"ratio_df_{unique_key}"] = ratio_df

    return fig, capped_seasonality, ratio_df


# ---------------------------------------------------------------------------
# FORECAST + CONFIG METADATA HELPERS
# ---------------------------------------------------------------------------

def flatten_dict(d: Mapping[str, Any], parent_key: str = "", sep: str = ":") -> Dict[str, Any]:
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def create_download_csv_with_metadata(forecast_df: pd.DataFrame, config: dict) -> str:
    csv_buffer = io.StringIO()
    forecast_df.to_csv(csv_buffer, index=False)
    forecast_csv_str = csv_buffer.getvalue()

    flat_config = flatten_dict(config)
    config_df = pd.DataFrame(flat_config.items(), columns=["Config_Key", "Config_Value"])

    config_buffer = io.StringIO()
    config_df.to_csv(config_buffer, index=False)
    config_csv_str = config_buffer.getvalue()

    combined_csv_str = (
        "### Forecast Results\n" + forecast_csv_str + "\n\n### Forecast Configuration\n" + config_csv_str
    )
    return combined_csv_str


def multiply_capped_with_contact_ratio(
    edited_capped_df: pd.DataFrame, contact_ratio_df: pd.DataFrame, base_volume: float | None = None
) -> pd.DataFrame:
    capped_df = edited_capped_df.set_index("Year").copy()
    ratio = contact_ratio_df.set_index("Year").copy()

    for col in ratio.columns:
        if ratio[col].dtype == "O":
            ratio[col] = ratio[col].astype(str).str.replace("%", "").replace("", "0")
            ratio[col] = pd.to_numeric(ratio[col], errors="coerce").fillna(0)

    exclude_cols = ["Avg", "Yearly_Avg", "Growth_%"]
    months = [col for col in capped_df.columns if col not in exclude_cols and col in ratio.columns]

    product = capped_df[months] * ratio[months]
    if base_volume is not None:
        product = product * base_volume

    product["Year"] = capped_df.index
    return product.reset_index(drop=True)


def clean_contact_ratio_df1(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_keep = ["Year"] + [
        m
        for m in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        if m in df.columns
    ]
    numeric_df = df[cols_to_keep].copy()

    for col in numeric_df.columns:
        if col != "Year":
            numeric_df[col] = (
                numeric_df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

    return numeric_df


# ---------------------------------------------------------------------------
# CONTACT RATIO BASE VOLUME + ENSURE NUMERIC
# ---------------------------------------------------------------------------

def add_editable_base_volume(contact_ratio_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    def clean_contact_ratio_column(col_series: pd.Series) -> pd.Series:
        cleaned = col_series.astype(str).str.extract(r"(\d+\.?\d*)")[0]
        return pd.to_numeric(cleaned, errors="coerce")

    def clean_contact_ratio_df(df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for col in df_copy.columns:
            if col != "Year":
                df_copy[col] = clean_contact_ratio_column(df_copy[col])
        return df_copy

    df_cleaned = clean_contact_ratio_df(contact_ratio_df)
    fallback_base_vol = df_cleaned.iloc[-1].drop(labels=["Year", "Growth_%"], errors="ignore").mean()

    return contact_ratio_df.copy(), float(fallback_base_vol)


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df.copy()
    for col in df_cleaned.columns:
        if col != "Year":
            df_cleaned[col] = df_cleaned[col].astype(str).str.replace("%", "", regex=False).replace("", "0")
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")
    return df_cleaned


# ---------------------------------------------------------------------------
# FORECAST RESULTS COMBINER
# ---------------------------------------------------------------------------

_MODEL_DISPLAY = {
    "prophet": "Prophet",
    "random_forest": "Rf",
    "randomforest": "Rf",
    "rf": "Rf",
    "xgboost": "Xgb",
    "xgb": "Xgb",
    "sarimax": "Sarimax",
    "var": "Var",
}

def process_forecast_results(
    forecast_results: Mapping[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined_forecast_df = pd.DataFrame()
    skip_keys = {"train", "test", "debug", "final_smoothed_values"}

    for model_name, df_fc in (forecast_results or {}).items():
        if model_name in skip_keys:
            continue
        if df_fc is None or getattr(df_fc, "empty", True):
            continue

        df_temp = df_fc.copy()
        if "ds" in df_temp.columns:
            df_temp = df_temp.rename(columns={"ds": "Month"})
        if "yhat" in df_temp.columns:
            df_temp = df_temp.rename(columns={"yhat": "Forecast"})

        if "Month" not in df_temp.columns or "Forecast" not in df_temp.columns:
            continue

        display = _MODEL_DISPLAY.get(str(model_name).strip().lower(), str(model_name).title())
        df_temp["Model"] = display
        df_temp["Month"] = pd.to_datetime(df_temp["Month"], errors="coerce")
        df_temp["Forecast"] = pd.to_numeric(df_temp["Forecast"], errors="coerce")
        df_temp = df_temp.dropna(subset=["Month", "Forecast"])

        combined_forecast_df = pd.concat(
            [combined_forecast_df, df_temp[["Model", "Month", "Forecast"]]],
            ignore_index=True,
        )

    wide_forecast_df = pd.DataFrame()
    pivot_smoothed_df = pd.DataFrame()

    if not combined_forecast_df.empty:
        combined_forecast_df = combined_forecast_df.copy()
        combined_forecast_df["Month"] = (
            pd.to_datetime(combined_forecast_df["Month"], errors="coerce")
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        combined_forecast_df["Month_Year"] = combined_forecast_df["Month"].dt.strftime("%b-%y")
        combined_forecast_df["Model"] = combined_forecast_df["Model"].astype(str).str.strip()
        combined_forecast_df["Month_Year"] = combined_forecast_df["Month_Year"].astype(str).str.strip()
        combined_forecast_df = combined_forecast_df.groupby(
            ["Model", "Month_Year"], as_index=False
        )["Forecast"].mean()

        wide_forecast_df = (
            combined_forecast_df.pivot(index="Model", columns="Month_Year", values="Forecast")
            .reset_index()
        )
        month_cols = [c for c in wide_forecast_df.columns if c not in ("Model", "Avg")]
        if month_cols:
            wide_forecast_df["Avg"] = wide_forecast_df[month_cols].apply(
                lambda r: pd.to_numeric(r, errors="coerce").mean(), axis=1
            )
        cols = list(wide_forecast_df.columns)
        if "Avg" in cols:
            cols = ["Model", "Avg"] + [c for c in cols if c not in ("Model", "Avg")]
            wide_forecast_df = wide_forecast_df[cols]

    if isinstance(forecast_results, Mapping) and "final_smoothed_values" in forecast_results:
        base = forecast_results.get("final_smoothed_values")
        if base is not None and not getattr(base, "empty", True):
            final_smoothed_df = base.copy()
            if "Date" in final_smoothed_df.columns:
                final_smoothed_df["Date"] = pd.to_datetime(
                    final_smoothed_df["Date"], errors="coerce"
                )
            else:
                final_smoothed_df["Date"] = pd.NaT
            if "Final_Smoothed_Value" in final_smoothed_df.columns:
                final_smoothed_df["Final_Smoothed_Value"] = pd.to_numeric(
                    final_smoothed_df["Final_Smoothed_Value"], errors="coerce"
                )
            final_smoothed_df = final_smoothed_df.dropna(subset=["Date", "Final_Smoothed_Value"])
            final_smoothed_df["Year"] = final_smoothed_df["Date"].dt.year.astype(int)
            final_smoothed_df["Month"] = final_smoothed_df["Date"].dt.strftime("%b")
            pivot_smoothed_df = final_smoothed_df.pivot(
                index="Year", columns="Month", values="Final_Smoothed_Value"
            ).reset_index()
            months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            keep = ["Year"] + [m for m in months_order if m in pivot_smoothed_df.columns]
            pivot_smoothed_df = pivot_smoothed_df[keep]

    return combined_forecast_df, wide_forecast_df, pivot_smoothed_df


# ---------------------------------------------------------------------------
# PERCENTAGE CLEAN / FORMAT + SIMPLE STYLE FORECAST
# ---------------------------------------------------------------------------

def clean_percentage_columns1(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    for col in df_copy.columns:
        if col != "Year":
            df_copy[col] = df_copy[col].astype(str).str.replace("%", "", regex=False).replace("", "0")
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
    return df_copy


def format_percentage_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    for col in df_copy.columns:
        if col != "Year":
            df_copy[col] = df_copy[col].astype(float).round(4)
            df_copy[col] = df_copy[col].apply(lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else "")
    return df_copy


def simple_style_month_forecast(df: pd.DataFrame, color: str = "darkblue", size: int = 30) -> pd.DataFrame:
    display_df = df.copy()

    for col in display_df.columns:
        if col in ["Accuracy(+−5%)", "Accuracy(+−7%)", "Accuracy(+−10%)"]:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.1f}%" if pd.notnull(x) and isinstance(x, (int, float)) else str(x) if pd.notnull(x) else ""
            )
        elif col in ["Count_Within(+−5%)", "Total_Months"]:
            display_df[col] = display_df[col].apply(
                lambda x: f"{int(x):d}" if pd.notnull(x) and isinstance(x, (int, float)) else str(x) if pd.notnull(x) else ""
            )
        elif col not in ["Model", "Metric"]:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.2f}%" if pd.notnull(x) and isinstance(x, (int, float)) else str(x) if pd.notnull(x) else ""
            )

    return display_df


# ---------------------------------------------------------------------------
# STYLED FORECAST TABLES
# ---------------------------------------------------------------------------

def format_percentage_series(series: pd.Series, col_name: str) -> pd.Series:
    if col_name == "Year":
        return series
    return series.astype(float).round(4).apply(lambda x: f"{x:.2f}%")


def prepare_styled_forecast(df: pd.DataFrame) -> pd.DataFrame:
    months_order = list(calendar.month_abbr[1:])
    month_cols = [m for m in months_order if m in df.columns]
    df_numeric = df[month_cols].astype(float)
    df = df[["Year"] + month_cols].copy()
    df["Avg"] = df_numeric.mean(axis=1)
    final_cols = ["Year", "Avg"] + month_cols
    df = df[final_cols]
    for col in df.select_dtypes(include="number").columns:
        if col != "Year":
            df[col] = (df[col] * 100).apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
    return df


def prepare_styled_forecast_for_Normalized_Contact_Ratio_2(df: pd.DataFrame) -> pd.DataFrame:
    months_order = list(calendar.month_abbr[1:])
    month_cols = [m for m in months_order if m in df.columns]
    df = df[["Year"] + month_cols].copy()
    df_numeric = df[month_cols].astype(float)
    df["Avg"] = df_numeric.mean(axis=1)
    final_cols = ["Year"] + month_cols + ["Avg"]
    df = df[final_cols]
    for col in df.select_dtypes(include="number").columns:
        if col != "Year":
            df[col] = df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
    return df


def prepare_styled_forecast_for_model(df: pd.DataFrame) -> pd.DataFrame:
    month_abbr = list(calendar.month_abbr[1:])
    month_cols = [col for col in df.columns if any(col.startswith(m) for m in month_abbr)]

    def parse_month_year(col: str):
        m, y = col.split("-")
        return pd.Timestamp(f"1 {m} 20{y}")

    month_cols_sorted = sorted(month_cols, key=parse_month_year)
    df_numeric = df[month_cols_sorted].astype(float)
    df = df[["Model"] + month_cols_sorted].copy()
    df["Avg"] = df_numeric.mean(axis=1)
    final_cols = ["Model", "Avg"] + month_cols_sorted
    df = df[final_cols]

    numeric_cols = [col for col in df.select_dtypes(include="number").columns if col != "Model"]
    df.loc[:, numeric_cols] = df[numeric_cols] * 100
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
    return df


# ---------------------------------------------------------------------------
# ACCURACY PHASE 1 (MONTH-LEVEL) + FILL FINAL SMOOTHED ROW
# ---------------------------------------------------------------------------

def accuracy_phase1(
    forecast_df: pd.DataFrame, baseline_df: pd.DataFrame
) -> pd.DataFrame:
    try:
        forecast_df = forecast_df.copy()
        baseline_df = baseline_df.copy()

        for col in forecast_df.columns:
            if col != "Model" and forecast_df[col].dtype == "object":
                try:
                    forecast_df[col] = (
                        forecast_df[col].astype(str).str.replace("%", "", regex=False).astype(float) / 100
                    )
                except Exception:
                    pass

        month_cols = [
            col
            for col in forecast_df.columns
            if col
            not in [
                "Model",
                "Avg",
                "Accuracy",
                "Count_Within(+−5%)",
                "Total_Months",
            ]
            and isinstance(col, str)
            and col.strip() != ""
        ]

        month_cols = sorted(
            [col for col in month_cols if isinstance(col, str) and pd.notnull(col) and col.strip() != ""],
            key=lambda x: pd.to_datetime(x, format="%b-%y", errors="coerce"),
        )

        baseline_melted = baseline_df.copy()
        baseline_melted = baseline_melted.melt(id_vars="Year", var_name="Month", value_name="Baseline")
        baseline_melted["Month_Year"] = baseline_melted.apply(
            lambda row: f"{row['Month']}-{str(row['Year'])[-2:]}"
            if pd.notnull(row["Month"]) and pd.notnull(row["Year"])
            else None,
            axis=1,
        )
        baseline_lookup = (
            baseline_melted.dropna(subset=["Month_Year"]).set_index("Month_Year")["Baseline"].to_dict()
        )

        ratio_df = forecast_df.copy()
        ratio_df = ratio_df.drop(columns=["Avg", "Count_Within(+−5%)"], errors="ignore")

        for col in month_cols:
            baseline_val = baseline_lookup.get(col)
            if baseline_val is not None and baseline_val != 0:
                try:
                    ratio_df[col] = (baseline_val) / ratio_df[col].astype(float)
                except Exception:
                    ratio_df[col] = None
            else:
                ratio_df[col] = None

        ratio_df = ratio_df[ratio_df["Model"] != "Final_smoothed_values"]

        def filter_and_summarize(row: pd.Series, lower: float, upper: float):
            filtered = [val for val in row[month_cols] if pd.notnull(val)]
            count = len([val for val in filtered if lower <= val <= upper])
            total = len(filtered)
            accuracy = count / total if total > 0 else None
            return accuracy, count, total

        ranges = {
            "Accuracy(+−5%)": (94.5, 105.5),
            "Accuracy(+−7%)": (92.5, 107.5),
            "Accuracy(+−10%)": (89.5, 110.5),
        }
        results = [
            ratio_df.apply(lambda row, r=r: filter_and_summarize(row, r[0], r[1]), axis=1)
            for r in ranges.values()
        ]

        ratio_df["Accuracy(+−5%)"] = [r[0] for r in results[0]]
        ratio_df["Count_Within(+−5%)"] = [r[1] for r in results[0]]
        ratio_df["Accuracy(+−7%)"] = [r[0] for r in results[1]]
        ratio_df["Accuracy(+−10%)"] = [r[0] for r in results[2]]
        ratio_df["Total_Months"] = [r[2] for r in results[0]]

        ratio_df["Count_Within(+−5%)"] = ratio_df["Count_Within(+−5%)"].astype(int)
        ratio_df["Total_Months"] = ratio_df["Total_Months"].astype(int)

        ratio_df.loc[:, "Accuracy(+−5%)"] = ratio_df["Accuracy(+−5%)"] * 100
        ratio_df.loc[:, "Accuracy(+−7%)"] = ratio_df["Accuracy(+−7%)"] * 100
        ratio_df.loc[:, "Accuracy(+−10%)"] = ratio_df["Accuracy(+−10%)"] * 100

        final_cols = [
            "Model",
            "Total_Months",
            "Count_Within(+−5%)",
            "Accuracy(+−5%)",
            "Accuracy(+−7%)",
            "Accuracy(+−10%)",
        ] + month_cols
        ratio_df = ratio_df[[col for col in final_cols if col in ratio_df.columns]]

        columns_to_format = ["Accuracy(+−5%)", "Accuracy(+−7%)", "Accuracy(+−10%)"] + month_cols
        for col in columns_to_format:
            if col in ratio_df.columns:
                ratio_df[col] = ratio_df[col].round(1)

        ratio_df = ratio_df.loc[:, ratio_df.columns.notna()]
        ratio_df.columns = ratio_df.columns.map(str)
        ratio_df = ratio_df.loc[:, ratio_df.columns != "nan"]

        return ratio_df

    except Exception:
        traceback.print_exc()
        raise


def fill_final_smoothed_row(wide_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    if wide_df is None or wide_df.empty or "Model" not in wide_df.columns:
        return wide_df
    if baseline_df is None or baseline_df.empty:
        return wide_df

    wide_df = wide_df.copy()
    base = baseline_df.copy()

    model_series = wide_df["Model"].astype(str).str.strip().str.lower()
    mask = model_series.str.startswith("final_smoot")
    if not mask.any():
        mask = model_series.eq("final_smoothed_values")
        if not mask.any():
            return wide_df

    if "Year" not in base.columns:
        if base.index.name and str(base.index.name).lower() == "year":
            base = base.reset_index()
        else:
            return wide_df

    full_to_abbrev = {
        "january": "Jan",
        "february": "Feb",
        "march": "Mar",
        "april": "Apr",
        "may": "May",
        "june": "Jun",
        "july": "Jul",
        "august": "Aug",
        "september": "Sep",
        "october": "Oct",
        "november": "Nov",
        "december": "Dec",
    }

    def is_month_col(col: Any) -> bool:
        s = str(col).strip()
        low = s.lower()
        if low in {"year", "model", "avg", "average", "total"}:
            return False
        if re.fullmatch(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", low):
            return True
        if low in full_to_abbrev:
            return True
        return False

    month_cols = [c for c in base.columns if is_month_col(c)]
    if not month_cols:
        return wide_df

    rename = {}
    for col in month_cols:
        low = str(col).strip().lower()
        if low in full_to_abbrev:
            rename[col] = full_to_abbrev[low]
    if rename:
        base = base.rename(columns=rename)
        month_cols = [rename.get(c, c) for c in month_cols]

    baseline_melted = base.melt(
        id_vars="Year",
        value_vars=month_cols,
        var_name="Month",
        value_name="Smoothed",
    )
    baseline_melted["Month_Year"] = baseline_melted.apply(
        lambda row: f"{row['Month']}-{str(row['Year'])[-2:]}",
        axis=1,
    )
    baseline_lookup = baseline_melted.groupby("Month_Year")["Smoothed"].mean().to_dict()

    for col in wide_df.columns:
        if col == "Model":
            continue
        if col in baseline_lookup:
            wide_df.loc[mask, col] = baseline_lookup[col] / 100.0

    if "Avg" in wide_df.columns:
        month_cols_in_wide = [c for c in wide_df.columns if c not in ("Model", "Avg")]
        if month_cols_in_wide:
            wide_df.loc[mask, "Avg"] = wide_df.loc[mask, month_cols_in_wide].apply(
                lambda r: pd.to_numeric(r, errors="coerce").mean(), axis=1
            )

    return wide_df


def map_normalized_volume_to_forecast(
    forecast_df: pd.DataFrame, smoothing_phase2_df: pd.DataFrame
) -> pd.DataFrame:
    forecast_df = forecast_df.copy()
    smoothing_phase2_df = smoothing_phase2_df.copy()

    forecast_df["Year"] = forecast_df["Year"].astype(int)
    smoothing_phase2_df["Year"] = smoothing_phase2_df["Year"].astype(int)
    forecast_df["Month"] = forecast_df["Month"].str.strip()
    smoothing_phase2_df["Month"] = smoothing_phase2_df["Month"].str.strip()

    merged_df = pd.merge(
        forecast_df,
        smoothing_phase2_df[["Year", "Month", "Normalized_Volume"]],
        how="left",
        on=["Year", "Month"],
    )

    return merged_df


# ---------------------------------------------------------------------------
# ACCURACY PHASE 2 (LONG FORMAT)
# ---------------------------------------------------------------------------

def accuracy_phase2_long(
    df: pd.DataFrame, group_col: str = "Model"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    def summarize_accuracy(sub_df: pd.DataFrame, volume_col: str) -> pd.Series:
        filtered = sub_df[
            sub_df[volume_col].notna()
            & (sub_df[volume_col] != 0)
            & sub_df["Base_Forecast"].notna()
            & (sub_df["Base_Forecast"] != 0)
        ].copy()

        filtered["ratio"] = (filtered[volume_col] / filtered["Base_Forecast"]) * 100

        results: Dict[str, Any] = {}
        total = len(filtered)
        results["Total_Months"] = total

        ranges = {
            "Met_Count(+−5%)": (94.5, 105.5),
            "Accuracy(+−5%)": (94.5, 105.5),
            "Accuracy(+−7%)": (92.5, 107.5),
            "Accuracy(+−10%)": (90.5, 110.5),
        }
        for key, (lower, upper) in ranges.items():
            met = filtered[(filtered["ratio"] >= lower) & (filtered["ratio"] <= upper)]
            count = len(met)
            results[key] = count if "Count" in key else (count / total * 100 if total > 0 else None)

        return pd.Series(results)

    orig_summary = (
        df.groupby(group_col)
        .apply(lambda g: summarize_accuracy(g, "Original_Volume"))
        .reset_index()
    )
    norm_summary = (
        df.groupby(group_col)
        .apply(lambda g: summarize_accuracy(g, "Normalized_Volume"))
        .reset_index()
    )

    for df_summary in (orig_summary, norm_summary):
        df_summary.dropna(subset=["Total_Months"], inplace=True)
        df_summary["Total_Months"] = df_summary["Total_Months"].astype(int)
        df_summary["Met_Count(+−5%)"] = df_summary["Met_Count(+−5%)"].astype(int)

        percent_cols = ["Accuracy(+−5%)", "Accuracy(+−7%)", "Accuracy(+−10%)"]
        for col in percent_cols:
            df_summary[col] = df_summary[col].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")

    return orig_summary, norm_summary


# ---------------------------------------------------------------------------
# CONFIG SNAPSHOTS + EXTRACTION
# ---------------------------------------------------------------------------

def config_to_dataframe(config_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame(config_dict)
    df = df.T
    df = df.reset_index()
    df = df.rename(columns={"index": "Model"})
    return df


def transform_model_snap_to_year_snap(df: pd.DataFrame) -> pd.DataFrame:
    month_year_cols = [c for c in df.columns if "-" in c]

    long_df = df.melt(
        id_vars=["Model", "Avg"],
        value_vars=month_year_cols,
        var_name="Month-Year",
        value_name="Value",
    )

    long_df["Month"], long_df["Year_suffix"] = long_df["Month-Year"].str.split("-", expand=True)
    long_df["Year"] = 2000 + long_df["Year_suffix"].astype(int)

    pivot_df = long_df.pivot_table(index="Year", columns="Month", values="Value", aggfunc="mean")

    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot_df = pivot_df.reindex(columns=month_order)

    pivot_df["Avg"] = pivot_df.mean(axis=1)

    pivot_df = pivot_df.reset_index()
    cols = ["Year", "Avg"] + month_order
    pivot_df = pivot_df[cols]

    return pivot_df


def add_config_summary(
    comparison_df: pd.DataFrame,
    phase1_run_results: Mapping[str, Any],
    phase2_run_results: Mapping[str, Any],
) -> pd.DataFrame:
    enhanced_df = comparison_df.copy()
    enhanced_df["Config_Summary"] = ""

    for idx, row in enhanced_df.iterrows():
        model_name = row["Model"].lower()
        source = row["Source"]

        if model_name == "rf":
            model_name = "random_forest"

        if source == "before":
            run_results = phase1_run_results
        elif source == "after":
            run_results = phase2_run_results
        else:
            continue

        config_summary = "Config not found"

        for key, model_result in run_results.items():
            if key in ["train", "test", "debug", "final_smoothed_values"]:
                continue
            if model_name in key.lower() or key.lower() in model_name:
                config_parts = []
                if hasattr(model_result, "items"):
                    for param_key, param_value in model_result.items():
                        config_parts.append(f"{param_key}:{param_value}")
                config_summary = " | ".join(config_parts) if config_parts else str(model_result)
                break

        enhanced_df.loc[idx, "Config_Summary"] = config_summary

    return enhanced_df


def extract_configs_from_summary(enhanced_comparison_df: pd.DataFrame) -> Dict[str, dict]:
    configs: Dict[str, dict] = {}

    for _, row in enhanced_comparison_df.iterrows():
        model_name = row["Model"].lower()
        config_string = row["Config_Summary"]

        model_mapping = {
            "prophet": "prophet",
            "rf": "random_forest",
            "xgb": "xgboost",
            "var": "var",
            "sarimax": "sarimax",
        }

        model_name = model_mapping.get(model_name, model_name)

        config_dict: Dict[str, Any] = {}
        if config_string and config_string != "Config not found":
            params = config_string.split(" | ")
            for param in params:
                if ":" in param:
                    key, value = param.split(":", 1)
                    value = value.strip()

                    if value.lower() == "true":
                        config_dict[key] = True
                    elif value.lower() == "false":
                        config_dict[key] = False
                    elif value.replace(".", "").replace("-", "").isdigit():
                        config_dict[key] = float(value) if "." in value else int(value)
                    else:
                        config_dict[key] = value

        configs[model_name] = config_dict

    return configs
