#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
META_DIR = ROOT / "data" / "metadata"

MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
TARGET_GEOS = {
    "AT",
    "BE",
    "BG",
    "CY",
    "CZ",
    "DE",
    "DK",
    "EE",
    "ES",
    "FI",
    "FR",
    "HR",
    "HU",
    "IE",
    "IS",
    "IT",
    "LT",
    "LU",
    "LV",
    "MT",
    "NL",
    "NO",
    "PL",
    "PT",
    "RO",
    "SE",
    "SI",
    "SK",
    "UK",
}


def eurostat_monthly_wide_to_long(df: pd.DataFrame, min_month: str = "2004-01") -> pd.DataFrame:
    geo_cols = [c for c in df.columns if "\\TIME_PERIOD" in str(c)]
    if len(geo_cols) != 1:
        raise ValueError(f"Expected one geo/time column, got {geo_cols}")
    geo_col = geo_cols[0]

    month_cols = [c for c in df.columns if MONTH_RE.match(str(c)) and str(c) >= min_month]
    id_cols = [c for c in df.columns if c not in month_cols]
    long_df = df.melt(id_vars=id_cols, value_vars=month_cols, var_name="month_str", value_name="passengers")
    key_name = str(geo_col).split("\\")[0]
    long_df = long_df.rename(columns={geo_col: key_name})
    long_df["passengers"] = pd.to_numeric(long_df["passengers"], errors="coerce")
    long_df["month"] = pd.PeriodIndex(long_df["month_str"], freq="M")
    return long_df


def build_route_shocks(route_df: pd.DataFrame, threshold: float = 100.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = route_df.copy()
    df["country"] = df["rep_airp"].astype(str).str[:2]
    df = df[df["country"].isin(TARGET_GEOS)].copy()
    df = df[df["partner"].isin(TARGET_GEOS)].copy()
    df = df[df["country"] != df["partner"]].copy()
    df = df[["country", "rep_airp", "partner", "month", "passengers"]].copy()
    df["passengers"] = pd.to_numeric(df["passengers"], errors="coerce").fillna(0.0)
    df["active"] = (df["passengers"] >= threshold).astype(int)
    df = df.sort_values(["rep_airp", "partner", "month"]).reset_index(drop=True)

    g = df.groupby(["rep_airp", "partner"], sort=False)
    df["active_l1"] = g["active"].shift(1).fillna(0).astype(int)
    prev3 = g["active"].transform(lambda s: s.shift(1).rolling(3, min_periods=3).sum())
    df["inactive_prev3"] = prev3.fillna(0).eq(0).astype(int)
    df["route_open_m"] = ((df["active"] == 1) & (df["active_l1"] == 0) & (df["inactive_prev3"] == 1)).astype(int)
    df["route_close_m_candidate"] = ((df["active"] == 0) & (df["active_l1"] == 1)).astype(int)
    df["active_f1"] = g["active"].shift(-1).fillna(0).astype(int)
    df["active_f2"] = g["active"].shift(-2).fillna(0).astype(int)
    df["active_f3"] = g["active"].shift(-3).fillna(0).astype(int)
    df["future_active_3"] = df["active_f1"] + df["active_f2"] + df["active_f3"]
    # Persistent openings survive into future months; persistent closures do not quickly reactivate.
    df["route_open_persist_m"] = ((df["route_open_m"] == 1) & (df["future_active_3"] >= 2)).astype(int)
    df["route_close_persist_m"] = ((df["route_close_m_candidate"] == 1) & (df["future_active_3"] == 0)).astype(int)
    df["active_route_passengers"] = np.where(df["active"] == 1, df["passengers"], 0.0)

    airport_month = (
        df.groupby(["country", "rep_airp", "month"], as_index=False)
        .agg(
            active_routes=("active", "sum"),
            route_open_m=("route_open_m", "sum"),
            route_close_m_candidate=("route_close_m_candidate", "sum"),
            route_open_persist_m=("route_open_persist_m", "sum"),
            route_close_persist_m=("route_close_persist_m", "sum"),
            passengers_total=("passengers", "sum"),
            active_route_passengers=("active_route_passengers", "sum"),
        )
        .sort_values(["rep_airp", "month"])
        .reset_index(drop=True)
    )
    airport_month["lag_active_routes"] = airport_month.groupby("rep_airp")["active_routes"].shift(1)
    airport_month["open_rate_norm"] = airport_month["route_open_m"] / airport_month["lag_active_routes"].replace(0, np.nan)
    airport_month["close_rate_norm"] = (
        airport_month["route_close_m_candidate"] / airport_month["lag_active_routes"].replace(0, np.nan)
    )
    airport_month["open_persist_rate_norm"] = (
        airport_month["route_open_persist_m"] / airport_month["lag_active_routes"].replace(0, np.nan)
    )
    airport_month["close_persist_rate_norm"] = (
        airport_month["route_close_persist_m"] / airport_month["lag_active_routes"].replace(0, np.nan)
    )
    airport_month["net_openings"] = airport_month["route_open_m"] - airport_month["route_close_m_candidate"]
    airport_month["net_openings_persist"] = airport_month["route_open_persist_m"] - airport_month["route_close_persist_m"]

    country_month = (
        df.groupby(["country", "month"], as_index=False)
        .agg(
            active_routes=("active", "sum"),
            route_open_m=("route_open_m", "sum"),
            route_close_m_candidate=("route_close_m_candidate", "sum"),
            route_open_persist_m=("route_open_persist_m", "sum"),
            route_close_persist_m=("route_close_persist_m", "sum"),
            passengers_total=("passengers", "sum"),
        )
        .sort_values(["country", "month"])
        .reset_index(drop=True)
    )
    country_month["lag_active_routes"] = country_month.groupby("country")["active_routes"].shift(1)
    country_month["lag_passengers_total"] = country_month.groupby("country")["passengers_total"].shift(1)
    country_month["open_rate_norm"] = country_month["route_open_m"] / country_month["lag_active_routes"].replace(0, np.nan)
    country_month["close_rate_norm"] = (
        country_month["route_close_m_candidate"] / country_month["lag_active_routes"].replace(0, np.nan)
    )
    country_month["open_persist_rate_norm"] = (
        country_month["route_open_persist_m"] / country_month["lag_active_routes"].replace(0, np.nan)
    )
    country_month["close_persist_rate_norm"] = (
        country_month["route_close_persist_m"] / country_month["lag_active_routes"].replace(0, np.nan)
    )
    country_month["net_openings"] = country_month["route_open_m"] - country_month["route_close_m_candidate"]
    country_month["net_openings_persist"] = country_month["route_open_persist_m"] - country_month["route_close_persist_m"]
    country_month["opening_passenger_intensity"] = (
        country_month["route_open_m"] / country_month["lag_passengers_total"].replace(0, np.nan) * 1_000_000
    )
    country_month["opening_persist_passenger_intensity"] = (
        country_month["route_open_persist_m"] / country_month["lag_passengers_total"].replace(0, np.nan) * 1_000_000
    )
    country_month["month_str"] = country_month["month"].astype(str)
    country_month["quarter"] = country_month["month"].dt.asfreq("Q")

    country_quarter = (
        country_month.groupby(["country", "quarter"], as_index=False)
        .agg(
            route_open_q=("route_open_m", "sum"),
            route_close_q=("route_close_m_candidate", "sum"),
            net_openings_q=("net_openings", "sum"),
            route_open_persist_q=("route_open_persist_m", "sum"),
            route_close_persist_q=("route_close_persist_m", "sum"),
            net_openings_persist_q=("net_openings_persist", "sum"),
            open_rate_norm_q=("open_rate_norm", "mean"),
            close_rate_norm_q=("close_rate_norm", "mean"),
            open_persist_rate_norm_q=("open_persist_rate_norm", "mean"),
            close_persist_rate_norm_q=("close_persist_rate_norm", "mean"),
            opening_passenger_intensity_q=("opening_passenger_intensity", "mean"),
            opening_persist_passenger_intensity_q=("opening_persist_passenger_intensity", "mean"),
            active_routes_q_mean=("active_routes", "mean"),
            passengers_total_q_sum=("passengers_total", "sum"),
        )
        .sort_values(["country", "quarter"])
        .reset_index(drop=True)
    )
    country_quarter["period_str"] = country_quarter["quarter"].astype(str)
    country_quarter["geo"] = country_quarter["country"]
    return df, airport_month, country_quarter


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DIR / "avia_paoac_m_passengers.csv"
    raw = pd.read_csv(raw_path)
    partner_col = next((c for c in raw.columns if str(c).startswith("partner")), None)
    if partner_col is None:
        raise KeyError("Could not locate partner column in AVIA_PAOAC monthly raw extract")
    # Row pre-filter before melt for memory.
    raw["rep_airp_country"] = raw["rep_airp"].astype(str).str[:2]
    raw = raw[
        raw["rep_airp_country"].isin(TARGET_GEOS)
        & raw[partner_col].isin(TARGET_GEOS)
        & (raw["unit"] == "PAS")
        & (raw["tra_meas"] == "PAS_CRD")
    ].copy()
    raw = raw.drop(columns=["rep_airp_country"])

    long_df = eurostat_monthly_wide_to_long(raw, min_month="2004-01")
    long_df = long_df.dropna(subset=["passengers"]).copy()
    route_month, airport_month, country_quarter = build_route_shocks(long_df, threshold=100.0)

    route_month_out = PROC_DIR / "airport_partner_route_monthly.parquet"
    airport_month_out = PROC_DIR / "airport_monthly_route_shocks.parquet"
    country_quarter_out = PROC_DIR / "country_quarterly_airport_route_shocks.parquet"
    route_month.to_parquet(route_month_out, index=False)
    airport_month.to_parquet(airport_month_out, index=False)
    country_quarter.to_parquet(country_quarter_out, index=False)
    country_quarter.to_csv(country_quarter_out.with_suffix(".csv"), index=False)

    # Merge onto the harmonized national quarterly panel for direct use.
    panel_q_path = PROC_DIR / "panel_quarterly_harmonized.parquet"
    if not panel_q_path.exists():
        raise FileNotFoundError(
            "Missing `panel_quarterly_harmonized.parquet`. Run scripts/harmonize_cross_frequency.py first."
        )
    panel_q = pd.read_parquet(panel_q_path)
    panel_q = panel_q.copy()
    panel_q["period_str"] = panel_q["period"].astype(str) if "period" in panel_q.columns else panel_q["period_str"]
    merged_q = panel_q.merge(
        country_quarter.drop(columns=["country", "quarter"]),
        on=["geo", "period_str"],
        how="left",
    )
    merged_q.to_parquet(PROC_DIR / "panel_quarterly_airport_shocks.parquet", index=False)
    merged_q.to_csv(PROC_DIR / "panel_quarterly_airport_shocks.csv", index=False)

    summary = {
        "raw_rows_filtered_wide": int(len(raw)),
        "route_month_rows": int(len(route_month)),
        "airport_month_rows": int(len(airport_month)),
        "country_quarter_rows": int(len(country_quarter)),
        "country_quarter_min": str(country_quarter["quarter"].min()) if len(country_quarter) else None,
        "country_quarter_max": str(country_quarter["quarter"].max()) if len(country_quarter) else None,
        "countries": int(country_quarter["geo"].nunique()) if len(country_quarter) else 0,
        "total_route_open_events": int(country_quarter["route_open_q"].sum()) if len(country_quarter) else 0,
        "total_route_close_candidates": int(country_quarter["route_close_q"].sum()) if len(country_quarter) else 0,
        "total_route_open_persistent": int(country_quarter["route_open_persist_q"].sum()) if len(country_quarter) else 0,
        "total_route_close_persistent": int(country_quarter["route_close_persist_q"].sum()) if len(country_quarter) else 0,
    }
    (META_DIR / "flight_shock_build_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("[done] Built monthly route shock datasets and merged quarterly shock panel")


if __name__ == "__main__":
    main()
