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

TIME_RE = re.compile(r"^\d{4}(?:-Q[1-4])?$")
COUNTRY_RE = re.compile(r"^[A-Z]{2}$")
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


ANNUAL_SPECS = {
    "prc_hpi_a_idx": "hpi_index",
    "prc_hpi_a_growth": "hpi_growth",
    "avia_paoc_a_passengers": "air_passengers",
    "tps00019_net_migration_rate": "net_migration_rate",
    "tps00176_immigration": "immigration",
    "tps00177_emigration": "emigration",
    "tec00115_gdp_pc_growth": "gdp_pc_growth",
    "une_rt_a_unemployment": "unemployment_rate",
    "prc_hicp_aind_inflation": "inflation_hicp",
    "tps00001_population": "population",
    "irt_lt_mcby_a_long_rate": "long_rate",
}

QUARTERLY_SPECS = {
    "prc_hpi_q_idx": "hpi_index",
    "prc_hpi_q_growth": "hpi_qoq_growth_eurostat",
    "avia_paoc_q_passengers": "air_passengers",
}


def load_eurostat_wide(name: str) -> pd.DataFrame:
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def eurostat_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    geo_cols = [c for c in df.columns if "\\TIME_PERIOD" in str(c)]
    if len(geo_cols) != 1:
        raise ValueError(f"Expected exactly one geo/time column, got: {geo_cols}")
    geo_time_col = geo_cols[0]

    time_cols = [c for c in df.columns if TIME_RE.match(str(c))]
    id_cols = [c for c in df.columns if c not in time_cols]

    long_df = df.melt(
        id_vars=id_cols,
        value_vars=time_cols,
        var_name="time_period",
        value_name="value",
    )
    long_df = long_df.rename(columns={geo_time_col: "geo"})
    long_df["geo"] = long_df["geo"].astype(str)
    long_df = long_df[long_df["geo"].str.match(COUNTRY_RE, na=False)].copy()
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    return long_df


def load_series(name: str, value_name: str) -> pd.DataFrame:
    long_df = eurostat_wide_to_long(load_eurostat_wide(name))
    out = long_df[["geo", "time_period", "value"]].rename(columns={"value": value_name})
    return out


def build_annual_panel() -> pd.DataFrame:
    frames = [load_series(src, dst) for src, dst in ANNUAL_SPECS.items()]
    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on=["geo", "time_period"], how="outer")

    panel = panel[~panel["time_period"].str.contains("-Q", na=False)].copy()
    panel["year"] = pd.to_numeric(panel["time_period"], errors="coerce").astype("Int64")
    panel = panel.drop(columns=["time_period"])
    panel = panel.dropna(subset=["year"])
    panel["year"] = panel["year"].astype(int)
    panel = panel.sort_values(["geo", "year"]).reset_index(drop=True)
    panel = panel[panel["geo"].isin(TARGET_GEOS)].copy()

    # Derived annual migration variables
    panel["net_migration_level"] = panel["immigration"] - panel["emigration"]
    panel["immigration_rate_per_1000"] = panel["immigration"] / panel["population"] * 1000.0
    panel["emigration_rate_per_1000"] = panel["emigration"] / panel["population"] * 1000.0
    panel["net_migration_level_per_1000"] = panel["net_migration_level"] / panel["population"] * 1000.0

    # Logs and growth transforms
    panel["log_hpi"] = np.where(panel["hpi_index"] > 0, np.log(panel["hpi_index"]), np.nan)
    panel["log_air_passengers"] = np.where(panel["air_passengers"] > 0, np.log(panel["air_passengers"]), np.nan)
    panel["log_population"] = np.where(panel["population"] > 0, np.log(panel["population"]), np.nan)

    g = panel.groupby("geo", sort=False)
    panel["dlog_hpi"] = g["log_hpi"].diff() * 100.0
    panel["air_growth"] = g["log_air_passengers"].diff() * 100.0
    panel["pop_growth"] = g["log_population"].diff() * 100.0

    lag_cols = [
        "hpi_growth",
        "dlog_hpi",
        "net_migration_rate",
        "net_migration_level_per_1000",
        "immigration_rate_per_1000",
        "emigration_rate_per_1000",
        "air_growth",
        "gdp_pc_growth",
        "unemployment_rate",
        "inflation_hicp",
        "long_rate",
        "pop_growth",
    ]
    for col in lag_cols:
        panel[f"L1_{col}"] = g[col].shift(1)

    panel["post_2020"] = (panel["year"] >= 2020).astype(int)
    panel["post_2022"] = (panel["year"] >= 2022).astype(int)
    return panel


def build_quarterly_panel() -> pd.DataFrame:
    frames = [load_series(src, dst) for src, dst in QUARTERLY_SPECS.items()]
    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on=["geo", "time_period"], how="outer")

    panel = panel[panel["time_period"].str.contains("-Q", na=False)].copy()
    panel["period"] = pd.PeriodIndex(panel["time_period"], freq="Q")
    panel["year"] = panel["period"].dt.year.astype(int)
    panel["quarter"] = panel["period"].dt.quarter.astype(int)
    panel = panel.sort_values(["geo", "period"]).reset_index(drop=True)
    panel = panel[panel["geo"].isin(TARGET_GEOS)].copy()

    panel["log_hpi"] = np.where(panel["hpi_index"] > 0, np.log(panel["hpi_index"]), np.nan)
    panel["log_air_passengers"] = np.where(panel["air_passengers"] > 0, np.log(panel["air_passengers"]), np.nan)
    g = panel.groupby("geo", sort=False)

    panel["dlog_hpi_qoq"] = g["log_hpi"].diff() * 100.0
    panel["hpi_yoy"] = g["log_hpi"].diff(4) * 100.0
    panel["air_qoq"] = g["log_air_passengers"].diff() * 100.0
    panel["air_yoy"] = g["log_air_passengers"].diff(4) * 100.0

    for col in ["air_qoq", "air_yoy", "hpi_yoy", "dlog_hpi_qoq"]:
        panel[f"L1_{col}"] = g[col].shift(1)
        panel[f"L2_{col}"] = g[col].shift(2)

    panel["period_str"] = panel["period"].astype(str)
    panel["quarter_end"] = panel["period"].dt.to_timestamp(how="end")
    panel["post_2020q1"] = ((panel["year"] > 2020) | ((panel["year"] == 2020) & (panel["quarter"] >= 1))).astype(int)
    panel["pandemic_shock"] = ((panel["year"] == 2020) | (panel["year"] == 2021)).astype(int)
    return panel


def coverage_summary(df: pd.DataFrame, time_col: str, vars_to_check: list[str]) -> pd.DataFrame:
    rows = []
    for geo, g in df.groupby("geo"):
        row = {
            "geo": geo,
            "start": str(g[time_col].min()),
            "end": str(g[time_col].max()),
            "n_rows": int(len(g)),
        }
        for col in vars_to_check:
            row[f"n_{col}"] = int(g[col].notna().sum()) if col in g.columns else 0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("geo").reset_index(drop=True)


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    annual = build_annual_panel()
    quarterly = build_quarterly_panel()

    annual_cov = coverage_summary(
        annual,
        "year",
        ["hpi_index", "air_passengers", "net_migration_rate", "gdp_pc_growth", "unemployment_rate"],
    )
    quarterly_cov = coverage_summary(
        quarterly,
        "period",
        ["hpi_index", "air_passengers", "hpi_yoy", "air_yoy"],
    )

    annual.to_csv(PROC_DIR / "panel_annual.csv", index=False)
    quarterly.to_csv(PROC_DIR / "panel_quarterly.csv", index=False)
    annual.to_parquet(PROC_DIR / "panel_annual.parquet", index=False)
    quarterly.to_parquet(PROC_DIR / "panel_quarterly.parquet", index=False)
    annual_cov.to_csv(META_DIR / "coverage_annual.csv", index=False)
    quarterly_cov.to_csv(META_DIR / "coverage_quarterly.csv", index=False)

    panel_meta = {
        "annual_rows": int(len(annual)),
        "annual_countries": int(annual["geo"].nunique()),
        "annual_year_min": int(annual["year"].min()),
        "annual_year_max": int(annual["year"].max()),
        "quarterly_rows": int(len(quarterly)),
        "quarterly_countries": int(quarterly["geo"].nunique()),
        "quarterly_period_min": str(quarterly["period"].min()),
        "quarterly_period_max": str(quarterly["period"].max()),
        "annual_full_sample_rows_for_baseline": int(
            annual[["hpi_growth", "L1_net_migration_rate"]].dropna().shape[0]
        ),
        "quarterly_full_sample_rows_for_baseline": int(
            quarterly[["hpi_yoy", "L1_air_yoy"]].dropna().shape[0]
        ),
    }
    (META_DIR / "panel_build_summary.json").write_text(json.dumps(panel_meta, indent=2))

    print("[done] Built annual and quarterly panels")
    print(json.dumps(panel_meta, indent=2))


if __name__ == "__main__":
    main()
