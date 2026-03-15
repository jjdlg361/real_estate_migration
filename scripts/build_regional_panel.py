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

YEAR_RE = re.compile(r"^\d{4}$")
NUTS2_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{2}$")


def eurostat_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    geo_cols = [c for c in df.columns if "\\TIME_PERIOD" in str(c)]
    if len(geo_cols) != 1:
        raise ValueError(f"Expected 1 geo/time col, got {geo_cols}")
    geo_col = geo_cols[0]
    time_cols = [c for c in df.columns if YEAR_RE.match(str(c))]
    id_cols = [c for c in df.columns if c not in time_cols]
    out = df.melt(id_vars=id_cols, value_vars=time_cols, var_name="year", value_name="value")
    out = out.rename(columns={geo_col: "geo"})
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["year"]).copy()
    out["year"] = out["year"].astype(int)
    return out


def load_eurostat_series(file_name: str, value_name: str, filters: dict[str, str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / file_name)
    long_df = eurostat_wide_to_long(df)
    if filters:
        mask = pd.Series(True, index=long_df.index)
        for col, val in filters.items():
            mask &= long_df[col].astype(str) == str(val)
        long_df = long_df[mask].copy()
    out = long_df[["geo", "year", "value"]].rename(columns={"value": value_name})
    return out


def build_oecd_rhpi_nuts2() -> pd.DataFrame:
    oecd = pd.read_csv(RAW_DIR / "oecd_rhpi_all.csv")
    oecd["TIME_PERIOD"] = oecd["TIME_PERIOD"].astype(str)
    oecd["OBS_VALUE"] = pd.to_numeric(oecd["OBS_VALUE"], errors="coerce")
    oecd["REF_AREA"] = oecd["REF_AREA"].astype(str)

    euro_nuts2_geo = load_eurostat_series("tgs00096_nuts2_population.csv", "population_tmp")
    euro_nuts2_set = set(euro_nuts2_geo["geo"].astype(str))
    euro_nuts2_set = {g for g in euro_nuts2_set if NUTS2_RE.match(g)}

    keep_base = (
        (oecd["MEASURE"] == "RHPI")
        & (oecd["DWELLINGS"] == "_T")
        & (oecd["VINTAGE"] == "_T")
        & (oecd["ADJUSTMENT"] == "N")
        & (oecd["REF_AREA"].isin(euro_nuts2_set))
    )
    oecd = oecd[keep_base].copy()

    # Prefer annual YoY growth directly when available.
    ann_gy = oecd[(oecd["FREQ"] == "A") & (oecd["TRANSFORMATION"] == "GY")].copy()
    ann_gy["year"] = pd.to_numeric(ann_gy["TIME_PERIOD"], errors="coerce").astype("Int64")
    ann_gy = ann_gy.dropna(subset=["year"]).copy()
    ann_gy["year"] = ann_gy["year"].astype(int)
    ann_gy = ann_gy.rename(columns={"REF_AREA": "geo", "OBS_VALUE": "rhpi_growth_oecd"})[
        ["geo", "year", "rhpi_growth_oecd"]
    ]

    ann_idx = oecd[(oecd["FREQ"] == "A") & (oecd["TRANSFORMATION"] == "_Z")].copy()
    ann_idx["year"] = pd.to_numeric(ann_idx["TIME_PERIOD"], errors="coerce").astype("Int64")
    ann_idx = ann_idx.dropna(subset=["year"]).copy()
    ann_idx["year"] = ann_idx["year"].astype(int)
    ann_idx = ann_idx.rename(columns={"REF_AREA": "geo", "OBS_VALUE": "rhpi_index_oecd"})[
        ["geo", "year", "rhpi_index_oecd"]
    ]
    ann_idx = ann_idx.sort_values(["geo", "year"]).reset_index(drop=True)
    ann_idx["rhpi_growth_from_index"] = (
        ann_idx.groupby("geo")["rhpi_index_oecd"].transform(lambda s: np.log(s.where(s > 0)).diff() * 100.0)
    )

    rhpi = ann_idx.merge(ann_gy, on=["geo", "year"], how="outer")
    rhpi["rhpi_growth"] = rhpi["rhpi_growth_oecd"].combine_first(rhpi["rhpi_growth_from_index"])
    rhpi["country"] = rhpi["geo"].str[:2]
    return rhpi


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    rhpi = build_oecd_rhpi_nuts2()

    regional_series = [
        load_eurostat_series("tgs00077_nuts2_air_passengers.csv", "air_passengers_ths"),
        load_eurostat_series("tgs00099_nuts2_net_migration_rate.csv", "net_migration_rate"),
        load_eurostat_series("tgs00003_nuts2_gdp.csv", "gdp_mio_eur"),
        load_eurostat_series("tgs00010_nuts2_unemployment.csv", "unemployment_rate"),
        load_eurostat_series("tgs00096_nuts2_population.csv", "population"),
        load_eurostat_series("tgs00026_nuts2_disposable_income.csv", "disp_income_mio_pps"),
    ]

    panel = rhpi[["geo", "country", "year", "rhpi_index_oecd", "rhpi_growth"]].copy()
    for s in regional_series:
        panel = panel.merge(s, on=["geo", "year"], how="left")

    panel = panel[panel["geo"].str.match(NUTS2_RE, na=False)].copy()
    panel = panel.sort_values(["geo", "year"]).reset_index(drop=True)

    panel["air_passengers"] = panel["air_passengers_ths"] * 1000.0
    panel["log_air_passengers"] = np.where(panel["air_passengers"] > 0, np.log(panel["air_passengers"]), np.nan)
    panel["log_population"] = np.where(panel["population"] > 0, np.log(panel["population"]), np.nan)
    panel["gdp_per_capita_eur"] = np.where(panel["population"] > 0, panel["gdp_mio_eur"] * 1_000_000 / panel["population"], np.nan)
    panel["log_gdp_per_capita"] = np.where(panel["gdp_per_capita_eur"] > 0, np.log(panel["gdp_per_capita_eur"]), np.nan)
    g = panel.groupby("geo", sort=False)
    panel["air_growth"] = g["log_air_passengers"].diff() * 100.0
    panel["pop_growth"] = g["log_population"].diff() * 100.0
    panel["gdp_pc_growth"] = g["log_gdp_per_capita"].diff() * 100.0

    for col in ["rhpi_growth", "net_migration_rate", "air_growth", "unemployment_rate", "pop_growth", "gdp_pc_growth"]:
        panel[f"L1_{col}"] = g[col].shift(1)

    panel["post_2020"] = (panel["year"] >= 2020).astype(int)
    panel["post_2022"] = (panel["year"] >= 2022).astype(int)

    out_csv = PROC_DIR / "panel_nuts2_annual.csv"
    out_parquet = PROC_DIR / "panel_nuts2_annual.parquet"
    panel.to_csv(out_csv, index=False)
    panel.to_parquet(out_parquet, index=False)

    coverage = (
        panel.groupby("geo")
        .agg(
            country=("country", "first"),
            start=("year", "min"),
            end=("year", "max"),
            n_rows=("year", "size"),
            n_rhpi=("rhpi_growth", lambda s: s.notna().sum()),
            n_air=("air_passengers", lambda s: s.notna().sum()),
            n_migr=("net_migration_rate", lambda s: s.notna().sum()),
            n_unemp=("unemployment_rate", lambda s: s.notna().sum()),
        )
        .reset_index()
        .sort_values("geo")
    )
    coverage.to_csv(META_DIR / "coverage_nuts2_annual.csv", index=False)

    summary = {
        "rows": int(len(panel)),
        "regions": int(panel["geo"].nunique()),
        "countries": int(panel["country"].nunique()),
        "year_min": int(panel["year"].min()) if len(panel) else None,
        "year_max": int(panel["year"].max()) if len(panel) else None,
        "complete_baseline_rows": int(
            panel[["rhpi_growth", "L1_net_migration_rate", "L1_air_growth"]].dropna().shape[0]
        ),
    }
    (META_DIR / "regional_panel_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("[done] Built NUTS2 annual panel (OECD RHPI + Eurostat regional mobility/macros)")


if __name__ == "__main__":
    main()
