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

NUTS2_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{2}$")


def build_region_quarter_shocks() -> pd.DataFrame:
    cw = pd.read_csv(META_DIR / "airport_nuts2_crosswalk.csv")
    cw = cw[cw["nuts2"].notna()].copy()
    cw["rep_airp"] = cw["rep_airp"].astype(str)
    cw["nuts2"] = cw["nuts2"].astype(str)

    airport_m = pd.read_parquet(PROC_DIR / "airport_monthly_route_shocks.parquet")
    airport_m["month"] = pd.PeriodIndex(airport_m["month"].astype(str), freq="M")
    merged = airport_m.merge(cw[["rep_airp", "nuts2"]], on="rep_airp", how="inner")
    merged["quarter"] = merged["month"].dt.asfreq("Q")

    reg_q = (
        merged.groupby(["nuts2", "quarter"], as_index=False)
        .agg(
            active_routes_q_mean=("active_routes", "mean"),
            route_open_q=("route_open_m", "sum"),
            route_close_q=("route_close_m_candidate", "sum"),
            route_open_persist_q=("route_open_persist_m", "sum"),
            route_close_persist_q=("route_close_persist_m", "sum"),
            passengers_total_q_sum=("passengers_total", "sum"),
            lag_active_routes_proxy=("lag_active_routes", "mean"),
        )
        .sort_values(["nuts2", "quarter"])
        .reset_index(drop=True)
    )
    reg_q["net_openings_q"] = reg_q["route_open_q"] - reg_q["route_close_q"]
    reg_q["net_openings_persist_q"] = reg_q["route_open_persist_q"] - reg_q["route_close_persist_q"]
    reg_q["open_persist_rate_norm_q"] = reg_q["route_open_persist_q"] / reg_q["lag_active_routes_proxy"].replace(0, np.nan)
    reg_q["close_persist_rate_norm_q"] = reg_q["route_close_persist_q"] / reg_q["lag_active_routes_proxy"].replace(0, np.nan)
    reg_q["period"] = reg_q["quarter"]
    reg_q["period_str"] = reg_q["quarter"].astype(str)
    reg_q = reg_q.rename(columns={"nuts2": "geo"})
    return reg_q


def _norm_quarter_str(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.replace("-", "", regex=False)
    # OECD sometimes provides YYYY-Q1 -> YYYYQ1, already handled by removing '-'
    return x


def build_oecd_nuts2_quarterly_hpi(valid_geos: set[str]) -> pd.DataFrame:
    oecd = pd.read_csv(RAW_DIR / "oecd_rhpi_all.csv")
    oecd["REF_AREA"] = oecd["REF_AREA"].astype(str)
    oecd["TIME_PERIOD"] = oecd["TIME_PERIOD"].astype(str)
    oecd["OBS_VALUE"] = pd.to_numeric(oecd["OBS_VALUE"], errors="coerce")

    keep = (
        (oecd["MEASURE"] == "RHPI")
        & (oecd["DWELLINGS"] == "_T")
        & (oecd["FREQ"] == "Q")
        & (oecd["VINTAGE"] == "_T")
        & (oecd["ADJUSTMENT"] == "N")
        & (oecd["REF_AREA"].isin(valid_geos))
    )
    oecd = oecd[keep].copy()
    oecd["period_str"] = _norm_quarter_str(oecd["TIME_PERIOD"])
    oecd["period"] = pd.PeriodIndex(oecd["period_str"], freq="Q")

    q_yoy = oecd[oecd["TRANSFORMATION"] == "GY"].copy()
    q_yoy = q_yoy.rename(columns={"REF_AREA": "geo", "OBS_VALUE": "rhpi_yoy"})[["geo", "period", "period_str", "rhpi_yoy"]]

    q_idx = oecd[oecd["TRANSFORMATION"] == "_Z"].copy()
    q_idx = q_idx.rename(columns={"REF_AREA": "geo", "OBS_VALUE": "rhpi_index"})[["geo", "period", "period_str", "rhpi_index"]]
    q_idx = q_idx.sort_values(["geo", "period"]).reset_index(drop=True)
    g = q_idx.groupby("geo", sort=False)
    q_idx["rhpi_qoq"] = g["rhpi_index"].transform(lambda s: np.log(s.where(s > 0)).diff() * 100.0)
    q_idx["rhpi_yoy_from_index"] = g["rhpi_index"].transform(lambda s: np.log(s.where(s > 0)).diff(4) * 100.0)

    panel = q_idx.merge(q_yoy, on=["geo", "period", "period_str"], how="outer")
    panel["rhpi_yoy"] = panel["rhpi_yoy"].combine_first(panel["rhpi_yoy_from_index"])
    panel["year"] = panel["period"].dt.year.astype(int)
    panel["quarter"] = panel["period"].dt.quarter.astype(int)
    panel["country"] = panel["geo"].str[:2]
    return panel


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    reg_q_shocks = build_region_quarter_shocks()
    valid_shock_geos = set(reg_q_shocks["geo"].astype(str))
    valid_shock_geos = {g for g in valid_shock_geos if NUTS2_RE.match(g)}

    oecd_q = build_oecd_nuts2_quarterly_hpi(valid_shock_geos)
    panel = oecd_q.merge(
        reg_q_shocks.drop(columns=["quarter"], errors="ignore"),
        on=["geo", "period", "period_str"],
        how="left",
    )
    panel = panel.sort_values(["geo", "period"]).reset_index(drop=True)

    g = panel.groupby("geo", sort=False)
    for col in [
        "rhpi_yoy",
        "rhpi_qoq",
        "route_open_q",
        "route_close_q",
        "route_open_persist_q",
        "route_close_persist_q",
        "net_openings_q",
        "net_openings_persist_q",
        "open_persist_rate_norm_q",
        "close_persist_rate_norm_q",
    ]:
        if col in panel.columns:
            panel[f"L1_{col}"] = g[col].shift(1)

    panel["quarter_id"] = panel["year"] * 4 + panel["quarter"]
    panel["post_2020q1"] = ((panel["year"] > 2020) | ((panel["year"] == 2020) & (panel["quarter"] >= 1))).astype(int)

    out_pq = PROC_DIR / "panel_nuts2_quarterly_route_shocks.parquet"
    out_csv = PROC_DIR / "panel_nuts2_quarterly_route_shocks.csv"
    panel.to_parquet(out_pq, index=False)
    panel.to_csv(out_csv, index=False)

    summary = {
        "rows": int(len(panel)),
        "regions": int(panel["geo"].nunique()),
        "countries": int(panel["country"].nunique()),
        "period_min": str(panel["period"].min()) if len(panel) else None,
        "period_max": str(panel["period"].max()) if len(panel) else None,
        "complete_eventstudy_rows": int(panel[["rhpi_yoy", "route_open_persist_q"]].dropna().shape[0]),
    }
    (META_DIR / "regional_route_quarterly_panel_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[done] {out_pq}")


if __name__ == "__main__":
    main()
