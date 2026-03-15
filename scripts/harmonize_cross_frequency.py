#!/usr/bin/env python3
from __future__ import annotations

import json
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
META_DIR = ROOT / "data" / "metadata"
RESULTS_DIR = ROOT / "results"

WB_UEM_FILE = RAW_DIR / "worldbank_unemployment_rate_annual.csv"
WB_GDPG_FILE = RAW_DIR / "worldbank_gdp_pc_growth_annual.csv"
WB_NETMIG_FILE = RAW_DIR / "worldbank_net_migration_level_annual.csv"
WB_LENDRATE_FILE = RAW_DIR / "worldbank_lending_rate_annual.csv"
WB_INFL_FILE = RAW_DIR / "worldbank_inflation_cpi_annual.csv"
WB_POPG_FILE = RAW_DIR / "worldbank_population_growth_annual.csv"
WB_POP_FILE = RAW_DIR / "worldbank_population_level_annual.csv"
WB_AIRPSG_FILE = RAW_DIR / "worldbank_air_passengers_annual.csv"

# World Bank indicators
WB_UEM_IND = "SL.UEM.TOTL.ZS"
WB_GDPG_IND = "NY.GDP.PCAP.KD.ZG"
WB_NETMIG_IND = "SM.POP.NETM"
WB_LENDRATE_IND = "FR.INR.LEND"
WB_INFL_IND = "FP.CPI.TOTL.ZG"
WB_POPG_IND = "SP.POP.GROW"
WB_POP_IND = "SP.POP.TOTL"
WB_AIRPSG_IND = "IS.AIR.PSGR"

ONS_LTIM_LATEST_URL = (
    "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/"
    "internationalmigration/datasets/longterminternationalimmigrationemigrationandnetmigrationflowsprovisional/"
    "yearendingjune2025/ltimnov25.xlsx"
)

# Poland BDL migration variables (official Statistics Poland API).
# Annual (full year): 269600 immigration, 269586 emigration
# First-half-year (H1): 269599 immigration, 269585 emigration
PL_BDL_IMM_ANNUAL_VAR = 269600
PL_BDL_EMI_ANNUAL_VAR = 269586
PL_BDL_IMM_H1_VAR = 269599
PL_BDL_EMI_H1_VAR = 269585

# WB2 -> Eurostat 2-letter codes for key targets (extendable).
WB2_TO_EU = {
    "GB": "UK",
    "PL": "PL",
    "ES": "ES",
    "IT": "IT",
    "PT": "PT",
    "FR": "FR",
    "DE": "DE",
    "NL": "NL",
    "BE": "BE",
    "SE": "SE",
    "NO": "NO",
    "IS": "IS",
    "IE": "IE",
    "FI": "FI",
    "DK": "DK",
    "AT": "AT",
    "CH": "CH",
    "CZ": "CZ",
    "SK": "SK",
    "SI": "SI",
    "HU": "HU",
    "RO": "RO",
    "BG": "BG",
    "HR": "HR",
    "LV": "LV",
    "LT": "LT",
    "EE": "EE",
    "LU": "LU",
    "CY": "CY",
    "MT": "MT",
    # Supplemental non-core countries to widen blended stream coverage.
    "TR": "TR",
    "IL": "IL",
    "JP": "JP",
    "MX": "MX",
    "AU": "AU",
    "NZ": "NZ",
    "CL": "CL",
    "BR": "BR",
    "CN": "CN",
    "IN": "IN",
    "SA": "SA",
    # New supplemental additions (for broader 2025-ready blended coverage)
    "US": "US",
    "CA": "CA",
    "KR": "KR",
    "CO": "CO",
    "ZA": "ZA",
    "GR": "EL",
}


def build_country_name_lookup() -> dict[str, str]:
    wb = fetch_wb(WB_UEM_IND, WB_UEM_FILE)
    x = wb[["country_code_wb2", "country_name"]].dropna().drop_duplicates()
    x["geo"] = x["country_code_wb2"].map(WB2_TO_EU)
    x = x.dropna(subset=["geo"]).drop_duplicates(subset=["geo"], keep="first")
    out = dict(zip(x["geo"], x["country_name"]))
    out.setdefault("UK", "United Kingdom")
    return out


def build_iso3_to_geo_map() -> dict[str, str]:
    wb = fetch_wb(WB_UEM_IND, WB_UEM_FILE)
    x = wb[["country_iso3", "country_code_wb2"]].dropna().drop_duplicates()
    x["geo"] = x["country_code_wb2"].map(WB2_TO_EU)
    x = x.dropna(subset=["country_iso3", "geo"]).drop_duplicates(subset=["country_iso3"], keep="first")
    return dict(zip(x["country_iso3"], x["geo"]))


def build_oecd_country_hpi_annual() -> pd.DataFrame:
    oecd_file = RAW_DIR / "oecd_rhpi_all.csv"
    if not oecd_file.exists():
        return pd.DataFrame(columns=["geo", "year", "oecd_hpi_growth_a"])

    usecols = [
        "REF_AREA_TYPE",
        "REF_AREA",
        "FREQ",
        "MEASURE",
        "ADJUSTMENT",
        "TRANSFORMATION",
        "VINTAGE",
        "DWELLINGS",
        "TIME_PERIOD",
        "OBS_VALUE",
    ]
    d = pd.read_csv(oecd_file, usecols=usecols)
    keep = (
        (d["REF_AREA_TYPE"] == "COU")
        & (d["MEASURE"] == "RHPI")
        & (d["DWELLINGS"].isin(["_T", "SINGLE_F", "MULTI_F"]))
        & (d["ADJUSTMENT"].isin(["N", "S"]))
    )
    d = d[keep].copy()
    d["geo"] = d["REF_AREA"].map(build_iso3_to_geo_map())
    d = d.dropna(subset=["geo"])
    d["adj_rank"] = np.where(d["ADJUSTMENT"] == "N", 0, 1)
    d["dw_rank"] = np.select(
        [d["DWELLINGS"] == "_T", d["DWELLINGS"] == "SINGLE_F", d["DWELLINGS"] == "MULTI_F"],
        [0, 1, 2],
        default=3,
    )
    d["vin_rank"] = np.where(d["VINTAGE"] == "_T", 0, 1)

    ann_gy = d[(d["FREQ"] == "A") & (d["TRANSFORMATION"] == "GY")].copy()
    ann_gy = ann_gy.sort_values(["geo", "TIME_PERIOD", "adj_rank", "dw_rank", "vin_rank"]).drop_duplicates(
        ["geo", "TIME_PERIOD"], keep="first"
    )
    ann_gy["year"] = pd.to_numeric(ann_gy["TIME_PERIOD"], errors="coerce").astype("Int64")
    ann_gy["oecd_hpi_growth_a"] = pd.to_numeric(ann_gy["OBS_VALUE"], errors="coerce")
    ann_gy = ann_gy.dropna(subset=["year"]).astype({"year": int})
    ann_gy = ann_gy[["geo", "year", "oecd_hpi_growth_a"]]

    ann_idx = d[(d["FREQ"] == "A") & (d["TRANSFORMATION"] == "_Z")].copy()
    ann_idx = ann_idx.sort_values(["geo", "TIME_PERIOD", "adj_rank", "dw_rank", "vin_rank"]).drop_duplicates(
        ["geo", "TIME_PERIOD"], keep="first"
    )
    ann_idx["year"] = pd.to_numeric(ann_idx["TIME_PERIOD"], errors="coerce").astype("Int64")
    ann_idx["oecd_hpi_index_a"] = pd.to_numeric(ann_idx["OBS_VALUE"], errors="coerce")
    ann_idx = ann_idx.dropna(subset=["year"]).astype({"year": int})
    ann_idx = ann_idx.sort_values(["geo", "year"]).reset_index(drop=True)
    ann_idx["oecd_hpi_growth_from_index_a"] = ann_idx.groupby("geo")["oecd_hpi_index_a"].transform(
        lambda s: np.log(s.where(s > 0)).diff() * 100.0
    )
    ann_idx = ann_idx[["geo", "year", "oecd_hpi_growth_from_index_a"]]

    out = ann_idx.merge(ann_gy, on=["geo", "year"], how="outer")
    out["oecd_hpi_growth_a"] = out["oecd_hpi_growth_a"].combine_first(out["oecd_hpi_growth_from_index_a"])
    out = out[["geo", "year", "oecd_hpi_growth_a"]].drop_duplicates(["geo", "year"])
    return out


def build_oecd_country_hpi_quarterly() -> pd.DataFrame:
    oecd_file = RAW_DIR / "oecd_rhpi_all.csv"
    if not oecd_file.exists():
        return pd.DataFrame(columns=["geo", "period_str", "oecd_hpi_yoy_q"])

    usecols = [
        "REF_AREA_TYPE",
        "REF_AREA",
        "FREQ",
        "MEASURE",
        "ADJUSTMENT",
        "TRANSFORMATION",
        "VINTAGE",
        "DWELLINGS",
        "TIME_PERIOD",
        "OBS_VALUE",
    ]
    d = pd.read_csv(oecd_file, usecols=usecols)
    keep = (
        (d["REF_AREA_TYPE"] == "COU")
        & (d["MEASURE"] == "RHPI")
        & (d["DWELLINGS"].isin(["_T", "SINGLE_F", "MULTI_F"]))
        & (d["ADJUSTMENT"].isin(["N", "S"]))
        & (d["FREQ"] == "Q")
    )
    d = d[keep].copy()
    d["geo"] = d["REF_AREA"].map(build_iso3_to_geo_map())
    d = d.dropna(subset=["geo"])
    d["adj_rank"] = np.where(d["ADJUSTMENT"] == "N", 0, 1)
    d["dw_rank"] = np.select(
        [d["DWELLINGS"] == "_T", d["DWELLINGS"] == "SINGLE_F", d["DWELLINGS"] == "MULTI_F"],
        [0, 1, 2],
        default=3,
    )
    d["vin_rank"] = np.where(d["VINTAGE"] == "_T", 0, 1)

    q_gy = d[d["TRANSFORMATION"] == "GY"].copy()
    q_gy = q_gy.sort_values(["geo", "TIME_PERIOD", "adj_rank", "dw_rank", "vin_rank"]).drop_duplicates(
        ["geo", "TIME_PERIOD"], keep="first"
    )
    q_gy["time"] = q_gy["TIME_PERIOD"].astype(str).str.replace("-", "", regex=False)
    q_gy = q_gy[q_gy["time"].str.match(r"^\d{4}Q[1-4]$", na=False)].copy()
    q_gy["period"] = pd.PeriodIndex(q_gy["time"], freq="Q")
    q_gy["oecd_hpi_yoy_q"] = pd.to_numeric(q_gy["OBS_VALUE"], errors="coerce")
    q_gy["period_str"] = q_gy["period"].astype(str)
    q_gy = q_gy[["geo", "period_str", "oecd_hpi_yoy_q"]]

    q_idx = d[d["TRANSFORMATION"] == "_Z"].copy()
    q_idx = q_idx.sort_values(["geo", "TIME_PERIOD", "adj_rank", "dw_rank", "vin_rank"]).drop_duplicates(
        ["geo", "TIME_PERIOD"], keep="first"
    )
    q_idx["time"] = q_idx["TIME_PERIOD"].astype(str).str.replace("-", "", regex=False)
    q_idx = q_idx[q_idx["time"].str.match(r"^\d{4}Q[1-4]$", na=False)].copy()
    q_idx["period"] = pd.PeriodIndex(q_idx["time"], freq="Q")
    q_idx["oecd_hpi_index_q"] = pd.to_numeric(q_idx["OBS_VALUE"], errors="coerce")
    q_idx = q_idx.sort_values(["geo", "period"]).reset_index(drop=True)
    q_idx["oecd_hpi_yoy_from_index_q"] = q_idx.groupby("geo")["oecd_hpi_index_q"].transform(
        lambda s: np.log(s.where(s > 0)).diff(4) * 100.0
    )
    q_idx["period_str"] = q_idx["period"].astype(str)
    q_idx = q_idx[["geo", "period_str", "oecd_hpi_yoy_from_index_q"]]

    out = q_idx.merge(q_gy, on=["geo", "period_str"], how="outer")
    out["oecd_hpi_yoy_q"] = out["oecd_hpi_yoy_q"].combine_first(out["oecd_hpi_yoy_from_index_q"])
    out = out[["geo", "period_str", "oecd_hpi_yoy_q"]].drop_duplicates(["geo", "period_str"])
    return out


def build_extra_country_rows(annual_base: pd.DataFrame, quarterly_base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Build supplemental country rows (outside core base geos) using public OECD/WB sources.
    """
    base_geos = set(annual_base["geo"].dropna().astype(str).unique())
    oecd_a = build_oecd_country_hpi_annual()
    oecd_q = build_oecd_country_hpi_quarterly()
    candidate_geos = sorted((set(oecd_a["geo"].dropna().astype(str)) | set(oecd_q["geo"].dropna().astype(str))) - base_geos)
    if not candidate_geos:
        return pd.DataFrame(columns=annual_base.columns), pd.DataFrame(columns=quarterly_base.columns), []

    # World Bank annual sources for supplemental macro/migration/air controls.
    wb_pop = fetch_wb(WB_POP_IND, WB_POP_FILE)
    wb_net = fetch_wb(WB_NETMIG_IND, WB_NETMIG_FILE)
    wb_gdp = fetch_wb(WB_GDPG_IND, WB_GDPG_FILE)
    wb_uem = fetch_wb(WB_UEM_IND, WB_UEM_FILE)
    wb_inf = fetch_wb(WB_INFL_IND, WB_INFL_FILE)
    wb_lng = fetch_wb(WB_LENDRATE_IND, WB_LENDRATE_FILE)
    wb_popg = fetch_wb(WB_POPG_IND, WB_POPG_FILE)
    wb_air = fetch_wb(WB_AIRPSG_IND, WB_AIRPSG_FILE)

    def _prep(wb: pd.DataFrame, col: str) -> pd.DataFrame:
        x = wb.copy()
        x["geo"] = x["country_code_wb2"].map(WB2_TO_EU)
        x = x.dropna(subset=["geo", "year"])
        x = x[x["geo"].isin(candidate_geos)].copy()
        x = x.rename(columns={"value": col})
        x["year"] = pd.to_numeric(x["year"], errors="coerce").astype("Int64")
        x = x.dropna(subset=["year"]).astype({"year": int})
        return x[["geo", "year", col]].drop_duplicates(["geo", "year"])

    pop = _prep(wb_pop, "population")
    net = _prep(wb_net, "wb_net_migration_level")
    gdp = _prep(wb_gdp, "gdp_pc_growth")
    uem = _prep(wb_uem, "unemployment_rate")
    inf = _prep(wb_inf, "inflation_hicp")
    lng = _prep(wb_lng, "long_rate")
    popg = _prep(wb_popg, "pop_growth")
    air = _prep(wb_air, "air_passengers_wb")

    # Annual supplemental rows.
    a = oecd_a[oecd_a["geo"].isin(candidate_geos)].copy()
    a = a.rename(columns={"oecd_hpi_growth_a": "hpi_growth"})
    a = a.merge(pop, on=["geo", "year"], how="outer")
    a = a.merge(net, on=["geo", "year"], how="left")
    a = a.merge(gdp, on=["geo", "year"], how="left")
    a = a.merge(uem, on=["geo", "year"], how="left")
    a = a.merge(inf, on=["geo", "year"], how="left")
    a = a.merge(lng, on=["geo", "year"], how="left")
    a = a.merge(popg, on=["geo", "year"], how="left")
    a = a.merge(air, on=["geo", "year"], how="left")
    a = a.sort_values(["geo", "year"]).reset_index(drop=True)
    a["air_growth"] = a.groupby("geo")["air_passengers_wb"].transform(lambda s: np.log(s.where(s > 0)).diff() * 100.0)
    a["net_migration_rate"] = np.where(
        pd.to_numeric(a["population"], errors="coerce") > 0,
        pd.to_numeric(a["wb_net_migration_level"], errors="coerce") / pd.to_numeric(a["population"], errors="coerce") * 1000.0,
        np.nan,
    )
    a["country_name"] = a["geo"].map(build_country_name_lookup())
    if "net_migration_rate_source" in annual_base.columns:
        a["net_migration_rate_source"] = pd.Series(index=a.index, dtype="object")
        a.loc[a["net_migration_rate"].notna(), "net_migration_rate_source"] = "wb_net_migration"
    if "air_passengers" in annual_base.columns:
        a["air_passengers"] = a["air_passengers_wb"]

    for col in annual_base.columns:
        if col not in a.columns:
            a[col] = np.nan
    a = a[annual_base.columns].drop_duplicates(["geo", "year"]).sort_values(["geo", "year"]).reset_index(drop=True)

    # Quarterly supplemental rows from OECD country-level RHPI.
    q = oecd_q[oecd_q["geo"].isin(candidate_geos)].copy()
    q = q.rename(columns={"oecd_hpi_yoy_q": "hpi_yoy"})
    q["period"] = pd.PeriodIndex(q["period_str"], freq="Q")
    q["year"] = q["period"].dt.year.astype(int)
    q["quarter"] = q["period"].dt.quarter.astype(int)
    q["country_name"] = q["geo"].map(build_country_name_lookup())

    for col in quarterly_base.columns:
        if col not in q.columns:
            q[col] = np.nan
    q = q[quarterly_base.columns].drop_duplicates(["geo", "year", "quarter"]).sort_values(["geo", "year", "quarter"]).reset_index(drop=True)

    return a, q, candidate_geos


def fetch_wb(indicator: str, out_file: Path) -> pd.DataFrame:
    if out_file.exists():
        return pd.read_csv(out_file)
    rows = []
    page = 1
    while True:
        url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&per_page=20000&page={page}"
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, list) or len(payload) < 2:
            raise RuntimeError(f"Unexpected WB payload for {indicator}")
        meta, data = payload
        for d in data:
            c = d.get("country") or {}
            rows.append(
                {
                    "country_code_wb2": c.get("id"),
                    "country_name": c.get("value"),
                    "country_iso3": d.get("countryiso3code"),
                    "year": pd.to_numeric(d.get("date"), errors="coerce"),
                    "value": pd.to_numeric(d.get("value"), errors="coerce"),
                    "indicator": indicator,
                }
            )
        if page >= int(meta["pages"]):
            break
        page += 1
    out = pd.DataFrame(rows).dropna(subset=["year"]).copy()
    out["year"] = out["year"].astype(int)
    out.to_csv(out_file, index=False)
    return out


def annualize_quarterly(q: pd.DataFrame) -> pd.DataFrame:
    d = q.copy()
    if "period_str" not in d.columns:
        d["period_str"] = d["year"].astype(int).astype(str) + "Q" + d["quarter"].astype(int).astype(str)
    d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")
    d = d.dropna(subset=["geo", "year"]).copy()
    d["year"] = d["year"].astype(int)

    agg = (
        d.groupby(["geo", "year"], as_index=False)
        .agg(
            q_hpi_yoy_mean=("hpi_yoy", "mean"),
            q_hpi_yoy_median=("hpi_yoy", "median"),
            q_air_yoy_mean=("air_yoy", "mean"),
            q_air_yoy_median=("air_yoy", "median"),
            q_hpi_obs=("hpi_yoy", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            q_air_obs=("air_yoy", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        )
        .sort_values(["geo", "year"])
    )

    # Q4 values often used as year-end nowcast.
    d2 = d.copy()
    d2["quarter"] = pd.to_numeric(d2["quarter"], errors="coerce")
    q4 = d2[d2["quarter"] == 4][["geo", "year", "hpi_yoy", "air_yoy"]].rename(
        columns={"hpi_yoy": "q4_hpi_yoy", "air_yoy": "q4_air_yoy"}
    )
    agg = agg.merge(q4, on=["geo", "year"], how="left")
    return agg


def uk_ons_annual_netmig_rate(pop_annual: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    # Source 1: historic UK ONS quarterly LTIM overlay already ingested in pipeline.
    q_overlay_path = PROC_DIR / "panel_quarterly_countryweb_overlay.parquet"
    if q_overlay_path.exists():
        q = pd.read_parquet(q_overlay_path)
        q = q[(q["geo"] == "UK") & q["uk_ons_net_migration_q"].notna()].copy()
        if not q.empty:
            q["year"] = pd.to_numeric(q["year"], errors="coerce").astype("Int64")
            q = q.dropna(subset=["year"]).copy()
            q["year"] = q["year"].astype(int)
            a_q = q.groupby(["geo", "year"], as_index=False)["uk_ons_net_migration_q"].sum().rename(
                columns={"uk_ons_net_migration_q": "uk_ons_net_migration_a"}
            )
            a_q = a_q.merge(pop_annual[["geo", "year", "population"]], on=["geo", "year"], how="left")
            a_q["uk_ons_net_migration_rate_a"] = np.where(
                pd.to_numeric(a_q["population"], errors="coerce") > 0,
                pd.to_numeric(a_q["uk_ons_net_migration_a"], errors="coerce")
                / pd.to_numeric(a_q["population"], errors="coerce")
                * 1000.0,
                np.nan,
            )
            a_q["uk_ons_net_migration_rate_source"] = "uk_ons_q_aggregated"
            a_q["source_priority"] = 1
            parts.append(
                a_q[
                    [
                        "geo",
                        "year",
                        "uk_ons_net_migration_a",
                        "uk_ons_net_migration_rate_a",
                        "uk_ons_net_migration_rate_source",
                        "source_priority",
                    ]
                ]
            )

    # Source 2: latest official ONS annual LTIM flows (YE Jun), preferred when available.
    try:
        r = requests.get(ONS_LTIM_LATEST_URL, timeout=60)
        r.raise_for_status()
        if r.content.startswith(b"PK"):
            t1 = pd.read_excel(io.BytesIO(r.content), sheet_name="1", header=5)
            t1 = t1.rename(
                columns={
                    "Flow\n[note 2]": "flow",
                    "Period\n[note 10]": "period",
                    "All Nationalities\n[note 1]": "all_nat",
                }
            )
            keep = {"flow", "period", "all_nat"}
            if keep.issubset(set(t1.columns)):
                t1 = t1[list(keep)].copy()
                t1["flow"] = t1["flow"].astype(str).str.strip()
                t1["period"] = t1["period"].astype(str).str.strip()
                t1["all_nat"] = pd.to_numeric(t1["all_nat"], errors="coerce")
                t1 = t1[t1["flow"] == "Net migration"].copy()
                t1 = t1[t1["period"].str.contains(r"^YE Jun", regex=True, na=False)].copy()
                if not t1.empty:
                    y2 = t1["period"].str.extract(r"YE Jun\s+(\d{2})", expand=False)
                    t1["year"] = pd.to_numeric(y2, errors="coerce")
                    t1 = t1.dropna(subset=["year"]).copy()
                    t1["year"] = (2000 + t1["year"].astype(int)).astype(int)
                    t1["geo"] = "UK"
                    t1 = t1.rename(columns={"all_nat": "uk_ons_net_migration_a"})
                    t1 = t1.merge(pop_annual[["geo", "year", "population"]], on=["geo", "year"], how="left")
                    t1["uk_ons_net_migration_rate_a"] = np.where(
                        pd.to_numeric(t1["population"], errors="coerce") > 0,
                        pd.to_numeric(t1["uk_ons_net_migration_a"], errors="coerce")
                        / pd.to_numeric(t1["population"], errors="coerce")
                        * 1000.0,
                        np.nan,
                    )
                    t1["uk_ons_net_migration_rate_source"] = "uk_ons_ltim_ye_jun"
                    t1["source_priority"] = 2
                    parts.append(
                        t1[
                            [
                                "geo",
                                "year",
                                "uk_ons_net_migration_a",
                                "uk_ons_net_migration_rate_a",
                                "uk_ons_net_migration_rate_source",
                                "source_priority",
                            ]
                        ]
                    )
    except Exception:
        # Keep harmonization resilient; quarterly overlay source remains available.
        pass

    if not parts:
        return pd.DataFrame(columns=["geo", "year", "uk_ons_net_migration_rate_a", "uk_ons_net_migration_rate_source"])

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["geo", "year", "source_priority"]).drop_duplicates(["geo", "year"], keep="last")

    # Backfill UK population from World Bank when base panel population is missing,
    # so the latest ONS annual net-migration flow can be converted to per-1,000 rates.
    wb_pop = fetch_wb(WB_POP_IND, WB_POP_FILE).copy()
    wb_pop = wb_pop[wb_pop["country_code_wb2"] == "GB"].copy()
    wb_pop["geo"] = "UK"
    wb_pop = wb_pop.rename(columns={"value": "wb_population"})
    out = out.merge(wb_pop[["geo", "year", "wb_population"]], on=["geo", "year"], how="left")
    pop = (
        pd.to_numeric(out["population"], errors="coerce")
        if "population" in out.columns
        else pd.Series(np.nan, index=out.index, dtype=float)
    )
    pop_wb = pd.to_numeric(out.get("wb_population"), errors="coerce")
    pop_use = pop.combine_first(pop_wb)
    # ONS latest year can appear before population vintages are updated; use last known population.
    pop_use = pd.Series(pop_use, index=out.index).sort_index().ffill()
    out["uk_ons_net_migration_rate_a"] = np.where(
        pop_use > 0,
        pd.to_numeric(out["uk_ons_net_migration_a"], errors="coerce") / pop_use * 1000.0,
        np.nan,
    )

    return out[["geo", "year", "uk_ons_net_migration_rate_a", "uk_ons_net_migration_rate_source"]]


def bdl_poland_series(var_id: int) -> pd.DataFrame:
    r = requests.get(
        "https://bdl.stat.gov.pl/api/v1/data/by-unit/000000000000",
        params={"format": "json", "var-id": var_id},
        timeout=60,
    )
    r.raise_for_status()
    j = r.json()
    rows = []
    for rec in j.get("results", []):
        for v in rec.get("values", []):
            y = pd.to_numeric(v.get("year"), errors="coerce")
            val = pd.to_numeric(v.get("val"), errors="coerce")
            if pd.notna(y) and pd.notna(val):
                rows.append({"geo": "PL", "year": int(y), "value": float(val)})
    return pd.DataFrame(rows)


def pl_bdl_annual_netmig_rate(pop_annual: pd.DataFrame) -> pd.DataFrame:
    try:
        imm_a = bdl_poland_series(PL_BDL_IMM_ANNUAL_VAR).rename(columns={"value": "pl_bdl_immigration_a"})
        emi_a = bdl_poland_series(PL_BDL_EMI_ANNUAL_VAR).rename(columns={"value": "pl_bdl_emigration_a"})
        imm_h1 = bdl_poland_series(PL_BDL_IMM_H1_VAR).rename(columns={"value": "pl_bdl_immigration_h1"})
        emi_h1 = bdl_poland_series(PL_BDL_EMI_H1_VAR).rename(columns={"value": "pl_bdl_emigration_h1"})
    except Exception:
        return pd.DataFrame(columns=["geo", "year", "pl_bdl_net_migration_rate_a", "pl_bdl_net_migration_rate_source"])

    if imm_a.empty and emi_a.empty and imm_h1.empty and emi_h1.empty:
        return pd.DataFrame(columns=["geo", "year", "pl_bdl_net_migration_rate_a", "pl_bdl_net_migration_rate_source"])

    annual = imm_a.merge(emi_a, on=["geo", "year"], how="outer")
    annual["pl_bdl_net_migration_a"] = pd.to_numeric(annual["pl_bdl_immigration_a"], errors="coerce") - pd.to_numeric(
        annual["pl_bdl_emigration_a"], errors="coerce"
    )
    h1 = imm_h1.merge(emi_h1, on=["geo", "year"], how="outer")
    h1["pl_bdl_net_migration_h1"] = pd.to_numeric(h1["pl_bdl_immigration_h1"], errors="coerce") - pd.to_numeric(
        h1["pl_bdl_emigration_h1"], errors="coerce"
    )

    d = annual.merge(h1[["geo", "year", "pl_bdl_net_migration_h1"]], on=["geo", "year"], how="outer")
    d = d.sort_values(["geo", "year"]).reset_index(drop=True)
    d["pl_bdl_net_migration_est_a"] = np.nan
    d["pl_bdl_net_migration_rate_source"] = np.where(d["pl_bdl_net_migration_a"].notna(), "pl_bdl_annual", None)

    for i in range(len(d)):
        if pd.notna(d.at[i, "pl_bdl_net_migration_a"]):
            continue
        h1_now = pd.to_numeric(d.at[i, "pl_bdl_net_migration_h1"], errors="coerce")
        if pd.isna(h1_now):
            continue
        prev = d.loc[: i - 1].copy()
        prev = prev.dropna(subset=["pl_bdl_net_migration_a", "pl_bdl_net_migration_h1"])
        if not prev.empty:
            p = prev.iloc[-1]
            denom = pd.to_numeric(p["pl_bdl_net_migration_h1"], errors="coerce")
            numer = pd.to_numeric(p["pl_bdl_net_migration_a"], errors="coerce")
            if pd.notna(denom) and denom != 0 and pd.notna(numer):
                ratio = float(numer / denom)
                d.at[i, "pl_bdl_net_migration_est_a"] = float(h1_now * ratio)
                d.at[i, "pl_bdl_net_migration_rate_source"] = "pl_bdl_h1_nowcast_ratio"
                continue
        d.at[i, "pl_bdl_net_migration_est_a"] = float(h1_now * 2.0)
        d.at[i, "pl_bdl_net_migration_rate_source"] = "pl_bdl_h1_nowcast_x2"

    d["pl_bdl_net_migration_used_a"] = pd.to_numeric(d["pl_bdl_net_migration_a"], errors="coerce").combine_first(
        pd.to_numeric(d["pl_bdl_net_migration_est_a"], errors="coerce")
    )
    d = d.merge(pop_annual[["geo", "year", "population"]], on=["geo", "year"], how="left")

    wb_pop = fetch_wb(WB_POP_IND, WB_POP_FILE).copy()
    wb_pop = wb_pop[wb_pop["country_code_wb2"] == "PL"].copy()
    wb_pop["geo"] = "PL"
    wb_pop = wb_pop.rename(columns={"value": "wb_population"})
    d = d.merge(wb_pop[["geo", "year", "wb_population"]], on=["geo", "year"], how="left")

    pop = (
        pd.to_numeric(d["population"], errors="coerce")
        if "population" in d.columns
        else pd.Series(np.nan, index=d.index, dtype=float)
    )
    pop_wb = pd.to_numeric(d.get("wb_population"), errors="coerce")
    pop_use = pop.combine_first(pop_wb)
    pop_use = pd.Series(pop_use, index=d.index).sort_index().ffill()
    d["pl_bdl_net_migration_rate_a"] = np.where(
        pop_use > 0,
        pd.to_numeric(d["pl_bdl_net_migration_used_a"], errors="coerce") / pop_use * 1000.0,
        np.nan,
    )
    d = d[d["pl_bdl_net_migration_rate_a"].notna()].copy()
    return d[["geo", "year", "pl_bdl_net_migration_rate_a", "pl_bdl_net_migration_rate_source"]]


def build_harmonized_panels() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    annual = pd.read_parquet(PROC_DIR / "panel_annual.parquet").copy()
    quarterly = pd.read_parquet(PROC_DIR / "panel_quarterly.parquet").copy()
    extra_a, extra_q, added_geos = build_extra_country_rows(annual, quarterly)
    if not extra_a.empty:
        annual = pd.concat([annual, extra_a], ignore_index=True).drop_duplicates(["geo", "year"], keep="first")
    if not extra_q.empty:
        quarterly = pd.concat([quarterly, extra_q], ignore_index=True).drop_duplicates(["geo", "year", "quarter"], keep="first")
    annual["country_name"] = annual["geo"].map(build_country_name_lookup())
    quarterly["country_name"] = quarterly["geo"].map(build_country_name_lookup())
    base_counts = (
        annual.groupby("geo", as_index=False)
        .agg(
            base_hpi_nonmissing=("hpi_growth", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            base_netmig_nonmissing=("net_migration_rate", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            base_gdp_nonmissing=("gdp_pc_growth", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            base_unemp_nonmissing=("unemployment_rate", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            base_long_nonmissing=("long_rate", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        )
        .sort_values("geo")
    )

    # Aggregate quarterly -> annual to align frequency definitions.
    qa = annualize_quarterly(quarterly)
    # Keep union of annual and quarterly-derived annualized rows so countries with
    # quarterly RHPI coverage (but sparse annual RHPI) can still enter 2025.
    annual = annual.merge(qa, on=["geo", "year"], how="outer")

    # Harmonized annual outcome and air variables:
    # prefer annual definitions, fallback to annualized quarterly means.
    annual["hpi_growth_harmonized_source"] = "missing"
    annual["hpi_growth_harmonized"] = annual["hpi_growth"].combine_first(annual["q_hpi_yoy_mean"])
    annual.loc[annual["hpi_growth"].notna(), "hpi_growth_harmonized_source"] = "panel_base_annual"
    annual.loc[annual["hpi_growth"].isna() & annual["q_hpi_yoy_mean"].notna(), "hpi_growth_harmonized_source"] = (
        "panel_base_quarterly_annualized"
    )
    oecd_hpi_a = build_oecd_country_hpi_annual()
    annual = annual.merge(oecd_hpi_a, on=["geo", "year"], how="left")
    hpi_oecd_fill = annual["hpi_growth_harmonized"].isna() & annual["oecd_hpi_growth_a"].notna()
    annual.loc[hpi_oecd_fill, "hpi_growth_harmonized"] = annual.loc[hpi_oecd_fill, "oecd_hpi_growth_a"]
    annual.loc[hpi_oecd_fill, "hpi_growth_harmonized_source"] = "oecd_rhpi_country"

    annual["air_growth_harmonized"] = annual["air_growth"].combine_first(annual["q_air_yoy_mean"])

    # UK annual migration fallback from ONS quarterly sums.
    uk_ons = uk_ons_annual_netmig_rate(annual)
    annual = annual.merge(uk_ons, on=["geo", "year"], how="left")
    annual["net_migration_rate_harmonized"] = annual["net_migration_rate"].copy()
    uk_fill = (annual["geo"] == "UK") & annual["net_migration_rate_harmonized"].isna() & annual["uk_ons_net_migration_rate_a"].notna()
    annual.loc[uk_fill, "net_migration_rate_harmonized"] = annual.loc[uk_fill, "uk_ons_net_migration_rate_a"]
    if "net_migration_rate_source" in annual.columns:
        annual["net_migration_rate_harmonized_source"] = annual["net_migration_rate_source"].fillna("eurostat")
    else:
        annual["net_migration_rate_harmonized_source"] = "eurostat"
    if "uk_ons_net_migration_rate_source" in annual.columns:
        annual.loc[uk_fill, "net_migration_rate_harmonized_source"] = annual.loc[uk_fill, "uk_ons_net_migration_rate_source"].fillna(
            "uk_ons_q_aggregated"
        )
    else:
        annual.loc[uk_fill, "net_migration_rate_harmonized_source"] = "uk_ons_q_aggregated"
    annual["net_migration_rate_harmonized_source"] = annual["net_migration_rate_harmonized_source"].fillna("eurostat")

    # Poland annual migration fallback from official BDL (annual + H1 nowcast when annual not released yet).
    pl_bdl = pl_bdl_annual_netmig_rate(annual)
    annual = annual.merge(pl_bdl, on=["geo", "year"], how="left")
    pl_fill = (
        (annual["geo"] == "PL")
        & annual["net_migration_rate_harmonized"].isna()
        & annual["pl_bdl_net_migration_rate_a"].notna()
    )
    annual.loc[pl_fill, "net_migration_rate_harmonized"] = annual.loc[pl_fill, "pl_bdl_net_migration_rate_a"]
    annual.loc[pl_fill, "net_migration_rate_harmonized_source"] = annual.loc[pl_fill, "pl_bdl_net_migration_rate_source"].fillna(
        "pl_bdl_annual"
    )

    # WB fallback for missing net migration rates (level -> per 1,000 via population).
    wb_n = fetch_wb(WB_NETMIG_IND, WB_NETMIG_FILE).copy()
    wb_n["geo"] = wb_n["country_code_wb2"].map(WB2_TO_EU)
    wb_n = wb_n.dropna(subset=["geo", "year"]).rename(columns={"value": "wb_net_migration_level"})
    annual = annual.merge(wb_n[["geo", "year", "wb_net_migration_level"]], on=["geo", "year"], how="left")
    annual["wb_net_migration_rate_per_1000"] = np.where(
        pd.to_numeric(annual["population"], errors="coerce") > 0,
        pd.to_numeric(annual["wb_net_migration_level"], errors="coerce") / pd.to_numeric(annual["population"], errors="coerce") * 1000.0,
        np.nan,
    )
    wb_net_fill = annual["net_migration_rate_harmonized"].isna() & annual["wb_net_migration_rate_per_1000"].notna()
    annual.loc[wb_net_fill, "net_migration_rate_harmonized"] = annual.loc[wb_net_fill, "wb_net_migration_rate_per_1000"]
    annual.loc[wb_net_fill, "net_migration_rate_harmonized_source"] = "wb_net_migration"

    # Macro fallbacks from WB (for UK and others with sparse annual macro series).
    wb_u = fetch_wb(WB_UEM_IND, WB_UEM_FILE)
    wb_g = fetch_wb(WB_GDPG_IND, WB_GDPG_FILE)
    wb_l = fetch_wb(WB_LENDRATE_IND, WB_LENDRATE_FILE)
    wb_i = fetch_wb(WB_INFL_IND, WB_INFL_FILE)
    wb_p = fetch_wb(WB_POPG_IND, WB_POPG_FILE)
    for wb, col, src, base in [
        (wb_u, "unemployment_rate_harmonized", "wb_unemployment", "unemployment_rate"),
        (wb_g, "gdp_pc_growth_harmonized", "wb_gdp_pc_growth", "gdp_pc_growth"),
        (wb_l, "long_rate_harmonized", "wb_lending_rate", "long_rate"),
        (wb_i, "inflation_hicp_harmonized", "wb_inflation_cpi", "inflation_hicp"),
        (wb_p, "pop_growth_harmonized", "wb_population_growth", "pop_growth"),
    ]:
        x = wb.copy()
        x["geo"] = x["country_code_wb2"].map(WB2_TO_EU)
        x = x.dropna(subset=["geo", "year"])
        x = x.rename(columns={"value": f"{col}_wb"})
        annual = annual.merge(x[["geo", "year", f"{col}_wb"]], on=["geo", "year"], how="left")
        annual[col] = annual[base].combine_first(annual[f"{col}_wb"])
        src_col = f"{col}_source"
        annual[src_col] = pd.Series(index=annual.index, dtype="object")
        annual.loc[annual[base].notna(), src_col] = "panel_base"
        annual.loc[annual[base].isna() & annual[f"{col}_wb"].notna(), src_col] = src

    # Carry-forward to 2025 when official annual vintages lag.
    # This is explicit in source tags so users can audit which values are nowcasts.
    annual = annual.sort_values(["geo", "year"]).reset_index(drop=True)
    g = annual.groupby("geo", sort=False)
    annual["population"] = g["population"].ffill().bfill()
    carry_targets = [
        ("net_migration_rate_harmonized", "net_migration_rate_harmonized_source"),
        ("gdp_pc_growth_harmonized", "gdp_pc_growth_harmonized_source"),
        ("unemployment_rate_harmonized", "unemployment_rate_harmonized_source"),
        ("inflation_hicp_harmonized", "inflation_hicp_harmonized_source"),
        ("long_rate_harmonized", "long_rate_harmonized_source"),
        ("pop_growth_harmonized", "pop_growth_harmonized_source"),
        ("air_growth_harmonized", None),
    ]
    for val_col, src_col in carry_targets:
        if val_col not in annual.columns:
            continue
        ff = g[val_col].ffill()
        mask = (annual["year"] == 2025) & annual[val_col].isna() & ff.notna()
        annual.loc[mask, val_col] = ff.loc[mask]
        if src_col is not None:
            if src_col not in annual.columns:
                annual[src_col] = np.nan
            annual.loc[mask, src_col] = "carry_forward_latest"

    # Ensure 2025 annual HPI exists where any historical HPI exists.
    hpi_ff = g["hpi_growth_harmonized"].ffill()
    hpi_cf_mask = (annual["year"] == 2025) & annual["hpi_growth_harmonized"].isna() & hpi_ff.notna()
    annual.loc[hpi_cf_mask, "hpi_growth_harmonized"] = hpi_ff.loc[hpi_cf_mask]
    annual.loc[hpi_cf_mask, "hpi_growth_harmonized_source"] = "carry_forward_latest"

    # Quarterly harmonized panel: attach annual harmonized controls to each quarter.
    ann_ctrl = annual[
        [
            "geo",
            "year",
            "country_name",
            "hpi_growth_harmonized",
            "air_growth_harmonized",
            "net_migration_rate_harmonized",
            "gdp_pc_growth_harmonized",
            "unemployment_rate_harmonized",
            "inflation_hicp_harmonized",
            "long_rate_harmonized",
            "pop_growth_harmonized",
        ]
    ].drop_duplicates(["geo", "year"])
    qh = quarterly.merge(ann_ctrl, on=["geo", "year"], how="left")

    # Within-year lag for quarterly models can still use quarterly air/hpi;
    # annual controls are aligned by same-year values and one-year lag.
    qh = qh.sort_values(["geo", "year", "quarter"]).reset_index(drop=True)

    # OECD quarterly RHPI growth fallback for missing quarterly HPI YoY.
    qh["hpi_yoy_harmonized"] = qh["hpi_yoy"]
    qh["hpi_yoy_harmonized_source"] = np.where(qh["hpi_yoy"].notna(), "panel_base", "missing")
    oecd_hpi_q = build_oecd_country_hpi_quarterly()
    qh = qh.merge(oecd_hpi_q, on=["geo", "period_str"], how="left")
    q_hpi_oecd_fill = qh["hpi_yoy_harmonized"].isna() & qh["oecd_hpi_yoy_q"].notna()
    qh.loc[q_hpi_oecd_fill, "hpi_yoy_harmonized"] = qh.loc[q_hpi_oecd_fill, "oecd_hpi_yoy_q"]
    qh.loc[q_hpi_oecd_fill, "hpi_yoy_harmonized_source"] = "oecd_rhpi_country_q"

    gq = qh.groupby("geo", sort=False)
    for c in [
        "net_migration_rate_harmonized",
        "gdp_pc_growth_harmonized",
        "unemployment_rate_harmonized",
        "inflation_hicp_harmonized",
        "long_rate_harmonized",
        "pop_growth_harmonized",
    ]:
        qh[f"L1y_{c}"] = gq[c].shift(4)

    annual = annual.sort_values(["geo", "year"]).reset_index(drop=True)
    ga = annual.groupby("geo", sort=False)
    for c in ["hpi_growth_harmonized", "air_growth_harmonized", "net_migration_rate_harmonized"]:
        annual[f"L1_{c}"] = ga[c].shift(1)

    # Country-level patch diagnostics for transparency.
    patch_rows = []
    for geo, g in annual.groupby("geo", sort=True):
        patch_rows.append(
            {
                "geo": geo,
                "annual_rows": int(len(g)),
                "hpi_filled_oecd": int((g["hpi_growth_harmonized_source"] == "oecd_rhpi_country").sum()),
                "hpi_filled_q_annualized": int((g["hpi_growth_harmonized_source"] == "panel_base_quarterly_annualized").sum()),
                "netmig_harmonized_nonmissing": int(g["net_migration_rate_harmonized"].notna().sum()),
                "netmig_filled_ons": int((g["net_migration_rate_harmonized_source"] == "uk_ons_q_aggregated").sum()),
                "netmig_filled_wb": int((g["net_migration_rate_harmonized_source"] == "wb_net_migration").sum()),
                "gdp_filled_wb": int((g["gdp_pc_growth_harmonized_source"] == "wb_gdp_pc_growth").sum()) if "gdp_pc_growth_harmonized_source" in g else 0,
                "unemp_filled_wb": int((g["unemployment_rate_harmonized_source"] == "wb_unemployment").sum()) if "unemployment_rate_harmonized_source" in g else 0,
                "long_filled_wb": int((g["long_rate_harmonized_source"] == "wb_lending_rate").sum()) if "long_rate_harmonized_source" in g else 0,
                "infl_filled_wb": int((g["inflation_hicp_harmonized_source"] == "wb_inflation_cpi").sum()) if "inflation_hicp_harmonized_source" in g else 0,
                "popg_filled_wb": int((g["pop_growth_harmonized_source"] == "wb_population_growth").sum()) if "pop_growth_harmonized_source" in g else 0,
            }
        )
    patch_df = pd.DataFrame(patch_rows).sort_values("geo").reset_index(drop=True)
    patch_df.to_csv(RESULTS_DIR / "country_blended_patch_coverage.csv", index=False)

    # Country data description table (estimation-ready windows + source recency).
    annual_core_cols = [
        "hpi_growth_harmonized",
        "net_migration_rate_harmonized",
        "air_growth_harmonized",
        "gdp_pc_growth_harmonized",
        "unemployment_rate_harmonized",
        "inflation_hicp_harmonized",
        "long_rate_harmonized",
        "pop_growth_harmonized",
    ]
    quarterly_core_cols = ["hpi_yoy_harmonized", "air_yoy"]
    annual["annual_fe_complete"] = annual[annual_core_cols].apply(lambda r: bool(pd.notna(r).all()), axis=1)
    qh["quarterly_fe_complete"] = qh[quarterly_core_cols].apply(lambda r: bool(pd.notna(r).all()), axis=1)

    a_bounds = (
        annual.groupby("geo", as_index=False)
        .agg(
            country_name=("country_name", "first"),
            a_year_min=("year", "min"),
            a_year_max=("year", "max"),
            a_rows=("year", "size"),
            a_hpi_nonmissing=("hpi_growth_harmonized", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            a_netmig_nonmissing=("net_migration_rate_harmonized", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            a_gdp_nonmissing=("gdp_pc_growth_harmonized", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            a_unemp_nonmissing=("unemployment_rate_harmonized", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            a_fe_rows=("annual_fe_complete", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).astype(int).sum())),
        )
        .sort_values("geo")
    )

    a_core = (
        annual[annual["annual_fe_complete"]]
        .groupby("geo", as_index=False)
        .agg(
            a_fe_year_min=("year", "min"),
            a_fe_year_max=("year", "max"),
        )
    )
    a_bounds = a_bounds.merge(a_core, on="geo", how="left")

    q_bounds = (
        qh.groupby("geo", as_index=False)
        .agg(
            q_period_min=("period_str", "min"),
            q_period_max=("period_str", "max"),
            q_rows=("period_str", "size"),
            q_hpi_yoy_nonmissing=("hpi_yoy_harmonized", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            q_air_yoy_nonmissing=("air_yoy", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            q_fe_rows=("quarterly_fe_complete", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).astype(int).sum())),
        )
        .sort_values("geo")
    )
    q_core = (
        qh[qh["quarterly_fe_complete"]]
        .groupby("geo", as_index=False)
        .agg(
            q_fe_period_min=("period_str", "first"),
            q_fe_period_max=("period_str", "last"),
        )
    )
    q_bounds = q_bounds.merge(q_core, on="geo", how="left")

    netmig_latest = (
        annual[annual["net_migration_rate_harmonized"].notna()]
        .sort_values(["geo", "year"])
        .groupby("geo", as_index=False)
        .tail(1)[["geo", "year", "net_migration_rate_harmonized_source"]]
        .rename(
            columns={
                "year": "netmig_latest_year",
                "net_migration_rate_harmonized_source": "netmig_latest_source",
            }
        )
    )
    hpi_latest = (
        qh[qh["hpi_yoy_harmonized"].notna()]
        .sort_values(["geo", "year", "quarter"])
        .groupby("geo", as_index=False)
        .tail(1)[["geo", "period_str", "hpi_yoy_harmonized_source"]]
        .rename(
            columns={
                "period_str": "hpiq_latest_period",
                "hpi_yoy_harmonized_source": "hpiq_latest_source",
            }
        )
    )
    desc = (
        a_bounds.merge(q_bounds, on="geo", how="outer")
        .merge(netmig_latest, on="geo", how="left")
        .merge(hpi_latest, on="geo", how="left")
        .sort_values("geo")
        .reset_index(drop=True)
    )
    desc.to_csv(RESULTS_DIR / "country_data_description_harmonized.csv", index=False)

    after_counts = (
        annual.groupby("geo", as_index=False)
        .agg(
            harmonized_hpi_nonmissing=("hpi_growth_harmonized", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            harmonized_netmig_nonmissing=("net_migration_rate_harmonized", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            harmonized_gdp_nonmissing=("gdp_pc_growth_harmonized", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            harmonized_unemp_nonmissing=("unemployment_rate_harmonized", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            harmonized_long_nonmissing=("long_rate_harmonized", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        )
        .sort_values("geo")
    )
    improvement = base_counts.merge(after_counts, on="geo", how="outer").merge(
        annual[["geo", "country_name"]].drop_duplicates("geo"), on="geo", how="left"
    )
    for v in ["hpi", "netmig", "gdp", "unemp", "long"]:
        improvement[f"delta_{v}_nonmissing"] = improvement[f"harmonized_{v}_nonmissing"] - improvement[f"base_{v}_nonmissing"]
    improvement = improvement[
        [
            "geo",
            "country_name",
            "base_hpi_nonmissing",
            "harmonized_hpi_nonmissing",
            "delta_hpi_nonmissing",
            "base_netmig_nonmissing",
            "harmonized_netmig_nonmissing",
            "delta_netmig_nonmissing",
            "base_gdp_nonmissing",
            "harmonized_gdp_nonmissing",
            "delta_gdp_nonmissing",
            "base_unemp_nonmissing",
            "harmonized_unemp_nonmissing",
            "delta_unemp_nonmissing",
            "base_long_nonmissing",
            "harmonized_long_nonmissing",
            "delta_long_nonmissing",
        ]
    ].sort_values("geo")
    improvement.to_csv(RESULTS_DIR / "country_harmonization_improvement_summary.csv", index=False)

    meta = {
        "annual_rows": int(len(annual)),
        "quarterly_rows": int(len(qh)),
        "annual_countries": int(annual["geo"].nunique()),
        "quarterly_countries": int(qh["geo"].nunique()),
        "supplemental_countries_added_count": int(len(added_geos)),
        "supplemental_countries_added": added_geos,
        "countries_with_hpi_fill_oecd": int((patch_df["hpi_filled_oecd"] > 0).sum()),
        "countries_with_hpi_gain_vs_base": int((improvement["delta_hpi_nonmissing"] > 0).sum()),
        "uk_netmig_filled_from_ons": int(uk_fill.sum()),
        "pl_netmig_filled_from_bdl": int(pl_fill.sum()),
        "countries_with_any_wb_netmig_fill": int((patch_df["netmig_filled_wb"] > 0).sum()),
        "countries_with_any_wb_unemp_fill": int((patch_df["unemp_filled_wb"] > 0).sum()),
        "countries_with_any_wb_gdp_fill": int((patch_df["gdp_filled_wb"] > 0).sum()),
        "countries_with_any_wb_long_fill": int((patch_df["long_filled_wb"] > 0).sum()),
        "countries_with_any_wb_infl_fill": int((patch_df["infl_filled_wb"] > 0).sum()),
        "countries_with_any_wb_popg_fill": int((patch_df["popg_filled_wb"] > 0).sum()),
        "uk_annual_netmig_nonmissing_after": int(annual.loc[annual["geo"] == "UK", "net_migration_rate_harmonized"].notna().sum()),
        "pl_annual_netmig_nonmissing_after": int(annual.loc[annual["geo"] == "PL", "net_migration_rate_harmonized"].notna().sum()),
        "uk_unemployment_nonmissing_after": int(annual.loc[annual["geo"] == "UK", "unemployment_rate_harmonized"].notna().sum()),
        "uk_gdppc_nonmissing_after": int(annual.loc[annual["geo"] == "UK", "gdp_pc_growth_harmonized"].notna().sum()),
    }
    return annual, qh, meta


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    annual_h, quarterly_h, meta = build_harmonized_panels()

    annual_h.to_parquet(PROC_DIR / "panel_annual_harmonized.parquet", index=False)
    annual_h.to_csv(PROC_DIR / "panel_annual_harmonized.csv", index=False)
    quarterly_h.to_parquet(PROC_DIR / "panel_quarterly_harmonized.parquet", index=False)
    quarterly_h.to_csv(PROC_DIR / "panel_quarterly_harmonized.csv", index=False)

    # Quick diagnostics table for UK/PL coverage before publication use.
    keep = ["geo", "year", "hpi_growth_harmonized", "net_migration_rate_harmonized", "gdp_pc_growth_harmonized", "unemployment_rate_harmonized"]
    diag = annual_h[annual_h["geo"].isin(["UK", "PL"])][keep].copy()
    diag.to_csv(RESULTS_DIR / "harmonized_uk_pl_annual_coverage.csv", index=False)

    # Write a paper-ready country data description table.
    desc = pd.read_csv(RESULTS_DIR / "country_data_description_harmonized.csv")
    tex_lines = [
        r"\begin{landscape}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{longtable}{lp{3.0cm}p{2.4cm}rp{2.8cm}rrr}",
        r"\caption{Country-level data description: harmonized annual and quarterly coverage}\label{tab:country_data_description}\\",
        r"\toprule",
        r"Geo & Country & Annual FE window & Annual FE rows & Quarterly FE window & Quarterly FE rows & Mig last year & HPI Q last \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{8}{l}{\textit{Table \ref{tab:country_data_description} continued}}\\",
        r"\toprule",
        r"Geo & Country & Annual FE window & Annual FE rows & Quarterly FE window & Quarterly FE rows & Mig last year & HPI Q last \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{8}{r}{\textit{Continued on next page}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\multicolumn{8}{p{0.96\linewidth}}{\footnotesize Notes: Annual and quarterly FE windows report complete-case coverage for the baseline FE regressions (dependent variable plus controls). ``Mig last year'' and ``HPI Q last'' report the most recent annual migration and quarterly HPI observations used by the harmonized pipeline.}\\",
        r"\endlastfoot",
    ]
    for _, r in desc.iterrows():
        a_min = int(r["a_fe_year_min"]) if pd.notna(r.get("a_fe_year_min")) else None
        a_max = int(r["a_fe_year_max"]) if pd.notna(r.get("a_fe_year_max")) else None
        a_window = f"{a_min}--{a_max}" if a_min is not None and a_max is not None else ""
        q_min = str(r.get("q_fe_period_min")) if pd.notna(r.get("q_fe_period_min")) else ""
        q_max = str(r.get("q_fe_period_max")) if pd.notna(r.get("q_fe_period_max")) else ""
        q_window = f"{q_min}--{q_max}" if q_min and q_max else ""
        tex_lines.append(
            f"{r['geo']} & {str(r['country_name']) if pd.notna(r['country_name']) else ''} & "
            f"{a_window} & "
            f"{int(r['a_fe_rows']) if pd.notna(r.get('a_fe_rows')) else ''} & "
            f"{q_window} & "
            f"{int(r['q_fe_rows']) if pd.notna(r.get('q_fe_rows')) else ''} & "
            f"{int(r['netmig_latest_year']) if pd.notna(r.get('netmig_latest_year')) else ''} & "
            f"{str(r.get('hpiq_latest_period')) if pd.notna(r.get('hpiq_latest_period')) else ''} \\\\"
        )
    tex_lines += [
        r"\end{longtable}",
        r"\end{landscape}",
        "",
    ]
    (ROOT / "paper_overleaf" / "tables" / "tab_country_data_description.tex").write_text("\n".join(tex_lines))

    (META_DIR / "harmonized_frequency_summary.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"[ok] wrote {PROC_DIR / 'panel_annual_harmonized.parquet'}")
    print(f"[ok] wrote {PROC_DIR / 'panel_quarterly_harmonized.parquet'}")
    print(f"[ok] wrote {RESULTS_DIR / 'country_data_description_harmonized.csv'}")
    print(f"[ok] wrote {RESULTS_DIR / 'country_harmonization_improvement_summary.csv'}")
    print(f"[ok] wrote {ROOT / 'paper_overleaf' / 'tables' / 'tab_country_data_description.tex'}")


if __name__ == "__main__":
    main()
