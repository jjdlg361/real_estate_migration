#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import requests
import eurostat
from matplotlib.patches import Patch
from linearmodels.panel import PanelOLS

# Reuse Eurostat OD parsing and WB code mapping already used in the IV pipeline.
from build_shiftshare_iv import COUNTRY2_RE, WB2_TO_EUROSTAT, load_od_migration, eurostat_wide_to_long


warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"linearmodels(\..*)?")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
META_DIR = ROOT / "data" / "metadata"

WB_GDP_LEVEL_FILE = RAW_DIR / "worldbank_origin_gdp_pc_ppp_const_long.csv"
WB_GDP_LEVEL_INDICATOR = "NY.GDP.PCAP.PP.KD"  # GDP per capita, PPP (constant intl $)
CASE_CTZ_PROXY_RAW = RAW_DIR / "migr_imm1ctz_case_proxy.csv"
UK_ONS_LTIM_YEJUN_RAW = RAW_DIR / "ons_ltim_ye_jun_2025.xlsx"
UK_ONS_LTIM_YEJUN_URL = (
    "https://www.ons.gov.uk/file?uri=%2Fpeoplepopulationandcommunity%2Fpopulationandmigration%2F"
    "internationalmigration%2Fdatasets%2Flongterminternationalimmigrationemigrationandnetmigrationflowsprovisional%2F"
    "yearendingjune2025%2Fltimnov25.xlsx"
)
OD_BLEND_PARQUET = PROC_DIR / "od_migration_blended_for_composition.parquet"
OD_BLEND_CSV = PROC_DIR / "od_migration_blended_for_composition.csv"

PAPER_DIR = ROOT / "paper_overleaf"
PAPER_TABLES_DIR = PAPER_DIR / "tables"
PAPER_FIGS_DIR = PAPER_DIR / "figures"

EU_EEA_CH_UK = {
    "AT","BE","BG","CY","CZ","DE","DK","EE","EL","ES","FI","FR","HR","HU","IE","IT","LT","LU","LV","MT","NL","PL","PT","RO","SE","SI","SK",
    "IS","NO","LI","CH","UK",
}
LATAM_CARIB = {
    "AR","BO","BR","CL","CO","CR","CU","DO","EC","SV","GT","HN","HT","JM","MX","NI","PA","PY","PE","PR","TT","UY","VE",
    "BS","BB","BZ","GY","SR","AW","CW","GD","LC","VC","AG","DM","KN","MQ","GP","GF",
}
MENA = {
    "DZ","MA","TN","LY","EG","SD","EH",
    "BH","IR","IQ","IL","JO","KW","LB","OM","PS","QA","SA","SY","AE","YE",
}
SSA = {
    "AO","BJ","BW","BF","BI","CM","CV","CF","TD","KM","CD","CG","CI","DJ","GQ","ER","SZ","ET","GA","GM","GH","GN","GW","KE",
    "LS","LR","MG","MW","ML","MR","MU","MZ","NA","NE","NG","RW","ST","SN","SC","SL","SO","ZA","SS","TZ","TG","UG","ZM","ZW",
}
SOUTH_ASIA = {"AF","BD","BT","IN","LK","MV","NP","PK"}
EAST_SE_ASIA = {
    "BN","KH","CN","HK","ID","JP","KP","KR","LA","MO","MN","MY","MM","PH","SG","TH","TL","TW","VN"
}
CENTRAL_ASIA_CAUCASUS = {"AM","AZ","GE","KZ","KG","TJ","TM","UZ"}
NON_EUROPE_EUR_CIS = {
    "AL","BA","BY","MD","ME","MK","RS","RU","TR","UA","XK",  # Balkan/CIS/non-EU Europe
} | CENTRAL_ASIA_CAUCASUS
N_AMERICA = {"US","CA"}
OCEANIA = {"AU","NZ","FJ","PG","WS","TO","VU","NC","PF"}

ORIGIN_GROUP_ORDER = [
    "EU_EEA_CH_UK",
    "LATAM_CARIB",
    "MENA",
    "SSA",
    "SOUTH_ASIA",
    "EAST_SE_ASIA",
    "NON_EUROPE_EUR_CIS",
    "N_AMERICA",
    "OCEANIA",
    "OTHER",
]

CASE_DESTS = ["ES", "UK"]
# Shared country-case focus set (Spain + UK): include major Spain LATAM/North Africa
# origins and major UK global inflow origins.
CASE_SELECTED_ORIGINS = [
    "CO", "MA", "VE", "AR", "PE", "UK",
    "IN", "CN", "RO", "PK", "PL", "NG", "US", "FR",
]
CASE_RANK_WINDOWS = {
    "ES": (2018, 2024),
    "UK": (2018, 2024),
}
# Apply year-level OD fallback where OD is zero/missing and proxy is available.
OD_PROXY_BLEND_GEOS = {"UK", "PT"}


def origin_group(code: str) -> str:
    c = str(code)
    if c in EU_EEA_CH_UK:
        return "EU_EEA_CH_UK"
    if c in LATAM_CARIB:
        return "LATAM_CARIB"
    if c in MENA:
        return "MENA"
    if c in SSA:
        return "SSA"
    if c in SOUTH_ASIA:
        return "SOUTH_ASIA"
    if c in EAST_SE_ASIA:
        return "EAST_SE_ASIA"
    if c in NON_EUROPE_EUR_CIS:
        return "NON_EUROPE_EUR_CIS"
    if c in N_AMERICA:
        return "N_AMERICA"
    if c in OCEANIA:
        return "OCEANIA"
    return "OTHER"


def winsorize_series(s: pd.Series, p_low: float = 0.01, p_high: float = 0.99) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    lo, hi = x.quantile([p_low, p_high])
    return x.clip(lo, hi)


def load_parquet_or_csv(base_name: str) -> pd.DataFrame:
    pq = PROC_DIR / f"{base_name}.parquet"
    csv = PROC_DIR / f"{base_name}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_csv(csv)


def load_harmonized_annual_base() -> pd.DataFrame:
    """
    Harmonized-first annual panel used as baseline for composition models.
    Falls back to legacy annual IV fields when harmonized columns are absent.
    """
    h = load_parquet_or_csv("panel_annual_harmonized").replace([np.inf, -np.inf], np.nan).copy()
    iv = load_parquet_or_csv("panel_annual_iv").replace([np.inf, -np.inf], np.nan).copy()

    key = ["geo", "year"]
    # Bring legacy-only fields for compatibility.
    iv_keep = [
        "geo",
        "year",
        "immigration",
        "emigration",
        "immigration_rate_per_1000",
        "emigration_rate_per_1000",
        "net_migration_level",
        "net_migration_level_per_1000",
        "hpi_index",
        "air_passengers",
        "population",
    ]
    iv_keep = [c for c in iv_keep if c in iv.columns]
    if iv_keep:
        h = h.merge(iv[iv_keep].drop_duplicates(key), on=key, how="left", suffixes=("", "_iv"))

    # Harmonized-first canonical variables.
    h["hpi_growth"] = pd.to_numeric(h.get("hpi_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(h.get("hpi_growth"), errors="coerce")
    )
    h["net_migration_rate"] = pd.to_numeric(h.get("net_migration_rate_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(h.get("net_migration_rate"), errors="coerce")
    )
    h["air_growth"] = pd.to_numeric(h.get("air_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(h.get("air_growth"), errors="coerce")
    )
    h["gdp_pc_growth"] = pd.to_numeric(h.get("gdp_pc_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(h.get("gdp_pc_growth"), errors="coerce")
    )
    h["unemployment_rate"] = pd.to_numeric(h.get("unemployment_rate_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(h.get("unemployment_rate"), errors="coerce")
    )
    h["inflation_hicp"] = pd.to_numeric(h.get("inflation_hicp_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(h.get("inflation_hicp"), errors="coerce")
    )
    h["long_rate"] = pd.to_numeric(h.get("long_rate_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(h.get("long_rate"), errors="coerce")
    )
    h["pop_growth"] = pd.to_numeric(h.get("pop_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(h.get("pop_growth"), errors="coerce")
    )

    h = h.sort_values(["geo", "year"]).reset_index(drop=True)
    g = h.groupby("geo", sort=False)
    for c in [
        "hpi_growth",
        "net_migration_rate",
        "air_growth",
        "gdp_pc_growth",
        "unemployment_rate",
        "inflation_hicp",
        "long_rate",
        "pop_growth",
        "immigration_rate_per_1000",
        "emigration_rate_per_1000",
        "net_migration_level_per_1000",
    ]:
        if c in h.columns:
            h[f"L1_{c}"] = g[c].shift(1)
    return h


def _parse_case_citizenship_proxy_raw(df: pd.DataFrame, geos: list[str]) -> pd.DataFrame:
    long_df = eurostat_wide_to_long(df)
    keep = (
        (long_df["age"] == "TOTAL")
        & (long_df["agedef"] == "REACH")
        & (long_df["unit"] == "NR")
        & (long_df["sex"] == "T")
    )
    long_df = long_df.loc[keep, ["geo", "citizen", "year", "value"]].copy()
    long_df = long_df.rename(columns={"citizen": "origin", "value": "immigration_proxy"})
    long_df["geo"] = long_df["geo"].astype(str)
    long_df["origin"] = long_df["origin"].astype(str)
    long_df["immigration_proxy"] = pd.to_numeric(long_df["immigration_proxy"], errors="coerce")
    long_df = long_df[long_df["geo"].isin(set(geos))].copy()
    long_df = long_df[long_df["origin"].str.match(COUNTRY2_RE, na=False)].copy()
    long_df = long_df[long_df["origin"] != long_df["geo"]].copy()
    long_df = long_df.dropna(subset=["immigration_proxy", "year"]).copy()
    long_df = long_df[long_df["immigration_proxy"] >= 0].copy()
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype(int)
    long_df = (
        long_df.groupby(["geo", "origin", "year"], as_index=False)["immigration_proxy"]
        .sum(min_count=1)
        .sort_values(["geo", "origin", "year"])
        .reset_index(drop=True)
    )
    long_df["source"] = "ctz_proxy"
    return long_df


def load_citizenship_proxy(geos: list[str]) -> pd.DataFrame:
    """
    Eurostat citizenship-based OD proxy for selected destinations.
    """
    expected_geos = set(geos)
    geo_list = sorted(expected_geos)

    def _fetch_and_cache() -> pd.DataFrame:
        df = eurostat.get_data_df(
            "MIGR_IMM1CTZ",
            filter_pars={"age": "TOTAL", "agedef": "REACH", "sex": "T", "unit": "NR", "geo": geo_list},
        )
        CASE_CTZ_PROXY_RAW.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CASE_CTZ_PROXY_RAW, index=False)
        return df

    if CASE_CTZ_PROXY_RAW.exists():
        df = pd.read_csv(CASE_CTZ_PROXY_RAW)
    else:
        df = _fetch_and_cache()

    long_df = _parse_case_citizenship_proxy_raw(df, geo_list)
    have_geos = set(long_df["geo"].astype(str).unique())
    if not expected_geos.issubset(have_geos):
        # Refresh stale cache generated with a different geo selection.
        try:
            long_df = _parse_case_citizenship_proxy_raw(_fetch_and_cache(), geo_list)
        except Exception:
            pass
    return long_df


def load_case_citizenship_proxy() -> pd.DataFrame:
    """
    Eurostat fallback for country-case decomposition when OD previous-residence
    detail is unavailable for one of the selected case destinations.
    """
    return load_citizenship_proxy(CASE_DESTS)


def load_uk_ons_ye_june_immigration() -> pd.DataFrame:
    """
    UK ONS LTIM YE-June immigration totals by nationality block (All, British, EU+, Non-EU+).
    Used to bridge UK OD composition after 2019 when Eurostat OD flows are zero.
    """
    if not UK_ONS_LTIM_YEJUN_RAW.exists():
        try:
            r = requests.get(UK_ONS_LTIM_YEJUN_URL, timeout=120)
            r.raise_for_status()
            if not r.content.startswith(b"PK"):
                raise RuntimeError("ONS YE-June workbook response is not XLSX")
            UK_ONS_LTIM_YEJUN_RAW.parent.mkdir(parents=True, exist_ok=True)
            UK_ONS_LTIM_YEJUN_RAW.write_bytes(r.content)
        except Exception:
            return pd.DataFrame(columns=["year", "all_nat", "british", "eu_plus", "non_eu_plus"])

    try:
        raw = pd.read_excel(UK_ONS_LTIM_YEJUN_RAW, sheet_name="1", header=None, engine="openpyxl")
    except Exception:
        return pd.DataFrame(columns=["year", "all_nat", "british", "eu_plus", "non_eu_plus"])
    if raw.empty or len(raw) < 8:
        return pd.DataFrame(columns=["year", "all_nat", "british", "eu_plus", "non_eu_plus"])

    header = raw.iloc[5].astype(str).str.replace("\n", " ", regex=False).str.strip().tolist()
    d = raw.iloc[6:].copy()
    d.columns = header

    def _pick_col(pattern: str) -> str | None:
        for c in d.columns:
            if re.search(pattern, str(c), flags=re.IGNORECASE):
                return str(c)
        return None

    flow_col = _pick_col(r"^Flow")
    period_col = _pick_col(r"^Period")
    all_col = _pick_col(r"All Nationalities")
    brit_col = _pick_col(r"^British")
    eu_col = _pick_col(r"EU\+")
    non_eu_col = _pick_col(r"Non-EU\+")
    req = [flow_col, period_col, all_col, brit_col, eu_col, non_eu_col]
    if any(c is None for c in req):
        return pd.DataFrame(columns=["year", "all_nat", "british", "eu_plus", "non_eu_plus"])

    d = d[[flow_col, period_col, all_col, brit_col, eu_col, non_eu_col]].copy()
    d.columns = ["flow", "period", "all_nat", "british", "eu_plus", "non_eu_plus"]
    for c in ["all_nat", "british", "eu_plus", "non_eu_plus"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["flow"].astype(str).str.contains("Immigration", case=False, na=False)].copy()
    d = d[d["period"].astype(str).str.contains("YE Jun", case=False, na=False)].copy()
    if d.empty:
        return pd.DataFrame(columns=["year", "all_nat", "british", "eu_plus", "non_eu_plus"])
    d["year"] = d["period"].astype(str).str.extract(r"(\d{2})(?!.*\d)")[0]
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d = d.dropna(subset=["year"]).copy()
    d["year"] = (2000 + d["year"].astype(int)).astype(int)
    d = d[["year", "all_nat", "british", "eu_plus", "non_eu_plus"]].drop_duplicates("year", keep="last")
    d = d.sort_values("year").reset_index(drop=True)
    return d


def build_uk_ons_synthetic_od(od: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    For UK years with OD zero/missing and no citizenship-proxy coverage, synthesize
    origin-level inflows from official ONS EU+/Non-EU+ immigration totals.
    EU and non-EU mass are scaled to OD level using overlap-years ratio, then
    distributed across origins using recent pre-gap OD corridor shares.
    """
    uk = od[od["geo"] == "UK"].copy()
    if uk.empty:
        return pd.DataFrame(columns=["geo", "origin", "year", "immigration_od", "source"]), {"uk_ons_years_used": []}

    uk["immigration_od"] = pd.to_numeric(uk["immigration_od"], errors="coerce")
    od_tot = uk.groupby("year", as_index=False)["immigration_od"].sum(min_count=1).rename(columns={"immigration_od": "od_total"})
    last_pos_year = pd.to_numeric(od_tot.loc[od_tot["od_total"] > 0, "year"], errors="coerce").max()
    if pd.isna(last_pos_year):
        return pd.DataFrame(columns=["geo", "origin", "year", "immigration_od", "source"]), {"uk_ons_years_used": []}
    last_pos_year = int(last_pos_year)

    ons = load_uk_ons_ye_june_immigration()
    if ons.empty:
        return pd.DataFrame(columns=["geo", "origin", "year", "immigration_od", "source"]), {"uk_ons_years_used": []}
    ons["nonbrit"] = pd.to_numeric(ons["eu_plus"], errors="coerce") + pd.to_numeric(ons["non_eu_plus"], errors="coerce")

    overlap = od_tot.merge(ons[["year", "nonbrit"]], on="year", how="inner")
    overlap["ratio"] = np.where(
        pd.to_numeric(overlap["nonbrit"], errors="coerce") > 0,
        pd.to_numeric(overlap["od_total"], errors="coerce") / pd.to_numeric(overlap["nonbrit"], errors="coerce"),
        np.nan,
    )
    ratio = float(overlap["ratio"].replace([np.inf, -np.inf], np.nan).dropna().median()) if overlap["ratio"].notna().any() else np.nan
    if not np.isfinite(ratio):
        ratio = 0.60
    ratio = float(np.clip(ratio, 0.30, 1.20))

    base_start = max(1998, last_pos_year - 4)
    base = uk[(uk["year"] >= base_start) & (uk["year"] <= last_pos_year)].copy()
    base = base[pd.to_numeric(base["immigration_od"], errors="coerce") > 0].copy()
    if base.empty:
        return pd.DataFrame(columns=["geo", "origin", "year", "immigration_od", "source"]), {"uk_ons_years_used": []}

    base["is_eu"] = base["origin"].astype(str).isin(EU_EEA_CH_UK)
    eu_base = base[base["is_eu"]].groupby("origin", as_index=False)["immigration_od"].sum(min_count=1)
    non_base = base[~base["is_eu"]].groupby("origin", as_index=False)["immigration_od"].sum(min_count=1)
    eu_mass = pd.to_numeric(eu_base["immigration_od"], errors="coerce").sum()
    non_mass = pd.to_numeric(non_base["immigration_od"], errors="coerce").sum()
    if eu_mass <= 0 or non_mass <= 0:
        return pd.DataFrame(columns=["geo", "origin", "year", "immigration_od", "source"]), {"uk_ons_years_used": []}
    eu_base["share"] = eu_base["immigration_od"] / eu_mass
    non_base["share"] = non_base["immigration_od"] / non_mass

    target = ons[ons["year"] > last_pos_year].copy()
    if target.empty:
        return pd.DataFrame(columns=["geo", "origin", "year", "immigration_od", "source"]), {"uk_ons_years_used": []}
    target = target[pd.to_numeric(target["nonbrit"], errors="coerce") > 0].copy()
    if target.empty:
        return pd.DataFrame(columns=["geo", "origin", "year", "immigration_od", "source"]), {"uk_ons_years_used": []}

    rows: list[dict[str, Any]] = []
    for _, r in target.iterrows():
        y = int(r["year"])
        eu_level = float(max(0.0, ratio * pd.to_numeric(r["eu_plus"], errors="coerce")))
        non_level = float(max(0.0, ratio * pd.to_numeric(r["non_eu_plus"], errors="coerce")))
        for _, e in eu_base.iterrows():
            rows.append(
                {
                    "geo": "UK",
                    "origin": str(e["origin"]),
                    "year": y,
                    "immigration_od": float(eu_level * float(e["share"])),
                    "source": "ons_uk_group_scaled",
                }
            )
        for _, e in non_base.iterrows():
            rows.append(
                {
                    "geo": "UK",
                    "origin": str(e["origin"]),
                    "year": y,
                    "immigration_od": float(non_level * float(e["share"])),
                    "source": "ons_uk_group_scaled",
                }
            )
    out = pd.DataFrame(rows)
    out = out[out["immigration_od"] >= 0].copy()
    out = (
        out.groupby(["geo", "origin", "year", "source"], as_index=False)["immigration_od"]
        .sum(min_count=1)
        .sort_values(["geo", "origin", "year"])
        .reset_index(drop=True)
    )
    return out, {
        "uk_ons_years_used": sorted(out["year"].astype(int).unique().tolist()),
        "uk_ons_ratio_od_to_nonbrit": ratio,
        "uk_ons_base_share_window": [int(base_start), int(last_pos_year)],
    }


def blend_od_with_case_proxy(od: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Year-level blend for destinations with OD zero/missing years:
    keep OD by default; replace geo-year blocks with citizenship proxy only when
    OD aggregate is non-positive or absent.
    """
    out = od.copy()
    out["geo"] = out["geo"].astype(str)
    out["origin"] = out["origin"].astype(str)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype(int)
    out["immigration_od"] = pd.to_numeric(out["immigration_od"], errors="coerce")
    out = out.dropna(subset=["geo", "origin", "year", "immigration_od"]).copy()
    out = out[out["immigration_od"] >= 0].copy()
    out["source"] = "od_prev_residence"

    blend_geos = sorted(set(OD_PROXY_BLEND_GEOS))
    replace_keys = pd.DataFrame(columns=["geo", "year"])
    meta: dict[str, Any] = {
        "blend_geos": blend_geos,
        "geo_years_replaced": 0,
        "geo_years_replaced_detail": [],
    }

    if blend_geos:
        proxy = pd.DataFrame()
        try:
            proxy = load_citizenship_proxy(blend_geos)
        except Exception as e:
            meta["proxy_error"] = str(e)
        if not proxy.empty:
            proxy = proxy[proxy["geo"].isin(blend_geos)].copy()
            if not proxy.empty:
                od_tot = (
                    out[out["geo"].isin(blend_geos)]
                    .groupby(["geo", "year"], as_index=False)["immigration_od"]
                    .sum(min_count=1)
                    .rename(columns={"immigration_od": "od_total"})
                )
                proxy_keys = proxy[["geo", "year"]].drop_duplicates()
                keys = proxy_keys.merge(od_tot, on=["geo", "year"], how="left")
                replace_keys = keys[
                    keys["od_total"].isna() | (pd.to_numeric(keys["od_total"], errors="coerce") <= 0)
                ][["geo", "year"]].copy()

                if not replace_keys.empty:
                    out = out.merge(replace_keys.assign(_drop=1), on=["geo", "year"], how="left")
                    out = out[out["_drop"].isna()].drop(columns="_drop")

                    proxy_use = proxy.merge(replace_keys, on=["geo", "year"], how="inner").copy()
                    proxy_use = proxy_use.rename(columns={"immigration_proxy": "immigration_od"})
                    proxy_use = proxy_use[["geo", "origin", "year", "immigration_od", "source"]]

                    out = pd.concat([out, proxy_use], ignore_index=True)
                    out = (
                        out.groupby(["geo", "origin", "year", "source"], as_index=False)["immigration_od"]
                        .sum(min_count=1)
                        .sort_values(["geo", "origin", "year"])
                        .reset_index(drop=True)
                    )
                    by_geo = (
                        replace_keys.groupby("geo", as_index=False)["year"]
                        .agg(
                            years_replaced=lambda s: ",".join(
                                str(int(x)) for x in sorted(set(pd.to_numeric(s, errors="coerce").dropna().astype(int)))
                            )
                        )
                    )
                    meta["geo_years_replaced"] = int(len(replace_keys))
                    meta["geo_years_replaced_detail"] = by_geo.to_dict(orient="records")
    # UK official bridge for remaining post-gap years not covered by Eurostat proxy.
    synth, synth_meta = build_uk_ons_synthetic_od(out)
    if not synth.empty:
        synth_keys = synth[["geo", "year"]].drop_duplicates()
        out = out.merge(synth_keys.assign(_drop=1), on=["geo", "year"], how="left")
        out = out[out["_drop"].isna()].drop(columns="_drop")
        out = pd.concat([out, synth], ignore_index=True)
        out = (
            out.groupby(["geo", "origin", "year", "source"], as_index=False)["immigration_od"]
            .sum(min_count=1)
            .sort_values(["geo", "origin", "year"])
            .reset_index(drop=True)
        )
        meta["uk_ons_synthetic_bridge"] = synth_meta
    else:
        meta["uk_ons_synthetic_bridge"] = synth_meta
    return out, meta


def fetch_world_bank_indicator(indicator: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    page = 1
    while True:
        url = (
            f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
            f"?format=json&per_page=20000&page={page}"
        )
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, list) or len(payload) < 2:
            raise RuntimeError(f"Unexpected World Bank response for {indicator}")
        meta, data = payload
        for item in data:
            country = item.get("country") or {}
            rows.append(
                {
                    "country_code_wb2": country.get("id"),
                    "country_name": country.get("value"),
                    "country_iso3": item.get("countryiso3code"),
                    "year": pd.to_numeric(item.get("date"), errors="coerce"),
                    "value": pd.to_numeric(item.get("value"), errors="coerce"),
                    "indicator": indicator,
                }
            )
        if page >= int(meta["pages"]):
            break
        page += 1
    out = pd.DataFrame(rows)
    out = out.dropna(subset=["year"]).copy()
    out["year"] = out["year"].astype(int)
    return out


def load_origin_gdp_levels() -> pd.DataFrame:
    if WB_GDP_LEVEL_FILE.exists():
        wb = pd.read_csv(WB_GDP_LEVEL_FILE)
    else:
        wb = fetch_world_bank_indicator(WB_GDP_LEVEL_INDICATOR)
        wb.to_csv(WB_GDP_LEVEL_FILE, index=False)

    wb["year"] = pd.to_numeric(wb["year"], errors="coerce")
    wb["value"] = pd.to_numeric(wb["value"], errors="coerce")
    wb = wb.dropna(subset=["year"]).copy()
    wb["year"] = wb["year"].astype(int)

    wb["origin_wb2"] = wb["country_code_wb2"].astype(str)
    wb["origin"] = wb["origin_wb2"].map(WB2_TO_EUROSTAT).fillna(wb["origin_wb2"])
    wb = wb[wb["origin"].str.match(COUNTRY2_RE, na=False)].copy()
    wb = wb.rename(columns={"value": "wb_gdp_pc_ppp_const"})

    keep = ["origin", "year", "wb_gdp_pc_ppp_const", "country_name", "country_iso3"]
    wb = wb[keep].drop_duplicates(subset=["origin", "year"]).sort_values(["origin", "year"]).reset_index(drop=True)
    return wb


def build_migration_composition_panel() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    annual = load_harmonized_annual_base()

    od = load_od_migration()
    od, blend_meta = blend_od_with_case_proxy(od)
    OD_BLEND_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    od.to_parquet(OD_BLEND_PARQUET, index=False)
    od.to_csv(OD_BLEND_CSV, index=False)

    wb_gdp = load_origin_gdp_levels()
    od = od.merge(wb_gdp[["origin", "year", "wb_gdp_pc_ppp_const"]], on=["origin", "year"], how="left")
    od["origin_group"] = od["origin"].map(origin_group)

    # Year-specific terciles for origin-country income level.
    q = (
        wb_gdp.dropna(subset=["wb_gdp_pc_ppp_const"])
        .groupby("year")["wb_gdp_pc_ppp_const"]
        .quantile([1 / 3, 2 / 3])
        .unstack()
        .rename(columns={1 / 3: "gdp_q33", 2 / 3: "gdp_q67"})
        .reset_index()
    )
    od = od.merge(q, on="year", how="left")

    od["w"] = od["immigration_od"].astype(float)
    od["gdp_known"] = od["wb_gdp_pc_ppp_const"].notna()
    od["w_cov"] = np.where(od["gdp_known"], od["w"], 0.0)
    od["log_origin_gdp_pc_ppp_const"] = np.where(od["wb_gdp_pc_ppp_const"] > 0, np.log(od["wb_gdp_pc_ppp_const"]), np.nan)
    od["w_log_origin_gdp"] = np.where(od["gdp_known"], od["w"] * od["log_origin_gdp_pc_ppp_const"], 0.0)
    od["w_origin_gdp"] = np.where(od["gdp_known"], od["w"] * od["wb_gdp_pc_ppp_const"], 0.0)

    od["origin_income_bin"] = np.where(
        ~od["gdp_known"] | od["gdp_q33"].isna() | od["gdp_q67"].isna(),
        pd.NA,
        np.where(
            od["wb_gdp_pc_ppp_const"] <= od["gdp_q33"],
            "low",
            np.where(od["wb_gdp_pc_ppp_const"] >= od["gdp_q67"], "high", "mid"),
        ),
    )
    od["w_low_bin"] = np.where(od["origin_income_bin"] == "low", od["w"], 0.0)
    od["w_mid_bin"] = np.where(od["origin_income_bin"] == "mid", od["w"], 0.0)
    od["w_high_bin"] = np.where(od["origin_income_bin"] == "high", od["w"], 0.0)

    # Group-level OD aggregation for origin-composition shares and group-specific inflow rates.
    grp = (
        od.groupby(["geo", "year", "origin_group"], as_index=False)["w"]
        .sum(min_count=1)
        .rename(columns={"w": "immigration_od_group"})
    )
    grp_total = grp.groupby(["geo", "year"], as_index=False)["immigration_od_group"].sum().rename(
        columns={"immigration_od_group": "od_total_groups"}
    )
    grp = grp.merge(grp_total, on=["geo", "year"], how="left")
    grp["share_group"] = np.where(grp["od_total_groups"] > 0, grp["immigration_od_group"] / grp["od_total_groups"], np.nan)
    grp_wide_levels = grp.pivot_table(index=["geo", "year"], columns="origin_group", values="immigration_od_group", aggfunc="first").reset_index()
    grp_wide_shares = grp.pivot_table(index=["geo", "year"], columns="origin_group", values="share_group", aggfunc="first").reset_index()
    grp_wide_levels.columns = [f"od_group_{c}" if c not in {"geo", "year"} else c for c in grp_wide_levels.columns]
    grp_wide_shares.columns = [f"share_group_{c}" if c not in {"geo", "year"} else c for c in grp_wide_shares.columns]

    # Selected-origin shares for country cases (Spain + UK-relevant origins).
    sel = od[od["origin"].isin(CASE_SELECTED_ORIGINS)].copy()
    sel_agg = (
        sel.groupby(["geo", "year", "origin"], as_index=False)["w"]
        .sum(min_count=1)
        .rename(columns={"w": "immigration_od_origin"})
    )
    sel_total = od.groupby(["geo", "year"], as_index=False)["w"].sum().rename(columns={"w": "od_total_all_origins"})
    sel_agg = sel_agg.merge(sel_total, on=["geo", "year"], how="left")
    sel_agg["share_origin"] = np.where(sel_agg["od_total_all_origins"] > 0, sel_agg["immigration_od_origin"] / sel_agg["od_total_all_origins"], np.nan)
    sel_wide = sel_agg.pivot_table(index=["geo", "year"], columns="origin", values="share_origin", aggfunc="first").reset_index()
    sel_wide.columns = [f"share_origin_{c}" if c not in {"geo", "year"} else c for c in sel_wide.columns]

    comp = (
        od.groupby(["geo", "year"], as_index=False)
        .agg(
            od_immigration_total=("w", "sum"),
            od_immigration_gdp_covered=("w_cov", "sum"),
            od_origin_gdp_wsum=("w_origin_gdp", "sum"),
            od_origin_log_gdp_wsum=("w_log_origin_gdp", "sum"),
            od_share_low_num=("w_low_bin", "sum"),
            od_share_mid_num=("w_mid_bin", "sum"),
            od_share_high_num=("w_high_bin", "sum"),
            od_origins_count=("origin", "nunique"),
            od_rows_gdp_known=("gdp_known", "sum"),
        )
        .sort_values(["geo", "year"])
        .reset_index(drop=True)
    )

    comp["od_gdp_coverage_share"] = np.where(
        comp["od_immigration_total"] > 0, comp["od_immigration_gdp_covered"] / comp["od_immigration_total"], np.nan
    )
    comp["origin_gdp_pc_ppp_const_wavg"] = np.where(
        comp["od_immigration_gdp_covered"] > 0, comp["od_origin_gdp_wsum"] / comp["od_immigration_gdp_covered"], np.nan
    )
    comp["origin_gdp_pc_ppp_const_wavg_log"] = np.where(
        comp["od_immigration_gdp_covered"] > 0, comp["od_origin_log_gdp_wsum"] / comp["od_immigration_gdp_covered"], np.nan
    )
    comp["share_low_gdp_origins"] = np.where(
        comp["od_immigration_gdp_covered"] > 0, comp["od_share_low_num"] / comp["od_immigration_gdp_covered"], np.nan
    )
    comp["share_mid_gdp_origins"] = np.where(
        comp["od_immigration_gdp_covered"] > 0, comp["od_share_mid_num"] / comp["od_immigration_gdp_covered"], np.nan
    )
    comp["share_high_gdp_origins"] = np.where(
        comp["od_immigration_gdp_covered"] > 0, comp["od_share_high_num"] / comp["od_immigration_gdp_covered"], np.nan
    )

    panel = annual.merge(comp, on=["geo", "year"], how="left")
    panel = panel.merge(grp_wide_levels, on=["geo", "year"], how="left")
    panel = panel.merge(grp_wide_shares, on=["geo", "year"], how="left")
    panel = panel.merge(sel_wide, on=["geo", "year"], how="left")
    panel = panel.sort_values(["geo", "year"]).reset_index(drop=True)

    g = panel.groupby("geo", sort=False)
    lag_cols = [
        "od_immigration_total",
        "od_gdp_coverage_share",
        "origin_gdp_pc_ppp_const_wavg",
        "origin_gdp_pc_ppp_const_wavg_log",
        "share_low_gdp_origins",
        "share_mid_gdp_origins",
        "share_high_gdp_origins",
    ]
    for col in lag_cols:
        panel[f"L1_{col}"] = g[col].shift(1)

    # Lag group shares and selected-origin shares.
    group_share_cols = [c for c in panel.columns if c.startswith("share_group_")]
    selected_share_cols = [c for c in panel.columns if c.startswith("share_origin_")]
    group_level_cols = [c for c in panel.columns if c.startswith("od_group_")]
    for col in group_share_cols + selected_share_cols + group_level_cols:
        panel[f"L1_{col}"] = g[col].shift(1)

    # Center the composition variable for interaction interpretation.
    if "L1_origin_gdp_pc_ppp_const_wavg_log" in panel.columns:
        mu = panel["L1_origin_gdp_pc_ppp_const_wavg_log"].mean(skipna=True)
        panel["L1_origin_gdp_pc_ppp_const_wavg_log_c"] = panel["L1_origin_gdp_pc_ppp_const_wavg_log"] - mu

    # Useful scale variable: share of OD-inflow captured relative to Eurostat aggregate immigration.
    panel["od_vs_agg_immig_ratio"] = np.where(
        panel["immigration"] > 0, panel["od_immigration_total"] / panel["immigration"], np.nan
    )
    for col in group_level_cols:
        suffix = col.replace("od_group_", "")
        panel[f"{col}_rate_per_1000"] = np.where(panel["population"] > 0, panel[col] / panel["population"] * 1000.0, np.nan)
        panel[f"L1_{col}_rate_per_1000"] = g[f"{col}_rate_per_1000"].shift(1)

    out_cols = sorted(panel.columns)
    panel = panel[out_cols]

    panel.to_csv(PROC_DIR / "panel_annual_migration_composition.csv", index=False)
    panel.to_parquet(PROC_DIR / "panel_annual_migration_composition.parquet", index=False)

    meta = {
        "panel_rows": int(len(panel)),
        "countries_total": int(panel["geo"].astype(str).nunique()),
        "years_min": int(pd.to_numeric(panel["year"], errors="coerce").min()),
        "years_max": int(pd.to_numeric(panel["year"], errors="coerce").max()),
        "od_rows": int(len(od)),
        "od_destination_countries": int(od["geo"].astype(str).nunique()),
        "od_origin_countries": int(od["origin"].astype(str).nunique()),
        "origin_groups_present": sorted([str(x) for x in od["origin_group"].dropna().unique()]),
        "wb_origin_gdp_rows": int(len(wb_gdp)),
        "wb_origin_gdp_origins": int(wb_gdp["origin"].astype(str).nunique()),
        "nonmissing_L1_origin_gdp_comp": int(panel["L1_origin_gdp_pc_ppp_const_wavg_log"].notna().sum()),
        "nonmissing_L1_share_high": int(panel["L1_share_high_gdp_origins"].notna().sum()),
        "median_od_gdp_coverage_share": float(panel["od_gdp_coverage_share"].median(skipna=True)),
        "median_od_vs_agg_immig_ratio": float(panel["od_vs_agg_immig_ratio"].median(skipna=True)),
        "od_blend_meta": blend_meta,
        "od_blended_output_parquet": str(OD_BLEND_PARQUET),
        "od_blended_output_csv": str(OD_BLEND_CSV),
    }
    return panel, meta, od


def _fit_panel_formula(df_panel: pd.DataFrame, formula: str, label: str, needed: list[str]) -> tuple[pd.DataFrame, str] | None:
    d = df_panel.dropna(subset=needed).copy()
    if len(d) == 0:
        return None
    mod = PanelOLS.from_formula(formula, data=d, drop_absorbed=True, check_rank=False)
    res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
    coef = pd.DataFrame(
        {
            "model": label,
            "term": res.params.index,
            "coef": res.params.values,
            "std_err": res.std_errors.values,
            "t_stat": res.tstats.values,
            "p_value": res.pvalues.values,
            "nobs": float(res.nobs),
            "r2_within": float(getattr(res, "rsquared_within", np.nan)),
            "r2_overall": float(getattr(res, "rsquared_overall", np.nan)),
        }
    )
    text = f"# {label}\nFormula: {formula}\n\n{res.summary}"
    return coef, text


def run_models(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    d = panel.replace([np.inf, -np.inf], np.nan).copy()

    for col in [
        "hpi_growth",
        "L1_immigration_rate_per_1000",
        "L1_net_migration_rate",
        "L1_air_growth",
        "L1_origin_gdp_pc_ppp_const_wavg_log",
        "L1_origin_gdp_pc_ppp_const_wavg_log_c",
    ]:
        if col in d.columns:
            d[col] = winsorize_series(d[col], 0.01, 0.99)

    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d = d.dropna(subset=["geo", "year"]).copy()
    d["year"] = d["year"].astype(int)
    d = d.set_index(["geo", "year"]).sort_index()

    controls = [
        "L1_air_growth",
        "L1_gdp_pc_growth",
        "L1_unemployment_rate",
        "L1_inflation_hicp",
        "L1_long_rate",
        "L1_pop_growth",
    ]
    key_group_share_cols = [
        "L1_share_group_LATAM_CARIB",
        "L1_share_group_EU_EEA_CH_UK",
        "L1_share_group_MENA",
    ]
    key_group_share_cols = [c for c in key_group_share_cols if c in d.columns]

    specs: list[tuple[str, str, list[str]]] = [
        (
            "annual_fe_immig_volume_origin_gdp",
            "hpi_growth ~ 1 + L1_immigration_rate_per_1000 + L1_origin_gdp_pc_ppp_const_wavg_log + EntityEffects + TimeEffects",
            ["hpi_growth", "L1_immigration_rate_per_1000", "L1_origin_gdp_pc_ppp_const_wavg_log"],
        ),
        (
            "annual_fe_immig_volume_origin_gdp_controls",
            (
                "hpi_growth ~ 1 + L1_immigration_rate_per_1000 + L1_origin_gdp_pc_ppp_const_wavg_log + "
                + " + ".join(controls)
                + " + EntityEffects + TimeEffects"
            ),
            ["hpi_growth", "L1_immigration_rate_per_1000", "L1_origin_gdp_pc_ppp_const_wavg_log"] + controls,
        ),
        (
            "annual_fe_immig_origin_gdp_interaction",
            (
                "hpi_growth ~ 1 + L1_immigration_rate_per_1000 + L1_origin_gdp_pc_ppp_const_wavg_log_c + "
                "L1_immigration_rate_per_1000:L1_origin_gdp_pc_ppp_const_wavg_log_c + "
                + " + ".join(controls)
                + " + EntityEffects + TimeEffects"
            ),
            ["hpi_growth", "L1_immigration_rate_per_1000", "L1_origin_gdp_pc_ppp_const_wavg_log_c"] + controls,
        ),
        (
            "annual_fe_immig_origin_income_shares",
            (
                "hpi_growth ~ 1 + L1_immigration_rate_per_1000 + L1_share_high_gdp_origins + L1_share_low_gdp_origins + "
                + " + ".join(controls)
                + " + EntityEffects + TimeEffects"
            ),
            ["hpi_growth", "L1_immigration_rate_per_1000", "L1_share_high_gdp_origins", "L1_share_low_gdp_origins"] + controls,
        ),
        (
            "annual_fe_netmig_controls_matchedsample",
            (
                "hpi_growth ~ 1 + L1_net_migration_rate + "
                + " + ".join(controls)
                + " + EntityEffects + TimeEffects"
            ),
            [
                "hpi_growth",
                "L1_net_migration_rate",
                "L1_origin_gdp_pc_ppp_const_wavg_log",
            ]
            + controls,
        ),
        (
            "annual_fe_netmig_plus_origin_gdpcomp",
            (
                "hpi_growth ~ 1 + L1_net_migration_rate + L1_origin_gdp_pc_ppp_const_wavg_log + "
                + " + ".join(controls)
                + " + EntityEffects + TimeEffects"
            ),
            ["hpi_growth", "L1_net_migration_rate", "L1_origin_gdp_pc_ppp_const_wavg_log"] + controls,
        ),
    ]

    if key_group_share_cols:
        specs.append(
            (
                "annual_fe_netmig_plus_origin_groupshares",
                (
                    "hpi_growth ~ 1 + L1_net_migration_rate + "
                    + " + ".join(key_group_share_cols)
                    + " + "
                    + " + ".join(controls)
                    + " + EntityEffects + TimeEffects"
                ),
                ["hpi_growth", "L1_net_migration_rate"] + key_group_share_cols + controls,
            )
        )
        specs.append(
            (
                "annual_fe_netmig_plus_gdpcomp_groupshares",
                (
                    "hpi_growth ~ 1 + L1_net_migration_rate + L1_origin_gdp_pc_ppp_const_wavg_log + "
                    + " + ".join(key_group_share_cols)
                    + " + "
                    + " + ".join(controls)
                    + " + EntityEffects + TimeEffects"
                ),
                ["hpi_growth", "L1_net_migration_rate", "L1_origin_gdp_pc_ppp_const_wavg_log"] + key_group_share_cols + controls,
            )
        )

    coef_frames: list[pd.DataFrame] = []
    texts: list[str] = []
    for label, formula, needed in specs:
        out = _fit_panel_formula(d, formula, label, needed)
        if out is None:
            continue
        coef, txt = out
        coef_frames.append(coef)
        texts.append(txt)

    coef_df = pd.concat(coef_frames, ignore_index=True) if coef_frames else pd.DataFrame()

    stats_cols = [
        "hpi_growth",
        "immigration_rate_per_1000",
        "net_migration_rate",
        "origin_gdp_pc_ppp_const_wavg",
        "origin_gdp_pc_ppp_const_wavg_log",
        "share_high_gdp_origins",
        "share_low_gdp_origins",
        "share_group_LATAM_CARIB",
        "share_group_EU_EEA_CH_UK",
        "share_group_MENA",
        "share_origin_CH",
        "share_origin_BR",
        "share_origin_MA",
        "od_gdp_coverage_share",
        "od_vs_agg_immig_ratio",
    ]
    srows = []
    reset = d.reset_index()
    for c in stats_cols:
        if c not in reset.columns:
            continue
        x = pd.to_numeric(reset[c], errors="coerce")
        srows.append(
            {
                "variable": c,
                "n": int(x.notna().sum()),
                "mean": float(x.mean()) if x.notna().any() else np.nan,
                "std": float(x.std()) if x.notna().any() else np.nan,
                "p25": float(x.quantile(0.25)) if x.notna().any() else np.nan,
                "p50": float(x.quantile(0.50)) if x.notna().any() else np.nan,
                "p75": float(x.quantile(0.75)) if x.notna().any() else np.nan,
            }
        )
    stats_df = pd.DataFrame(srows)
    return coef_df, texts, stats_df


def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def _coef_entry(coef_df: pd.DataFrame, model: str, term: str) -> tuple[str, str]:
    row = coef_df[(coef_df["model"] == model) & (coef_df["term"] == term)]
    if row.empty:
        return "", ""
    r = row.iloc[0]
    return f"{r['coef']:.3f}{_stars(float(r['p_value']))}", f"({r['std_err']:.3f})"


def write_case_outputs(panel: pd.DataFrame, od: pd.DataFrame) -> dict[str, pd.DataFrame]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}

    od_case = od[od["geo"].isin(CASE_DESTS)].copy()
    if not od_case.empty:
        od_case = od_case.copy()
        od_case["source"] = od_case.get("source", "od_prev_residence")
        od_case["flow"] = pd.to_numeric(od_case["immigration_od"], errors="coerce")
        od_case = od_case.dropna(subset=["flow"]).copy()
    else:
        od_case = pd.DataFrame(columns=["geo", "origin", "year", "flow", "source"])

    case = od_case[["geo", "origin", "year", "flow", "source"]].copy()
    case["geo"] = case["geo"].astype(str)
    case["year"] = pd.to_numeric(case["year"], errors="coerce").astype(int)

    # Safety fallback for case visuals/tables: fill missing/zero geo-years with proxy.
    try:
        ctz = load_case_citizenship_proxy()
        ctz = ctz[ctz["geo"].isin(CASE_DESTS)].copy()
        if not ctz.empty:
            ctz["flow"] = pd.to_numeric(ctz["immigration_proxy"], errors="coerce")
            ctz = ctz.dropna(subset=["flow"]).copy()
            case_tot = (
                case.groupby(["geo", "year"], as_index=False)["flow"]
                .sum(min_count=1)
                .rename(columns={"flow": "od_total"})
            )
            ctz_keys = ctz[["geo", "year"]].drop_duplicates()
            k = ctz_keys.merge(case_tot, on=["geo", "year"], how="left")
            fill_keys = k[k["od_total"].isna() | (pd.to_numeric(k["od_total"], errors="coerce") <= 0)][["geo", "year"]]
            if not fill_keys.empty:
                ctz_use = ctz.merge(fill_keys, on=["geo", "year"], how="inner")
                ctz_use = ctz_use[["geo", "origin", "year", "flow", "source"]]
                case = case.merge(fill_keys.assign(_drop=1), on=["geo", "year"], how="left")
                case = case[case["_drop"].isna()].drop(columns="_drop")
                case = pd.concat([case, ctz_use], ignore_index=True)
    except Exception as e:
        print(f"[warn] unable to build citizenship proxy case decomposition: {e}")

    if case.empty:
        return out

    case["origin_group"] = case["origin"].map(origin_group)
    totals = case.groupby(["geo", "year"], as_index=False)["flow"].sum().rename(columns={"flow": "od_total"})
    src_unique = case.groupby("geo", as_index=False)["source"].nunique().rename(columns={"source": "source_n"})
    source_map = case.groupby("geo", as_index=False)["source"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
    source_map = source_map.merge(src_unique, on="geo", how="left")
    source_map["source"] = np.where(source_map["source_n"] > 1, "multi_source_blend", source_map["source"])
    source_map = source_map[["geo", "source"]]
    out["case_source_map"] = source_map

    group_case = (
        case.groupby(["geo", "year", "origin_group"], as_index=False)["flow"]
        .sum(min_count=1)
        .merge(totals, on=["geo", "year"], how="left")
    )
    group_case = group_case.merge(source_map, on="geo", how="left")
    group_case["share"] = np.where(group_case["od_total"] > 0, group_case["flow"] / group_case["od_total"], np.nan)
    group_case = group_case.rename(columns={"flow": "immigration_case"})
    group_case = group_case.sort_values(["geo", "year", "origin_group"]).reset_index(drop=True)
    group_case.to_csv(RESULTS_DIR / "migration_case_es_pt_group_shares.csv", index=False)
    out["group_case"] = group_case

    sel_case = case[case["origin"].isin(CASE_SELECTED_ORIGINS)].copy()
    sel_case = (
        sel_case.groupby(["geo", "year", "origin"], as_index=False)["flow"]
        .sum(min_count=1)
        .merge(totals, on=["geo", "year"], how="left")
    )
    sel_case = sel_case.merge(source_map, on="geo", how="left")
    sel_case["share"] = np.where(sel_case["od_total"] > 0, sel_case["flow"] / sel_case["od_total"], np.nan)
    sel_case = sel_case.rename(columns={"flow": "immigration_case"})
    sel_case = sel_case.sort_values(["geo", "year", "origin"]).reset_index(drop=True)
    sel_case.to_csv(RESULTS_DIR / "migration_case_es_pt_selected_origin_shares.csv", index=False)
    out["selected_case"] = sel_case

    top_full = (
        case.groupby(["geo", "origin"], as_index=False)["flow"]
        .sum(min_count=1)
        .rename(columns={"flow": "immig_total_full"})
    )
    win_rows = []
    for geo, (y0, y1) in CASE_RANK_WINDOWS.items():
        mask = (case["geo"] == geo) & (case["year"] >= y0) & (case["year"] <= y1)
        if mask.any():
            x = case.loc[mask].copy()
            yy0 = int(pd.to_numeric(x["year"], errors="coerce").min())
            yy1 = int(pd.to_numeric(x["year"], errors="coerce").max())
            x["rank_window_label"] = f"{yy0}--{yy1}"
            win_rows.append(x)
    rank_window = pd.concat(win_rows, ignore_index=True) if win_rows else case.iloc[0:0].copy()
    top_rank = (
        rank_window.groupby(["geo", "origin"], as_index=False)["flow"]
        .sum(min_count=1)
        .rename(columns={"flow": "immig_total_rank_window"})
    )
    top = top_full.merge(top_rank, on=["geo", "origin"], how="outer").fillna(0)
    top = top.merge(
        case.groupby(["geo", "origin"], as_index=False)["origin_group"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
        on=["geo", "origin"],
        how="left",
    )
    top = top.merge(source_map, on="geo", how="left")
    if not rank_window.empty and "rank_window_label" in rank_window.columns:
        geo_window = (
            rank_window.groupby("geo", as_index=False)["rank_window_label"]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
        )
        top = top.merge(geo_window, on="geo", how="left")
    top = top.sort_values(["geo", "immig_total_rank_window", "immig_total_full"], ascending=[True, False, False]).reset_index(drop=True)
    top.to_csv(RESULTS_DIR / "migration_case_es_pt_top_origins.csv", index=False)
    out["top_case"] = top

    out["case_proxy_used_for"] = sorted(case.loc[case["source"] == "ctz_proxy", "geo"].astype(str).unique().tolist())

    return out


def _plot_composition_coefficients(coef_df: pd.DataFrame) -> None:
    if not PAPER_FIGS_DIR.exists():
        return
    rows = []
    specs = [
        ("annual_fe_netmig_controls_matchedsample", "Net migration (matched baseline)", "L1_net_migration_rate", "Net migration rate"),
        ("annual_fe_netmig_plus_origin_gdpcomp", "Net migration + origin GDP", "L1_net_migration_rate", "Net migration rate"),
        ("annual_fe_netmig_plus_origin_gdpcomp", "Net migration + origin GDP", "L1_origin_gdp_pc_ppp_const_wavg_log", "Origin GDPpc composition"),
        ("annual_fe_netmig_plus_origin_groupshares", "Net migration + origin groups", "L1_share_group_LATAM_CARIB", "Share from Latin America/Carib"),
        ("annual_fe_netmig_plus_origin_groupshares", "Net migration + origin groups", "L1_share_group_EU_EEA_CH_UK", "Share from EU/EEA/CH/UK"),
        ("annual_fe_netmig_plus_origin_groupshares", "Net migration + origin groups", "L1_share_group_MENA", "Share from MENA"),
        ("annual_fe_netmig_plus_gdpcomp_groupshares", "Net migration + GDP + groups", "L1_origin_gdp_pc_ppp_const_wavg_log", "Origin GDPpc composition"),
    ]
    for model, mlabel, term, tlabel in specs:
        r = coef_df[(coef_df["model"] == model) & (coef_df["term"] == term)]
        if r.empty:
            continue
        x = r.iloc[0]
        rows.append(
            {
                "model_label": mlabel,
                "term_label": tlabel,
                "coef": float(x["coef"]),
                "se": float(x["std_err"]),
                "p": float(x["p_value"]),
            }
        )
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["label"] = df["model_label"] + " | " + df["term_label"]
    y = np.arange(len(df))[::-1]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    colors = ["#b22222" if "Net migration" in l else "#1f4e79" if "GDP" in l else "#2f7d32" for l in df["term_label"]]
    ax.axvline(0, color="#444444", lw=1, ls="--")
    ax.errorbar(df["coef"], y, xerr=1.96 * df["se"], fmt="none", ecolor="#333333", elinewidth=1.5, capsize=3)
    ax.scatter(df["coef"], y, c=colors, s=45, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=9)
    ax.set_xlabel("Coefficient estimate (95% CI)")
    ax.set_title("Who arrives matters: migration volume, origin-income composition, and origin-group shares")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(PAPER_FIGS_DIR / "fig_origin_composition_coefficients.pdf")
    plt.close(fig)


def _plot_case_origin_mix(case_outputs: dict[str, pd.DataFrame]) -> None:
    if not PAPER_FIGS_DIR.exists():
        return
    gdf = case_outputs.get("group_case")
    sdf = case_outputs.get("selected_case")
    src_map = case_outputs.get("case_source_map")
    if gdf is None or gdf.empty:
        return
    src_lookup = {}
    if src_map is not None and not src_map.empty:
        src_lookup = dict(zip(src_map["geo"].astype(str), src_map["source"].astype(str)))

    n_geo = len(CASE_DESTS)
    fig, axes = plt.subplots(2, n_geo, figsize=(4.0 * n_geo, 7.8), sharex=True)
    if n_geo == 1:
        axes = np.array([[axes[0]], [axes[1]]])  # pragma: no cover
    group_focus = ["LATAM_CARIB", "EU_EEA_CH_UK", "MENA", "N_AMERICA"]
    colors = {
        "LATAM_CARIB": "#b22222",
        "EU_EEA_CH_UK": "#1f4e79",
        "MENA": "#c78b00",
        "N_AMERICA": "#2f7d32",
    }
    for i, geo in enumerate(CASE_DESTS):
        ax = axes[0, i]
        any_series = False
        for grp in group_focus:
            s = gdf[(gdf["geo"] == geo) & (gdf["origin_group"] == grp)].sort_values("year")
            if s.empty:
                continue
            any_series = True
            plot_kw = {"lw": 2, "label": grp.replace("_", " "), "color": colors.get(grp)}
            if len(s) < 2:
                plot_kw.update({"marker": "o", "markersize": 4})
            ax.plot(s["year"], s["share"], **plot_kw)
        src = src_lookup.get(geo, "od_prev_residence")
        if src == "od_prev_residence":
            src_lbl = "OD previous residence"
        elif src == "ctz_proxy":
            src_lbl = "citizenship proxy"
        elif src == "ons_uk_group_scaled":
            src_lbl = "ONS YE-June bridge"
        else:
            src_lbl = "multi-source blend"
        ax.set_title(f"{geo}: origin-group shares ({src_lbl})")
        ax.set_ylabel("Share of inflows")
        ax.grid(alpha=0.2)
        ax.set_ylim(bottom=0)
        if any_series and i == 0:
            ax.legend(fontsize=8, ncol=2, loc="upper right")
        if not any_series:
            ax.text(0.5, 0.5, "No country-level case decomposition\\navailable in current source", ha="center", va="center", transform=ax.transAxes, fontsize=9)
    if sdf is not None and not sdf.empty:
        focus_by_geo = {
            "ES": ["CO", "MA", "VE", "AR", "PE", "UK"],
            "UK": ["IN", "CN", "RO", "PK", "PL", "NG"],
        }
        for i, geo in enumerate(CASE_DESTS):
            ax = axes[1, i]
            any_series = False
            for origin in focus_by_geo.get(geo, []):
                s = sdf[(sdf["geo"] == geo) & (sdf["origin"] == origin)].sort_values("year")
                if s.empty:
                    continue
                any_series = True
                plot_kw = {"lw": 1.8, "label": origin}
                if len(s) < 2:
                    plot_kw.update({"marker": "o", "markersize": 4})
                ax.plot(s["year"], s["share"], **plot_kw)
            src = src_lookup.get(geo, "od_prev_residence")
            if src == "ctz_proxy":
                ax.set_title(f"{geo}: selected-country citizenship shares (proxy)")
            elif src == "ons_uk_group_scaled":
                ax.set_title(f"{geo}: selected-origin shares (ONS bridge)")
            elif src == "multi_source_blend":
                ax.set_title(f"{geo}: selected-origin shares (multi-source blend)")
            else:
                ax.set_title(f"{geo}: selected-origin shares (including CH)")
            ax.set_ylabel("Share of inflows")
            ax.grid(alpha=0.2)
            ax.set_ylim(bottom=0)
            if any_series:
                ax.legend(fontsize=8, ncol=3, loc="upper right")
            else:
                ax.text(0.5, 0.5, "No country-level case decomposition\\navailable in current source", ha="center", va="center", transform=ax.transAxes, fontsize=9)
    for row in axes:
        for ax in np.atleast_1d(row):
            ax.set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(PAPER_FIGS_DIR / "fig_es_pt_origin_mix.pdf")
    plt.close(fig)


def _plot_country_who_arrives_contributions(panel: pd.DataFrame, coef_df: pd.DataFrame) -> None:
    if not PAPER_FIGS_DIR.exists():
        return
    row_m = coef_df[(coef_df["model"] == "annual_fe_netmig_plus_origin_gdpcomp") & (coef_df["term"] == "L1_net_migration_rate")]
    row_c = coef_df[
        (coef_df["model"] == "annual_fe_netmig_plus_origin_gdpcomp")
        & (coef_df["term"] == "L1_origin_gdp_pc_ppp_const_wavg_log")
    ]
    if row_m.empty or row_c.empty:
        return
    beta_m = float(row_m.iloc[0]["coef"])
    beta_c = float(row_c.iloc[0]["coef"])

    needed = [
        "geo",
        "year",
        "hpi_growth",
        "L1_net_migration_rate",
        "L1_origin_gdp_pc_ppp_const_wavg_log",
        "L1_air_growth",
        "L1_gdp_pc_growth",
        "L1_unemployment_rate",
        "L1_inflation_hicp",
        "L1_long_rate",
        "L1_pop_growth",
    ]
    d = panel.copy()
    for c in ["hpi_growth", "L1_net_migration_rate", "L1_origin_gdp_pc_ppp_const_wavg_log"]:
        if c in d.columns:
            d[c] = winsorize_series(d[c], 0.01, 0.99)
    d = d.dropna(subset=[c for c in needed if c in d.columns]).copy()
    if d.empty:
        return

    # Use centered components so bars reflect relative contribution differences across countries.
    m_mean = d["L1_net_migration_rate"].mean()
    c_mean = d["L1_origin_gdp_pc_ppp_const_wavg_log"].mean()
    d["contrib_netmig_rel_pp"] = beta_m * (d["L1_net_migration_rate"] - m_mean)
    d["contrib_origin_gdp_rel_pp"] = beta_c * (d["L1_origin_gdp_pc_ppp_const_wavg_log"] - c_mean)
    d["contrib_total_rel_pp"] = d["contrib_netmig_rel_pp"] + d["contrib_origin_gdp_rel_pp"]

    agg = (
        d.groupby("geo", as_index=False)
        .agg(
            contrib_total_rel_pp=("contrib_total_rel_pp", "mean"),
            contrib_netmig_rel_pp=("contrib_netmig_rel_pp", "mean"),
            contrib_origin_gdp_rel_pp=("contrib_origin_gdp_rel_pp", "mean"),
            avg_hpi_growth=("hpi_growth", "mean"),
            nobs=("hpi_growth", "size"),
        )
        .sort_values("contrib_total_rel_pp")
        .reset_index(drop=True)
    )
    if agg.empty:
        return

    # Keep all countries if compact enough; otherwise show tails.
    if len(agg) > 18:
        show = pd.concat([agg.head(9), agg.tail(9)], ignore_index=True)
    else:
        show = agg.copy()
    show = show.sort_values("contrib_total_rel_pp").reset_index(drop=True)
    y = np.arange(len(show))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.8, 5.8), gridspec_kw={"width_ratios": [1.5, 1.0]}
    )
    # Stacked horizontal bars (relative fitted contribution decomposition).
    left = np.minimum(show["contrib_netmig_rel_pp"], 0)
    right = np.maximum(show["contrib_netmig_rel_pp"], 0)
    ax1.barh(y, right, color="#b22222", alpha=0.88, label="Net migration component")
    ax1.barh(y, left, color="#b22222", alpha=0.88)
    ax1.barh(
        y,
        show["contrib_origin_gdp_rel_pp"],
        left=show["contrib_netmig_rel_pp"],
        color="#1f4e79",
        alpha=0.88,
        label="Origin-income composition component",
    )
    ax1.axvline(0, color="#444", lw=1, ls="--")
    ax1.set_yticks(y)
    ax1.set_yticklabels(show["geo"])
    ax1.set_xlabel("Average model-implied contribution relative to sample mean (pp)")
    ax1.set_title("Country ranking in the `who arrives` model")
    ax1.grid(axis="x", alpha=0.2)
    ax1.legend(
        handles=[
            Patch(facecolor="#b22222", label="Net migration component"),
            Patch(facecolor="#1f4e79", label="Origin-income composition component"),
        ],
        fontsize=8,
        loc="lower right",
    )

    # Companion scatter: composition vs migration contribution, sized by sample n.
    sizes = 35 + 7 * np.sqrt(show["nobs"].clip(lower=1))
    sc = ax2.scatter(
        show["contrib_netmig_rel_pp"],
        show["contrib_origin_gdp_rel_pp"],
        s=sizes,
        c=show["avg_hpi_growth"],
        cmap="RdYlBu_r",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.95,
    )
    ax2.axhline(0, color="#666", lw=0.8, ls="--")
    ax2.axvline(0, color="#666", lw=0.8, ls="--")
    for _, r in show.iterrows():
        ax2.text(
            r["contrib_netmig_rel_pp"] + 0.01,
            r["contrib_origin_gdp_rel_pp"] + 0.01,
            str(r["geo"]),
            fontsize=8,
            alpha=0.85,
        )
    ax2.set_xlabel("Net-migration component (pp, rel. mean)")
    ax2.set_ylabel("Origin-income component (pp, rel. mean)")
    ax2.set_title("Decomposition across countries")
    ax2.grid(alpha=0.2)
    cbar = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Avg annual HPI growth (%)")

    fig.suptitle("Where the `who arrives` signal is strongest (descriptive, model-implied)", y=0.99, fontsize=12)
    fig.tight_layout()
    fig.savefig(PAPER_FIGS_DIR / "fig_country_who_arrives_contrib.pdf")
    plt.close(fig)


def _build_country_immigrant_type_effects(panel: pd.DataFrame, coef_df: pd.DataFrame, od: pd.DataFrame) -> pd.DataFrame:
    model = "annual_fe_netmig_plus_origin_groupshares"
    term_map = {
        "LATAM_CARIB": "L1_share_group_LATAM_CARIB",
        "EU_EEA_CH_UK": "L1_share_group_EU_EEA_CH_UK",
        "MENA": "L1_share_group_MENA",
    }
    beta = {}
    for grp, term in term_map.items():
        row = coef_df[(coef_df["model"] == model) & (coef_df["term"] == term)]
        if row.empty:
            continue
        beta[grp] = float(row.iloc[0]["coef"])
    if not beta:
        return pd.DataFrame()

    keep_cols = ["geo", "year"] + [f"L1_share_group_{g}" for g in beta.keys()]
    d = panel[[c for c in keep_cols if c in panel.columns]].copy()
    req = [f"L1_share_group_{g}" for g in beta.keys()]
    d = d.dropna(subset=req).copy()
    if d.empty:
        return pd.DataFrame()

    sample_means = {g: float(pd.to_numeric(d[f"L1_share_group_{g}"], errors="coerce").mean()) for g in beta.keys()}
    agg = d.groupby("geo", as_index=False).agg(**{f"avg_share_{g}": (f"L1_share_group_{g}", "mean") for g in beta.keys()})
    agg["nobs"] = d.groupby("geo", as_index=False)["year"].size()["size"]

    for g, b in beta.items():
        agg[f"contrib_{g}_pp"] = b * (agg[f"avg_share_{g}"] - sample_means[g])
    contrib_cols = [f"contrib_{g}_pp" for g in beta.keys()]
    agg["total_comp_effect_pp"] = agg[contrib_cols].sum(axis=1)
    agg["dominant_inflow_group"] = agg[[f"avg_share_{g}" for g in beta.keys()]].idxmax(axis=1).str.replace("avg_share_", "", regex=False)
    agg["dominant_positive_channel"] = agg[contrib_cols].idxmax(axis=1).str.replace("contrib_", "", regex=False).str.replace("_pp", "", regex=False)
    agg["dominant_negative_channel"] = agg[contrib_cols].idxmin(axis=1).str.replace("contrib_", "", regex=False).str.replace("_pp", "", regex=False)
    agg["direction"] = np.where(agg["total_comp_effect_pp"] >= 0, "Increase", "Decrease")

    # Add within-umbrella origin-country detail so umbrella channels are not over-aggregated.
    if od is not None and not od.empty:
        top = od[["geo", "origin", "year", "immigration_od"]].copy()
        top["year"] = pd.to_numeric(top["year"], errors="coerce")
        top["immigration_od"] = pd.to_numeric(top["immigration_od"], errors="coerce")
        top = top.dropna(subset=["geo", "origin", "year", "immigration_od"]).copy()
        top["year"] = top["year"].astype(int)
        top["origin_group"] = top["origin"].map(origin_group)
        top = top[top["origin_group"].isin(beta.keys())].copy()
        if not top.empty:
            y_max = int(top["year"].max())
            y_min = int(y_max - 6)
            top = top[top["year"] >= y_min].copy()
            grp = (
                top.groupby(["geo", "origin_group", "origin"], as_index=False)["immigration_od"]
                .sum(min_count=1)
                .rename(columns={"immigration_od": "flow_sum"})
            )
            grp_tot = grp.groupby(["geo", "origin_group"], as_index=False)["flow_sum"].sum().rename(columns={"flow_sum": "group_sum"})
            grp = grp.merge(grp_tot, on=["geo", "origin_group"], how="left")
            grp["share_in_group"] = np.where(grp["group_sum"] > 0, grp["flow_sum"] / grp["group_sum"], np.nan)
            grp = grp.sort_values(["geo", "origin_group", "flow_sum", "origin"], ascending=[True, True, False, True])
            top1 = grp.groupby(["geo", "origin_group"], as_index=False).head(1).copy()
            for g in beta.keys():
                x = top1[top1["origin_group"] == g][["geo", "origin", "share_in_group"]].copy()
                x = x.rename(
                    columns={
                        "origin": f"top_origin_{g}",
                        "share_in_group": f"top_origin_share_{g}",
                    }
                )
                agg = agg.merge(x, on="geo", how="left")
            agg["top_origin_window"] = f"{y_min}--{y_max}"
    agg = agg.sort_values("total_comp_effect_pp", ascending=False).reset_index(drop=True)
    return agg


def _plot_country_immigrant_type_heatmap(country_fx: pd.DataFrame) -> None:
    if country_fx.empty:
        return
    if not PAPER_FIGS_DIR.exists():
        return
    plot_cols = ["contrib_LATAM_CARIB_pp", "contrib_EU_EEA_CH_UK_pp", "contrib_MENA_pp", "total_comp_effect_pp"]
    use = country_fx[["geo"] + [c for c in plot_cols if c in country_fx.columns]].copy()
    if use.empty:
        return
    use = use.sort_values("total_comp_effect_pp").reset_index(drop=True)
    mat = use[[c for c in plot_cols if c in use.columns]].to_numpy(dtype=float)
    labels = {
        "contrib_LATAM_CARIB_pp": "Latin America/Caribbean",
        "contrib_EU_EEA_CH_UK_pp": "EU/EEA/CH/UK",
        "contrib_MENA_pp": "Middle East/North Africa",
        "total_comp_effect_pp": "Total composition",
    }
    x_labels = [labels[c] for c in use.columns if c != "geo"]

    vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
    vmax = max(vmax, 0.01)
    fig, ax = plt.subplots(figsize=(9.8, max(4.5, 0.26 * len(use))))
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(use)))
    ax.set_yticklabels(use["geo"].tolist(), fontsize=8)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_title("Country-level immigrant-type composition effects (relative pp, FE group-share model)")
    ax.set_xlabel("Immigrant-type channel")
    ax.set_ylabel("Country")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Relative contribution to annual HPI growth (pp)")
    fig.tight_layout()
    fig.savefig(PAPER_FIGS_DIR / "fig_country_immigrant_type_heatmap.pdf")
    plt.close(fig)


def _write_country_immigrant_type_table(country_fx: pd.DataFrame) -> None:
    if country_fx.empty:
        return
    if not PAPER_TABLES_DIR.exists():
        return
    show = country_fx.copy().sort_values("total_comp_effect_pp", ascending=False)
    if len(show) > 24:
        show = pd.concat([show.head(12), show.tail(12)], ignore_index=True)
        show = show.sort_values("total_comp_effect_pp", ascending=False).reset_index(drop=True)
    group_label = {
        "LATAM_CARIB": "Latin America/Caribbean",
        "EU_EEA_CH_UK": "EU/EEA/CH/UK",
        "MENA": "Middle East/North Africa",
    }

    window_label = ""
    if "top_origin_window" in show.columns:
        nonnull_window = show["top_origin_window"].dropna()
        if len(nonnull_window):
            window_label = str(nonnull_window.iloc[0])

    def _fmt_top(origin: Any, share: Any) -> str:
        if pd.isna(origin) or str(origin).strip() == "":
            return "--"
        if pd.isna(share):
            return str(origin)
        return f"{origin} ({100.0 * float(share):.1f}\\%)"

    lines = [
        r"\begin{landscape}",
        r"\begin{center}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{llrrrrp{12.2cm}}",
        r"\caption{Immigrant-type composition by country: model-implied channels with within-umbrella origin detail}\label{tab:country_immigrant_type_effects}\\",
        r"\toprule",
        r"Geo & Direction & Total comp. (pp) & LATAM/Carib (pp) & EU/EEA/CH/UK (pp) & MENA (pp) & Top origin within each umbrella (LATAM $|$ EU/EEA/CH/UK $|$ MENA) \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{7}{l}{\textit{Table \thetable\ (continued)}}\\",
        r"\toprule",
        r"Geo & Direction & Total comp. (pp) & LATAM/Carib (pp) & EU/EEA/CH/UK (pp) & MENA (pp) & Top origin within each umbrella (LATAM $|$ EU/EEA/CH/UK $|$ MENA) \\",
        r"\midrule",
        r"\endhead",
    ]
    for _, r in show.iterrows():
        top_detail = (
            _fmt_top(r.get("top_origin_LATAM_CARIB"), r.get("top_origin_share_LATAM_CARIB"))
            + " | "
            + _fmt_top(r.get("top_origin_EU_EEA_CH_UK"), r.get("top_origin_share_EU_EEA_CH_UK"))
            + " | "
            + _fmt_top(r.get("top_origin_MENA"), r.get("top_origin_share_MENA"))
        )
        lines.append(
            f"{r['geo']} & {r['direction']} & "
            f"{float(r.get('total_comp_effect_pp', np.nan)):.3f} & "
            f"{float(r.get('contrib_LATAM_CARIB_pp', np.nan)):.3f} & "
            f"{float(r.get('contrib_EU_EEA_CH_UK_pp', np.nan)):.3f} & "
            f"{float(r.get('contrib_MENA_pp', np.nan)):.3f} & "
            f"{top_detail} \\\\"
        )
    note_line_1 = (
        r"\multicolumn{7}{p{24.5cm}}{\footnotesize Notes: Entries use the annual FE model with lagged net migration plus lagged origin-group shares. "
        r"Values are country-average model-implied contributions relative to the composition-sample mean: positive values indicate composition states associated "
        r"with higher next-year house-price growth, negative values indicate composition states associated with lower next-year growth.}\\"
    )
    note_line_2 = (
        rf"\multicolumn{{7}}{{p{{24.5cm}}}}{{\footnotesize Within-umbrella top-origin detail is computed from cumulative Eurostat OD previous-residence inflows "
        rf"(\texttt{{MIGR\_IMM5PRV}}) over {window_label if window_label else 'the latest available 7-year window'} and reported as \texttt{{origin (share within umbrella)}}. "
        r"This directly separates countries inside each umbrella rather than treating umbrella channels as internally homogeneous.}\\"
    )
    note_line_3 = r"\multicolumn{7}{p{24.5cm}}{\footnotesize These are descriptive fitted decompositions, not causal country effects.}\\"
    lines += [
        r"\bottomrule",
        note_line_1,
        note_line_2,
        note_line_3,
        r"\end{longtable}",
        r"\end{center}",
        r"\end{landscape}",
    ]
    (PAPER_TABLES_DIR / "tab_country_immigrant_type_effects.tex").write_text("\n".join(lines))


def _write_paper_tables(coef_df: pd.DataFrame, case_outputs: dict[str, pd.DataFrame]) -> None:
    if not PAPER_TABLES_DIR.exists():
        return
    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Composition regression table
    models = [
        ("annual_fe_netmig_controls_matchedsample", "Matched FE"),
        ("annual_fe_netmig_plus_origin_gdpcomp", "+ GDP comp."),
        ("annual_fe_netmig_plus_origin_groupshares", "+ Groups"),
        ("annual_fe_netmig_plus_gdpcomp_groupshares", "+ GDP+Groups"),
    ]
    term_rows = [
        ("L1_net_migration_rate", "Lagged net migration rate"),
        ("L1_origin_gdp_pc_ppp_const_wavg_log", "Lagged weighted avg. origin GDPpc (log, PPP)"),
        ("L1_share_group_LATAM_CARIB", "Lagged share of inflows from Latin America/Caribbean"),
        ("L1_share_group_EU_EEA_CH_UK", "Lagged share of inflows from EU/EEA/CH/UK"),
        ("L1_share_group_MENA", "Lagged share of inflows from MENA"),
    ]
    col_spec = "p{5.0cm}" + "".join(["p{2.0cm}"] * len(models))
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Who arrives matters: origin-income composition and origin-group shares in country FE models}",
        r"\label{tab:who_arrives_composition}",
        r"\begin{threeparttable}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & " + " & ".join(lbl for _, lbl in models) + r" \\",
        r"\midrule",
    ]
    for term, label in term_rows:
        coef_line = [label]
        se_line = [""]
        for model, _ in models:
            coef_s, se_s = _coef_entry(coef_df, model, term)
            coef_line.append(coef_s)
            se_line.append(se_s)
        lines.append(" & ".join(coef_line) + r" \\")
        lines.append(" & ".join(se_line) + r" \\")
    lines.append(r"\midrule")
    for stat_name, label in [("Country FE", "Country FE"), ("Year FE", "Year FE"), ("Controls", "Macro + air controls"), ("Observations", "Observations")]:
        row = [label]
        for model, _ in models:
            if stat_name in {"Country FE", "Year FE"}:
                row.append("Yes" if not coef_df[coef_df["model"] == model].empty else "")
            elif stat_name == "Controls":
                row.append("Yes" if ("controls" in model or "plus_" in model) else "Yes")
            else:
                sub = coef_df[coef_df["model"] == model]
                row.append("" if sub.empty else f"{int(sub['nobs'].max())}")
        lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item Notes: Dependent variable is annual house-price growth. All models include country and year fixed effects and clustered standard errors (country and year in estimation; table reports coefficient estimates and standard errors). The matched baseline uses the same country-year sample as the origin-GDP composition model. All specifications include lagged macro controls and air-passenger controls. Origin-group shares are based on Eurostat origin-destination immigration inflows (\texttt{MIGR\_IMM5PRV}). Significance stars use conventional cutoffs (* $p<0.10$, ** $p<0.05$, *** $p<0.01$).",
        r"\item Exact p-values are taken directly from the audited coefficient CSV generated in the same run; stars are attached from those p-values.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    (PAPER_TABLES_DIR / "tab_who_arrives_composition.tex").write_text("\n".join(lines))

    # Spain-UK top origins table (recent period)
    top = case_outputs.get("top_case")
    if top is None or top.empty:
        return
    top_recent = (
        top.sort_values(["geo", "immig_total_rank_window"], ascending=[True, False])
        .groupby("geo", as_index=False)
        .head(8)
        .copy()
    )
    top_recent["rank"] = top_recent.groupby("geo").cumcount() + 1
    pivot = top_recent.pivot(index="rank", columns="geo", values=["origin", "origin_group", "immig_total_rank_window"])

    def cell(rank: int, geo: str) -> str:
        try:
            o = pivot.loc[rank, ("origin", geo)]
            g = pivot.loc[rank, ("origin_group", geo)]
            v = pivot.loc[rank, ("immig_total_rank_window", geo)]
        except Exception:
            return ""
        if pd.isna(o):
            return ""
        return f"{o} ({g.replace('_',' ')}, {int(v):,})"

    top_recent = top_recent.copy()
    if "rank_window_label" not in top_recent.columns:
        top_recent["rank_window_label"] = top_recent["geo"].map(lambda g: f"{CASE_RANK_WINDOWS.get(str(g), ('2018','2024'))[0]}--{CASE_RANK_WINDOWS.get(str(g), ('2018','2024'))[1]}")

    geos_present = [g for g in CASE_DESTS if g in set(top_recent["geo"].astype(str))]
    if "ES" not in geos_present and "ES" in set(top_recent["geo"].astype(str)):
        geos_present = ["ES"] + [g for g in geos_present if g != "ES"]

    def header_for_geo(geo: str) -> str:
        label = "Spain" if geo == "ES" else "United Kingdom" if geo == "UK" else geo
        sub = top_recent[top_recent["geo"] == geo]
        win = ""
        if not sub.empty and "rank_window_label" in sub.columns:
            w = sub["rank_window_label"].dropna().astype(str)
            if not w.empty:
                win = w.mode().iat[0]
        win = win or "2018--2024"
        return f"{label} ({geo}, OD {win})"

    def cell_geo(rank: int, geo: str) -> str:
        return cell(rank, geo)

    col_specs = " ".join(["p{4.1cm}"] * max(1, len(geos_present)))
    header_cells = " & ".join(header_for_geo(g) for g in geos_present)
    lines2 = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Spain and United Kingdom: top country composition entries for immigration}",
        r"\label{tab:iberia_top_origins}",
        r"\begin{threeparttable}",
        r"\small",
        rf"\begin{{tabular}}{{c {col_specs}}}",
        r"\toprule",
        "Rank & " + header_cells + r" \\",
        r"\midrule",
    ]
    geo_source = {}
    if "source" in top.columns:
        geo_source = (
            top.dropna(subset=["geo", "source"])
            .groupby("geo", as_index=False)["source"]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
            .set_index("geo")["source"]
            .to_dict()
        )
    for rank in range(1, 9):
        row_cells = []
        for geo in geos_present:
            gc = cell_geo(rank, geo)
            src_tag = geo_source.get(geo)
            if src_tag == "ctz_proxy" and gc:
                gc = gc + r" [ctz]"
            elif src_tag == "ons_uk_group_scaled" and gc:
                gc = gc + r" [ons]"
            elif src_tag == "multi_source_blend" and gc:
                gc = gc + r" [blend]"
            if (not gc) and rank == 1:
                gc = rf"\emph{{No country-level {geo} entries available in current sources}}"
            row_cells.append(gc)
        lines2.append(str(rank) + " & " + " & ".join(row_cells) + r"\\")
    lines2 += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item Notes: Entries show origin-country code (origin group, cumulative immigration counts in the country-specific OD ranking window shown in the header). Baseline source is Eurostat \texttt{MIGR\_IMM5PRV} (previous residence, OD). Cells marked [ctz] use citizenship-based proxy flows (\texttt{MIGR\_IMM1CTZ}); [ons] marks UK ONS YE-June EU+/non-EU immigration bridge scaling; [blend] indicates multi-source year-level blending. Ranking windows are country-specific and data-availability constrained. This table is descriptive and is included to visualize composition differences, not to establish causal country-by-country rankings.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    (PAPER_TABLES_DIR / "tab_iberia_top_origins.tex").write_text("\n".join(lines2))


def write_paper_assets(panel: pd.DataFrame, od: pd.DataFrame, coef_df: pd.DataFrame) -> None:
    if not PAPER_DIR.exists():
        return
    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGS_DIR.mkdir(parents=True, exist_ok=True)
    case_outputs = write_case_outputs(panel, od)
    _plot_composition_coefficients(coef_df)
    _plot_case_origin_mix(case_outputs)
    _plot_country_who_arrives_contributions(panel, coef_df)
    country_fx = _build_country_immigrant_type_effects(panel, coef_df, od)
    if not country_fx.empty:
        country_fx.to_csv(RESULTS_DIR / "country_immigrant_type_effects.csv", index=False)
        _plot_country_immigrant_type_heatmap(country_fx)
        _write_country_immigrant_type_table(country_fx)
    _write_paper_tables(coef_df, case_outputs)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    panel, meta, od = build_migration_composition_panel()
    coef_df, texts, stats_df = run_models(panel)

    coef_path = RESULTS_DIR / "migration_composition_coefficients.csv"
    txt_path = RESULTS_DIR / "migration_composition_summaries.txt"
    stats_path = RESULTS_DIR / "migration_composition_sample_stats.csv"
    meta_path = META_DIR / "migration_composition_summary.json"

    coef_df.to_csv(coef_path, index=False)
    txt_path.write_text("\n\n".join(texts))
    stats_df.to_csv(stats_path, index=False)

    # Add simple model-level sample summary for quick inspection.
    if not coef_df.empty:
        model_n = coef_df.groupby("model", as_index=False)["nobs"].max().rename(columns={"nobs": "nobs_model"})
        meta["models"] = model_n.to_dict(orient="records")
    meta_path.write_text(json.dumps(meta, indent=2))

    # If the paper package exists in the repo, refresh paper-ready tables/figures for the new section.
    write_paper_assets(panel, od, coef_df)

    print(f"[ok] wrote {coef_path}")
    print(f"[ok] wrote {txt_path}")
    print(f"[ok] wrote {stats_path}")
    print(f"[ok] wrote {meta_path}")


if __name__ == "__main__":
    main()
    group_label = {
        "LATAM_CARIB": "Latin America/Caribbean",
        "EU_EEA_CH_UK": "EU/EEA/CH/UK",
        "MENA": "Middle East/North Africa",
    }
