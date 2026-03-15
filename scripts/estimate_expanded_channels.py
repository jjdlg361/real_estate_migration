#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import eurostat
from linearmodels.panel import PanelOLS

from build_shiftshare_iv import COUNTRY2_RE, TARGET_GEOS, WB2_TO_EUROSTAT, load_od_migration

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"linearmodels(\..*)?")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
META_DIR = ROOT / "data" / "metadata"
PAPER_DIR = ROOT / "paper_overleaf"
PAPER_TABLES_DIR = PAPER_DIR / "tables"
PAPER_FIGS_DIR = PAPER_DIR / "figures"

M_COL_RE = re.compile(r"^\d{4}-\d{2}$")
A_COL_RE = re.compile(r"^\d{4}$")

RAW_TOUR_NIGHTS = RAW_DIR / "tour_occ_nim_m_nights.csv"
RAW_ASY_M = RAW_DIR / "migr_asyappctzm_m_first_total.csv"
RAW_TPS_M = RAW_DIR / "migr_asytpsm_m_total_ua.csv"
RAW_ACQ_TOTAL = RAW_DIR / "migr_acq_a_total.csv"
RAW_ACQ_BY_ORIGIN = RAW_DIR / "migr_acq_a_by_former_citizenship.csv"
RAW_CAR_REG = RAW_DIR / "road_eqr_carpda_a.csv"
RAW_WB_EXTRA = RAW_DIR / "worldbank_remittance_long.csv"

WB_GDP_LEVEL_FILE = RAW_DIR / "worldbank_origin_gdp_pc_ppp_const_long.csv"
WB_GDP_LEVEL_INDICATOR = "NY.GDP.PCAP.PP.KD"

WB_EXTRA_INDICATORS = {
    "BX.TRF.PWKR.DT.GD.ZS": "wb_remit_in_pct_gdp",
    "NY.GDP.PCAP.KD.ZG": "wb_gdp_pc_growth",
    "SP.POP.TOTL": "wb_population",
    "SL.UEM.TOTL.ZS": "wb_unemployment",
}


def _safe_geo_name(df: pd.DataFrame) -> str:
    geo_cols = [c for c in df.columns if "\\TIME_PERIOD" in str(c)]
    if len(geo_cols) != 1:
        raise ValueError(f"Expected one geo/time column, got {geo_cols}")
    return geo_cols[0]


def _wide_to_long(df: pd.DataFrame, col_pattern: re.Pattern, period_name: str, value_name: str) -> pd.DataFrame:
    geo_col = _safe_geo_name(df)
    time_cols = [c for c in df.columns if col_pattern.match(str(c))]
    id_cols = [c for c in df.columns if c not in time_cols]
    out = df.melt(id_vars=id_cols, value_vars=time_cols, var_name=period_name, value_name=value_name)
    out = out.rename(columns={geo_col: "geo"})
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    return out


def winsorize_series(s: pd.Series, p_low: float = 0.01, p_high: float = 0.99) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    lo, hi = x.quantile([p_low, p_high])
    return x.clip(lo, hi)


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
                    "indicator": indicator,
                    "country_code_wb2": country.get("id"),
                    "country_name": country.get("value"),
                    "country_iso3": item.get("countryiso3code"),
                    "year": pd.to_numeric(item.get("date"), errors="coerce"),
                    "value": pd.to_numeric(item.get("value"), errors="coerce"),
                }
            )
        if page >= int(meta["pages"]):
            break
        page += 1
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    return df


def fetch_or_load_tourism_nights() -> pd.DataFrame:
    if RAW_TOUR_NIGHTS.exists():
        return pd.read_csv(RAW_TOUR_NIGHTS)
    df = eurostat.get_data_df(
        "TOUR_OCC_NIM",
        filter_pars={"freq": "M", "unit": "NR", "c_resid": ["TOTAL", "FOR"], "nace_r2": "I551-I553"},
    )
    RAW_TOUR_NIGHTS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_TOUR_NIGHTS, index=False)
    return df


def fetch_or_load_asylum_monthly() -> pd.DataFrame:
    if RAW_ASY_M.exists():
        return pd.read_csv(RAW_ASY_M)
    df = eurostat.get_data_df(
        "MIGR_ASYAPPCTZM",
        filter_pars={
            "freq": "M",
            "unit": "PER",
            "citizen": "TOTAL",
            "sex": "T",
            "applicant": "FRST",
            "age": "TOTAL",
        },
    )
    RAW_ASY_M.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_ASY_M, index=False)
    return df


def fetch_or_load_tps_monthly() -> pd.DataFrame:
    if RAW_TPS_M.exists():
        return pd.read_csv(RAW_TPS_M)
    df = eurostat.get_data_df(
        "MIGR_ASYTPSM",
        filter_pars={"freq": "M", "unit": "PER", "sex": "T", "age": "TOTAL", "citizen": ["TOTAL", "UA"]},
    )
    RAW_TPS_M.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_TPS_M, index=False)
    return df


def fetch_or_load_naturalization_total() -> pd.DataFrame:
    if RAW_ACQ_TOTAL.exists():
        return pd.read_csv(RAW_ACQ_TOTAL)
    df = eurostat.get_data_df(
        "MIGR_ACQ",
        filter_pars={"freq": "A", "citizen": "TOTAL", "agedef": "REACH", "age": "TOTAL", "unit": "NR", "sex": "T"},
    )
    RAW_ACQ_TOTAL.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_ACQ_TOTAL, index=False)
    return df


def fetch_or_load_naturalization_by_origin() -> pd.DataFrame:
    if RAW_ACQ_BY_ORIGIN.exists():
        return pd.read_csv(RAW_ACQ_BY_ORIGIN)
    df = eurostat.get_data_df(
        "MIGR_ACQ",
        filter_pars={"freq": "A", "agedef": "REACH", "age": "TOTAL", "unit": "NR", "sex": "T"},
    )
    RAW_ACQ_BY_ORIGIN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_ACQ_BY_ORIGIN, index=False)
    return df


def fetch_or_load_car_registrations() -> pd.DataFrame:
    if RAW_CAR_REG.exists():
        return pd.read_csv(RAW_CAR_REG)
    df = eurostat.get_data_df(
        "ROAD_EQR_CARPDA",
        filter_pars={"freq": "A", "unit": "NR", "mot_nrg": ["TOTAL", "ELC"], "geo": sorted(TARGET_GEOS)},
    )
    RAW_CAR_REG.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CAR_REG, index=False)
    return df


def fetch_or_load_wb_extra() -> pd.DataFrame:
    if RAW_WB_EXTRA.exists():
        return pd.read_csv(RAW_WB_EXTRA)
    frames = []
    for ind, alias in WB_EXTRA_INDICATORS.items():
        df = fetch_world_bank_indicator(ind)
        df["indicator_alias"] = alias
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    RAW_WB_EXTRA.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(RAW_WB_EXTRA, index=False)
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
    keep = ["origin", "year", "wb_gdp_pc_ppp_const"]
    return wb[keep].drop_duplicates(["origin", "year"]).sort_values(["origin", "year"]).reset_index(drop=True)


def build_tourism_quarterly() -> pd.DataFrame:
    raw = fetch_or_load_tourism_nights()
    d = _wide_to_long(raw, M_COL_RE, "month_str", "tour_nights")
    d = d[(d["unit"] == "NR") & (d["nace_r2"] == "I551-I553")].copy()
    d = d[d["geo"].astype(str).isin(TARGET_GEOS)].copy()
    d["month"] = pd.PeriodIndex(d["month_str"].astype(str), freq="M")
    d["quarter"] = d["month"].dt.asfreq("Q")
    q = (
        d.groupby(["geo", "c_resid", "quarter"], as_index=False)["tour_nights"]
        .sum(min_count=1)
    )
    q["period_str"] = q["quarter"].astype(str)
    q_w = q.pivot_table(index=["geo", "period_str", "quarter"], columns="c_resid", values="tour_nights", aggfunc="first").reset_index()
    q_w.columns = [f"tour_nights_{c.lower()}_q" if c not in {"geo", "period_str", "quarter"} else c for c in q_w.columns]
    for c in ["tour_nights_total_q", "tour_nights_for_q"]:
        if c not in q_w.columns:
            q_w[c] = np.nan
    q_w = q_w.sort_values(["geo", "quarter"]).reset_index(drop=True)
    q_w["tour_total_yoy"] = q_w.groupby("geo")["tour_nights_total_q"].transform(lambda s: np.log(s.where(s > 0)).diff(4) * 100.0)
    q_w["tour_for_yoy"] = q_w.groupby("geo")["tour_nights_for_q"].transform(lambda s: np.log(s.where(s > 0)).diff(4) * 100.0)
    return q_w


def build_asylum_quarterly() -> pd.DataFrame:
    asy_raw = fetch_or_load_asylum_monthly()
    asy = _wide_to_long(asy_raw, M_COL_RE, "month_str", "asylum_first_apps_m")
    asy = asy[(asy["unit"] == "PER") & (asy["sex"] == "T") & (asy["age"] == "TOTAL") & (asy["applicant"] == "FRST") & (asy["citizen"] == "TOTAL")].copy()
    asy = asy[asy["geo"].astype(str).isin(TARGET_GEOS)].copy()
    asy["month"] = pd.PeriodIndex(asy["month_str"].astype(str), freq="M")
    asy["quarter"] = asy["month"].dt.asfreq("Q")
    asy_q = (
        asy.groupby(["geo", "quarter"], as_index=False)["asylum_first_apps_m"]
        .sum(min_count=1)
        .rename(columns={"asylum_first_apps_m": "asylum_first_apps_q"})
    )
    asy_q["period_str"] = asy_q["quarter"].astype(str)
    asy_q = asy_q.sort_values(["geo", "quarter"]).reset_index(drop=True)
    asy_q["asylum_first_apps_yoy_q"] = asy_q.groupby("geo")["asylum_first_apps_q"].transform(lambda s: np.log1p(s).diff(4) * 100.0)

    tps_raw = fetch_or_load_tps_monthly()
    tps = _wide_to_long(tps_raw, M_COL_RE, "month_str", "tps_stock_m")
    tps = tps[(tps["unit"] == "PER") & (tps["sex"] == "T") & (tps["age"] == "TOTAL")].copy()
    tps = tps[tps["geo"].astype(str).isin(TARGET_GEOS)].copy()
    tps = tps[tps["citizen"].isin(["TOTAL", "UA"])].copy()
    tps["month"] = pd.PeriodIndex(tps["month_str"].astype(str), freq="M")
    tps["quarter"] = tps["month"].dt.asfreq("Q")
    # End-of-quarter stock proxy.
    tps = tps.sort_values(["geo", "citizen", "month"]).reset_index(drop=True)
    tps_q = tps.groupby(["geo", "citizen", "quarter"], as_index=False).tail(1).copy()
    tps_q["period_str"] = tps_q["quarter"].astype(str)
    tps_q = tps_q[["geo", "period_str", "quarter", "citizen", "tps_stock_m"]].copy()
    tps_w = tps_q.pivot_table(index=["geo", "period_str", "quarter"], columns="citizen", values="tps_stock_m", aggfunc="first").reset_index()
    tps_w.columns = [f"tps_stock_{c.lower()}_qe" if c not in {"geo", "period_str", "quarter"} else c for c in tps_w.columns]
    for c in ["tps_stock_total_qe", "tps_stock_ua_qe"]:
        if c not in tps_w.columns:
            tps_w[c] = np.nan
    tps_w = tps_w.sort_values(["geo", "quarter"]).reset_index(drop=True)
    tps_w["tps_stock_ua_yoy_q"] = tps_w.groupby("geo")["tps_stock_ua_qe"].transform(lambda s: np.log1p(s).diff(4) * 100.0)

    out = asy_q.merge(tps_w.drop(columns=["quarter"], errors="ignore"), on=["geo", "period_str"], how="outer")
    return out


def build_annual_type_features(base_annual: pd.DataFrame) -> pd.DataFrame:
    annual = base_annual.copy()

    # Asylum annual from monthly first-time apps.
    asy_q = build_asylum_quarterly()
    asy_y = asy_q.copy()
    asy_y["year"] = pd.PeriodIndex(asy_y["period_str"], freq="Q").year
    asy_y = asy_y.groupby(["geo", "year"], as_index=False)["asylum_first_apps_q"].sum(min_count=1)
    annual = annual.merge(asy_y, on=["geo", "year"], how="left")
    annual["asylum_rate_per_1000"] = np.where(annual["population"] > 0, annual["asylum_first_apps_q"] / annual["population"] * 1000.0, np.nan)
    annual["non_asylum_immigration_rate_per_1000"] = annual["immigration_rate_per_1000"] - annual["asylum_rate_per_1000"]

    # Citizenship acquisition (total + composition by former citizenship GDPpc).
    acq_tot_raw = fetch_or_load_naturalization_total()
    acq_tot = _wide_to_long(acq_tot_raw, A_COL_RE, "year", "acq_total")
    acq_tot = acq_tot[(acq_tot["agedef"] == "REACH") & (acq_tot["age"] == "TOTAL") & (acq_tot["unit"] == "NR") & (acq_tot["sex"] == "T") & (acq_tot["citizen"] == "TOTAL")].copy()
    acq_tot["year"] = pd.to_numeric(acq_tot["year"], errors="coerce").astype("Int64")
    acq_tot = acq_tot.dropna(subset=["year"]).copy()
    acq_tot["year"] = acq_tot["year"].astype(int)
    acq_tot = acq_tot[["geo", "year", "acq_total"]]

    annual = annual.merge(acq_tot, on=["geo", "year"], how="left")
    annual["naturalization_rate_per_1000"] = np.where(annual["population"] > 0, annual["acq_total"] / annual["population"] * 1000.0, np.nan)

    acq_origin_raw = fetch_or_load_naturalization_by_origin()
    acq_o = _wide_to_long(acq_origin_raw, A_COL_RE, "year", "acq_flow")
    acq_o = acq_o[(acq_o["agedef"] == "REACH") & (acq_o["age"] == "TOTAL") & (acq_o["unit"] == "NR") & (acq_o["sex"] == "T")].copy()
    acq_o = acq_o[acq_o["geo"].astype(str).isin(TARGET_GEOS)].copy()
    acq_o["origin"] = acq_o["citizen"].astype(str)
    acq_o = acq_o[acq_o["origin"].str.match(COUNTRY2_RE, na=False)].copy()
    acq_o = acq_o[acq_o["origin"] != acq_o["geo"]].copy()
    acq_o["year"] = pd.to_numeric(acq_o["year"], errors="coerce")
    acq_o = acq_o.dropna(subset=["year", "acq_flow"]).copy()
    acq_o["year"] = acq_o["year"].astype(int)

    wb_gdp = load_origin_gdp_levels()
    acq_o = acq_o.merge(wb_gdp, on=["origin", "year"], how="left")
    acq_o["w"] = pd.to_numeric(acq_o["acq_flow"], errors="coerce")
    acq_o["w_log_gdp"] = np.where(acq_o["wb_gdp_pc_ppp_const"] > 0, acq_o["w"] * np.log(acq_o["wb_gdp_pc_ppp_const"]), np.nan)
    acq_o["w_cov"] = np.where(acq_o["wb_gdp_pc_ppp_const"].notna(), acq_o["w"], 0.0)
    acq_comp = (
        acq_o.groupby(["geo", "year"], as_index=False)
        .agg(acq_gdp_wsum=("w_log_gdp", "sum"), acq_gdp_cov=("w_cov", "sum"))
    )
    acq_comp["acq_origin_gdp_wavg_log"] = np.where(acq_comp["acq_gdp_cov"] > 0, acq_comp["acq_gdp_wsum"] / acq_comp["acq_gdp_cov"], np.nan)
    annual = annual.merge(acq_comp[["geo", "year", "acq_origin_gdp_wavg_log"]], on=["geo", "year"], how="left")

    # Car registrations.
    car_raw = fetch_or_load_car_registrations()
    car = _wide_to_long(car_raw, A_COL_RE, "year", "car_reg")
    car = car[(car["unit"] == "NR") & (car["mot_nrg"].isin(["TOTAL", "ELC"]))].copy()
    car["year"] = pd.to_numeric(car["year"], errors="coerce").astype("Int64")
    car = car.dropna(subset=["year"]).copy()
    car["year"] = car["year"].astype(int)
    car = car[car["geo"].astype(str).isin(TARGET_GEOS)].copy()
    car_w = car.pivot_table(index=["geo", "year"], columns="mot_nrg", values="car_reg", aggfunc="first").reset_index()
    car_w.columns = [f"car_reg_{c.lower()}" if c not in {"geo", "year"} else c for c in car_w.columns]
    if "car_reg_total" not in car_w.columns:
        car_w["car_reg_total"] = np.nan
    if "car_reg_elc" not in car_w.columns:
        car_w["car_reg_elc"] = np.nan
    car_w["car_reg_elc_share"] = np.where(car_w["car_reg_total"] > 0, car_w["car_reg_elc"] / car_w["car_reg_total"], np.nan)
    annual = annual.merge(car_w, on=["geo", "year"], how="left")
    annual["car_reg_total_per_1000"] = np.where(annual["population"] > 0, annual["car_reg_total"] / annual["population"] * 1000.0, np.nan)

    annual = annual.sort_values(["geo", "year"]).reset_index(drop=True)
    g = annual.groupby("geo", sort=False)
    lag_cols = [
        "asylum_rate_per_1000",
        "non_asylum_immigration_rate_per_1000",
        "naturalization_rate_per_1000",
        "acq_origin_gdp_wavg_log",
        "car_reg_total_per_1000",
        "car_reg_elc_share",
    ]
    for c in lag_cols:
        annual[f"L1_{c}"] = g[c].shift(1)
    return annual


def build_quarterly_extended_panel() -> pd.DataFrame:
    base = pd.read_parquet(PROC_DIR / "panel_quarterly_traveler_quality.parquet")
    base = base.replace([np.inf, -np.inf], np.nan).copy()
    if "period_str" not in base.columns:
        base["period_str"] = base["period"].astype(str)

    tour_q = build_tourism_quarterly()
    asy_q = build_asylum_quarterly()
    panel = base.merge(tour_q.drop(columns=["quarter"], errors="ignore"), on=["geo", "period_str"], how="left")
    panel = panel.merge(asy_q.drop(columns=["quarter"], errors="ignore"), on=["geo", "period_str"], how="left")

    # Annual population for per-capita scaling (harmonized-first).
    annual_pop = pd.read_parquet(PROC_DIR / "panel_annual_harmonized.parquet").copy()
    pop = pd.to_numeric(annual_pop["population"], errors="coerce") if "population" in annual_pop.columns else pd.Series(np.nan, index=annual_pop.index)
    if "population_harmonized" in annual_pop.columns:
        pop = pop.combine_first(pd.to_numeric(annual_pop["population_harmonized"], errors="coerce"))
    annual_pop["population"] = pop
    annual_pop = annual_pop[["geo", "year", "population"]].drop_duplicates(["geo", "year"])
    q = pd.PeriodIndex(panel["period_str"], freq="Q")
    panel["year"] = q.year
    panel = panel.merge(annual_pop, on=["geo", "year"], how="left")

    panel["asylum_first_q_per100k"] = np.where(panel["population"] > 0, panel["asylum_first_apps_q"] / panel["population"] * 100000.0, np.nan)
    panel["tps_ua_stock_qe_per100k"] = np.where(panel["population"] > 0, panel["tps_stock_ua_qe"] / panel["population"] * 100000.0, np.nan)

    panel = panel.sort_values(["geo", "period_str"]).reset_index(drop=True)
    g = panel.groupby("geo", sort=False)
    lag_cols = [
        "tour_total_yoy",
        "tour_for_yoy",
        "asylum_first_q_per100k",
        "asylum_first_apps_yoy_q",
        "tps_ua_stock_qe_per100k",
        "tps_stock_ua_yoy_q",
    ]
    for c in lag_cols:
        panel[f"L1_{c}"] = g[c].shift(1)
    return panel


def build_origin_remittance_panel() -> pd.DataFrame:
    od = load_od_migration().copy()
    outflow = od.groupby(["origin", "year"], as_index=False)["immigration_od"].sum(min_count=1).rename(columns={"immigration_od": "outflow_to_europe"})

    wb = fetch_or_load_wb_extra().copy()
    wb["year"] = pd.to_numeric(wb["year"], errors="coerce")
    wb["value"] = pd.to_numeric(wb["value"], errors="coerce")
    wb = wb.dropna(subset=["year"]).copy()
    wb["year"] = wb["year"].astype(int)
    wb["origin"] = wb["country_code_wb2"].astype(str).map(WB2_TO_EUROSTAT).fillna(wb["country_code_wb2"].astype(str))
    wb = wb[wb["origin"].str.match(COUNTRY2_RE, na=False)].copy()

    wb_w = wb.pivot_table(index=["origin", "year"], columns="indicator_alias", values="value", aggfunc="first").reset_index()
    rem = outflow.merge(wb_w, on=["origin", "year"], how="left")
    rem["outflow_rate_per_1000"] = np.where(rem["wb_population"] > 0, rem["outflow_to_europe"] / rem["wb_population"] * 1000.0, np.nan)

    rem = rem.sort_values(["origin", "year"]).reset_index(drop=True)
    g = rem.groupby("origin", sort=False)
    for c in ["outflow_rate_per_1000", "wb_remit_in_pct_gdp", "wb_unemployment", "wb_gdp_pc_growth"]:
        rem[f"L1_{c}"] = g[c].shift(1)

    rem["L1_outflow_x_remit_in"] = rem["L1_outflow_rate_per_1000"] * rem["L1_wb_remit_in_pct_gdp"]
    return rem


def _fit_panel(df_panel: pd.DataFrame, formula: str, label: str, needed: list[str], entity: str, time: str) -> tuple[pd.DataFrame, str] | None:
    d = df_panel.dropna(subset=needed).copy()
    if d.empty:
        return None
    d = d.set_index([entity, time]).sort_index()
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
            "entity_dim": entity,
            "time_dim": time,
        }
    )
    txt = f"# {label}\nFormula: {formula}\n\n{res.summary}"
    return coef, txt


def estimate_annual_models(annual: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = annual.copy()
    for c in [
        "hpi_growth",
        "L1_net_migration_rate",
        "L1_asylum_rate_per_1000",
        "L1_non_asylum_immigration_rate_per_1000",
        "L1_naturalization_rate_per_1000",
        "L1_acq_origin_gdp_wavg_log",
        "L1_car_reg_total_per_1000",
    ]:
        if c in d.columns:
            d[c] = winsorize_series(d[c], 0.01, 0.99)

    controls = ["L1_air_growth", "L1_gdp_pc_growth", "L1_unemployment_rate", "L1_inflation_hicp", "L1_long_rate", "L1_pop_growth"]

    specs = [
        (
            "annual_ext_baseline",
            "hpi_growth ~ 1 + L1_net_migration_rate + " + " + ".join(controls) + " + EntityEffects + TimeEffects",
            ["hpi_growth", "L1_net_migration_rate"] + controls,
        ),
        (
            "annual_ext_mig_types",
            "hpi_growth ~ 1 + L1_non_asylum_immigration_rate_per_1000 + L1_asylum_rate_per_1000 + L1_naturalization_rate_per_1000 + " + " + ".join(controls) + " + EntityEffects + TimeEffects",
            ["hpi_growth", "L1_non_asylum_immigration_rate_per_1000", "L1_asylum_rate_per_1000", "L1_naturalization_rate_per_1000"] + controls,
        ),
        (
            "annual_ext_mig_types_cars",
            "hpi_growth ~ 1 + L1_non_asylum_immigration_rate_per_1000 + L1_asylum_rate_per_1000 + L1_naturalization_rate_per_1000 + L1_acq_origin_gdp_wavg_log + L1_car_reg_total_per_1000 + L1_car_reg_elc_share + " + " + ".join(controls) + " + EntityEffects + TimeEffects",
            ["hpi_growth", "L1_non_asylum_immigration_rate_per_1000", "L1_asylum_rate_per_1000", "L1_naturalization_rate_per_1000", "L1_acq_origin_gdp_wavg_log", "L1_car_reg_total_per_1000", "L1_car_reg_elc_share"] + controls,
        ),
    ]

    coefs = []
    texts = []
    for label, formula, needed in specs:
        out = _fit_panel(d, formula, label, needed, "geo", "year")
        if out is None:
            continue
        c, t = out
        coefs.append(c)
        texts.append(t)
    return (pd.concat(coefs, ignore_index=True) if coefs else pd.DataFrame()), texts


def estimate_quarterly_models(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = panel.copy()
    d["quarter_idx"] = pd.PeriodIndex(d["period_str"], freq="Q").to_timestamp(how="end")
    for c in [
        "hpi_yoy",
        "L1_air_yoy",
        "L1_tour_total_yoy",
        "L1_asylum_first_q_per100k",
        "L1_tps_ua_stock_qe_per100k",
        "L1_pax_per_move_total",
    ]:
        if c in d.columns:
            d[c] = winsorize_series(d[c], 0.01, 0.99)

    specs = [
        (
            "quarterly_ext_air_only",
            "hpi_yoy ~ 1 + L1_air_yoy + EntityEffects + TimeEffects",
            ["hpi_yoy", "L1_air_yoy"],
        ),
        (
            "quarterly_ext_nonair_mobility",
            "hpi_yoy ~ 1 + L1_air_yoy + L1_tour_total_yoy + L1_asylum_first_q_per100k + L1_tps_ua_stock_qe_per100k + EntityEffects + TimeEffects",
            ["hpi_yoy", "L1_air_yoy", "L1_tour_total_yoy", "L1_asylum_first_q_per100k", "L1_tps_ua_stock_qe_per100k"],
        ),
        (
            "quarterly_ext_full",
            "hpi_yoy ~ 1 + L1_air_yoy + L1_tour_total_yoy + L1_asylum_first_q_per100k + L1_tps_ua_stock_qe_per100k + L1_pax_per_move_total + L1_airfare_yoy_q + EntityEffects + TimeEffects",
            ["hpi_yoy", "L1_air_yoy", "L1_tour_total_yoy", "L1_asylum_first_q_per100k", "L1_tps_ua_stock_qe_per100k", "L1_pax_per_move_total", "L1_airfare_yoy_q"],
        ),
    ]

    coefs = []
    texts = []
    for label, formula, needed in specs:
        out = _fit_panel(d, formula, label, needed, "geo", "quarter_idx")
        if out is None:
            continue
        c, t = out
        coefs.append(c)
        texts.append(t)
    return (pd.concat(coefs, ignore_index=True) if coefs else pd.DataFrame()), texts


def estimate_origin_remittance_models(rem: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = rem.copy()
    for c in ["wb_gdp_pc_growth", "L1_outflow_rate_per_1000", "L1_wb_remit_in_pct_gdp", "L1_outflow_x_remit_in"]:
        if c in d.columns:
            d[c] = winsorize_series(d[c], 0.01, 0.99)

    specs = [
        (
            "origin_remit_baseline",
            "wb_gdp_pc_growth ~ 1 + L1_outflow_rate_per_1000 + L1_wb_remit_in_pct_gdp + EntityEffects + TimeEffects",
            ["wb_gdp_pc_growth", "L1_outflow_rate_per_1000", "L1_wb_remit_in_pct_gdp"],
        ),
        (
            "origin_remit_interaction",
            "wb_gdp_pc_growth ~ 1 + L1_outflow_rate_per_1000 + L1_wb_remit_in_pct_gdp + L1_outflow_x_remit_in + L1_wb_unemployment + EntityEffects + TimeEffects",
            ["wb_gdp_pc_growth", "L1_outflow_rate_per_1000", "L1_wb_remit_in_pct_gdp", "L1_outflow_x_remit_in", "L1_wb_unemployment"],
        ),
    ]

    coefs = []
    texts = []
    for label, formula, needed in specs:
        out = _fit_panel(d, formula, label, needed, "origin", "year")
        if out is None:
            continue
        c, t = out
        coefs.append(c)
        texts.append(t)
    return (pd.concat(coefs, ignore_index=True) if coefs else pd.DataFrame()), texts


def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def _coef_entry(df: pd.DataFrame, model: str, term: str) -> tuple[str, str]:
    r = df[(df["model"] == model) & (df["term"] == term)]
    if r.empty:
        return "", ""
    x = r.iloc[0]
    return f"{x['coef']:.3f}{_stars(float(x['p_value']))}", f"({x['std_err']:.3f})"


def write_latex_tables(coef_df: pd.DataFrame) -> None:
    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Annual extended channels.
    models = ["annual_ext_baseline", "annual_ext_mig_types", "annual_ext_mig_types_cars"]
    headers = ["Baseline", "+ Types", "+ Types+Cars"]
    rows = [
        ("Lagged net migration rate", "L1_net_migration_rate"),
        ("Lagged non-asylum immigration rate per 1,000", "L1_non_asylum_immigration_rate_per_1000"),
        ("Lagged asylum first-applicant rate per 1,000", "L1_asylum_rate_per_1000"),
        ("Lagged naturalization rate per 1,000", "L1_naturalization_rate_per_1000"),
        ("Lagged naturalization origin GDPpc (log)", "L1_acq_origin_gdp_wavg_log"),
        ("Lagged car registrations per 1,000", "L1_car_reg_total_per_1000"),
        ("Lagged EV share in new car registrations", "L1_car_reg_elc_share"),
    ]
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Annual migration-type decomposition with naturalization and car-market controls}",
        r"\label{tab:extended_annual}",
        r"\begin{threeparttable}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{p{6.0cm}ccc}",
        r"\toprule",
        " & " + " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for label, term in rows:
        coef_row, se_row = [label], [""]
        for m in models:
            c, s = _coef_entry(coef_df, m, term)
            coef_row.append(c)
            se_row.append(s)
        lines.append(" & ".join(coef_row) + r" \\")
        lines.append(" & ".join(se_row) + r" \\")

    for stat, key in [("Country FE", "Yes"), ("Year FE", "Yes"), ("Macro + air controls", "Yes")]:
        lines.append(f"{stat} & {key} & {key} & {key} \\\\")

    for stat in ["nobs", "r2_within"]:
        vals = []
        for m in models:
            d = coef_df[coef_df["model"] == m]
            if d.empty:
                vals.append("")
            else:
                vals.append(f"{float(d.iloc[0][stat]):.0f}" if stat == "nobs" else f"{float(d.iloc[0][stat]):.3f}")
        lbl = "Observations" if stat == "nobs" else r"$R^2$ (within)"
        lines.append(lbl + " & " + " & ".join(vals) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item Notes: Dependent variable is annual house-price growth. Asylum is first-time applications (monthly data aggregated to annual rates). Non-asylum immigration is total immigration rate minus asylum-applicant rate. Naturalization variables come from Eurostat acquisition-of-citizenship data. Car controls use Eurostat passenger-car registrations. Standard errors are two-way clustered by country and year.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        "",
    ]
    (PAPER_TABLES_DIR / "tab_extended_channels_annual.tex").write_text("\n".join(lines))

    # Quarterly high-frequency non-air channels.
    qmodels = ["quarterly_ext_air_only", "quarterly_ext_nonair_mobility", "quarterly_ext_full"]
    qheaders = ["Air only", "+ Tourism+Asylum", "Full"]
    qrows = [
        ("Lagged air-passenger YoY growth", "L1_air_yoy"),
        ("Lagged tourism nights YoY growth", "L1_tour_total_yoy"),
        ("Lagged asylum first-applicant rate per 100k", "L1_asylum_first_q_per100k"),
        ("Lagged temporary-protection stock (UA) per 100k", "L1_tps_ua_stock_qe_per100k"),
        ("Lagged passengers per movement", "L1_pax_per_move_total"),
        ("Lagged airfare inflation (CP0733 YoY)", "L1_airfare_yoy_q"),
    ]
    ql = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Quarterly high-frequency channels beyond air-passenger counts}",
        r"\label{tab:extended_quarterly}",
        r"\begin{threeparttable}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{p{6.0cm}ccc}",
        r"\toprule",
        " & " + " & ".join(qheaders) + r" \\",
        r"\midrule",
    ]
    for label, term in qrows:
        coef_row, se_row = [label], [""]
        for m in qmodels:
            c, s = _coef_entry(coef_df, m, term)
            coef_row.append(c)
            se_row.append(s)
        ql.append(" & ".join(coef_row) + r" \\")
        ql.append(" & ".join(se_row) + r" \\")

    ql.append("Country FE & Yes & Yes & Yes \\\\")
    ql.append("Quarter FE & Yes & Yes & Yes \\\\")
    for stat in ["nobs", "r2_within"]:
        vals = []
        for m in qmodels:
            d = coef_df[coef_df["model"] == m]
            if d.empty:
                vals.append("")
            else:
                vals.append(f"{float(d.iloc[0][stat]):.0f}" if stat == "nobs" else f"{float(d.iloc[0][stat]):.3f}")
        lbl = "Observations" if stat == "nobs" else r"$R^2$ (within)"
        ql.append(lbl + " & " + " & ".join(vals) + r" \\")

    ql += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item Notes: Dependent variable is quarterly house-price YoY growth. Tourism nights come from Eurostat monthly accommodation data. Asylum and temporary-protection variables are monthly Eurostat migration-protection series aggregated to quarter. Standard errors are two-way clustered by country and quarter.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        "",
    ]
    (PAPER_TABLES_DIR / "tab_extended_channels_quarterly.tex").write_text("\n".join(ql))

    # Origin remittance-growth channel.
    rmodels = ["origin_remit_baseline", "origin_remit_interaction"]
    rh = ["Baseline", "+ Interaction"]
    rrows = [
        ("Lagged outflow to Europe (per 1,000)", "L1_outflow_rate_per_1000"),
        ("Lagged remittance inflows (% of GDP)", "L1_wb_remit_in_pct_gdp"),
        ("Lagged outflow $\\times$ remittance inflows", "L1_outflow_x_remit_in"),
        ("Lagged unemployment rate", "L1_wb_unemployment"),
    ]
    rl = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Origin-country channel: outmigration, remittances, and GDP-per-capita growth}",
        r"\label{tab:origin_remittance}",
        r"\begin{threeparttable}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{p{6.0cm}cc}",
        r"\toprule",
        " & " + " & ".join(rh) + r" \\",
        r"\midrule",
    ]
    for label, term in rrows:
        coef_row, se_row = [label], [""]
        for m in rmodels:
            c, s = _coef_entry(coef_df, m, term)
            coef_row.append(c)
            se_row.append(s)
        rl.append(" & ".join(coef_row) + r" \\")
        rl.append(" & ".join(se_row) + r" \\")
    rl.append("Origin FE & Yes & Yes \\\\")
    rl.append("Year FE & Yes & Yes \\\\")
    for stat in ["nobs", "r2_within"]:
        vals = []
        for m in rmodels:
            d = coef_df[coef_df["model"] == m]
            if d.empty:
                vals.append("")
            else:
                vals.append(f"{float(d.iloc[0][stat]):.0f}" if stat == "nobs" else f"{float(d.iloc[0][stat]):.3f}")
        lbl = "Observations" if stat == "nobs" else r"$R^2$ (within)"
        rl.append(lbl + " & " + " & ".join(vals) + r" \\")
    rl += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item Notes: Dependent variable is origin-country GDP per-capita growth. Outflow to Europe is summed from destination-country immigration records by previous residence. Remittance variables are World Bank indicators. Standard errors are two-way clustered by origin and year.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        "",
    ]
    (PAPER_TABLES_DIR / "tab_origin_remittance_channel.tex").write_text("\n".join(rl))


def plot_key_coefficients(coef_df: pd.DataFrame) -> None:
    PAPER_FIGS_DIR.mkdir(parents=True, exist_ok=True)
    picks = [
        ("annual_ext_mig_types_cars", "L1_non_asylum_immigration_rate_per_1000", "Annual: non-asylum immigration"),
        ("annual_ext_mig_types_cars", "L1_asylum_rate_per_1000", "Annual: asylum-applicant rate"),
        ("annual_ext_mig_types_cars", "L1_naturalization_rate_per_1000", "Annual: naturalization rate"),
        ("annual_ext_mig_types_cars", "L1_car_reg_total_per_1000", "Annual: car registrations"),
        ("quarterly_ext_full", "L1_air_yoy", "Quarterly: air-passenger growth"),
        ("quarterly_ext_full", "L1_tour_total_yoy", "Quarterly: tourism nights growth"),
        ("quarterly_ext_full", "L1_asylum_first_q_per100k", "Quarterly: asylum first apps per 100k"),
        ("quarterly_ext_full", "L1_tps_ua_stock_qe_per100k", "Quarterly: temporary protection stock (UA)"),
        ("origin_remit_interaction", "L1_wb_remit_in_pct_gdp", "Origin growth: remittance inflows"),
        ("origin_remit_interaction", "L1_outflow_x_remit_in", "Origin growth: outflow x remittance"),
    ]

    rows = []
    for m, t, lbl in picks:
        r = coef_df[(coef_df["model"] == m) & (coef_df["term"] == t)]
        if r.empty:
            continue
        x = r.iloc[0]
        rows.append({"label": lbl, "coef": float(x["coef"]), "se": float(x["std_err"])})
    if not rows:
        return

    d = pd.DataFrame(rows)
    y = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.axvline(0, color="#555555", lw=1.0, ls="--")
    ax.errorbar(d["coef"], y, xerr=1.96 * d["se"], fmt="none", ecolor="#222222", elinewidth=1.4, capsize=3)
    colors = ["#1f4e79" if "Annual" in l else "#b22222" if "Quarterly" in l else "#2f7d32" for l in d["label"]]
    ax.scatter(d["coef"], y, c=colors, s=40, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"], fontsize=9)
    ax.set_xlabel("Coefficient (95% confidence interval)")
    ax.set_title("Expanded channels: migration type, non-air high-frequency mobility, and origin remittance channel")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(PAPER_FIGS_DIR / "fig_extended_channels_coefficients.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    annual_base = pd.read_parquet(PROC_DIR / "panel_annual_migration_composition.parquet")
    annual_ext = build_annual_type_features(annual_base)
    quarter_ext = build_quarterly_extended_panel()
    rem_panel = build_origin_remittance_panel()

    annual_ext.to_parquet(PROC_DIR / "panel_annual_extended_channels.parquet", index=False)
    annual_ext.to_csv(PROC_DIR / "panel_annual_extended_channels.csv", index=False)
    quarter_ext.to_parquet(PROC_DIR / "panel_quarterly_extended_channels.parquet", index=False)
    quarter_ext.to_csv(PROC_DIR / "panel_quarterly_extended_channels.csv", index=False)
    rem_panel.to_parquet(PROC_DIR / "panel_origin_remittance_growth.parquet", index=False)
    rem_panel.to_csv(PROC_DIR / "panel_origin_remittance_growth.csv", index=False)

    annual_coef, annual_txt = estimate_annual_models(annual_ext)
    q_coef, q_txt = estimate_quarterly_models(quarter_ext)
    rem_coef, rem_txt = estimate_origin_remittance_models(rem_panel)

    coef_df = pd.concat([annual_coef, q_coef, rem_coef], ignore_index=True, sort=False)
    coef_df.to_csv(RESULTS_DIR / "extended_channels_coefficients.csv", index=False)
    (RESULTS_DIR / "extended_channels_summaries.txt").write_text("\n\n".join(annual_txt + q_txt + rem_txt))

    summary_rows = []
    for name, df in [
        ("annual_extended", annual_ext),
        ("quarterly_extended", quarter_ext),
        ("origin_remittance", rem_panel),
    ]:
        summary_rows.append(
            {
                "sample": name,
                "rows": int(len(df)),
                "countries_or_origins": int(df[df.columns[0]].astype(str).nunique()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "extended_channels_sample_stats.csv", index=False)

    write_latex_tables(coef_df)
    plot_key_coefficients(coef_df)

    meta = {
        "annual_extended_rows": int(len(annual_ext)),
        "quarterly_extended_rows": int(len(quarter_ext)),
        "origin_remittance_rows": int(len(rem_panel)),
        "annual_countries": int(annual_ext["geo"].astype(str).nunique()),
        "quarterly_countries": int(quarter_ext["geo"].astype(str).nunique()),
        "origin_panel_origins": int(rem_panel["origin"].astype(str).nunique()),
        "quarterly_period_min": str(quarter_ext["period_str"].dropna().min()) if "period_str" in quarter_ext else "",
        "quarterly_period_max": str(quarter_ext["period_str"].dropna().max()) if "period_str" in quarter_ext else "",
    }
    (META_DIR / "extended_channels_summary.json").write_text(json.dumps(meta, indent=2))

    print(f"[ok] wrote {RESULTS_DIR / 'extended_channels_coefficients.csv'}")
    print(f"[ok] wrote {PAPER_TABLES_DIR / 'tab_extended_channels_annual.tex'}")
    print(f"[ok] wrote {PAPER_TABLES_DIR / 'tab_extended_channels_quarterly.tex'}")
    print(f"[ok] wrote {PAPER_TABLES_DIR / 'tab_origin_remittance_channel.tex'}")
    print(f"[ok] wrote {PAPER_FIGS_DIR / 'fig_extended_channels_coefficients.pdf'}")


if __name__ == "__main__":
    main()
