#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from linearmodels.panel import PanelOLS

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
META_DIR = ROOT / "data" / "metadata"
PAPER_TABLES_DIR = ROOT / "paper_overleaf" / "tables"

UK_HPI_PAGE = "https://www.gov.uk/government/statistical-data-sets/uk-house-price-index-data-downloads-november-2025"
UK_HPI_LINK_PATTERN = re.compile(r"https://publicdata\.landregistry\.gov\.uk/market-trend-data/house-price-index-data/UK-HPI-full-file-[0-9]{4}-[0-9]{2}\.csv")
ONS_LTIM_2020_URL = "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/internationalmigration/datasets/longterminternationalmigrationprovisional/yearendingdecember2020/modelledestimatesforukinternationalmigration.xlsx"
ONS_LTIM_CURRENT_URL = "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/internationalmigration/datasets/longterminternationalmigrationprovisional/current/modelledestimatesforukinternationalmigration.xlsx"
BDL_UNIT_URL = "https://bdl.stat.gov.pl/api/v1/data/by-unit/000000000000"
INE_TABLE_URL_TMPL = "https://servicios.ine.es/wstempus/js/ES/DATOS_TABLA/{table_id}?tip=AM"

RAW_UK_HPI = RAW_DIR / "uk_hpi_full_latest.csv"
RAW_ONS_LTIM = RAW_DIR / "ons_ltim_yearendingdec2020.xlsx"
RAW_BDL_PL_IMM = RAW_DIR / "pl_bdl_immigration_annual.csv"
RAW_BDL_PL_EMI = RAW_DIR / "pl_bdl_emigration_annual.csv"
RAW_INE_ES_HPI_Q = RAW_DIR / "es_ine_hpi_quarterly.csv"
RAW_INE_ES_HPI_A = RAW_DIR / "es_ine_hpi_annual.csv"
RAW_INE_ES_IMM_OLD = RAW_DIR / "es_ine_immigration_annual_legacy.csv"
RAW_INE_ES_EMI_OLD = RAW_DIR / "es_ine_emigration_annual_legacy.csv"
RAW_INE_ES_IMM_NEW = RAW_DIR / "es_ine_immigration_annual_new.csv"
RAW_INE_ES_EMI_NEW = RAW_DIR / "es_ine_emigration_annual_new.csv"

# Poland official BDL variable ids (annual, permanent external migration)
PL_BDL_IMMIGR_ID = 269600
PL_BDL_EMIGR_ID = 269586

# Spain INE table ids
ES_INE_HPI_Q_TABLE_ID = 76201
ES_INE_HPI_A_TABLE_ID = 25173
ES_INE_IMM_OLD_TABLE_ID = 24282
ES_INE_EMI_OLD_TABLE_ID = 24296
ES_INE_IMM_NEW_TABLE_ID = 69687
ES_INE_EMI_NEW_TABLE_ID = 69702


def _safe_get(url: str, *, params: dict | None = None) -> requests.Response:
    r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    return r


def fetch_uk_hpi_latest() -> tuple[pd.DataFrame, str]:
    html = _safe_get(UK_HPI_PAGE).text
    links = sorted(set(UK_HPI_LINK_PATTERN.findall(html)))
    if not links:
        raise RuntimeError("Could not find UK HPI full file link on GOV.UK page")
    link = links[-1]
    df = pd.read_csv(link)
    RAW_UK_HPI.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_UK_HPI, index=False)
    return df, link


def fetch_ons_ltim_2020() -> pd.DataFrame:
    if not RAW_ONS_LTIM.exists():
        # Try current ONS workbook first; fall back to yearendingdec2020 if current is unavailable.
        last_err: Exception | None = None
        for url in [ONS_LTIM_CURRENT_URL, ONS_LTIM_2020_URL]:
            try:
                r = _safe_get(url)
                # XLSX is a zip container and usually starts with PK.
                if not r.content.startswith(b"PK"):
                    raise RuntimeError(f"ONS LTIM response is not XLSX for {url}")
                RAW_ONS_LTIM.parent.mkdir(parents=True, exist_ok=True)
                RAW_ONS_LTIM.write_bytes(r.content)
                break
            except Exception as e:
                last_err = e
        if not RAW_ONS_LTIM.exists():
            raise RuntimeError(f"Could not fetch ONS LTIM workbook (current or 2020 fallback). Last error: {last_err}")

    raw = pd.read_excel(RAW_ONS_LTIM, sheet_name="Quarterly Data", header=None)
    header = raw.iloc[3].astype(str).str.strip().tolist()
    q = raw.iloc[4:].copy()
    q.columns = header
    q = q.rename(columns={
        "Year": "year",
        "Quarter": "quarter",
        "Total immigration\nEstimate": "uk_ons_immigration_q",
        "Total emigration\nEstimate": "uk_ons_emigration_q",
        "Total net migration\nEstimate": "uk_ons_net_migration_q",
    })
    keep = ["year", "quarter", "uk_ons_immigration_q", "uk_ons_emigration_q", "uk_ons_net_migration_q"]
    q = q[keep].dropna(subset=["year", "quarter"]).copy()
    q["year"] = pd.to_numeric(q["year"], errors="coerce").astype("Int64")
    q["quarter"] = pd.to_numeric(q["quarter"], errors="coerce").astype("Int64")
    q = q.dropna(subset=["year", "quarter"]).copy()
    q["year"] = q["year"].astype(int)
    q["quarter"] = q["quarter"].astype(int)
    q["period_str"] = q["year"].astype(str) + "Q" + q["quarter"].astype(str)
    q["geo"] = "UK"
    return q


def fetch_poland_bdl_var(var_id: int) -> pd.DataFrame:
    r = _safe_get(BDL_UNIT_URL, params={"format": "json", "var-id": var_id})
    payload = r.json()
    rows = []
    for rec in payload.get("results", []):
        for val in rec.get("values", []):
            rows.append({"year": int(val["year"]), "value": pd.to_numeric(val.get("val"), errors="coerce")})
    return pd.DataFrame(rows)


def fetch_ine_table(table_id: int) -> list[dict]:
    url = INE_TABLE_URL_TMPL.format(table_id=table_id)
    return _safe_get(url).json()


def _ine_row_to_series(raw_rows: list[dict], row_name: str, value_col: str) -> pd.DataFrame:
    rows = [r for r in raw_rows if str(r.get("Nombre", "")).strip() == row_name.strip()]
    if not rows:
        raise RuntimeError(f"Could not find INE row '{row_name}'")
    vals = pd.DataFrame(rows[0].get("Data", []))
    out = vals.rename(columns={"Anyo": "year", "Valor": value_col})[["year", value_col]].copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    return out.dropna(subset=["year"]).astype({"year": int})


def _ine_row_to_quarterly_index(raw_rows: list[dict], row_name: str, value_col: str) -> pd.DataFrame:
    rows = [r for r in raw_rows if str(r.get("Nombre", "")).strip() == row_name.strip()]
    if not rows:
        raise RuntimeError(f"Could not find INE row '{row_name}'")
    vals = pd.DataFrame(rows[0].get("Data", []))
    q = vals.rename(columns={"Valor": value_col, "Fecha": "date"})[["date", value_col]].copy()
    q["date"] = pd.to_datetime(q["date"], errors="coerce", utc=True)
    q = q.dropna(subset=["date"]).sort_values("date")
    q["value"] = pd.to_numeric(q[value_col], errors="coerce")
    q["year"] = q["date"].dt.year
    q["quarter"] = q["date"].dt.quarter
    q["period_str"] = q["year"].astype(str) + "Q" + q["quarter"].astype(str)
    q[value_col] = q["value"]
    return q[["year", "quarter", "period_str", value_col]].copy()


def _load_preferred_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prefer core stream panels so official overlays can be blended directly into
    the main pipeline, not only into side datasets.
    """
    annual_core = PROC_DIR / "panel_annual_harmonized.parquet"
    quarterly_core = PROC_DIR / "panel_quarterly_harmonized.parquet"
    annual_alt = PROC_DIR / "panel_annual_migration_composition.parquet"
    quarterly_alt = PROC_DIR / "panel_quarterly_traveler_quality.parquet"

    if annual_core.exists():
        annual = pd.read_parquet(annual_core)
    elif annual_alt.exists():
        annual = pd.read_parquet(annual_alt)
    else:
        raise FileNotFoundError("Could not find annual base panel (`panel_annual_harmonized.parquet` or fallback panel).")

    if quarterly_core.exists():
        quarterly = pd.read_parquet(quarterly_core)
    elif quarterly_alt.exists():
        quarterly = pd.read_parquet(quarterly_alt)
    else:
        raise FileNotFoundError("Could not find quarterly base panel (`panel_quarterly_harmonized.parquet` or fallback panel).")

    if "period_str" not in quarterly.columns:
        if {"year", "quarter"}.issubset(set(quarterly.columns)):
            quarterly["period_str"] = quarterly["year"].astype(int).astype(str) + "Q" + quarterly["quarter"].astype(int).astype(str)
        elif "period" in quarterly.columns:
            p = pd.PeriodIndex(quarterly["period"], freq="Q")
            quarterly["period_str"] = p.astype(str)
            quarterly["year"] = p.year.astype(int)
            quarterly["quarter"] = p.quarter.astype(int)
        else:
            raise ValueError("Quarterly panel missing period fields needed for country-web overlay merge.")
    return annual, quarterly


def build_official_overlays() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    annual, quarterly = _load_preferred_panel()

    # UK official HPI monthly -> quarterly/annual yoy
    uk_hpi_raw, uk_hpi_link = fetch_uk_hpi_latest()
    uk = uk_hpi_raw[uk_hpi_raw["RegionName"].astype(str).str.upper() == "UNITED KINGDOM"].copy()
    uk["date"] = pd.to_datetime(uk["Date"], errors="coerce", dayfirst=True)
    uk = uk.dropna(subset=["date"]).sort_values("date")
    uk["AveragePrice"] = pd.to_numeric(uk["AveragePrice"], errors="coerce")
    uk["year"] = uk["date"].dt.year
    uk["quarter"] = uk["date"].dt.quarter
    uk["period_str"] = uk["year"].astype(str) + "Q" + uk["quarter"].astype(str)

    uk_q = uk.groupby(["year", "quarter", "period_str"], as_index=False)["AveragePrice"].mean()
    uk_q["uk_hpi_yoy_official_q"] = np.log(uk_q["AveragePrice"]).diff(4) * 100.0
    uk_q["geo"] = "UK"

    uk_a = uk.groupby("year", as_index=False)["AveragePrice"].mean()
    uk_a["uk_hpi_growth_official_a"] = np.log(uk_a["AveragePrice"]).diff(1) * 100.0
    uk_a["geo"] = "UK"

    # Poland official migration (annual counts)
    pl_imm = fetch_poland_bdl_var(PL_BDL_IMMIGR_ID).rename(columns={"value": "pl_immigration_official"})
    pl_emi = fetch_poland_bdl_var(PL_BDL_EMIGR_ID).rename(columns={"value": "pl_emigration_official"})
    pl = pl_imm.merge(pl_emi, on="year", how="outer")
    pl["pl_net_migration_official"] = pl["pl_immigration_official"] - pl["pl_emigration_official"]
    pl["geo"] = "PL"

    RAW_BDL_PL_IMM.parent.mkdir(parents=True, exist_ok=True)
    pl_imm.to_csv(RAW_BDL_PL_IMM, index=False)
    pl_emi.to_csv(RAW_BDL_PL_EMI, index=False)

    # Spain official overlays from INE (HPI + migration).
    es_hpi_q_raw = fetch_ine_table(ES_INE_HPI_Q_TABLE_ID)
    es_hpi_a_raw = fetch_ine_table(ES_INE_HPI_A_TABLE_ID)
    es_imm_old_raw = fetch_ine_table(ES_INE_IMM_OLD_TABLE_ID)
    es_emi_old_raw = fetch_ine_table(ES_INE_EMI_OLD_TABLE_ID)
    es_imm_new_raw = fetch_ine_table(ES_INE_IMM_NEW_TABLE_ID)
    es_emi_new_raw = fetch_ine_table(ES_INE_EMI_NEW_TABLE_ID)

    es_hpi_q = _ine_row_to_quarterly_index(es_hpi_q_raw, "Total Nacional. General. Índice.", "es_hpi_index_official_q")
    es_hpi_q["es_hpi_yoy_official_q"] = np.log(es_hpi_q["es_hpi_index_official_q"]).diff(4) * 100.0
    es_hpi_q["geo"] = "ES"

    es_hpi_a = _ine_row_to_series(es_hpi_a_raw, "Total Nacional. Media anual. General.", "es_hpi_index_official_a")
    es_hpi_a["es_hpi_growth_official_a"] = np.log(es_hpi_a["es_hpi_index_official_a"]).diff(1) * 100.0
    es_hpi_a["geo"] = "ES"

    es_imm_old = _ine_row_to_series(
        es_imm_old_raw,
        "Total. Flujo de inmigraciones procedentes del extranjero. Total Nacional. Todas las edades.",
        "es_immigration_official_old",
    )
    es_emi_old = _ine_row_to_series(
        es_emi_old_raw,
        "Total Nacional. Total. Flujo de emigraciones con destino el extranjero. Todas las edades.",
        "es_emigration_official_old",
    )
    es_imm_new = _ine_row_to_series(
        es_imm_new_raw,
        "Todas las edades. Total. Dato base. Inmigraciones procedentes del extranjero.",
        "es_immigration_official_new",
    )
    es_emi_new = _ine_row_to_series(
        es_emi_new_raw,
        "Todas las edades. Total. Dato base. Emigraciones con destino al extranjero.",
        "es_emigration_official_new",
    )

    es = es_imm_old.merge(es_emi_old, on="year", how="outer").merge(es_imm_new, on="year", how="outer").merge(es_emi_new, on="year", how="outer")
    es["es_immigration_official"] = es["es_immigration_official_new"].combine_first(es["es_immigration_official_old"])
    es["es_emigration_official"] = es["es_emigration_official_new"].combine_first(es["es_emigration_official_old"])
    es["es_net_migration_official"] = es["es_immigration_official"] - es["es_emigration_official"]
    es["geo"] = "ES"

    es_hpi_q.to_csv(RAW_INE_ES_HPI_Q, index=False)
    es_hpi_a.to_csv(RAW_INE_ES_HPI_A, index=False)
    es_imm_old.to_csv(RAW_INE_ES_IMM_OLD, index=False)
    es_emi_old.to_csv(RAW_INE_ES_EMI_OLD, index=False)
    es_imm_new.to_csv(RAW_INE_ES_IMM_NEW, index=False)
    es_emi_new.to_csv(RAW_INE_ES_EMI_NEW, index=False)

    # Merge overlays into annual panel
    annual_ext = annual.merge(uk_a[["geo", "year", "uk_hpi_growth_official_a"]], on=["geo", "year"], how="left")
    annual_ext = annual_ext.merge(pl[["geo", "year", "pl_immigration_official", "pl_emigration_official", "pl_net_migration_official"]], on=["geo", "year"], how="left")
    annual_ext = annual_ext.merge(es_hpi_a[["geo", "year", "es_hpi_growth_official_a"]], on=["geo", "year"], how="left")
    annual_ext = annual_ext.merge(es[["geo", "year", "es_immigration_official", "es_emigration_official", "es_net_migration_official"]], on=["geo", "year"], how="left")
    annual_ext["pl_net_migration_rate_official_per_1000"] = np.where(
        annual_ext["population"] > 0,
        annual_ext["pl_net_migration_official"] / annual_ext["population"] * 1000.0,
        np.nan,
    )
    annual_ext["es_net_migration_rate_official_per_1000"] = np.where(
        annual_ext["population"] > 0,
        annual_ext["es_net_migration_official"] / annual_ext["population"] * 1000.0,
        np.nan,
    )

    annual_ext["net_migration_rate_countryweb_patch"] = annual_ext["net_migration_rate"]
    mask_pl = (annual_ext["geo"] == "PL") & annual_ext["pl_net_migration_rate_official_per_1000"].notna()
    annual_ext.loc[mask_pl, "net_migration_rate_countryweb_patch"] = annual_ext.loc[mask_pl, "pl_net_migration_rate_official_per_1000"]
    mask_es = (annual_ext["geo"] == "ES") & annual_ext["es_net_migration_rate_official_per_1000"].notna()
    annual_ext.loc[mask_es, "net_migration_rate_countryweb_patch"] = annual_ext.loc[mask_es, "es_net_migration_rate_official_per_1000"]
    annual_ext = annual_ext.sort_values(["geo", "year"]).reset_index(drop=True)
    annual_ext["L1_net_migration_rate_countryweb_patch"] = annual_ext.groupby("geo")["net_migration_rate_countryweb_patch"].shift(1)

    # Merge overlays into quarterly panel
    q_ext = quarterly.merge(uk_q[["geo", "period_str", "uk_hpi_yoy_official_q"]], on=["geo", "period_str"], how="left")
    q_ext = q_ext.merge(fetch_ons_ltim_2020()[["geo", "period_str", "uk_ons_net_migration_q"]], on=["geo", "period_str"], how="left")
    q_ext = q_ext.merge(es_hpi_q[["geo", "period_str", "es_hpi_yoy_official_q"]], on=["geo", "period_str"], how="left")
    for col in ["uk_hpi_yoy_official_q", "uk_ons_net_migration_q", "es_hpi_yoy_official_q"]:
        if col not in q_ext.columns:
            q_ext[col] = np.nan

    meta = {
        "uk_hpi_source": uk_hpi_link,
        "pl_bdl_imm_var_id": PL_BDL_IMMIGR_ID,
        "pl_bdl_emi_var_id": PL_BDL_EMIGR_ID,
        "es_ine_hpi_q_table_id": ES_INE_HPI_Q_TABLE_ID,
        "es_ine_hpi_a_table_id": ES_INE_HPI_A_TABLE_ID,
        "es_ine_imm_old_table_id": ES_INE_IMM_OLD_TABLE_ID,
        "es_ine_emi_old_table_id": ES_INE_EMI_OLD_TABLE_ID,
        "es_ine_imm_new_table_id": ES_INE_IMM_NEW_TABLE_ID,
        "es_ine_emi_new_table_id": ES_INE_EMI_NEW_TABLE_ID,
        "uk_hpi_quarterly_obs": int(q_ext["uk_hpi_yoy_official_q"].notna().sum()),
        "uk_ons_migration_quarterly_obs": int(q_ext["uk_ons_net_migration_q"].notna().sum()),
        "pl_official_migration_annual_obs": int(annual_ext["pl_net_migration_rate_official_per_1000"].notna().sum()),
        "es_hpi_quarterly_obs": int(q_ext["es_hpi_yoy_official_q"].notna().sum()),
        "es_official_migration_annual_obs": int(annual_ext["es_net_migration_rate_official_per_1000"].notna().sum()),
    }
    return annual_ext, q_ext, pl, meta


def _apply_blend_to_core_streams(annual_ext: pd.DataFrame, q_ext: pd.DataFrame) -> dict:
    """
    Persist blended official overlays directly into core stream panels.
    """
    annual_core_path_pq = PROC_DIR / "panel_annual_harmonized.parquet"
    annual_core_path_csv = PROC_DIR / "panel_annual_harmonized.csv"
    quarterly_core_path_pq = PROC_DIR / "panel_quarterly_harmonized.parquet"
    quarterly_core_path_csv = PROC_DIR / "panel_quarterly_harmonized.csv"

    annual = pd.read_parquet(annual_core_path_pq) if annual_core_path_pq.exists() else pd.read_csv(annual_core_path_csv)
    quarterly = pd.read_parquet(quarterly_core_path_pq) if quarterly_core_path_pq.exists() else pd.read_csv(quarterly_core_path_csv)

    # Annual blend: patch migration rate directly in core stream.
    for c in [
        "net_migration_rate_countryweb_patch",
        "pl_net_migration_rate_official_per_1000",
        "es_net_migration_rate_official_per_1000",
        "net_migration_rate_eurostat",
        "net_migration_rate_source",
    ]:
        if c in annual.columns:
            annual = annual.drop(columns=[c])
    ann_patch_cols = [
        "geo",
        "year",
        "net_migration_rate_countryweb_patch",
        "pl_net_migration_rate_official_per_1000",
        "es_net_migration_rate_official_per_1000",
    ]
    ann_patch = annual_ext[ann_patch_cols].drop_duplicates(["geo", "year"])
    annual = annual.merge(ann_patch, on=["geo", "year"], how="left")
    annual["net_migration_rate_eurostat"] = annual["net_migration_rate"]

    annual["net_migration_rate_source"] = "eurostat"
    annual.loc[(annual["geo"] == "PL") & annual["pl_net_migration_rate_official_per_1000"].notna(), "net_migration_rate_source"] = "poland_bdl"
    annual.loc[(annual["geo"] == "ES") & annual["es_net_migration_rate_official_per_1000"].notna(), "net_migration_rate_source"] = "spain_ine"

    annual["net_migration_rate"] = annual["net_migration_rate_countryweb_patch"].combine_first(annual["net_migration_rate"])
    if "net_migration_rate_harmonized" in annual.columns:
        annual["net_migration_rate_harmonized"] = annual["net_migration_rate_countryweb_patch"].combine_first(
            annual["net_migration_rate_harmonized"]
        )
    annual = annual.sort_values(["geo", "year"]).reset_index(drop=True)
    annual["L1_net_migration_rate"] = annual.groupby("geo", sort=False)["net_migration_rate"].shift(1)
    if "net_migration_rate_harmonized" in annual.columns:
        annual["L1_net_migration_rate_harmonized"] = annual.groupby("geo", sort=False)["net_migration_rate_harmonized"].shift(1)

    # Quarterly blend: patch hpi_yoy where official series exist (UK, ES).
    for c in ["uk_hpi_yoy_official_q", "es_hpi_yoy_official_q", "hpi_yoy_eurostat", "hpi_yoy_source"]:
        if c in quarterly.columns:
            quarterly = quarterly.drop(columns=[c])
    for col in ["uk_hpi_yoy_official_q", "es_hpi_yoy_official_q"]:
        if col not in q_ext.columns:
            q_ext[col] = np.nan
    q_patch = q_ext[["geo", "period_str", "uk_hpi_yoy_official_q", "es_hpi_yoy_official_q"]].drop_duplicates(["geo", "period_str"])
    quarterly = quarterly.copy()
    if "period_str" not in quarterly.columns:
        quarterly["period_str"] = quarterly["year"].astype(int).astype(str) + "Q" + quarterly["quarter"].astype(int).astype(str)
    quarterly = quarterly.merge(q_patch, on=["geo", "period_str"], how="left")

    quarterly["hpi_yoy_eurostat"] = quarterly["hpi_yoy"]
    quarterly["hpi_yoy_source"] = "eurostat"
    uk_mask = (quarterly["geo"] == "UK") & quarterly["uk_hpi_yoy_official_q"].notna()
    es_mask = (quarterly["geo"] == "ES") & quarterly["es_hpi_yoy_official_q"].notna()
    quarterly.loc[uk_mask, "hpi_yoy"] = quarterly.loc[uk_mask, "uk_hpi_yoy_official_q"]
    quarterly.loc[es_mask, "hpi_yoy"] = quarterly.loc[es_mask, "es_hpi_yoy_official_q"]
    if "hpi_yoy_harmonized" in quarterly.columns:
        quarterly.loc[uk_mask, "hpi_yoy_harmonized"] = quarterly.loc[uk_mask, "uk_hpi_yoy_official_q"]
        quarterly.loc[es_mask, "hpi_yoy_harmonized"] = quarterly.loc[es_mask, "es_hpi_yoy_official_q"]
    quarterly.loc[uk_mask, "hpi_yoy_source"] = "uk_hmlr"
    quarterly.loc[es_mask, "hpi_yoy_source"] = "spain_ine"

    quarterly = quarterly.sort_values(["geo", "year", "quarter"]).reset_index(drop=True)

    annual.to_parquet(annual_core_path_pq, index=False)
    annual.to_csv(annual_core_path_csv, index=False)
    quarterly.to_parquet(quarterly_core_path_pq, index=False)
    quarterly.to_csv(quarterly_core_path_csv, index=False)

    return {
        "annual_rows_core": int(len(annual)),
        "quarterly_rows_core": int(len(quarterly)),
        "annual_migration_rate_patched_rows": int((annual["net_migration_rate_source"] != "eurostat").sum()),
        "quarterly_hpi_yoy_patched_rows": int((quarterly["hpi_yoy_source"] != "eurostat").sum()),
        "annual_pl_patched_rows": int((annual["net_migration_rate_source"] == "poland_bdl").sum()),
        "annual_es_patched_rows": int((annual["net_migration_rate_source"] == "spain_ine").sum()),
        "quarterly_uk_hpi_patched_rows": int((quarterly["hpi_yoy_source"] == "uk_hmlr").sum()),
        "quarterly_es_hpi_patched_rows": int((quarterly["hpi_yoy_source"] == "spain_ine").sum()),
    }


def _fit_annual_overlay_models(annual_ext: pd.DataFrame) -> pd.DataFrame:
    d = annual_ext.copy()
    num_cols = d.select_dtypes(include=[np.number]).columns
    d[num_cols] = d[num_cols].replace([np.inf, -np.inf], np.nan)
    controls = ["L1_air_growth", "L1_gdp_pc_growth", "L1_unemployment_rate", "L1_inflation_hicp", "L1_long_rate", "L1_pop_growth"]

    out_rows = []
    specs = [
        (
            "annual_countryweb_baseline",
            "hpi_growth ~ 1 + L1_net_migration_rate + " + " + ".join(controls) + " + EntityEffects + TimeEffects",
            ["hpi_growth", "L1_net_migration_rate"] + controls,
            "L1_net_migration_rate",
        ),
        (
            "annual_countryweb_patch",
            "hpi_growth ~ 1 + L1_net_migration_rate_countryweb_patch + " + " + ".join(controls) + " + EntityEffects + TimeEffects",
            ["hpi_growth", "L1_net_migration_rate_countryweb_patch"] + controls,
            "L1_net_migration_rate_countryweb_patch",
        ),
    ]

    for model, formula, need, term in specs:
        x = d.dropna(subset=need).copy().set_index(["geo", "year"]).sort_index()
        if x.empty:
            continue
        res = PanelOLS.from_formula(formula, data=x, drop_absorbed=True, check_rank=False).fit(
            cov_type="clustered", cluster_entity=True, cluster_time=True
        )
        if term in res.params.index:
            out_rows.append(
                {
                    "model": model,
                    "term": term,
                    "coef": float(res.params[term]),
                    "std_err": float(res.std_errors[term]),
                    "p_value": float(res.pvalues[term]),
                    "nobs": float(res.nobs),
                    "r2_within": float(getattr(res, "rsquared_within", np.nan)),
                }
            )
    return pd.DataFrame(out_rows)


def write_table(overlay_stats: dict, model_df: pd.DataFrame) -> None:
    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    def _fmt(x, d=3):
        return "" if pd.isna(x) else f"{x:.{d}f}"

    c_base = model_df[model_df["model"] == "annual_countryweb_baseline"]
    c_patch = model_df[model_df["model"] == "annual_countryweb_patch"]

    bcoef = _fmt(c_base["coef"].iloc[0], 3) if not c_base.empty else ""
    bse = _fmt(c_base["std_err"].iloc[0], 3) if not c_base.empty else ""
    bp = _fmt(c_base["p_value"].iloc[0], 3) if not c_base.empty else ""
    pcoef = _fmt(c_patch["coef"].iloc[0], 3) if not c_patch.empty else ""
    pse = _fmt(c_patch["std_err"].iloc[0], 3) if not c_patch.empty else ""
    pp = _fmt(c_patch["p_value"].iloc[0], 3) if not c_patch.empty else ""

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Country-web official overlays (UK, Poland, and Spain): coverage and annual migration-coefficient robustness}",
        r"\label{tab:country_web_overlay}",
        r"\begin{threeparttable}",
        r"\small",
        r"\begin{tabular}{p{8.1cm}cc}",
        r"\toprule",
        r"Metric & Baseline panel & Country-web overlay panel \\",
        r"\midrule",
        f"Annual migration coefficient & {bcoef} & {pcoef} \\\\",
        f"Std. error & ({bse}) & ({pse}) \\\\",
        f"p-value & [{bp}] & [{pp}] \\\\",
        f"Poland official migration years (BDL) &  & {overlay_stats.get('pl_years','')} \\\\",
        f"Spain official migration years (INE stitched old+new) &  & {overlay_stats.get('es_years','')} \\\\",
        f"UK official HPI quarterly points (Land Registry) &  & {overlay_stats.get('uk_hpi_q_obs','')} \\\\",
        f"Spain official HPI quarterly points (INE IPV) &  & {overlay_stats.get('es_hpi_q_obs','')} \\\\",
        f"UK official migration quarterly points (ONS LTIM 2020 file) &  & {overlay_stats.get('uk_ons_q_obs','')} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item Notes: Overlay panel replaces Poland and Spain net migration with official annual series where available (Statistics Poland BDL; Spain INE migration tables) and adds UK/Spain official house-price overlays (HM Land Registry UK HPI; INE IPV) plus UK ONS LTIM migration overlap diagnostics. Baseline and overlay models use identical FE controls except for migration-series source in patched country-years.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        "",
    ]
    (PAPER_TABLES_DIR / "tab_country_web_overlay.tex").write_text("\n".join(lines))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    annual_ext, q_ext, pl, meta = build_official_overlays()
    annual_ext.to_parquet(PROC_DIR / "panel_annual_countryweb_overlay.parquet", index=False)
    q_ext.to_parquet(PROC_DIR / "panel_quarterly_countryweb_overlay.parquet", index=False)
    blend_meta = _apply_blend_to_core_streams(annual_ext, q_ext)

    model_df = _fit_annual_overlay_models(annual_ext)
    model_df.to_csv(RESULTS_DIR / "country_web_overlay_coefficients.csv", index=False)

    # Correlation diagnostics for overlap windows.
    pl_overlap = annual_ext[(annual_ext["geo"] == "PL") & annual_ext["pl_net_migration_rate_official_per_1000"].notna() & annual_ext["net_migration_rate"].notna()].copy()
    es_overlap = annual_ext[(annual_ext["geo"] == "ES") & annual_ext["es_net_migration_rate_official_per_1000"].notna() & annual_ext["net_migration_rate"].notna()].copy()
    uk_overlap = q_ext[(q_ext["geo"] == "UK") & q_ext["uk_hpi_yoy_official_q"].notna() & q_ext["hpi_yoy"].notna()].copy()
    es_hpi_overlap = q_ext[(q_ext["geo"] == "ES") & q_ext["es_hpi_yoy_official_q"].notna() & q_ext["hpi_yoy"].notna()].copy()
    uk_ons_overlap = q_ext[(q_ext["geo"] == "UK") & q_ext["uk_ons_net_migration_q"].notna() & q_ext["air_yoy"].notna()].copy()

    overlay_stats = {
        "pl_years": int(pl["year"].nunique()),
        "es_years": int(annual_ext[(annual_ext["geo"] == "ES") & annual_ext["es_net_migration_rate_official_per_1000"].notna()]["year"].nunique()),
        "uk_hpi_q_obs": int(uk_overlap.shape[0]),
        "es_hpi_q_obs": int(es_hpi_overlap.shape[0]),
        "uk_ons_q_obs": int(uk_ons_overlap.shape[0]),
        "pl_corr_eurostat_official": float(pl_overlap[["net_migration_rate", "pl_net_migration_rate_official_per_1000"]].corr().iloc[0, 1]) if len(pl_overlap) >= 3 else np.nan,
        "es_corr_eurostat_official": float(es_overlap[["net_migration_rate", "es_net_migration_rate_official_per_1000"]].corr().iloc[0, 1]) if len(es_overlap) >= 3 else np.nan,
        "uk_hpi_corr_eurostat_official": float(uk_overlap[["hpi_yoy", "uk_hpi_yoy_official_q"]].corr().iloc[0, 1]) if len(uk_overlap) >= 3 else np.nan,
        "es_hpi_corr_eurostat_official": float(es_hpi_overlap[["hpi_yoy", "es_hpi_yoy_official_q"]].corr().iloc[0, 1]) if len(es_hpi_overlap) >= 3 else np.nan,
        "sources": {
            "uk_hpi": meta["uk_hpi_source"],
            "ons_ltim": ONS_LTIM_2020_URL,
            "bdl_api": BDL_UNIT_URL,
            "es_ine_hpi_q": INE_TABLE_URL_TMPL.format(table_id=ES_INE_HPI_Q_TABLE_ID),
            "es_ine_hpi_a": INE_TABLE_URL_TMPL.format(table_id=ES_INE_HPI_A_TABLE_ID),
            "es_ine_imm_old": INE_TABLE_URL_TMPL.format(table_id=ES_INE_IMM_OLD_TABLE_ID),
            "es_ine_emi_old": INE_TABLE_URL_TMPL.format(table_id=ES_INE_EMI_OLD_TABLE_ID),
            "es_ine_imm_new": INE_TABLE_URL_TMPL.format(table_id=ES_INE_IMM_NEW_TABLE_ID),
            "es_ine_emi_new": INE_TABLE_URL_TMPL.format(table_id=ES_INE_EMI_NEW_TABLE_ID),
        },
    }

    write_table(overlay_stats, model_df)
    (META_DIR / "country_web_overlay_summary.json").write_text(json.dumps({**meta, **overlay_stats, **blend_meta}, indent=2))
    pd.DataFrame([
        {
            "series": "PL official vs Eurostat net migration rate",
            "corr": overlay_stats["pl_corr_eurostat_official"],
            "nobs": int(pl_overlap.shape[0]),
        },
        {
            "series": "UK official vs Eurostat HPI YoY",
            "corr": overlay_stats["uk_hpi_corr_eurostat_official"],
            "nobs": int(uk_overlap.shape[0]),
        },
        {
            "series": "ES official vs Eurostat net migration rate",
            "corr": overlay_stats["es_corr_eurostat_official"],
            "nobs": int(es_overlap.shape[0]),
        },
        {
            "series": "ES official vs Eurostat HPI YoY",
            "corr": overlay_stats["es_hpi_corr_eurostat_official"],
            "nobs": int(es_hpi_overlap.shape[0]),
        },
    ]).to_csv(RESULTS_DIR / "country_web_overlay_correlations.csv", index=False)

    print(f"[ok] wrote {PROC_DIR / 'panel_annual_countryweb_overlay.parquet'}")
    print(f"[ok] wrote {RESULTS_DIR / 'country_web_overlay_coefficients.csv'}")
    print(f"[ok] wrote {PAPER_TABLES_DIR / 'tab_country_web_overlay.tex'}")


if __name__ == "__main__":
    main()
