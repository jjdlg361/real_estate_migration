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

TIME_RE = re.compile(r"^\d{4}$")
COUNTRY2_RE = re.compile(r"^[A-Z]{2}$")
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
EUROSTAT_TO_WB2 = {"EL": "GR", "UK": "GB"}
WB2_TO_EUROSTAT = {v: k for k, v in EUROSTAT_TO_WB2.items()}


def eurostat_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    geo_cols = [c for c in df.columns if "\\TIME_PERIOD" in str(c)]
    if len(geo_cols) != 1:
        raise ValueError(f"Expected one geo/time col, got {geo_cols}")
    geo_col = geo_cols[0]
    time_cols = [c for c in df.columns if TIME_RE.match(str(c))]
    id_cols = [c for c in df.columns if c not in time_cols]
    long_df = df.melt(id_vars=id_cols, value_vars=time_cols, var_name="year", value_name="value")
    long_df = long_df.rename(columns={geo_col: "geo"})
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["year"]).copy()
    long_df["year"] = long_df["year"].astype(int)
    return long_df


def load_od_migration() -> pd.DataFrame:
    path = RAW_DIR / "migr_imm5prv_od.csv"
    df = pd.read_csv(path)
    long_df = eurostat_wide_to_long(df)
    keep = (
        (long_df["age"] == "TOTAL")
        & (long_df["agedef"] == "REACH")
        & (long_df["unit"] == "NR")
        & (long_df["sex"] == "T")
    )
    long_df = long_df.loc[keep, ["geo", "partner", "year", "value"]].copy()
    long_df = long_df.rename(columns={"partner": "origin", "value": "immigration_od"})
    long_df["geo"] = long_df["geo"].astype(str)
    long_df["origin"] = long_df["origin"].astype(str)
    long_df = long_df[long_df["geo"].isin(TARGET_GEOS)].copy()
    long_df = long_df[long_df["origin"].str.match(COUNTRY2_RE, na=False)].copy()
    long_df = long_df[long_df["origin"] != long_df["geo"]].copy()
    # Deduplicate if any accidental repeated rows remain
    long_df = (
        long_df.groupby(["geo", "origin", "year"], as_index=False)["immigration_od"]
        .sum(min_count=1)
        .sort_values(["geo", "origin", "year"])
        .reset_index(drop=True)
    )
    return long_df


def load_asylum_od() -> pd.DataFrame:
    path = RAW_DIR / "migr_asyappctza_od.csv"
    if not path.exists():
        return pd.DataFrame(columns=["geo", "origin", "year", "asylum_apps_od"])
    df = pd.read_csv(path)
    long_df = eurostat_wide_to_long(df)
    keep = (
        (long_df["applicant"] == "FRST")
        & (long_df["age"] == "TOTAL")
        & (long_df["sex"] == "T")
        & (long_df["unit"] == "PER")
    )
    # citizen is the origin/citizenship dimension here
    long_df = long_df.loc[keep, ["geo", "citizen", "year", "value"]].copy()
    long_df = long_df.rename(columns={"citizen": "origin", "value": "asylum_apps_od"})
    long_df["geo"] = long_df["geo"].astype(str)
    long_df["origin"] = long_df["origin"].astype(str)
    long_df = long_df[long_df["geo"].isin(TARGET_GEOS)].copy()
    long_df = long_df[long_df["origin"].str.match(COUNTRY2_RE, na=False)].copy()
    long_df = long_df[long_df["origin"] != long_df["geo"]].copy()
    long_df = (
        long_df.groupby(["geo", "origin", "year"], as_index=False)["asylum_apps_od"]
        .sum(min_count=1)
        .sort_values(["geo", "origin", "year"])
        .reset_index(drop=True)
    )
    return long_df


def load_world_bank_push() -> pd.DataFrame:
    path = RAW_DIR / "worldbank_push_shocks_long.csv"
    wb = pd.read_csv(path)
    wb["year"] = pd.to_numeric(wb["year"], errors="coerce")
    wb["value"] = pd.to_numeric(wb["value"], errors="coerce")
    wb = wb.dropna(subset=["year"]).copy()
    wb["year"] = wb["year"].astype(int)

    # Prefer 2-letter WB code; map to Eurostat country code style where needed.
    wb["origin_wb2"] = wb["country_code_wb2"].astype(str)
    wb["origin"] = wb["origin_wb2"].map(WB2_TO_EUROSTAT).fillna(wb["origin_wb2"])
    wb = wb[wb["origin"].str.match(COUNTRY2_RE, na=False)].copy()

    wide = wb.pivot_table(
        index=["origin", "year"],
        columns="indicator_alias",
        values="value",
        aggfunc="first",
    ).reset_index()
    for c in ["wb_gdp_pc_growth", "wb_battle_deaths", "wb_population", "wb_unemployment"]:
        if c not in wide.columns:
            wide[c] = np.nan
    wide["wb_battle_deaths_per_million"] = np.where(
        wide["wb_population"] > 0,
        wide["wb_battle_deaths"] / wide["wb_population"] * 1_000_000.0,
        np.nan,
    )
    wide["push_gdp_downturn"] = -wide["wb_gdp_pc_growth"]
    wide["push_conflict_log"] = np.log1p(wide["wb_battle_deaths_per_million"].clip(lower=0))
    wide["push_unemp"] = wide["wb_unemployment"]

    for col in ["push_gdp_downturn", "push_conflict_log", "push_unemp"]:
        mu = wide.groupby("year")[col].transform("mean")
        sd = wide.groupby("year")[col].transform("std")
        wide[f"{col}_z"] = (wide[col] - mu) / sd.replace(0, np.nan)
    wide["push_index_wb"] = (
        wide["push_gdp_downturn_z"].fillna(0)
        + wide["push_conflict_log_z"].fillna(0)
        + 0.5 * wide["push_unemp_z"].fillna(0)
    )
    return wide


def build_shiftshare(
    od: pd.DataFrame,
    wb_push: pd.DataFrame,
    asylum_od: pd.DataFrame | None = None,
    base_start: int = 2005,
    base_end: int = 2009,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Base shares by destination-country and origin-country
    base = od[(od["year"] >= base_start) & (od["year"] <= base_end)].copy()
    base = base.dropna(subset=["immigration_od"])
    base_mean = (
        base.groupby(["geo", "origin"], as_index=False)["immigration_od"]
        .mean()
        .rename(columns={"immigration_od": "base_mean_od"})
    )
    dest_tot = base_mean.groupby("geo", as_index=False)["base_mean_od"].sum().rename(
        columns={"base_mean_od": "base_total_dest"}
    )
    shares = base_mean.merge(dest_tot, on="geo", how="left")
    shares["share_base"] = np.where(
        shares["base_total_dest"] > 0, shares["base_mean_od"] / shares["base_total_dest"], np.nan
    )
    shares = shares[(shares["share_base"] > 0) & shares["share_base"].notna()].copy()

    # Leave-one-out origin supply shocks using Eurostat OD flows
    od_nonmissing = od.dropna(subset=["immigration_od"]).copy()
    origin_year_total = (
        od_nonmissing.groupby(["origin", "year"], as_index=False)["immigration_od"]
        .sum()
        .rename(columns={"immigration_od": "origin_year_inflow_total"})
    )
    od_with_supply = od.merge(origin_year_total, on=["origin", "year"], how="left")
    od_with_supply["origin_supply_loo"] = (
        od_with_supply["origin_year_inflow_total"] - od_with_supply["immigration_od"].fillna(0)
    )

    ss = od_with_supply.merge(shares[["geo", "origin", "share_base"]], on=["geo", "origin"], how="left")
    ss = ss[ss["share_base"].notna()].copy()
    ss["ss_component_loo_supply"] = ss["share_base"] * ss["origin_supply_loo"]
    ss_loo = (
        ss.groupby(["geo", "year"], as_index=False)["ss_component_loo_supply"]
        .sum(min_count=1)
        .rename(columns={"ss_component_loo_supply": "ss_loo_origin_supply"})
    )

    # World Bank origin push shocks
    ss_wb = od.merge(shares[["geo", "origin", "share_base"]], on=["geo", "origin"], how="inner")
    ss_wb = ss_wb.merge(wb_push, on=["origin", "year"], how="left")
    for col in ["push_index_wb", "push_gdp_downturn", "push_conflict_log", "push_unemp"]:
        ss_wb[f"comp_{col}"] = ss_wb["share_base"] * ss_wb[col]
    wb_agg = (
        ss_wb.groupby(["geo", "year"], as_index=False)[
            [f"comp_{c}" for c in ["push_index_wb", "push_gdp_downturn", "push_conflict_log", "push_unemp"]]
        ]
        .sum(min_count=1)
        .rename(
            columns={
                "comp_push_index_wb": "ss_push_index_wb",
                "comp_push_gdp_downturn": "ss_push_gdp_downturn",
                "comp_push_conflict_log": "ss_push_conflict_log",
                "comp_push_unemp": "ss_push_unemployment",
            }
        )
    )

    out = ss_loo.merge(wb_agg, on=["geo", "year"], how="outer")

    # Asylum-specific origin shocks (refugee/asylum-driven push proxy within Eurostat)
    if asylum_od is not None and not asylum_od.empty:
        asy = asylum_od.merge(shares[["geo", "origin", "share_base"]], on=["geo", "origin"], how="inner")
        asy_nonmissing = asy.dropna(subset=["asylum_apps_od"]).copy()
        asy_origin_total = (
            asy_nonmissing.groupby(["origin", "year"], as_index=False)["asylum_apps_od"]
            .sum()
            .rename(columns={"asylum_apps_od": "origin_year_asylum_total"})
        )
        asy = asy.merge(asy_origin_total, on=["origin", "year"], how="left")
        asy["origin_asylum_loo"] = asy["origin_year_asylum_total"] - asy["asylum_apps_od"].fillna(0)
        asy["comp_asylum_loo"] = asy["share_base"] * asy["origin_asylum_loo"]
        asy_agg = (
            asy.groupby(["geo", "year"], as_index=False)["comp_asylum_loo"]
            .sum(min_count=1)
            .rename(columns={"comp_asylum_loo": "ss_asylum_loo"})
        )
        out = out.merge(asy_agg, on=["geo", "year"], how="outer")

    out = out.sort_values(["geo", "year"]).reset_index(drop=True)
    out["ss_loo_origin_supply_logdiff"] = (
        out.groupby("geo")["ss_loo_origin_supply"].transform(lambda s: np.log(s.where(s > 0)).diff() * 100.0)
    )
    if "ss_asylum_loo" in out.columns:
        out["ss_asylum_loo_logdiff"] = out.groupby("geo")["ss_asylum_loo"].transform(
            lambda s: np.log1p(s.clip(lower=0)).diff() * 100.0
        )

    share_diag = shares.copy().sort_values(["geo", "share_base"], ascending=[True, False]).reset_index(drop=True)
    return out, share_diag


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    od = load_od_migration()
    wb_push = load_world_bank_push()
    asylum_od = load_asylum_od()
    iv_panel, share_diag = build_shiftshare(od, wb_push, asylum_od=asylum_od)

    annual_path = PROC_DIR / "panel_annual_harmonized.parquet"
    if not annual_path.exists():
        raise FileNotFoundError(
            "Missing `panel_annual_harmonized.parquet`. Run scripts/harmonize_cross_frequency.py first."
        )
    annual = pd.read_parquet(annual_path)
    annual["hpi_growth"] = pd.to_numeric(annual.get("hpi_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(annual.get("hpi_growth"), errors="coerce")
    )
    annual["net_migration_rate"] = pd.to_numeric(
        annual.get("net_migration_rate_harmonized"), errors="coerce"
    ).combine_first(pd.to_numeric(annual.get("net_migration_rate"), errors="coerce"))
    annual_iv = annual.merge(iv_panel, on=["geo", "year"], how="left")
    annual_iv = annual_iv.sort_values(["geo", "year"]).reset_index(drop=True)
    for col in [
        "ss_loo_origin_supply",
        "ss_loo_origin_supply_logdiff",
        "ss_push_index_wb",
        "ss_push_gdp_downturn",
        "ss_push_conflict_log",
        "ss_push_unemployment",
        "ss_asylum_loo",
        "ss_asylum_loo_logdiff",
    ]:
        if col in annual_iv.columns:
            annual_iv[f"L1_{col}"] = annual_iv.groupby("geo")[col].shift(1)

    annual_iv.to_csv(PROC_DIR / "panel_annual_iv.csv", index=False)
    annual_iv.to_parquet(PROC_DIR / "panel_annual_iv.parquet", index=False)
    iv_panel.to_csv(PROC_DIR / "shiftshare_country_year_iv.csv", index=False)
    share_diag.to_csv(META_DIR / "shiftshare_base_shares_top.csv", index=False)

    summary = {
        "od_rows": int(len(od)),
        "od_destinations": int(od["geo"].nunique()),
        "od_origins": int(od["origin"].nunique()),
        "od_year_min": int(od["year"].min()),
        "od_year_max": int(od["year"].max()),
        "iv_rows": int(len(iv_panel)),
        "iv_nonmissing_ss_loo": int(iv_panel["ss_loo_origin_supply"].notna().sum()),
        "iv_nonmissing_ss_push_index": int(iv_panel["ss_push_index_wb"].notna().sum()),
        "iv_nonmissing_ss_asylum_loo": int(iv_panel["ss_asylum_loo"].notna().sum()) if "ss_asylum_loo" in iv_panel else 0,
        "annual_iv_rows": int(len(annual_iv)),
        "annual_iv_baseline_complete": int(
            annual_iv[["hpi_growth", "net_migration_rate", "ss_push_index_wb"]].dropna().shape[0]
        ),
    }
    (META_DIR / "shiftshare_iv_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("[done] Wrote IV panel and merged annual panel")


if __name__ == "__main__":
    main()
