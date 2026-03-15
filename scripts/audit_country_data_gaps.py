#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
META_DIR = ROOT / "data" / "metadata"

KEY_ANNUAL_SERIES = {
    "prc_hpi_a_growth": "hpi_growth",
    "tps00019_net_migration_rate": "net_migration_rate",
    "avia_paoc_a_passengers": "air_passengers",
    "tec00115_gdp_pc_growth": "gdp_pc_growth",
    "une_rt_a_unemployment": "unemployment_rate",
    "prc_hicp_aind_inflation": "inflation_hicp",
    "irt_lt_mcby_a_long_rate": "long_rate",
}
TIME_RE = re.compile(r"^\d{4}$")


def _series_geo_coverage(name: str, min_year: int = 2010) -> dict[str, int]:
    df = pd.read_csv(RAW_DIR / f"{name}.csv")
    geo_col = [c for c in df.columns if "\\TIME_PERIOD" in str(c)][0]
    year_cols = [c for c in df.columns if TIME_RE.match(str(c)) and int(c) >= min_year]
    d = df[[geo_col] + year_cols].copy()
    d[geo_col] = d[geo_col].astype(str)
    d = d[d[geo_col].str.match(r"^[A-Z]{2}$", na=False)].copy()
    d["n_nonmissing"] = d[year_cols].apply(pd.to_numeric, errors="coerce").notna().sum(axis=1)
    return d.set_index(geo_col)["n_nonmissing"].to_dict()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    cov = {src: _series_geo_coverage(src, min_year=2010) for src in KEY_ANNUAL_SERIES}
    geos = sorted({g for d in cov.values() for g in d.keys()})

    rows = []
    for geo in geos:
        row = {"geo": geo}
        for src, label in KEY_ANNUAL_SERIES.items():
            row[f"n_{label}_2010plus"] = int(cov[src].get(geo, 0))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("geo").reset_index(drop=True)
    out["core_min_obs_2010plus"] = out[[c for c in out.columns if c.startswith("n_")]].min(axis=1)
    out["core_mean_obs_2010plus"] = out[[c for c in out.columns if c.startswith("n_")]].mean(axis=1)

    # Candidate additions: countries with decent core coverage in the three headline channels.
    headline_cols = ["n_hpi_growth_2010plus", "n_net_migration_rate_2010plus", "n_air_passengers_2010plus"]
    out["headline_min_obs_2010plus"] = out[headline_cols].min(axis=1)
    add_candidates = out[out["headline_min_obs_2010plus"] >= 8].copy()
    near_candidates = out[out["headline_min_obs_2010plus"] >= 7].copy()

    # Current target list inferred from blended harmonized panel.
    panel_h = PROC_DIR / "panel_annual_harmonized.parquet"
    if not panel_h.exists():
        raise FileNotFoundError(
            "Missing `panel_annual_harmonized.parquet`. Run scripts/harmonize_cross_frequency.py first."
        )
    panel_annual = pd.read_parquet(panel_h)
    current_targets = sorted(panel_annual["geo"].dropna().astype(str).unique().tolist())
    missing_candidates = add_candidates[~add_candidates["geo"].isin(current_targets)].copy()

    out.to_csv(RESULTS_DIR / "country_data_gap_audit.csv", index=False)

    recommendations = {
        "current_targets_n": len(current_targets),
        "current_targets": current_targets,
        "candidate_missing_geos": missing_candidates["geo"].tolist(),
        "candidate_missing_summary": missing_candidates[
            ["geo", "headline_min_obs_2010plus", "n_long_rate_2010plus", "n_unemployment_rate_2010plus"]
        ].to_dict(orient="records"),
        "near_candidate_missing_geos": near_candidates[~near_candidates["geo"].isin(current_targets)]["geo"].tolist(),
        "near_candidate_missing_summary": near_candidates[~near_candidates["geo"].isin(current_targets)][
            ["geo", "headline_min_obs_2010plus", "n_long_rate_2010plus", "n_unemployment_rate_2010plus"]
        ].to_dict(orient="records"),
        "priority_data_fixes": [
            {
                "geo": "UK",
                "issue": "Weak annual Eurostat unemployment and migration depth in this sample.",
                "recommended_official_source": "ONS API time series (labor + migration releases) with transparent concept mapping.",
            },
            {
                "geo": "PL",
                "issue": "Eurostat and official BDL migration series diverge in overlap window.",
                "recommended_official_source": "Keep BDL patch with documented definition bridge; avoid naive one-to-one replacement beyond overlap checks.",
            },
            {
                "geo": "IS/NO/CH",
                "issue": "Long-rate control missing/short in Eurostat stream.",
                "recommended_official_source": "OECD/central-bank long-term rates as explicit control fallback.",
            },
        ],
    }
    (META_DIR / "country_data_gap_recommendations.json").write_text(json.dumps(recommendations, indent=2))

    print(f"[ok] wrote {RESULTS_DIR / 'country_data_gap_audit.csv'}")
    print(f"[ok] wrote {META_DIR / 'country_data_gap_recommendations.json'}")


if __name__ == "__main__":
    main()
