#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"


def fit(df: pd.DataFrame, label: str, formula: str, need: list[str]) -> pd.DataFrame:
    x = df.dropna(subset=need).copy().set_index(["geo", "time_id"]).sort_index()
    if x.empty:
        return pd.DataFrame()
    res = PanelOLS.from_formula(formula, data=x, drop_absorbed=True, check_rank=False).fit(
        cov_type="clustered", cluster_entity=True, cluster_time=True
    )
    out = pd.DataFrame(
        {
            "model": label,
            "term": res.params.index,
            "coef": res.params.values,
            "std_err": res.std_errors.values,
            "p_value": res.pvalues.values,
            "nobs": float(res.nobs),
            "r2_within": float(getattr(res, "rsquared_within", np.nan)),
        }
    )
    return out


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    a = pd.read_parquet(PROC_DIR / "panel_annual_harmonized.parquet").copy()
    q = pd.read_parquet(PROC_DIR / "panel_quarterly_harmonized.parquet").copy()

    a["time_id"] = pd.to_numeric(a["year"], errors="coerce")
    q["time_id"] = pd.to_numeric(q["year"], errors="coerce") * 4 + pd.to_numeric(q["quarter"], errors="coerce")
    a = a.replace([np.inf, -np.inf], np.nan)
    q = q.replace([np.inf, -np.inf], np.nan)

    ac = [
        "L1_air_growth_harmonized",
        "gdp_pc_growth_harmonized",
        "unemployment_rate_harmonized",
        "inflation_hicp_harmonized",
        "long_rate_harmonized",
        "pop_growth_harmonized",
    ]
    # lag remaining annual controls
    ga = a.sort_values(["geo", "year"]).groupby("geo", sort=False)
    for c in [
        "hpi_growth_harmonized",
        "gdp_pc_growth_harmonized",
        "unemployment_rate_harmonized",
        "inflation_hicp_harmonized",
        "long_rate_harmonized",
        "pop_growth_harmonized",
    ]:
        a[f"L1_{c}"] = ga[c].shift(1)
    if "L1_air_growth_harmonized" not in a.columns:
        a["L1_air_growth_harmonized"] = ga["air_growth_harmonized"].shift(1)
    gq = q.sort_values(["geo", "time_id"]).groupby("geo", sort=False)
    annual_formula = (
        "hpi_growth_harmonized ~ 1 + L1_net_migration_rate_harmonized + L1_air_growth_harmonized + "
        "L1_gdp_pc_growth_harmonized + L1_unemployment_rate_harmonized + L1_inflation_hicp_harmonized + "
        "L1_long_rate_harmonized + L1_pop_growth_harmonized + "
        "EntityEffects + TimeEffects"
    )
    annual_need = ["hpi_growth_harmonized", "L1_net_migration_rate_harmonized", "L1_air_growth_harmonized"] + [f"L1_{c}" for c in ac[1:]]
    annual_coef = fit(a, "annual_fe_harmonized_full", annual_formula, annual_need)

    quarterly_formula = (
        "hpi_yoy_harmonized ~ 1 + L1_air_yoy + L1y_net_migration_rate_harmonized + L1y_gdp_pc_growth_harmonized + "
        "L1y_unemployment_rate_harmonized + EntityEffects + TimeEffects"
    )
    quarterly_need = [
        "hpi_yoy_harmonized",
        "L1_air_yoy",
        "L1y_net_migration_rate_harmonized",
        "L1y_gdp_pc_growth_harmonized",
        "L1y_unemployment_rate_harmonized",
    ]
    quarterly_coef = fit(q, "quarterly_fe_harmonized_plus_annual_ctrls", quarterly_formula, quarterly_need)

    coef = pd.concat([annual_coef, quarterly_coef], ignore_index=True)
    coef.to_csv(RESULTS_DIR / "model_coefficients_harmonized.csv", index=False)

    # Coverage diagnostics focused on UK/PL in harmonized models.
    cov_rows = []
    for geo in ["UK", "PL"]:
        aa = a[a["geo"] == geo]
        qq = q[q["geo"] == geo]
        cov_rows.append(
            {
                "geo": geo,
                "annual_hpi_nonmissing": int(aa["hpi_growth_harmonized"].notna().sum()),
                "annual_netmig_nonmissing": int(aa["net_migration_rate_harmonized"].notna().sum()),
                "annual_unemp_nonmissing": int(aa["unemployment_rate_harmonized"].notna().sum()),
                "quarterly_hpi_yoy_nonmissing": int(qq["hpi_yoy"].notna().sum()),
                "quarterly_hpi_yoy_harmonized_nonmissing": int(qq["hpi_yoy_harmonized"].notna().sum()),
                "quarterly_air_yoy_nonmissing": int(qq["air_yoy"].notna().sum()),
                "quarterly_l1y_netmig_nonmissing": int(qq["L1y_net_migration_rate_harmonized"].notna().sum()),
            }
        )
    pd.DataFrame(cov_rows).to_csv(RESULTS_DIR / "harmonized_uk_pl_model_coverage.csv", index=False)

    print(f"[ok] wrote {RESULTS_DIR / 'model_coefficients_harmonized.csv'}")
    print(f"[ok] wrote {RESULTS_DIR / 'harmonized_uk_pl_model_coverage.csv'}")


if __name__ == "__main__":
    main()
