#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd
from linearmodels.panel import PanelOLS
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"linearmodels(\..*)?")


def load_panel(name: str) -> pd.DataFrame:
    parquet_path = PROC_DIR / f"{name}.parquet"
    csv_path = PROC_DIR / f"{name}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(csv_path)


def fit_formula(df: pd.DataFrame, formula: str, label: str):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"linearmodels\..*")
        model = PanelOLS.from_formula(formula, data=df, drop_absorbed=True, check_rank=False)
        result = model.fit(cov_type="clustered", cluster_entity=True)
    return {"label": label, "formula": formula, "result": result}


def coef_frame(model_pack: dict) -> pd.DataFrame:
    res = model_pack["result"]
    out = pd.DataFrame(
        {
            "model": model_pack["label"],
            "term": res.params.index,
            "coef": res.params.values,
            "std_err": res.std_errors.values,
            "t_stat": res.tstats.values,
            "p_value": res.pvalues.values,
        }
    )
    out["nobs"] = float(res.nobs)
    out["r2_within"] = float(getattr(res, "rsquared_within", float("nan")))
    out["r2_overall"] = float(getattr(res, "rsquared_overall", float("nan")))
    return out


def sample_stats(df: pd.DataFrame, vars_to_keep: list[str], panel_name: str) -> pd.DataFrame:
    rows = []
    for col in vars_to_keep:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "panel": panel_name,
                "variable": col,
                "n": int(s.notna().sum()),
                "mean": s.mean(),
                "std": s.std(),
                "p10": s.quantile(0.10),
                "p50": s.quantile(0.50),
                "p90": s.quantile(0.90),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    annual = load_panel("panel_annual_harmonized")
    quarterly = load_panel("panel_quarterly_harmonized")

    annual = annual.replace([np.inf, -np.inf], np.nan)
    quarterly = quarterly.replace([np.inf, -np.inf], np.nan)

    # Harmonized-first canonical variables (fallback to raw where harmonized missing).
    annual["hpi_growth"] = annual.get("hpi_growth_harmonized").combine_first(annual.get("hpi_growth"))
    annual["net_migration_rate"] = annual.get("net_migration_rate_harmonized").combine_first(annual.get("net_migration_rate"))
    annual["air_growth"] = annual.get("air_growth_harmonized").combine_first(annual.get("air_growth"))
    annual["gdp_pc_growth"] = annual.get("gdp_pc_growth_harmonized").combine_first(annual.get("gdp_pc_growth"))
    annual["unemployment_rate"] = annual.get("unemployment_rate_harmonized").combine_first(annual.get("unemployment_rate"))
    annual["inflation_hicp"] = annual.get("inflation_hicp_harmonized").combine_first(annual.get("inflation_hicp"))
    annual["long_rate"] = annual.get("long_rate_harmonized").combine_first(annual.get("long_rate"))
    annual["pop_growth"] = annual.get("pop_growth_harmonized").combine_first(annual.get("pop_growth"))

    ga = annual.sort_values(["geo", "year"]).groupby("geo", sort=False)
    for c in [
        "net_migration_rate",
        "air_growth",
        "gdp_pc_growth",
        "unemployment_rate",
        "inflation_hicp",
        "long_rate",
        "pop_growth",
        "hpi_growth",
    ]:
        annual[f"L1_{c}"] = ga[c].shift(1)

    quarterly["hpi_yoy"] = quarterly.get("hpi_yoy_harmonized").combine_first(quarterly.get("hpi_yoy"))
    gq = quarterly.sort_values(["geo", "year", "quarter"]).groupby("geo", sort=False)
    if "hpi_yoy" in quarterly.columns:
        quarterly["L1_hpi_yoy"] = gq["hpi_yoy"].shift(1)
    if "air_yoy" in quarterly.columns:
        quarterly["L1_air_yoy"] = gq["air_yoy"].shift(1)
        quarterly["L2_air_yoy"] = gq["air_yoy"].shift(2)
    if "air_qoq" in quarterly.columns:
        quarterly["L1_air_qoq"] = gq["air_qoq"].shift(1)
        quarterly["L2_air_qoq"] = gq["air_qoq"].shift(2)

    annual = annual.set_index(["geo", "year"]).sort_index()
    quarterly["quarter_id"] = quarterly["year"].astype(int) * 4 + quarterly["quarter"].astype(int)
    quarterly = quarterly.set_index(["geo", "quarter_id"]).sort_index()

    annual_models = []
    annual_models.append(
        fit_formula(
            annual.dropna(subset=["hpi_growth", "L1_net_migration_rate"]),
            "hpi_growth ~ 1 + L1_net_migration_rate + EntityEffects + TimeEffects",
            "annual_fe_migration",
        )
    )
    annual_models.append(
        fit_formula(
            annual.dropna(subset=["hpi_growth", "L1_net_migration_rate", "L1_air_growth"]),
            "hpi_growth ~ 1 + L1_net_migration_rate + L1_air_growth + EntityEffects + TimeEffects",
            "annual_fe_migration_flights",
        )
    )
    annual_models.append(
        fit_formula(
            annual.dropna(
                subset=[
                    "hpi_growth",
                    "L1_net_migration_rate",
                    "L1_air_growth",
                    "L1_gdp_pc_growth",
                    "L1_unemployment_rate",
                    "L1_inflation_hicp",
                    "L1_long_rate",
                    "L1_pop_growth",
                ]
            ),
            (
                "hpi_growth ~ 1 + L1_net_migration_rate + L1_air_growth + "
                "L1_gdp_pc_growth + L1_unemployment_rate + L1_inflation_hicp + "
                "L1_long_rate + L1_pop_growth + EntityEffects + TimeEffects"
            ),
            "annual_fe_full_controls",
        )
    )

    quarterly_models = []
    quarterly_models.append(
        fit_formula(
            quarterly.dropna(subset=["hpi_yoy", "L1_air_yoy"]),
            "hpi_yoy ~ 1 + L1_air_yoy + EntityEffects + TimeEffects",
            "quarterly_fe_air_yoy",
        )
    )
    quarterly_models.append(
        fit_formula(
            quarterly.dropna(subset=["hpi_yoy", "L1_air_yoy", "L2_air_yoy"]),
            "hpi_yoy ~ 1 + L1_air_yoy + L2_air_yoy + EntityEffects + TimeEffects",
            "quarterly_fe_air_yoy_lags",
        )
    )

    all_models = annual_models + quarterly_models

    coef_tables = [coef_frame(m) for m in all_models]
    pd.concat(coef_tables, ignore_index=True).to_csv(RESULTS_DIR / "model_coefficients.csv", index=False)

    annual_stats = sample_stats(
        annual.reset_index(),
        [
            "hpi_growth",
            "net_migration_rate",
            "air_growth",
            "gdp_pc_growth",
            "unemployment_rate",
            "inflation_hicp",
            "long_rate",
            "pop_growth",
        ],
        "annual",
    )
    quarterly_stats = sample_stats(
        quarterly.reset_index(),
        ["hpi_yoy", "air_yoy", "dlog_hpi_qoq", "air_qoq"],
        "quarterly",
    )
    pd.concat([annual_stats, quarterly_stats], ignore_index=True).to_csv(
        RESULTS_DIR / "sample_stats.csv", index=False
    )

    summary_path = RESULTS_DIR / "model_summaries.txt"
    with summary_path.open("w") as f:
        for pack in all_models:
            f.write(f"# {pack['label']}\n")
            f.write(f"Formula: {pack['formula']}\n\n")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                f.write(str(pack["result"].summary))
            f.write("\n\n" + "=" * 100 + "\n\n")

    print(f"[done] Wrote {summary_path}")
    print(f"[done] Wrote {RESULTS_DIR / 'model_coefficients.csv'}")
    print(f"[done] Wrote {RESULTS_DIR / 'sample_stats.csv'}")


if __name__ == "__main__":
    main()
