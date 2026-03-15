#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
TABLE_DIRS = [
    ROOT / "paper_overleaf" / "tables",
    ROOT / "paper_overleaf_v45_slim_backup_20260305_200335" / "tables",
]


def stars(p: float | None) -> str:
    if p is None or pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def _fit(df: pd.DataFrame, formula: str) -> tuple[pd.DataFrame, object]:
    res = PanelOLS.from_formula(formula, data=df, drop_absorbed=True, check_rank=False).fit(
        cov_type="clustered", cluster_entity=True, cluster_time=True
    )
    out = pd.DataFrame(
        {
            "term": res.params.index,
            "coef": res.params.values,
            "std_err": res.std_errors.values,
            "p_value": res.pvalues.values,
        }
    )
    out["nobs"] = float(res.nobs)
    out["r2_within"] = float(getattr(res, "rsquared_within", np.nan))
    return out, res


def _term(df: pd.DataFrame, name: str) -> tuple[float | None, float | None, float | None]:
    x = df[df["term"] == name]
    if x.empty:
        return None, None, None
    r = x.iloc[0]
    return float(r["coef"]), float(r["std_err"]), float(r["p_value"])


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    panel_path = PROC_DIR / "panel_annual_harmonized.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing panel: {panel_path}")

    a = pd.read_parquet(panel_path).copy()
    a = a.replace([np.inf, -np.inf], np.nan)
    a["time_id"] = pd.to_numeric(a["year"], errors="coerce")
    a = a.sort_values(["geo", "year"])

    # Ensure lag columns exist (same convention as main harmonized estimation script).
    g = a.groupby("geo", sort=False)
    for c in [
        "hpi_growth_harmonized",
        "gdp_pc_growth_harmonized",
        "unemployment_rate_harmonized",
        "inflation_hicp_harmonized",
        "long_rate_harmonized",
        "pop_growth_harmonized",
    ]:
        lag_col = f"L1_{c}"
        if lag_col not in a.columns:
            a[lag_col] = g[c].shift(1)
    if "L1_air_growth_harmonized" not in a.columns:
        a["L1_air_growth_harmonized"] = g["air_growth_harmonized"].shift(1)
    if "L1_net_migration_rate_harmonized" not in a.columns:
        a["L1_net_migration_rate_harmonized"] = g["net_migration_rate_harmonized"].shift(1)

    need = [
        "hpi_growth_harmonized",
        "L1_net_migration_rate_harmonized",
        "L1_air_growth_harmonized",
        "L1_gdp_pc_growth_harmonized",
        "L1_unemployment_rate_harmonized",
        "L1_inflation_hicp_harmonized",
        "L1_long_rate_harmonized",
        "L1_pop_growth_harmonized",
    ]
    x = a.dropna(subset=need).copy()
    # Year-specific median split, mirroring lottery-paper heterogeneity style.
    x["fast_pop_growth"] = (
        x["L1_pop_growth_harmonized"]
        > x.groupby("year")["L1_pop_growth_harmonized"].transform("median")
    ).astype(float)
    x["mig_x_fast_pop"] = x["L1_net_migration_rate_harmonized"] * x["fast_pop_growth"]
    x = x.set_index(["geo", "time_id"]).sort_index()

    controls = (
        "L1_air_growth_harmonized + L1_gdp_pc_growth_harmonized + "
        "L1_unemployment_rate_harmonized + L1_inflation_hicp_harmonized + L1_long_rate_harmonized + "
        "L1_pop_growth_harmonized"
    )
    f_base = f"hpi_growth_harmonized ~ 1 + L1_net_migration_rate_harmonized + {controls} + EntityEffects + TimeEffects"
    f_int = (
        "hpi_growth_harmonized ~ 1 + L1_net_migration_rate_harmonized + fast_pop_growth + mig_x_fast_pop + "
        f"{controls} + EntityEffects + TimeEffects"
    )

    slow = x[x["fast_pop_growth"] == 0.0]
    fast = x[x["fast_pop_growth"] == 1.0]

    slow_coef, slow_res = _fit(slow, f_base)
    fast_coef, fast_res = _fit(fast, f_base)
    int_coef, int_res = _fit(x, f_int)

    slow_coef["model"] = "annual_hetero_pop_slow"
    fast_coef["model"] = "annual_hetero_pop_fast"
    int_coef["model"] = "annual_hetero_pop_interaction"
    coef_all = pd.concat([slow_coef, fast_coef, int_coef], ignore_index=True)
    coef_path = RESULTS_DIR / "h1b_style_population_heterogeneity_coefficients.csv"
    coef_all.to_csv(coef_path, index=False)

    b_slow, se_slow, p_slow = _term(slow_coef, "L1_net_migration_rate_harmonized")
    b_fast, se_fast, p_fast = _term(fast_coef, "L1_net_migration_rate_harmonized")
    b_base, se_base, p_base = _term(int_coef, "L1_net_migration_rate_harmonized")
    b_int, se_int, p_int = _term(int_coef, "mig_x_fast_pop")

    mig_sd = float(x["L1_net_migration_rate_harmonized"].std())
    y_abs_mean = float(x["hpi_growth_harmonized"].abs().mean())
    implied_base = np.nan if b_base is None else b_base * mig_sd
    implied_fast = np.nan if (b_base is None or b_int is None) else (b_base + b_int) * mig_sd
    share_base = np.nan if y_abs_mean == 0 or np.isnan(implied_base) else implied_base / y_abs_mean
    share_fast = np.nan if y_abs_mean == 0 or np.isnan(implied_fast) else implied_fast / y_abs_mean

    summary = pd.DataFrame(
        [
            {
                "migration_sd": mig_sd,
                "mean_abs_hpi_growth": y_abs_mean,
                "implied_pp_effect_base_1sd": implied_base,
                "implied_pp_effect_fast_1sd": implied_fast,
                "implied_share_of_mean_abs_base": share_base,
                "implied_share_of_mean_abs_fast": share_fast,
                "nobs_slow": float(slow_res.nobs),
                "nobs_fast": float(fast_res.nobs),
                "nobs_interaction": float(int_res.nobs),
            }
        ]
    )
    summary_path = RESULTS_DIR / "h1b_style_population_heterogeneity_summary.csv"
    summary.to_csv(summary_path, index=False)

    table_tex = f"""
\\begin{{table}}[!htbp]
\\centering
\\caption{{Heterogeneity check (H-1B style): migration effect by population-growth regime}}
\\label{{tab:h1b_style_heterogeneity}}
\\begin{{threeparttable}}
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{p{{5.8cm}}ccc}}
\\toprule
 & Slow-pop-growth countries & Fast-pop-growth countries & Full sample + interaction \\\\
\\midrule
Lagged net migration rate & {'' if b_slow is None else f"{b_slow:.3f}{stars(p_slow)}"} & {'' if b_fast is None else f"{b_fast:.3f}{stars(p_fast)}"} & {'' if b_base is None else f"{b_base:.3f}{stars(p_base)}"} \\\\
 & {'' if se_slow is None else f"({se_slow:.3f})"} & {'' if se_fast is None else f"({se_fast:.3f})"} & {'' if se_base is None else f"({se_base:.3f})"} \\\\
Lagged net migration $\\times$ fast-pop-growth &  &  & {'' if b_int is None else f"{b_int:.3f}{stars(p_int)}"} \\\\
 &  &  & {'' if se_int is None else f"({se_int:.3f})"} \\\\
\\midrule
Country FE & Yes & Yes & Yes \\\\
Year FE & Yes & Yes & Yes \\\\
Macro + air controls & Yes & Yes & Yes \\\\
Observations & {int(round(float(slow_res.nobs))):,} & {int(round(float(fast_res.nobs))):,} & {int(round(float(int_res.nobs))):,} \\\\
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}[flushleft]
\\footnotesize
\\item Notes: Annual FE models on harmonized country-year panel. ``Fast-pop-growth'' is an indicator for countries above the year-specific median lagged population growth. Controls match the annual harmonized baseline (lagged air growth and lagged macro controls). A one-standard-deviation migration change in this sample is {mig_sd:.2f} net migrants per 1,000 residents.
\\item Economic scale (interaction model): implied effect of a 1-SD migration increase is {implied_base:.3f} pp in baseline-pop-growth regimes and {implied_fast:.3f} pp in fast-pop-growth regimes (about {100*share_base:.1f}\\% and {100*share_fast:.1f}\\% of mean absolute annual house-price growth, respectively).
\\item Significance stars: * $p<0.10$, ** $p<0.05$, *** $p<0.01$.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
""".strip() + "\n"

    for td in TABLE_DIRS:
        if td.exists():
            td.mkdir(parents=True, exist_ok=True)
            (td / "tab_h1b_style_heterogeneity.tex").write_text(table_tex)

    print(f"[ok] wrote {coef_path}")
    print(f"[ok] wrote {summary_path}")
    for td in TABLE_DIRS:
        if td.exists():
            print(f"[ok] wrote {td / 'tab_h1b_style_heterogeneity.tex'}")


if __name__ == "__main__":
    main()
