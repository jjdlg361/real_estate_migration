#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from build_shiftshare_iv import load_od_migration

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
META_DIR = ROOT / "data" / "metadata"
PAPER_TABLES_DIR = ROOT / "paper_overleaf" / "tables"
PAPER_FIGS_DIR = ROOT / "paper_overleaf" / "figures"
OD_BLEND_PARQUET = PROC_DIR / "od_migration_blended_for_composition.parquet"


def winsorize(s: pd.Series, p_low: float = 0.01, p_high: float = 0.99) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() == 0:
        return x
    lo, hi = x.quantile([p_low, p_high])
    return x.clip(lo, hi)


def stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def build_network_factors() -> pd.DataFrame:
    if OD_BLEND_PARQUET.exists():
        od = pd.read_parquet(OD_BLEND_PARQUET).copy()
    else:
        od = load_od_migration().copy()
    od["immigration_od"] = pd.to_numeric(od["immigration_od"], errors="coerce")
    od = od.dropna(subset=["geo", "origin", "year", "immigration_od"]).copy()
    od = od[od["immigration_od"] >= 0].copy()

    od = (
        od.groupby(["geo", "origin", "year"], as_index=False)["immigration_od"]
        .sum(min_count=1)
        .sort_values(["geo", "origin", "year"])
        .reset_index(drop=True)
    )
    od["network_prev_origin"] = od.groupby(["geo", "origin"], sort=False)["immigration_od"].cumsum() - od["immigration_od"]
    od["network_prev_origin"] = od["network_prev_origin"].clip(lower=0)

    agg = (
        od.groupby(["geo", "year"], as_index=False)
        .agg(
            network_stock_proxy=("network_prev_origin", "sum"),
            network_origins_active=("network_prev_origin", lambda s: int((s > 0).sum())),
        )
        .sort_values(["geo", "year"])
    )

    od = od.merge(agg[["geo", "year", "network_stock_proxy"]], on=["geo", "year"], how="left")
    od["p"] = np.where(od["network_stock_proxy"] > 0, od["network_prev_origin"] / od["network_stock_proxy"], np.nan)
    hhi = od.groupby(["geo", "year"], as_index=False)["p"].apply(lambda s: np.nansum(np.square(s))).rename(columns={"p": "network_hhi"})
    def _shannon(s: pd.Series) -> float:
        x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(x) & (x > 0)
        if not m.any():
            return np.nan
        z = x[m]
        return float(-np.sum(z * np.log(z)))

    shannon = od.groupby(["geo", "year"], as_index=False)["p"].apply(_shannon).rename(columns={"p": "network_shannon"})
    out = agg.merge(hhi, on=["geo", "year"], how="left").merge(shannon, on=["geo", "year"], how="left")
    return out


def fit_model(df: pd.DataFrame, label: str, formula: str, need: list[str]) -> tuple[pd.DataFrame, dict, str] | None:
    x = df.dropna(subset=need).copy()
    if x.empty:
        return None
    x = x.set_index(["geo", "year"]).sort_index()
    res = PanelOLS.from_formula(formula, data=x, drop_absorbed=True, check_rank=False).fit(
        cov_type="clustered", cluster_entity=True, cluster_time=True
    )
    coef = pd.DataFrame(
        {
            "model": label,
            "term": res.params.index,
            "coef": res.params.values,
            "std_err": res.std_errors.values,
            "p_value": res.pvalues.values,
            "t_stat": res.tstats.values,
            "nobs": float(res.nobs),
            "r2_within": float(getattr(res, "rsquared_within", np.nan)),
            "r2_overall": float(getattr(res, "rsquared_overall", np.nan)),
        }
    )
    resid = pd.to_numeric(res.resids, errors="coerce")
    rmse = float(np.sqrt(np.nanmean(np.square(resid))))
    mae = float(np.nanmean(np.abs(resid)))
    stats = {
        "model": label,
        "nobs": float(res.nobs),
        "r2_within": float(getattr(res, "rsquared_within", np.nan)),
        "r2_overall": float(getattr(res, "rsquared_overall", np.nan)),
        "rmse_resid": rmse,
        "mae_resid": mae,
    }
    text = f"# {label}\nFormula: {formula}\n\n{res.summary}\n"
    return coef, stats, text


def coef_cell(df: pd.DataFrame, model: str, term: str) -> tuple[str, str]:
    x = df[(df["model"] == model) & (df["term"] == term)]
    if x.empty:
        return "", ""
    r = x.iloc[0]
    b = f"{r['coef']:.3f}{stars(float(r['p_value']))}"
    se = f"({r['std_err']:.3f})"
    return b, se


def write_table(coef: pd.DataFrame, fit_df: pd.DataFrame) -> None:
    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    models = [
        "annual_fe_dehaas_baseline_matched",
        "annual_fe_dehaas_core",
        "annual_fe_dehaas_core_interactions",
    ]
    headers = ["Matched baseline", "Core factor model", "Factor model + interactions"]
    rows = [
        ("Lagged net migration rate", "L1_net_migration_rate"),
        ("Lagged origin-income hump index", "L1_origin_income_hump"),
        ("Lagged diaspora network stock (per 1k)", "L1_network_stock_per_1000"),
        ("Lagged origin concentration (HHI)", "L1_network_hhi"),
        ("Lagged asylum share in inflows", "L1_asylum_share_inflows"),
        ("Lagged naturalization rate (per 1k)", "L1_naturalization_rate_per_1000"),
        ("Lagged migration x origin-income hump", "L1_net_migration_rate:L1_origin_income_hump"),
    ]

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Migration-system factor framework in annual country FE models}",
        r"\label{tab:dehaas_factors}",
        r"\begin{threeparttable}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{p{5.2cm}p{2.3cm}p{2.3cm}p{2.3cm}}",
        r"\toprule",
        " & " + " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for label, term in rows:
        est = []
        se = []
        for m in models:
            b, s = coef_cell(coef, m, term)
            est.append(b)
            se.append(s)
        lines.append(label + " & " + " & ".join(est) + r" \\")
        lines.append(" & " + " & ".join(se) + r" \\")

    def fval(m: str, c: str, d: int = 3) -> str:
        x = fit_df[fit_df["model"] == m]
        if x.empty or pd.isna(x.iloc[0][c]):
            return ""
        return f"{x.iloc[0][c]:.{d}f}"

    lines += [
        r"\midrule",
        "Country FE & Yes & Yes & Yes \\\\",
        "Year FE & Yes & Yes & Yes \\\\",
        "Macro + air controls & Yes & Yes & Yes \\\\",
        "Observations & " + " & ".join([fval(m, "nobs", 0) for m in models]) + r" \\",
        r"$R^2$ within & " + " & ".join([fval(m, "r2_within", 3) for m in models]) + r" \\",
        "Residual RMSE & " + " & ".join([fval(m, "rmse_resid", 3) for m in models]) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item Notes: Dependent variable is annual house-price growth. Models include country and year fixed effects with clustered SE (country and year), plus lagged macro and air controls. The origin-income hump index is $H_{c,t-1}=s^{mid}_{c,t-1}-0.5(s^{low}_{c,t-1}+s^{high}_{c,t-1})$, where $s$ are destination-year OD inflow shares by origin-income tercile (low/mid/high) built from Eurostat previous-residence OD immigration flows (\texttt{MIGR\_IMM5PRV}) merged with World Bank origin GDP per capita, PPP, constant dollars (\texttt{NY.GDP.PCAP.PP.KD}); tercile cutoffs are year-specific across origins and shares are normalized by GDP-covered OD inflows. Network stock and concentration use lagged cumulative OD previous-residence inflow proxies.",
        r"\item Significance stars: * $p<0.10$, ** $p<0.05$, *** $p<0.01$.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        "",
    ]
    (PAPER_TABLES_DIR / "tab_dehaas_factors.tex").write_text("\n".join(lines))


def write_figure(coef: pd.DataFrame) -> None:
    PAPER_FIGS_DIR.mkdir(parents=True, exist_ok=True)
    keep = [
        "L1_net_migration_rate",
        "L1_origin_income_hump",
        "L1_network_stock_per_1000",
        "L1_network_hhi",
        "L1_asylum_share_inflows",
        "L1_naturalization_rate_per_1000",
    ]
    d = coef[(coef["model"] == "annual_fe_dehaas_core") & (coef["term"].isin(keep))].copy()
    if d.empty:
        return
    d["lo"] = d["coef"] - 1.96 * d["std_err"]
    d["hi"] = d["coef"] + 1.96 * d["std_err"]
    label_map = {
        "L1_net_migration_rate": "Net migration rate",
        "L1_origin_income_hump": "Origin-income hump",
        "L1_network_stock_per_1000": "Network stock per 1k",
        "L1_network_hhi": "Origin concentration (HHI)",
        "L1_asylum_share_inflows": "Asylum share in inflows",
        "L1_naturalization_rate_per_1000": "Naturalization rate per 1k",
    }
    d["label"] = d["term"].map(label_map)
    d = d.sort_values("coef")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = np.arange(len(d))
    ax.errorbar(d["coef"], y, xerr=1.96 * d["std_err"], fmt="o", color="#1f77b4", capsize=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.set_xlabel("Coefficient (95% CI)")
    ax.set_title("Migration-system factors: annual FE core model")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    fig.savefig(PAPER_FIGS_DIR / "fig_dehaas_factors_coefficients.pdf")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PROC_DIR / "panel_annual_extended_channels.parquet").copy()
    h = pd.read_parquet(PROC_DIR / "panel_annual_harmonized.parquet").copy()
    h = h[[
        "geo",
        "year",
        "hpi_growth",
        "hpi_growth_harmonized",
        "net_migration_rate",
        "net_migration_rate_harmonized",
        "net_migration_rate_harmonized_source",
        "air_growth",
        "air_growth_harmonized",
        "gdp_pc_growth",
        "gdp_pc_growth_harmonized",
        "gdp_pc_growth_harmonized_source",
        "unemployment_rate",
        "unemployment_rate_harmonized",
        "unemployment_rate_harmonized_source",
        "inflation_hicp",
        "inflation_hicp_harmonized",
        "inflation_hicp_harmonized_source",
        "long_rate",
        "long_rate_harmonized",
        "long_rate_harmonized_source",
        "pop_growth",
        "pop_growth_harmonized",
        "pop_growth_harmonized_source",
        "population",
    ]].drop_duplicates(["geo", "year"])
    df = df.merge(h, on=["geo", "year"], how="left", suffixes=("", "_h"))

    # Prefer harmonized source tags from the refreshed harmonized panel when available.
    for sc in [
        "net_migration_rate_harmonized_source",
        "gdp_pc_growth_harmonized_source",
        "unemployment_rate_harmonized_source",
        "inflation_hicp_harmonized_source",
        "long_rate_harmonized_source",
        "pop_growth_harmonized_source",
    ]:
        sh = f"{sc}_h"
        if sh in df.columns:
            df[sc] = df[sh].combine_first(df.get(sc))

    # Harmonized-first canonical annual fields (prefer freshly merged *_h values).
    def _pick_num(*cols: str) -> pd.Series:
        s = pd.Series(np.nan, index=df.index, dtype=float)
        for c in cols:
            if c in df.columns:
                s = s.combine_first(pd.to_numeric(df[c], errors="coerce"))
        return s

    df["hpi_growth"] = _pick_num("hpi_growth_harmonized_h", "hpi_growth_harmonized", "hpi_growth_h", "hpi_growth")
    df["net_migration_rate"] = _pick_num(
        "net_migration_rate_harmonized_h",
        "net_migration_rate_harmonized",
        "net_migration_rate_h",
        "net_migration_rate",
    )
    df["air_growth"] = _pick_num("air_growth_harmonized_h", "air_growth_harmonized", "air_growth_h", "air_growth")
    df["gdp_pc_growth"] = _pick_num("gdp_pc_growth_harmonized_h", "gdp_pc_growth_harmonized", "gdp_pc_growth_h", "gdp_pc_growth")
    df["unemployment_rate"] = _pick_num(
        "unemployment_rate_harmonized_h",
        "unemployment_rate_harmonized",
        "unemployment_rate_h",
        "unemployment_rate",
    )
    df["inflation_hicp"] = _pick_num("inflation_hicp_harmonized_h", "inflation_hicp_harmonized", "inflation_hicp_h", "inflation_hicp")
    df["long_rate"] = _pick_num("long_rate_harmonized_h", "long_rate_harmonized", "long_rate_h", "long_rate")
    df["pop_growth"] = _pick_num("pop_growth_harmonized_h", "pop_growth_harmonized", "pop_growth_h", "pop_growth")
    df = df.sort_values(["geo", "year"]).reset_index(drop=True)

    # Recompute core lags from harmonized-first canonical fields.
    g0 = df.groupby("geo", sort=False)
    for c in [
        "net_migration_rate",
        "air_growth",
        "gdp_pc_growth",
        "unemployment_rate",
        "inflation_hicp",
        "long_rate",
        "pop_growth",
    ]:
        df[f"L1_{c}"] = g0[c].shift(1)

    # Migration-system factors: migration-transition hump, network effects, and migration-type composition.
    df["origin_income_hump"] = (
        pd.to_numeric(df["share_mid_gdp_origins"], errors="coerce")
        - 0.5 * pd.to_numeric(df["share_low_gdp_origins"], errors="coerce")
        - 0.5 * pd.to_numeric(df["share_high_gdp_origins"], errors="coerce")
    )
    denom = pd.to_numeric(df["asylum_rate_per_1000"], errors="coerce") + pd.to_numeric(
        df["non_asylum_immigration_rate_per_1000"], errors="coerce"
    )
    df["asylum_share_inflows"] = np.where(denom > 0, pd.to_numeric(df["asylum_rate_per_1000"], errors="coerce") / denom, np.nan)

    net = build_network_factors()
    df = df.merge(net, on=["geo", "year"], how="left")
    pop_use = pd.to_numeric(df.get("population"), errors="coerce")
    pop_use = pop_use.groupby(df["geo"]).ffill().groupby(df["geo"]).bfill()
    df["network_stock_per_1000"] = np.where(
        pop_use > 0,
        pd.to_numeric(df["network_stock_proxy"], errors="coerce") / pop_use * 1000.0,
        np.nan,
    )

    g = df.groupby("geo", sort=False)
    lag_new = [
        "net_migration_rate",
        "origin_income_hump",
        "network_stock_per_1000",
        "network_hhi",
        "network_shannon",
        "asylum_share_inflows",
        "naturalization_rate_per_1000",
    ]
    for c in lag_new:
        df[f"L1_{c}"] = g[c].shift(1)
    to_winsor = [
        "hpi_growth",
        "L1_net_migration_rate",
        "L1_origin_income_hump",
        "L1_network_stock_per_1000",
        "L1_network_hhi",
        "L1_asylum_share_inflows",
        "L1_naturalization_rate_per_1000",
    ]
    for c in to_winsor:
        if c in df.columns:
            df[c] = winsorize(df[c], 0.01, 0.99)

    controls = [
        "L1_air_growth",
        "L1_gdp_pc_growth",
        "L1_unemployment_rate",
        "L1_inflation_hicp",
        "L1_long_rate",
        "L1_pop_growth",
    ]
    core_terms = [
        "L1_net_migration_rate",
        "L1_origin_income_hump",
        "L1_network_stock_per_1000",
        "L1_network_hhi",
        "L1_asylum_share_inflows",
        "L1_naturalization_rate_per_1000",
    ]
    common_need = ["hpi_growth"] + core_terms + controls

    specs = [
        (
            "annual_fe_dehaas_baseline_matched",
            "hpi_growth ~ 1 + L1_net_migration_rate + " + " + ".join(controls) + " + EntityEffects + TimeEffects",
            common_need,
        ),
        (
            "annual_fe_dehaas_core",
            "hpi_growth ~ 1 + " + " + ".join(core_terms) + " + " + " + ".join(controls) + " + EntityEffects + TimeEffects",
            common_need,
        ),
        (
            "annual_fe_dehaas_core_interactions",
            (
                "hpi_growth ~ 1 + "
                + " + ".join(core_terms)
                + " + L1_net_migration_rate:L1_origin_income_hump"
                + " + "
                + " + ".join(controls)
                + " + EntityEffects + TimeEffects"
            ),
            common_need,
        ),
    ]

    coef_list: list[pd.DataFrame] = []
    fit_stats: list[dict] = []
    summary_chunks: list[str] = []
    for label, formula, need in specs:
        out = fit_model(df, label, formula, need)
        if out is None:
            continue
        c, s, t = out
        coef_list.append(c)
        fit_stats.append(s)
        summary_chunks.append(t)

    coef = pd.concat(coef_list, ignore_index=True) if coef_list else pd.DataFrame()
    fit_df = pd.DataFrame(fit_stats)

    coef.to_csv(RESULTS_DIR / "dehaas_factors_coefficients.csv", index=False)
    fit_df.to_csv(RESULTS_DIR / "dehaas_factors_model_fit.csv", index=False)
    (RESULTS_DIR / "dehaas_factors_summaries.txt").write_text("\n\n" + ("\n" + "=" * 100 + "\n\n").join(summary_chunks))
    df.to_parquet(PROC_DIR / "panel_annual_dehaas_factors.parquet", index=False)
    df.to_csv(PROC_DIR / "panel_annual_dehaas_factors.csv", index=False)

    if not coef.empty and not fit_df.empty:
        write_table(coef, fit_df)
        write_figure(coef)

    meta = {
        "rows_panel": int(len(df)),
        "countries": int(df["geo"].astype(str).nunique()),
        "years_min": int(pd.to_numeric(df["year"], errors="coerce").min()),
        "years_max": int(pd.to_numeric(df["year"], errors="coerce").max()),
        "nonmissing_l1_origin_income_hump": int(df["L1_origin_income_hump"].notna().sum()),
        "nonmissing_l1_network_stock_per_1000": int(df["L1_network_stock_per_1000"].notna().sum()),
        "nonmissing_l1_network_hhi": int(df["L1_network_hhi"].notna().sum()),
        "nonmissing_l1_asylum_share_inflows": int(df["L1_asylum_share_inflows"].notna().sum()),
    }
    (META_DIR / "dehaas_factors_summary.json").write_text(json.dumps(meta, indent=2))

    print(f"[ok] wrote {RESULTS_DIR / 'dehaas_factors_coefficients.csv'}")
    print(f"[ok] wrote {PAPER_TABLES_DIR / 'tab_dehaas_factors.tex'}")
    print(f"[ok] wrote {PAPER_FIGS_DIR / 'fig_dehaas_factors_coefficients.pdf'}")


if __name__ == "__main__":
    main()
