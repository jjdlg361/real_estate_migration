#!/usr/bin/env python3
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS
from linearmodels.panel import PanelOLS


warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"linearmodels(\..*)?")

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"


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


def save_text(path: Path, chunks: list[str]) -> None:
    path.write_text("\n\n".join(chunks))


def _country_iv_sample() -> pd.DataFrame:
    df = load_parquet_or_csv("panel_annual_iv")
    h = load_parquet_or_csv("panel_annual_harmonized")
    df = df.replace([np.inf, -np.inf], np.nan).copy()
    h = h.replace([np.inf, -np.inf], np.nan).copy()

    h_keep = [
        "geo",
        "year",
        "hpi_growth",
        "hpi_growth_harmonized",
        "net_migration_rate",
        "net_migration_rate_harmonized",
        "air_growth",
        "air_growth_harmonized",
        "gdp_pc_growth",
        "gdp_pc_growth_harmonized",
        "unemployment_rate",
        "unemployment_rate_harmonized",
        "inflation_hicp",
        "inflation_hicp_harmonized",
        "long_rate",
        "long_rate_harmonized",
        "pop_growth",
        "pop_growth_harmonized",
    ]
    h_keep = [c for c in h_keep if c in h.columns]
    h = h[h_keep].drop_duplicates(["geo", "year"])
    df = df.merge(h, on=["geo", "year"], how="left", suffixes=("", "_h"))

    # Harmonized-first canonical fields.
    df["hpi_growth"] = pd.to_numeric(df.get("hpi_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(df.get("hpi_growth"), errors="coerce")
    )
    df["net_migration_rate"] = pd.to_numeric(df.get("net_migration_rate_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(df.get("net_migration_rate"), errors="coerce")
    )
    df["air_growth"] = pd.to_numeric(df.get("air_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(df.get("air_growth"), errors="coerce")
    )
    df["gdp_pc_growth"] = pd.to_numeric(df.get("gdp_pc_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(df.get("gdp_pc_growth"), errors="coerce")
    )
    df["unemployment_rate"] = pd.to_numeric(df.get("unemployment_rate_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(df.get("unemployment_rate"), errors="coerce")
    )
    df["inflation_hicp"] = pd.to_numeric(df.get("inflation_hicp_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(df.get("inflation_hicp"), errors="coerce")
    )
    df["long_rate"] = pd.to_numeric(df.get("long_rate_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(df.get("long_rate"), errors="coerce")
    )
    df["pop_growth"] = pd.to_numeric(df.get("pop_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(df.get("pop_growth"), errors="coerce")
    )

    df = df.sort_values(["geo", "year"]).reset_index(drop=True)
    g = df.groupby("geo", sort=False)
    for c in [
        "hpi_growth",
        "net_migration_rate",
        "air_growth",
        "gdp_pc_growth",
        "unemployment_rate",
        "inflation_hicp",
        "long_rate",
        "pop_growth",
    ]:
        df[f"L1_{c}"] = g[c].shift(1)
    keep = [
        "geo",
        "year",
        "hpi_growth",
        "L1_net_migration_rate",
        "L1_air_growth",
        "L1_gdp_pc_growth",
        "L1_unemployment_rate",
        "L1_inflation_hicp",
        "L1_long_rate",
        "L1_pop_growth",
        "L1_ss_push_index_wb",
        "L1_ss_loo_origin_supply_logdiff",
        "L1_ss_asylum_loo_logdiff",
    ]
    d = df[keep].copy()
    # Winsorize volatile variables (pandemic years produce extreme growth rates).
    for col in ["hpi_growth", "L1_air_growth", "L1_ss_loo_origin_supply_logdiff"]:
        if col in d.columns:
            d[col] = winsorize_series(d[col], 0.01, 0.99)
    return d


def _fit_country_iv_with_instruments(d: pd.DataFrame, instrument_terms: list[str], label: str) -> tuple[pd.DataFrame, str]:
    needed = [
        "hpi_growth",
        "L1_net_migration_rate",
        "L1_air_growth",
        "L1_gdp_pc_growth",
        "L1_unemployment_rate",
        "L1_inflation_hicp",
        "L1_long_rate",
        "L1_pop_growth",
    ] + instrument_terms
    d = d.dropna(subset=[c for c in needed if c in d.columns]).copy()
    iv_formula = (
        "hpi_growth ~ 1 + L1_air_growth + L1_gdp_pc_growth + L1_unemployment_rate + "
        "L1_inflation_hicp + L1_long_rate + L1_pop_growth + C(geo) + C(year) "
        f"[L1_net_migration_rate ~ {' + '.join(instrument_terms)}]"
    )
    model = IV2SLS.from_formula(iv_formula, data=d)
    res = model.fit(cov_type="clustered", clusters=d["geo"])

    coef = pd.DataFrame(
        {
            "model": [label] * len(res.params),
            "term": res.params.index,
            "coef": res.params.values,
            "std_err": res.std_errors.values,
            "t_stat": res.tstats.values,
            "p_value": res.pvalues.values,
            "nobs": float(res.nobs),
        }
    )
    out_text = f"# {label}\n" + str(res.summary)
    return coef, out_text


def run_country_iv() -> tuple[pd.DataFrame, list[str]]:
    d = _country_iv_sample()
    outputs = []
    texts = []

    # Composite/WB-style IV (with optional second instrument if present)
    wb_instruments = ["L1_ss_push_index_wb"]
    if "L1_ss_loo_origin_supply_logdiff" in d.columns and d["L1_ss_loo_origin_supply_logdiff"].notna().sum() > 100:
        wb_instruments.append("L1_ss_loo_origin_supply_logdiff")
    coef_wb, txt_wb = _fit_country_iv_with_instruments(d, wb_instruments, "country_iv_fe")
    outputs.append(coef_wb)
    texts.append(txt_wb)

    # Asylum-specific IV (cleaner exclusion story, single instrument)
    if "L1_ss_asylum_loo_logdiff" in d.columns and d["L1_ss_asylum_loo_logdiff"].notna().sum() > 100:
        coef_asy, txt_asy = _fit_country_iv_with_instruments(d, ["L1_ss_asylum_loo_logdiff"], "country_iv_fe_asylum")
        outputs.append(coef_asy)
        texts.append(txt_asy)

    return pd.concat(outputs, ignore_index=True), texts


def run_quarterly_route_shock_fe() -> tuple[pd.DataFrame, list[str]]:
    df = load_parquet_or_csv("panel_quarterly_airport_shocks")
    qh = load_parquet_or_csv("panel_quarterly_harmonized")
    df = df.replace([np.inf, -np.inf], np.nan).copy()
    qh = qh.replace([np.inf, -np.inf], np.nan).copy()
    if "period_str" not in df.columns and "period" in df.columns:
        df["period_str"] = df["period"].astype(str)
    if "period_str" not in qh.columns:
        if "period" in qh.columns:
            qh["period_str"] = qh["period"].astype(str)
        else:
            qh["period_str"] = qh["year"].astype(int).astype(str) + "Q" + qh["quarter"].astype(int).astype(str)

    qh_keep = ["geo", "period_str", "hpi_yoy", "hpi_yoy_harmonized", "air_yoy"]
    qh_keep = [c for c in qh_keep if c in qh.columns]
    qh = qh[qh_keep].drop_duplicates(["geo", "period_str"])
    df = df.merge(qh, on=["geo", "period_str"], how="left", suffixes=("", "_h"))
    df["hpi_yoy"] = pd.to_numeric(df.get("hpi_yoy_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(df.get("hpi_yoy"), errors="coerce")
    )
    if "air_yoy_h" in df.columns:
        df["air_yoy"] = pd.to_numeric(df["air_yoy_h"], errors="coerce").combine_first(pd.to_numeric(df.get("air_yoy"), errors="coerce"))
    if "year" not in df.columns or "quarter" not in df.columns:
        p = pd.PeriodIndex(df["period_str"], freq="Q")
        df["year"] = p.year
        df["quarter"] = p.quarter
    df["quarter_id"] = df["year"].astype(int) * 4 + df["quarter"].astype(int)
    df = df.sort_values(["geo", "quarter_id"]).reset_index(drop=True)
    g = df.groupby("geo", sort=False)
    if "air_yoy" in df.columns:
        df["L1_air_yoy"] = g["air_yoy"].shift(1)
        df["L2_air_yoy"] = g["air_yoy"].shift(2)
    for col in ["route_open_q", "route_close_q", "net_openings_q", "open_rate_norm_q", "close_rate_norm_q"]:
        if col in df.columns:
            df[f"L1_{col}"] = g[col].shift(1)

    for col in ["hpi_yoy", "L1_air_yoy", "L1_net_openings_q", "L1_open_rate_norm_q"]:
        if col in df.columns:
            df[col] = winsorize_series(df[col], 0.01, 0.99)

    df_panel = df.set_index(["geo", "quarter_id"]).sort_index()
    outputs: list[dict] = []

    specs = [
        (
            "quarterly_fe_airplus_routecounts",
            "hpi_yoy ~ 1 + L1_air_yoy + L1_net_openings_q + EntityEffects + TimeEffects",
            ["hpi_yoy", "L1_air_yoy", "L1_net_openings_q"],
        ),
        (
            "quarterly_fe_airplus_route_rates",
            "hpi_yoy ~ 1 + L1_air_yoy + L1_open_rate_norm_q + L1_close_rate_norm_q + EntityEffects + TimeEffects",
            ["hpi_yoy", "L1_air_yoy", "L1_open_rate_norm_q", "L1_close_rate_norm_q"],
        ),
    ]

    coef_frames = []
    text_blocks = []
    for label, formula, needed in specs:
        d = df_panel.dropna(subset=needed).copy()
        if len(d) == 0:
            continue
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
        coef_frames.append(coef)
        text_blocks.append(f"# {label}\n{res.summary}")

    if coef_frames:
        return pd.concat(coef_frames, ignore_index=True), text_blocks
    return pd.DataFrame(), []


def run_regional_twfe() -> tuple[pd.DataFrame, list[str]]:
    df = load_parquet_or_csv("panel_nuts2_annual")
    df = df.replace([np.inf, -np.inf], np.nan).copy()
    df = df.sort_values(["geo", "year"]).reset_index(drop=True)
    g = df.groupby("geo", sort=False)
    for col in ["rhpi_growth", "L1_air_growth", "L1_net_migration_rate", "L1_gdp_pc_growth"]:
        if col in df.columns:
            df[col] = winsorize_series(df[col], 0.01, 0.99)
    df_panel = df.set_index(["geo", "year"]).sort_index()

    specs = [
        (
            "nuts2_twfe_baseline",
            (
                "rhpi_growth ~ 1 + L1_net_migration_rate + L1_air_growth + "
                "L1_unemployment_rate + L1_gdp_pc_growth + L1_pop_growth + EntityEffects + TimeEffects"
            ),
            ["rhpi_growth", "L1_net_migration_rate", "L1_air_growth", "L1_unemployment_rate", "L1_gdp_pc_growth", "L1_pop_growth"],
        ),
        (
            "nuts2_twfe_post2020_interaction",
            (
                "rhpi_growth ~ 1 + L1_net_migration_rate + L1_air_growth + "
                "L1_net_migration_rate:post_2020 + L1_air_growth:post_2020 + "
                "L1_unemployment_rate + L1_gdp_pc_growth + EntityEffects + TimeEffects"
            ),
            ["rhpi_growth", "L1_net_migration_rate", "L1_air_growth", "post_2020", "L1_unemployment_rate", "L1_gdp_pc_growth"],
        ),
    ]

    coef_frames = []
    text_blocks = []
    for label, formula, needed in specs:
        d = df_panel.dropna(subset=needed).copy()
        if len(d) == 0:
            continue
        mod = PanelOLS.from_formula(formula, data=d, drop_absorbed=True, check_rank=False)
        res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        coef_frames.append(
            pd.DataFrame(
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
        )
        text_blocks.append(f"# {label}\n{res.summary}")
    if coef_frames:
        return pd.concat(coef_frames, ignore_index=True), text_blocks
    return pd.DataFrame(), []


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    coef_frames = []
    text_blocks = []
    summary: dict[str, object] = {}

    try:
        coef, txts = run_country_iv()
        coef_frames.append(coef)
        text_blocks.extend(txts)
        summary["country_iv_fe"] = {"n_terms": int(len(coef))}
    except Exception as e:
        text_blocks.append(f"# country_iv_fe\nFAILED: {e}")
        summary["country_iv_fe"] = {"failed": str(e)}

    try:
        coef_q, txt_q = run_quarterly_route_shock_fe()
        if not coef_q.empty:
            coef_frames.append(coef_q)
        text_blocks.extend(txt_q)
        summary["quarterly_route_shock_fe"] = {"n_terms": int(len(coef_q))}
    except Exception as e:
        text_blocks.append(f"# quarterly_route_shock_fe\nFAILED: {e}")
        summary["quarterly_route_shock_fe"] = {"failed": str(e)}

    try:
        coef_r, txt_r = run_regional_twfe()
        if not coef_r.empty:
            coef_frames.append(coef_r)
        text_blocks.extend(txt_r)
        summary["nuts2_twfe"] = {"n_terms": int(len(coef_r))}
    except Exception as e:
        text_blocks.append(f"# nuts2_twfe\nFAILED: {e}")
        summary["nuts2_twfe"] = {"failed": str(e)}

    if coef_frames:
        pd.concat(coef_frames, ignore_index=True).to_csv(RESULTS_DIR / "advanced_model_coefficients.csv", index=False)
    save_text(RESULTS_DIR / "advanced_model_summaries.txt", text_blocks)
    (RESULTS_DIR / "advanced_model_run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("[done] Advanced estimations complete")


if __name__ == "__main__":
    main()
